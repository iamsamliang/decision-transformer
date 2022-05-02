import csv
import logging
# make deterministic
from gpt.utils import set_seed
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.data import Dataset
from gpt.model_graph import GPT, GPTConfig
from gpt.trainer_graph import Trainer, TrainerConfig
from gpt.utils import sample
from collections import deque
import random
import torch
import pickle
import blosc
import argparse
import networkx as nx
import matplotlib.pyplot as plt

def create_graph_dataset(num_nodes, graph_density, num_trajs, max_traj_length, seed_val, goal_node):
    # Assumptions: Dataset derived from choosing only valid actions (ones where a directed edge exists)
    
    rng = np.random.default_rng(seed_val) # Set seed so results are reproducible
    graph = rng.binomial(1, graph_density, size=(num_nodes, num_nodes)) # Create random graph

    #G = nx.from_numpy_array(graph, create_using=nx.DiGraph)
    #plt.figure(figsize=(15, 12))
    #pos = nx.spring_layout(G, k=1.5, scale=1, seed=np.random)
    #nx.draw_networkx(G, pos, node_size=1200, node_color="#006400", width=0.5, font_color="white", font_size=16, arrowsize=15, font_family="serif")
    #plt.savefig(f'graph_seed{seed_val}.png', dpi=800)

    # trajectories = [] # store all trajectories
    observations = []
    actions = []
    returns = [0]
    done_idxs = []
    stepwise_returns = []
    
    goal_node = goal_node # Set a fixed goal node
    T = max_traj_length # Maximum trajectory length (does this refer to # of edges or nodes) --> going with EDGES for now
    num_trajs = num_trajs
    # NOTE THESE SPECIAL TOKENS CANNOT BE NEGATIVE OR THE NN.EMBEDDING LAYER DOESN'T WORK BC IT TAKES INDICES AS INPUTS
    goal_action_index = len(graph[0]) + 1 # special action index when in goal state
    fail_action_index = len(graph[0]) # special action index when fail to reach goal state
    all_actions = np.arange(len(graph[0]))
  
    for _ in range(num_trajs):
      # curr_node = rng.integers(0, 19)
      
      # this is uniformly selecting starting nodes with outdegree of at least 1 and that isn't the goal node
      nodes_with_outedges = np.sum(graph, axis=1).nonzero()[0]
      nodes_with_outedges = np.delete(nodes_with_outedges, np.where(nodes_with_outedges == goal_node))
      curr_node = rng.choice(nodes_with_outedges)
    
      reached_goal = False
      traj_length = 0
      
      # build one trajectory
      # trajectory length is counted by number of nodes
      for _ in range(T): # T
        observations.append(curr_node)
        traj_length += 1

        if curr_node == goal_node:
          reached_goal = True
          actions.append(goal_action_index)
          stepwise_returns.append(0)
          break

        valid_actions = all_actions[graph[curr_node] > 0]
        if len(valid_actions) == 0:
#          done_idxs.append(len(observations))
          # so EITHER we can add another stepwise return and action, or change the one we just added to the appropriate values
          actions.append(fail_action_index) #placeholder because outside the for loop, we set actions[-1] = fail_action_index
          stepwise_returns.append(-10000) # do we need to change all of them to -10000 for this trajectory?
#          returns[-1] += -10000 # or infinty?
          break
        action = rng.choice(valid_actions)
        actions.append(action)
        stepwise_returns.append(-1) # reward is defined by what state you are in when you take the action 
        returns[-1] += -1
        curr_node = action
#        if curr_node == goal_node:
#          reached_goal = True
#          stepwise_returns.append(-1) # how is reward defined: by what state you are in when you take the action or the state you go to; should last stepwise be 0 or -1
#          returns[-1] += -1 # reward of 0 when hit goal state, but still need to account for penalty of taking the action that got to goal state
#          observations.append(curr_node)
#          actions.append(goal_action_index)
#          stepwise_returns.append(0) # need to also append 0 for lengths to work out
#          done_idxs.append(len(observations))
#          break
    
      ### NOTE: This code block will pad 0s at the beginning of the trajectory until it is of length max_traj_length. This only makes sense to do if context_length = max_traj_length ###
#      if traj_length != T:
#        num_zeros = T - traj_length
#        for i in range(num_zeros):
#          observations.insert(done_idxs[-1] + i, 0)
      # We also need to pad stepwise_returns, returns, and actions
      ##############################################################################################################

      done_idxs.append(len(observations))
      if not reached_goal:
        returns[-1] += -10000 # or infinity
        stepwise_returns[-1] = -10000
        actions[-1] = fail_action_index
      returns.append(0)

    actions = np.array(actions)
    returns = np.array(returns)
    stepwise_returns = np.array(stepwise_returns)
    done_idxs = np.array(done_idxs)
    
    # -- create reward-to-go dataset
    start_index = 0
    rtg = np.zeros_like(stepwise_returns)
    for i in done_idxs:
        i = int(i)
        curr_traj_returns = stepwise_returns[start_index:i]
        for j in range(i-1, start_index-1, -1): # start from i-1
            rtg_j = curr_traj_returns[j-start_index:i-start_index]
            if stepwise_returns[i-1] == -10000:
              rtg[j] = -10000
            else:
              rtg[j] = sum(rtg_j)
        start_index = i
    print('max returns-to-go is %d' % max(rtg))

    # -- create timestep dataset
    start_index = 0
    timesteps = np.zeros(len(actions)+1, dtype=int)
    for i in done_idxs:
        i = int(i)
        timesteps[start_index:i+1] = np.arange(i+1 - start_index)
        start_index = i+1
    print('max timestep is %d' % max(timesteps))

    return observations, actions, returns, done_idxs, rtg, timesteps, graph
