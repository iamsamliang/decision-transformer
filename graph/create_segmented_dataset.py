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

def create_segmented_dataset(num_nodes, graph_density, num_trajs, max_traj_length, seed_val, self_cycles):
    # Assumptions: Dataset derived from choosing only valid actions (ones where a directed edge exists)
    
    # Given a graph G (self cycles allowed or prohibited), create trajectories of varying length 1 to max_traj_length. Then, test if the DT can find the shortest path between any start and goal node.

    rng = np.random.default_rng(seed_val) # Set seed so results are reproducible
    graph = rng.binomial(1, graph_density, size=(num_nodes, num_nodes)) # Create random graph
    
    observations = []
    actions = []
    stepwise_returns = []
    returns = []
    done_idxs = []
    
    # No more fixed goal node
    T = max_traj_length # Maximum trajectory length in terms of NUMBER OF EDGES (not nodes)
    num_trajs = num_trajs
    
    # NOTE THE SPECIAL ACTION TOKEN CANNOT BE NEGATIVE OR THE NN.EMBEDDING LAYER DOESN'T WORK BC IT TAKES INDICES AS INPUTS
    no_action_index = len(graph[0]) + 1 # special action index when in node with no possible action
    
    all_actions = np.arange(len(graph[0]))
  
    for _ in range(num_trajs):
      # this is uniformly selecting starting nodes with outdegree of at least 1
      nodes_with_outedges = np.sum(graph, axis=1).nonzero()[0]
      curr_node = rng.choice(nodes_with_outedges)
      
      # set the trajectory length randomly which will be between 1 - max_traj_length (inclusive)
      traj_length = rng.choice(np.arange(1, T + 1))

      # append 0 to the returns (total reward), so we start at 0 for each new trajectory and we can add to it
      returns.append(0)
       
      # build one trajectory (need traj_length + 1 bc we are counting by edges but just traj_length would be one short as for loop counts by nodes)
      for _ in range(traj_length + 1):
        # first append the node we are now at
        observations.append(curr_node)
        # find all directed edges we can take from current node
        valid_actions = all_actions[graph[curr_node] > 0]

        # if this node has no outward directed edges, then we append a special action token instead and a special reward; we also make the return (total reward) of this trajectory .......
        # we also break out of the for loop bc we are done building this trajectory
        if len(valid_actions) == 0:
          actions.append(no_action_index)
          stepwise_returns.append(-1) # NOTE: change the reward we append later
          returns[-1] += -1 # NOTE: change the reward we add later
          break

        # if we are here, that means the current node has at least one outward directed edge; sample one of these edges to take uniformly
        # next, append the action we are taking at the current node and the reward we get from taking this action (-1). Also, increment the return (total reward) of this trajectory by the reward we got from taking this action
        action = rng.choice(valid_actions)
        actions.append(action)
        stepwise_returns.append(-1) # how is reward defined: by what state you are in when you take the action or the state you go to; should last stepwise be 0 or -1
        returns[-1] += -1

        # next, update the current node because we are at a new node from taking the action
        curr_node = action

      # no matter what the outcome of the trajectory (we either hit a node with no outward edges or we built of trajectory of the desired length), 
      # we append the new number of observations seen across all trajectories built so that we know which observations belong to which trajectory
      done_idxs.append(len(observations))
      # we also append 0 to the returns (total reward), so the next trajectory can start adding to it  
      #returns.append(0)

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
    print('longest path length is %d' % (max(-rtg)))

    # -- create timestep dataset
    start_index = 0
    timesteps = np.zeros(len(actions)+1, dtype=int)
    for i in done_idxs:
        i = int(i)
        timesteps[start_index:i+1] = np.arange(i+1 - start_index)
        start_index = i+1
    print('max timestep is %d' % max(timesteps))

    return observations, actions, returns, done_idxs, rtg, timesteps, graph
