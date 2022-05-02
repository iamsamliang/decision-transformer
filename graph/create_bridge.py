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
from networkx.drawing.nx_pydot import graphviz_layout


# This creates random walks in two different undirected subgraphs. The two subgraphs are then connected together.
def create_bridge(num_nodes, graph_density, num_trajs, max_traj_length, seed_val, is_directed):
    # Assumptions: Dataset derived from choosing only valid actions (ones where a directed edge exists)
    # Seed has already been set prior to creating this graph dataset
     
    # Create two random undirected graphs
    graph1 = nx.fast_gnp_random_graph(num_nodes, graph_density, seed=np.random, directed=is_directed)
    graph2 = nx.fast_gnp_random_graph(num_nodes, graph_density, seed=np.random, directed=is_directed)
    new_labels = np.arange(len(graph1.nodes()), len(graph1.nodes()) + len(graph2.nodes()))
    the_mapping = dict(zip(graph2.nodes(), new_labels))
    nx.relabel_nodes(graph2, the_mapping, copy=False)

    plt.figure(figsize=(15, 12))
    pos = graphviz_layout(graph1, prog="dot")
    nx.draw_networkx(graph1, pos, node_size=1200, node_color="#006400", width=0.5, font_color="white", font_size=16, arrowsize=15, font_family="serif")
    plt.savefig(f'undirected1_seed{seed_val}', dpi=500, bbox_inches="tight")

    plt.figure(figsize=(15, 12))
    pos = graphviz_layout(graph2, prog="dot")
    nx.draw_networkx(graph2, pos, node_size=1200, node_color="#006400", width=0.5, font_color="white", font_size=16, arrowsize=15, font_family="serif")
    plt.savefig(f'undirected2_seed{seed_val}', dpi=500, bbox_inches="tight")
    
    # combine the two graphs and link them with one edge
    combined_graph = nx.compose(graph1, graph2)
    combined_graph.add_edge(np.random.choice(graph1.nodes()), np.random.choice(graph2.nodes()))

    plt.figure(figsize=(15, 12))
    pos = graphviz_layout(combined_graph, prog="dot")
    nx.draw_networkx(combined_graph, pos, node_size=1200, node_color="#006400", width=0.5, font_color="white", font_size=16, arrowsize=15, font_family="serif")
    plt.savefig(f'combined_seed{seed_val}', dpi=500, bbox_inches="tight")

    all_actions1 = np.array(graph1.nodes())
    all_actions2 = np.array(graph2.nodes())

    graph1 = nx.to_numpy_array(graph1, dtype=np.int64)
    graph2 = nx.to_numpy_array(graph2, dtype=np.int64)
    combined_graph = nx.to_numpy_array(combined_graph, dtype=np.int64)

    observations = []
    actions = []
    returns = [0]
    done_idxs = []
    stepwise_returns = []
    
    T = max_traj_length # Maximum trajectory length (does this refer to # of edges or nodes) --> going with EDGES for now
    num_trajs = num_trajs
    
    goal_action_index = len(graph1[0]) + len(graph2[0]) # special action index when fail to reach goal state
    num_trajs1 = num_trajs // 2
    num_trajs2 = num_trajs // 2

    # random trajectories within graph 1
    for _ in range(num_trajs1):
      # this is uniformly selecting starting nodes with outdegree of at least 1
      nodes_with_outedges = np.sum(graph1, axis=1).nonzero()[0]
      curr_node = np.random.choice(nodes_with_outedges)

      traj_length = 0
      
      # build one trajectory
      # trajectory length is counted by number of nodes
      for _ in range(T):
        observations.append(curr_node)
        traj_length += 1

        valid_actions = all_actions1[graph1[curr_node] > 0]
        if len(valid_actions) == 0:
          actions.append(goal_action_index) 
          stepwise_returns.append(0) 
          break

        action = np.random.choice(valid_actions)
        actions.append(action)
        stepwise_returns.append(-1) # reward is defined by what state you are in when you take the action 
        returns[-1] += -1
        curr_node = action
   
      stepwise_returns[-1] = 0
      actions[-1] = goal_action_index

      ### NOTE: This code block will pad 0s at the beginning of the trajectory until it is of length max_traj_length. This only makes sense to do if context_length = max_traj_length ###
      if traj_length != T:
        num_zeros = T - traj_length
        for i in range(num_zeros):
          observations.append(-10)
          stepwise_returns.append(0)
          actions.append(goal_action_index + 1)
      # We also need to pad stepwise_returns, returns, and actions
      ##############################################################################################################

      done_idxs.append(len(observations))
      returns.append(0)

    # random trajectories within graph 2
    for _ in range(num_trajs2):
      # this is uniformly selecting starting nodes with outdegree of at least 1
      nodes_with_outedges = np.sum(graph2, axis=1).nonzero()[0]
      curr_node = np.random.choice(nodes_with_outedges)

      traj_length = 0
      
      # build one trajectory
      # trajectory length is counted by number of nodes
      for _ in range(T):
        observations.append(curr_node + num_nodes)
        traj_length += 1
        
        valid_actions = all_actions1[graph2[curr_node] > 0]
        if len(valid_actions) == 0:
          actions.append(goal_action_index) 
          stepwise_returns.append(0) 
          break

        action = np.random.choice(valid_actions)
        actions.append(action + num_nodes)
        stepwise_returns.append(-1) # reward is defined by what state you are in when you take the action 
        returns[-1] += -1
        curr_node = action
   
      stepwise_returns[-1] = 0
      actions[-1] = goal_action_index

      ### NOTE: This code block will pad 0s at the beginning of the trajectory until it is of length max_traj_length. This only makes sense to do if context_length = max_traj_length ###
      if traj_length != T:
        num_zeros = T - traj_length
        for i in range(num_zeros):
          observations.append(-10)
          stepwise_returns.append(0)
          actions.append(goal_action_index + 1)
      # We also need to pad stepwise_returns, returns, and actions
      ##############################################################################################################

      done_idxs.append(len(observations))
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

    # -- create timestep dataset
    start_index = 0
    timesteps = np.zeros(len(actions)+1, dtype=int)
    for i in done_idxs:
        i = int(i)
        timesteps[start_index:i+1] = np.arange(i+1 - start_index)
        start_index = i+1
    print('max timestep is %d' % max(timesteps))

    return observations, actions, returns, done_idxs, rtg, timesteps, combined_graph
