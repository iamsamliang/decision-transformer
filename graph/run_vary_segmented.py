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
#from create_segmented_dataset import create_segmented_dataset
from create_varying import create_varying_fixed_graph

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=340)
parser.add_argument('--context_length', type=int, default=5)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--model_type', type=str, default='reward_conditioned')
parser.add_argument('--batch_size', type=int, default=128)

parser.add_argument('--dataset', type=str, default='varying_segments')

# segmented graph dataset specific arguments
parser.add_argument('--num_nodes', type=int, default=24)
parser.add_argument('--graph_density', type=float, default=0.09)
parser.add_argument('--num_trajs', type=int, default=1000)
parser.add_argument('--max_traj_length', type=int, default=5)
parser.add_argument('--goal_node', type=int, required=True)
parser.add_argument('--is_directed', type=int, required=True)

args = parser.parse_args()

set_seed(args.seed)

is_directed = -1
if args.is_directed == 0:
    is_directed = False
elif args.is_directed == 1:
    is_directed = True

# Note, the context_length argument is misleading. The GPT context length is not context_length. They used context_length * 3 as the GPT context_length. context_length * 3 is also the block_size passed into the StateActionReturnDatset, GPTConfig, and TrainerConfig 

# The reason why context_length * 3 is used is because each timestep has 3 tokens instead of 1 (as in language prediction). The 3 tokens are returns-to-go, state, and action. Therefore, if you want a context length of 30, you need the GPT model to actually have a context length of 30 * 3 = 90

class StateActionReturnDataset(Dataset):

    def __init__(self, data, block_size, actions, done_idxs, rtgs, timesteps, max_traj_length): 
        self.block_size = block_size
        #self.vocab_size = max(actions) + 3 # in the graph setting, some actions are illegal in certain states; does that change this line?
        self.data = data
        self.actions = actions
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps
        self.max_traj_length = max_traj_length
    
    def __len__(self):
        return len(self.data) - self.block_size # this is because the chunks of data are allowed to overlap and are still unique "hello" -> "hell" and "ello" are different data items; shouldn't this be self.block_size // 3?

    def __getitem__(self, idx):
        orig_idx = idx
        block_size = self.block_size // 3
        done_idx = idx + block_size
        for i in self.done_idxs:
            if i > idx: # first done_idx greater than idx
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - block_size # how do they make sure this isn't negative? - The first trajetory must be of lengtha at least block size otherwise it will be negative
        
        # This code will right align trajectories that are smaller than block_size by taking tokens from the next trajectory and adding them to the end (hence, overlapping trajectories are possible)
        # The if statement only occurs w/ values in self.done_idxs < block_size
        # in all other cases, it left aligns by taking tokens from previous trajectory and using them at the front of the current trajectory of interest
        if idx < 0:
            idx = orig_idx
            done_idx = block_size + idx

        # each item also needs to be of same block_size, so cannot change block size
        # apparently, they do not care about splitting the trajectories. It feeds in tokens from overlapping trajectories sometimes
        states = torch.tensor(np.array(self.data[idx:done_idx]), dtype=torch.float32).reshape(block_size, -1) # (block_size, 4*84*84)
#        states = states / 255. # why divide by 255? I think it's because the states are images, so they're normalizing. I don't think we need to normalize graph nodes
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1) # (block_size, 1)
        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:idx+1], dtype=torch.int64).unsqueeze(1)

        return states, actions, rtgs, timesteps

# args.goal_node is not used
obss, actions, returns, done_idxs, rtgs, timesteps, graph, best_goal = create_varying_fixed_graph(args.num_nodes, args.graph_density, args.num_trajs, args.max_traj_length, args.seed, None)
vocab_size = len(graph) + 3

# set up logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

train_dataset = StateActionReturnDataset(obss, args.context_length*3, actions, done_idxs, rtgs, timesteps, args.max_traj_length)

mconf = GPTConfig(vocab_size, train_dataset.block_size,
                  n_layer=6, n_head=8, n_embd=128, model_type=args.model_type, max_timestep=max(timesteps))
model = GPT(mconf)

# initialize a trainer instance and kick off training
epochs = args.epochs
tconf = TrainerConfig(max_epochs=epochs, batch_size=args.batch_size, learning_rate=6e-4,
                      lr_decay=True, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*args.context_length*3,
                      num_workers=2, seed=args.seed, model_type=args.model_type, max_timestep=max(timesteps), dataset=args.dataset)
trainer = Trainer(model, train_dataset, None, tconf, graph, best_goal, is_directed)

trainer.train()

eval_returns = []
eval_return = trainer.get_returns_vary_fixed(True, args.max_traj_length)
eval_returns.append(eval_return)
print()
print(eval_returns)
