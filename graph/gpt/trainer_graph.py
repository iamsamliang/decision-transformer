"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)

from gpt.utils import sample
import atari_py
from collections import deque
import random
import cv2
import torch
from PIL import Image
import wandb
from scipy.sparse.csgraph import shortest_path
import matplotlib.pyplot as plt

class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1 # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9 # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0 # for DataLoader

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config, graph, goal_node, is_directed):
        self.model = model
        self.train_dataset = train_dataset
        # test dataset should be the shortest path from each node to the fixed goal
        self.test_dataset = test_dataset
        self.config = config
        self.graph = graph
        self.goal_node = goal_node
        self.dist_matrix, self.predecessors = shortest_path(self.graph, directed=is_directed, return_predecessors=True, unweighted=True)

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)
            print(f'Using CUDA device')
        else:
            print(f'Using CPU device')

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        # torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        def run_epoch(split, epoch_num=0):
            is_train = split == 'train'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            #for (states, actions, rtgs, timesteps) in loader:
                #print(states)
                #print()
                #print(actions)
                #print()
                #print(rtgs)
                #print(timesteps)
                #break
            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
       
            # x = states
            # y = actions
            # r = returns-to-go
            # t = timesteps
            for it, (x, y, r, t) in pbar:

                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)
                r = r.to(self.device)
                t = t.to(self.device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    # logits, loss = model(x, y, r)
                    logits, loss = model(x, y, y, r, t)
                    loss = loss.mean() # collapse all losses if they are scattered on multiple gpus
                    wandb.log({"train_loss": loss.item()})
                    losses.append(loss.item())

                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (y >= 0).sum() # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate
                    
                    #wandb.watch(model, criterion=F.cross_entropy, log="all", log_freq="100", idx=None, log_graph=(False))		    
                    #wandb.config.update({"learning_rate": lr}, allow_val_change=True)

                    # report progress
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

            if not is_train:
                test_loss = float(np.mean(losses))
                wandb.log({"test_loss": test_loss})
                logger.info("test loss: %f", test_loss)
                return test_loss

        # best_loss = float('inf')
        
        best_return = -float('inf')

        self.tokens = 0 # counter used for learning rate decay

        wandb.init(project="dt-graph", name=f'{self.config.dataset} seed {self.config.seed}', entity="iamsamliang", config={"learning_rate": config.learning_rate, "architecture": "Decision Tranformer", 
        "epochs": config.max_epochs, "batch_size": config.batch_size})
#        wandb.watch(model, log="all", log_freq="100")
        for epoch in range(config.max_epochs):
            run_epoch('train', epoch_num=epoch)
            # if self.test_dataset is not None:
            #     test_loss = run_epoch('test')

            # -- pass in target returns
            # original == graph dataset used for DT paper results
            if self.config.dataset == 'original':
                eval_return = self.get_returns_orig(False, self.train_dataset.max_traj_length)
            elif self.config.dataset == 'varying_segments':
                eval_return = self.get_returns_vary_fixed(False, self.train_dataset.max_traj_length)
            elif self.config.dataset == 'bridge':
                eval_return = self.get_returns_bridge(False)

        wandb.finish()

    def get_returns_orig(self, plot_hist, max_traj_length):
        self.model.train(False)
        env = Env(self.device, self.config.seed, self.graph, self.goal_node)
        env.eval()
        
        T_rewards = []
        done = True
        succeeded = [] # succeeded will indicate whether the generated paths are same length as shortest paths because the generated paths will need to be at least shortest path length
        generated_paths = []
        shortest_paths = []
        exists_shortest = [True] * len(self.graph)
        exists_shortest[self.goal_node] = False # we do not generate a shortest path starting from goal node
        gen_path_lengths = [] # keep track of the length of the generated paths
        random_path_lengths = [] # keep track of length of random graphs
        all_actions = np.arange(len(self.graph[0]))

        if plot_hist == True:
            # see what a random agent would do
            for i in range(len(self.graph)):
                if i == self.goal_node:
                    continue
    
                curr_node = i
                random_done = False
    
                for i in range(max_traj_length):
                    if curr_node == self.goal_node:
                        random_path_lengths.append(i)
                        random_done = True
                        break
                        
                    valid_actions = all_actions[self.graph[curr_node] > 0]
                    if len(valid_actions) == 0:
                        random_path_lengths.append(np.inf)
                        random_done = True
                        break
    
                    action = np.random.choice(valid_actions)
                    curr_node = action
                
                if not random_done:
                    random_path_lengths.append(np.inf)

        # see if the model can generate shortest path from each node, except goal node
        for i in range(len(self.graph)):
            if i == self.goal_node:
                continue
            env.set_state(i)
            done = False
            reward_sum = 0
            state = i
            path = [state]
            shortest_dist = - self.dist_matrix[state, self.goal_node] # negative length of shortest path from start_node to goal_node
            curr_node = self.goal_node
            # generate the shortest path
            # if it there is no shortest path, continue to next iteration
            if self.predecessors[state, curr_node] == -9999:
                exists_shortest[state] = False
                shortest_paths.append([None])
                generated_paths.append([None])
                gen_path_lengths.append(np.inf)
                T_rewards.append(-999999)
                continue
            else:
                shortest_path = [curr_node]
                while self.predecessors[state, curr_node] != state:
                    next_node = self.predecessors[state, curr_node]
                    shortest_path.insert(0, next_node)
                    curr_node = next_node
                shortest_path.insert(0, state)
                shortest_paths.append(shortest_path)

            state = torch.tensor(state)
            state = state.type(torch.float32).to(self.device).unsqueeze(0).unsqueeze(0)
            rtgs = [shortest_dist]
            # first state is from env, first rtg is target return (negative shortest path length), and first timestep is 0
            sampled_action = sample(self.model.module, state, 1, self.graph[path[-1]], temperature=1.0, sample=False, actions=None, 
                rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1), 
                timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(self.device))

            j = 0
            all_states = state
            actions = []

            while len(path) < max_traj_length:
#                if done:
#                    state, reward_sum, done = env.reset(), 0, False
                action = sampled_action.cpu().numpy()[0,-1]
                actions += [sampled_action]
                #if action == 21:
                    #print(i)
                    #print(self.graph[path[-1]])
                    #print(path)
                    #print(actions)
                state, reward, done = env.step(action)
                reward_sum += reward
                j += 1
                path.append(state)
                rtgs += [rtgs[-1] - reward]

                if done:
                    T_rewards.append(reward_sum)
                    if len(path) <= -shortest_dist + 1:
                        succeeded.append(True)
                    else:
                        succeeded.append(False)
                    break
                
                if len(all_actions[self.graph[state] > 0]) == 0:
                    break

                state = torch.tensor(state)
                state = state.unsqueeze(0).unsqueeze(0).to(self.device)

                all_states = torch.cat([all_states, state], dim=0)

                # all_states has all previous states and rtgs has all previous rtgs (will be cut to block_size in utils.sample)
                # timestep is just current timestep
                sampled_action = sample(self.model.module, all_states.unsqueeze(0), 1, self.graph[path[-1]], temperature=1.0, sample=False, 
                    actions=torch.tensor(actions, dtype=torch.long).to(self.device).unsqueeze(1).unsqueeze(0), 
                    rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1), 
                    timesteps=(min(j, self.config.max_timestep) * torch.ones((1, 1, 1), dtype=torch.int64).to(self.device)))

            generated_paths.append(path)

            if not done:
                succeeded.append(False)
                T_rewards.append(-10000)
                gen_path_lengths.append(np.inf)
            else:
                gen_path_lengths.append(len(path) - 1)

        
        print(f'\nTotal Rewards Per Simulation: {T_rewards}')
        print(f'These are the runs that succeeded/failed: {succeeded}\n')
        print(f'Generated shortest path for {sum(succeeded)} simulations out of {len(succeeded)}\n')    
        print(f'Generated Paths: {generated_paths}')
        print(f'Shortest Paths: {shortest_paths}\n\n')

        # plot the histogram
        if plot_hist==True:
            short_path_lengths = self.dist_matrix
            short_path_lengths = np.delete(short_path_lengths, self.goal_node, axis=0)
            short_path_lengths = short_path_lengths[:, self.goal_node]
            short_path_lengths = short_path_lengths.reshape(-1)
            short_path_lengths = np.delete(short_path_lengths, np.where(short_path_lengths==0))
            self.plot_lengths(np.array(gen_path_lengths), np.array(short_path_lengths), np.array(random_path_lengths), max_traj_length)

        self.model.train(True)
        return T_rewards

    def get_returns_vary_fixed(self, plot_hist, max_traj_length):
        self.model.train(False)

        # first check if there are shortest paths with length greater than max_traj_length to the fixed goal node
        shortest_to_goal = self.dist_matrix[:, self.goal_node]
        if np.max(shortest_to_goal[np.where(shortest_to_goal != np.inf)]) <= max_traj_length:
            print(f'There are no evaluations because there are no shortest paths with length greater than {max_traj_length} to the fixed goal node')
            return None

        # next, find all shortest paths with length greater than max_traj_length
        # then, test to see if the DT can find the shortest paths from the corresponding start to goal nodes

        start_nodes = np.where((shortest_to_goal > max_traj_length) & (shortest_to_goal != np.inf))[0]

        env = Env(self.device, self.config.seed, self.graph, self.goal_node)
        env.eval()
        
        T_rewards = []
        done = True
        succeeded = [] # succeeded will indicate whether the generated paths are same length as shortest paths because the generated paths will need to be at least shortest path length
        generated_paths = []
        shortest_paths = []
        exists_shortest = [True] * len(self.graph)
        exists_shortest[self.goal_node] = False # we do not generate a shortest path starting from goal node
        gen_path_lengths = [] # keep track of the length of the generated paths
        random_path_lengths = [] # keep track of length of random graphs
        all_actions = np.arange(len(self.graph[0]))
        max_rel = -1

        if plot_hist == True:
            # see what a random agent would do
            for i in start_nodes:
                if i == self.goal_node:
                    continue
                
                shortest_dist = int(-self.dist_matrix[i, self.goal_node])
                curr_node = i
                random_done = False
    
                for i in range(1, -shortest_dist + 6):
                    if curr_node == self.goal_node:
                        random_path_lengths.append(i)
                        random_done = True
                        break
                        
                    valid_actions = all_actions[self.graph[curr_node] > 0]
                    if len(valid_actions) == 0:
                        random_path_lengths.append(np.inf)
                        random_done = True
                        break
    
                    action = np.random.choice(valid_actions)
                    curr_node = action
                
                if not random_done:
                    random_path_lengths.append(np.inf)

        # see if the model can generate shortest path from each node, except goal node
        for i in start_nodes:
            if i == self.goal_node:
                continue
            env.set_state(i)
            done = False
            reward_sum = 0
            state = i
            path = [state]
            shortest_dist = - self.dist_matrix[state, self.goal_node] # negative length of shortest path from start_node to goal_node

            if -shortest_dist > max_rel:
                max_rel = -shortest_dist

            curr_node = self.goal_node
            # generate the shortest path
            # if it there is no shortest path, continue to next iteration
            if self.predecessors[state, curr_node] == -9999:
                exists_shortest[state] = False
                shortest_paths.append([None])
                generated_paths.append([None])
                gen_path_lengths.append(np.inf)
                T_rewards.append(-999999)
                continue
            else:
                shortest_path = [curr_node]
                while self.predecessors[state, curr_node] != state:
                    next_node = self.predecessors[state, curr_node]
                    shortest_path.insert(0, next_node)
                    curr_node = next_node
                shortest_path.insert(0, state)
                shortest_paths.append(shortest_path)

            state = torch.tensor(state)
            state = state.type(torch.float32).to(self.device).unsqueeze(0).unsqueeze(0)
            rtgs = [shortest_dist]
            # first state is from env, first rtg is target return (negative shortest path length), and first timestep is 0
            sampled_action = sample(self.model.module, state, 1, self.graph[path[-1]], temperature=1.0, sample=False, actions=None, 
                rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1), 
                timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(self.device))

            j = 0
            all_states = state
            actions = []

            while len(path) < -shortest_dist + 5:
#                if done:
#                    state, reward_sum, done = env.reset(), 0, False
                action = sampled_action.cpu().numpy()[0,-1]
                actions += [sampled_action]
                state, reward, done = env.step(action)
                reward_sum += reward
                j += 1
                path.append(state)
                rtgs += [rtgs[-1] - reward]

                if done:
                    T_rewards.append(reward_sum)
                    if len(path) <= -shortest_dist + 1:
                        succeeded.append(True)
                    else:
                        succeeded.append(False)
                    break
                
                if len(all_actions[self.graph[state] > 0]) == 0:
                    break

                state = torch.tensor(state)
                state = state.unsqueeze(0).unsqueeze(0).to(self.device)

                all_states = torch.cat([all_states, state], dim=0)

                # all_states has all previous states and rtgs has all previous rtgs (will be cut to block_size in utils.sample)
                # timestep is just current timestep
                sampled_action = sample(self.model.module, all_states.unsqueeze(0), 1, self.graph[path[-1]], temperature=1.0, sample=False, 
                    actions=torch.tensor(actions, dtype=torch.long).to(self.device).unsqueeze(1).unsqueeze(0), 
                    rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1), 
                    timesteps=(min(j, self.config.max_timestep) * torch.ones((1, 1, 1), dtype=torch.int64).to(self.device)))

            generated_paths.append(path)

            if not done:
                succeeded.append(False)
                T_rewards.append(-10000)
                gen_path_lengths.append(np.inf)
            else:
                gen_path_lengths.append(len(path) - 1)

        
        print(f'\nTotal Rewards Per Simulation: {T_rewards}')
        print(f'These are the runs that succeeded/failed: {succeeded}\n')
        print(f'Generated shortest path for {sum(succeeded)} simulations out of {len(succeeded)}\n')    
        print(f'Generated Paths: {generated_paths}')
        print(f'Shortest Paths: {shortest_paths}\n\n')

        # plot the histogram
        if plot_hist==True:
            short_path_lengths = shortest_to_goal[start_nodes]
            short_path_lengths = short_path_lengths.reshape(-1)
            short_path_lengths = np.delete(short_path_lengths, np.where(short_path_lengths==0))
            self.plot_lengths(np.array(gen_path_lengths), np.array(short_path_lengths), np.array(random_path_lengths), max_rel + 5)

        self.model.train(True)
        return T_rewards


    def get_returns_bridge(self, plot_hist):
        self.model.train(False)

        start_nodes = []
        goal_nodes = []

        # pick 3 starting nodes in one subgraph and goal nodes in the other subgraph
        which_graph = np.random.choice([0, 1]) == 1
        if which_graph:
            start_nodes = np.random.choice(np.arange(0, len(self.graph) // 2), size=3, replace=False)
            goal_nodes = np.random.choice(np.arange(len(self.graph) // 2, len(self.graph)), size=3, replace=False)
        else:
            goal_nodes = np.random.choice(np.arange(0, len(self.graph) // 2), size=3, replace=False)
            start_nodes = np.random.choice(np.arange(len(self.graph) // 2, len(self.graph)), size=3, replace=False)
        
        start_goal_pairs = tuple(zip(start_nodes, goal_nodes)) # ((start, goal), (start, goal), ...., (start, goal))

        env = Env(self.device, self.config.seed, self.graph, None)
        env.eval()
        
        T_rewards = []
        done = True
        succeeded = [] # succeeded will indicate whether the generated paths are same length as shortest paths because the generated paths will need to be at least shortest path length
        generated_paths = []
        shortest_paths = []
        exists_shortest = [True] * len(start_nodes)
        gen_path_lengths = [] # keep track of the length of the generated paths
        random_path_lengths = [] # keep track of length of random graphs
        all_actions = np.arange(len(self.graph[0]))
        max_rel = -1
        short_path_lengths = []

        if plot_hist == True:
            # see what a random agent would do
            for start, goal in start_goal_pairs:
                
                shortest_dist = int(-self.dist_matrix[start, goal])
                curr_node = start
                random_done = False
    
                for i in range(1, -shortest_dist + 6):
                    if curr_node == goal:
                        random_path_lengths.append(i)
                        random_done = True
                        break
                        
                    valid_actions = all_actions[self.graph[curr_node] > 0]
                    if len(valid_actions) == 0:
                        random_path_lengths.append(np.inf)
                        random_done = True
                        break
    
                    action = np.random.choice(valid_actions)
                    curr_node = action
                
                if not random_done:
                    random_path_lengths.append(np.inf)

        # see if the model can generate shortest path from each start node in start_goal_pairs to its corresponding goal node
        for start_node, goal_node in start_goal_pairs:
            env.set_state(start_node)
            env.set_goal(goal_node)
            done = False
            reward_sum = 0
            state = start_node
            path = [state]
            shortest_dist = - self.dist_matrix[state, goal_node] # negative length of shortest path from start_node to goal_node
            short_path_lengths.append( -shortest_dist)

            if -shortest_dist > max_rel:
                max_rel = -shortest_dist

            curr_node = goal_node
            # generate the shortest path
            # if it there is no shortest path, continue to next iteration
            if self.predecessors[state, curr_node] == -9999:
                exists_shortest[state] = False
                shortest_paths.append([None])
                generated_paths.append([None])
                gen_path_lengths.append(np.inf)
                T_rewards.append(-999999)
                continue
            else:
                shortest_path = [curr_node]
                while self.predecessors[state, curr_node] != state:
                    next_node = self.predecessors[state, curr_node]
                    shortest_path.insert(0, next_node)
                    curr_node = next_node
                shortest_path.insert(0, state)
                shortest_paths.append(shortest_path)

            state = torch.tensor(state)
            state = state.type(torch.float32).to(self.device).unsqueeze(0).unsqueeze(0)
            rtgs = [shortest_dist]
            # first state is from env, first rtg is target return (negative shortest path length), and first timestep is 0
            sampled_action = sample(self.model.module, state, 1, self.graph[path[-1]], temperature=1.0, sample=False, actions=None, 
                rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1), 
                timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(self.device))

            j = 0
            all_states = state
            actions = []

            while len(path) < -shortest_dist + 5:
                action = sampled_action.cpu().numpy()[0,-1]
                actions += [sampled_action]
                #if action == 21:
                    #print(i)
                    #print(self.graph[path[-1]])
                    #print(path)
                    #print(actions)
                state, reward, done = env.step(action)
                reward_sum += reward
                j += 1
                path.append(state)
                rtgs += [rtgs[-1] - reward]

                if done:
                    T_rewards.append(reward_sum)
                    if len(path) <= -shortest_dist + 1:
                        succeeded.append(True)
                    else:
                        succeeded.append(False)
                    break
                
                if len(all_actions[self.graph[state] > 0]) == 0:
                    break
                
                state = torch.tensor(state)
                state = state.unsqueeze(0).unsqueeze(0).to(self.device)

                all_states = torch.cat([all_states, state], dim=0)

                # all_states has all previous states and rtgs has all previous rtgs (will be cut to block_size in utils.sample)
                # timestep is just current timestep
                sampled_action = sample(self.model.module, all_states.unsqueeze(0), 1, self.graph[path[-1]], temperature=1.0, sample=False, 
                    actions=torch.tensor(actions, dtype=torch.long).to(self.device).unsqueeze(1).unsqueeze(0), 
                    rtgs=torch.tensor(rtgs, dtype=torch.long).to(self.device).unsqueeze(0).unsqueeze(-1), 
                    timesteps=(min(j, self.config.max_timestep) * torch.ones((1, 1, 1), dtype=torch.int64).to(self.device)))

            generated_paths.append(path)
            if not done:
                succeeded.append(False)
                T_rewards.append(-10000)
                gen_path_lengths.append(np.inf)
            else:
                gen_path_lengths.append(len(path) - 1)

        print(f'\nTotal Rewards Per Simulation: {T_rewards}')
        print(f'These are the runs that succeeded/failed: {succeeded}\n')
        print(f'Generated shortest path for {sum(succeeded)} simulations out of {len(succeeded)}\n')    
        print(f'Generated Paths: {generated_paths}')
        print(f'Shortest Paths: {shortest_paths}\n\n')

        # plot the histogram
        if plot_hist==True:
            self.plot_lengths(np.array(gen_path_lengths), np.array(short_path_lengths), np.array(random_path_lengths), max_rel + 5)

        self.model.train(True)
        return T_rewards

    def plot_lengths(self, gen_path_lengths, short_path_lengths, random_path_lengths, max_traj_length):
        bins = np.arange(1, max_traj_length + 2)

        fig, ax = plt.subplots(figsize=(9, 5))
        y_val, bins, patches = plt.hist([np.clip(gen_path_lengths, bins[0], bins[-1]), np.clip(short_path_lengths, bins[0], bins[-1]), np.clip(random_path_lengths, bins[0], bins[-1])], density=True, bins=bins, color=['#013220', '#F46B00', '#1E2F97'], label=['transformer', 'shortest path', 'random walk'], align='left')

        xlabels = bins[0:-1].astype(str)
        xlabels[-1] = r"$\infty$"

        plt.xlim([0, max_traj_length + 1])
        plt.xticks(np.arange(1, max_traj_length + 1))
        ax.set_xticklabels(xlabels)
        plt.xlabel("Number of Steps to Goal")

        plt.ylim([0, min(np.max(y_val) + 0.1, 1)])
        plt.ylabel("Proportion of Paths")
        plt.title('')
        plt.setp(patches, linewidth=0)
        plt.legend(loc='upper left', fontsize=10)

        fig.tight_layout()
        plt.savefig(f'histogram_varyfixed_seed{self.config.seed}.png', format='png', dpi=600)

class Env():
    # either build graph and goal_node from seed or pass them in
    def __init__(self, device, seed, graph, goal_node):
        self.device = device
        self.seed = seed
        self.graph = graph
        self.goal_node = goal_node
        # do we want to use seed to generate start state? yea, so it's reproducible
        self.rng = np.random.default_rng(seed)
        #nodes_with_outedges = np.sum(graph, axis=1).nonzero()[0]
        #nodes_with_outedges = np.delete(nodes_with_outedges, np.where(nodes_with_outedges == goal_node))
        #self.node_bucket = nodes_with_outedges
        #self.curr_node = self.rng.choice(self.node_bucket)
        self.curr_node = None
        self.actions = np.arange(len(graph))
#        self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
        self.training = True  # Consistent with model training mode

    def get_state(self):
        return self.curr_node

    def set_state(self, state):
        self.curr_node = state

    def set_goal(self, goal_node):
        self.goal_node = goal_node

    def reset(self):
        self.curr_node = self.rng.choice(self.node_bucket)
        return self.curr_node

    def step(self, action):
        # output the reward of taking this action
        # update the current state
        # check if we reached the goal state, and output correct signal
   
        reward, done = -1, False
        if self.graph[self.curr_node, action] == 1:
            self.curr_node = action
        else:
            self.curr_node = self.curr_node # state doesn't change if we take illegal action
        
        if self.curr_node == self.goal_node:
            done = True
        
        # the original code returns a state_buffer of the previous 3 states and the new state
        return self.curr_node, reward, done

    # Uses loss of life as terminal signal
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False

    def action_space(self):
        return len(self.actions)

