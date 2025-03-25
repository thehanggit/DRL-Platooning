#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 01:18:36 2019

@author: ccyen
"""

import gym, random, pickle, os.path, math, glob
import numpy as np
import itertools

from timeit import default_timer as timer
from datetime import timedelta
from timeit import default_timer as timer

import torch
import torch.optim as optim

import matplotlib
from IPython.display import clear_output

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


from C3PO_DRL.utils.wrappers import *
from C3PO_DRL.utils.hyperparameters import Config
from C3PO_DRL.utils.plot import plot_reward

from C3PO_DRL.C3PO_BaseAgent import BaseAgent
from C3PO_DRL.C3PO_Network import CNN, CNN_Actor, CNN_Critic, DQN, NN, DuelingDQN, DuelingDQN_CA, BDN, ActorCritic, DuelingDQN_with_Noisy
from C3PO_DRL.C3PO_MemReplay import ReplayBuffer, ExperienceReplayMemory, PrioritizedReplayMemory

class Model_3DQN(BaseAgent):
    def __init__(self, static_policy=False, config=None, shapeOfState=None, sizeOfActions=None):
        super(Model_3DQN, self).__init__(config=config, shapeOfState=shapeOfState, sizeOfActions=sizeOfActions)
        self.device = config.device

        #self.noisy=config.USE_NOISY_NETS
        self.priority_replay=config.USE_PRIORITY_REPLAY

        self.gamma = config.GAMMA
        self.lr = config.LR
        self.target_net_update_freq = config.TARGET_NET_UPDATE_FREQ
        self.experience_replay_size = config.EXP_REPLAY_SIZE
        self.batch_size = config.BATCH_SIZE
        self.learn_start = config.LEARN_START
        self.update_freq = config.UPDATE_FREQ
        self.sigma_init= config.SIGMA_INIT
        self.priority_beta_start = config.PRIORITY_BETA_START
        self.priority_beta_frames = config.PRIORITY_BETA_FRAMES
        self.priority_alpha = config.PRIORITY_ALPHA

        self.static_policy = static_policy
        self.num_feats = shapeOfState
        self.num_actions = sizeOfActions
        
        self.declare_networks()
            
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        #move to correct device
        self.model = self.model.to(self.device)
        self.target_model.to(self.device)

        if self.static_policy:
            self.model.eval()
            self.target_model.eval()
        else:
            self.model.train()
            self.target_model.train()

        self.update_count = 0

        self.declare_memory()

        self.nsteps = config.N_STEPS
        self.nstep_buffer = []

    def declare_networks(self):
        #self.model = DuelingDQN_with_Noisy(self.num_feats, self.num_actions, noisy=self.noisy, sigma_init=self.sigma_init)
        #self.target_model = DuelingDQN_with_Noisy(self.num_feats, self.num_actions, noisy=self.noisy, sigma_init=self.sigma_init)
        self.model = DuelingDQN(self.num_feats, self.num_actions)
        self.target_model = DuelingDQN(self.num_feats, self.num_actions)

    def declare_memory(self):
        self.memory = ExperienceReplayMemory(self.experience_replay_size) if not self.priority_replay else PrioritizedReplayMemory(self.experience_replay_size, self.priority_alpha, self.priority_beta_start, self.priority_beta_frames)

    def append_to_replay(self, s, a, r, s_):
        self.nstep_buffer.append((s, a, r, s_))

        if(len(self.nstep_buffer)<self.nsteps):
            return
        
        R = sum([self.nstep_buffer[i][2]*(self.gamma**i) for i in range(self.nsteps)])
        state, action, _, _ = self.nstep_buffer.pop(0)

        self.memory.push((state, action, R, s_))

    def prep_minibatch(self):
        # random transition batch is taken from experience replay memory
        transitions, indices, weights = self.memory.sample(self.batch_size)
        
        batch_state, batch_action, batch_reward, batch_next_state = zip(*transitions)

        shape = (-1,)+self.num_feats

        batch_state = torch.tensor(batch_state, device=self.device, dtype=torch.float).view(shape)
        batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.long).squeeze().view(-1, 1)
        batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float).squeeze().view(-1, 1)
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), device=self.device, dtype=torch.uint8)
        try: #sometimes all next states are false
            non_final_next_states = torch.tensor([s for s in batch_next_state if s is not None], device=self.device, dtype=torch.float).view(shape)
            empty_next_state_values = False
        except:
            non_final_next_states = None
            empty_next_state_values = True

        return batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights

    
    def compute_loss(self, batch_vars): #faster
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights = batch_vars

        #estimate
        #self.model.sample_noise()
        current_q_values = self.model(batch_state).gather(1, batch_action)
        
        #target
        with torch.no_grad():
            max_next_q_values = torch.zeros(self.batch_size, device=self.device, dtype=torch.float).unsqueeze(dim=1)
            if not empty_next_state_values:
                max_next_action = self.get_max_next_state_action(non_final_next_states)
                #self.target_model.sample_noise()
                max_next_q_values[non_final_mask] = self.target_model(non_final_next_states).gather(1, max_next_action)
            expected_q_values = batch_reward + ((self.gamma**self.nsteps)*max_next_q_values)

        diff = (expected_q_values - current_q_values)
        if self.priority_replay:
            self.memory.update_priorities(indices, diff.detach().squeeze().abs().cpu().numpy().tolist())
            loss = self.MSE(diff).squeeze() * weights
        else:
            loss = self.MSE(diff)
        loss = loss.mean()

        return loss

    def update(self, s, a, r, s_, frame=0):
        if self.static_policy:
            return None

        self.append_to_replay(s, a, r, s_)

        if frame < self.learn_start or frame % self.update_freq != 0:
            return None

        batch_vars = self.prep_minibatch()

        loss = self.compute_loss(batch_vars)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.update_target_model()
        '''
        self.save_td(loss.item(), frame)
        self.save_sigma_param_magnitudes(frame)
        '''

    def get_action(self, s, eps=0.1): #faster
        with torch.no_grad():
            if np.random.random() >= eps or self.static_policy:# or self.noisy:
                X = torch.tensor([s], device=self.device, dtype=torch.float)
                X = X.unsqueeze(0)
                #self.model.sample_noise()
                a = self.model(X).max(1)[1].view(1, 1)
                return a.item()
            else:
                return np.random.randint(0, self.num_actions)

    def update_target_model(self):
        self.update_count+=1
        self.update_count = self.update_count % self.target_net_update_freq
        if self.update_count == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def get_max_next_state_action(self, next_states):
        return self.model(next_states).max(dim=1)[1].view(-1, 1)

    def finish_nstep(self):
        while len(self.nstep_buffer) > 0:
            R = sum([self.nstep_buffer[i][2]*(self.gamma**i) for i in range(len(self.nstep_buffer))])
            state, action, _, _ = self.nstep_buffer.pop(0)

            self.memory.push((state, action, R, None))

    def reset_hx(self):
        pass

    def MSE(self, x):
        return 0.5 * x.pow(2)
    
class Model_2DQN(BaseAgent):
    def __init__(self, static_policy=False, config=None, shapeOfState=None, sizeOfActions=None):
        super(Model_2DQN, self).__init__(config=config, shapeOfState=shapeOfState, sizeOfActions=sizeOfActions)
        self.device = config.device
        
        self.gamma = config.GAMMA
        self.lr = config.LR
        self.target_net_update_freq = config.TARGET_NET_UPDATE_FREQ
        self.experience_replay_size = config.EXP_REPLAY_SIZE
        self.batch_size = config.BATCH_SIZE
        self.learn_start = config.LEARN_START
        self.update_freq = config.UPDATE_FREQ

        self.static_policy = static_policy
        self.num_feats = shapeOfState # 1 => gray scale, 3 => color scale
        self.num_actions = sizeOfActions
        
        self.declare_networks()
        
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        #move to correct device
        self.model = self.model.to(self.device)
        self.target_model.to(self.device)

        if self.static_policy:
            self.model.eval()
            self.target_model.eval()
        else:
            self.model.train()
            self.target_model.train()

        self.update_count = 0

        self.declare_memory()

    def declare_networks(self):
        self.model = DuelingDQN(self.num_feats, self.num_actions)
        self.target_model = DuelingDQN(self.num_feats, self.num_actions)
        #self.model = DQN(self.num_feats, self.num_actions)
        #self.target_model = DQN(self.num_feats, self.num_actions)
    
    def declare_memory(self):
        self.memory = ExperienceReplayMemory(self.experience_replay_size)
    
    def append_to_replay(self, s, a, r, s_):
        self.memory.push((s, a, r, s_))
    
    def prep_minibatch(self):
        # random transition batch is taken from experience replay memory
        transitions, indices, weights = self.memory.sample(self.batch_size)
        
        batch_state, batch_action, batch_reward, batch_next_state = zip(*transitions)

        shape = (-1,)+self.num_feats

        batch_state = torch.tensor(batch_state, device=self.device, dtype=torch.float).view(shape)
        batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.long).squeeze().view(-1, 1)
        batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float).squeeze().view(-1, 1)
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), device=self.device, dtype=torch.uint8)
        try: #sometimes all next states are false
            non_final_next_states = torch.tensor([s for s in batch_next_state if s is not None], device=self.device, dtype=torch.float).view(shape)
            empty_next_state_values = False
        except:
            non_final_next_states = None
            empty_next_state_values = True

        return batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights
    
    def compute_loss(self, batch_vars):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights = batch_vars

        #estimate
        current_q_values = self.model(batch_state).gather(1, batch_action)
        
        #target
        with torch.no_grad():
            max_next_q_values = torch.zeros(self.batch_size, device=self.device, dtype=torch.float).unsqueeze(dim=1)
            if not empty_next_state_values:
                max_next_action = self.get_max_next_state_action(non_final_next_states)
                max_next_q_values[non_final_mask] = self.target_model(non_final_next_states).gather(1, max_next_action)
            expected_q_values = batch_reward + self.gamma*max_next_q_values

        diff = (expected_q_values - current_q_values)
        loss = self.MSE(diff)
        loss = loss.mean()

        return loss

    def update(self, s, a, r, s_, epi):
        if self.static_policy:
            return None
        
        self.append_to_replay(s, a, r, s_)
        
        if ((epi < self.learn_start) or (epi % self.update_freq != 0)):
            return None

        batch_vars = self.prep_minibatch()

        loss = self.compute_loss(batch_vars)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        self.update_target_model()
        """
        self.save_td(loss.item(), epi)
        self.save_sigma_param_magnitudes(epi)
        """

    def get_action(self, s, eps=0.1):
        with torch.no_grad():
            if ((np.random.random() >= eps) or (self.static_policy)):
                X = torch.tensor([s], device=self.device, dtype=torch.float)
                X = X.unsqueeze(0)
                a = self.model(X).max(1)[1].view(1, 1)
                return a.item()
            else:
                return np.random.randint(0, self.num_actions)

    def update_target_model(self):
        self.update_count+=1
        self.update_count = self.update_count % self.target_net_update_freq
        if self.update_count == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def get_max_next_state_action(self, next_states):
        return self.target_model(next_states).max(dim=1)[1].view(-1, 1)

    def finish_nstep(self):
        pass

    def reset_hx(self):
        pass
    
    def MSE(self, x):
        return 0.5 * x.pow(2)
    

class Model_2DSARSA(BaseAgent):    
    def __init__(self, static_policy=False, config=None, shapeOfState=None, sizeOfActions=None):
        super(Model_2DSARSA, self).__init__(config=config, shapeOfState=shapeOfState, sizeOfActions=sizeOfActions)
        self.device = config.device

        self.priority_replay=config.USE_PRIORITY_REPLAY
        
        self.gamma = config.GAMMA
        self.lr = config.LR
        self.target_net_update_freq = config.TARGET_NET_UPDATE_FREQ
        self.experience_replay_size = config.EXP_REPLAY_SIZE
        self.batch_size = config.BATCH_SIZE
        self.learn_start = config.LEARN_START
        self.update_freq = config.UPDATE_FREQ
        self.priority_beta_start = config.PRIORITY_BETA_START
        self.priority_beta_frames = config.PRIORITY_BETA_FRAMES
        self.priority_alpha = config.PRIORITY_ALPHA
        
        self.static_policy = static_policy
        self.num_feats = shapeOfState
        self.num_actions = sizeOfActions
        
        self.declare_networks()
        
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        #move to correct device
        self.model = self.model.to(self.device)
        self.target_model.to(self.device)
        
        if self.static_policy:
            self.model.eval()
            self.target_model.eval()
        else:
            self.model.train()
            self.target_model.train()
        
        self.update_count = 0
        
        self.declare_memory()
        
    def declare_networks(self):
        self.model = NN(self.num_feats, self.num_actions)
        self.target_model = NN(self.num_feats, self.num_actions)

    def declare_memory(self):
        self.memory = ExperienceReplayMemory(self.experience_replay_size) if not self.priority_replay else PrioritizedReplayMemory(self.experience_replay_size, self.priority_alpha, self.priority_beta_start, self.priority_beta_frames)

    def append_to_replay(self, s, a, r, s_, a_):
        self.memory.push((s, a, r, s_, a_))
    
    def prep_minibatch(self):
        # random transition batch is taken from experience replay memory
        transitions, indices, weights = self.memory.sample(self.batch_size)
        
        batch_state, batch_action, batch_reward, batch_next_state, batch_next_action = zip(*transitions)

        shape = (-1,)+self.num_feats

        batch_state = torch.tensor(batch_state, device=self.device, dtype=torch.float).view(shape)
        batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.long).squeeze().view(-1, 1)
        batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float).squeeze().view(-1, 1)
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), device=self.device, dtype=torch.uint8)
        try: #sometimes all next states are false
            non_final_next_states = torch.tensor([s for s in batch_next_state if s is not None], device=self.device, dtype=torch.float).view(shape)
            empty_next_state_values = False
        except:
            non_final_next_states = None
            empty_next_state_values = True

        batch_next_action = torch.tensor(batch_next_action, device=self.device, dtype=torch.long).squeeze().view(-1, 1)

        return batch_state, batch_action, batch_reward, non_final_next_states, batch_next_action, non_final_mask, empty_next_state_values, indices, weights
    
    def compute_loss(self, batch_vars): #faster
        batch_state, batch_action, batch_reward, non_final_next_states, batch_next_action, non_final_mask, empty_next_state_values, indices, weights = batch_vars

        #estimate
        current_q_values = self.model(batch_state).gather(1, batch_action)
        
        #target
        with torch.no_grad():
            next_q_values = torch.zeros(self.batch_size, device=self.device, dtype=torch.float).unsqueeze(dim=1)
            if not empty_next_state_values:
                next_q_values[non_final_mask] = self.target_model(non_final_next_states).gather(1, batch_next_action)
            expected_q_values = batch_reward + self.gamma*next_q_values
        
        diff = (expected_q_values - current_q_values)
        if self.priority_replay:
            self.memory.update_priorities(indices, diff.detach().squeeze().abs().cpu().numpy().tolist())
            loss = self.MSE(diff).squeeze() * weights
        else:
            loss = self.MSE(diff)
        loss = loss.mean()

        return loss

    def update(self, s, a, r, s_, a_, epi):
        if self.static_policy:
            return None

        self.append_to_replay(s, a, r, s_, a_)

        if ((epi < self.learn_start) or (epi % self.update_freq != 0)):
            return None

        batch_vars = self.prep_minibatch()

        loss = self.compute_loss(batch_vars)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.update_target_model()
        
        '''
        self.save_td(loss.item(), frame)
        self.save_sigma_param_magnitudes(frame)
        '''

    def get_action(self, s, eps=0.1): #faster
        
        with torch.no_grad():
            if ((np.random.random() >= eps) or (self.static_policy)):
                X = torch.tensor([s], device=self.device, dtype=torch.float)
                X = X.unsqueeze(0)
                a = self.model(X).max(1)[1].view(1, 1)
                return a.item()
            else:
                return np.random.randint(0, self.num_actions)
        
    def update_target_model(self):
        self.update_count+=1
        self.update_count = self.update_count % self.target_net_update_freq
        if self.update_count == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def get_max_next_state_action(self, next_states):
        return self.target_model(next_states).max(dim=1)[1].view(-1, 1)

    def finish_nstep(self):
        pass
    
    def reset_hx(self):
        pass
    
    def MSE(self, x):
        return 0.5 * x.pow(2)

class Model_BayesianCNN(BaseAgent):    
    def __init__(self, static_policy=False, config=None, shapeOfState=None, sizeOfActions=None):
        super(Model_BayesianCNN, self).__init__(config=config, shapeOfState=shapeOfState, sizeOfActions=sizeOfActions)
        self.device = config.device

        self.priority_replay=config.USE_PRIORITY_REPLAY

        self.gamma = config.GAMMA
        self.lr = config.LR
        self.target_net_update_freq = config.TARGET_NET_UPDATE_FREQ
        self.experience_replay_size = config.EXP_REPLAY_SIZE
        self.batch_size = config.BATCH_SIZE
        self.learn_start = config.LEARN_START
        self.update_freq = config.UPDATE_FREQ
        self.priority_beta_start = config.PRIORITY_BETA_START
        self.priority_beta_frames = config.PRIORITY_BETA_FRAMES
        self.priority_alpha = config.PRIORITY_ALPHA
        
        self.activation_type = config.ACTIVATION_TYPE
        self.priors = config.PRIORS
        self.layer_type = config.LAYER_TYPE

        self.static_policy = static_policy
        self.num_feats = shapeOfState
        self.num_actions = sizeOfActions

        self.declare_networks()
            
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        #move to correct device
        self.model = self.model.to(self.device)
        self.target_model.to(self.device)

        if self.static_policy:
            self.model.eval()
            self.target_model.eval()
        else:
            self.model.train()
            self.target_model.train()

        self.update_count = 0

        self.declare_memory()

    def declare_networks(self):
        self.model = BayesianCNN(self.num_feats, self.num_actions, self.layer_type, self.activation_type, self.priors)
        self.target_model = BayesianCNN(self.num_feats, self.num_actions, self.layer_type, self.activation_type, self.priors)

    def declare_memory(self):
        self.memory = ExperienceReplayMemory(self.experience_replay_size) if not self.priority_replay else PrioritizedReplayMemory(self.experience_replay_size, self.priority_alpha, self.priority_beta_start, self.priority_beta_frames)

    def append_to_replay(self, s, a, r, s_, a_):
        self.memory.push((s, a, r, s_, a_))
    
    def prep_minibatch(self):
        # random transition batch is taken from experience replay memory
        transitions, indices, weights = self.memory.sample(self.batch_size)
        
        batch_state, batch_action, batch_reward, batch_next_state, batch_next_action = zip(*transitions)

        shape = (-1,)+self.num_feats

        batch_state = torch.tensor(batch_state, device=self.device, dtype=torch.float).view(shape)
        batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.long).squeeze().view(-1, 1)
        batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float).squeeze().view(-1, 1)
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), device=self.device, dtype=torch.uint8)
        try: #sometimes all next states are false
            non_final_next_states = torch.tensor([s for s in batch_next_state if s is not None], device=self.device, dtype=torch.float).view(shape)
            empty_next_state_values = False
        except:
            non_final_next_states = None
            empty_next_state_values = True

        batch_next_action = torch.tensor(batch_next_action, device=self.device, dtype=torch.long).squeeze().view(-1, 1)

        return batch_state, batch_action, batch_reward, non_final_next_states, batch_next_action, non_final_mask, empty_next_state_values, indices, weights
    
    def compute_loss(self, batch_vars): #faster
        batch_state, batch_action, batch_reward, non_final_next_states, batch_next_action, non_final_mask, empty_next_state_values, indices, weights = batch_vars

        #estimate
        current_q_values = self.model(batch_state).gather(1, batch_action)
        
        #target
        with torch.no_grad():
            next_q_values = torch.zeros(self.batch_size, device=self.device, dtype=torch.float).unsqueeze(dim=1)
            if not empty_next_state_values:
                next_q_values[non_final_mask] = self.target_model(non_final_next_states).gather(1, batch_next_action)
            expected_q_values = batch_reward + self.gamma*next_q_values
        
        diff = (expected_q_values - current_q_values)
        if self.priority_replay:
            self.memory.update_priorities(indices, diff.detach().squeeze().abs().cpu().numpy().tolist())
            loss = self.MSE(diff).squeeze() * weights
        else:
            loss = self.MSE(diff)
        loss = loss.mean()

        return loss

    def update(self, s, a, r, s_, a_, frame=0):
        if self.static_policy:
            return None

        self.append_to_replay(s, a, r, s_, a_)

        if frame < self.learn_start or frame % self.update_freq != 0:
            return None

        batch_vars = self.prep_minibatch()

        loss = self.compute_loss(batch_vars)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.update_target_model()
        
        '''
        self.save_td(loss.item(), frame)
        self.save_sigma_param_magnitudes(frame)
        '''
        
    def get_action(self, s, eps=0.1): #faster
        if ((np.random.random() >= eps) or self.static_policy):    
            with torch.no_grad():
                X = torch.tensor([s], device=self.device, dtype=torch.float)
                X = X.unsqueeze(0)
                log_probs = torch.zeros(self.num_actions).to(self.device)
                logits = self.model(X)
                dist = torch.distributions.Categorical(logits=logits)
                
                action_list = torch.zeros(self.num_actions).to(self.device)
                
                for a in range(self.num_actions):
                    actions = dist.sample().view(-1, 1)
                    log_probs = F.log_softmax(logits, dim=1)
                    action_log_probs = log_probs.gather(1, actions)
                    action_list[a] = action_log_probs
                
            return action_list.cpu().data.numpy()
        else:
            return [random.random() for i in range(self.num_actions)]

    def update_target_model(self):
        self.update_count+=1
        self.update_count = self.update_count % self.target_net_update_freq
        if self.update_count == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def get_max_next_state_action(self, next_states):
        return self.target_model(next_states).max(dim=1)[1].view(-1, 1)

    def finish_nstep(self):
        pass
    
    def reset_hx(self):
        pass
    
    def MSE(self, x):
        return 0.5 * x.pow(2)

class Model_BDN(BaseAgent):    
    def __init__(self, static_policy=False, config=None, shapeOfState=None, sizeOfActions=None):
        super(Model_BDN, self).__init__(config=config, shapeOfState=shapeOfState, sizeOfActions=sizeOfActions)
        self.device = config.device

        self.priority_replay=config.USE_PRIORITY_REPLAY

        self.gamma = config.GAMMA
        self.lr = config.LR
        self.target_net_update_freq = config.TARGET_NET_UPDATE_FREQ
        self.experience_replay_size = config.EXP_REPLAY_SIZE
        self.batch_size = config.BATCH_SIZE
        self.learn_start = config.LEARN_START
        self.update_freq = config.UPDATE_FREQ
        self.priority_beta_start = config.PRIORITY_BETA_START
        self.priority_beta_frames = config.PRIORITY_BETA_FRAMES
        self.priority_alpha = config.PRIORITY_ALPHA

        self.static_policy = static_policy
        self.num_feats = shapeOfState
        self.num_actions = sizeOfActions

        self.declare_networks()
            
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        #move to correct device
        self.model = self.model.to(self.device)
        self.target_model.to(self.device)

        if self.static_policy:
            self.model.eval()
            self.target_model.eval()
        else:
            self.model.train()
            self.target_model.train()

        self.update_count = 0

        self.declare_memory()

    def declare_networks(self):
        self.model = BDN(self.num_feats, self.num_actions)
        self.target_model = BDN(self.num_feats, self.num_actions)

    def declare_memory(self):
        self.memory = ExperienceReplayMemory(self.experience_replay_size) if not self.priority_replay else PrioritizedReplayMemory(self.experience_replay_size, self.priority_alpha, self.priority_beta_start, self.priority_beta_frames)

    def append_to_replay(self, s, a, r, s_, a_):
        self.memory.push((s, a, r, s_, a_))
    
    def prep_minibatch(self):
        # random transition batch is taken from experience replay memory
        transitions, indices, weights = self.memory.sample(self.batch_size)
        
        batch_state, batch_action, batch_reward, batch_next_state, batch_next_action = zip(*transitions)

        shape = (-1,)+self.num_feats

        batch_state = torch.tensor(batch_state, device=self.device, dtype=torch.float).view(shape)
        batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.long).squeeze().view(-1, 1)
        batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float).squeeze().view(-1, 1)
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), device=self.device, dtype=torch.uint8)
        try: #sometimes all next states are false
            non_final_next_states = torch.tensor([s for s in batch_next_state if s is not None], device=self.device, dtype=torch.float).view(shape)
            empty_next_state_values = False
        except:
            non_final_next_states = None
            empty_next_state_values = True

        batch_next_action = torch.tensor(batch_next_action, device=self.device, dtype=torch.long).squeeze().view(-1, 1)

        return batch_state, batch_action, batch_reward, non_final_next_states, batch_next_action, non_final_mask, empty_next_state_values, indices, weights
    
    def compute_loss(self, batch_vars): #faster
        batch_state, batch_action, batch_reward, non_final_next_states, batch_next_action, non_final_mask, empty_next_state_values, indices, weights = batch_vars

        #estimate
        current_q_values = self.model(batch_state).gather(1, batch_action)
        
        #target
        with torch.no_grad():
            next_q_values = torch.zeros(self.batch_size, device=self.device, dtype=torch.float).unsqueeze(dim=1)
            if not empty_next_state_values:
                next_q_values[non_final_mask] = self.target_model(non_final_next_states).gather(1, batch_next_action)
            expected_q_values = batch_reward + self.gamma*next_q_values
        
        diff = (expected_q_values - current_q_values)
        if self.priority_replay:
            self.memory.update_priorities(indices, diff.detach().squeeze().abs().cpu().numpy().tolist())
            loss = self.MSE(diff).squeeze() * weights
        else:
            loss = self.MSE(diff)
        loss = loss.mean()

        return loss

    def update(self, s, a, r, s_, a_, frame=0):
        if self.static_policy:
            return None

        self.append_to_replay(s, a, r, s_, a_)

        if frame < self.learn_start or frame % self.update_freq != 0:
            return None

        batch_vars = self.prep_minibatch()

        loss = self.compute_loss(batch_vars)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.update_target_model()
        
        '''
        self.save_td(loss.item(), frame)
        self.save_sigma_param_magnitudes(frame)
        '''

    def get_action(self, s, eps=0.1): #faster
        with torch.no_grad():
            if ((np.random.random() >= eps) or self.static_policy):
                X = torch.tensor([s], device=self.device, dtype=torch.float)
                X = X.unsqueeze(0)
                w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16, w17, w18 = self.model(X)
                return [w1.item(), w2.item(), w3.item(), w4.item(), w5.item(), w6.item(), w7.item(), w8.item(), w9.item(), w10.item(), w11.item(), w12.item(), w13.item(), w14.item(), w15.item(), w16.item(), w17.item(), w18.item()]
            else:
                return [random.random() for i in range(18)]

    def update_target_model(self):
        self.update_count+=1
        self.update_count = self.update_count % self.target_net_update_freq
        if self.update_count == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def get_max_next_state_action(self, next_states):
        return self.target_model(next_states).max(dim=1)[1].view(-1, 1)

    def finish_nstep(self):
        pass
    
    def reset_hx(self):
        pass
    
    def MSE(self, x):
        return 0.5 * x.pow(2)

class Model_A2C(BaseAgent):
    def __init__(self, static_policy=False, config=None, shapeOfState=None, sizeOfActions=None):
        super(Model_A2C, self).__init__(config=config, shapeOfState=shapeOfState, sizeOfActions=sizeOfActions)
        self.device = config.device

        self.noisy=config.USE_NOISY_NETS
        self.priority_replay=config.USE_PRIORITY_REPLAY

        self.gamma = config.GAMMA
        self.lr = config.LR
        self.target_net_update_freq = config.TARGET_NET_UPDATE_FREQ
        self.learn_start = config.LEARN_START
        self.sigma_init= config.SIGMA_INIT
        self.num_agents = config.num_agents
        self.value_loss_weight = config.value_loss_weight
        self.entropy_loss_weight = config.entropy_loss_weight
        self.rollout = config.rollout
        self.grad_norm_max = config.grad_norm_max

        self.static_policy = static_policy
        self.num_feats = shapeOfState
        self.num_actions = sizeOfActions

        self.declare_networks()
            
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr, alpha=0.99, eps=1e-5)
        
        #move to correct device
        self.model = self.model.to(self.device)

        if self.static_policy:
            self.model.eval()
        else:
            self.model.train()

        self.rollouts = RolloutStorage(self.rollout, self.num_agents,
            self.num_feats, self.num_actions, self.device)

        self.value_losses = []
        self.entropy_losses = []
        self.policy_losses = []


    def declare_networks(self):
        self.model = ActorCritic(self.num_feats, self.num_actions)

    def get_action(self, s, deterministic=False):
        with torch.no_grad():
            #X = torch.tensor([s], device=self.device, dtype=torch.float)
            #X = X.unsqueeze(0)
            logits, values = self.model(s)
            dist = torch.distributions.Categorical(logits=logits)
    
            if deterministic:
                actions = dist.probs.argmax(dim=1, keepdim=True)
            else:
                actions = dist.sample().view(-1, 1)
    
            log_probs = F.log_softmax(logits, dim=1)
            action_log_probs = log_probs.gather(1, actions)
            
            return values, actions, action_log_probs        

    def evaluate_actions(self, s, actions):
        logits, values = self.model(s)

        dist = torch.distributions.Categorical(logits=logits)

        log_probs = F.log_softmax(logits, dim=1)
        action_log_probs = log_probs.gather(1, actions)

        dist_entropy = dist.entropy().mean()

        return values, action_log_probs, dist_entropy

    def get_values(self, s):
        _, values = self.model(s)

        return values

    def compute_loss(self, rollouts):
        obs_shape = rollouts.observations.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        values, action_log_probs, dist_entropy = self.evaluate_actions(
            rollouts.observations[:-1].view(-1, *obs_shape),
            rollouts.actions.view(-1, 1))

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()

        action_loss = -(advantages.detach() * action_log_probs).mean()

        loss = action_loss + self.value_loss_weight * value_loss - self.entropy_loss_weight * dist_entropy

        return loss, action_loss, value_loss, dist_entropy

    def update(self, rollout):
        loss, action_loss, value_loss, dist_entropy = self.compute_loss(rollout)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_max)
        self.optimizer.step()

        #self.save_loss(loss.item(), action_loss.item(), value_loss.item(), dist_entropy.item())
        #self.save_sigma_param_magnitudes()

        return value_loss.item(), action_loss.item(), dist_entropy.item()

    '''def save_loss(self, loss, policy_loss, value_loss, entropy_loss):
        super(Model, self).save_loss(loss)
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)
        self.entropy_losses.append(entropy_loss)'''
        
class Model_2DSARSA_CA(BaseAgent):    
    def __init__(self, static_policy=False, config=None, shapeOfState=None, sizeOfActions=None):
        super(Model_2DSARSA_CA, self).__init__(config=config, shapeOfState=shapeOfState, sizeOfActions=sizeOfActions)
        self.device = config.device

        self.priority_replay=config.USE_PRIORITY_REPLAY

        self.gamma = config.GAMMA
        self.lr = config.LR
        self.target_net_update_freq = config.TARGET_NET_UPDATE_FREQ
        self.experience_replay_size = config.EXP_REPLAY_SIZE
        self.batch_size = config.BATCH_SIZE
        self.learn_start = config.LEARN_START
        self.update_freq = config.UPDATE_FREQ
        self.priority_beta_start = config.PRIORITY_BETA_START
        self.priority_beta_frames = config.PRIORITY_BETA_FRAMES
        self.priority_alpha = config.PRIORITY_ALPHA

        self.static_policy = static_policy
        self.num_feats = shapeOfState
        self.num_actions = sizeOfActions

        self.declare_networks()
            
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        #move to correct device
        self.model = self.model.to(self.device)
        self.target_model.to(self.device)

        if self.static_policy:
            self.model.eval()
            self.target_model.eval()
        else:
            self.model.train()
            self.target_model.train()

        self.update_count = 0

        self.declare_memory()

    def declare_networks(self):
        self.model = DuelingDQN_CA(self.num_feats, self.num_actions)
        self.target_model = DuelingDQN_CA(self.num_feats, self.num_actions)

    def declare_memory(self):
        self.memory = ExperienceReplayMemory(self.experience_replay_size) if not self.priority_replay else PrioritizedReplayMemory(self.experience_replay_size, self.priority_alpha, self.priority_beta_start, self.priority_beta_frames)

    def append_to_replay(self, s, a, r, s_, a_):
        self.memory.push((s, a, r, s_, a_))
    
    def prep_minibatch(self):
        # random transition batch is taken from experience replay memory
        transitions, indices, weights = self.memory.sample(self.batch_size)
        
        batch_state, batch_action, batch_reward, batch_next_state, batch_next_action = zip(*transitions)

        state_shape = (-1,)+self.num_feats
        action_shape = (-1,)+(1, 1, self.num_actions)

        batch_state = torch.tensor(batch_state, device=self.device, dtype=torch.float).view(state_shape)
        batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.float).view(action_shape)
        batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float).squeeze().view(-1, 1)
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), device=self.device, dtype=torch.uint8)
        try: #sometimes all next states are false
            non_final_next_states = torch.tensor([s for s in batch_next_state if s is not None], device=self.device, dtype=torch.float).view(shape)
            empty_next_state_values = False
        except:
            non_final_next_states = None
            empty_next_state_values = True

        batch_next_action = torch.tensor(batch_next_action, device=self.device, dtype=torch.float).view(action_shape)

        return batch_state, batch_action, batch_reward, non_final_next_states, batch_next_action, non_final_mask, empty_next_state_values, indices, weights
    
    def compute_loss(self, batch_vars): #faster
        batch_state, batch_action, batch_reward, non_final_next_states, batch_next_action, non_final_mask, empty_next_state_values, indices, weights = batch_vars

        #estimate
        current_q_values, _ = self.model(batch_state)
        
        #target
        with torch.no_grad():
            next_q_values = torch.zeros(self.batch_size, device=self.device, dtype=torch.float).unsqueeze(dim=1)
            if not empty_next_state_values:
                next_q_values[non_final_mask], _ = self.target_model(non_final_next_states)
            expected_q_values = batch_reward + self.gamma*next_q_values
        
        diff = (expected_q_values - current_q_values)
        if self.priority_replay:
            self.memory.update_priorities(indices, diff.detach().squeeze().abs().cpu().numpy().tolist())
            loss = self.MSE(diff).squeeze() * weights
        else:
            loss = self.MSE(diff)
        loss = loss.mean()

        return loss

    def update(self, s, a, r, s_, a_, frame=0):
        if self.static_policy:
            return None

        self.append_to_replay(s, a, r, s_, a_)

        if frame < self.learn_start or frame % self.update_freq != 0:
            return None

        batch_vars = self.prep_minibatch()

        loss = self.compute_loss(batch_vars)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.update_target_model()
        
        '''
        self.save_td(loss.item(), frame)
        self.save_sigma_param_magnitudes(frame)
        '''

    def get_action(self, s, eps=0.1): #faster
        with torch.no_grad():
            if ((np.random.random() >= eps) or (self.static_policy)):
                X = torch.tensor([s], device=self.device, dtype=torch.float)
                X = X.unsqueeze(0)
                _, a = self.model(X)
                a = a.cpu().data.numpy()
                return a
            else:
                return np.array([random.random() for i in range(self.num_actions)]).reshape(1, self.num_actions)

    def update_target_model(self):
        self.update_count+=1
        self.update_count = self.update_count % self.target_net_update_freq
        if self.update_count == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def get_max_next_state_action(self, next_states):
        return self.target_model(next_states).max(dim=1)[1].view(-1, 1)

    def finish_nstep(self):
        pass
    
    def reset_hx(self):
        pass
    
    def MSE(self, x):
        return 0.5 * x.pow(2)


class Model_DSARSA_CA(BaseAgent):
    def __init__(self, static_policy=False, config=None, shapeOfState=None, sizeOfActions=None):
        super(Model_DSARSA_CA, self).__init__(config=config, shapeOfState=shapeOfState, sizeOfActions=sizeOfActions)
        self.device = config.device
        
        self.gamma = config.GAMMA
        self.lr = config.LR
        self.LR_ACTOR = 1e-3         # learning rate of the actor 
        self.LR_CRITIC = 1e-3        # learning rate of the critic
        self.WEIGHT_DECAY = 0.0000   # L2 weight decay
        self.TAU = 1e-3              # for soft update of target parameters
        self.target_net_update_freq = config.TARGET_NET_UPDATE_FREQ
        self.experience_replay_size = config.EXP_REPLAY_SIZE
        self.batch_size = config.BATCH_SIZE
        self.learn_start = config.LEARN_START
        self.update_freq = config.UPDATE_FREQ

        self.static_policy = static_policy
        self.num_feats = shapeOfState # 1 => gray scale, 3 => color scale
        self.num_actions = sizeOfActions
        
        self.declare_networks()
        
        self.actor_target.load_state_dict(self.actor_net.state_dict())
        self.critic_target.load_state_dict(self.critic_net.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=self.LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=self.LR_CRITIC, weight_decay=self.WEIGHT_DECAY)
        
        #move to correct device
        self.actor_net = self.actor_net.to(self.device)
        self.critic_net = self.critic_net.to(self.device)
        self.actor_target.to(self.device)
        self.critic_target.to(self.device)

        if self.static_policy:
            self.actor_net.eval()
            self.critic_net.eval()
            self.actor_target.eval()
            self.critic_target.eval()
        else:
            self.actor_net.train()
            self.critic_net.train()
            self.actor_target.train()
            self.critic_target.train()

        self.update_count = 0

        self.declare_memory()

    def declare_networks(self):
        self.actor_net = CNN_Actor(self.num_feats, self.num_actions)
        self.actor_target = CNN_Actor(self.num_feats, self.num_actions)
        
        self.critic_net = CNN_Critic(self.num_feats, self.num_actions)
        self.critic_target = CNN_Critic(self.num_feats, self.num_actions)        
    
    def declare_memory(self):
        self.memory = ExperienceReplayMemory(self.experience_replay_size)
        #self.memory = ReplayBuffer(self.num_actions, self.experience_replay_size, self.batch_size)
        
    def append_to_replay(self, s, a, r, s_, a_):
        self.memory.push((s, a, r, s_, a_))
        #self.memory.add(s, a, r, s_, a_)
    
    def prep_minibatch(self):
        # random transition batch is taken from experience replay memory
        transitions, indices, weights = self.memory.sample(self.batch_size)
        
        batch_state, batch_action, batch_reward, batch_next_state, batch_next_action = zip(*transitions)

        shape = (-1,)+self.num_feats

        batch_state = torch.tensor(batch_state, device=self.device, dtype=torch.float).view(shape)
        batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.float).squeeze()
        batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float).squeeze().view(-1, 1)
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), device=self.device, dtype=torch.uint8)
        try: #sometimes all next states are false
            non_final_next_states = torch.tensor([s for s in batch_next_state if s is not None], device=self.device, dtype=torch.float).view(shape)
            empty_next_state_values = False
        except:
            non_final_next_states = None
            empty_next_state_values = True
        
        batch_next_action = torch.tensor(batch_next_action, device=self.device, dtype=torch.float).squeeze()

        return batch_state, batch_action, batch_reward, non_final_next_states, batch_next_action, non_final_mask, empty_next_state_values, indices, weights
    
    def compute_loss(self, batch_vars):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        
        batch_state, batch_action, batch_reward, non_final_next_states, batch_next_action, non_final_mask, empty_next_state_values, indices, weights = batch_vars
        
        #print(batch_action)
        #print(batch_action.shape)
        Q_expected = self.critic_net(batch_state, batch_action)
        
        # Get predicted next-state actions and Q values from target models
        with torch.no_grad():
            #actions_next = self.actor_target(non_final_next_states)
            #print(actions_next)
            #print(actions_next.shape)
            
            if not empty_next_state_values:
                Q_targets_next = self.critic_target(non_final_next_states, batch_next_action)
                
            # Compute Q targets for current states (y_i)
            Q_targets = batch_reward + (self.gamma * Q_targets_next)
            # Compute critic loss
        
        c_loss = F.mse_loss(Q_expected, Q_targets)
        
        # Compute actor loss
        #actions_pred = self.actor_net(batch_state)
        a_loss = -self.critic_net(batch_state, batch_action).mean()
        
        return c_loss, a_loss
    
    def update(self, s, a, r, s_, a_, epi):
        if self.static_policy:
            return None
        
        self.append_to_replay(s, a, r, s_, a_)
        
        if ((epi < self.learn_start) or (epi % self.update_freq != 0)):
            return None

        batch_vars = self.prep_minibatch()
        
        critic_loss, actor_loss = self.compute_loss(batch_vars)
        

        # Optimize the model
        self.critic_optimizer.zero_grad()
        self.actor_optimizer.zero_grad()
        
        critic_loss.backward()
        actor_loss.backward()
        
        torch.nn.utils.clip_grad_norm(self.critic_net.parameters(), 1)
        torch.nn.utils.clip_grad_norm(self.actor_net.parameters(), 1)
        
        self.critic_optimizer.step()
        self.actor_optimizer.step()
        """
        self.save_td(loss.item(), epi)
        self.save_sigma_param_magnitudes(epi)
        """

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_net, self.critic_target, self.TAU)
        self.soft_update(self.actor_net, self.actor_target, self.TAU) 
    
    def soft_update(self, net, target, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            net: PyTorch model (weights will be copied from)
            target: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, net_param in zip(target.parameters(), net.parameters()):
            target_param.data.copy_(tau*net_param.data + (1.0-tau)*target_param.data)

    
    def get_action(self, s, eps=0.1):
        with torch.no_grad():
            if ((np.random.random() >= eps) or (self.static_policy)):
                X = torch.tensor([s], device=self.device, dtype=torch.float)
                X = X.unsqueeze(0)
                a = self.actor_net(X).cpu().data.numpy()
                return a
            else:
                return np.random.rand(1, self.num_actions)
    
    def update_target_model(self):
        self.update_count+=1
        self.update_count = self.update_count % self.target_net_update_freq
        if self.update_count == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def get_max_next_state_action(self, next_states):
        return self.target_model(next_states).max(dim=1)[1].view(-1, 1)

    def finish_nstep(self):
        pass

    def reset_hx(self):
        pass
    
    def MSE(self, x):
        return 0.5 * x.pow(2)
    
    def save_w_ca(self, modelName):
        torch.save(self.actor_net.state_dict(), './saved_agents/act_model_' + modelName + '.pth')
        torch.save(self.actor_optimizer.state_dict(), './saved_agents/act_optim_' + modelName + '.pth')
        
        torch.save(self.critic_net.state_dict(), './saved_agents/cri_model_' + modelName + '.pth')
        torch.save(self.critic_optimizer.state_dict(), './saved_agents/cri_optim_' + modelName + '.pth')
    
    def load_w_ca(self, modelName):
        fname_act_model = "./saved_agents/act_model_" +  modelName + ".pth"
        fname_act_optim = "./saved_agents/act_optim_" +  modelName + ".pth"
        
        fname_cri_model = "./saved_agents/cri_model_" +  modelName + ".pth"
        fname_cri_optim = "./saved_agents/cri_optim_" +  modelName + ".pth"

        if os.path.isfile(fname_act_model):
            self.actor_net.load_state_dict(torch.load(fname_act_model, map_location='cpu'))
            self.actor_target.load_state_dict(self.actor_net.state_dict())    

        if os.path.isfile(fname_act_optim):
            self.actor_optimizer.load_state_dict(torch.load(fname_act_optim, map_location='cpu'))
        
        if os.path.isfile(fname_cri_model):
            self.critic_net.load_state_dict(torch.load(fname_cri_model, map_location='cpu'))
            self.critic_target.load_state_dict(self.critic_net.state_dict())    

        if os.path.isfile(fname_cri_optim):
            self.critic_optimizer.load_state_dict(torch.load(fname_cri_optim, map_location='cpu'))
    

class Model_DSARSA(BaseAgent):
    def __init__(self, static_policy=False, config=None, shapeOfState=None, sizeOfActions=None):
        super(Model_DSARSA, self).__init__(config=config, shapeOfState=shapeOfState, sizeOfActions=sizeOfActions)
        self.device = config.device
        
        self.gamma = config.GAMMA
        self.lr = config.LR
        self.target_net_update_freq = config.TARGET_NET_UPDATE_FREQ
        self.experience_replay_size = config.EXP_REPLAY_SIZE
        self.batch_size = config.BATCH_SIZE
        self.learn_start = config.LEARN_START
        self.update_freq = config.UPDATE_FREQ

        self.static_policy = static_policy
        self.num_feats = shapeOfState # 1 => gray scale, 3 => color scale
        self.num_actions = sizeOfActions
        
        self.declare_networks()
        
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        #move to correct device
        self.model = self.model.to(self.device)
        self.target_model.to(self.device)

        if self.static_policy:
            self.model.eval()
            self.target_model.eval()
        else:
            self.model.train()
            self.target_model.train()

        self.update_count = 0

        self.declare_memory()

    def declare_networks(self):
        self.model = CNN(self.num_feats, self.num_actions)
        self.target_model = CNN(self.num_feats, self.num_actions)
        #self.model = DQN(self.num_feats, self.num_actions)
        #self.target_model = DQN(self.num_feats, self.num_actions)
    
    def declare_memory(self):
        self.memory = ExperienceReplayMemory(self.experience_replay_size)
    
    def append_to_replay(self, s, a, r, s_, a_):
        self.memory.push((s, a, r, s_, a_))
    
    def prep_minibatch(self):
        # random transition batch is taken from experience replay memory
        transitions, indices, weights = self.memory.sample(self.batch_size)
        
        batch_state, batch_action, batch_reward, batch_next_state, batch_next_action = zip(*transitions)

        shape = (-1,)+self.num_feats

        batch_state = torch.tensor(batch_state, device=self.device, dtype=torch.float).view(shape)
        batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.long).squeeze().view(-1, 1)
        batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float).squeeze().view(-1, 1)
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), device=self.device, dtype=torch.uint8)
        try: #sometimes all next states are false
            non_final_next_states = torch.tensor([s for s in batch_next_state if s is not None], device=self.device, dtype=torch.float).view(shape)
            empty_next_state_values = False
        except:
            non_final_next_states = None
            empty_next_state_values = True
        
        batch_next_action = torch.tensor(batch_next_action, device=self.device, dtype=torch.long).squeeze().view(-1, 1)

        return batch_state, batch_action, batch_reward, non_final_next_states, batch_next_action, non_final_mask, empty_next_state_values, indices, weights
    
    def compute_loss(self, batch_vars):
        batch_state, batch_action, batch_reward, non_final_next_states, batch_next_action, non_final_mask, empty_next_state_values, indices, weights = batch_vars

        #estimate
        current_q_values = self.model(batch_state).gather(1, batch_action)
        
        #target
        with torch.no_grad():
            next_q_values = torch.zeros(self.batch_size, device=self.device, dtype=torch.float).unsqueeze(dim=1)
            if not empty_next_state_values:
                next_q_values[non_final_mask] = self.target_model(non_final_next_states).gather(1, batch_next_action)
            expected_q_values = batch_reward + self.gamma*next_q_values

        diff = (expected_q_values - current_q_values)
        loss = self.MSE(diff)
        loss = loss.mean()

        return loss

    def update(self, s, a, r, s_, a_, epi):
        if self.static_policy:
            return None
        
        self.append_to_replay(s, a, r, s_, a_)
        
        if ((epi < self.learn_start) or (epi % self.update_freq != 0)):
            return None

        batch_vars = self.prep_minibatch()

        loss = self.compute_loss(batch_vars)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        self.update_target_model()
        """
        self.save_td(loss.item(), epi)
        self.save_sigma_param_magnitudes(epi)
        """

    def get_action(self, s, eps=0.1):
        with torch.no_grad():
            if ((np.random.random() >= eps) or (self.static_policy)):
                X = torch.tensor([s], device=self.device, dtype=torch.float)
                X = X.unsqueeze(0)
                a = self.model(X).max(1)[1].view(1, 1)
                return a.item()
            else:
                return np.random.randint(0, self.num_actions)

    def update_target_model(self):
        self.update_count+=1
        self.update_count = self.update_count % self.target_net_update_freq
        if self.update_count == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def get_max_next_state_action(self, next_states):
        return self.target_model(next_states).max(dim=1)[1].view(-1, 1)

    def finish_nstep(self):
        pass

    def reset_hx(self):
        pass
    
    def MSE(self, x):
        return 0.5 * x.pow(2)
    
    

class Model_N_Step_DQN(BaseAgent):
    def __init__(self, static_policy=False, config=None, shapeOfState=None, sizeOfActions=None):
        super(Model_N_Step_DQN, self).__init__(config=config, shapeOfState=shapeOfState, sizeOfActions=sizeOfActions)
        self.device = config.device
        
        self.gamma = config.GAMMA
        self.lr = config.LR
        self.target_net_update_freq = config.TARGET_NET_UPDATE_FREQ
        self.experience_replay_size = config.EXP_REPLAY_SIZE
        self.batch_size = config.BATCH_SIZE
        self.learn_start = config.LEARN_START
        self.update_freq = config.UPDATE_FREQ

        self.static_policy = static_policy
        self.num_feats = shapeOfState # 1 => gray scale, 3 => color scale
        self.num_actions = sizeOfActions
        
        self.declare_networks()
        
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        #move to correct device
        self.model = self.model.to(self.device)
        self.target_model.to(self.device)

        if self.static_policy:
            self.model.eval()
            self.target_model.eval()
        else:
            self.model.train()
            self.target_model.train()

        self.update_count = 0

        self.declare_memory()
        
        self.nsteps = config.N_STEPS
        self.nstep_buffer = []

    def declare_networks(self):
        self.model = CNN(self.num_feats, self.num_actions)
        self.target_model = CNN(self.num_feats, self.num_actions)
        
    def declare_memory(self):
        self.memory = ExperienceReplayMemory(self.experience_replay_size)
    
    def append_to_replay(self, s, a, r, s_):
        self.nstep_buffer.append((s, a, r, s_))

        if(len(self.nstep_buffer)<self.nsteps):
            return
        
        R = sum([self.nstep_buffer[i][2]*(self.gamma**i) for i in range(self.nsteps)])
        state, action, _, _ = self.nstep_buffer.pop(0)
        
        self.memory.push((state, action, R, s_))
        
    def prep_minibatch(self):
        # random transition batch is taken from experience replay memory
        transitions, indices, weights = self.memory.sample(self.batch_size)
        
        batch_state, batch_action, batch_reward, batch_next_state = zip(*transitions)

        shape = (-1,)+self.num_feats

        batch_state = torch.tensor(batch_state, device=self.device, dtype=torch.float).view(shape)
        batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.long).squeeze().view(-1, 1)
        batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float).squeeze().view(-1, 1)
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), device=self.device, dtype=torch.uint8)
        try: #sometimes all next states are false
            non_final_next_states = torch.tensor([s for s in batch_next_state if s is not None], device=self.device, dtype=torch.float).view(shape)
            empty_next_state_values = False
        except:
            non_final_next_states = None
            empty_next_state_values = True

        return batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights
    
    def compute_loss(self, batch_vars):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights = batch_vars

        #estimate
        current_q_values = self.model(batch_state).gather(1, batch_action)
        
        #target
        with torch.no_grad():
            max_next_q_values = torch.zeros(self.batch_size, device=self.device, dtype=torch.float).unsqueeze(dim=1)
            if not empty_next_state_values:
                max_next_action = self.get_max_next_state_action(non_final_next_states)
                max_next_q_values[non_final_mask] = self.target_model(non_final_next_states).gather(1, max_next_action)
            expected_q_values = batch_reward + ((self.gamma**self.nsteps)*max_next_q_values)

        diff = (expected_q_values - current_q_values)
        loss = self.MSE(diff)
        loss = loss.mean()

        return loss

    def update(self, s, a, r, s_, epi):
        if self.static_policy:
            return None
        
        self.append_to_replay(s, a, r, s_)
        
        if ((epi < self.learn_start) or (epi % self.update_freq != 0)):
            return None

        batch_vars = self.prep_minibatch()

        loss = self.compute_loss(batch_vars)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        self.update_target_model()
        """
        self.save_td(loss.item(), epi)
        self.save_sigma_param_magnitudes(epi)
        """

    def get_action(self, s, eps=0.1):
        with torch.no_grad():
            if ((np.random.random() >= eps) or (self.static_policy)):
                X = torch.tensor([s], device=self.device, dtype=torch.float)
                X = X.unsqueeze(0)
                a = self.model(X).max(1)[1].view(1, 1)
                return a.item()
            else:
                return np.random.randint(0, self.num_actions)

    def update_target_model(self):
        self.update_count+=1
        self.update_count = self.update_count % self.target_net_update_freq
        if self.update_count == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def get_max_next_state_action(self, next_states):
        return self.target_model(next_states).max(dim=1)[1].view(-1, 1)

    def finish_nstep(self):
        while len(self.nstep_buffer) > 0:
            R = sum([self.nstep_buffer[i][2]*(self.gamma**i) for i in range(len(self.nstep_buffer))])
            state, action, _, _ = self.nstep_buffer.pop(0)

            self.memory.push((state, action, R, None))

    def reset_hx(self):
        pass
    
    def MSE(self, x):
        return 0.5 * x.pow(2)
    
    

class Model_DQN(BaseAgent):
    def __init__(self, static_policy=False, config=None, shapeOfState=None, sizeOfActions=None):
        super(Model_DQN, self).__init__(config=config, shapeOfState=shapeOfState, sizeOfActions=sizeOfActions)
        self.device = config.device
        
        self.gamma = config.GAMMA
        self.lr = config.LR
        self.target_net_update_freq = config.TARGET_NET_UPDATE_FREQ
        self.experience_replay_size = config.EXP_REPLAY_SIZE
        self.batch_size = config.BATCH_SIZE
        self.learn_start = config.LEARN_START
        self.update_freq = config.UPDATE_FREQ

        self.static_policy = static_policy
        self.num_feats = shapeOfState # 1 => gray scale, 3 => color scale
        self.num_actions = sizeOfActions
        
        self.declare_networks()
        
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        #move to correct device
        self.model = self.model.to(self.device)
        self.target_model.to(self.device)

        if self.static_policy:
            self.model.eval()
            self.target_model.eval()
        else:
            self.model.train()
            self.target_model.train()

        self.update_count = 0

        self.declare_memory()

    def declare_networks(self):
        self.model = CNN(self.num_feats, self.num_actions)
        self.target_model = CNN(self.num_feats, self.num_actions)
        #self.model = DQN(self.num_feats, self.num_actions)
        #self.target_model = DQN(self.num_feats, self.num_actions)
    
    def declare_memory(self):
        self.memory = ExperienceReplayMemory(self.experience_replay_size)
    
    def append_to_replay(self, s, a, r, s_):
        self.memory.push((s, a, r, s_))
    
    def prep_minibatch(self):
        # random transition batch is taken from experience replay memory
        transitions, indices, weights = self.memory.sample(self.batch_size)
        
        batch_state, batch_action, batch_reward, batch_next_state = zip(*transitions)

        shape = (-1,)+self.num_feats

        batch_state = torch.tensor(batch_state, device=self.device, dtype=torch.float).view(shape)
        batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.long).squeeze().view(-1, 1)
        batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float).squeeze().view(-1, 1)
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch_next_state)), device=self.device, dtype=torch.uint8)
        try: #sometimes all next states are false
            non_final_next_states = torch.tensor([s for s in batch_next_state if s is not None], device=self.device, dtype=torch.float).view(shape)
            empty_next_state_values = False
        except:
            non_final_next_states = None
            empty_next_state_values = True

        return batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights
    
    def compute_loss(self, batch_vars):
        batch_state, batch_action, batch_reward, non_final_next_states, non_final_mask, empty_next_state_values, indices, weights = batch_vars

        #estimate
        current_q_values = self.model(batch_state).gather(1, batch_action)
        
        #target
        with torch.no_grad():
            max_next_q_values = torch.zeros(self.batch_size, device=self.device, dtype=torch.float).unsqueeze(dim=1)
            if not empty_next_state_values:
                max_next_action = self.get_max_next_state_action(non_final_next_states)
                max_next_q_values[non_final_mask] = self.target_model(non_final_next_states).gather(1, max_next_action)
            expected_q_values = batch_reward + self.gamma*max_next_q_values

        diff = (expected_q_values - current_q_values)
        loss = self.MSE(diff)
        loss = loss.mean()

        return loss

    def update(self, s, a, r, s_, epi):
        if self.static_policy:
            return None
        
        self.append_to_replay(s, a, r, s_)
        
        if ((epi < self.learn_start) or (epi % self.update_freq != 0)):
            return None

        batch_vars = self.prep_minibatch()

        loss = self.compute_loss(batch_vars)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        self.update_target_model()
        """
        self.save_td(loss.item(), epi)
        self.save_sigma_param_magnitudes(epi)
        """

    def get_action(self, s, eps=0.1):
        with torch.no_grad():
            if ((np.random.random() >= eps) or (self.static_policy)):
                X = torch.tensor([s], device=self.device, dtype=torch.float)
                X = X.unsqueeze(0)
                a = self.model(X).max(1)[1].view(1, 1)
                return a.item()
            else:
                return np.random.randint(0, self.num_actions)

    def update_target_model(self):
        self.update_count+=1
        self.update_count = self.update_count % self.target_net_update_freq
        if self.update_count == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    def get_max_next_state_action(self, next_states):
        return self.target_model(next_states).max(dim=1)[1].view(-1, 1)

    def finish_nstep(self):
        pass

    def reset_hx(self):
        pass
    
    def MSE(self, x):
        return 0.5 * x.pow(2)
    
    