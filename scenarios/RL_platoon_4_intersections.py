#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 01:34:51 2020

@author: ccyen
"""

import numpy as np
import math

import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy as sp
import scipy.io as sio

import os
import sys
import os.path as op

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import copy as cp
import itertools
import argparse
import pandas as pd

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

import traci

#sys.path.append(os.path.join(os.path.dirname(__file__),'/C3PO_DRL'))
#sys.path.append("/Users/ccyen/.spyder-py3/C3PO_DRL_FUEL_OPT/C3PO_DRL/")
#sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)),'C3PO_DRL/'))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from sumo_rl_src.environment.env import SumoEnvironment
from sumo_rl_src.agents.ql_agent import QLAgent
from sumo_rl_src.exploration.epsilon_greedy import EpsilonGreedy

from C3PO_DRL.C3PO_PlatoonAgent import Model_DQN, Model_N_Step_DQN, Model_DSARSA, Model_DSARSA_CA, Model_2DSARSA, Model_3DQN, Model_2DQN
from C3PO_DRL.utils.hyperparameters import Config

if __name__ == '__main__':
    
    alpha = 0.1
    gamma = 0.99
    decay = 1
    runs = 1000
    
    env = SumoEnvironment(net_file='../maps/SingleIntersections/single-intersection.net.xml',
                              route_file='../maps/SingleIntersections/single-intersection.rou.xml',
                              use_gui=True,
                              num_seconds=40000,
                              max_depart_delay=0,
                              time_to_load_vehicles=300,
                              num_veh_plat=5,
                              gap_between_vehicles=1,
                              phases=[
                                traci.trafficlight.Phase(args.ns, "GGrr"),   # north-south
                                traci.trafficlight.Phase(2, "yyrr"),
                                traci.trafficlight.Phase(args.we, "rrGG"),   # west-east
                                traci.trafficlight.Phase(2, "rryy")
                                ])
    
    
    R_history = []
    Avg_delay_history = []
    Avg_queue_history = []
    
    for episode in range(1, runs + 1):
        
        episode_reward = 0.0
        
        initial_states = env.reset()
        
        ql_agent = QLAgent(starting_state=env.encode(initial_states),
                           state_space=env.observation_space,
                           action_space=env.action_space,
                           alpha=alpha,
                           gamma=gamma,
                           exploration_strategy=EpsilonGreedy(initial_epsilon=0.05, min_epsilon=0.005, decay=decay))
        infos = []
        done = {'__all__': False}
        while not done['__all__']:
            
            action = ql_agent.act()

            state, reward, done, info = env.step(action=action)
            infos.append(info)
            
            episode_reward += reward

            ql_agent.learn(next_state=env.encode(state), reward=reward)
        
        env.close()
        
        """"" Save Data and Save history """""
        R_history.append(episode_reward)
        sio.savemat('../outputs/RL_platoon/RL_platoon_result.mat', {'R_history':R_history, 'Avg_delay_history': Avg_delay_history, 'Avg_queue_history_drl':Avg_queue_history})
        
        
        print('Episode: %d' %episode)
        print('Reward: %.2f' %episode_reward)
        #print('Avg Queue: %d' %np.sum(Avg_total_queue_length))

        df = pd.DataFrame(infos)
        df.to_csv('../outputs/RL_platoon/c3podrl_alpha{}_gamma{}_decay{}_episode{}.csv'.format(alpha, gamma, decay, episode), index=False)

