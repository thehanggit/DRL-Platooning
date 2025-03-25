#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 01:34:51 2020

@author: ccyen
"""

import numpy as np
import math
# import random

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

# sys.path.append(os.path.join(os.path.dirname(__file__),'/C3PO_DRL'))
# sys.path.append("/Users/ccyen/.spyder-py3/C3PO_DRL_FUEL_OPT/C3PO_DRL/")
# sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)),'C3PO_DRL/'))
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from sumo_rl_src.environment.platoon_env import PlatoonEnvironment

from C3PO_DRL.C3PO_PlatoonAgent import Model_DQN, Model_N_Step_DQN, Model_DSARSA, Model_DSARSA_CA, Model_2DSARSA, \
    Model_3DQN, Model_2DQN
from C3PO_DRL.utils.hyperparameters import Config

##### DQN Parameters #####
config = Config()

config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# epsilon variables
config.epsilon_start = 1.0
config.epsilon_final = 0.01
config.epsilon_decay = 300  # 30000
config.epsilon_by_time_slot = lambda ep: config.epsilon_final + (
            config.epsilon_start - config.epsilon_final) * math.exp(-1. * ep / config.epsilon_decay)

# misc agent variables
config.GAMMA = 0.99
config.LR = 1e-4

# memory
config.TARGET_NET_UPDATE_FREQ = 1000  # 1000
config.EXP_REPLAY_SIZE = 300000  # 100000
config.BATCH_SIZE = 1024  # 32

# For 3D
config.USE_NOISY_NETS = False
config.SIGMA_INIT = 0.5
config.USE_PRIORITY_REPLAY = False
config.PRIORITY_ALPHA = 0.6
config.PRIORITY_BETA_START = 0.4
config.PRIORITY_BETA_FRAMES = 100000

# Learning control variables
config.LEARN_START = 100
config.MAX_EPISODES = 1
config.UPDATE_FREQ = 1

# Nstep controls
config.N_STEPS = 3

output_folder = '../outputs/RL_platoon/'

if __name__ == '__main__':

    env = PlatoonEnvironment(net_file='../maps/SingleIntersection/network_scale/corridor.net.xml',
                             route_file='../maps/SingleIntersection/network_scale/corridor.rou.xml',
                             cfg_file='../maps/SingleIntersection/network_scale/corridor.sumocfg',
                             use_gui=True,
                             direct_start=True,
                             num_seconds=600,
                             max_depart_delay=0,
                             time_to_load_vehicles=10,
                             num_green_phases=2,
                             phases=[
                                 traci.trafficlight.Phase(30, "GGGgrrrrGGGgrrrr"),  # north-south
                                 traci.trafficlight.Phase(2, "yyyyrrrryyyyrrrr"),
                                 traci.trafficlight.Phase(30, "rrrrGGGgrrrrGGGg"),  # west-east
                                 traci.trafficlight.Phase(2, "rrrryyyyrrrryyyy")
                             ])

    # traci.gui.setZoom(traci.gui.DEFAULT_VIEW, 150)

    model_name = "2DSARSA"

    model = Model_2DSARSA(config=config, shapeOfState=env.observation_space, sizeOfActions=env.action_space)

    model.load_w(model_name)
    model.model.eval()
    model.target_model.eval()

    R_history = []
    Delay_history = []
    Fuel_history = []

    save_per_step_time = 100
    step_time_slot = 2
    num_step_time = (int)(save_per_step_time / step_time_slot)

    input_size = env.observation_space[0]

    # for episode in range(1, config.MAX_EPISODES + 1):

    epsilon = config.epsilon_by_time_slot(0)

    initial_states = env.reset()

    cur_state = initial_states

    for p in cur_state:
        if len(cur_state[p]) < input_size:
            cur_state[p] += [0 for i in range(input_size - len(cur_state[p]))]

    # cur_state = random.sample(range(1, 1000), 152)

    prev_action = {}
    for p in cur_state:
        prev_action[p] = 0

    all_rewards = []
    all_delays = []
    all_fuels = []

    done = {'__all__': False}
    while not done['__all__']:

        cur_action = {}
        for p in cur_state:
            cur_action[p] = model.get_action(cur_state[p], epsilon)

        prev_state = cp.deepcopy(cur_state)

        env.no_op_step(step_time_slot)

        cur_state, reward, done, info = env.step(joint_actions=cur_action)

        for p in cur_state:
            if len(cur_state[p]) < input_size:
                cur_state[p] += [cur_state[p][-1] for i in range(input_size - len(cur_state[p]))]

        # epsilon = config.epsilon_by_time_slot(episode)

        for p in cur_state:
            prev_s = prev_state[p] if p in prev_state else [0 for i in range(input_size)]
            prev_a = prev_action[p] if p in prev_action else 2  # No-op
            r = reward[p] if p in reward else 0
            cur_s = cur_state[p]
            cur_a = cur_action[p] if p in cur_action else 2  # No-op

            # model.update(prev_s, prev_a, r, cur_s, cur_a, episode)

        ##### For N DQN #####
        # model.update(prev_state, cur_action, reward, cur_state, episode)

        all_rewards.append(sum(reward.values()))
        all_delays.append(info['avg_wait_time'])
        all_fuels.append(info['avg_fuel_consumption'])

        prev_action = cp.deepcopy(cur_action)

        # episode_reward = np.mean(all_rewards)
        # episode_delay = np.mean(all_delays)
        # episode_fuel = np.mean(all_fuels)

        # print('Episode: %d, Epsilon: %f' % (episode, epsilon))
        # print('Average reward: %f' % episode_reward)
        # print('Average delay: %.2f' % episode_delay)
        # print('Average fuel: %.2f' % episode_fuel)
        #
        # ### Save Data and Save history ###
        # R_history.append(episode_reward)
        # Delay_history.append(episode_delay)
        # Fuel_history.append(episode_fuel)
        # sio.savemat(output_folder + 'RL_platoon_result.mat',
        #             {'R_history': R_history, 'Delay_history': Delay_history, 'Fuel_history': Fuel_history})
        #
        # out_csv = output_folder + 'c3podrl_alpha{}_gamma{}_decay{}.csv'.format(config.LR, config.GAMMA, epsilon)
        # # out_csv2 = output_folder + 'trajectory_episode{}.csv'.format(episode)
        # env.save_csv(out_csv, info['step_time'])
        # env.save_csv2(out_csv2, info['step_time'])
    env.close()


