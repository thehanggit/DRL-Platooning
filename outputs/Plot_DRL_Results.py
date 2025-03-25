#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 07:41:22 2019

@author: ccyen
"""

import itertools
import numpy as np
import scipy.io as sio
import os.path as op
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve


R = np.array([])
Avg_delay = np.array([])
Avg_fuel = np.array([])
if op.isfile('RL_platoon/RL_platoon_result.mat'):
    loadData = sio.loadmat('RL_platoon/RL_platoon_result.mat')
    R = loadData['R_history']
    R.shape = (-1, 1)
    Avg_delay = loadData['Delay_history']
    Avg_delay.shape = (-1, 1)
    Avg_fuel = loadData['Fuel_history']
    Avg_fuel.shape = (-1, 1)


episodes = np.size(R)

t = np.arange(1, episodes + 1, 1)

##### Avg End-to-end Delay #####
fig, ax1 = plt.subplots()
color = 'tab:red'
ax1.set_xlabel('Episodes')
ax1.set_ylabel('Reward', color=color)
ax1.plot(t, R, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Average Delay', color=color)  # we already handled the x-label with ax1
ax2.plot(t, Avg_delay, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


##### Avg Fuel Consumptionn #####
fig_2, ax3 = plt.subplots()
color = 'tab:red'
ax3.set_xlabel('Episodes')
ax3.set_ylabel('Reward', color=color)
ax3.plot(t, R, color=color)
ax3.tick_params(axis='y', labelcolor=color)

ax4 = ax3.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:green'
ax4.set_ylabel('Average Fuel Consumption', color=color)  # we already handled the x-label with ax1
ax4.plot(t, Avg_fuel, color=color)
ax4.tick_params(axis='y', labelcolor=color)

fig_2.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()



t2 = np.arange(1, int(episodes / 10) + 1, 1)

R_list = []
Avg_delay_list = []
Avg_fuel_list = []
for epi_idx in range(1, int(episodes / 10) + 1):
    R_list.append(np.mean(R[epi_idx*10 - 9:epi_idx*10]))
    Avg_delay_list.append(np.mean(Avg_delay[epi_idx*10 - 9:epi_idx*10]))
    Avg_fuel_list.append(np.mean(Avg_fuel[epi_idx*10 - 9:epi_idx*10]))


##### Avg Delay #####
fig_5, ax9 = plt.subplots()
color = 'tab:red'
ax9.set_xlabel('Tens of episodes')
ax9.set_ylabel('Reward', color=color)
ax9.plot(t2, R_list, color=color)
ax9.tick_params(axis='y', labelcolor=color)

ax10 = ax9.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax10.set_ylabel('Average Delay', color=color)  # we already handled the x-label with ax9
ax10.plot(t2, Avg_delay_list, color=color)
ax10.tick_params(axis='y', labelcolor=color)

fig_5.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


##### Average Quene Length ####
fig_8, ax15 = plt.subplots()

color = 'tab:red'
ax15.set_xlabel('Tens of episodes')
ax15.set_ylabel('Reward', color=color)
ax15.plot(t2, R_list, color=color)
ax15.tick_params(axis='y', labelcolor=color)

ax16 = ax15.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:green'
ax16.set_ylabel('Average Fuel Consumption', color=color)  # we already handled the x-label with ax15
ax16.plot(t2, Avg_fuel_list, color=color)
ax16.tick_params(axis='y', labelcolor=color)

fig_8.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()
