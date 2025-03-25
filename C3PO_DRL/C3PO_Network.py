#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 01:25:54 2019

@author: ccyen
"""

import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import gym, random, pickle, os.path, math, glob

from torch.autograd import Variable
from timeit import default_timer as timer
from datetime import timedelta
from timeit import default_timer as timer
from IPython.display import clear_output

from C3PO_DRL.utils.wrappers import *
from C3PO_DRL.utils.hyperparameters import Config
from C3PO_DRL.utils.plot import plot_reward
from C3PO_DRL.agents.BaseAgent import BaseAgent
from C3PO_DRL.networks.layers import NoisyLinear


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DuelingDQN_with_Noisy(nn.Module):
    def __init__(self, shapeOfState, numOfActions, noisy=False, sigma_init=0.5):
        super(DuelingDQN_with_Noisy, self).__init__()
        
        self.shape_state = shapeOfState
        self.num_actions = numOfActions
        self.noisy=noisy

        self.layer1 = nn.Conv2d(self.shape_state[0], 32, kernel_size=5, stride=1)
        self.layer2 = nn.Conv2d(32, 64, kernel_size=4, stride=1)
        self.layer3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.adv1 = nn.Linear(self.feature_size(), 1024) if not self.noisy else NoisyLinear(self.feature_size(), 1024, sigma_init)
        self.adv2 = nn.Linear(1024, self.num_actions) if not self.noisy else NoisyLinear(1024, self.num_actions, sigma_init)

        self.val1 = nn.Linear(self.feature_size(), 1024) if not self.noisy else NoisyLinear(self.feature_size(), 1024, sigma_init)
        self.val2 = nn.Linear(1024, 1) if not self.noisy else NoisyLinear(1024, 1, sigma_init)
        
        print(self.feature_size())
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = x.view(x.size(0), -1)

        adv = F.relu(self.adv1(x))
        adv = self.adv2(adv)

        val = F.relu(self.val1(x))
        val = self.val2(val)

        return val + adv - adv.mean()
    
    def feature_size(self):
        return self.layer3(self.layer2(self.layer1(torch.zeros(1, *self.shape_state)))).view(1, -1).size(1)

    def sample_noise(self):
        if self.noisy:
            self.adv1.sample_noise()
            self.adv2.sample_noise()
            self.val1.sample_noise()
            self.val2.sample_noise()


class NN(nn.Module):
    def __init__(self, shapeOfState, numOfActions):
        super(NN, self).__init__()
        
        self.shape_state = shapeOfState
        self.num_actions = numOfActions

        # """Hang"""
        # self.layer0 = nn.Linear(self.shape_state[0], 64)
        # self.layer1 = nn.Linear(64,256)
        self.layer1 = nn.Linear(self.shape_state[0], 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        
        self.adv1 = nn.Linear(64, 32)
        self.adv2 = nn.Linear(32, self.num_actions)

        self.val1 = nn.Linear(64, 32)
        self.val2 = nn.Linear(32, 1)
        
    def forward(self, x):
        # x = F.relu(self.layer0(x))
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = x.view(x.size(0), -1)

        adv = F.relu(self.adv1(x))
        adv = self.adv2(adv)

        val = F.relu(self.val1(x))
        val = self.val2(val)

        return val + adv - adv.mean()
    
    def sample_noise(self):
        #ignore this for now
        pass


class DuelingDQN(nn.Module):
    def __init__(self, shapeOfState, numOfActions):
        super(DuelingDQN, self).__init__()
        
        self.shape_state = shapeOfState
        self.num_actions = numOfActions
        
        self.layer1 = nn.Conv2d(self.shape_state[0], 32, kernel_size=5, stride=1)
        self.layer2 = nn.Conv2d(32, 64, kernel_size=4, stride=1)
        self.layer3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.adv1 = nn.Linear(self.feature_size(), 1024)
        self.adv2 = nn.Linear(1024, self.num_actions)

        self.val1 = nn.Linear(self.feature_size(), 1024)
        self.val2 = nn.Linear(1024, 1)
        
        print(self.feature_size())
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = x.view(x.size(0), -1)

        adv = F.relu(self.adv1(x))
        adv = self.adv2(adv)

        val = F.relu(self.val1(x))
        val = self.val2(val)

        return val + adv - adv.mean()
    
    def get_hidden_layer(self, state):
        y = F.relu(self.layer1(state))
        y = F.relu(self.layer2(y))
        y = F.relu(self.layer3(y))
        y = y.view(y.size(0), -1)

        output_hidden = F.relu(self.adv1(y))
        
        return output_hidden
    
    def feature_size(self):
        return self.layer3(self.layer2(self.layer1(torch.zeros(1, *self.shape_state)))).view(1, -1).size(1)
    
    def sample_noise(self):
        #ignore this for now
        pass

class DuelingDQN_CA(nn.Module):
    def __init__(self, shapeOfState, numOfActions):
        super(DuelingDQN_CA, self).__init__()
        
        self.shape_state = shapeOfState
        self.num_actions = numOfActions
        
        self.layer1 = nn.Conv2d(self.shape_state[0], 32, kernel_size=5, stride=1)
        self.layer2 = nn.Conv2d(32, 64, kernel_size=4, stride=1)
        self.layer3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.adv1 = nn.Linear(self.feature_size(), 1024)
        self.adv2 = nn.Linear(1024, self.num_actions)

        self.val1 = nn.Linear(self.feature_size(), 1024)
        self.val2 = nn.Linear(1024, 1)
        
        print(self.feature_size())
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = x.view(x.size(0), -1)

        adv = F.relu(self.adv1(x))
        adv = F.sigmoid(self.adv2(adv))

        val = F.relu(self.val1(x))
        val = F.sigmoid(self.val2(val))

        return val + adv - adv.mean()
    
    def get_weights(self, state):
        y = F.relu(self.layer1(state))
        y = F.relu(self.layer2(y))
        y = F.relu(self.layer3(y))
        y = y.view(y.size(0), -1)

        output_hidden = F.relu(self.adv1(y))
        
        return output_hidden
    
    def feature_size(self):
        return self.layer3(self.layer2(self.layer1(torch.zeros(1, *self.shape_state)))).view(1, -1).size(1)
    
    def sample_noise(self):
        #ignore this for now
        pass

class CNN(nn.Module):
    def __init__(self, shapeOfState, numOfActions):
        super(CNN, self).__init__()
        
        self.shape_state = shapeOfState
        self.num_actions = numOfActions
        #TO DO Declare your layers here
        self.layer1 = nn.Conv2d(self.shape_state[0], 32, kernel_size=5, stride=1)
        #self.layer2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = nn.Conv2d(32, 64, kernel_size=4, stride=1)
        self.layer3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        #self.layer4 = nn.Linear(self.feature_size(), self.num_actions)
        self.layer4 = nn.Linear(self.feature_size(), 1024)
        self.layer5 = nn.Linear(1024, self.num_actions)
        print(self.feature_size())

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = x.view(x.size(0), -1) # transform x to one dimension
        #x = self.layer4(x)
        x = F.relu(self.layer4(x))
        x = self.layer5(x)
        
        return x
    
    def get_hidden_layer(self, state):
        y = F.relu(self.layer1(state))
        y = F.relu(self.layer2(y))
        y = F.relu(self.layer3(y))
        y = y.view(y.size(0), -1)

        output_hidden = F.relu(self.layer4(y))
        
        return output_hidden
    
    def feature_size(self):
        return self.layer3(self.layer2(self.layer1(torch.zeros(1, *self.shape_state)))).view(1, -1).size(1)

class ActorCritic(nn.Module):
    def __init__(self, shapeOfState, numOfActions):
        super(ActorCritic, self).__init__()
        
        self.shape_state = shapeOfState
        self.num_actions = numOfActions
        
        init_ = lambda m: self.layer_init(m, nn.init.orthogonal_,
                    lambda x: nn.init.constant_(x, 0),
                    nn.init.calculate_gain('relu'))

        self.layer1 = init_(nn.Conv2d(self.shape_state[0], 32, kernel_size=5, stride=1))
        self.layer2 = init_(nn.Conv2d(32, 64, kernel_size=4, stride=1))
        self.layer3 = init_(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        self.fc1 = init_(nn.Linear(self.feature_size(), 512))

        init_ = lambda m: self.layer_init(m, nn.init.orthogonal_,
          lambda x: nn.init.constant_(x, 0))

        self.critic_linear = init_(nn.Linear(512, 1))

        init_ = lambda m: self.layer_init(m, nn.init.orthogonal_,
              lambda x: nn.init.constant_(x, 0), gain=0.01)

        self.actor_linear = init_(nn.Linear(512, self.num_actions))

        self.train()

    def forward(self, x):
        x = F.relu(self.layer1(x/255.0))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))

        value = self.critic_linear(x)
        logits = self.actor_linear(x)

        return logits, value

    def feature_size(self):
        return self.layer3(self.layer2(self.layer1(torch.zeros(1, *self.shape_state)))).view(1, -1).size(1)

    def layer_init(self, module, weight_init, bias_init, gain=1):
        weight_init(module.weight.data, gain=gain)
        bias_init(module.bias.data)
        return module

""" Branching Dueling Network """
class BDN(nn.Module):
    def __init__(self, shapeOfState, numOfActions):
        super(BDN, self).__init__()
        
        self.shape_state = shapeOfState
        self.num_actions = numOfActions
        
        self.layer1 = nn.Conv2d(self.shape_state[0], 32, kernel_size=5, stride=1)
        self.layer2 = nn.Conv2d(32, 64, kernel_size=4, stride=1)
        self.layer3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        self.layer4 = nn.Linear(self.feature_size(), 1024)
        self.layer5 = nn.Linear(1024, 512)

        self.bdn1_layer1 = nn.Linear(self.layer5, 256)
        self.bdn1_layer2 = nn.Linear(self.bdn1_layer1, 128)
        self.bdn1_output1 = nn.Linear(self.bdn1_layer2, 1)
        self.bdn1_output2 = nn.Linear(self.bdn1_layer2, 1)
        
        self.bdn2_layer1 = nn.Linear(self.layer5, 256)
        self.bdn2_layer2 = nn.Linear(self.bdn2_layer1, 128)
        self.bdn2_output1 = nn.Linear(self.bdn2_layer2, 1)
        self.bdn2_output2 = nn.Linear(self.bdn2_layer2, 1)
        
        self.bdn3_layer1 = nn.Linear(self.layer5, 256)
        self.bdn3_layer2 = nn.Linear(self.bdn3_layer1, 128)
        self.bdn3_output1 = nn.Linear(self.bdn3_layer2, 1)
        self.bdn3_output2 = nn.Linear(self.bdn3_layer2, 1)
        
        self.bdn4_layer1 = nn.Linear(self.layer5, 256)
        self.bdn4_layer2 = nn.Linear(self.bdn4_layer1, 128)
        self.bdn4_output1 = nn.Linear(self.bdn4_layer2, 1)
        self.bdn4_output2 = nn.Linear(self.bdn4_layer2, 1)
        
        self.bdn5_layer1 = nn.Linear(self.layer5, 256)
        self.bdn5_layer2 = nn.Linear(self.bdn5_layer1, 128)
        self.bdn5_output1 = nn.Linear(self.bdn5_layer2, 1)
        self.bdn5_output2 = nn.Linear(self.bdn5_layer2, 1)
        
        self.bdn6_layer1 = nn.Linear(self.layer5, 256)
        self.bdn6_layer2 = nn.Linear(self.bdn6_layer1, 128)
        self.bdn6_output1 = nn.Linear(self.bdn6_layer2, 1)
        self.bdn6_output2 = nn.Linear(self.bdn6_layer2, 1)
        
        self.bdn7_layer1 = nn.Linear(self.layer5, 256)
        self.bdn7_layer2 = nn.Linear(self.bdn7_layer1, 128)
        self.bdn7_output1 = nn.Linear(self.bdn7_layer2, 1)
        self.bdn7_output2 = nn.Linear(self.bdn7_layer2, 1)
        
        self.bdn8_layer1 = nn.Linear(self.layer5, 256)
        self.bdn8_layer2 = nn.Linear(self.bdn8_layer1, 128)
        self.bdn8_output1 = nn.Linear(self.bdn8_layer2, 1)
        self.bdn8_output2 = nn.Linear(self.bdn8_layer2, 1)
        
        self.bdn9_layer1 = nn.Linear(self.layer5, 256)
        self.bdn9_layer2 = nn.Linear(self.bdn9_layer1, 128)
        self.bdn9_output1 = nn.Linear(self.bdn9_layer2, 1)
        self.bdn9_output2 = nn.Linear(self.bdn9_layer2, 1)
        

        #self.val1 = nn.Linear(self.feature_size(), 1024)
        #self.val2 = nn.Linear(1024, 512)
        #self.val3 = nn.Linear(512, 256)
        #self.val4 = nn.Linear(256, 128)
        #self.val5 = nn.Linear(128, 1)
        
        print(self.feature_size())
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = x.view(x.size(0), -1)

        bdn_1 = F.relu(self.bdn1_layer1(x))
        bdn_1 = self.bdn1_layer2(bdn_1)
        bdn_1_o1 = F.sigmoid(self.bdn1_output1(bdn_1))
        bdn_1_o2 = F.sigmoid(self.bdn1_output2(bdn_1))
        
        bdn_2 = F.relu(self.bdn2_layer1(x))
        bdn_2 = self.bdn2_layer2(bdn_2)
        bdn_2_o1 = F.sigmoid(self.bdn2_output1(bdn_2))
        bdn_2_o2 = F.sigmoid(self.bdn2_output2(bdn_2))
        
        bdn_3 = F.relu(self.bdn3_layer1(x))
        bdn_3 = self.bdn3_layer2(bdn_3)
        bdn_3_o1 = F.sigmoid(self.bdn3_output1(bdn_3))
        bdn_3_o2 = F.sigmoid(self.bdn3_output2(bdn_3))
        
        bdn_4 = F.relu(self.bdn4_layer1(x))
        bdn_4 = self.bdn4_layer2(bdn_4)
        bdn_4_o1 = F.sigmoid(self.bdn4_output1(bdn_4))
        bdn_4_o2 = F.sigmoid(self.bdn4_output2(bdn_4))
        
        bdn_5 = F.relu(self.bdn5_layer1(x))
        bdn_5 = self.bdn5_layer2(bdn_5)
        bdn_5_o1 = F.sigmoid(self.bdn5_output1(bdn_5))
        bdn_5_o2 = F.sigmoid(self.bdn5_output2(bdn_5))
        
        bdn_6 = F.relu(self.bdn6_layer1(x))
        bdn_6 = self.bdn6_layer2(bdn_6)
        bdn_6_o1 = F.sigmoid(self.bdn6_output1(bdn_6))
        bdn_6_o2 = F.sigmoid(self.bdn6_output2(bdn_6))
        
        bdn_7 = F.relu(self.bdn7_layer1(x))
        bdn_7 = self.bdn7_layer2(bdn_7)
        bdn_7_o1 = F.sigmoid(self.bdn7_output1(bdn_7))
        bdn_7_o2 = F.sigmoid(self.bdn7_output2(bdn_7))
        
        bdn_8 = F.relu(self.bdn8_layer1(x))
        bdn_8 = self.bdn8_layer2(bdn_8)
        bdn_8_o1 = F.sigmoid(self.bdn8_output1(bdn_8))
        bdn_8_o2 = F.sigmoid(self.bdn8_output2(bdn_8))
        
        bdn_9 = F.relu(self.bdn9_layer1(x))
        bdn_9 = self.bdn9_layer2(bdn_9)
        bdn_9_o1 = F.sigmoid(self.bdn9_output1(bdn_9))
        bdn_9_o2 = F.sigmoid(self.bdn9_output2(bdn_9))


        #val = F.relu(self.val1(x))
        #val = self.val2(val)
        #val = self.val3(val)
        #val = self.val4(val)
        #val = self.val5(val)

        return bdn_1_o1, bdn_1_o2, bdn_2_o1, bdn_2_o2, bdn_3_o1, bdn_3_o2, bdn_4_o1, bdn_4_o2, bdn_5_o1, bdn_5_o2, bdn_6_o1, bdn_6_o2, bdn_7_o1, bdn_7_o2, bdn_8_o1, bdn_8_o2, bdn_9_o1, bdn_9_o2 
    
    def get_hidden_layer(self, state):
        y = F.relu(self.layer1(state))
        y = F.relu(self.layer2(y))
        y = F.relu(self.layer3(y))
        y = y.view(y.size(0), -1)

        output_hidden = F.relu(self.adv1(y))
        
        return output_hidden
    
    def feature_size(self):
        return self.layer3(self.layer2(self.layer1(torch.zeros(1, *self.shape_state)))).view(1, -1).size(1)
    
    def sample_noise(self):
        #ignore this for now
        pass

class CNN_Actor(nn.Module):    
    def __init__(self, shapeOfState, numOfActions, fc1_units=400, fc2_units=300):
        super(CNN_Actor, self).__init__()
        
        self.shape_state = shapeOfState
        self.num_actions = numOfActions
        
        #TO DO Declare your layers here
        self.layer1 = nn.Conv2d(self.shape_state[0], 32, kernel_size=5, stride=1)
        #self.layer2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = nn.Conv2d(32, 64, kernel_size=4, stride=1)
        self.layer3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        #self.layer4 = nn.Linear(self.feature_size(), self.num_actions)
        self.layer4 = nn.Linear(self.feature_size(), 1024)
        self.layer5 = nn.Linear(1024, fc1_units)
        self.layer6 = nn.Linear(fc1_units, fc2_units)
        self.layer7 = nn.Linear(fc2_units, self.num_actions)
        #self.reset_parameters()
        
        print(self.feature_size())

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = x.view(x.size(0), -1) # transform x to one dimension
        #x = self.layer4(x)
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        x = F.relu(self.layer6(x))
        x = F.sigmoid(self.layer7(x))
        
        return x
    
    def get_hidden_layer(self, state):
        y = F.relu(self.layer1(state))
        y = F.relu(self.layer2(y))
        y = F.relu(self.layer3(y))
        y = y.view(y.size(0), -1)
        
        y = F.relu(self.layer4(y))
        y = F.relu(self.layer5(y))
        y = F.relu(self.layer6(y))
        output_hidden = y
        
        return output_hidden
    
    def feature_size(self):
        return self.layer3(self.layer2(self.layer1(torch.zeros(1, *self.shape_state)))).view(1, -1).size(1)
    
    def hidden_init(self, layer):
        fan_in = layer.weight.data.size()[0]
        lim = 1. / np.sqrt(fan_in)
        return (-lim, lim)
    
    def reset_parameters(self):
        self.layer4.weight.data.uniform_(*self.hidden_init(self.layer4))
        self.layer5.weight.data.uniform_(*self.hidden_init(self.layer5))
        self.layer6.weight.data.uniform_(*self.hidden_init(self.layer6))
        self.layer7.weight.data.uniform_(0, 1)

class CNN_Critic(nn.Module):
    def __init__(self, shapeOfState, numOfActions, fc1_units=400, fc2_units=300):
        super(CNN_Critic, self).__init__()
        
        self.shape_state = shapeOfState
        self.num_actions = numOfActions
        
        #TO DO Declare your layers here
        self.layer1 = nn.Conv2d(self.shape_state[0], 32, kernel_size=5, stride=1)
        #self.layer2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = nn.Conv2d(32, 64, kernel_size=4, stride=1)
        self.layer3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        self.fc1 = nn.Linear(self.feature_size(), 1024)
        self.fc2 = nn.Linear(1024, fc1_units)
        self.fc3 = nn.Linear(fc1_units+numOfActions, fc2_units)
        self.fc4 = nn.Linear(fc2_units, 1)
        #self.reset_parameters()
        
    def forward(self, x, a):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = x.view(x.size(0), -1) # transform x to one dimension
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.cat((x, a), dim=1)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        
        return x
    
    def feature_size(self):
        return self.layer3(self.layer2(self.layer1(torch.zeros(1, *self.shape_state)))).view(1, -1).size(1)
    
    def hidden_init(self, layer):
        fan_in = layer.weight.data.size()[0]
        lim = 1. / np.sqrt(fan_in)
        return (-lim, lim)
    
    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*self.hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*self.hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*self.hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(0, 1)

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        
        self.input_shape = input_shape
        self.num_actions = num_actions

        self.conv1 = nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(self.feature_size(), 512)
        self.fc2 = nn.Linear(512, self.num_actions)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
    def feature_size(self):
        return self.conv3(self.conv2(self.conv1(torch.zeros(1, *self.input_shape)))).view(1, -1).size(1)
