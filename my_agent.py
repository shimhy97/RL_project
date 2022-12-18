# This is agent code by using Deep Reinforcement Learning
# Path: my_agent.py
# Compare this snippet from my_env.py:
# Algorithm: Actor-Critic
#

import gym
import os
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from my_env import TicTacToeEnv


class TicTacToeActor(nn.Module):
    def __init__(self, state_size, action_size):
        super(TicTacToeActor, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

class TicTacToeCritic(nn.Module):
    def __init__(self, state_size):
        super(TicTacToeCritic, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class TicTacToeAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.actor = TicTacToeActor(state_size, action_size)
        self.critic = TicTacToeCritic(state_size)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)
        self.writer = SummaryWriter()

    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0) # convert state vector to tensor
        policy = self.actor(state) # get policy from actor model
        m = Categorical(probs = policy) # get distribution from policy
        action = m.sample() # get action from distribution
        log_prob = m.log_prob(action) # get log probability of action
        return action[0], log_prob

    def train_model(self, state, reward, next_state, done):
        state = torch.from_numpy(state).float().unsqueeze(0)
        next_state = torch.from_numpy(next_state).float().unsqueeze(0)
        reward = torch.FloatTensor([reward])
        if done:
            target = reward 
        else:
            target = reward + 0.9 * self.critic(next_state)
        value = self.critic(state)
        advantage = target - value
        actor_loss = -self.actor_optimizer(state) * advantage
        critic_loss = advantage.pow(2)
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()