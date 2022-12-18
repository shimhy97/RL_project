# This code is main code to run the game
# Path: my_main.py
# Compare this snippet from my_env.py and my_agent.py:
#

import gym
import numpy as np
import matplotlib.pyplot as plt
from my_env import TicTacToeEnv
from my_agent import TicTacToeAgent
import os 
import time 

env = TicTacToeEnv()
agent = TicTacToeAgent(env.observation_space.shape[0], action_size = 1)
max_episode = 1000

def plot(self, scores):
    plt.plot(scores)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.show()
    # save the plot
    # graph folder is created automatically
    # add date and time to the file name
    if not os.path.exists('graph'):
        os.makedirs('graph')
    plt.savefig('./graph/Episode-Score{}.png'.format(time.strftime("%Y%m%d-%H%M%S")))

def main():
    scores = []
    for e in range(max_episode):
        state = env.reset()
        done = False
        score = 0
        while not done:
            action, log_prob = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.train_model(state, action, reward, next_state, done)
            state = next_state
            score += reward
        scores.append(score)
        agent.writer.add_scalar('Score', score, e)
        if e % 10 == 0:
            print('Episode: {}, Score: {}'.format(e, score))

if __name__ == "__main__":
    main()
