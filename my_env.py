import sys

import gym
import copy
import numpy as np
from gym import spaces
from PIL import Image
import matplotlib.pyplot as plt

## This is the environment for the game of Tic Tac Toe

## The board is represented as a 3x3 matrix with the O's and X's

## What I need for the environment ?
# 1. A way to represent the board
# 2. A way to check if the game is over
# 3. A way to check if the game is a draw
# 4. Check whose turn it is
# 5. reset the board
# 6. How to give rewards
# 7. How to take actions

class TicTacToeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, size=3, win=3):
        self.size = size
        self.win = win
        self.board = np.zeros((self.size, self.size))
        self.action_space = spaces.Discrete(self.size * self.size)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.size, self.size), dtype=np.int)
        self.state = None
        self.done = False
        self.reward = None
        # not flipped = 0 , player = 1, opponent = 2
        self.player = 1
        self.opponent = 2
        self.winner = None
        self.action = None
        self.observation = None
        self.player_reward = 1
        self.opponent_reward = -1

    # Transition function
    def step(self, action):
        self.action = action
        self.board = self.state
        # player's move
        self.board[action // self.size, action % self.size] = self.player
        self.observation = copy.deepcopy(self.board)
        # check if player won
        self.reward, self.done, self.winner = self.check_win(self.board, self.player) 
        if self.done:
            return self.observation, self.reward, self.done, self.winner
        # opponent's move
        self.board = self.opponent_move(self.board, self.opponent)
        self.observation = copy.deepcopy(self.board)
        # check if opponent won
        self.reward, self.done, self.winner = self.check_win(self.board, self.opponent)
        return self.observation, self.reward, self.done, self.winner

    def reset(self):
        self.board = np.zeros((self.size, self.size))
        self.state = copy.deepcopy(self.board)
        self.done = False
        self.reward = None
        self.player = 1
        self.opponent = 2
        self.winner = None
        self.action = None
        self.observation = None
        return self.state

    def render(self, mode='human'):
        plt.imshow(self.board)
        plt.show()

    def check_win(self, board, player):
        # return 1 if player wins, -1 if opponent wins, 0 if draw, None if game not over
        if player == self.player:
            reward = self.player_reward
        elif player == self.opponent:
            reward = self.opponent_reward

        # check rows
        for i in range(self.size):
            if np.sum(board[i, :] == player) == self.size:
                return reward, True, player
        # check columns
        for i in range(self.size):
            if np.sum(board[:, i] == player) == self.size:
                return reward, True, player
        # check diagonals
        if np.sum(np.diag(board) == player) == self.size:
            return reward, True, player
        if np.sum(np.diag(np.fliplr(board)) == player) == self.size:
            return reward, True, player
        # check if draw
        if np.all(board != 0):
            return 0, True, None
        return None, False, None
    
    # opponent's turn - Random selection of opponent
    def opponent_move(self, board, player):
        # check rows
        for i in range(self.size):
            if np.sum(board[i, :] == player) == self.size - 1:
                for j in range(self.size):
                    if board[i, j] == 0:
                        board[i, j] = player
                        return board
        # check columns
        for i in range(self.size):
            if np.sum(board[:, i] == player) == self.size - 1:
                for j in range(self.size):
                    if board[j, i] == 0:
                        board[j, i] = player
                        return board
        # check diagonals
        if np.sum(np.diag(board) == player) == self.size - 1:
            for i in range(self.size):
                if board[i, i] == 0:
                    board[i, i] = player
                    return board
        if np.sum(np.diag(np.fliplr(board)) == player) == self.size - 1:
            for i in range(self.size):
                if board[i, self.size - i - 1] == 0:
                    board[i, self.size - i - 1] = player
                    return board
        # check if draw
        if np.all(board != 0):
            return board
        # random move
        while True:
            i = np.random.randint(self.size)
            j = np.random.randint(self.size)
            if board[i, j] == 0:
                board[i, j] = self.opponent
                return board
    
    def close(self):
        pass

