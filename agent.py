import random
import torch
import numpy as np
from collections import deque
from game import Direction, Game, Point

MAXMEMORY = 100000
BATCHSIZE = 1000
LEARNINGRATE = 0.001


class Agent:

    def __init__(self):
        self.epsilon = 0
        # Discount rate
        self.gamma = 0
        self.numGames = 0
        self.memory = deque(maxlen=MAXMEMORY)

    def getState(self, game):
        pass

    def remember(self, state, action, reward, nextState, gameOver):
        pass

    def trainLongMemory(self):
        pass

    def trainShortMemory(self):
        pass

    def getAction(self, state):
        pass


def train():
    bestScore = 0
    totalScore = 0
    plotScores = []
    plotMeanScores = []
    agent = Agent()
    game = Game()
    while True:
        pass


if __name__ == '__main__':
    train()
