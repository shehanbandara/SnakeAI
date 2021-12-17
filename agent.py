import random
import torch
import numpy as np
from collections import deque
from game import Direction, Game, Point

MAXMEMORY = 100000
BATCHSIZE = 1000
LEARNINGRATE = 0.001


def train():
    pass


if __name__ == '__main__':
    train()


class Agent:

    def __init__(self):
        pass

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
