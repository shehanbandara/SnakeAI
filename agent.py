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
        # Related to the epsilon-greedy action selection procedure in the Q-learning algorithm
        # Introduces randomness into the algorithm
        self.epsilon = 0

        # Discount rate
        self.gamma = 0

        # Number of gane iterations
        self.numGames = 0

        # If memory is exceeded, popleft()
        self.memory = deque(maxlen=MAXMEMORY)

    def getAction(self, state):
        pass

    def getState(self, game):
        pass

    def remember(self, state, action, reward, nextState, gameOver):
        pass

    def trainShortTermMemory(self, state, action, reward, nextState, gameOver):
        pass

    def trainLongTermMemory(self):
        pass


def train():
    # Initialize varibles for the high score and total score
    highScore = 0
    totalScore = 0

    #
    plotScores = []
    plotMeanScores = []

    #
    agent = Agent()
    game = Game()

    #
    while True:

        # Get the old state
        oldState = agent.getState(game)

        # Get the action
        action = agent.getAction(oldState)

        # Preform the action and get the new state
        reward, gameOver, score = game.play(action)
        newState = agent.getState(game)

        # Train the short term memory
        agent.trainShortTermMemory(
            oldState, action, reward, newState, gameOver)

        #
        agent.remember(oldState, action, reward, newState, gameOver)

        # If the game is over
        if gameOver:

            # Train the long term memory and plot the results
            game.reset()
            agent.numGames += 1
            agent.trainLongTermMemory()

            # If new high score
            if score > highScore:
                highScore = score

            # Print the game iteration, game score, and high score
            print('Game: ', agent.numGames, ', Score:',
                  score, ', High Score:', highScore)


if __name__ == '__main__':
    train()
