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
        # Store the head of the snake
        head = game.snake[0]

        # Create points in all 4 directions around the head to be used to check for potential dangers
        pointR = Point(head.x + 20, head.y)
        pointL = Point(head.x - 20, head.y)
        pointU = Point(head.x, head.y - 20)
        pointD = Point(head.x, head.y + 20)

        # Create boolean variables which express the current direction
        directionR = game.direction == Direction.RIGHT
        directionL = game.direction == Direction.LEFT
        directionU = game.direction == Direction.UP
        directionD = game.direction == Direction.DOWN

        # Create the state list
        state = [
            # Danger is straight
            (directionR and game.is_collision(pointR)) or
            (directionL and game.is_collision(pointL)) or
            (directionU and game.is_collision(pointU)) or
            (directionD and game.is_collision(pointD)),

            # Danger is right
            (directionU and game.is_collision(pointR)) or
            (directionD and game.is_collision(pointL)) or
            (directionL and game.is_collision(pointU)) or
            (directionR and game.is_collision(pointD)),

            # Danger is left
            (directionD and game.is_collision(pointR)) or
            (directionU and game.is_collision(pointL)) or
            (directionR and game.is_collision(pointU)) or
            (directionL and game.is_collision(pointD)),

            # Move direction
            directionL,
            directionR,
            directionU,
            directionD,

            # Food location
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        ]

        # Return list as a numpy array
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, nextState, gameOver):
        # If memory is exceeded, popleft()
        self.memory.append(self, state, action, reward, nextState, gameOver)

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
