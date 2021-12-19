import random
import torch
import numpy as np
from collections import deque
from game import Direction, Game, Point
from model import LinearQNet, QTrainer
from plot import plot

MAXMEMORY = 100000
BATCHSIZE = 1000
LEARNINGRATE = 0.001


class Agent:

    def __init__(self):
        # Related to the epsilon-greedy action selection procedure in the Q-learning algorithm
        # Introduces randomness into the algorithm
        self.epsilon = 0

        # Discount rate
        self.gamma = 0.9

        # Number of gane iterations
        self.numGames = 0

        # If memory is exceeded, popleft()
        self.memory = deque(maxlen=MAXMEMORY)

        # Create an instance of the LinearQNet model
        self.model = LinearQNet(11, 256, 3)

        # Create an instance of the QTrainer trainer
        self.trainer = QTrainer(
            self.model, learningRate=LEARNINGRATE, gamma=self.gamma)

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

    def getAction(self, state):
        # The more games the agent plays, the less random actions the agent makes
        self.epsilon = 80 - self.numGames

        # Initialize the action
        action = [0, 0, 0]

        # If a random number between 0 and 200 is less than epsilon
        if random.randint(0, 200) < self.epsilon:
            # Get a random index
            index = random.randint(0, 2)

            # Overwrite action with a random action
            action[index] = 1

        # If a random number between 0 and 200 is greater than epsilon
        else:
            # Convert state to a tensor
            stateTensor = torch.tensor(state, dtype=torch.float)

            # Predict which action to take
            prediction = self.model(stateTensor)

            # Get index with the best predicted action
            index = torch.argmax(prediction).item()

            # Overwrite action with a predicted action
            action[index] = 1

        # Return the action
        return action

    def remember(self, state, action, reward, nextState, gameOver):
        # If memory is exceeded, popleft()
        self.memory.append((state, action, reward, nextState, gameOver))

    def trainShortTermMemory(self, state, action, reward, nextState, gameOver):
        # Train for one game iteration
        self.trainer.trainStep(state, action, reward, nextState, gameOver)

    def trainLongTermMemory(self):
        # If there are more 1000 samples in memory
        if len(self.memory) > BATCHSIZE:
            # Get a random sample from memory
            sample = random.sample(self.memory, BATCHSIZE)

        # If there are less than 1000 samples in memory
        else:
            # Get all samples in memory
            sample = self.memory

        # Train for one game iteration
        states, actions, rewards, nextStates, gameOvers = zip(*sample)
        self.trainer.trainStep(states, actions, rewards, nextStates, gameOvers)


def train():
    # Initialize varibles for the high score and total score
    highScore = 0
    totalScore = 0

    # Initialize lists to be used for plotting game statistics
    plotScores = []
    plotMeanScores = []

    # Initialize the agent and game
    agent = Agent()
    game = Game()

    # Game loop
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

        # Store game iteration information in memory
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

            # Plot the game score and mean game score
            plotScores.append(score)
            totalScore += score
            plotMeanScores = totalScore / agent.numGames
            plotMeanScores.append(plotMeanScores)
            plot(plotScores, plotMeanScores)


if __name__ == '__main__':
    train()
