import pygame
import random
from collections import namedtuple
from enum import Enum


pygame.init()

Point = namedtuple('Point', 'x, y')
BLOCKSIZE = 20


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


class Game:

    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height

        # Initialize the display
        self.display = pygame.display.set_model(self.width, self.height)
        pygame.display.set_caption('SnakeAI')
        self.clock = pygame.time.Clock()

        # Initialize the game state
        self.score = 0
        self.food = None
        self.direction = Direction.RIGHT
        self.head = Point(self.width/2, self.height/2)
        self.snake = [self.head, Point(
            self.head.x - BLOCKSIZE, self.head.y), Point(self.head.x - (2*BLOCKSIZE), self.head.y)]
        self.placeFood()

    def placeFood(self):
        x = random.randint(0, (self.width - BLOCKSIZE) //
                           BLOCKSIZE) * BLOCKSIZE
        y = random.randint(0, (self.height - BLOCKSIZE) //
                           BLOCKSIZE) * BLOCKSIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self.placeFood()

    def play(self):
        # Collect user input
        pass

        # Move the snake

        # Check if the game is over

        # Place new food or move

        # Update UI and clock

        # Return gameOver and score
        return gameOver, self.score


if __name__ == '__main__':
    game = Game()

    while True:
        gameOver, score = game.play()

        if gameOver == True:
            break

    print("Final Score: ", score)

    pygame.quit()
