import pygame
import random
from collections import namedtuple
from enum import Enum
import numpy as np

pygame.init()

Point = namedtuple('Point', 'x, y')
Font = pygame.font.SysFont('arial', 25)
BLOCKSIZE = 20
SPEED = 15

# RGB Colours
BLACK = (0, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
RED = (200, 0, 0)
WHITE = (255, 255, 255)


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
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('SnakeAI')
        self.clock = pygame.time.Clock()

        self.reset()

    def reset(self):
        # Reset the game state
        self.score = 0
        self.food = None
        self.direction = Direction.RIGHT
        self.head = Point(self.width/2, self.height/2)
        self.snake = [self.head, Point(
            self.head.x - BLOCKSIZE, self.head.y), Point(self.head.x - (2*BLOCKSIZE), self.head.y)]
        self.frameIteration = 0
        self.placeFood()

    def placeFood(self):
        x = random.randint(0, (self.width - BLOCKSIZE) //
                           BLOCKSIZE) * BLOCKSIZE
        y = random.randint(0, (self.height - BLOCKSIZE) //
                           BLOCKSIZE) * BLOCKSIZE
        self.food = Point(x, y)

        if self.food in self.snake:
            self.placeFood()

    def play(self, action):
        self.frameIteration += 1

        # Collect user input
        for i in pygame.event.get():
            if i.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Move the snake
        self.move(action)
        self.snake.insert(0, self.head)

        # Check if the game is over
        reward = 0
        gameOver = False
        if self.collision() or self.frameIteration > 100*len(self.snake):
            gameOver = True
            reward = -10
            return reward, gameOver, self.score

        # Place new food or move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self.placeFood()
        else:
            self.snake.pop()

        # Update UI and clock
        self.updateUI()
        self.clock.tick(SPEED)

        # Return gameOver and score
        return reward, gameOver, self.score

    def move(self, action):
        # [straight, right, left]
        clockwise = [Direction.RIGHT, Direction.DOWN,
                     Direction.LEFT, Direction.UP]
        currentIndex = clockwise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            # No change in direction
            newDirection = clockwise[currentIndex]

        elif np.array_equal(action, [0, 1, 0]):
            nextIndex = (currentIndex + 1) % 4
            # Turn right
            newDirection = clockwise[nextIndex]

        else:
            nextIndex = (currentIndex - 1) % 4
            # Turn left
            newDirection = clockwise[nextIndex]

        self.direction = newDirection

        x = self.head.x
        y = self.head.y

        if self.direction == Direction.RIGHT:
            x += BLOCKSIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCKSIZE
        elif self.direction == Direction.DOWN:
            y += BLOCKSIZE
        elif self.direction == Direction.UP:
            y -= BLOCKSIZE

        self.head = Point(x, y)

    def collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Does the snake hit the boundary
        if pt.x > self.width - BLOCKSIZE or pt.x < 0 or pt.y > self.height - BLOCKSIZE or pt.y < 0:
            return True

        # Does the snake hit itself
        if pt in self.snake[1:]:
            return True

        return False

    def updateUI(self):
        self.display.fill(BLACK)

        for i in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(
                i.x, i.y, BLOCKSIZE, BLOCKSIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(
                i.x+4, i.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(
            self.food.x, self.food.y, BLOCKSIZE, BLOCKSIZE))

        text = Font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
