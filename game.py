import pygame
import random
from collections import namedtuple
from enum import Enum

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
        for i in pygame.event.get():
            if i.type == pygame.QUIT:
                pygame.quit()
                quit()
            if i.type == pygame.KEYDOWN:
                if i.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif i.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                elif i.key == pygame.K_UP:
                    self.direction = Direction.UP
                elif i.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN

        # Move the snake
        self.move(self.direction)
        self.snake.insert(0, self.head)

        # Check if the game is over
        gameOver = False
        if self.collision():
            gameOver = True
            return gameOver, self.score

        # Place new food or move
        if self.head == self.food:
            self.score += 1
            self.placeFood()
        else:
            self.snake.pop()

        # Update UI and clock
        self.updateUI()
        self.clock.tick(SPEED)

        # Return gameOver and score
        return gameOver, self.score

    def move(self, direction):
        x = self.head.x
        y = self.head.y

        if direction == Direction.RIGHT:
            x += BLOCKSIZE
        elif direction == Direction.LEFT:
            x -= BLOCKSIZE
        elif direction == Direction.DOWN:
            y += BLOCKSIZE
        elif direction == Direction.UP:
            y -= BLOCKSIZE

        self.head = Point(x, y)

    def collision(self):
        # Does the snake hit the boundary
        if self.head.x > self.width - BLOCKSIZE or self.head.x < 0 or self.head.y > self.height - BLOCKSIZE or self.head.y < 0:
            return True

        # Does the snake hit itself
        if self.head in self.snake[1:]:
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


if __name__ == '__main__':
    game = Game()

    while True:
        gameOver, score = game.play()

        if gameOver == True:
            break

    print("Final Score: ", score)

    pygame.quit()
