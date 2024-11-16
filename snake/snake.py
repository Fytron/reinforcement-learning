import pygame
import random
import numpy as np
from enum import Enum
import os

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

class SnakeGame:
    def __init__(self, width=640, height=480):
        pygame.init()
        self.width = width
        self.height = height
        self.block_size = 20
        self.display = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Snake AI')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.record = 0
        self.reset()
        
    def reset(self):
        self.direction = Direction.RIGHT
        self.head = [self.width/2, self.height/2]
        self.snake = [self.head.copy(),
                     [self.head[0]-self.block_size, self.head[1]],
                     [self.head[0]-(2*self.block_size), self.head[1]]]
        self.score = 0
        self.food = self._place_food()
        self.frame_iteration = 0
        return self._get_state()

    def _place_food(self):
        x = round(random.randint(0, (self.width-self.block_size)//self.block_size)*self.block_size)
        y = round(random.randint(0, (self.height-self.block_size)//self.block_size)*self.block_size)
        return [x, y]

    def play_step(self, action):
        self.frame_iteration += 1
        
        # 1. Collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. Move
        self._move(action)
        self.snake.insert(0, self.head.copy())
        
        # 3. Check if game over
        reward = 0
        game_over = False
        if self._is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. Place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self.food = self._place_food()
            # Update record if current score is higher
            if self.score > self.record:
                self.record = self.score
        else:
            self.snake.pop()
        
        # 5. Update UI and clock
        self._update_ui()
        self.clock.tick(20)
        
        # 6. Return game over and score
        return reward, game_over, self.score

    def _update_ui(self):
        self.display.fill((0,0,0))
        
        for pt in self.snake:
            pygame.draw.rect(self.display, (0,255,0), 
                           pygame.Rect(pt[0], pt[1], self.block_size, self.block_size))
        
        pygame.draw.rect(self.display, (255,0,0), 
                        pygame.Rect(self.food[0], self.food[1], self.block_size, self.block_size))
        
        score_text = self.font.render(f'Score: {self.score}', True, (255, 255, 255))
        record_text = self.font.render(f'Record: {self.record}', True, (255, 255, 255))
        self.display.blit(score_text, (10, 10))
        self.display.blit(record_text, (10, 50))
        
        pygame.display.flip()

    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]
            
        self.direction = new_dir
        
        x = self.head[0]
        y = self.head[1]
        if self.direction == Direction.RIGHT:
            x += self.block_size
        elif self.direction == Direction.LEFT:
            x -= self.block_size
        elif self.direction == Direction.DOWN:
            y += self.block_size
        elif self.direction == Direction.UP:
            y -= self.block_size
            
        self.head = [x, y]

    def _is_collision(self):
        if (self.head[0] > self.width - self.block_size or self.head[0] < 0 or
            self.head[1] > self.height - self.block_size or self.head[1] < 0):
            return True
        if self.head in self.snake[1:]:
            return True
        return False

    def _get_state(self):
        head = self.head
        point_l = [head[0] - self.block_size, head[1]]
        point_r = [head[0] + self.block_size, head[1]]
        point_u = [head[0], head[1] - self.block_size]
        point_d = [head[0], head[1] + self.block_size]
        
        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        state = [
            (dir_r and self._is_collision_point(point_r)) or 
            (dir_l and self._is_collision_point(point_l)) or 
            (dir_u and self._is_collision_point(point_u)) or 
            (dir_d and self._is_collision_point(point_d)),

            (dir_u and self._is_collision_point(point_r)) or 
            (dir_d and self._is_collision_point(point_l)) or 
            (dir_l and self._is_collision_point(point_u)) or 
            (dir_r and self._is_collision_point(point_d)),

            (dir_d and self._is_collision_point(point_r)) or 
            (dir_u and self._is_collision_point(point_l)) or 
            (dir_r and self._is_collision_point(point_u)) or 
            (dir_l and self._is_collision_point(point_d)),
            
            dir_l, dir_r, dir_u, dir_d,
            
            self.food[0] < head[0],
            self.food[0] > head[0],
            self.food[1] < head[1],
            self.food[1] > head[1]
        ]
        return np.array(state, dtype=int)

    def _is_collision_point(self, point):
        if (point[0] > self.width - self.block_size or point[0] < 0 or
            point[1] > self.height - self.block_size or point[1] < 0):
            return True
        if point in self.snake[1:]:
            return True
        return False