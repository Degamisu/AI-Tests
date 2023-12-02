import pygame
import numpy as np
from collections import deque
import random
import tensorflow as tf
from tensorflow.keras import layers, models

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 400, 400
GRID_SIZE = 20
FPS = 10

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Define directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

# Create the game window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Game with DRL by Degamisu (2023)")
clock = pygame.time.Clock()

# Define the Snake class
class Snake:
    def __init__(self):
        self.body = deque([(WIDTH // 2, HEIGHT // 2)])
        self.direction = RIGHT

    def move(self, direction):
        new_head = (
            (self.body[0][0] + direction[0]) % WIDTH,
            (self.body[0][1] + direction[1]) % HEIGHT
        )
        self.body.appendleft(new_head)
        self.body.pop()

    def grow(self):
        tail_direction = (
            self.body[-1][0] - self.body[-2][0],
            self.body[-1][1] - self.body[-2][1]
        )
        new_tail = (
            (self.body[-1][0] + tail_direction[0]) % WIDTH,
            (self.body[-1][1] + tail_direction[1]) % HEIGHT
        )
        self.body.append(new_tail)

    def check_collision(self):
        return len(self.body) != len(set(self.body))

# Define the Food class
class Food:
    def __init__(self):
        self.position = (random.randint(0, WIDTH // GRID_SIZE - 1) * GRID_SIZE,
                         random.randint(0, HEIGHT // GRID_SIZE - 1) * GRID_SIZE)

    def respawn(self):
        self.position = (random.randint(0, WIDTH // GRID_SIZE - 1) * GRID_SIZE,
                         random.randint(0, HEIGHT // GRID_SIZE - 1) * GRID_SIZE)

# Define the DQN model
def create_model():
    model = models.Sequential()
    model.add(layers.Dense(128, input_shape=(11,), activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(4, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    return model

# Initialize DQN model and target model
model = create_model()
target_model = create_model()

# Initialize replay buffer
replay_buffer = deque(maxlen=10000)

# Define the DQN Agent
class DQNAgent:
    def __init__(self):
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.95
        self.batch_size = 32
        self.update_rate = 1000  # Update target model every 1000 steps
        self.step = 0

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randint(0, 3)
        q_values = model.predict(state)
        return np.argmax(q_values[0])

    def train(self, state, action, reward, next_state, done):
        replay_buffer.append((state, action, reward, next_state, done))
        self.step += 1

        if len(replay_buffer) >= self.batch_size and self.step % 5 == 0:
            self.replay()

        if self.step % self.update_rate == 0:
            self.update_target_model()

    def replay(self):
        minibatch = random.sample(replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = np.concatenate(states)
        next_states = np.concatenate(next_states)

        targets = model.predict(states)
        next_q_values = target_model.predict(next_states)

        for i in range(self.batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])

        model.fit(states, targets, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        target_model.set_weights(model.get_weights())

# Initialize Snake and Food
snake = Snake()
food = Food()

# Initialize DQN agent
agent = DQNAgent()

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Get current state
    state = np.array([snake.body[0][0], snake.body[0][1],
                      food.position[0], food.position[1],
                      snake.direction[0], snake.direction[1],
                      WIDTH, HEIGHT,
                      snake.body[-1][0], snake.body[-1][1],
                      food.position[0] - snake.body[0][0], food.position[1] - snake.body[0][1]])

    # Choose an action using DQN
    action = agent.act(np.reshape(state, [1, 11]))

    # Perform action
    if action == 0 and snake.direction != RIGHT:
        snake.direction = LEFT
    elif action == 1 and snake.direction != LEFT:
        snake.direction = RIGHT
    elif action == 2 and snake.direction != DOWN:
        snake.direction = UP
    elif action == 3 and snake.direction != UP:
        snake.direction = DOWN

    # Move the snake
    snake.move(snake.direction)

    # Check for collision with the food
    if snake.body[0] == food.position:
        snake.grow()
        food.respawn()

    # Check for collision with the walls or itself
    if snake.check_collision():
        running = False

    # Get the next state
    next_state = np.array([snake.body[0][0], snake.body[0][1],
                           food.position[0], food.position[1],
                           snake.direction[0], snake.direction[1],
                           WIDTH, HEIGHT,
                           snake.body[-1][0], snake.body[-1][1],
                           food.position[0] - snake.body[0][0], food.position[1] - snake.body[0][1]])

    # Get the reward
    reward = 1 if snake.body[0] == food.position else 0

    # Check if the game is done
    done = not running

    # Train the DQN agent
    agent.train(np.reshape(state, [1, 11]), action, reward, np.reshape(next_state, [1, 11]), done)

    # Draw everything
    screen.fill(WHITE)
    for segment in snake.body:
        pygame.draw.rect(screen, GREEN, (*segment, GRID_SIZE, GRID_SIZE))

    pygame.draw.rect(screen, RED, (*food.position, GRID_SIZE, GRID_SIZE))

    pygame.display.flip()
    clock.tick(FPS)

# Quit Pygame
pygame.quit()
