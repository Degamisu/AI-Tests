import pygame
import random

# Initialize Pygame
pygame.init()

# Debugging variables
debug_font = pygame.font.SysFont(None, 24)
debug_color = (255, 255, 255)

# Constants
WIDTH, HEIGHT = 400, 400
GRID_SIZE = 20
FPS = 15  # Increase the speed

# Colors
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Create the game window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("DRL Snake Game")
clock = pygame.time.Clock()

# Define directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

# Define the Snake class
class Snake:
    def __init__(self):
        self.body = [(WIDTH // 2, HEIGHT // 2)]
        self.direction = RIGHT

    def move(self):
        # Move in the current direction
        new_head = (
            (self.body[0][0] + self.direction[0] * GRID_SIZE) % WIDTH,
            (self.body[0][1] + self.direction[1] * GRID_SIZE) % HEIGHT
        )
        self.body.insert(0, new_head)
        self.body.pop()

    def rotate_left(self):
        # Rotate the snake to the left
        self.direction = (self.direction[1], -self.direction[0])

    def rotate_right(self):
        # Rotate the snake to the right
        self.direction = (-self.direction[1], self.direction[0])

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

# Define the Q-learning agent
class QLearningAgent:
    def __init__(self):
        self.q_table = {}
        self.prev_state = None
        self.prev_action = None

    def get_state_key(self, snake_head, food_position):
        return (snake_head, food_position)

    def get_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0, 0, 0, 0]  # Q-values for UP, DOWN, LEFT, RIGHT

        # Choose the action with the highest Q-value (exploration vs. exploitation)
        if random.uniform(0, 1) < 0.1:  # Exploration (random action)
            return random.choice([UP, DOWN, LEFT, RIGHT])
        else:  # Exploitation (choose the best action)
            return max((value, action) for action, value in enumerate(self.q_table[state]))[1]

    def update_q_value(self, reward, next_state):
        # Q-learning update rule
        if self.prev_state is not None and self.prev_action is not None:
            current_q_value = self.q_table[self.prev_state][self.prev_action]
            best_next_action = max((value, a) for a, value in enumerate(self.q_table[next_state]))[1]
            next_q_value = self.q_table[next_state][best_next_action]
            updated_q_value = current_q_value + 0.1 * (reward + 0.9 * next_q_value - current_q_value)

            # Update Q-value in the Q-table
            self.q_table[self.prev_state][self.prev_action] = updated_q_value

    def reset_memory(self):
        self.prev_state = None
        self.prev_action = None

# Initialize Snake, Food, and Q-learning agent
snake = Snake()
food = Food()
q_agent = QLearningAgent()

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Get current state
    state = q_agent.get_state_key(snake.body[0], food.position)

    # Choose action using Q-learning
    action = q_agent.get_action(state)

    # Perform action
    if action == UP:
        snake.rotate_left()
    elif action == DOWN:
        snake.rotate_right()

    snake.move()

    # Check for collision with the food
    if snake.body[0] == food.position:
        snake.grow()
        food.respawn()

    # Check for collision with the walls or itself
    if snake.check_collision():
        # Respawn snake and food
        snake = Snake()
        food.respawn()

    # Get the next state
    next_state = q_agent.get_state_key(snake.body[0], food.position)

    # Get the reward (you may need to adjust the reward function based on your goals)
    reward = 1 if snake.body[0] == food.position else 0

    # Update Q-value based on the observed reward
    q_agent.update_q_value(reward, next_state)

    # Draw everything
    screen.fill(BLACK)
    for segment in snake.body:
        pygame.draw.rect(screen, GREEN, (*segment, GRID_SIZE, GRID_SIZE))

    pygame.draw.rect(screen, RED, (*food.position, GRID_SIZE, GRID_SIZE))

    # Draw debugging information
    fps_text = debug_font.render(f"FPS: {int(clock.get_fps())}", True, debug_color)
    direction_text = debug_font.render(f"Direction: {action}", True, debug_color)
    brain_text = debug_font.render(f"Snake Brain: {q_agent.q_table[state]}", True, debug_color)

    # Draw debugging information
    screen.blit(fps_text, (10, 10))
    screen.blit(direction_text, (10, 30))
    screen.blit(brain_text, (10, 50))

    pygame.display.flip()
    clock.tick(FPS)

    # Reset Q-learning agent's memory
    q_agent.reset_memory()

# Quit Pygame
pygame.quit()
