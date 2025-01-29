import pygame
import random
import numpy as np
from gymnasium import Env, spaces
from PIL import Image

class DinoGame:
    def __init__(self, width=600, height=150):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.Surface((width, height))
        self.clock = pygame.time.Clock()
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        
        # Dino properties
        self.dino_width = 40
        self.dino_height = 40
        self.dino_x = 50
        self.dino_y = height - self.dino_height - 10
        self.dino_vel_y = 0
        self.gravity = 0.8
        self.jump_strength = -12
        self.is_jumping = False
        
        # Obstacle properties
        self.obstacle_width = 20
        self.obstacle_height = 40
        self.obstacles = []
        self.spawn_timer = 0
        self.spawn_interval = 60  # Frames between obstacle spawns
        
        # Game properties
        self.score = 0
        self.game_speed = 5
        self.is_done = False
        
    def reset(self):
        self.dino_y = self.height - self.dino_height - 10
        self.dino_vel_y = 0
        self.is_jumping = False
        self.obstacles = []
        self.score = 0
        self.game_speed = 5
        self.is_done = False
        self.spawn_timer = 0
        return self._get_state()
        
    def step(self, action):
        reward = 0.1  # Small positive reward for surviving
        
        # Handle jumping
        if action == 1 and not self.is_jumping:  # Jump action
            self.dino_vel_y = self.jump_strength
            self.is_jumping = True
        
        # Apply gravity
        self.dino_vel_y += self.gravity
        self.dino_y += self.dino_vel_y
        
        # Ground collision
        if self.dino_y > self.height - self.dino_height - 10:
            self.dino_y = self.height - self.dino_height - 10
            self.dino_vel_y = 0
            self.is_jumping = False
            
        # Spawn and update obstacles
        self.spawn_timer += 1
        if self.spawn_timer >= self.spawn_interval:
            self.spawn_timer = 0
            self.obstacles.append([self.width, self.height - self.obstacle_height - 10])
            
        # Update obstacles and check collisions
        for obstacle in self.obstacles[:]:
            obstacle[0] -= self.game_speed
            
            # Check collision
            dino_rect = pygame.Rect(self.dino_x, self.dino_y, self.dino_width, self.dino_height)
            obstacle_rect = pygame.Rect(obstacle[0], obstacle[1], 
                                     self.obstacle_width, self.obstacle_height)
            
            if dino_rect.colliderect(obstacle_rect):
                self.is_done = True
                reward = -1
            
            # Remove off-screen obstacles
            if obstacle[0] + self.obstacle_width < 0:
                self.obstacles.remove(obstacle)
                self.score += 1
                reward = 1  # Reward for passing obstacle
                
        # Increase game speed gradually
        if self.score > 0 and self.score % 5 == 0:
            self.game_speed = min(self.game_speed + 0.01, 10)
            
        return self._get_state(), reward, self.is_done, {}
        
    def render(self):
        # Clear screen
        self.screen.fill(self.WHITE)
        
        # Draw dino
        pygame.draw.rect(self.screen, self.BLACK, 
                        (self.dino_x, self.dino_y, self.dino_width, self.dino_height))
        
        # Draw obstacles
        for obstacle in self.obstacles:
            pygame.draw.rect(self.screen, self.BLACK,
                           (obstacle[0], obstacle[1], self.obstacle_width, self.obstacle_height))
        
        # Convert surface to numpy array
        state = pygame.surfarray.array3d(self.screen)
        return np.transpose(state, (1, 0, 2))  # Transpose to (H, W, C)
        
    def _get_state(self):
        # Render the game state
        state = self.render()
        
        # Convert to grayscale
        state = np.dot(state[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        
        # Resize to 84x84
        state = Image.fromarray(state).resize((84, 84))
        state = np.array(state)
        
        # Add channel dimension and normalize
        state = state[..., np.newaxis].astype(np.float32) / 255.0
        return state

class DinoEnv(Env):
    def __init__(self, render_mode="human"):
        super().__init__()
        self.game = DinoGame()
        self.render_mode = render_mode
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(2)  # 0: do nothing, 1: jump
        self.observation_space = spaces.Box(
            low=0, high=1,
            shape=(84, 84, 1),
            dtype=np.float32
        )
        
        # Create display if render mode is human
        if render_mode == "human":
            pygame.display.init()
            self.window = pygame.display.set_mode((self.game.width, self.game.height))
            pygame.display.set_caption("Dino Game")
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        observation = self.game.reset()
        return observation, {}
        
    def step(self, action):
        observation, reward, done, info = self.game.step(action)
        
        if self.render_mode == "human":
            self._render_frame()
            
        return observation, reward, done, False, info
        
    def _render_frame(self):
        # Get the game surface and display it
        game_surface = pygame.surfarray.make_surface(
            np.transpose(self.game.render(), (1, 0, 2))
        )
        self.window.blit(game_surface, (0, 0))
        pygame.display.flip()
        
    def render(self):
        if self.render_mode == "human":
            self._render_frame()
        return self.game.render()
        
    def close(self):
        if self.render_mode == "human":
            pygame.display.quit()
            pygame.quit()
