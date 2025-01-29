import numpy as np
import gym
from gym import spaces
import pyglet
from pyglet import shapes
import math

class SimpleNavigationEnv(gym.Env):
    """
    A simple 2D navigation environment where an agent needs to reach a target.
    The agent can move in 2D space and the target is randomly placed.
    """
    def __init__(self):
        super(SimpleNavigationEnv, self).__init__()
        
        # Define action and observation space
        # Action space: [move_x, move_y] each in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
        # Observation space: [agent_x, agent_y, target_x, target_y]
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(4,), dtype=np.float32
        )
        
        # Environment parameters
        self.max_steps = 100
        self.current_step = 0
        self.agent_pos = [0.0, 0.0]
        self.target_pos = [0.0, 0.0]
        
        # Visualization
        self.window = None
        self.grid_lines = []
        self.shapes = []
    
    def reset(self):
        """Reset the environment to initial state."""
        self.current_step = 0
        
        # Random initial position for agent
        self.agent_pos = [
            float(np.random.uniform(-8, 8)),
            float(np.random.uniform(-8, 8))
        ]
        
        # Random target position
        self.target_pos = [
            float(np.random.uniform(-8, 8)),
            float(np.random.uniform(-8, 8))
        ]
        
        # Ensure target is not too close to agent
        while self._get_distance() < 2.0:
            self.target_pos = [
                float(np.random.uniform(-8, 8)),
                float(np.random.uniform(-8, 8))
            ]
        
        return self._get_observation()
    
    def step(self, action):
        """Execute action and return new state, reward, done flag."""
        self.current_step += 1
        
        # Move agent
        self.agent_pos[0] = float(np.clip(self.agent_pos[0] + action[0], -10, 10))
        self.agent_pos[1] = float(np.clip(self.agent_pos[1] + action[1], -10, 10))
        
        # Calculate distance to target
        distance = self._get_distance()
        
        # Calculate reward
        if distance < 0.5:  # Agent reached target
            reward = 1.0
            done = True
        else:
            reward = -0.01  # Small negative reward for each step
            done = self.current_step >= self.max_steps
        
        return self._get_observation(), float(reward), done, {}
    
    def _get_distance(self):
        """Calculate distance between agent and target."""
        dx = self.agent_pos[0] - self.target_pos[0]
        dy = self.agent_pos[1] - self.target_pos[1]
        return float(np.sqrt(dx * dx + dy * dy))
    
    def _get_observation(self):
        """Return current observation."""
        return [
            self.agent_pos[0],
            self.agent_pos[1],
            self.target_pos[0],
            self.target_pos[1]
        ]
    
    def render(self, mode='human'):
        """Render the environment."""
        if mode == 'human':
            if self.window is None:
                self.window = pyglet.window.Window(800, 800, caption='Navigation Environment')
                self._create_grid_lines()
                
                @self.window.event
                def on_draw():
                    self.window.clear()
                    self._draw_environment()
            
            # Keep the window open and update
            pyglet.clock.tick()
            self.window.switch_to()
            self.window.dispatch_events()
            self.window.dispatch_event('on_draw')
            self.window.flip()
    
    def _create_grid_lines(self):
        """Create grid lines using shapes"""
        # Create vertical and horizontal grid lines
        for i in range(21):
            x = i * 40
            # Vertical line
            self.grid_lines.append(shapes.Line(x, 0, x, 800, color=(50, 50, 50)))
            # Horizontal line
            self.grid_lines.append(shapes.Line(0, x, 800, x, color=(50, 50, 50)))
    
    def _draw_environment(self):
        """Draw the entire environment"""
        # Draw grid
        for line in self.grid_lines:
            line.draw()
        
        # Convert coordinates to window space
        def world_to_screen(x, y):
            screen_x = int((x + 10) * 40)  # Scale and shift to fit window
            screen_y = int((y + 10) * 40)
            return screen_x, screen_y
        
        # Draw agent (blue circle)
        agent_x, agent_y = world_to_screen(self.agent_pos[0], self.agent_pos[1])
        agent_circle = shapes.Circle(agent_x, agent_y, 10, color=(0, 0, 255))
        agent_circle.draw()
        
        # Draw target (red circle)
        target_x, target_y = world_to_screen(self.target_pos[0], self.target_pos[1])
        target_circle = shapes.Circle(target_x, target_y, 10, color=(255, 0, 0))
        target_circle.draw()
    
    def close(self):
        """Close the environment."""
        if self.window is not None:
            self.window.close()
            self.window = None

if __name__ == "__main__":
    # Test the environment
    env = SimpleNavigationEnv()
    obs = env.reset()
    print("Initial observation:", obs)
    
    try:
        for i in range(100):
            action = [float(x) for x in env.action_space.sample()]
            obs, reward, done, _ = env.step(action)
            env.render()
            pyglet.clock.tick(30)  # Limit to 30 FPS
            
            if done:
                print("\nEpisode finished!")
                break
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    finally:
        env.close()
