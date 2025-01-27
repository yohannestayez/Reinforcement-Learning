import gym
from gym_chrome_dino import ChromeDinoEnv
import numpy as np

class DinoEnv(ChromeDinoEnv):
    def __init__(self):
        super().__init__(render_mode="human")  # Use "human" for visualization
        self.action_space = gym.spaces.Discrete(3)  # 0: do nothing, 1: jump, 2: duck
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)
        
    def reset(self):
        obs = super().reset()
        return self.preprocess(obs)
    
    def step(self, action):
        obs, reward, done, info = super().step(action)
        return self.preprocess(obs), reward, done, info
    
    def preprocess(self, obs):
        # Convert to grayscale, resize, and add channel dimension
        obs = np.dot(obs[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)  # RGB to grayscale
        obs = np.array(Image.fromarray(obs).resize((84, 84)))  # Resize to 84x84
        return obs[..., np.newaxis]  # Shape: (84, 84, 1)