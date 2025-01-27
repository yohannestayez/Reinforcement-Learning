import sys
import os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from envs.dino_env import DinoEnv
from models.dqn_agent import DQNAgent

def evaluate(checkpoint_path, n_episodes=10):
    # Environment setup
    env = DinoEnv()
    input_shape = (1, 84, 84)
    n_actions = env.action_space.n
    
    # Agent setup
    agent = DQNAgent(input_shape, n_actions)
    agent.policy_net.load_state_dict(torch.load(checkpoint_path))
    agent.epsilon = 0.0  # Disable exploration
    
    rewards = []
    for _ in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        
        rewards.append(episode_reward)
    
    # Plot rewards
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Evaluation Rewards")
    plt.savefig(Path("data/logs") / "evaluation_rewards.png")
    plt.show()

if __name__ == "__main__":
    checkpoint = "data/checkpoints/dqn_episode_950.pth"  # Example checkpoint
    evaluate(checkpoint)