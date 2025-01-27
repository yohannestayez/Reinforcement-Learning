import sys
import os
import numpy as np
import torch
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from envs.dino_env import DinoEnv
from models.dqn_agent import DQNAgent

def train():
    # Environment setup
    env = DinoEnv()
    input_shape = (1, 84, 84)  # (C, H, W)
    n_actions = env.action_space.n
    
    # Agent setup
    agent = DQNAgent(input_shape, n_actions)
    episodes = 1000
    batch_size = 64
    update_target_freq = 1000  # Update target network every 1000 steps
    
    # Logging setup
    log_dir = Path("data/logs") / datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    total_steps = 0
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.buffer.push(state, action, reward, next_state, done)
            episode_reward += reward
            state = next_state
            total_steps += 1
            
            # Train on batch
            loss = agent.train_step(batch_size)
            
            # Update target network
            if total_steps % update_target_freq == 0:
                agent.update_target_network()
            
            # Decay epsilon
            agent.update_epsilon()
        
        # Save checkpoint
        if episode % 50 == 0:
            checkpoint_dir = Path("data/checkpoints")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(agent.policy_net.state_dict(), checkpoint_dir / f"dqn_episode_{episode}.pth")
        
        print(f"Episode {episode}, Reward: {episode_reward}, Epsilon: {agent.epsilon:.2f}, Loss: {loss:.4f}")

if __name__ == "__main__":
    train()