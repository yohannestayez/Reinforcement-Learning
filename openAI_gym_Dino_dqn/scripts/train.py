import sys
import os
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from collections import deque

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from envs.dino_env import DinoEnv
from models.dqn_agent import DQNAgent

def plot_metrics(scores, avg_scores, losses, save_dir):
    # Plot scores
    plt.figure(figsize=(10, 5))
    plt.plot(scores, label='Score')
    plt.plot(avg_scores, label='Average Score')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Training Progress')
    plt.legend()
    plt.savefig(str(save_dir / 'scores.png'))
    plt.close()
    
    # Plot losses
    if losses:
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.savefig(str(save_dir / 'losses.png'))
        plt.close()

def train(render=False):
    # Create directories for saving
    save_dir = Path("data")
    model_dir = save_dir / "models"
    log_dir = save_dir / "logs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    for dir_path in [model_dir, log_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize environment and agent
    env = DinoEnv(render_mode="human" if render else None)
    agent = DQNAgent()
    
    # Training parameters
    episodes = 500
    max_steps = 5000
    save_freq = 100
    print_freq = 10
    
    # Metrics
    scores = []
    losses = []
    avg_scores = []
    score_window = deque(maxlen=100)
    best_avg_score = float('-inf')
    
    # Training loop
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = []
        
        for step in range(max_steps):
            # Select and perform action
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition and train
            agent.buffer.push(state, action, reward, next_state, done)
            loss = agent.train_step()
            if loss is not None:
                episode_loss.append(loss)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # Update metrics
        scores.append(episode_reward)
        score_window.append(episode_reward)
        avg_score = np.mean(score_window)
        avg_scores.append(avg_score)
        if episode_loss:
            losses.extend(episode_loss)
        
        # Print progress
        if (episode + 1) % print_freq == 0:
            print(f"Episode {episode + 1}/{episodes}")
            print(f"Score: {episode_reward:.2f}")
            print(f"Average Score: {avg_score:.2f}")
            print(f"Epsilon: {agent.epsilon:.2f}")
            if episode_loss:
                print(f"Average Loss: {np.mean(episode_loss):.4f}")
            print("-" * 40)
        
        # Save best model
        if avg_score > best_avg_score:
            best_avg_score = avg_score
            agent.save(str(model_dir / "best_model.pth"))
        
        # Save checkpoint
        if (episode + 1) % save_freq == 0:
            agent.save(str(model_dir / f"checkpoint_{episode + 1}.pth"))
            plot_metrics(scores, avg_scores, losses, log_dir)
    
    # Final save and plot
    agent.save(str(model_dir / "final_model.pth"))
    plot_metrics(scores, avg_scores, losses, log_dir)
    
    return agent

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="Render the environment")
    args = parser.parse_args()
    
    train(render=args.render)