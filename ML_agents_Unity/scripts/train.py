import os
import numpy as np
import yaml
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import pyglet

import sys
from pathlib import Path
# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from enviroment import NavigationEnvironment
from agent.agent import PPOAgent

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class Logger:
    def __init__(self, log_dir='data/logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.rewards = []
        self.lengths = []
        self.actor_losses = []
        self.critic_losses = []
        
    def log_episode(self, reward, length):
        self.rewards.append(reward)
        self.lengths.append(length)
        
    def log_training(self, actor_loss, critic_loss):
        self.actor_losses.append(actor_loss)
        self.critic_losses.append(critic_loss)
        
    def plot_metrics(self):
        if not self.rewards:  # Skip if no data
            return
            
        # Plot episode rewards
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(self.rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        # Plot episode lengths
        plt.subplot(2, 2, 2)
        plt.plot(self.lengths)
        plt.title('Episode Lengths')
        plt.xlabel('Episode')
        plt.ylabel('Length')
        
        # Plot actor losses
        plt.subplot(2, 2, 3)
        plt.plot(self.actor_losses)
        plt.title('Actor Loss')
        plt.xlabel('Update')
        plt.ylabel('Loss')
        
        # Plot critic losses
        plt.subplot(2, 2, 4)
        plt.plot(self.critic_losses)
        plt.title('Critic Loss')
        plt.xlabel('Update')
        plt.ylabel('Loss')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'training_metrics.png'))
        plt.close()
    
    def get_stats(self):
        """Get training statistics safely"""
        stats = {
            'last_10_avg': 0.0,
            'best_reward': 0.0,
            'total_episodes': len(self.rewards)
        }
        
        if self.rewards:
            stats['last_10_avg'] = float(np.mean(self.rewards[-10:]))
            stats['best_reward'] = float(max(self.rewards))
        
        return stats

def main():
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/models', exist_ok=True)
    
    # Initialize logger
    logger = Logger()
    
    # Load configuration
    config = load_config('config/behavior.yaml')
    hyperparams = config['behaviors']['NavigationAgent']['hyperparameters']
    
    # Initialize environment and agent
    env = NavigationEnvironment()
    state_dim = env.spec.observation_specs[0].shape[0]
    action_dim = env.spec.action_spec.continuous_size
    
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=hyperparams['learning_rate'],
        gamma=config['behaviors']['NavigationAgent']['reward_signals']['extrinsic']['gamma'],
        epsilon=hyperparams['epsilon']
    )
    
    # Training parameters
    num_episodes = 100
    batch_size = hyperparams['batch_size']
    
    states = []
    actions = []
    log_probs = []
    rewards = []
    dones = []
    
    print("\nStarting training for 100 episodes...")
    print("=" * 50)
    
    try:
        for episode in range(num_episodes):
            episode_reward = 0
            episode_length = 0
            state = env.reset()
            
            while True:
                # Collect experience
                action, log_prob = agent.get_action(state)
                next_state, reward, done = env.step(action)
                
                # Visualize the environment
                env.render()
                pyglet.clock.tick(30)  # Limit to 30 FPS
                
                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                dones.append(done)
                
                episode_reward += reward
                episode_length += 1
                
                # Update policy if we have enough experience
                if len(states) >= batch_size:
                    actor_loss, critic_loss = agent.update(
                        states=states,
                        actions=actions,
                        old_log_probs=log_probs,
                        rewards=rewards,
                        dones=dones
                    )
                    logger.log_training(actor_loss, critic_loss)
                    
                    states.clear()
                    actions.clear()
                    log_probs.clear()
                    rewards.clear()
                    dones.clear()
                
                if done:
                    break
                    
                state = next_state
            
            # Log episode results
            logger.log_episode(episode_reward, episode_length)
            
            # Print progress
            print(f"Episode {episode+1}/100 - Reward: {episode_reward:.2f}, Length: {episode_length}")
            
            # Save checkpoint and plot metrics every 10 episodes
            if (episode + 1) % 10 == 0:
                checkpoint_path = f'data/models/checkpoint_episode_{episode+1}.pt'
                agent.save(checkpoint_path)
                logger.plot_metrics()
                print(f"Saved checkpoint and plots at episode {episode+1}")
                print("-" * 50)
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    finally:
        # Final save
        agent.save('data/models/final_model.pt')
        logger.plot_metrics()
        env.close()
        
        # Get and print final statistics
        stats = logger.get_stats()
        print("\nTraining completed!")
        if stats['total_episodes'] > 0:
            print(f"Final average reward (last 10 episodes): {stats['last_10_avg']:.2f}")
            print(f"Best episode reward: {stats['best_reward']:.2f}")
        print(f"Total episodes completed: {stats['total_episodes']}")
        print("=" * 50)

if __name__ == "__main__":
    main()
