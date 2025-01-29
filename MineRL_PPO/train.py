import os
import torch
import numpy as np
import logging
import time
from datetime import datetime
from envs.minerl_env import CampfireEnv
from agent.ppo_agent import PPOAgent
from utils.visualization import plot_training_rewards, plot_success_rate

def setup_logging():
    """Setup logging configuration."""
    log_dir = "data/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def train(num_episodes=1000, render_interval=10):
    """Train the PPO agent."""
    logger = setup_logging()
    
    # Create directories for saving
    os.makedirs("data/models", exist_ok=True)
    os.makedirs("data/plots", exist_ok=True)
    
    # Initialize environment and agent
    env = CampfireEnv()
    agent = PPOAgent(env)
    
    # Training loop
    best_reward = float('-inf')
    episode_rewards = []
    episode_successes = []
    
    logger.info("Starting training...")
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        # Episode loop
        while not done:
            # Render every render_interval episodes
            if episode % render_interval == 0:
                env.render()
            
            # Select action
            action, value, log_prob = agent.select_action(obs)
            
            # Execute action
            next_obs, reward, done, info = env.step(action)
            
            # Store transition
            agent.store_transition(obs, action, reward, value, log_prob, done)
            
            # Update observation and accumulate reward
            obs = next_obs
            episode_reward += reward
            step += 1
            
            # Log step information
            if step % 100 == 0:
                logger.debug(f"Episode {episode}, Step {step}: Action={action}, Reward={reward}")
        
        # Update policy
        agent.update_policy(next_obs)
        
        # Store episode metrics
        episode_rewards.append(episode_reward)
        episode_successes.append(info.get('campfire_lit', False))
        
        # Log episode information
        logger.info(f"Episode {episode}: Reward={episode_reward:.2f}, Steps={step}, Success={info.get('campfire_lit', False)}")
        
        # Save best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(os.path.join("data/models", "best_model.pth"))
            logger.info(f"New best model saved with reward: {best_reward:.2f}")
        
        # Save checkpoint every 100 episodes
        if episode % 100 == 0:
            agent.save(os.path.join("data/models", f"checkpoint_{episode}.pth"))
            
        # Calculate and log moving average
        if len(episode_rewards) >= 100:
            avg_reward = np.mean(episode_rewards[-100:])
            logger.info(f"Last 100 episodes average reward: {avg_reward:.2f}")
        
        # Plot and save metrics every 100 episodes
        if episode % 100 == 0 and episode > 0:
            plot_training_rewards(episode_rewards, save_path=os.path.join("data/plots", f"rewards_{episode}.png"))
            plot_success_rate(episode_successes, save_path=os.path.join("data/plots", f"success_rate_{episode}.png"))
    
    # Save final model
    agent.save(os.path.join("data/models", "final_model.pth"))
    
    # Save final plots
    plot_training_rewards(episode_rewards, save_path=os.path.join("data/plots", "final_rewards.png"))
    plot_success_rate(episode_successes, save_path=os.path.join("data/plots", "final_success_rate.png"))
    
    logger.info("Training completed!")
    
    return episode_rewards, episode_successes

if __name__ == "__main__":
    rewards, successes = train()
