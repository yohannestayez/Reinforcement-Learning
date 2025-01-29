import sys
from pathlib import Path
import numpy as np
import torch

# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from envs.dino_env import DinoEnv
from models.dqn_agent import DQNAgent

def evaluate(model_path, num_episodes=10, render=True):
    # Initialize environment and agent
    env = DinoEnv(render_mode="human" if render else None)
    agent = DQNAgent()
    
    # Load trained model
    agent.load(model_path)
    agent.epsilon = 0.0  # No exploration during evaluation
    
    scores = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Select action
            action = agent.act(state)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            state = next_state
            episode_reward += reward
        
        scores.append(episode_reward)
        print(f"Episode {episode + 1}: Score = {episode_reward}")
    
    print("\nEvaluation Results:")
    print(f"Average Score: {np.mean(scores):.2f}")
    print(f"Standard Deviation: {np.std(scores):.2f}")
    print(f"Max Score: {np.max(scores):.2f}")
    print(f"Min Score: {np.min(scores):.2f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to the model file")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to evaluate")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    args = parser.parse_args()
    
    evaluate(args.model, args.episodes, not args.no_render)