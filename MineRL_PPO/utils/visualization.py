import matplotlib.pyplot as plt
import numpy as np
from typing import List
import os

def plot_training_rewards(rewards: List[float], save_path: str = None):
    """Plot training rewards over episodes."""
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.title('Training Rewards over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    # Add moving average
    window_size = min(100, len(rewards))
    if window_size > 0:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(rewards)), moving_avg, 'r', label='Moving Average')
    
    plt.legend()
    plt.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def plot_success_rate(successes: List[bool], window_size: int = 100, save_path: str = None):
    """Plot success rate over episodes."""
    success_rate = []
    for i in range(len(successes) - window_size + 1):
        rate = sum(successes[i:i+window_size]) / window_size
        success_rate.append(rate)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(window_size-1, len(successes)), success_rate)
    plt.title(f'Success Rate over Episodes (Window Size: {window_size})')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')
    plt.grid(True)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()
