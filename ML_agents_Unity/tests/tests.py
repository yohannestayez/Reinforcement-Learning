import numpy as np
import pyglet

import sys
from pathlib import Path
# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from scripts.enviroment import NavigationEnvironment


def test_environment():
    """Test the navigation environment."""
    print("\nInitializing environment...")
    env = NavigationEnvironment(use_unity=False)  # Use simple environment for testing
    
    print("\nTesting environment reset...")
    initial_state = env.reset()
    print(f"Initial state shape: {initial_state.shape if isinstance(initial_state, np.ndarray) else len(initial_state)}")
    print(f"Initial state: {initial_state}")
    
    print("\nTesting environment step...")
    try:
        num_episodes = 3
        steps_per_episode = 50
        
        for episode in range(num_episodes):
            print(f"\nEpisode {episode + 1}/{num_episodes}")
            state = env.reset()
            episode_reward = 0
            
            for step in range(steps_per_episode):
                # Sample a random action
                action = np.random.uniform(-1, 1, size=2)
                
                # Take a step
                next_state, reward, done = env.step(action)
                episode_reward += reward
                
                # Render the environment
                env.render()
                pyglet.clock.tick(30)  # Limit to 30 FPS
                
                print(f"Step {step + 1}: Action={action}, Reward={reward:.3f}")
                
                if done:
                    print(f"Episode finished after {step + 1} steps")
                    break
                
                state = next_state
            
            print(f"Episode {episode + 1} finished with total reward: {episode_reward:.3f}")
        
        print("\nEnvironment test completed successfully!")
    
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    
    finally:
        env.close()
        print("Environment closed")

if __name__ == "__main__":
    test_environment()
