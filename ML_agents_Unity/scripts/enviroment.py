from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple
import numpy as np
import sys
from pathlib import Path
# Add parent directory to path for imports
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from envs.unity_env import SimpleNavigationEnv

class NavigationEnvironment:
    def __init__(self, worker_id=0, use_unity=False):
        """Initialize the environment."""
        self.use_unity = use_unity
        
        if use_unity:
            # Unity environment setup
            self.engine_configuration_channel = EngineConfigurationChannel()
            self.env = UnityEnvironment(
                file_name=None,  # Use editor
                worker_id=worker_id,
                side_channels=[self.engine_configuration_channel]
            )
            self.env.reset()
            
            # Get the first behavior name
            self.behavior_name = list(self.env.behavior_specs)[0]
            self.spec = self.env.behavior_specs[self.behavior_name]
        else:
            # Use simple environment for testing
            self.env = SimpleNavigationEnv()
            self.spec = type('EnvSpec', (), {
                'observation_specs': [type('ObsSpec', (), {'shape': (4,)})()],
                'action_spec': type('ActionSpec', (), {'continuous_size': 2})()
            })
    
    def reset(self):
        """Reset the environment and return initial observations."""
        if self.use_unity:
            self.env.reset()
            decision_steps, _ = self.env.get_steps(self.behavior_name)
            return decision_steps.obs[0][0]
        else:
            return self.env.reset()
    
    def step(self, action):
        """Execute action and return new state, reward, done flag."""
        if self.use_unity:
            # Convert action to Unity format
            action_tuple = ActionTuple(continuous=np.array([action]))
            
            # Set the actions
            self.env.set_actions(self.behavior_name, action_tuple)
            
            # Move the simulation forward
            self.env.step()
            
            # Get the new simulation results
            decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
            
            if len(terminal_steps.agent_id) > 0:  # Episode ended
                reward = terminal_steps.reward[0]
                done = True
                next_state = terminal_steps.obs[0][0]
            else:
                reward = decision_steps.reward[0]
                done = False
                next_state = decision_steps.obs[0][0]
            
            return next_state, reward, done
        else:
            next_obs, reward, done, _ = self.env.step(action)
            return next_obs, reward, done
    
    def render(self, mode='human'):
        """Render the environment."""
        if not self.use_unity:
            self.env.render(mode)
    
    def close(self):
        """Close the environment."""
        if self.use_unity:
            self.env.close()
        else:
            pass  # No need to close the simple environment
