import numpy as np
import minerl
import gym
from gym import spaces
import logging

class CampfireEnv(gym.Env):
    """
    Custom MineRL environment for training an agent to gather resources and light a campfire.
    The agent must collect wood and coal, then return to a designated zone to light a campfire.
    """
    
    def __init__(self):
        super(CampfireEnv, self).__init__()
        
        # Initialize the base MineRL environment
        self.env = minerl.data.make("MineRLObtainDiamond-v0")  # Using this as base since it has tree/mining mechanics
        
        # Define action space (simplified for our task)
        self.action_space = spaces.Discrete(8)  # move forward/back, turn left/right, jump, mine, place, craft
        
        # Define base actions for MineRL
        self.base_actions = {
            'forward': [0, 1],  # Not moving, moving forward
            'back': [0, 1],     # Not moving, moving backward
            'left': [0, 1],     # Not turning, turning left
            'right': [0, 1],    # Not turning, turning right
            'jump': [0, 1],     # Not jumping, jumping
            'attack': [0, 1],   # Not attacking, attacking
            'craft': [0, 1],    # Not crafting, crafting
            'place': [0, 1]     # Not placing, placing
        }
        
        # Define observation space
        self.observation_space = spaces.Dict({
            'pov': spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            'inventory': spaces.Dict({
                'wood': spaces.Box(low=0, high=64, shape=(1,), dtype=np.int32),
                'coal': spaces.Box(low=0, high=64, shape=(1,), dtype=np.int32)
            }),
            'compass': spaces.Box(low=-180, high=180, shape=(1,), dtype=np.float32)
        })
        
        # Task-specific variables
        self.campfire_zone = None  # Will be set during reset
        self.max_steps = 2000
        self.current_step = 0
        self.last_action = None
        self.last_position = None
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
    
    def _get_observation(self, obs):
        """Process MineRL observation into our format."""
        inventory = obs.get('inventory', {})
        return {
            'pov': obs['pov'],
            'inventory': {
                'wood': np.array([inventory.get('log', 0)]),
                'coal': np.array([inventory.get('coal', 0)])
            },
            'compass': np.array([self._calculate_compass_angle()])
        }
    
    def reset(self):
        """Reset the environment and return initial observation."""
        # Reset the base environment
        obs = self.env.reset()
        
        # Reset task-specific variables
        self.current_step = 0
        self.campfire_zone = self._generate_campfire_zone()
        self.last_action = None
        self.last_position = self._get_agent_position(obs)
        
        # Process and return the initial observation
        return self._get_observation(obs)
    
    def step(self, action):
        """Execute action and return new state, reward, done, info."""
        # Convert our simplified action to MineRL action
        minerl_action = self._convert_action(action)
        
        # Execute action in base environment
        obs, reward, done, info = self.env.step(minerl_action)
        
        # Get current position
        current_position = self._get_agent_position(obs)
        
        # Process observation
        processed_obs = self._get_observation(obs)
        
        # Calculate reward
        reward = self._calculate_reward(processed_obs, current_position, info)
        
        # Update step counter and position
        self.current_step += 1
        self.last_position = current_position
        self.last_action = action
        
        # Check if episode should end
        done = done or self.current_step >= self.max_steps
        
        # Update info dict
        info.update({
            'steps': self.current_step,
            'in_campfire_zone': self._in_campfire_zone(current_position),
            'has_materials': self._has_required_materials(processed_obs)
        })
        
        return processed_obs, reward, done, info
    
    def _convert_action(self, action):
        """Convert simplified action to MineRL action format."""
        # Basic action mappings
        actions = {
            0: {'forward': 1},                  # Move forward
            1: {'back': 1},                     # Move backward
            2: {'left': 1},                     # Turn left
            3: {'right': 1},                    # Turn right
            4: {'jump': 1},                     # Jump
            5: {'attack': 1},                   # Mine
            6: {'place': 1},                    # Place
            7: {'craft': 1}                     # Craft
        }
        
        # Start with all actions set to 0
        minerl_action = {k: 0 for k in self.base_actions.keys()}
        
        # Update with the selected action
        if action in actions:
            minerl_action.update(actions[action])
        
        return minerl_action
    
    def _get_agent_position(self, obs):
        """Extract agent position from observation."""
        # In practice, you would need to implement position tracking
        # This is a placeholder that returns a dummy position
        return {'x': 0, 'y': 0, 'z': 0}
    
    def _generate_campfire_zone(self):
        """Generate coordinates for campfire zone."""
        return {
            'x': 0,
            'z': 0,
            'radius': 5
        }
    
    def _in_campfire_zone(self, position):
        """Check if agent is in campfire zone."""
        if not position or not self.campfire_zone:
            return False
        
        dx = position['x'] - self.campfire_zone['x']
        dz = position['z'] - self.campfire_zone['z']
        distance = (dx * dx + dz * dz) ** 0.5
        
        return distance <= self.campfire_zone['radius']
    
    def _has_required_materials(self, obs):
        """Check if agent has required materials."""
        return (obs['inventory']['wood'][0] > 0 and 
                obs['inventory']['coal'][0] > 0)
    
    def _calculate_reward(self, obs, position, info):
        """Calculate reward based on current state and actions."""
        reward = 0
        
        # Resource collection rewards
        if info.get('wood_collected', False):
            reward += 10
        if info.get('coal_collected', False):
            reward += 10
        
        # Position-based rewards
        in_zone = self._in_campfire_zone(position)
        has_materials = self._has_required_materials(obs)
        
        if in_zone and has_materials:
            reward += 20
        
        # Campfire placement reward
        if info.get('campfire_lit', False):
            reward += 50
        
        # Penalties
        if self._too_far_from_task_area(position):
            reward -= 5
        
        if self._is_idle():
            reward -= 1
        
        return reward
    
    def _too_far_from_task_area(self, position):
        """Check if agent has wandered too far."""
        if not position or not self.campfire_zone:
            return False
        
        dx = position['x'] - self.campfire_zone['x']
        dz = position['z'] - self.campfire_zone['z']
        distance = (dx * dx + dz * dz) ** 0.5
        
        return distance > 50  # Arbitrary threshold
    
    def _is_idle(self):
        """Check if agent is idle (not making progress)."""
        if self.last_position is None or self.last_action is None:
            return False
            
        # Consider the agent idle if it's been in the same position
        # with the same action for multiple steps
        return (self.last_action == self.last_action and
                self.last_position == self._get_agent_position(None))
    
    def _calculate_compass_angle(self):
        """Calculate angle to campfire zone."""
        if not self.last_position or not self.campfire_zone:
            return 0.0
            
        dx = self.campfire_zone['x'] - self.last_position['x']
        dz = self.campfire_zone['z'] - self.last_position['z']
        
        angle = np.arctan2(dz, dx) * 180 / np.pi
        return angle
    
    def render(self, mode='human'):
        """Render the environment."""
        return self.env.render(mode)
    
    def close(self):
        """Clean up resources."""
        self.env.close()
