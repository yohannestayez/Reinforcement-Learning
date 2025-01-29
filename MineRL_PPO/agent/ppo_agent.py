import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import logging

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        # Adjusted CNN layers for 64x64 input
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)  # Output: 15x15
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)  # Output: 6x6
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)  # Output: 4x4
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1024, 512)  # 64 * 4 * 4 = 1024
        
    def forward(self, x):
        # Ensure input is float and normalized
        x = x.float() / 255.0
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.flatten(x)
        x = torch.relu(self.fc(x))
        return x

class PolicyNetwork(nn.Module):
    def __init__(self, num_actions):
        super(PolicyNetwork, self).__init__()
        self.cnn = CNNFeatureExtractor()
        
        # Additional inputs processing
        self.inventory_encoder = nn.Linear(2, 64)  # wood and coal counts
        self.compass_encoder = nn.Linear(1, 64)    # compass angle
        
        # Combine all features
        self.feature_combiner = nn.Linear(512 + 64 + 64, 512)
        
        # Policy head
        self.policy = nn.Linear(512, num_actions)
        self.value = nn.Linear(512, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, pov, inventory, compass):
        # Process visual input
        pov_features = self.cnn(pov)
        
        # Process inventory and compass
        inventory_features = torch.relu(self.inventory_encoder(inventory))
        compass_features = torch.relu(self.compass_encoder(compass))
        
        # Combine features
        combined = torch.cat([pov_features, inventory_features, compass_features], dim=1)
        features = torch.relu(self.feature_combiner(combined))
        
        # Get policy and value
        action_logits = self.policy(features)
        action_probs = torch.softmax(action_logits, dim=-1)
        value = self.value(features)
        
        return action_probs, value

class PPOAgent:
    def __init__(self, env, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.env = env
        self.device = device
        self.num_actions = env.action_space.n
        
        # Initialize networks
        self.policy = PolicyNetwork(self.num_actions).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=2.5e-4, eps=1e-5)
        
        # PPO parameters
        self.clip_epsilon = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        self.ppo_epochs = 10
        self.batch_size = 32
        self.gamma = 0.99
        self.gae_lambda = 0.95
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Storage for experience
        self.reset_storage()
    
    def reset_storage(self):
        """Reset storage for new episode."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.advantages = None
        self.returns = None
    
    def preprocess_observation(self, obs):
        """Convert observation to tensor format."""
        # Process POV image
        pov = torch.FloatTensor(obs['pov']).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # Process inventory
        inventory = torch.FloatTensor([
            obs['inventory']['wood'][0],
            obs['inventory']['coal'][0]
        ]).unsqueeze(0).to(self.device)
        
        # Process compass
        compass = torch.FloatTensor([obs['compass'][0]]).unsqueeze(0).to(self.device)
        
        return pov, inventory, compass
    
    def select_action(self, obs):
        """Select action using current policy."""
        with torch.no_grad():
            pov, inventory, compass = self.preprocess_observation(obs)
            action_probs, value = self.policy(pov, inventory, compass)
            
            # Sample action from probability distribution
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            return action.item(), value.item(), log_prob.item()
    
    def store_transition(self, obs, action, reward, value, log_prob, done):
        """Store transition in episode storage."""
        self.states.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def compute_gae(self, next_value):
        """Compute Generalized Advantage Estimation."""
        rewards = torch.FloatTensor(self.rewards).to(self.device)
        values = torch.FloatTensor(self.values + [next_value]).to(self.device)
        dones = torch.FloatTensor(self.dones + [0]).to(self.device)
        
        advantages = []
        gae = 0
        
        for step in reversed(range(len(self.rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + torch.FloatTensor(self.values).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update_policy(self, next_obs):
        """Update policy using PPO algorithm."""
        # Compute advantages and returns
        with torch.no_grad():
            pov, inventory, compass = self.preprocess_observation(next_obs)
            _, next_value = self.policy(pov, inventory, compass)
            advantages, returns = self.compute_gae(next_value.item())
        
        # Convert episode storage to tensors
        batch_obs = [self.preprocess_observation(obs) for obs in self.states]
        batch_pov = torch.cat([obs[0] for obs in batch_obs])
        batch_inventory = torch.cat([obs[1] for obs in batch_obs])
        batch_compass = torch.cat([obs[2] for obs in batch_obs])
        
        batch_actions = torch.LongTensor(self.actions).to(self.device)
        batch_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        
        # PPO update
        for _ in range(self.ppo_epochs):
            # Get current policy distribution and values
            action_probs, values = self.policy(batch_pov, batch_inventory, batch_compass)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(batch_actions)
            entropy = dist.entropy().mean()
            
            # Calculate ratio and surrogate loss
            ratio = torch.exp(new_log_probs - batch_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            
            # Calculate losses
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = 0.5 * (returns - values.squeeze()).pow(2).mean()
            
            # Combined loss
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            # Log statistics
            self.logger.debug(f"Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}, Entropy: {entropy.item():.4f}")
        
        # Reset storage
        self.reset_storage()
    
    def save(self, path):
        """Save model to disk."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'hyperparameters': {
                'clip_epsilon': self.clip_epsilon,
                'value_coef': self.value_coef,
                'entropy_coef': self.entropy_coef,
                'max_grad_norm': self.max_grad_norm,
                'ppo_epochs': self.ppo_epochs,
                'batch_size': self.batch_size,
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda
            }
        }, path)
    
    def load(self, path):
        """Load model from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load hyperparameters
        hyperparameters = checkpoint.get('hyperparameters', {})
        for key, value in hyperparameters.items():
            setattr(self, key, value)
