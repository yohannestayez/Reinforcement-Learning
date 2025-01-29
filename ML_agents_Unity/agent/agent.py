import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim * 2)  # Mean and std for each action
        )
        
    def forward(self, state):
        x = self.net(state)
        mean, log_std = torch.chunk(x, 2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std.exp()

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, state):
        return self.net(state)

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, epsilon=0.2):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        self.gamma = gamma
        self.epsilon = epsilon
        
    def get_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            mean, std = self.actor(state)
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)  # Sum log probs for each action dimension
            return (
                action.squeeze(0).cpu().tolist(),
                log_prob.squeeze(0).cpu().item()
            )
    
    def update(self, states, actions, old_log_probs, rewards, dones):
        # Convert all inputs to tensors
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs).unsqueeze(-1)  # Add dimension for broadcasting
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)
        
        # Compute returns and advantages
        returns = []
        advantages = []
        value = 0
        
        for r, d in zip(reversed(rewards.tolist()), reversed(dones.tolist())):
            value = r + self.gamma * value * (1 - d)
            returns.insert(0, value)
            
        returns = torch.FloatTensor(returns).unsqueeze(-1)  # Add dimension for broadcasting
        values = self.critic(states)
        advantages = returns - values.detach()
        
        # Update policy
        mean, std = self.actor(states)
        dist = Normal(mean, std)
        new_log_probs = dist.log_prob(actions).sum(dim=-1, keepdim=True)  # Sum log probs and keep dimension
        
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Update value function
        critic_loss = nn.MSELoss()(values, returns)
        
        # Perform updates
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        return actor_loss.item(), critic_loss.item()
    
    def save(self, path):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
