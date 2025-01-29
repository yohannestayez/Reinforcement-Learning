import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

# Print CUDA information
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA Device:", torch.cuda.get_device_name(0))
    device = torch.device("cuda")
else:
    print("CUDA not available, using CPU")
    device = torch.device("cpu")

print(f"PyTorch Version: {torch.__version__}")
print(f"Using device: {device}")

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        self.input_shape = input_shape
        
        # Move to device before creating layers
        self.to(device)
        
        # Create layers
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4).to(device),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2).to(device),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1).to(device),
            nn.ReLU()
        )
        
        # Calculate the size of the convolution output
        conv_out_size = self._get_conv_out(input_shape)
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512).to(device),
            nn.ReLU(),
            nn.Linear(512, n_actions).to(device)
        )
        
        print(f"Model device: {next(self.parameters()).device}")
        
    def _get_conv_out(self, shape):
        # Run a test forward pass to get the conv output size
        o = self.conv(torch.zeros(1, *shape, device=device))
        return int(np.prod(o.size()))
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Run through convolutional layers
        conv_out = self.conv(x)
        
        # Flatten the output while preserving batch size
        conv_out = conv_out.reshape(batch_size, -1)
        
        # Run through fully connected layers
        return self.fc(conv_out)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, input_shape=(1, 84, 84), n_actions=2):
        self.input_shape = input_shape
        self.n_actions = n_actions
        
        # Networks
        self.policy_net = DQN(input_shape, n_actions)
        self.target_net = DQN(input_shape, n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        print(f"Policy net device: {next(self.policy_net.parameters()).device}")
        print(f"Target net device: {next(self.target_net.parameters()).device}")
        
        # Training parameters
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update = 1000
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.00025)
        
        # Initialize replay buffer
        self.buffer = ReplayBuffer(50000)
        self.steps = 0
        
    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
            
        with torch.no_grad():
            # Convert state to tensor and ensure it's on the correct device
            state_tensor = torch.tensor(np.array(state), dtype=torch.float32, device=device).unsqueeze(0).permute(0, 3, 1, 2)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
            
    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None
            
        # Sample from replay buffer
        transitions = self.buffer.sample(self.batch_size)
        batch = list(zip(*transitions))
        
        # Prepare batch and ensure all tensors are on the correct device
        state_batch = torch.tensor(np.array(batch[0]), dtype=torch.float32, device=device).permute(0, 3, 1, 2)
        action_batch = torch.tensor(batch[1], dtype=torch.long, device=device)
        reward_batch = torch.tensor(batch[2], dtype=torch.float32, device=device)
        next_state_batch = torch.tensor(np.array(batch[3]), dtype=torch.float32, device=device).permute(0, 3, 1, 2)
        done_batch = torch.tensor(batch[4], dtype=torch.float32, device=device)
        
        # Compute current Q values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Compute next Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            next_q_values[done_batch == 1] = 0.0
            expected_q_values = reward_batch + self.gamma * next_q_values
            
        # Compute loss and update
        loss = nn.MSELoss()(current_q_values, expected_q_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
        
    def save(self, path):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, path)
        
    def load(self, path):
        checkpoint = torch.load(path, map_location=device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']