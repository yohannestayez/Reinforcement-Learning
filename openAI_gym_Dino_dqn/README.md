# Chrome Dinosaur Game DQN

This mini project implements a Deep Q-Network (DQN) agent that learns to play a Pygame-based version of the Chrome Dinosaur game. The agent learns to jump over obstacles to achieve the highest possible score.

## Project Structure

```
.
├── data/                  # Directory for saving models and training logs
│   ├── models/           # Saved model checkpoints
│   └── logs/             # Training logs and visualizations
├── envs/                 # Environment implementations
│   └── dino_game.py     # Pygame-based Dinosaur game environment
├── models/              # Model implementations
│   └── dqn_agent.py    # DQN agent implementation
├── scripts/             # Training and evaluation scripts
│   ├── train.py        # Script for training the agent
│   └── evaluate.py     # Script for evaluating trained models
└── requirements.txt    # Project dependencies
```
## Training

To train the agent:

```bash
# Train without rendering
python scripts/train.py

# Train with game visualization
python scripts/train.py --render
```

During training, the script will:
- Save the best performing model
- Save periodic checkpoints
- Generate training progress plots
- Display episode rewards and training metrics

## Evaluation

To evaluate a trained model:

```bash
# Evaluate the best model
python scripts/evaluate.py --model data/models/best_model.pth

# Evaluate specific checkpoint without rendering
python scripts/evaluate.py --model data/models/checkpoint_500.pth --no-render

# Run more evaluation episodes
python scripts/evaluate.py --model data/models/best_model.pth --episodes 20
```

## Environment

The game environment is a simplified version of the Chrome Dinosaur game:
- The dinosaur can jump over obstacles
- The game speed increases as the score increases
- The state is represented as an 84x84 grayscale image
- Actions: [0: do nothing, 1: jump]
- Rewards:
  - +0.1 for surviving each timestep
  - +1.0 for successfully passing an obstacle
  - -1.0 for collision with an obstacle

## Model Architecture

The DQN uses a convolutional neural network:
- 3 convolutional layers with ReLU activation
- 2 fully connected layers
- Input: 84x84x1 grayscale image
- Output: Q-values for each action

## Training Features

- Experience Replay Buffer (50,000 transitions)
- Target Network (updated every 1000 steps)
- Epsilon-greedy exploration (1.0 to 0.01)
- Adam optimizer with learning rate 0.00025
- Reward clipping
- Frame preprocessing (grayscale + resize)

