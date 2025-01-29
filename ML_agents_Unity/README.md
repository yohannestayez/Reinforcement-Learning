# ML-Agents Navigation Project

A reinforcement learning project using Unity ML-Agents for training an agent to navigate to a target. Features both a simple Python environment and a visually identical Unity environment.

## Project Structure
```
.
├── agent
│   └── agent.py               # PPO agent implementation
├── config                # ML-Agents behavior parameters
├── data
├── envs
│   └── unity_env.py    # Simple environment implementation
├── scripts
│   ├── environment.py          # Environment wrapper
│   └── train.py # Training script
├── tests
│   └── tests.py
├──UnityProject           # Unity ML-Agents implementation
│    └── Assets
│        ├── Resources
│        │   └── MLAgents Settings.asset
│        └── Scripts
││           └── NavigationAgent.cs
├── .gitignore
└── README.md
```

## Usage

1. Test Environment
```bash
python scripts/test_env.py
```

2. Train Agent
```bash
# Start Unity Editor and press Play
python scripts/train.py
```

## Configuration

- `behavior.yaml`: PPO hyperparameters
  - Network: 2 layers, 128 units each
  - Learning rate: 0.0003
  - Batch size: 64
  - Max steps: 500,000

## Implementation Notes

- Unity environment matches simple environment
- NavigationAgent.cs handles automatic visual matching
- Training can use either environment for consistency
