# MineRL Campfire Challenge

A Proximal Policy Optimization (PPO) agent that learns to gather resources and light a campfire in Minecraft using MineRL.



## Project Structure
```
MineRL_Implementation/
├── agents/
│   └── ppo_agent.py        # PPO agent implementation
├── env/
│   └── minerl_env.py # Custom environment
├── utils/
│   └── visualization.py    # Training visualization
├── train.py             # Main training script
└── .gitignore          
```


## Requirements

### Python Dependencies
```
minerl         # Minecraft environment
torch          # Deep learning framework
numpy          # Numerical computations
gym            # Environment interface
matplotlib     # Training visualization
pygame         # Window management
```

###  **Start Training**:
```bash
python train.py
```


## Features

- **Environment**: Flat world with trees, coal, and campfire zone
- **Agent**: PPO with CNN for visual processing
- **Real-time**: Watch the agent learn in Minecraft
- **Rewards**:
  - +10: Collect wood/coal
  - +20: Reach campfire zone with materials
  - +50: Light campfire
  - -5: Wander too far
  - -10: Idle behavior

## Monitoring

- Watch real-time agent actions in Minecraft
- Check `data/plots/` for training metrics
- Review logs in `data/logs/` directory

