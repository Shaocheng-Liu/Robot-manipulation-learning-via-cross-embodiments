# World Model Integration Documentation

## Overview

This implementation adds a world model to the collective learning framework. The world model consists of three main components:

1. **Encoder**: Maps (state, task_encoding) → latent_state
2. **Dynamics Model**: Predicts next_latent_state = f(latent_state, action, task_encoding)  
3. **Reward Model**: Predicts reward = g(latent_state, action, task_encoding)

The world model follows the TD-MPC2 architecture with MLP-based components and proper normalization.

## Architecture Details

### Encoder: z = h(s, e)
- Input: State (39-dim) + Task encoding (6-dim) 
- Output: Latent state (512-dim)
- Architecture: 2-5 layer MLP with LayerNorm + Mish activation
- Final layer uses simplicial normalization (normalized to unit simplex)

### Dynamics Model: z' = d(z, a, e)  
- Input: Latent state (512-dim) + Action (4-dim) + Task encoding (6-dim)
- Output: Next latent state (512-dim)
- Architecture: 3-layer MLP with LayerNorm + Mish activation
- Final layer uses simplicial normalization

### Reward Model: r̂ = R(z, a, e)
- Input: Latent state (512-dim) + Action (4-dim) + Task encoding (6-dim)
- Output: Reward prediction (continuous value)
- Architecture: 3-layer MLP with discretized reward prediction (101 bins)
- Uses cross-entropy loss on discretized reward targets

## Integration with Collective Learning

When the world model is enabled:

1. **Actor Input**: Instead of raw states, the actor receives latent states from the encoder
2. **Critic Input**: Similarly uses latent states instead of raw states
3. **Training**: World model is trained alongside actor/critic during collective learning
4. **Loss Function**: Total loss = Actor loss + Critic loss + World model loss

## Configuration

### Enabling World Model

In `config/transformer_collective_network/transformer_collective_sac.yaml`:

```yaml
builder:
  # ... other parameters ...
  use_world_model: True  # Set to True to enable world model
  world_model_cfg: ${transformer_collective_network.world_model}
  world_model_optimizer_cfg: ${transformer_collective_network.optimizers.world_model}
```

### World Model Configuration

In `config/transformer_collective_network/components/world_model.yaml`:

```yaml
# Latent space dimension
latent_dim: 512

# Encoder configuration
encoder_layers: 2  # 2-5 layers as per TD-MPC2
encoder_hidden_dim: 256

# Dynamics model configuration  
dynamics_hidden_dim: 512

# Reward model configuration
reward_hidden_dim: 512
reward_bins: 101  # Discretized reward prediction bins

# World model training parameters
reward_bounds: [-10.0, 10.0]  # Min/max reward for discretization
```

### Optimizer Configuration

In `config/transformer_collective_network/optimizers/world_model.yaml`:

```yaml
_target_: torch.optim.Adam
lr: 1e-4
weight_decay: 1e-5
```

## Training Process

1. **Expert Training**: Regular SAC agents train on individual tasks
2. **Data Collection**: Expert trajectories are collected for collective learning
3. **Collective Learning**: 
   - Transformer trains to predict task encodings (CLS tokens)
   - World model trains to predict latent dynamics and rewards
   - SAC trains on latent states instead of raw states
4. **Student Learning**: Student agents learn from the collective network

## Benefits

1. **Latent Representation**: More compact and structured state representation
2. **Task-Aware Dynamics**: Dynamics model conditions on task encoding
3. **Sample Efficiency**: Latent space learning can be more sample efficient
4. **Transfer Learning**: Shared latent representation across tasks
5. **Model-Based Planning**: Enables future model-based planning capabilities

## Implementation Details

### Key Files Modified:
- `mtrl/agent/components/world_model.py`: World model components
- `mtrl/agent/transformer_agent.py`: Integration with transformer agent
- `mtrl/experiment/collective_learning.py`: Training loop integration
- Configuration files for world model settings

### Training Integration:
- World model training is called during both actor and critic training phases
- Loss computation includes dynamics loss (MSE) + reward loss (cross-entropy)
- Gradients flow through the world model during collective learning

### Latent State Usage:
- When `use_world_model=True`, actor and critic use latent states
- Raw states are encoded to latent space using the encoder
- Task encoding (CLS token) is concatenated with latent states for actor/critic input

## Usage Example

```python
# In your experiment configuration, set:
use_world_model: True

# The world model will automatically:
# 1. Encode raw states to latent space
# 2. Train dynamics and reward models
# 3. Provide latent states to actor/critic
# 4. Optimize world model alongside policy
```

## Testing

The implementation has been tested with:
- Component-level tests for encoder, dynamics, and reward models
- End-to-end forward and backward passes
- Loss computation and gradient flow
- Training step validation
- Latent space normalization checks

All tests pass successfully, confirming the world model is ready for integration with the collective learning pipeline.