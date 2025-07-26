# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""World Model components for collective learning with latent dynamics."""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mtrl.agent.components import base as base_component
from mtrl.utils.types import TensorType


class NormedLinear(nn.Module):
    """Linear layer with LayerNorm and activation as used in TD-MPC2."""
    
    def __init__(self, in_features: int, out_features: int, act: str = "Mish"):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        
        if act == "Mish":
            self.activation = nn.Mish()
        elif act == "ReLU":
            self.activation = nn.ReLU()
        elif act == "SimNorm":
            # Simplicial normalization - normalize to unit simplex
            self.activation = lambda x: F.normalize(x, p=1, dim=-1)
        else:
            raise ValueError(f"Unsupported activation: {act}")
    
    def forward(self, x: TensorType) -> TensorType:
        x = self.linear(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class WorldModelEncoder(base_component.Component):
    """Encoder that maps (state, task_encoding) -> latent_state.
    
    Following TD-MPC2 architecture: z = h(s, e)
    """
    
    def __init__(
        self,
        state_dim: int,
        task_encoding_dim: int,
        latent_dim: int,
        num_layers: int = 2,
        hidden_dim: int = 256,
    ):
        """Initialize the world model encoder.
        
        Args:
            state_dim: Dimension of the state observation
            task_encoding_dim: Dimension of the task encoding (CLS token)
            latent_dim: Dimension of the latent representation
            num_layers: Number of hidden layers (2-5 as per TD-MPC2)
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.task_encoding_dim = task_encoding_dim
        self.latent_dim = latent_dim
        
        # Input dimension is state + task encoding
        input_dim = state_dim + task_encoding_dim
        
        # Build encoder network
        layers = []
        in_dim = input_dim
        
        # Hidden layers with NormedLinear
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else latent_dim
            act = "Mish" if i < num_layers - 1 else "SimNorm"
            layers.append(NormedLinear(in_dim, out_dim, act=act))
            in_dim = out_dim
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, state: TensorType, task_encoding: TensorType) -> TensorType:
        """Encode state and task encoding to latent representation.
        
        Args:
            state: State observation [batch_size, state_dim]
            task_encoding: Task encoding (CLS token) [batch_size, task_encoding_dim]
            
        Returns:
            Latent representation [batch_size, latent_dim]
        """
        # Concatenate state and task encoding
        x = torch.cat([state, task_encoding], dim=-1)
        
        # Encode to latent space
        latent = self.encoder(x)
        
        return latent


class WorldModelDynamics(base_component.Component):
    """Dynamics model that predicts next latent state.
    
    Following TD-MPC2 architecture: z' = d(z, a, e)
    """
    
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        task_encoding_dim: int,
        hidden_dim: int = 512,
    ):
        """Initialize the dynamics model.
        
        Args:
            latent_dim: Dimension of the latent representation
            action_dim: Dimension of the action space
            task_encoding_dim: Dimension of the task encoding
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.task_encoding_dim = task_encoding_dim
        
        # Input is latent + action + task encoding
        input_dim = latent_dim + action_dim + task_encoding_dim
        
        # 3-layer MLP as per TD-MPC2
        self.dynamics = nn.Sequential(
            NormedLinear(input_dim, hidden_dim, act="Mish"),
            NormedLinear(hidden_dim, hidden_dim, act="Mish"),
            NormedLinear(hidden_dim, latent_dim, act="SimNorm")
        )
    
    def forward(
        self, 
        latent: TensorType, 
        action: TensorType, 
        task_encoding: TensorType
    ) -> TensorType:
        """Predict next latent state.
        
        Args:
            latent: Current latent state [batch_size, latent_dim]
            action: Action taken [batch_size, action_dim]
            task_encoding: Task encoding [batch_size, task_encoding_dim]
            
        Returns:
            Next latent state [batch_size, latent_dim]
        """
        # Concatenate inputs
        x = torch.cat([latent, action, task_encoding], dim=-1)
        
        # Predict next latent state
        next_latent = self.dynamics(x)
        
        return next_latent


class WorldModelReward(base_component.Component):
    """Reward model that predicts reward from latent state and action.
    
    Following TD-MPC2 architecture: r̂ = R(z, a, e)
    """
    
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        task_encoding_dim: int,
        hidden_dim: int = 512,
        reward_bins: int = 101,  # Discretized reward prediction as in TD-MPC2
    ):
        """Initialize the reward model.
        
        Args:
            latent_dim: Dimension of the latent representation
            action_dim: Dimension of the action space  
            task_encoding_dim: Dimension of the task encoding
            hidden_dim: Hidden layer dimension
            reward_bins: Number of bins for discretized reward prediction
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.task_encoding_dim = task_encoding_dim
        self.reward_bins = reward_bins
        
        # Input is latent + action + task encoding
        input_dim = latent_dim + action_dim + task_encoding_dim
        
        # 3-layer MLP with final linear layer (no activation)
        self.reward_net = nn.Sequential(
            NormedLinear(input_dim, hidden_dim, act="Mish"),
            NormedLinear(hidden_dim, hidden_dim, act="Mish"),
            nn.Linear(hidden_dim, reward_bins)
        )
    
    def forward(
        self, 
        latent: TensorType, 
        action: TensorType, 
        task_encoding: TensorType
    ) -> TensorType:
        """Predict reward from latent state and action.
        
        Args:
            latent: Latent state [batch_size, latent_dim]
            action: Action taken [batch_size, action_dim]
            task_encoding: Task encoding [batch_size, task_encoding_dim]
            
        Returns:
            Reward logits [batch_size, reward_bins]
        """
        # Concatenate inputs
        x = torch.cat([latent, action, task_encoding], dim=-1)
        
        # Predict reward
        reward_logits = self.reward_net(x)
        
        return reward_logits
    
    def predict_reward(
        self, 
        latent: TensorType, 
        action: TensorType, 
        task_encoding: TensorType,
        reward_bounds: Tuple[float, float] = (-10.0, 10.0)
    ) -> TensorType:
        """Predict continuous reward value.
        
        Args:
            latent: Latent state [batch_size, latent_dim]
            action: Action taken [batch_size, action_dim]
            task_encoding: Task encoding [batch_size, task_encoding_dim]
            reward_bounds: (min_reward, max_reward) for discretization
            
        Returns:
            Predicted reward values [batch_size, 1]
        """
        reward_logits = self.forward(latent, action, task_encoding)
        
        # Convert logits to probabilities
        reward_probs = F.softmax(reward_logits, dim=-1)
        
        # Create reward bins
        min_reward, max_reward = reward_bounds
        reward_bins = torch.linspace(
            min_reward, max_reward, self.reward_bins, 
            device=reward_logits.device
        )
        
        # Compute expected reward
        reward_pred = torch.sum(reward_probs * reward_bins, dim=-1, keepdim=True)
        
        return reward_pred


class WorldModel(base_component.Component):
    """Complete world model with encoder, dynamics, and reward components."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        task_encoding_dim: int,
        latent_dim: int = 512,
        encoder_layers: int = 2,
        encoder_hidden_dim: int = 256,
        dynamics_hidden_dim: int = 512,
        reward_hidden_dim: int = 512,
        reward_bins: int = 101,
    ):
        """Initialize the complete world model.
        
        Args:
            state_dim: Dimension of state observations
            action_dim: Dimension of action space
            task_encoding_dim: Dimension of task encoding (CLS token)
            latent_dim: Dimension of latent representation
            encoder_layers: Number of encoder layers
            encoder_hidden_dim: Encoder hidden dimension
            dynamics_hidden_dim: Dynamics hidden dimension  
            reward_hidden_dim: Reward model hidden dimension
            reward_bins: Number of reward discretization bins
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.task_encoding_dim = task_encoding_dim
        self.latent_dim = latent_dim
        
        # Initialize components
        self.encoder = WorldModelEncoder(
            state_dim=state_dim,
            task_encoding_dim=task_encoding_dim,
            latent_dim=latent_dim,
            num_layers=encoder_layers,
            hidden_dim=encoder_hidden_dim,
        )
        
        self.dynamics = WorldModelDynamics(
            latent_dim=latent_dim,
            action_dim=action_dim,
            task_encoding_dim=task_encoding_dim,
            hidden_dim=dynamics_hidden_dim,
        )
        
        self.reward = WorldModelReward(
            latent_dim=latent_dim,
            action_dim=action_dim,
            task_encoding_dim=task_encoding_dim,
            hidden_dim=reward_hidden_dim,
            reward_bins=reward_bins,
        )
    
    def encode(self, state: TensorType, task_encoding: TensorType) -> TensorType:
        """Encode state to latent representation."""
        return self.encoder(state, task_encoding)
    
    def predict_next_latent(
        self, 
        latent: TensorType, 
        action: TensorType, 
        task_encoding: TensorType
    ) -> TensorType:
        """Predict next latent state."""
        return self.dynamics(latent, action, task_encoding)
    
    def predict_reward(
        self, 
        latent: TensorType, 
        action: TensorType, 
        task_encoding: TensorType,
        reward_bounds: Tuple[float, float] = (-10.0, 10.0)
    ) -> TensorType:
        """Predict reward."""
        return self.reward.predict_reward(latent, action, task_encoding, reward_bounds)
    
    def forward(
        self,
        state: TensorType,
        action: TensorType,
        task_encoding: TensorType,
        reward_bounds: Tuple[float, float] = (-10.0, 10.0)
    ) -> Tuple[TensorType, TensorType, TensorType]:
        """Full forward pass of world model.
        
        Args:
            state: Current state [batch_size, state_dim]
            action: Action taken [batch_size, action_dim]
            task_encoding: Task encoding [batch_size, task_encoding_dim]
            reward_bounds: Reward discretization bounds
            
        Returns:
            (latent_state, next_latent_state, predicted_reward)
        """
        # Encode current state
        latent = self.encode(state, task_encoding)
        
        # Predict next latent state
        next_latent = self.predict_next_latent(latent, action, task_encoding)
        
        # Predict reward
        reward_pred = self.predict_reward(latent, action, task_encoding, reward_bounds)
        
        return latent, next_latent, reward_pred
    
    def compute_loss(
        self,
        state: TensorType,
        action: TensorType,
        next_state: TensorType,
        reward: TensorType,
        task_encoding: TensorType,
        reward_bounds: Tuple[float, float] = (-10.0, 10.0)
    ) -> Tuple[TensorType, TensorType, TensorType]:
        """Compute world model training losses.
        
        Args:
            state: Current state [batch_size, state_dim]
            action: Action taken [batch_size, action_dim]
            next_state: Next state [batch_size, state_dim]
            reward: Actual reward [batch_size, 1]
            task_encoding: Task encoding [batch_size, task_encoding_dim]
            reward_bounds: Reward discretization bounds
            
        Returns:
            (dynamics_loss, reward_loss, total_loss)
        """
        # Encode current and next states
        latent = self.encode(state, task_encoding)
        next_latent_target = self.encode(next_state, task_encoding)
        
        # Predict next latent state
        next_latent_pred = self.predict_next_latent(latent, action, task_encoding)
        
        # Dynamics loss (MSE between predicted and target next latent)
        dynamics_loss = F.mse_loss(next_latent_pred, next_latent_target.detach())
        
        # Reward loss (cross-entropy for discretized rewards)
        reward_logits = self.reward(latent, action, task_encoding)
        
        # Discretize target rewards
        min_reward, max_reward = reward_bounds
        reward_bins = torch.linspace(
            min_reward, max_reward, self.reward.reward_bins,
            device=reward.device
        )
        
        # Find closest bin for each reward
        reward_targets = torch.clamp(reward, min_reward, max_reward)
        bin_indices = torch.round(
            (reward_targets - min_reward) / (max_reward - min_reward) * (self.reward.reward_bins - 1)
        ).long().squeeze(-1)
        
        reward_loss = F.cross_entropy(reward_logits, bin_indices)
        
        # Total loss
        total_loss = dynamics_loss + reward_loss
        
        return dynamics_loss, reward_loss, total_loss