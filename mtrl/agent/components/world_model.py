from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from mtrl.agent.components import base as base_component
from mtrl.utils.types import TensorType

# 这些函数来自你项目的 wm_math.py（与 TD-MPC2 等价）
from .wm_math import DRegCfg, soft_ce, two_hot_inv, symlog


class SimNorm(nn.Module):
    """Simplicial normalization: reshape [..., K*G] -> [..., K, G], softmax over G."""
    def __init__(self, simnorm_dim: int):
        super().__init__()
        self.dim = simnorm_dim

    def forward(self, x):
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shp)

    def __repr__(self):
        return f"SimNorm(dim={self.dim})"


class NormedLinear(nn.Module):
    """Linear + (optional Dropout) + LayerNorm + Activation (default Mish)."""
    def __init__(self, in_features: int, out_features: int, dropout: float = 0., act=None):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.ln = nn.LayerNorm(out_features)
        if act is None:
            act = nn.Mish(inplace=False)
        self.act = act
        self.dropout = nn.Dropout(dropout, inplace=False) if dropout else None

    def forward(self, x):
        x = self.linear(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.ln(x)
        return self.act(x)

    def __repr__(self):
        repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
        return f"NormedLinear(in_features={self.linear.in_features}, " \
               f"out_features={self.linear.out_features}, " \
               f"bias={self.linear.bias is not None}{repr_dropout}, " \
               f"act={self.act.__class__.__name__})"


def mlp(in_dim: int, mlp_dims: List[int], out_dim: int, act=None, dropout: float = 0.):
    """TD-MPC2 风格的 MLP 块。"""
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]
    dims = [in_dim] + mlp_dims + [out_dim]
    layers = nn.ModuleList()
    for i in range(len(dims) - 2):
        layers.append(NormedLinear(dims[i], dims[i+1], dropout=dropout*(i==0)))
    layers.append(NormedLinear(dims[-2], dims[-1], act=act) if act else nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


class WorldModelEncoder(base_component.Component):
    """z = h([s, e])"""
    def __init__(self, state_dim: int, task_encoding_dim: int, latent_dim: int,
                 num_enc_layers: int = 2, enc_dim: int = 256, simnorm_dim: int = 8):
        super().__init__()
        assert latent_dim % simnorm_dim == 0, \
            f"latent_dim ({latent_dim}) must be divisible by simnorm_dim ({simnorm_dim})"
        input_dim = state_dim + task_encoding_dim
        hidden_dims = [enc_dim] * max(num_enc_layers - 1, 1)
        self.encoder = mlp(input_dim, hidden_dims, latent_dim, act=SimNorm(simnorm_dim))

    def forward(self, state: TensorType, task_encoding: TensorType) -> TensorType:
        x = torch.cat([state, task_encoding], dim=-1)
        return self.encoder(x)


class WorldModelDynamics(base_component.Component):
    """z' = d([z, a, e])"""
    def __init__(self, latent_dim: int, action_dim: int, task_encoding_dim: int,
                 mlp_dim: int = 512, simnorm_dim: int = 8):
        super().__init__()
        assert latent_dim % simnorm_dim == 0, \
            f"latent_dim ({latent_dim}) must be divisible by simnorm_dim ({simnorm_dim})"
        input_dim = latent_dim + action_dim + task_encoding_dim
        self.dynamics = mlp(input_dim, [mlp_dim, mlp_dim], latent_dim, act=SimNorm(simnorm_dim))

    def forward(self, latent: TensorType, action: TensorType, task_encoding: TensorType) -> TensorType:
        x = torch.cat([latent, action, task_encoding], dim=-1)
        return self.dynamics(x)


class WorldModelReward(base_component.Component):
    """r̂ = R([z, a, e]) → logits (two-hot) 或标量（num_bins=1）"""
    def __init__(self, latent_dim: int, action_dim: int, task_encoding_dim: int,
                 mlp_dim: int = 512, reward_bins: int = 101):
        super().__init__()
        input_dim = latent_dim + action_dim + task_encoding_dim
        self.reward_bins = reward_bins
        self.reward_net = mlp(input_dim, [mlp_dim, mlp_dim], max(reward_bins, 1))

    def forward(self, latent: TensorType, action: TensorType, task_encoding: TensorType) -> TensorType:
        x = torch.cat([latent, action, task_encoding], dim=-1)
        return self.reward_net(x)


class WorldModel(base_component.Component):
    """
    TD-MPC2 风格的隐式世界模型（简化版）：
      - z = h([s,e])
      - z' = d([z,a,e])
      - r̂ = R([z,a,e]) （two-hot 离散回归）
    """
    def __init__(self, state_dim: int, action_dim: int, task_encoding_dim: int,
                 latent_dim: int = 512, num_enc_layers: int = 2, enc_dim: int = 256,
                 mlp_dim: int = 512, reward_bins: int = 101, simnorm_dim: int = 8,
                 reward_bounds: Tuple[float, float] = (-10.0, 10.0)):
        super().__init__()
        self.latent_dim = latent_dim
        self.reward_bins = reward_bins

        # modules
        self.encoder = WorldModelEncoder(state_dim, task_encoding_dim, latent_dim,
                                         num_enc_layers, enc_dim, simnorm_dim)
        self.dynamics = WorldModelDynamics(latent_dim, action_dim, task_encoding_dim,
                                           mlp_dim, simnorm_dim)
        self.reward_head = WorldModelReward(latent_dim, action_dim, task_encoding_dim,
                                            mlp_dim, reward_bins)

        # dreg（two-hot 离散回归配置）
        if self.reward_bins > 1:
            rmin, rmax = reward_bounds
            vmin = symlog(torch.tensor([rmin])).item()
            vmax = symlog(torch.tensor([rmax])).item()
            bin_size = (vmax - vmin) / (self.reward_bins - 1)
            self.dreg_cfg = DRegCfg(num_bins=self.reward_bins, vmin=vmin, vmax=vmax, bin_size=bin_size)
        else:
            self.dreg_cfg = None

    # ---------- public API ----------
    def encode(self, state: TensorType, task_encoding: TensorType) -> TensorType:
        return self.encoder(state, task_encoding)

    def predict_next_latent(self, latent: TensorType, action: TensorType, task_encoding: TensorType) -> TensorType:
        return self.dynamics(latent, action, task_encoding)

    def reward_logits(self, latent: TensorType, action: TensorType, task_encoding: TensorType) -> TensorType:
        return self.reward_head(latent, action, task_encoding)

    def predict_reward(self, latent: TensorType, action: TensorType, task_encoding: TensorType) -> TensorType:
        """返回连续标量奖励（[B,1]），two-hot 反变换。"""
        logits = self.reward_logits(latent, action, task_encoding)
        if self.reward_bins <= 1:
            return logits  # [B,1]
        return two_hot_inv(logits, self.dreg_cfg)  # [B,1]

    def forward(self, state: TensorType, action: TensorType, task_encoding: TensorType):
        z = self.encode(state, task_encoding)
        z_next = self.predict_next_latent(z, action, task_encoding)
        r_hat = self.predict_reward(z, action, task_encoding)
        return z, z_next, r_hat

    @torch.no_grad()
    def latent_rollout(self, z0: TensorType, actions: TensorType, task_encoding: TensorType) -> TensorType:
        """可选：多步 latent rollout（诊断用）"""
        T = actions.shape[0]
        B = z0.shape[0]
        z = z0
        traj = [z0]
        for t in range(T):
            z = self.predict_next_latent(z, actions[t].view(B, -1), task_encoding)
            traj.append(z)
        return torch.stack(traj, dim=0)  # [T+1, B, latent_dim]

    # ---------- loss ----------
    def compute_loss(self, state: TensorType, action: TensorType, next_state: TensorType,
                     reward: TensorType, task_encoding: TensorType):
        """返回 (dyn_loss, rew_loss, total_loss)"""
        # 1) encode
        z = self.encode(state, task_encoding)
        with torch.no_grad():
            z_next_tgt = self.encode(next_state, task_encoding)

        # 2) dynamics 一致性
        z_next_pred = self.predict_next_latent(z, action, task_encoding)
        dyn_loss = F.mse_loss(z_next_pred, z_next_tgt)

        # 3) reward 两邻居 soft CE（或 num_bins=1 下的 MSE）
        logits = self.reward_logits(z, action, task_encoding)
        if self.reward_bins > 1:
            r = reward if reward.ndim == 2 else reward.unsqueeze(-1)
            rew_loss = soft_ce(logits, r, self.dreg_cfg).mean()
        else:
            r = reward if reward.ndim == 2 else reward.unsqueeze(-1)
            rew_loss = F.mse_loss(logits, r)

        total = dyn_loss + rew_loss
        return dyn_loss, rew_loss, total