from typing import List, Tuple, Optional
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mtrl.agent.components import base as base_component
from mtrl.utils.types import TensorType

from .wm_math import DRegCfg, soft_ce, two_hot_inv, symlog


# ----------------- small blocks -----------------
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
    """Linear (+ optional Dropout) + LayerNorm + Activation (default Mish)."""
    def __init__(self, in_features: int, out_features: int, dropout: float = 0., act=None):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.ln = nn.LayerNorm(out_features)
        if act is None:
            act = nn.Mish(inplace=False)
        self.act = act
        self.dropout = nn.Dropout(dropout, inplace=False) if dropout and dropout > 1e-8 else None

    def forward(self, x):
        x = self.linear(x)
        if self.dropout is not None:
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
    """TD-MPC2 风格的 MLP 块（首层可选 dropout）。"""
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]
    dims = [in_dim] + mlp_dims + [out_dim]
    layers = nn.ModuleList()
    for i in range(len(dims) - 2):
        # 只在第一层放一点 dropout（可配），稳定些
        layers.append(NormedLinear(dims[i], dims[i+1], dropout=dropout if i == 0 else 0.0))
    layers.append(NormedLinear(dims[-2], dims[-1], act=act) if act else nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


# ----------------- submodules -----------------
class WorldModelEncoder(base_component.Component):
    """z = h([s, e])"""
    def __init__(self, state_dim: int, task_encoding_dim: int, latent_dim: int,
                 num_enc_layers: int = 2, enc_dim: int = 256, simnorm_dim: int = 8, dropout: float = 0.):
        super().__init__()
        assert latent_dim % simnorm_dim == 0, \
            f"latent_dim ({latent_dim}) must be divisible by simnorm_dim ({simnorm_dim})"
        input_dim = state_dim + task_encoding_dim
        hidden_dims = [enc_dim] * max(num_enc_layers - 1, 1)
        self.encoder = mlp(input_dim, hidden_dims, latent_dim, act=SimNorm(simnorm_dim), dropout=dropout)

    def forward(self, state: TensorType, task_encoding: TensorType) -> TensorType:
        x = torch.cat([state, task_encoding], dim=-1)
        return self.encoder(x)


class WorldModelDynamics(base_component.Component):
    """z' = d([z, a, e])"""
    def __init__(self, latent_dim: int, action_dim: int, task_encoding_dim: int,
                 mlp_dim: int = 512, simnorm_dim: int = 8, dropout: float = 0.):
        super().__init__()
        assert latent_dim % simnorm_dim == 0, \
            f"latent_dim ({latent_dim}) must be divisible by simnorm_dim ({simnorm_dim})"
        input_dim = latent_dim + action_dim + task_encoding_dim
        self.dynamics = mlp(input_dim, [mlp_dim, mlp_dim], latent_dim, act=SimNorm(simnorm_dim), dropout=dropout)

    def forward(self, latent: TensorType, action: TensorType, task_encoding: TensorType) -> TensorType:
        x = torch.cat([latent, action, task_encoding], dim=-1)
        return self.dynamics(x)


class WorldModelReward(base_component.Component):
    """r̂ = R([z, a, e]) → logits (two-hot) 或标量（reward_bins=1）"""
    def __init__(self, latent_dim: int, action_dim: int, task_encoding_dim: int,
                 mlp_dim: int = 512, reward_bins: int = 101, dropout: float = 0.):
        super().__init__()
        input_dim = latent_dim + action_dim + task_encoding_dim
        self.reward_bins = reward_bins
        self.reward_net = mlp(input_dim, [mlp_dim, mlp_dim], max(reward_bins, 1), dropout=dropout)

    def forward(self, latent: TensorType, action: TensorType, task_encoding: TensorType) -> TensorType:
        x = torch.cat([latent, action, task_encoding], dim=-1)
        return self.reward_net(x)


# ----------------- world model -----------------
class WorldModel(base_component.Component):
    """
    TD-MPC2 风格的隐式世界模型（轻改版）：
      - z = h([s,e])
      - z' = d([z,a,e])
      - r̂ = R([z,a,e]) （two-hot 离散回归；reward_bins=1 时退化为标量回归）
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        task_encoding_dim: int,
        # ✅ 统一从 cfg 读取超参；以下键名按你之前的写法给出默认值
        latent_dim: int = 512,
        num_enc_layers: int = 2,
        enc_dim: int = 256,
        mlp_dim: int = 512,
        reward_bins: int = 101,
        simnorm_dim: int = 8,
        reward_bounds: Tuple[float, float] = (-10.0, 10.0),
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.reward_bins = reward_bins
        self.task_encoding_dim = task_encoding_dim

        # modules
        self.encoder = WorldModelEncoder(
            state_dim=state_dim,
            task_encoding_dim=task_encoding_dim,
            latent_dim=latent_dim,
            num_enc_layers=num_enc_layers,
            enc_dim=enc_dim,
            simnorm_dim=simnorm_dim,
            dropout=dropout,
        )
        self.dynamics = WorldModelDynamics(
            latent_dim=latent_dim,
            action_dim=action_dim,
            task_encoding_dim=task_encoding_dim,
            mlp_dim=mlp_dim,
            simnorm_dim=simnorm_dim,
            dropout=dropout,
        )
        self.reward_head = WorldModelReward(
            latent_dim=latent_dim,
            action_dim=action_dim,
            task_encoding_dim=task_encoding_dim,
            mlp_dim=mlp_dim,
            reward_bins=reward_bins,
            dropout=dropout,
        )

        # dreg（two-hot 离散回归配置）
        if self.reward_bins > 1:
            rmin, rmax = reward_bounds
            vmin = symlog(torch.tensor([rmin])).item()
            vmax = symlog(torch.tensor([rmax])).item()
            bin_size = (vmax - vmin) / max(self.reward_bins - 1, 1)
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
            # 标量回归（MSE）
            return logits if logits.ndim == 2 else logits.unsqueeze(-1)
        return two_hot_inv(logits, self.dreg_cfg)  # [B,1]

    def forward(self, state: TensorType, action: TensorType, task_encoding: TensorType):
        z = self.encode(state, task_encoding)
        z_next = self.predict_next_latent(z, action, task_encoding)
        r_hat = self.predict_reward(z, action, task_encoding)
        return z, z_next, r_hat

    @torch.no_grad()
    def latent_rollout(self, z0: TensorType, actions: TensorType, task_encoding: TensorType) -> TensorType:
        """多步 latent rollout（诊断用）"""
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

        # 3) reward two-hot（或 reward_bins=1 的 MSE）
        logits = self.reward_logits(z, action, task_encoding)
        if self.reward_bins > 1:
            # 确保形状为 [B,1]
            r = reward if reward.ndim == 2 else reward.unsqueeze(-1)
            rew_loss = soft_ce(logits, r, self.dreg_cfg).mean()
        else:
            r = reward if reward.ndim == 2 else reward.unsqueeze(-1)
            rew_loss = F.mse_loss(logits, r)

        total = dyn_loss + rew_loss
        return dyn_loss, rew_loss, total

    # ---------- reward-bounds 工具 ----------
    def set_reward_bounds(self, vmin: float, vmax: float):
        """在线更新 two-hot 的 symlog(bins) 边界；仅在 reward_bins>1 时生效。"""
        if self.reward_bins <= 1:
            self.dreg_cfg = None
            return
        # 避免边界相等
        if math.isclose(vmin, vmax):
            eps = 1e-3
            vmin, vmax = vmin - eps, vmax + eps
        bin_size = (vmax - vmin) / max(self.reward_bins - 1, 1)
        self.dreg_cfg = DRegCfg(num_bins=self.reward_bins, vmin=vmin, vmax=vmax, bin_size=bin_size)

    @staticmethod
    def estimate_reward_bounds_from_buffer(
        replay_buffer,
        max_samples: int = 100_000,
        q_low: float = 0.01,
        q_high: float = 0.99,
        print_stats: bool = True,
    ) -> Tuple[float, float]:
        """
        从 buffer 里估计 symlog(reward) 的分位数，用于 two-hot 的 (vmin, vmax)。
        兼容两类访问方式：
          1) 直接有 numpy 数组属性：buffer.rewards 形如 [E, T, 1] 或 [N, 1]
          2) 只能采样：用 buffer.sample_indices() / buffer.rewards[...] 收集
        """
        def to_flat_numpy_rewards(buf, max_n) -> np.ndarray:
            # 优先走直接数组
            if hasattr(buf, "rewards") and isinstance(buf.rewards, np.ndarray):
                r = buf.rewards
                r = r.reshape(-1)  # [E*T] or [N]
                if r.size > max_n:
                    idx = np.random.RandomState(0).choice(r.size, size=max_n, replace=False)
                    r = r[idx]
                return r.astype(np.float32)

            # 次选：通过 indices 抽样（需要 rewards 数组存在）
            if hasattr(buf, "rewards") and isinstance(buf.rewards, np.ndarray) and hasattr(buf, "sample_indices"):
                total = buf.rewards.reshape(-1).shape[0]
                n = min(total, max_n)
                idxs = buf.sample_indices() if callable(getattr(buf, "sample_indices", None)) else np.random.randint(0, total, size=n)
                # 兼容你自定义的二维索引布局 [ep, t]
                try:
                    r = buf.rewards[idxs // 400, idxs % 400]  # 你的 TransformerReplayBuffer 典型布局
                except Exception:
                    r = buf.rewards.reshape(-1)[idxs]
                return r.reshape(-1).astype(np.float32)

            # 兜底：尝试 sample()（效率最低，而且很多实现返回 torch）
            if hasattr(buf, "sample"):
                rs = []
                rng = np.random.RandomState(0)
                bs = getattr(buf, "batch_size", 1024)
                need = max_n
                while need > 0:
                    batch = buf.sample()
                    if hasattr(batch, "reward"):
                        r = batch.reward
                        if hasattr(r, "cpu"):
                            r = r.cpu().numpy()
                        rs.append(r.reshape(-1))
                        need -= r.size
                        if len(rs) > 20_000:  # 防止死循环
                            break
                    else:
                        break
                if rs:
                    r = np.concatenate(rs, axis=0)
                    if r.size > max_n:
                        idx = rng.choice(r.size, size=max_n, replace=False)
                        r = r[idx]
                    return r.astype(np.float32)

            raise RuntimeError("Cannot extract rewards from replay_buffer reliably.")

        # 取样
        rewards_np = to_flat_numpy_rewards(replay_buffer, max_samples)
        if rewards_np.size == 0:
            raise RuntimeError("No rewards found in buffer to estimate bounds.")

        # symlog 再取分位数
        r_tensor = torch.from_numpy(rewards_np)
        r_symlog = symlog(r_tensor).numpy()
        vmin = float(np.quantile(r_symlog, q_low))
        vmax = float(np.quantile(r_symlog, q_high))

        # 防“边界重合”
        if math.isclose(vmin, vmax):
            pad = 1e-3
            vmin, vmax = vmin - pad, vmax + pad

        if print_stats:
            print(f"[WorldModel] Reward stats: "
                  f"raw_min={rewards_np.min():.4f}, raw_max={rewards_np.max():.4f}, "
                  f"symlog_q{int(q_low*100)}={vmin:.4f}, symlog_q{int(q_high*100)}={vmax:.4f}, "
                  f"N={rewards_np.size}")

        return vmin, vmax