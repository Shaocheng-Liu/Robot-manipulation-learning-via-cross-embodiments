from typing import List, Tuple, Optional
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mtrl.agent.components import base as base_component
from mtrl.utils.types import TensorType

from .wm_math import DRegCfg, soft_ce, two_hot_inv, symlog

try:
    Mish = nn.Mish  # PyTorch 里有就直接用
except AttributeError:
    class Mish(nn.Module):
        def __init__(self, inplace: bool = False):
            super().__init__()
            self.inplace = inplace  # 保留接口以兼容调用
        def forward(self, x):
            # Mish(x) = x * tanh(softplus(x))
            return x * torch.tanh(F.softplus(x))


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
            act = Mish(inplace=False)
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
        if not hasattr(self, "_dbg_printed"):
            print(f"[WM-ENC/DEBUG] state{tuple(state.shape)} + task{tuple(task_encoding.shape)} -> x{tuple(x.shape)}")
            assert x.shape[-1] == self.expected_in, \
                f"encoder input {x.shape[-1]} != expected_in {self.expected_in}"
            self._dbg_printed = True
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

        self.expected_in = state_dim + task_encoding_dim  # 记录预期输入
        # 找到 encoder 里的第一层 Linear（不管是 nn.Linear 还是包在 NormedLinear 里）
        first_linear_in = None
        for m in self.encoder.modules():
            if isinstance(m, nn.Linear):
                first_linear_in = m.in_features
                break
        print(f"[WM-ENC/DEBUG] expect_in={self.expected_in}, first_linear_in={first_linear_in}, latent_dim={latent_dim}")

        # dreg（two-hot 离散回归配置）
        if self.reward_bins > 1:
            rmin, rmax = reward_bounds
            vmin = symlog(torch.tensor([rmin])).item()
            vmax = symlog(torch.tensor([rmax])).item()
            self.dreg_cfg = DRegCfg(num_bins=self.reward_bins, vmin=vmin, vmax=vmax)
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
        self.dreg_cfg = DRegCfg(num_bins=self.reward_bins, vmin=vmin, vmax=vmax)

    @staticmethod
    def estimate_reward_bounds_from_buffer(
        replay_buffer,
        max_samples: int = 100_000,
        q_low: float = 0.01,
        q_high: float = 0.99,
        print_stats: bool = True,
    ):
        """
        仅适配 TransformerReplayBuffer：
        - 直接从 replay_buffer.rewards 读取（形状 [E, T, 1]）
        - 只使用有效区间（full ? capacity : idx）
        - 先 symlog，再在 symlog 轴上取分位数作为 (vmin, vmax)
        返回：vmin, vmax（float，位于 symlog 轴）
        """
        # 有效样本数
        N_valid = replay_buffer.capacity if getattr(replay_buffer, "full", False) \
                else getattr(replay_buffer, "idx", 0)
        if N_valid <= 0:
            raise RuntimeError("Replay buffer 里还没有有效的 reward 样本。")

        # 拉平拿前 N_valid 个（避免把未写入的尾部 0 算进去）
        r_np = replay_buffer.rewards.reshape(-1)[:N_valid].astype(np.float32)
        if r_np.size == 0:
            raise RuntimeError("从 replay_buffer.rewards 取到的 reward 为空。")

        # 如样本太多，下采样（固定随机种子保证可复现）
        if r_np.size > max_samples:
            rng = np.random.RandomState(0)
            idx = rng.choice(r_np.size, size=max_samples, replace=False)
            r_np = r_np[idx]

        # 到 symlog 轴
        r_symlog = symlog(torch.from_numpy(r_np)).numpy()

        # 分位数做边界（更抗离群）
        vmin = float(np.quantile(r_symlog, q_low))
        vmax = float(np.quantile(r_symlog, q_high))

        # 防止数值重合
        if np.isclose(vmin, vmax):
            eps = 1e-3
            vmin, vmax = vmin - eps, vmax + eps

        if print_stats:
            print(
                "[Reward bounds] "
                f"raw_min={r_np.min():.4f}, raw_max={r_np.max():.4f}, "
                f"symlog_q{int(q_low*100)}={vmin:.4f}, symlog_q{int(q_high*100)}={vmax:.4f}, "
                f"N={r_np.size}"
            )

        return vmin, vmax