import torch
import torch.nn.functional as F

@torch.jit.script
def symlog(x: torch.Tensor) -> torch.Tensor:
    # sign(x) * log(1 + |x|)
    return torch.sign(x) * torch.log1p(torch.abs(x))

@torch.jit.script
def symexp(x: torch.Tensor) -> torch.Tensor:
    # sign(x) * (exp(|x|) - 1)
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)

@torch.jit.script
def log_std(x: torch.Tensor, low: torch.Tensor, dif: torch.Tensor) -> torch.Tensor:
    # low + 0.5 * dif * (tanh(x) + 1)
    return low + 0.5 * dif * (torch.tanh(x) + 1.0)

@torch.jit.script
def gaussian_logprob(eps: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    # sum_i [ -0.5 eps_i^2 - log_std_i - 0.5*log(2*pi) ]
    residual = -0.5 * eps.pow(2) - log_std
    log_prob = residual - 0.9189385332046727  # 0.5*ln(2*pi)
    return log_prob.sum(-1, keepdim=True)

@torch.jit.script
def squash(mu: torch.Tensor, pi: torch.Tensor, log_pi: torch.Tensor):
    mu = torch.tanh(mu)
    pi = torch.tanh(pi)
    # log det |Jacobian| = \sum_i log(1 - tanh(pi_i)^2)
    log_pi = log_pi - torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi

class DRegCfg:
    """离散回归(two-hot)的配置，等价 TD-MPC2 的 cfg 里 num_bins/vmin/vmax/bin_size 那部分。"""
    __slots__ = ("num_bins", "vmin", "vmax", "bin_size")
    def __init__(self, num_bins: int, vmin: float, vmax: float):
        self.num_bins = int(num_bins)
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.bin_size = (self.vmax - self.vmin) / max(self.num_bins - 1, 1)

@torch.jit.script
def two_hot(x: torch.Tensor, num_bins: int, vmin: float, vmax: float, bin_size: float) -> torch.Tensor:
    """把标量回归目标 x(形状 [B,1] 或 [B]) 映射成 soft two-hot 向量 [B, num_bins]。"""
    if num_bins == 0:
        return x
    if num_bins == 1:
        return symlog(x)
    x = torch.clamp(symlog(x).squeeze(-1), vmin, vmax)
    # 落在哪个 bin
    f = (x - vmin) / bin_size
    bin_idx = torch.floor(f)
    bin_off = (f - bin_idx).unsqueeze(-1)          # [B,1]
    bin_idx = bin_idx.long().unsqueeze(-1)         # [B,1]
    B = x.shape[0]
    oh = torch.zeros(B, num_bins, device=x.device, dtype=x.dtype)
    oh.scatter_(1, bin_idx.clamp(min=0, max=num_bins-1), 1.0 - bin_off)
    oh.scatter_(1, (bin_idx + 1) % num_bins, bin_off)
    return oh

@torch.jit.script
def two_hot_inv(x: torch.Tensor, num_bins: int, vmin: float, vmax: float) -> torch.Tensor:
    """把 two-hot 概率分布反推回标量（对数值域再做 symexp）。"""
    if num_bins == 0:
        return x
    if num_bins == 1:
        return symexp(x)
    bins = torch.linspace(vmin, vmax, num_bins, device=x.device, dtype=x.dtype)
    x = F.softmax(x, dim=-1)
    val = torch.sum(x * bins, dim=-1, keepdim=True)
    return symexp(val)

def soft_ce(pred_logits: torch.Tensor, target_scalar: torch.Tensor, dreg: DRegCfg) -> torch.Tensor:
    """离散回归的 soft cross-entropy（TD-MPC2 同款）。"""
    # pred_logits: [B, num_bins]
    # target_scalar: [B, 1]（实数）
    logp = F.log_softmax(pred_logits, dim=-1)
    tgt = two_hot(target_scalar, dreg.num_bins, dreg.vmin, dreg.vmax, dreg.bin_size)  # [B, num_bins]
    return -(tgt * logp).sum(-1, keepdim=True)  # [B,1]