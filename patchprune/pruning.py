"""Patch importance scores and selection rules for inference-time pruning.

Every score maps a batch of series to one number per patch, higher meaning
"more worth keeping":

* ``variance``  - variance of the instance-normalised patch; flat patches score low.
* ``entropy``   - mean surprisal -log p(x_t+1 | x_t) of the patch under a
  first-order Markov model over quantised values, fitted on the training split.
  This is a lightweight stand-in for the learned entropy model in EntroPE:
  predictable patches carry little information and score low.
* ``attention`` - attention each patch receives in the preceding encoder layer,
  averaged over heads and queries (as in EViT-style token pruning).
* ``random``    - uniform random scores (baseline).
* ``recency``   - patch position, i.e. keep the most recent patches. Equivalent
  to shortening the look-back window (baseline).

Two selection rules are supported: a fixed keep ratio (top-k per series) and an
adaptive rule that keeps, per series, the smallest set of patches holding a
fraction ``top_p`` of the (min-shifted) score mass.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

METHODS = ("variance", "entropy", "attention", "random", "recency")
FILL_MODES = ("skip", "mean", "zero")


class MarkovEntropyModel:
    """First-order Markov model over quantile-binned, instance-normalised values."""

    def __init__(self, edges: torch.Tensor, log_trans: torch.Tensor):
        self.edges = edges
        self.log_trans = log_trans

    @classmethod
    def fit(cls, train: np.ndarray, seq_len: int, n_bins: int = 16, max_windows: int = 4000,
            seed: int = 0):
        rng = np.random.default_rng(seed)
        starts = np.arange(0, len(train) - seq_len + 1)
        starts = rng.choice(starts, size=min(max_windows, len(starts)), replace=False)
        windows = np.stack([train[s:s + seq_len] for s in starts])          # [W, L, C]
        mu = windows.mean(1, keepdims=True)
        sd = np.sqrt(windows.var(1, keepdims=True) + 1e-5)
        z = torch.from_numpy(((windows - mu) / sd).transpose(0, 2, 1).copy())  # [W, C, L]

        qs = torch.linspace(0, 1, n_bins + 1)[1:-1]
        edges = torch.quantile(z.flatten()[:2_000_000], qs).contiguous()
        bins = torch.bucketize(z, edges)
        pairs = bins[..., :-1] * n_bins + bins[..., 1:]
        counts = torch.bincount(pairs.flatten(), minlength=n_bins * n_bins).float()
        counts = counts.view(n_bins, n_bins) + 1.0                          # Laplace smoothing
        log_trans = torch.log(counts / counts.sum(1, keepdim=True))
        return cls(edges, log_trans)

    def to(self, device):
        return MarkovEntropyModel(self.edges.to(device), self.log_trans.to(device))

    def surprisal(self, patches: torch.Tensor) -> torch.Tensor:
        """[B, N, P] -> mean surprisal per patch [B, N] (nats)."""
        bins = torch.bucketize(patches.contiguous(), self.edges)
        logp = self.log_trans[bins[..., :-1], bins[..., 1:]]
        return -logp.mean(-1)

    def state_dict(self):
        return {"edges": self.edges, "log_trans": self.log_trans}

    @classmethod
    def from_state_dict(cls, state):
        return cls(state["edges"], state["log_trans"])


@dataclass
class PruningPlan:
    method: str
    keep_ratio: float = 1.0
    top_p: Optional[float] = None
    prune_layer: Optional[int] = None
    fill: str = "skip"
    min_keep: int = 1
    entropy_model: Optional[MarkovEntropyModel] = None
    seed: int = 0
    _gen: Optional[torch.Generator] = field(default=None, repr=False)

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"unknown method {self.method!r}; choose from {METHODS}")
        if self.fill not in FILL_MODES:
            raise ValueError(f"unknown fill {self.fill!r}; choose from {FILL_MODES}")
        if self.prune_layer is None:
            self.prune_layer = 1 if self.method == "attention" else 0
        if self.method == "attention" and self.prune_layer < 1:
            raise ValueError("attention scores need at least one unpruned layer (prune_layer >= 1)")
        if self.method == "entropy" and self.entropy_model is None:
            raise ValueError("entropy scoring needs a fitted MarkovEntropyModel")
        if self.method in ("random", "recency") and self.top_p is not None:
            raise ValueError(f"adaptive selection is not defined for {self.method!r}")
        self._gen = torch.Generator().manual_seed(self.seed)

    # ------------------------------------------------------------------ scores
    def score(self, patches: torch.Tensor, attn: Optional[torch.Tensor]) -> torch.Tensor:
        bc, n, _ = patches.shape
        if self.method == "variance":
            return patches.var(-1, unbiased=False)
        if self.method == "entropy":
            return self.entropy_model.surprisal(patches)
        if self.method == "attention":
            return attn.mean(dim=(1, 2))  # attention received, averaged over heads and queries
        if self.method == "random":
            return torch.rand(bc, n, generator=self._gen).to(patches.device)
        return torch.arange(n, device=patches.device, dtype=patches.dtype).expand(bc, n)

    # --------------------------------------------------------------- selection
    def select(self, scores: torch.Tensor):
        """Return (indices [B, K], valid [B, K]) of the patches to keep."""
        bc, n = scores.shape
        if self.top_p is None:
            k = min(n, max(self.min_keep, int(round(self.keep_ratio * n))))
            idx = scores.topk(k, dim=1).indices.sort(dim=1).values
            return idx, torch.ones_like(idx, dtype=torch.bool)

        s = scores - scores.min(dim=1, keepdim=True).values
        s = s + 1e-12 * s.amax(dim=1, keepdim=True).clamp_min(1e-12)
        sorted_s, order = s.sort(dim=1, descending=True)
        mass = sorted_s.cumsum(1) / sorted_s.sum(1, keepdim=True)
        count = ((mass < self.top_p).sum(1) + 1).clamp(self.min_keep, n)
        kmax = int(count.max())
        idx = order[:, :kmax]
        valid = torch.arange(kmax, device=scores.device)[None, :] < count[:, None]
        return idx, valid

    @property
    def label(self) -> str:
        rule = f"top_p={self.top_p}" if self.top_p is not None else f"keep={self.keep_ratio:.2f}"
        return f"{self.method}[{rule}, layer={self.prune_layer}, fill={self.fill}]"
