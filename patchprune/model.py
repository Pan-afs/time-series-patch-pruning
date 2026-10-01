"""PatchTST (supervised) with optional inference-time patch pruning.

The architecture follows Nie et al., "A Time Series is Worth 64 Words" (ICLR
2023): reversible instance normalisation, channel independence, overlapping
patches, a post-norm Transformer encoder with BatchNorm and residual attention,
and a flatten-linear forecasting head.

Pruning is applied only at inference. At layer ``plan.prune_layer`` the encoder
keeps a subset of patch tokens per series and runs the remaining layers on that
subset only, so attention and feed-forward cost shrink with the number of
patches kept. Dropped tokens are written back before the head (see ``fill``).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pruning import PruningPlan

MASK_VALUE = -1e9


class RevIN(nn.Module):
    def __init__(self, n_vars: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(n_vars))
        self.bias = nn.Parameter(torch.zeros(n_vars))

    def normalize(self, x):
        """Instance-normalise [B, L, C]; the affine part is applied separately."""
        self.mean = x.mean(1, keepdim=True).detach()
        self.std = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + self.eps).detach()
        return (x - self.mean) / self.std

    def denormalize(self, x):
        x = (x - self.bias) / (self.weight + self.eps * self.eps)
        return x * self.std + self.mean


class BatchNormTokens(nn.Module):
    """BatchNorm over the feature dimension of a [batch, tokens, d_model] tensor."""

    def __init__(self, d_model: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(d_model)

    def forward(self, x):
        return self.bn(x.transpose(1, 2)).transpose(1, 2)


class MultiheadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.n_heads, self.d_k = n_heads, d_model // n_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.out = nn.Sequential(nn.Linear(d_model, d_model), nn.Dropout(dropout))
        self.scale = self.d_k ** -0.5

    def forward(self, x, prev=None, key_padding_mask=None):
        b, n, _ = x.shape
        q = self.W_Q(x).view(b, n, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_K(x).view(b, n, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_V(x).view(b, n, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if prev is not None:  # residual attention (Liu et al., RealFormer)
            scores = scores + prev
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], MASK_VALUE)
        attn = F.softmax(scores, dim=-1)

        ctx = torch.matmul(attn, v).transpose(1, 2).reshape(b, n, -1)
        return self.out(ctx), attn, scores


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn = MultiheadAttention(d_model, n_heads, dropout)
        self.drop_attn = nn.Dropout(dropout)
        self.norm_attn = BatchNormTokens(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(d_ff, d_model))
        self.drop_ff = nn.Dropout(dropout)
        self.norm_ff = BatchNormTokens(d_model)

    def forward(self, x, prev=None, key_padding_mask=None):
        a, attn, scores = self.attn(x, prev, key_padding_mask)
        x = self.norm_attn(x + self.drop_attn(a))
        x = self.norm_ff(x + self.drop_ff(self.ff(x)))
        return x, attn, scores


def _gather_tokens(x, idx):
    return x.gather(1, idx[..., None].expand(-1, -1, x.size(-1)))


def _gather_scores(scores, idx):
    h = scores.size(1)
    rows = idx[:, None, :, None].expand(-1, h, -1, scores.size(-1))
    scores = scores.gather(2, rows)
    cols = idx[:, None, None, :].expand(-1, h, scores.size(2), -1)
    return scores.gather(3, cols)


class PatchTST(nn.Module):
    def __init__(self, n_vars, seq_len, pred_len, patch_len=16, stride=8, d_model=16,
                 n_heads=4, d_ff=128, e_layers=3, dropout=0.3, head_dropout=0.0, **_):
        super().__init__()
        self.patch_len, self.stride = patch_len, stride
        self.n_patches = (seq_len - patch_len) // stride + 2  # +1 for end padding
        self.revin = RevIN(n_vars)
        self.pad = nn.ReplicationPad1d((0, stride))
        self.W_P = nn.Linear(patch_len, d_model)
        self.W_pos = nn.Parameter(torch.empty(self.n_patches, d_model).uniform_(-0.02, 0.02))
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(e_layers))
        self.head = nn.Sequential(nn.Flatten(start_dim=-2),
                                  nn.Linear(d_model * self.n_patches, pred_len),
                                  nn.Dropout(head_dropout))

    def patchify(self, x):
        """[B, L, C] -> (model input patches, instance-normalised patches), each [B*C, N, P].

        The second tensor omits RevIN's learnable affine transform and is what the
        pruning scores are computed on.
        """
        b, _, c = x.shape
        z = self.pad(self.revin.normalize(x).permute(0, 2, 1))
        plain = z.unfold(-1, self.patch_len, self.stride)  # [B, C, N, P]
        patches = plain * self.revin.weight[None, :, None, None] + self.revin.bias[None, :, None, None]
        shape = (b * c, self.n_patches, self.patch_len)
        return patches.reshape(shape), plain.reshape(shape)

    def forward(self, x, plan: PruningPlan = None, return_kept: bool = False):
        b, _, c = x.shape
        patches, plain = self.patchify(x)
        h = self.drop(self.W_P(patches) + self.W_pos)
        h, kept = self.encode(h, plain, plan)

        h = h.reshape(b, c, self.n_patches, -1).transpose(2, 3)  # [B, C, d_model, N]
        y = self.head(h).transpose(1, 2)                          # [B, pred_len, C]
        y = self.revin.denormalize(y)
        return (y, kept) if return_kept else y

    def encode(self, h, patches, plan):
        n = h.size(1)
        prev = attn = None
        kpm = idx = full = None
        for i, layer in enumerate(self.layers):
            if plan is not None and i == plan.prune_layer:
                importance = plan.score(patches=patches, attn=attn)
                idx, valid = plan.select(importance)
                full = h
                h = _gather_tokens(h, idx)
                if prev is not None:
                    prev = _gather_scores(prev, idx)
                kpm = None if bool(valid.all()) else ~valid
            h, attn, prev = layer(h, prev, kpm)

        if idx is None:
            return h, None
        out = self._fill(h, full, idx, kpm, plan.fill)
        kept = torch.zeros(h.size(0), n, dtype=torch.bool, device=h.device)
        kept.scatter_(1, idx, True if kpm is None else ~kpm)
        return out, kept

    @staticmethod
    def _fill(h, full, idx, kpm, mode):
        """Write kept tokens back to their positions; decide what dropped ones become."""
        if kpm is not None:  # padded slots point at a kept token; neutralise them
            idx = idx.masked_fill(kpm, -1)
        if mode == "skip":    # dropped tokens bypass the remaining layers
            out = full.clone()
        elif mode == "mean":  # dropped tokens take the mean of kept tokens
            w = (~kpm).float()[..., None] if kpm is not None else torch.ones_like(h[..., :1])
            mean = (h * w).sum(1, keepdim=True) / w.sum(1, keepdim=True)
            out = mean.expand_as(full).clone()
        elif mode == "zero":
            out = torch.zeros_like(full)
        else:
            raise ValueError(f"unknown fill mode {mode!r}")
        if kpm is not None:
            # route padded slots to a scratch column that is discarded afterwards
            out = torch.cat([out, out[:, :1]], dim=1)
            idx = torch.where(idx < 0, torch.full_like(idx, out.size(1) - 1), idx)
            out.scatter_(1, idx[..., None].expand(-1, -1, h.size(-1)), h)
            return out[:, :-1]
        out.scatter_(1, idx[..., None].expand(-1, -1, h.size(-1)), h)
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

