"""Attention as a mixer — the entry that makes "N-D Transformer" literal.

The kernel family in :mod:`torch_dimensions.compose.attention` builds one
attention operator *per axis* and contracts them; this is the other half, and
a much smaller idea: multi-head self-attention along whichever single axis a
layer is sweeping, used exactly like an LSTM or a selective scan.

That gives the two constructions the literature actually means by "axial
transformer" a name each:

    td.Transformer(64, 12, lattice)                    # attention sweeps every axis
    td.LSTM(64, 12, lattice, method=td.axial_attention)  # kernels over space, RNN over time

The first is this file. Attention is quadratic in the swept axis and linear in
the number of lines, which for a lattice means ``O(cells · A)`` — the cost
BENCHMARKS.md measures against the factorized path, and the reason the kernel
family exists for large lattices.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["AttentionMixer"]


class AttentionMixer(nn.Module):
    """Pre-norm multi-head self-attention over the swept axis, plus an MLP.

    A transformer block, with one deliberate difference from the textbook: the
    residual connections live *here* rather than being left to the composition
    layer, because a transformer block is defined as attention-plus-MLP with
    its own residuals, and splitting them across two layers of abstraction
    would make the module something else wearing the name.

    Args:
        d_model: feature width, unchanged in and out.
        n_heads: must divide ``d_model``.
        causal: mask out future positions. **Off by default**, which is the
            right default here and the opposite of a language model's: a mixer
            does not know which axis it is sweeping, and masking "the future"
            of a spatial axis is meaningless. Causality along time is the
            schedule's business — build the model with a causal mixer for the
            time axis, or use the hybrid form where the time mixer is separate.
        mlp_ratio: hidden width of the feed-forward block, as a multiple.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        *,
        causal: bool = False,
        dropout: float = 0.0,
        mlp_ratio: float = 2.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"n_heads={n_heads} does not divide d_model={d_model}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.causal = causal
        self.head_dim = d_model // n_heads

        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (M, A, H)
        m, a, h = x.shape
        if h != self.d_model:
            raise ValueError(f"expected {self.d_model} features, got {h}")

        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        # (M, heads, A, head_dim)
        q, k, v = (t.reshape(m, a, self.n_heads, self.head_dim).transpose(1, 2) for t in (q, k, v))
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if self.causal:
            mask = torch.ones(a, a, dtype=torch.bool, device=x.device).triu(1)
            scores = scores.masked_fill(mask, float("-inf"))
        attended = F.softmax(scores, dim=-1) @ v
        attended = attended.transpose(1, 2).reshape(m, a, h)

        x = x + self.drop(self.proj(attended))
        return x + self.drop(self.mlp(self.norm2(x)))

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, n_heads={self.n_heads}, causal={self.causal}"
