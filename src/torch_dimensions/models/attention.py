"""The N-D transformer: attention sweeping one axis per layer.

``td.Transformer`` is to :class:`~torch_dimensions.AxialKernel` what
``td.Mamba`` is to a 1-D selective scan — the same abstraction, a different
1-D operator. Both constructions in the literature now have a name:

    td.Transformer(64, 12, lattice)                       # attention sweeps every axis
    td.LSTM(64, 12, lattice, method=td.axial_attention)   # kernels over space, RNN over time

Which to reach for is a size question with a measured answer in BENCHMARKS.md:
per-axis attention is cheaper below roughly 50 cells per axis, and the
factorized kernel family wins above it.
"""

from __future__ import annotations

from torch_dimensions.mixers.attention import AttentionMixer
from torch_dimensions.models.base import LatticeModel

__all__ = ["Transformer", "TransformerND"]


class Transformer(LatticeModel):
    """Axial self-attention over a sequence, or over an N-D lattice.

    Each layer attends along the single axis its schedule assigns it, so a
    rank-3 lattice costs ``O(cells · A)`` rather than the ``O(cells²)`` of
    attention over the flattened grid — the reason axial transformers exist.

    Takes the same arguments as :class:`~torch_dimensions.LSTM`. Mixer options
    go through ``mixer_kwargs``::

        td.Transformer(64, 12, lattice, mixer_kwargs={"n_heads": 8})

    **On causality.** The mixer is non-causal by default, and that is the
    correct default for a mixer that is not told which axis it sweeps —
    masking "the future" of a spatial axis is meaningless. For a causal time
    axis, use the hybrid form (``method=td.axial_attention`` with a causal
    mixer along time) or pass ``mixer_kwargs={"causal": True}`` and accept that
    it applies to every axis, which for spatial axes is a lower-triangular
    receptive field rather than a bug.
    """

    _mixer = AttentionMixer


TransformerND = Transformer
"""Alias. Unlike ``S4ND``/``MambaND`` this needs no separate class with a
mandatory ``dim``: those names exist in the literature as distinct models, so
the library refuses to let ``S4ND(dim=1)`` quietly be S4. "TransformerND" is
not a published model name, so there is nothing to be mistaken for."""
