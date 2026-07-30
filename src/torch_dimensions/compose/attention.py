"""The kernel family: per-axis attention contracted across the lattice.

Where :class:`~torch_dimensions.AxialScan` sweeps a 1-D mixer along one axis
per layer, this builds an explicit ``(A, A)`` operator per axis and contracts
them all every layer — the factorized joint operator is a Kronecker product,
which is what keeps cost quadratic in axial size rather than in cell count.

Two variants, differing in where the kernel comes from:

- **axial attention** (``per_line=True``): scores are computed per line, so
  every row of the fold gets its own attention pattern.
- **CaFA** (``per_line=False``): features are pooled over the *other* spatial
  axes first, giving one kernel per axis per (batch, timestep) — the
  factorized-attention construction, cheaper and more structured.

The hybrid form: on a lattice with a time axis, the kernels own the spatial
axes and the model's 1-D mixer runs along time, each layer. This is why
``td.LSTM(nd_method=td.cafa)`` is meaningful — CaFA never consumes the LSTM,
it handles the axes the LSTM does not. Pooling (CaFA) deliberately keeps the
time dimension unpooled so the kernel at time ``t`` sees only time ``t``:
causality along time is the mixer's property and must not leak away through a
pooled kernel.

Scores carry a learnable relative-position bias per axis (spatial axes have
static sizes, so the table is well-defined); gating is ``"softmax"`` or
``"leaky_relu"`` (the CaFA paper's default). Sparse lattices are handled by
:func:`~torch_dimensions.axial_contract`'s per-line renormalization — for a
softmax kernel that renormalization *is* masked softmax, and for a signed
gate the relative cancellation guard applies.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_dimensions.compose.kernel import axial_contract
from torch_dimensions.compose.scan import axial_apply
from torch_dimensions.lattice import Lattice
from torch_dimensions.plan import ScanPlan

__all__ = ["AxialKernel"]

_GATES = ("softmax", "leaky_relu")


class AxialKernel(nn.Module):
    """Kernel-family block: per-axis attention over the lattice, optional
    mixer along time. See the module docstring.

    Args:
        mixer: zero-arg factory (or module) for the per-layer *time* mixer.
            Requires the lattice to have a time axis — on a purely spatial
            lattice the kernels are the whole model and a mixer would be
            silently dead weight, which is refused rather than allowed.
            Pass ``None`` for a kernel-only block.
        plan: depth and axis coverage. Each layer contracts every *spatial*
            axis the plan mentions, in the plan's first-appearance order;
            an axis the plan never names is never contracted (and
            ``plan.resolve`` warns, same as the scan family).
        per_line: per-line scores (axial attention) vs pooled per-axis
            kernels (CaFA).
        gate: ``"softmax"`` or ``"leaky_relu"``.
    """

    def __init__(
        self,
        mixer: Callable[[], nn.Module] | nn.Module | None,
        plan: ScanPlan,
        lattice: Lattice,
        d_model: int,
        *,
        per_line: bool = True,
        gate: str = "softmax",
        dropout: float = 0.0,
        norm: bool = True,
        residual: bool = True,
        chunk: int | None = None,
    ) -> None:
        super().__init__()
        if gate not in _GATES:
            raise ValueError(f"gate must be one of {_GATES}; got {gate!r}")
        self.lattice = lattice
        self.plan = plan.resolve(lattice)
        self.d_model = d_model
        self.per_line = per_line
        self.gate = gate
        self.residual = residual
        self.chunk = chunk

        time_index = 0 if lattice.time else None
        self.spatial_axes = [int(a) for a in self.plan.axes if a != time_index]
        if not self.spatial_axes:
            raise ValueError("the kernel family needs at least one spatial axis in the plan")

        n = len(self.plan)
        h = d_model
        # Flat (layer, axis) indexing — layer * n_axes + j — keeps every
        # per-axis module addressable without nested ModuleLists.
        n_ax = len(self.spatial_axes)
        self.q = nn.ModuleList(nn.Linear(h, h, bias=False) for _ in range(n * n_ax))
        self.k = nn.ModuleList(nn.Linear(h, h, bias=False) for _ in range(n * n_ax))
        sizes = [lattice.axis_size(a) for a in self.spatial_axes]
        self.bias = nn.ParameterList(
            nn.Parameter(torch.zeros(a, a)) for _ in range(n) for a in sizes
        )
        self.out = nn.ModuleList(nn.Linear(h, h) for _ in range(n))
        self.norms = nn.ModuleList(nn.LayerNorm(h) for _ in range(n)) if norm else None
        self.drop = nn.Dropout(dropout)

        if lattice.time:
            if mixer is None:
                self.mixers = None
            elif isinstance(mixer, nn.Module):
                self.mixers = nn.ModuleList([mixer] * n)  # shared, as in AxialScan
            else:
                self.mixers = nn.ModuleList([mixer() for _ in range(n)])
            self.time_norms = (
                nn.ModuleList(nn.LayerNorm(h) for _ in range(n))
                if norm and self.mixers is not None
                else None
            )
        else:
            if mixer is not None:
                raise ValueError(
                    "a mixer was given but the lattice has no time axis for it to sweep; "
                    "the kernel family owns every spatial axis, so the mixer would be dead "
                    "weight. Use a lattice with time=True (the hybrid form) or mixer=None."
                )
            self.mixers = None
            self.time_norms = None

        if not lattice.is_dense:
            self.register_buffer("cell_mask", lattice.mask(torch.bool), persistent=False)
        else:
            self.cell_mask = None

    def _masked(self, x: torch.Tensor) -> torch.Tensor:
        if self.cell_mask is None:
            return x
        return x * self.cell_mask.to(x.dtype)

    def _kernel(self, layer: int, j: int, axis: int, h: torch.Tensor) -> torch.Tensor:
        """Build the ``(M, A, A)`` operator for one axis of one layer."""
        lat = self.lattice
        scale = 1.0 / math.sqrt(self.d_model)
        idx = layer * len(self.spatial_axes) + j
        bias = self.bias[idx]

        if self.per_line:
            seq, _ = lat.to_sequence(h, axis)  # (M, A, H)
            q = self.q[idx](seq)
            k = self.k[idx](seq)
            scores = q @ k.transpose(1, 2) * scale + bias
        else:
            # Pool over the *other spatial* axes only. Batch stays batch, and
            # time deliberately stays unpooled: a kernel at time t built from
            # future timesteps would leak the future into a "causal" model.
            d = lat.tensor_dim(axis)
            keep = {0, d, h.ndim - 1}
            if lat.time:
                keep.add(1)
            reduce_dims = tuple(i for i in range(h.ndim) if i not in keep)
            counts = lat.valid_counts(axis).to(h.dtype).to(h.device)
            pooled = h.sum(reduce_dims) if reduce_dims else h  # (B, [T,] A, H)
            pooled = pooled / counts.unsqueeze(-1)
            q = self.q[idx](pooled)
            k = self.k[idx](pooled)
            scores = q @ k.transpose(-1, -2) * scale + bias  # (B, [T,] A, A)
            a = scores.shape[-1]
            # Expand to one kernel per folded line. The fold orders leading
            # dims as (B, [T,] *others), batch-major, so lines sharing a
            # (batch, timestep) are contiguous.
            lines_per = 1
            for i in range(1, h.ndim - 1):
                if i != d:
                    lines_per *= h.shape[i]
            if lat.time:
                lines_per //= h.shape[1]
                scores = scores.reshape(-1, 1, a, a).expand(-1, lines_per, a, a)
            else:
                scores = scores.reshape(-1, 1, a, a).expand(-1, lines_per, a, a)
            scores = scores.reshape(-1, a, a)

        if self.gate == "softmax":
            return F.softmax(scores, dim=-1)
        return F.leaky_relu(scores)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.d_model:
            raise ValueError(f"expected {self.d_model} features, got {x.shape[-1]}")
        x = self._masked(x)
        valid = None if self.cell_mask is None else self.cell_mask.to(x.dtype)
        for i in range(len(self.plan)):
            h = self.norms[i](x) if self.norms is not None else x
            for j, axis in enumerate(self.spatial_axes):
                kernel = self._kernel(i, j, axis, h)
                h = axial_contract(h, self.lattice, axis, kernel, valid=valid)
            h = self.out[i](h)
            x = x + self.drop(h) if self.residual else self.drop(h)
            x = self._masked(x)
            if self.mixers is not None:
                h = self.time_norms[i](x) if self.time_norms is not None else x
                h = axial_apply(h, self.lattice, 0, self.mixers[i], chunk=self.chunk)
                x = x + self.drop(h)
                x = self._masked(x)
        return x

    def extra_repr(self) -> str:
        kind = "per-line" if self.per_line else "pooled (CaFA)"
        return f"d_model={self.d_model}, {kind}, gate={self.gate}, lattice={self.lattice}"
