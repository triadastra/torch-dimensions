"""Sequential composition: sweep a 1-D mixer along one lattice axis at a time.

This is what makes N-D tractable without an N-D kernel. Permute the swept axis
to the sequence position, fold every other axis into the batch, run an ordinary
1-D operator, and permute back. Alternating the axis across layers recovers N-D
context, so a 1-D CUDA kernel is reused unchanged.

:func:`axial_apply` is the whole mechanism and is deliberately a pure function:
no parameters, no state, nothing to configure. :class:`AxialScan` is the thin
module that wraps it in pre-norm residual layers and walks a
:class:`~torch_dimensions.ScanPlan`. Keeping them separate means the axis
bookkeeping can be tested against an explicit per-line reference without any
norm or residual in the way.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from torch_dimensions.lattice import AxisSpec, Lattice
from torch_dimensions.plan import ScanPlan

__all__ = ["AxialScan", "axial_apply"]


def axial_apply(
    x: torch.Tensor,
    lattice: Lattice,
    axis: AxisSpec,
    fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    reverse: bool = False,
    chunk: int | None = None,
) -> torch.Tensor:
    """Apply a 1-D operator independently along every line of ``axis``.

    ``fn`` receives ``(M, A, H)`` — a plain batch of 1-D sequences — and must
    return the same shape. It is never told which axis it is sweeping, how many
    axes exist, or how long the other axes are. That ignorance is the extension
    point: any 1-D module is a valid mixer.

    ``reverse`` flips the sequence before the call and unflips after, so the
    operator sees the line back-to-front while the output stays in lattice
    order.

    ``chunk`` caps how many folded rows go through ``fn`` at once. Scanning a
    short axis folds every other axis into ``M``, which can reach tens of
    thousands of rows and overrun a fused kernel's grid limits. Pure-torch
    mixers have no such limit, so this defaults to off; the kernel adapters
    that need it set it themselves rather than exposing a magic constant.
    """
    if chunk is not None and chunk < 1:
        raise ValueError(f"chunk must be >= 1 or None; got {chunk}")
    seq, restore = lattice.to_sequence(x, axis)
    if reverse:
        seq = seq.flip(1)

    if chunk is None or seq.shape[0] <= chunk:
        out = fn(seq)
    else:
        out = torch.cat([fn(seq[i : i + chunk]) for i in range(0, seq.shape[0], chunk)], dim=0)

    if out.shape != seq.shape:
        raise ValueError(
            f"mixer changed shape: got {tuple(out.shape)}, expected {tuple(seq.shape)}. "
            "A mixer must map (M, A, H) -> (M, A, H)."
        )
    if reverse:
        out = out.flip(1)
    return lattice.from_sequence(out, restore)


class AxialScan(nn.Module):
    """Stack of pre-norm residual layers, each sweeping one axis of a lattice.

    Args:
        mixer: either a zero-argument factory called once per layer (each layer
            gets its own weights, as in Mamba-ND), or an already-built
            ``nn.Module``, in which case every layer *shares* it.
        plan: which axis each layer sweeps and in which direction. Resolved
            against ``lattice`` at construction, so unknown axes fail here
            rather than at the first forward pass.
        lattice: the grid being swept.
        d_model: feature width. A mixer maps ``d_model -> d_model``; this stack
            never changes the feature dimension.
        chunk: see :func:`axial_apply`.

    On a lattice with absent cells, those cells are zeroed on entry and after
    every layer. Zeroing on *entry* is what makes the outputs at present cells
    independent of whatever values were sitting in absent ones. Note that an
    absent cell still occupies a position in its line, so it advances a
    recurrence — the guarantee is invariance to its *value*, not to its
    existence.
    """

    def __init__(
        self,
        mixer: Callable[[], nn.Module] | nn.Module,
        plan: ScanPlan,
        lattice: Lattice,
        d_model: int,
        *,
        dropout: float = 0.0,
        norm: bool = True,
        residual: bool = True,
        chunk: int | None = None,
    ) -> None:
        super().__init__()
        self.lattice = lattice
        self.plan = plan.resolve(lattice)
        self.d_model = d_model
        self.residual = residual
        self.chunk = chunk

        n = len(self.plan)
        if isinstance(mixer, nn.Module):
            self.mixers = nn.ModuleList([mixer] * n)  # shared weights, by request
        else:
            self.mixers = nn.ModuleList([mixer() for _ in range(n)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n)]) if norm else None
        self.drop = nn.Dropout(dropout)

        if not lattice.is_dense:
            self.register_buffer("cell_mask", lattice.mask(torch.bool), persistent=False)
        else:
            self.cell_mask = None

    def _masked(self, x: torch.Tensor) -> torch.Tensor:
        if self.cell_mask is None:
            return x
        return x * self.cell_mask.to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.d_model:
            raise ValueError(f"expected {self.d_model} features, got {x.shape[-1]}")
        x = self._masked(x)
        for i, step in enumerate(self.plan):
            h = self.norms[i](x) if self.norms is not None else x
            h = axial_apply(
                h,
                self.lattice,
                step.axis,
                self.mixers[i],
                reverse=step.reverse,
                chunk=self.chunk,
            )
            x = x + self.drop(h) if self.residual else self.drop(h)
            x = self._masked(x)
        return x

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, plan={self.plan}, lattice={self.lattice}"
