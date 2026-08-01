"""Joint composition: no factorization at all.

The third method of multidimensionality, and the honest baseline for the other
two. Fold **every** lattice axis into one sequence and run the 1-D mixer over
the whole thing. A rank-3 lattice of 8×8×8 becomes one sequence of 512 tokens;
the mixer sees every cell and has no idea a lattice was ever involved.

This is what a Vision Transformer does. ViT does not attend along rows and then
along columns — it flattens the patch grid into one sequence and attends over
all of it, which is why ``td.ViT`` needs this method and not
:func:`~torch_dimensions.axial_scan`. It is also what every "flatten it and
use a sequence model" baseline does, which makes it the comparison the axial
methods have to beat rather than a strawman nobody implemented.

**The cost, stated plainly.** Attention over the flattened lattice is
``O(cells²)``; axial attention is ``O(cells · A)``; the factorized kernel
family is ``O(Σ A²)``. At 8×8 that ordering barely matters and joint attention
is the most expressive of the three. At 64×64×64 it is 2.6e5 tokens and the
scores alone do not fit in memory. BENCHMARKS.md has the crossover.

**Sparse lattices become genuinely cheaper here**, and that is not true of the
other methods. Absent cells are dropped from the sequence rather than masked
into it: a lattice at 40% occupancy is a 40%-length sequence, and quadratic
attention over it costs 16% as much. The scan and kernel families must keep
absent cells in place — a recurrence has to step over them — so they can only
mask. This is the one composition where sparsity is a saving rather than a
bookkeeping obligation.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn

from torch_dimensions.lattice import Lattice
from torch_dimensions.plan import ScanPlan

__all__ = ["Flatten"]


class Flatten(nn.Module):
    """Stack of pre-norm residual layers over the fully flattened lattice.

    Args:
        mixer: a zero-argument factory (one per layer) or a built module
            (shared), as in :class:`~torch_dimensions.AxialScan`.
        plan: contributes its **depth only**. There is no axis to choose —
            every layer mixes every axis — so the schedule's axis assignments
            and directions are not used. Kept in the signature because depth
            is a property of the plan everywhere else in the library, and a
            second way to say "how many layers" would be a second way to
            disagree.
        join_time: fold the time axis into the same sequence as the spatial
            cells (joint space-time attention, ViViT's first variant). When
            ``False``, time folds into the batch instead and each timestep is
            mixed independently — the model is then not a sequence model along
            time at all, which is right for per-frame encoders and wrong for
            forecasting.

    On a sparse lattice the sequence contains only cells that exist. Absent
    cells never reach the mixer, so their values cannot influence anything —
    the same guarantee the other families give by masking, obtained here by
    construction.
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
        join_time: bool = True,
    ) -> None:
        super().__init__()
        if mixer is None:
            raise ValueError(
                "the flatten method is nothing but a mixer over the flattened lattice; "
                "with mixer=None there would be no operator at all"
            )
        self.lattice = lattice
        # Resolved for consistency with the other families (it validates axis
        # names against the lattice), then used only for its length. Silently:
        # every axis is mixed in every layer here, so an "axis never swept"
        # warning would be true of the schedule and false of the model.
        self.plan = plan.resolve(lattice, warn=False)
        self.d_model = d_model
        self.residual = residual
        self.chunk = chunk
        self.join_time = join_time and lattice.time

        n = len(self.plan)
        if isinstance(mixer, nn.Module):
            self.mixers = nn.ModuleList([mixer] * n)
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

    @property
    def seq_len(self) -> int:
        """Tokens the mixer sees per row, excluding time when it is not joined."""
        return self.lattice.n_valid

    def _to_tokens(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        """``(B, [T,] *shape, H)`` -> ``(rows, L, H)`` over present cells."""
        lat = self.lattice
        g = lat.gather(x)  # (B, [T,] G, H) — absent cells dropped
        if lat.time:
            b, t, cells, h = g.shape
            # Joined: one sequence of T·G tokens, time-major, so the flattened
            # order agrees with time order and a causal mixer stays causal in
            # time. Not joined: time folds into the batch and each timestep is
            # mixed on its own.
            seq = g.reshape(b, t * cells, h) if self.join_time else g.reshape(b * t, cells, h)
            return seq, (b, t, cells, h)
        b, cells, h = g.shape
        return g, (b, cells, h)

    def _from_tokens(self, seq: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
        return self.lattice.scatter(seq.reshape(*shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.d_model:
            raise ValueError(f"expected {self.d_model} features, got {x.shape[-1]}")
        # Dropping absent cells from the *sequence* keeps their values out of
        # the mixer, but the residual stream still carries whatever was sitting
        # in them, so the output at an absent cell would echo its input. The
        # other families zero on entry and after every layer; so does this one.
        # Caught by the conformance suite's mask-invariance check, which is
        # exactly the reasoning error that check exists for: "they never reach
        # the mixer" is not the same claim as "they cannot influence output".
        x = self._masked(x)
        for i in range(len(self.plan)):
            h = self.norms[i](x) if self.norms is not None else x
            seq, shape = self._to_tokens(h)
            if self.chunk is None or seq.shape[0] <= self.chunk:
                out = self.mixers[i](seq)
            else:
                rows = range(0, seq.shape[0], self.chunk)
                out = torch.cat([self.mixers[i](seq[j : j + self.chunk]) for j in rows], dim=0)
            if out.shape != seq.shape:
                raise ValueError(
                    f"mixer changed shape: got {tuple(out.shape)}, expected {tuple(seq.shape)}. "
                    "A mixer must map (M, A, H) -> (M, A, H)."
                )
            h = self._from_tokens(out, shape)
            x = x + self.drop(h) if self.residual else self.drop(h)
            x = self._masked(x)
        return x

    def extra_repr(self) -> str:
        span = "space+time" if self.join_time else "space only"
        return (
            f"d_model={self.d_model}, layers={len(self.plan)}, tokens={self.seq_len} "
            f"({span}), lattice={self.lattice}"
        )
