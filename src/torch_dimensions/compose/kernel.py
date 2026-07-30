"""Kernel composition: per-axis operators contracted into a joint one.

The other family. Where :mod:`~torch_dimensions.compose.scan` sweeps a 1-D
operator along one axis per layer, this builds one kernel per axis and
contracts them all in a single pass. On a dense lattice the result is exactly
the Kronecker product ``A_0 ⊗ A_1 ⊗ … ⊗ A_{n-1}`` — which is the whole point,
because it means cost is quadratic in *axial* size rather than in the number of
cells. Materializing one attention matrix per lattice line is what runs out of
memory at rank 4; this does not.

Every existing implementation hardcodes its contraction to one rank, via einsum
strings keyed by axis index. Here the contraction reuses the same fold as the
scan family, so it works at any rank with no per-rank table.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import reduce

import torch

from torch_dimensions.lattice import AxisSpec, Lattice

__all__ = ["axial_contract", "kron_operator"]

_EPS = 1e-6


def axial_contract(
    x: torch.Tensor,
    lattice: Lattice,
    axis: AxisSpec,
    kernel: torch.Tensor,
    *,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Contract ``kernel`` into ``x`` along ``axis``.

    ``kernel`` is ``(A, A)`` for the swept axis, or anything broadcasting to
    ``(M, A, A)`` where ``M`` is the folded batch. Output position ``q`` becomes
    ``sum_k kernel[q, k] * x[k]`` along that axis, every other axis untouched.

    ``valid`` is the broadcast cell mask. When given, absent cells are zeroed
    *before* the contraction and each output line is then renormalized by the
    kernel mass that landed on cells which actually exist, so an output stays a
    convex combination of present values. That per-line rescale is the one
    departure from a strict Kronecker product, and it costs ``O(N · A)``
    elementwise work rather than ``O(N · A²)`` of score memory — which is
    precisely why the factorized path survives where a per-line implementation
    does not.

    The zeroing has to happen on *every* contraction, not once at entry. A
    contraction leaves absent cells holding a weighted average of present ones
    — legitimate scratch, but nonzero. Carry that into the next axis and its
    numerator sums those scratch values while its denominator still counts only
    present keys, so the result is no longer a convex combination and inflates
    with sparsity. Zeroing once at entry is only sufficient for a rank-1
    lattice.
    """
    if valid is not None:
        x = x * valid

    seq, restore = lattice.to_sequence(x, axis)  # (M, A, H)
    out = kernel @ seq

    if valid is not None:
        mass, _ = lattice.to_sequence(valid.expand(*x.shape[:-1], 1), axis)  # (M, A, 1)
        den = kernel @ mass
        # Guard the magnitude, not the lower bound. `clamp_min` assumes the
        # denominator is a non-negative mass, which holds only for a
        # non-negative kernel. A signed kernel — LeakyReLU-gated scores, say —
        # can drive this to zero by cancellation while the numerator stays
        # nonzero, and clamping to +eps then divides by ~0 and explodes by
        # orders of magnitude. Leaving degenerate lines unscaled is the honest
        # fallback: a genuinely dead line has a zero numerator and stays zero.
        den = torch.where(den.abs() < _EPS, torch.ones_like(den), den)
        # No nan_to_num here. The where-guard already keeps |den| >= eps, so
        # this division cannot create a fresh NaN or inf — the only NaNs that
        # could reach a nan_to_num are ones already in `x`, and zeroing those
        # would silently launder a diverging model into finite numbers mid-
        # network. A NaN that arrives must leave; that is what makes divergence
        # debuggable.
        out = out / den

    return lattice.from_sequence(out, restore)


def kron_operator(kernels: Sequence[torch.Tensor]) -> torch.Tensor:
    """The joint operator per-axis kernels are equivalent to, built explicitly.

    Only usable on small lattices — it is ``(∏S, ∏S)``, which is exactly the
    cost the factorized path exists to avoid. It exists so the factorization can
    be *checked* against the thing it claims to equal, rather than against
    another call to itself.
    """
    return reduce(torch.kron, kernels)
