"""How full is the lattice, really — answered before a model is built.

A lattice is dense when every combination of its axes exists and sparse when
most of them do not, and which one you have decides whether the sparse
machinery is doing anything for you. That is a property of the *data*, not a
setting, so it is measured rather than declared:

    report = td.data.sparsity(table)
    print(report)          # a summary, including the percentage
    report.percent_sparse  # 32.1

The per-axis breakdown is the part worth reading twice. A lattice can be 30%
sparse because observations are scattered, or because one station in twelve
reports nothing at all — the first is ordinary, the second is usually a join
that went wrong upstream, and `empty_slices` tells them apart.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from torch_dimensions.lattice import Lattice

__all__ = ["SparsityReport", "sparsity"]


@dataclass
class SparsityReport:
    """What a pre-run over the data found about lattice occupancy."""

    shape: tuple[int, ...]
    names: tuple[str, ...]
    present: int
    total: int
    per_axis: dict[str, list[int]] = field(default_factory=dict)
    """Per axis, how many cells are present at each index along it."""
    empty_slices: dict[str, list[int]] = field(default_factory=dict)
    """Per axis, the indices with no present cell at all."""
    observed: float | None = None
    """Fraction of (time, cell, feature) entries actually observed, when the
    data carried a time axis. ``None`` when only structure was inspected."""

    @property
    def absent(self) -> int:
        return self.total - self.present

    @property
    def fraction_present(self) -> float:
        return self.present / self.total if self.total else 1.0

    @property
    def percent_sparse(self) -> float:
        """The headline number: what percentage of the lattice is absent."""
        return 100.0 * (1.0 - self.fraction_present)

    @property
    def dense(self) -> bool:
        return self.present == self.total

    def summary(self) -> str:
        head = (
            f"lattice {' × '.join(str(s) for s in self.shape) or '—'}  "
            f"{self.present}/{self.total} cells present  "
            f"({self.percent_sparse:.1f}% sparse)"
        )
        if self.dense:
            head += "  — dense; a validity mask would do nothing"
        lines = [head]
        if self.observed is not None:
            lines.append(f"  observed entries: {100.0 * self.observed:.1f}% of the series")
        for name in self.names:
            counts = self.per_axis.get(name, [])
            if not counts:
                continue
            per = self.total // len(counts) if counts else 0
            worst = min(counts) if counts else 0
            empty = self.empty_slices.get(name, [])
            note = f"  {name:<12} {min(counts)}–{max(counts)} of {per} per index"
            if empty:
                note += f"   ⚠ {len(empty)} empty: {empty[:6]}{'…' if len(empty) > 6 else ''}"
            elif worst == per:
                note += "   (full)"
            lines.append(note)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"SparsityReport(shape={self.shape}, present={self.present}/{self.total}, "
            f"percent_sparse={self.percent_sparse:.1f})"
        )


def _lattice_mask(lattice: Lattice) -> torch.Tensor:
    """The presence mask in the lattice's own shape.

    Not ``lattice.mask()``, which is broadcast-shaped ``(1, 1, *shape, 1)`` for
    multiplying against data — right for arithmetic, wrong here, where the
    singleton batch/time/feature axes would be reported as lattice axes and
    push the real names off the end.
    """
    if lattice.valid is None:
        return torch.ones(tuple(lattice.shape), dtype=torch.bool)
    return lattice.valid.bool()


def _mask_from_values(
    values: torch.Tensor, shape: tuple[int, ...], missing: float | None
) -> tuple[torch.Tensor, float]:
    """Reduce a data tensor to a per-cell presence mask.

    The lattice axes are located as a contiguous run inside ``values.shape``;
    everything before them is time and everything after is features, both of
    which are reduced away — a cell counts as present when *any* observation of
    it exists. Requiring the caller to state the shape rather than guessing it
    is deliberate: a (6, 8) lattice inside a (10, 6, 8, 1) tensor has exactly
    one sensible reading, but a (6, 6) one does not, and silently picking is
    how a transposed axis survives to training.
    """
    rank = len(shape)
    dims = values.shape
    start = None
    for s in range(len(dims) - rank + 1):
        if tuple(dims[s : s + rank]) == shape:
            if start is not None:
                raise ValueError(
                    f"lattice shape {shape} appears more than once in tensor shape "
                    f"{tuple(dims)}; slice the tensor so the placement is unambiguous"
                )
            start = s
    if start is None:
        raise ValueError(f"lattice shape {shape} does not appear in tensor shape {tuple(dims)}")

    observed = (
        torch.isfinite(values)
        if values.is_floating_point()
        else torch.ones_like(values, dtype=torch.bool)
    )
    if missing is not None:
        observed = observed & (values != missing)

    reduce_dims = [d for d in range(values.ndim) if not (start <= d < start + rank)]
    fraction = float(observed.float().mean()) if observed.numel() else 1.0
    mask = observed.any(dim=reduce_dims) if reduce_dims else observed
    return mask.bool(), fraction


def sparsity(
    data,
    *,
    shape: Sequence[int] | None = None,
    names: Sequence[str] | None = None,
    missing: float | None = None,
) -> SparsityReport:
    """Measure how much of a lattice its data actually occupies.

    Args:
        data: a :class:`~torch_dimensions.Lattice`, a
            :class:`~torch_dimensions.data.LatticeTable`, a boolean presence
            mask shaped like the lattice, or a data tensor (with ``shape``
            given) whose non-finite entries mark absence.
        shape: the lattice shape, required only for the data-tensor form.
        names: axis names, when the source does not carry them.
        missing: an additional sentinel counted as absent (e.g. ``0.0`` for a
            table that filled gaps with zeros rather than NaNs).

    Returns:
        A :class:`SparsityReport`; ``report.percent_sparse`` is the headline.
    """
    from torch_dimensions.data.table import LatticeTable

    observed: float | None = None

    if isinstance(data, LatticeTable):
        lattice = data.lattice
        mask = _lattice_mask(lattice)
        _, observed = _mask_from_values(data.series, tuple(lattice.shape), missing)
        axis_names = tuple(lattice.names or ())
    elif isinstance(data, Lattice):
        lattice = data
        mask = _lattice_mask(lattice)
        axis_names = tuple(lattice.names or ())
    else:
        values = torch.as_tensor(data)
        if values.dtype == torch.bool and shape is None:
            mask = values
        else:
            if shape is None:
                raise ValueError(
                    "measuring a data tensor needs the lattice shape: "
                    "sparsity(values, shape=(6, 8), names=('h', 'w'))"
                )
            mask, observed = _mask_from_values(values, tuple(shape), missing)
        axis_names = tuple(names or ())

    mask = mask.bool()
    shape_t = tuple(mask.shape)
    if names is not None:
        axis_names = tuple(names)
    # Lattice names include the time axis in front of the spatial ones; only
    # the spatial names describe the mask's dimensions.
    if len(axis_names) > len(shape_t):
        axis_names = axis_names[len(axis_names) - len(shape_t) :]
    if len(axis_names) != len(shape_t):
        axis_names = tuple(f"dim{i}" for i in range(len(shape_t)))

    per_axis: dict[str, list[int]] = {}
    empty: dict[str, list[int]] = {}
    for i, name in enumerate(axis_names):
        others = [d for d in range(mask.ndim) if d != i]
        counts = mask.sum(dim=others).tolist() if others else mask.long().tolist()
        per_axis[name] = [int(c) for c in counts]
        empty[name] = [j for j, c in enumerate(counts) if int(c) == 0]

    return SparsityReport(
        shape=shape_t,
        names=axis_names,
        present=int(mask.sum()),
        total=int(mask.numel()),
        per_axis=per_axis,
        empty_slices=empty,
        observed=observed,
    )
