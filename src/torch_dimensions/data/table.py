"""Long-format rows to a dense lattice-shaped series.

Input is what a database or CSV actually gives you: one row per observation,
carrying its coordinates, its timestamp, and its features. Output is the
``(T, *shape, F)`` tensor the models want, plus the lattice describing it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from torch_dimensions.data.coords import CoordMap, from_coords
from torch_dimensions.lattice import Lattice

__all__ = ["LatticeTable", "from_table"]


@dataclass
class LatticeTable:
    """A lattice-shaped series and everything needed to interpret it."""

    lattice: Lattice
    series: torch.Tensor  # (T, *shape, F); exactly zero at absent cells
    times: tuple  # length T, sorted unique time values
    coords: CoordMap

    @property
    def n_features(self) -> int:
        return self.series.shape[-1]

    def __len__(self) -> int:
        return self.series.shape[0]

    def __repr__(self) -> str:
        return (
            f"LatticeTable(T={len(self)}, shape={self.lattice.shape}, "
            f"F={self.n_features}, cells={self.lattice.n_valid}/{self.lattice.n_cells})"
        )


def from_table(
    coords: Sequence[Sequence],
    times: Sequence,
    values: torch.Tensor | Sequence[Sequence[float]],
    names: Sequence[str] | None = None,
    *,
    dense: bool = False,
    dtype: torch.dtype = torch.float32,
) -> LatticeTable:
    """Build a lattice and a dense series from long-format rows.

    Args:
        coords: ``(N, K)`` coordinate tuples, one per row.
        times: ``(N,)`` timestamps. Any orderable, hashable type; the distinct
            values become the time axis in sorted order.
        values: ``(N, F)`` features.
        names: axis names.
        dense: mark every combination present even if never observed.

    Duplicate ``(time, cell)`` pairs raise. Silently keeping the last one is the
    kind of default that turns a join bug into a plausible-looking dataset.
    """
    values = torch.as_tensor(values, dtype=dtype)
    if values.ndim != 2:
        raise ValueError(f"values must be (N, F); got {tuple(values.shape)}")
    n = values.shape[0]
    if len(coords) != n or len(times) != n:
        raise ValueError(
            f"coords ({len(coords)}), times ({len(times)}) and values ({n}) "
            "must describe the same number of rows"
        )

    cmap = from_coords(coords, names, dense=dense)
    order = sorted(dict.fromkeys(times))
    t_lookup = {t: i for i, t in enumerate(order)}
    t_index = torch.tensor([t_lookup[t] for t in times], dtype=torch.long)

    n_cells = cmap.lattice.n_cells
    flat_slot = t_index * n_cells + cmap.index
    if len(torch.unique(flat_slot)) != n:
        dupes = n - len(torch.unique(flat_slot))
        raise ValueError(
            f"{dupes} duplicate (time, cell) rows; aggregate them before building a lattice"
        )

    series = torch.zeros(len(order) * n_cells, values.shape[1], dtype=dtype)
    series[flat_slot] = values
    series = series.reshape(len(order), *cmap.lattice.shape, values.shape[1])

    return LatticeTable(lattice=cmap.lattice, series=series, times=tuple(order), coords=cmap)
