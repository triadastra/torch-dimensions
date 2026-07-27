"""Where the data actually comes from.

:class:`LatticeSource` is a **protocol, not a base class**. That is the whole
customization story: a memory-mapped array, a zarr store, an HDF5 file, or a
database cursor all batch correctly if they satisfy three members, and none of
them has to inherit from or even know about this library.

Two reference implementations ship — an in-memory tensor and a
:class:`~torch_dimensions.data.LatticeTable` — because a protocol with no
implementations is a promise rather than a feature.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from torch.utils.data import Dataset

from torch_dimensions.data.window import LatticeWindow, Window
from torch_dimensions.lattice import Lattice

__all__ = ["LatticeDataset", "LatticeSource", "Sample", "TensorSource"]


@runtime_checkable
class LatticeSource(Protocol):
    """A time-indexed store of lattice-shaped observations."""

    @property
    def lattice(self) -> Lattice:
        """The grid every timestep is shaped by. Static, not per-sample."""

    def __len__(self) -> int:
        """Number of timesteps."""

    def __getitem__(self, index: slice) -> torch.Tensor:
        """``(T, *lattice.shape, F)`` for the requested timestep range."""


class TensorSource:
    """The simplest source: a tensor already in memory."""

    def __init__(self, series: torch.Tensor, lattice: Lattice) -> None:
        expected = (*lattice.shape,)
        if tuple(series.shape[1:-1]) != expected:
            raise ValueError(
                f"series has lattice dims {tuple(series.shape[1:-1])}, expected {expected}"
            )
        self.series, self._lattice = series, lattice

    @property
    def lattice(self) -> Lattice:
        return self._lattice

    def __len__(self) -> int:
        return self.series.shape[0]

    def __getitem__(self, index: slice) -> torch.Tensor:
        return self.series[index]


class Sample(dict):
    """One window. A dict so it survives any collate function, with attribute
    access because ``sample.x`` reads better than ``sample["x"]``."""

    __getattr__ = dict.__getitem__


class LatticeDataset(Dataset):
    """A ``torch.utils.data.Dataset`` over windows of a source.

    Deliberately thin. It does not shuffle, batch, normalize, or prefetch —
    ``DataLoader`` already does the first two and the rest are the caller's
    policy, not ours.

    The lattice is *not* in each sample. It is static metadata; stacking it
    once per item and again per batch would be pure waste. Read it from
    ``dataset.lattice``.
    """

    def __init__(self, source: LatticeSource, windows: LatticeWindow) -> None:
        if len(windows) == 0:
            raise ValueError("windows is empty; nothing to iterate")
        over = [w for w in windows if w.y1 > len(source)]
        if over:
            raise ValueError(
                f"{len(over)} windows run past the end of the source "
                f"({len(source)} timesteps); build LatticeWindow with the source's length"
            )
        self.source, self.windows = source, windows

    @property
    def lattice(self) -> Lattice:
        return self.source.lattice

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, i: int) -> Sample:
        w: Window = self.windows[i]
        sample = Sample(x=self.source[w.x0 : w.x1], window=w)
        if w.y1 > w.y0:
            sample["y"] = self.source[w.y0 : w.y1]
        return sample
