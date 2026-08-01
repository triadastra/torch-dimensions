"""An on-disk source, and masked normalization statistics.

Two things the data layer promised and did not have. The ``LatticeSource``
protocol's claim is that "a memory-mapped array, a zarr store, or a database
cursor all batch correctly" — a claim with two in-memory implementations behind
it, which is a promise rather than a feature. :class:`MemmapSource` is the
on-disk one, and it is written to fail the way real on-disk sources fail so
that :func:`~torch_dimensions.testing.check_data_source` has something honest
to check.

Normalization is here for a narrower reason: a mean taken over a sparse
lattice's zeros is wrong, invisibly. The absent cells are exactly zero by
construction, so they drag every statistic toward zero in proportion to how
sparse the lattice is, and nothing about the resulting model looks broken.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from torch_dimensions.lattice import Lattice

__all__ = ["MemmapSource", "Normalizer", "masked_stats"]


class MemmapSource:
    """A ``.npy`` file on disk, memory-mapped, as a :class:`LatticeSource`.

        td.data.MemmapSource.write("series.npy", tensor)
        source = td.data.MemmapSource("series.npy", lattice)

    Only the requested slice is read, so the file may be far larger than
    memory.

    **The handle is opened lazily and dropped on pickling.** That is the whole
    difficulty of an on-disk source and the reason this class exists as a
    reference: ``DataLoader(num_workers>0)`` pickles the source into each
    worker, and a live mmap or file handle either fails to pickle or —
    worse — pickles into a handle that is invalid in the child. DEBUG.md #9
    records what that failure mode looks like from the outside: not an
    exception, a hang. Each worker reopens the file itself.

    Needs ``numpy`` (for the ``.npy`` container only); torch ships with it in
    every practical install, but it is imported lazily so this module never
    breaks an import that would otherwise work.
    """

    def __init__(
        self,
        path: str | Path,
        lattice: Lattice,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"no such file: {self.path}")
        self._lattice = lattice
        self.dtype = dtype
        self._array: Any = None
        head = self._open()
        got = tuple(head.shape[1:-1])
        if got != tuple(lattice.shape):
            raise ValueError(
                f"{self.path.name} has lattice dims {got}, but the lattice declares "
                f"{tuple(lattice.shape)}"
            )

    @staticmethod
    def write(path: str | Path, series: torch.Tensor) -> Path:
        """Write a ``(T, *shape, F)`` tensor to a ``.npy`` this can read."""
        import numpy as np

        path = Path(path)
        np.save(path, series.detach().cpu().numpy())
        return path if path.suffix == ".npy" else path.with_suffix(".npy")

    def _open(self) -> Any:
        if self._array is None:
            import numpy as np

            self._array = np.load(self.path, mmap_mode="r")
        return self._array

    @property
    def lattice(self) -> Lattice:
        return self._lattice

    def __len__(self) -> int:
        return int(self._open().shape[0])

    def __getitem__(self, index: slice) -> torch.Tensor:
        # `.copy()` because torch cannot take ownership of a read-only mmap
        # view, and a tensor that aliases one would be a use-after-close the
        # moment the handle is dropped.
        return torch.from_numpy(self._open()[index].copy()).to(self.dtype)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_array"] = None  # the child process opens its own
        return state

    def __repr__(self) -> str:
        return f"MemmapSource({self.path.name}, {self._lattice})"


@dataclass(frozen=True)
class Normalizer:
    """Per-cell mean and scale, applied and inverted.

    Deliberately a value object with no state beyond the statistics: the
    library computes them and applies them, and never decides *when* — fitting
    on the wrong split is the caller's classic mistake to make, and hiding it
    inside a training loop this library does not have would only make it
    harder to see.
    """

    mean: torch.Tensor
    scale: torch.Tensor

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / self.scale.to(x.device)

    def invert(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale.to(x.device) + self.mean.to(x.device)


def masked_stats(
    series: torch.Tensor,
    lattice: Lattice,
    *,
    eps: float = 1e-6,
    per_cell: bool = True,
) -> Normalizer:
    """Mean and standard deviation over **present cells only**.

    Args:
        series: ``(T, *lattice.shape, F)``.
        lattice: supplies the validity mask.
        per_cell: statistics per cell (the default — each series has its own
            scale) or one set shared across the lattice.

    Absent cells hold exactly zero, so a plain ``series.mean()`` on a lattice
    that is 30% absent is pulled 30% toward zero and the standard deviation
    with it. Nothing about the resulting model looks wrong; it is simply
    trained on data centred on a number that means nothing. NaNs are treated
    as absent too, so a real gap and a structural absence are handled the same
    way.
    """
    if series.ndim != lattice.rank + 2:
        raise ValueError(
            f"expected a (T, *{lattice.shape}, F) tensor; got shape {tuple(series.shape)}"
        )
    present = lattice.mask(torch.bool).reshape(*lattice.shape, 1).to(series.device)
    known = present.unsqueeze(0) & ~series.isnan()
    values = torch.nan_to_num(series, nan=0.0)

    dims: tuple[int, ...] = (0,) if per_cell else tuple(range(series.ndim - 1))
    count = known.expand_as(values).sum(dims).clamp_min(1)
    mean = values.sum(dims) / count
    # Var over the same masked set: subtract the mean only where a value exists,
    # or the absent zeros contribute (0 - mean)^2 and inflate the scale.
    centered = (values - mean) * known
    var = (centered * centered).sum(dims) / count
    scale = var.sqrt().clamp_min(eps)
    return Normalizer(mean=mean, scale=scale)
