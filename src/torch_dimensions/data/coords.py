"""Turning observed coordinates into a lattice.

This is the piece that exists to prevent one specific, silent failure. Given
rows keyed by ``(state, commodity, flow)`` — or any other tuple — a hand-written
mapping from those keys to grid positions is easy to get subtly wrong, and a
mis-shuffled lattice still trains, still converges, and is quietly wrong.
Deriving the mapping once, here, makes that class of bug impossible downstream.

Coordinates may be integers or any hashable categoricals (strings, dates,
tuples). Each axis gets its own vocabulary, sorted where the values are
orderable and first-seen otherwise, so a lattice built twice from the same data
is identical.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from torch_dimensions.lattice import Lattice

__all__ = ["CoordMap", "from_coords"]


def _vocab(values: Sequence) -> tuple:
    """Distinct values in a deterministic order."""
    seen = list(dict.fromkeys(values))
    try:
        return tuple(sorted(seen))
    except TypeError:
        # Mixed or unorderable types: fall back to first appearance, which is
        # still deterministic for a given input ordering.
        return tuple(seen)


@dataclass
class CoordMap:
    """The lattice implied by a set of observed coordinates, plus the mapping
    back and forth."""

    lattice: Lattice
    vocabs: tuple[tuple, ...]
    index: torch.Tensor  # (N,) flat cell index, one per input row

    @property
    def names(self) -> tuple[str, ...]:
        assert self.lattice.names is not None
        return self.lattice.names

    def encode(self, coords: Sequence[Sequence]) -> torch.Tensor:
        """Map coordinate tuples to flat cell indices, for data the lattice was
        not built from. Unknown values raise rather than silently landing in
        cell zero."""
        lookups = [{v: i for i, v in enumerate(v_)} for v_ in self.vocabs]
        strides = _strides(self.lattice.shape)
        out = []
        for row in coords:
            if len(row) != len(self.vocabs):
                raise ValueError(f"expected {len(self.vocabs)} coordinates, got {len(row)}")
            flat = 0
            for k, (value, lookup) in enumerate(zip(row, lookups, strict=True)):
                if value not in lookup:
                    raise KeyError(f"{value!r} is not a known {self.names[k]!r} value")
                flat += lookup[value] * strides[k]
            out.append(flat)
        return torch.tensor(out, dtype=torch.long)

    def decode(self, flat: int) -> tuple:
        """Flat cell index back to the original coordinate values."""
        out = []
        for size, vocab in zip(reversed(self.lattice.shape), reversed(self.vocabs), strict=True):
            flat, r = divmod(flat, size)
            out.append(vocab[r])
        return tuple(reversed(out))


def _strides(shape: Sequence[int]) -> list[int]:
    strides, acc = [], 1
    for size in reversed(shape):
        strides.append(acc)
        acc *= size
    return list(reversed(strides))


def from_coords(
    coords: Sequence[Sequence],
    names: Sequence[str] | None = None,
    *,
    dense: bool = False,
    time: bool = True,
) -> CoordMap:
    """Infer a lattice from observed coordinate tuples.

    Args:
        coords: ``(N, K)`` — one tuple per observation. Rows may repeat.
        names: axis names; defaults to ``dim0..dimK``.
        dense: treat every combination as present. By default a combination
            that never appears is marked absent, which is the honest reading of
            "we have no data for it" and is what lets the model mask it.
        time: give the lattice a time axis. On by default because coordinates
            almost always index a series, but a static grid is legitimate.

    Returns:
        A :class:`CoordMap` whose ``.lattice`` is ready to hand to a model.
    """
    if isinstance(coords, torch.Tensor):
        coords = coords.tolist()
    rows = [tuple(r) for r in coords]
    if not rows:
        raise ValueError("need at least one coordinate row")
    k = len(rows[0])
    if any(len(r) != k for r in rows):
        raise ValueError("all coordinate rows must have the same length")

    vocabs = tuple(_vocab([r[axis] for r in rows]) for axis in range(k))
    shape = tuple(len(v) for v in vocabs)
    lookups = [{v: i for i, v in enumerate(v_)} for v_ in vocabs]
    strides = _strides(shape)

    index = torch.tensor(
        [sum(lookups[a][r[a]] * strides[a] for a in range(k)) for r in rows],
        dtype=torch.long,
    )

    valid = None
    if not dense:
        flags = torch.zeros(math.prod(shape), dtype=torch.bool)
        flags[index] = True
        # Every combination observed means the grid really is dense; saying so
        # is more honest than carrying an all-True mask around.
        valid = None if bool(flags.all()) else flags.reshape(shape)

    lattice = Lattice(shape=shape, names=tuple(names) if names else None, valid=valid, time=time)
    return CoordMap(lattice=lattice, vocabs=vocabs, index=index)
