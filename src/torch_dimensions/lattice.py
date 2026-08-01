"""The N-D lattice: axis naming, permutation, and sparse-cell bookkeeping.

Every block in the library delegates its axis handling here, so that no model
ever hardcodes a rank or an axis meaning. This module holds no learnable
parameters and builds no modules — it is pure tensor bookkeeping, and it is
deliberately the most heavily tested part of the package. Almost every bug in
an N-D model presents as "the model is bad" and is actually an axis-order bug.

Canonical tensor layout::

    (B, [T,] *shape, H)

Batch first, an optional time axis when ``time=True``, then one dim per lattice
axis, then features last. Time is a normal sweepable axis in every respect
except that its length is dynamic and therefore absent from ``shape``;
causality is a property of the mixer, not of the axis.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import torch

__all__ = ["Lattice", "Restore", "Sub"]

AxisSpec = int | str


class Restore(NamedTuple):
    """Opaque handle returned by :meth:`Lattice.to_sequence`, consumed by
    :meth:`Lattice.from_sequence` to undo the fold."""

    axis: int
    shape: torch.Size


class Sub(NamedTuple):
    """A sub-lattice and the tensor selection that produces it.

    Returned by :meth:`Lattice.sliced`. The pair travels together on purpose:
    a sub-lattice whose selection is re-derived by the caller is a lattice and
    a tensor that agree only as long as nobody edits one of them.
    """

    lattice: Lattice
    indices: tuple[torch.Tensor, ...]
    """One index vector per lattice axis, into that axis of the parent."""

    def take(self, x: torch.Tensor) -> torch.Tensor:
        """Select the sub-lattice out of a parent-shaped ``(B, [T,] *shape, H)``."""
        lead = 1 + (1 if self.lattice.time else 0)
        for i, idx in enumerate(self.indices):
            dim = lead + i
            if idx.numel() == x.shape[dim] and bool(
                torch.equal(idx, torch.arange(idx.numel(), device=idx.device))
            ):
                continue  # a full-axis selection is a copy nobody asked for
            x = x.index_select(dim, idx.to(x.device))
        return x


@dataclass(eq=False)
class Lattice:
    """Describes the N-D grid a block operates over.

    Args:
        shape: size of each lattice axis. Excludes batch, features, and time.
        names: per-axis names; defaults to ``("dim0", "dim1", ...)``. Used for
            readable axis references (``lattice.permutation("width")``).
        valid: optional bool tensor of shape ``shape`` marking which cells
            exist. ``None`` means dense. This is what lets blocks operate on
            grids that are not fully populated.
        time: prepend a dynamic-length time axis named ``"time"``.
    """

    shape: tuple[int, ...]
    names: tuple[str, ...] | None = None
    valid: torch.Tensor | None = None
    time: bool = False

    def __post_init__(self) -> None:
        self.shape = tuple(int(s) for s in self.shape)
        if not self.shape and not self.time:
            # shape=() with time=True is the degenerate "just a sequence"
            # lattice, which is what makes the 1-D case fall out of the same
            # machinery instead of needing its own code path.
            raise ValueError("a lattice needs at least one axis; got shape=() with time=False")
        if any(s <= 0 for s in self.shape):
            raise ValueError(f"lattice axes must be positive; got shape={self.shape}")

        if self.names is None:
            self.names = tuple(f"dim{i}" for i in range(len(self.shape)))
        else:
            self.names = tuple(str(n) for n in self.names)
            if len(self.names) != len(self.shape):
                raise ValueError(
                    f"got {len(self.names)} names for {len(self.shape)} axes: "
                    f"names={self.names}, shape={self.shape}"
                )
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"axis names must be unique; got {self.names}")
        if self.time and "time" in self.names:
            raise ValueError("'time' is reserved when time=True; rename that lattice axis")

        if self.valid is not None:
            if not self.shape:
                raise ValueError("a time-only lattice has no cells to mark valid")
            if tuple(self.valid.shape) != self.shape:
                raise ValueError(
                    f"valid mask has shape {tuple(self.valid.shape)}, expected {self.shape}"
                )
            # Clone, don't alias. `.to(torch.bool)` returns the caller's own
            # tensor when it is already bool, and a caller who later reuses or
            # edits that tensor would silently desync every cache derived from
            # it here — the exact misplacement this class exists to prevent.
            self.valid = self.valid.to(torch.bool).clone()
            if not bool(self.valid.any()):
                raise ValueError("valid mask selects no cells")

        self._cache: dict[str, torch.Tensor] = {}
        self._frozen = True

    # A lattice is a value object, and blocks derive buffers from it at
    # construction — ``AxialScan`` registers its cell mask once, and the
    # derived ``flat_idx`` is cached here. Mutating a field afterwards would
    # leave both stale and silently misplace data, so mutation is refused.
    # Build a new lattice, or use :meth:`to` to move devices.
    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError(f"Lattice is immutable; cannot set {name!r}. Construct a new one.")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"Lattice is immutable; cannot delete {name!r}")

    # -- structure ----------------------------------------------------------

    @property
    def rank(self) -> int:
        """Number of lattice axes, excluding time."""
        return len(self.shape)

    @property
    def n_axes(self) -> int:
        """Number of sweepable axes, including time when present."""
        return self.rank + (1 if self.time else 0)

    @property
    def axis_names(self) -> tuple[str, ...]:
        """Sweepable axis names, time first when present."""
        assert self.names is not None
        return ("time", *self.names) if self.time else self.names

    @property
    def tensor_ndim(self) -> int:
        """Rank of a well-formed tensor for this lattice: batch + axes + features."""
        return self.n_axes + 2

    @property
    def n_cells(self) -> int:
        """Total lattice cells, valid or not."""
        return math.prod(self.shape)

    @property
    def is_dense(self) -> bool:
        return self.valid is None

    @property
    def n_valid(self) -> int:
        """Number of cells that exist. Equals :attr:`n_cells` when dense."""
        if self.valid is None:
            return self.n_cells
        return int(self.valid.sum())

    @property
    def device(self) -> torch.device | None:
        return None if self.valid is None else self.valid.device

    def to(self, device: torch.device | str) -> Lattice:
        """Return a lattice whose derived tensors live on ``device``.

        Dense lattices hold no tensors, so this is a no-op for them. Blocks are
        expected to ``register_buffer`` whatever they need at construction, so
        that moving the *module* keeps the buffers in step.
        """
        if self.valid is None:
            return self
        return Lattice(self.shape, self.names, self.valid.to(device), self.time)

    # -- sub-lattices -------------------------------------------------------

    def sliced(self, **selection: slice | Sequence[int] | torch.Tensor) -> Sub:
        """Cut a sub-lattice out along one or more named axes.

        Splitting an experiment over *space* — hold out a region, train on the
        rest — is otherwise hand-rolled index arithmetic against a validity
        mask, which is the same class of bug this module exists to remove.
        Axes not mentioned are kept whole::

            sub = lat.sliced(row=slice(0, 4), col=[0, 2, 5])
            model = td.Mamba(64, 12, lattice=sub.lattice)
            y = model(sub.take(x))

        The rank never changes: an axis is narrowed, never dropped, so a
        sub-lattice is interchangeable with its parent everywhere a lattice is
        accepted. Select a single position with ``slice(i, i + 1)`` — an
        integer would silently change the rank of every tensor downstream, and
        a rank that changes because of a train/test split is not a rank.
        """
        if self.time and "time" in selection:
            raise ValueError(
                "'time' has no static size, so it cannot be sliced here; "
                "window the data instead (td.data.LatticeWindow)"
            )
        idx: list[torch.Tensor] = []
        for i, size in enumerate(self.shape):
            assert self.names is not None
            name = self.names[i]
            sel = selection.pop(name, None)
            if sel is None:
                idx.append(torch.arange(size))
            elif isinstance(sel, slice):
                idx.append(torch.arange(size)[sel])
            elif isinstance(sel, int):
                raise TypeError(
                    f"axis {name!r}: an integer index would drop the axis and change the "
                    f"lattice rank; use slice({sel}, {sel + 1}) to keep it"
                )
            else:
                v = torch.as_tensor(sel, dtype=torch.long).reshape(-1)
                if v.numel() and (int(v.min()) < 0 or int(v.max()) >= size):
                    raise IndexError(f"axis {name!r}: indices out of range for size {size}")
                idx.append(v)
            if idx[-1].numel() == 0:
                raise ValueError(f"axis {name!r}: selection is empty; a lattice axis needs a cell")
        if selection:
            raise KeyError(f"unknown axes {sorted(selection)}; lattice has {self.names}")

        valid = None
        if self.valid is not None:
            valid = self.valid
            for i, v in enumerate(idx):
                valid = valid.index_select(i, v.to(valid.device))
            if not bool(valid.any()):
                raise ValueError("the selected sub-lattice contains no existing cells")
        sub = Lattice(tuple(int(v.numel()) for v in idx), self.names, valid, self.time)
        return Sub(sub, tuple(idx))

    @classmethod
    def merge(cls, lattices: Sequence[Lattice], axis: AxisSpec) -> Lattice:
        """Concatenate lattices along ``axis`` — the inverse of :meth:`sliced`.

        Every input must agree on names, on ``time``, and on every axis size
        but ``axis``. Sparsity survives: a dense input contributes a full block,
        so merging dense with sparse yields the honest mixed mask rather than
        quietly dropping either claim. Concatenate the data yourself along
        ``lattice.tensor_dim(axis)`` — the lattice describes tensors, it does
        not hold them.
        """
        lats = list(lattices)
        if not lats:
            raise ValueError("need at least one lattice to merge")
        head = lats[0]
        i = head.lattice_index(axis)
        for other in lats[1:]:
            if other.names != head.names or other.time != head.time:
                raise ValueError(f"lattices must agree on names and time; {head!r} vs {other!r}")
            if len(other.shape) != len(head.shape) or any(
                a != b
                for j, (a, b) in enumerate(zip(head.shape, other.shape, strict=True))
                if j != i
            ):
                raise ValueError(
                    f"lattices may differ only along {head.axis_names[head.axis_index(axis)]!r}; "
                    f"got {head.shape} and {other.shape}"
                )
        shape = list(head.shape)
        shape[i] = sum(lat.shape[i] for lat in lats)
        valid = None
        if any(not lat.is_dense for lat in lats):
            blocks = [
                lat.valid if lat.valid is not None else torch.ones(lat.shape, dtype=torch.bool)
                for lat in lats
            ]
            valid = torch.cat([b.to(blocks[0].device) for b in blocks], dim=i)
        return cls(tuple(shape), head.names, valid, head.time)

    # -- axis resolution ----------------------------------------------------

    def axis_index(self, axis: AxisSpec) -> int:
        """Resolve a name or index to a sweep-axis index in ``[0, n_axes)``.

        Index 0 is time when ``time=True``, otherwise the first lattice axis.
        Negative indices count from the end.
        """
        if isinstance(axis, str):
            names = self.axis_names
            if axis not in names:
                raise KeyError(f"unknown axis {axis!r}; lattice has {names}")
            return names.index(axis)
        i = int(axis)
        if i < 0:
            i += self.n_axes
        if not 0 <= i < self.n_axes:
            raise IndexError(f"axis {axis} out of range for {self.n_axes} axes")
        return i

    def tensor_dim(self, axis: AxisSpec) -> int:
        """Position of ``axis`` within a ``(B, [T,] *shape, H)`` tensor."""
        return 1 + self.axis_index(axis)

    def lattice_index(self, axis: AxisSpec) -> int:
        """Resolve to an index into :attr:`shape`. Raises for the time axis."""
        i = self.axis_index(axis)
        if self.time:
            if i == 0:
                raise ValueError("'time' is not a lattice axis; it has no static size")
            return i - 1
        return i

    def axis_size(self, axis: AxisSpec) -> int:
        """Static length of ``axis``. Raises for time, whose length is dynamic."""
        return self.shape[self.lattice_index(axis)]

    # -- permutation --------------------------------------------------------

    def permutation(self, axis: AxisSpec) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Permutation moving ``axis`` to the sequence position, and its inverse.

        The forward permutation maps ``(B, [T,] *shape, H)`` to
        ``(B, *others, A, H)`` so that everything but the swept axis folds into
        the batch. The inverse is built directly rather than derived twice; the
        test suite is what checks it against ``torch.argsort``.
        """
        d = self.tensor_dim(axis)
        nd = self.tensor_ndim
        others = [i for i in range(1, nd - 1) if i != d]
        perm = (0, *others, d, nd - 1)

        inv = [0] * nd
        for new_pos, old in enumerate(perm):
            inv[old] = new_pos
        return perm, tuple(inv)

    def _check(self, x: torch.Tensor) -> None:
        if x.ndim != self.tensor_ndim:
            raise ValueError(
                f"expected a {self.tensor_ndim}-D tensor (B, "
                f"{'T, ' if self.time else ''}*{self.shape}, H); got shape {tuple(x.shape)}"
            )
        start = 1 + (1 if self.time else 0)
        got = tuple(x.shape[start : start + self.rank])
        if got != self.shape:
            raise ValueError(f"tensor has lattice dims {got}, expected {self.shape}")

    def to_sequence(self, x: torch.Tensor, axis: AxisSpec) -> tuple[torch.Tensor, Restore]:
        """Fold ``x`` into ``(M, A, H)`` batched 1-D sequences along ``axis``.

        ``M`` is the batch times every other axis, ``A`` is the length of the
        swept axis. This is the shape every mixer sees; a mixer never learns
        which axis it is sweeping or how many axes exist.
        """
        self._check(x)
        i = self.axis_index(axis)
        perm, _ = self.permutation(i)
        d = x.permute(*perm).contiguous()
        return d.reshape(-1, d.shape[-2], d.shape[-1]), Restore(i, d.shape)

    def from_sequence(self, seq: torch.Tensor, restore: Restore) -> torch.Tensor:
        """Invert :meth:`to_sequence`."""
        _, inv = self.permutation(restore.axis)
        return seq.reshape(restore.shape).permute(*inv).contiguous()

    # -- sparse cells -------------------------------------------------------

    @property
    def flat_idx(self) -> torch.Tensor:
        """Indices of existing cells into the flattened lattice, shape ``(G,)``."""
        if "flat_idx" not in self._cache:
            if self.valid is None:
                idx = torch.arange(self.n_cells)
            else:
                idx = self.valid.reshape(-1).nonzero(as_tuple=True)[0]
            self._cache["flat_idx"] = idx
        return self._cache["flat_idx"]

    def scatter(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, [T,] G, H)`` of existing cells -> dense ``(B, [T,] *shape, H)``.

        Cells that do not exist are exactly zero, which is what lets a plain
        sum double as a masked sum downstream.
        """
        lead, h = x.shape[:-2], x.shape[-1]
        if x.shape[-2] != self.n_valid:
            raise ValueError(f"expected {self.n_valid} cells, got {x.shape[-2]}")
        out = x.new_zeros(*lead, self.n_cells, h)
        # Index on x's device. The cache lives wherever `valid` lives, and
        # torch only tolerates the CPU-index-into-device-tensor direction —
        # a device lattice indexing a CPU tensor raises. `.to` is a no-op
        # when they already agree.
        out[..., self.flat_idx.to(x.device), :] = x
        return out.reshape(*lead, *self.shape, h)

    def gather(self, x: torch.Tensor) -> torch.Tensor:
        """Dense ``(B, [T,] *shape, H)`` -> ``(B, [T,] G, H)`` of existing cells."""
        lead, h = x.shape[: -(self.rank + 1)], x.shape[-1]
        flat = x.reshape(*lead, self.n_cells, h)
        return flat[..., self.flat_idx.to(x.device), :]

    def mask(self, dtype: torch.dtype = torch.bool) -> torch.Tensor:
        """Validity mask shaped to broadcast over ``(B, [T,] *shape, H)``.

        Always a fresh tensor. The cached version of this returned — for the
        bool case — a reshaped *view* of ``valid``, so a caller writing into
        "their" mask corrupted the lattice through it. Callers hold or buffer
        the result anyway (it is built at module construction, not per
        forward), so there is nothing worth caching.
        """
        base = torch.ones(self.shape, dtype=torch.bool) if self.valid is None else self.valid
        lead = (1, 1) if self.time else (1,)
        return base.reshape(*lead, *self.shape, 1).to(dtype, copy=True)

    def valid_counts(self, axis: AxisSpec) -> torch.Tensor:
        """Existing cells at each position of ``axis``, shape ``(axis_size,)``.

        This is the denominator for masked-mean pooling over the other axes.
        Never zero: dead lines are clamped to 1 so callers divide safely, and
        their numerators are zero anyway.
        """
        i = self.lattice_index(axis)
        if self.valid is None:
            n = self.n_cells // self.shape[i]
            return torch.full((self.shape[i],), float(n))
        others = tuple(j for j in range(self.rank) if j != i)
        counts = self.valid.sum(dim=others) if others else self.valid.to(torch.long)
        return counts.to(torch.float32).clamp_min(1.0)

    # -- display ------------------------------------------------------------

    def __repr__(self) -> str:
        parts = [f"shape={self.shape}", f"names={self.axis_names}"]
        if self.time:
            parts.append("time=True")
        if not self.is_dense:
            parts.append(f"valid={self.n_valid}/{self.n_cells}")
        return f"Lattice({', '.join(parts)})"
