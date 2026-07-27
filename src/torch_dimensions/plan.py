"""The scan schedule: which axis each layer sweeps, and in which direction.

A ``ScanPlan`` is **data, not control flow**. In every existing N-D
implementation this schedule is a pair of inline list comprehensions welded to
the model, which is why none of them can be inspected, serialized, or swapped
without editing the module. Here it is a plain value: printable, comparable,
round-trippable, and testable with no tensors and no modules in sight.

Axes may be given by name or index. A plan stays unresolved until
:meth:`ScanPlan.resolve` binds it to a :class:`~torch_dimensions.Lattice`,
which keeps plans serializable and reusable across lattices of the same rank.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from torch_dimensions.lattice import Lattice

__all__ = ["ScanPlan", "Step"]

AxisSpec = int | str


class Step(NamedTuple):
    """One layer's sweep: an axis, and whether to traverse it backwards."""

    axis: AxisSpec
    reverse: bool = False


class ScanPlan:
    """An ordered schedule of per-layer axis sweeps.

    Construct with :meth:`cyclic`, :meth:`paired`, or :meth:`from_list` rather
    than passing steps directly.
    """

    __slots__ = ("steps",)

    def __init__(self, steps: Sequence[Step]) -> None:
        steps = tuple(Step(s.axis, bool(s.reverse)) for s in steps)
        if not steps:
            raise ValueError("a scan plan needs at least one step")
        object.__setattr__(self, "steps", steps)

    # -- constructors -------------------------------------------------------

    @staticmethod
    def _check_axes(axes: Sequence[AxisSpec], n_layers: int) -> tuple[AxisSpec, ...]:
        axes = tuple(axes)
        if not axes:
            raise ValueError("need at least one axis to scan")
        if len(set(axes)) != len(axes):
            raise ValueError(f"axes must be distinct; got {axes}")
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1; got {n_layers}")
        return axes

    @classmethod
    def cyclic(
        cls,
        axes: Sequence[AxisSpec],
        n_layers: int,
        bidirectional: bool = False,
    ) -> ScanPlan:
        """Cycle through ``axes``, one axis per layer.

        When ``bidirectional``, direction flips after each *full cycle* rather
        than after each layer. Flipping per layer looks equivalent and is not:
        with an even number of axes it aliases against the cycle, so every axis
        is pinned to one direction forever and the plan is silently
        unidirectional. Flipping per cycle gives every axis both directions.
        """
        axes = cls._check_axes(axes, n_layers)
        n = len(axes)
        return cls(
            [Step(axes[i % n], bidirectional and (i // n) % 2 == 1) for i in range(n_layers)]
        )

    @classmethod
    def paired(cls, axes: Sequence[AxisSpec], n_layers: int) -> ScanPlan:
        """Sweep each axis forward then immediately backward before moving on.

        Gives an axis both directions within adjacent layers, where
        :meth:`cyclic` spreads them a full cycle apart.
        """
        axes = cls._check_axes(axes, n_layers)
        n = len(axes)
        return cls([Step(axes[(i // 2) % n], bool(i % 2)) for i in range(n_layers)])

    @classmethod
    def from_list(cls, steps: Sequence[Step | tuple | AxisSpec]) -> ScanPlan:
        """Build from explicit steps: ``Step`` objects, ``(axis, reverse)``
        pairs, or bare axes (taken as forward)."""
        out = []
        for s in steps:
            if isinstance(s, Step):
                out.append(s)
            elif isinstance(s, tuple):
                if len(s) != 2:
                    raise ValueError(f"expected (axis, reverse) pairs; got {s!r}")
                out.append(Step(s[0], bool(s[1])))
            else:
                out.append(Step(s, False))
        return cls(out)

    # -- inspection ---------------------------------------------------------

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self) -> Iterator[Step]:
        return iter(self.steps)

    def __getitem__(self, i: int) -> Step:
        return self.steps[i]

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ScanPlan) and self.steps == other.steps

    def __hash__(self) -> int:
        return hash(self.steps)

    @property
    def axes(self) -> tuple[AxisSpec, ...]:
        """Distinct axes touched, in order of first appearance."""
        seen: dict[AxisSpec, None] = {}
        for s in self.steps:
            seen.setdefault(s.axis, None)
        return tuple(seen)

    def counts(self) -> dict[AxisSpec, int]:
        """How many layers sweep each axis."""
        out: dict[AxisSpec, int] = {}
        for s in self.steps:
            out[s.axis] = out.get(s.axis, 0) + 1
        return out

    def is_resolved(self) -> bool:
        return all(isinstance(s.axis, int) for s in self.steps)

    # -- binding to a lattice -----------------------------------------------

    def resolve(self, lattice: Lattice) -> ScanPlan:
        """Return an equivalent plan with every axis as an integer index.

        Raises if an axis does not exist on ``lattice``. Warns — but does not
        fail — when the plan leaves an axis unswept: that is legal and
        occasionally intended, and usually a mistake.
        """
        resolved = [Step(lattice.axis_index(s.axis), s.reverse) for s in self.steps]
        touched = {s.axis for s in resolved}
        missing = [lattice.axis_names[i] for i in range(lattice.n_axes) if i not in touched]
        if missing:
            warnings.warn(
                f"scan plan never sweeps {missing}; those axes get no mixing",
                UserWarning,
                stacklevel=2,
            )
        return ScanPlan(resolved)

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict:
        return {"steps": [[s.axis, s.reverse] for s in self.steps]}

    @classmethod
    def from_dict(cls, d: dict) -> ScanPlan:
        if "steps" not in d:
            raise KeyError(f"plan dict needs a 'steps' key; got {sorted(d)}")
        return cls.from_list([tuple(s) for s in d["steps"]])

    def __repr__(self) -> str:
        body = ", ".join(f"{s.axis}{'-' if s.reverse else '+'}" for s in self.steps)
        return f"ScanPlan({body})"
