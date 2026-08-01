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
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from torch_dimensions.lattice import Lattice

__all__ = ["AxisCoverage", "Coverage", "ScanPlan", "Step"]

AxisSpec = int | str


class Step(NamedTuple):
    """One layer's sweep: an axis, and whether to traverse it backwards."""

    axis: AxisSpec
    reverse: bool = False


@dataclass(frozen=True)
class AxisCoverage:
    """What one axis actually receives from a plan."""

    name: str
    index: int
    layers: tuple[int, ...]
    """Layer indices that sweep this axis, in order."""
    forward: int
    backward: int

    @property
    def n_sweeps(self) -> int:
        return self.forward + self.backward

    @property
    def direction(self) -> str:
        """``"both"``, ``"forward"``, ``"backward"``, or ``"none"``."""
        if self.forward and self.backward:
            return "both"
        if self.forward:
            return "forward"
        return "backward" if self.backward else "none"


@dataclass(frozen=True)
class Coverage:
    """A machine-readable report of what a plan does to a lattice.

    The question "does this schedule actually reach every axis, in both
    directions" is asked by the constructor's warning, by the spec the viewer
    renders, and by anyone reading a plan — three places that each used to
    recompute it. This is the one answer they share.

    Purely descriptive: unlike :meth:`ScanPlan.resolve` it never warns, because
    a report that emits warnings cannot be used to decide whether to warn.
    """

    n_layers: int
    axes: tuple[AxisCoverage, ...]
    """Every lattice axis, swept or not, in lattice order."""

    @property
    def unswept(self) -> tuple[str, ...]:
        return tuple(a.name for a in self.axes if a.n_sweeps == 0)

    @property
    def pinned(self) -> tuple[str, ...]:
        """Axes swept in exactly one direction — half the receptive field."""
        return tuple(a.name for a in self.axes if a.direction in ("forward", "backward"))

    def directions(self) -> dict[str, str]:
        """Axis name to direction, swept axes only (the spec's shape)."""
        return {a.name: a.direction for a in self.axes if a.n_sweeps}

    def __getitem__(self, name: str) -> AxisCoverage:
        for a in self.axes:
            if a.name == name:
                return a
        raise KeyError(f"no axis {name!r} in coverage; has {[a.name for a in self.axes]}")

    def to_dict(self) -> dict:
        return {
            "n_layers": self.n_layers,
            "axes": [
                {
                    "name": a.name,
                    "index": a.index,
                    "layers": list(a.layers),
                    "forward": a.forward,
                    "backward": a.backward,
                    "direction": a.direction,
                }
                for a in self.axes
            ],
            "unswept": list(self.unswept),
            "pinned": list(self.pinned),
        }

    def __repr__(self) -> str:
        width = max((len(a.name) for a in self.axes), default=1)
        rows = "\n".join(
            f"  {a.name:<{width}}  {a.forward:>3}→ {a.backward:>3}←  {a.direction}"
            for a in self.axes
        )
        return f"Coverage({self.n_layers} layers)\n{rows}"


class ScanPlan:
    """An ordered schedule of per-layer axis sweeps.

    Construct with :meth:`cyclic`, :meth:`paired`, or :meth:`from_list` rather
    than passing steps directly.
    """

    __slots__ = ("steps",)

    steps: tuple[Step, ...]

    def __init__(self, steps: Sequence[Step]) -> None:
        built = tuple(Step(s.axis, bool(s.reverse)) for s in steps)
        if not built:
            raise ValueError("a scan plan needs at least one step")
        object.__setattr__(self, "steps", built)

    # Immutable because it is hashable. Mutating a plan would change its hash,
    # silently losing it from any dict or set holding it — and worse, a block
    # builds one mixer per step at construction, so a plan edited afterwards
    # would no longer describe the layers that actually run.
    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(
            f"ScanPlan is immutable; cannot set {name!r}. "
            "Build a new plan with ScanPlan.from_list(...)."
        )

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"ScanPlan is immutable; cannot delete {name!r}")

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

    @staticmethod
    def _bidi_set(
        bidirectional: bool | AxisSpec | Sequence[AxisSpec] | None,
        axes: Sequence[AxisSpec],
    ) -> set[AxisSpec]:
        """Normalize the ``bidirectional`` argument to a set of axes.

        Bidirectionality is per-axis on purpose. Forward-only along time is
        correct — that is causality — while forward-only along a spatial or
        categorical axis is just lost receptive field. An all-or-nothing flag
        cannot express that distinction.
        """
        if bidirectional is None or bidirectional is False:
            return set()
        if bidirectional is True:
            return set(axes)
        # A bare axis is a common shorthand. Guard the string case explicitly:
        # set("time") is {'t','i','m','e'}, which would silently match nothing.
        if isinstance(bidirectional, (str, int)):
            bidirectional = [bidirectional]
        chosen = set(bidirectional)
        unknown = chosen - set(axes)
        if unknown:
            raise ValueError(f"bidirectional axes {sorted(map(str, unknown))} are not in {axes}")
        return chosen

    @staticmethod
    def _warn_if_pinned(steps: Sequence[Step], wanted: set[AxisSpec]) -> None:
        """Warn when an axis was asked to be bidirectional but only got one way.

        Bidirectional coverage costs layers. Below that budget the request is
        silently downgraded, which is exactly the failure this class exists to
        make visible.
        """
        seen: dict[AxisSpec, set[bool]] = {}
        for s in steps:
            seen.setdefault(s.axis, set()).add(s.reverse)
        pinned = [a for a in wanted if len(seen.get(a, set())) < 2]
        if pinned:
            warnings.warn(
                f"axes {sorted(map(str, pinned))} were requested bidirectional but only get "
                f"one direction in {len(steps)} layers; bidirectional coverage of k axes needs "
                f"roughly 2k layers, so either add layers or scan fewer axes",
                UserWarning,
                stacklevel=3,
            )

    @classmethod
    def cyclic(
        cls,
        axes: Sequence[AxisSpec],
        n_layers: int,
        bidirectional: bool | AxisSpec | Sequence[AxisSpec] = False,
        warn: bool = True,
    ) -> ScanPlan:
        """Cycle through ``axes``, one axis per layer.

        ``bidirectional`` accepts ``True``/``False`` or an explicit collection
        of axes, so time can stay causal while spatial axes get both
        directions.

        Direction flips after each *full cycle*, not after each layer. Flipping
        per layer looks equivalent and is not: with an even number of axes the
        two periods phase-lock, every axis is pinned to one direction forever,
        and the plan is silently unidirectional.
        """
        axes = cls._check_axes(axes, n_layers)
        bidi = cls._bidi_set(bidirectional, axes)
        n = len(axes)
        steps = [
            Step(axes[i % n], axes[i % n] in bidi and (i // n) % 2 == 1) for i in range(n_layers)
        ]
        if warn:
            cls._warn_if_pinned(steps, bidi)
        return cls(steps)

    @classmethod
    def paired(
        cls,
        axes: Sequence[AxisSpec],
        n_layers: int,
        bidirectional: bool | AxisSpec | Sequence[AxisSpec] = True,
        warn: bool = True,
    ) -> ScanPlan:
        """Sweep each bidirectional axis forward then immediately backward.

        This is the schedule used by the official Mamba-ND implementation,
        which advances the axis ordering every *two* layers while flipping
        direction every layer — so each ordering is used once forward and once
        backward, and no phase-locking is possible for any axis count. Axes
        outside ``bidirectional`` take a single forward layer instead of two.

        Prefer this over :meth:`cyclic` when layers are plentiful: it gives an
        axis both directions in adjacent layers rather than a full cycle apart.
        The cost is coverage — pairing k axes needs 2k layers before the
        schedule repeats, so at shallow depth it reaches fewer distinct axes.
        """
        axes = cls._check_axes(axes, n_layers)
        bidi = cls._bidi_set(bidirectional, axes)
        template = [
            Step(ax, rev) for ax in axes for rev in ((False, True) if ax in bidi else (False,))
        ]
        steps = [template[i % len(template)] for i in range(n_layers)]
        if warn:
            cls._warn_if_pinned(steps, bidi)
        return cls(steps)

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

    # -- algebra ------------------------------------------------------------

    def __add__(self, other: ScanPlan) -> ScanPlan:
        """Concatenate two schedules: ``a + b`` runs a's layers, then b's."""
        if not isinstance(other, ScanPlan):
            return NotImplemented
        return ScanPlan((*self.steps, *other.steps))

    def __mul__(self, k: int) -> ScanPlan:
        """Repeat the schedule ``k`` times."""
        if not isinstance(k, int) or isinstance(k, bool):
            return NotImplemented
        if k < 1:
            raise ValueError(f"a plan must be repeated at least once; got {k}")
        return ScanPlan(self.steps * k)

    __rmul__ = __mul__

    def reversed(self) -> ScanPlan:
        """The same layers in the opposite **order**.

        Layer order, not sweep direction — the axes and their ``reverse`` flags
        are untouched. For the mirror-image sweep use :meth:`flipped`; the two
        are different plans and naming only one of them "reversed" is how they
        get confused.
        """
        return ScanPlan(tuple(reversed(self.steps)))

    def flipped(self) -> ScanPlan:
        """The same layers in the same order, every sweep **direction** negated.

        Makes bidirectionality composable: ``plan + plan.flipped()`` gives every
        axis both directions in a schedule of twice the depth, whatever the
        original was.
        """
        return ScanPlan(tuple(Step(s.axis, not s.reverse) for s in self.steps))

    # -- coverage -----------------------------------------------------------

    def coverage(self, lattice: Lattice) -> Coverage:
        """Report what this plan does to every axis of ``lattice``.

        Includes axes the plan never touches — the interesting ones are exactly
        the ones absent from the schedule, so a report keyed only by what the
        plan mentions cannot show them.
        """
        seen: dict[int, list[tuple[int, bool]]] = {}
        for layer, s in enumerate(self.steps):
            seen.setdefault(lattice.axis_index(s.axis), []).append((layer, s.reverse))
        axes = []
        for i in range(lattice.n_axes):
            hits = seen.get(i, [])
            axes.append(
                AxisCoverage(
                    name=lattice.axis_names[i],
                    index=i,
                    layers=tuple(layer for layer, _ in hits),
                    forward=sum(1 for _, rev in hits if not rev),
                    backward=sum(1 for _, rev in hits if rev),
                )
            )
        return Coverage(n_layers=len(self.steps), axes=tuple(axes))

    # -- binding to a lattice -----------------------------------------------

    def resolve(self, lattice: Lattice, warn: bool = True) -> ScanPlan:
        """Return an equivalent plan with every axis as an integer index.

        Raises if an axis does not exist on ``lattice``. Warns — but does not
        fail — when the plan leaves an axis unswept: that is legal and
        occasionally intended, and usually a mistake.

        ``warn=False`` is for composition strategies that do not sweep at all.
        The joint (flatten) family mixes every axis in every layer and uses the
        plan only for its depth, so "never sweeps w" is both true and
        completely misleading there — a warning nobody can act on trains
        readers to ignore warnings.
        """
        resolved = [Step(lattice.axis_index(s.axis), s.reverse) for s in self.steps]
        touched = {s.axis for s in resolved}
        missing = [lattice.axis_names[i] for i in range(lattice.n_axes) if i not in touched]
        if missing and warn:
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
