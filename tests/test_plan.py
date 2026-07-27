"""Phase 2 acceptance for ScanPlan. See PLAN.md."""

import math

import pytest

from torch_dimensions import Lattice, ScanPlan, Step

# -- construction -----------------------------------------------------------


def test_cyclic_visits_axes_in_order():
    plan = ScanPlan.cyclic(("a", "b", "c"), n_layers=5)
    assert [s.axis for s in plan] == ["a", "b", "c", "a", "b"]
    assert not any(s.reverse for s in plan)


@pytest.mark.parametrize("n_axes", [1, 2, 3, 4])
@pytest.mark.parametrize("n_layers", [1, 2, 5, 12, 13])
def test_cyclic_spreads_layers_evenly(n_axes, n_layers):
    axes = tuple(f"a{i}" for i in range(n_axes))
    counts = ScanPlan.cyclic(axes, n_layers).counts()
    lo, hi = math.floor(n_layers / n_axes), math.ceil(n_layers / n_axes)
    assert sum(counts.values()) == n_layers
    for a in axes[: min(n_axes, n_layers)]:
        assert lo <= counts[a] <= hi


@pytest.mark.parametrize("n_axes", [1, 2, 3, 4])
def test_bidirectional_cyclic_gives_every_axis_both_directions(n_axes):
    """Flipping direction per *layer* rather than per *cycle* aliases against
    an even axis count and silently pins each axis to one direction. This is
    the test that catches it."""
    axes = tuple(f"a{i}" for i in range(n_axes))
    plan = ScanPlan.cyclic(axes, n_layers=4 * n_axes, bidirectional=True)
    seen: dict[str, set[bool]] = {a: set() for a in axes}
    for s in plan:
        seen[s.axis].add(s.reverse)
    for a in axes:
        assert seen[a] == {False, True}, f"{a} only ever swept one way in {plan}"


def test_cyclic_without_bidirectional_is_all_forward():
    plan = ScanPlan.cyclic(("a", "b"), n_layers=8)
    assert not any(s.reverse for s in plan)


def test_paired_alternates_within_adjacent_layers():
    plan = ScanPlan.paired(("a", "b"), n_layers=6)
    assert list(plan) == [
        Step("a", False),
        Step("a", True),
        Step("b", False),
        Step("b", True),
        Step("a", False),
        Step("a", True),
    ]


def test_from_list_accepts_steps_pairs_and_bare_axes():
    assert ScanPlan.from_list([Step("a", True)]) == ScanPlan.from_list([("a", True)])
    assert ScanPlan.from_list(["a", "b"]) == ScanPlan.from_list([("a", False), ("b", False)])


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: ScanPlan.cyclic((), 3), "at least one axis"),
        (lambda: ScanPlan.cyclic(("a", "a"), 3), "distinct"),
        (lambda: ScanPlan.cyclic(("a",), 0), "n_layers"),
        (lambda: ScanPlan.from_list([]), "at least one step"),
        (lambda: ScanPlan.from_list([("a", True, 1)]), "axis, reverse"),
    ],
)
def test_construction_errors(call, match):
    with pytest.raises(ValueError, match=match):
        call()


# -- inspection -------------------------------------------------------------


def test_axes_are_distinct_and_in_first_appearance_order():
    plan = ScanPlan.from_list(["c", "a", "c", "b"])
    assert plan.axes == ("c", "a", "b")


def test_counts_and_len():
    plan = ScanPlan.cyclic(("a", "b"), n_layers=5)
    assert len(plan) == 5
    assert plan.counts() == {"a": 3, "b": 2}


def test_equality_and_hashing_ignore_construction_route():
    a = ScanPlan.cyclic(("x", "y"), n_layers=2)
    b = ScanPlan.from_list([("x", False), ("y", False)])
    assert a == b and hash(a) == hash(b)
    assert a != ScanPlan.from_list([("y", False), ("x", False)])
    assert a != "not a plan"


def test_repr_shows_axis_and_direction():
    assert repr(ScanPlan.from_list([("h", False), ("w", True)])) == "ScanPlan(h+, w-)"


# -- resolution against a lattice -------------------------------------------


def test_resolve_maps_names_to_indices():
    lat = Lattice(shape=(3, 4), names=("h", "w"), time=True)
    plan = ScanPlan.cyclic(("time", "h", "w"), n_layers=3).resolve(lat)
    assert plan.is_resolved()
    assert [s.axis for s in plan] == [0, 1, 2]


def test_resolve_is_idempotent():
    lat = Lattice(shape=(3, 4), names=("h", "w"))
    once = ScanPlan.cyclic(("h", "w"), 4).resolve(lat)
    assert once.resolve(lat) == once


def test_resolve_rejects_unknown_axes():
    lat = Lattice(shape=(3, 4), names=("h", "w"))
    with pytest.raises(KeyError, match="depth"):
        ScanPlan.from_list(["depth"]).resolve(lat)


def test_resolve_warns_when_an_axis_is_never_swept():
    lat = Lattice(shape=(3, 4), names=("h", "w"))
    with pytest.warns(UserWarning, match=r"never sweeps \['w'\]"):
        ScanPlan.from_list(["h"]).resolve(lat)


def test_resolve_is_silent_when_every_axis_is_covered(recwarn):
    lat = Lattice(shape=(3, 4), names=("h", "w"), time=True)
    ScanPlan.cyclic(("time", "h", "w"), n_layers=6).resolve(lat)
    assert len(recwarn) == 0


def test_unresolved_plan_reports_itself_as_such():
    assert not ScanPlan.cyclic(("h", "w"), 2).is_resolved()
    assert ScanPlan.from_list([0, 1]).is_resolved()


# -- serialization ----------------------------------------------------------


@pytest.mark.parametrize(
    "plan",
    [
        ScanPlan.cyclic(("a", "b", "c"), n_layers=7, bidirectional=True),
        ScanPlan.paired(("h", "w"), n_layers=4),
        ScanPlan.from_list([(0, True), (1, False)]),
    ],
)
def test_round_trips_through_a_dict(plan):
    assert ScanPlan.from_dict(plan.to_dict()) == plan


def test_to_dict_is_plain_json_types():
    d = ScanPlan.cyclic(("a",), 2, bidirectional=True).to_dict()
    assert d == {"steps": [["a", False], ["a", True]]}


def test_from_dict_needs_a_steps_key():
    with pytest.raises(KeyError, match="steps"):
        ScanPlan.from_dict({"layers": []})
