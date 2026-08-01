"""Phase 2 acceptance for ScanPlan. See PLAN.md."""

import json
import math
import warnings

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


def test_bidirectional_accepts_an_explicit_axis_set():
    """Time stays causal while spatial axes get both directions — the reason
    bidirectionality is per-axis rather than a single flag."""
    plan = ScanPlan.cyclic(("time", "h", "w"), n_layers=12, bidirectional=("h", "w"))
    seen: dict[str, set[bool]] = {}
    for s in plan:
        seen.setdefault(s.axis, set()).add(s.reverse)
    assert seen["time"] == {False}
    assert seen["h"] == {False, True}
    assert seen["w"] == {False, True}


def test_bidirectional_accepts_a_bare_axis_name():
    """set('time') is {'t','i','m','e'}; a bare string must not be iterated."""
    plan = ScanPlan.cyclic(("time", "h"), n_layers=8, bidirectional="time")
    seen: dict[str, set[bool]] = {}
    for s in plan:
        seen.setdefault(s.axis, set()).add(s.reverse)
    assert seen["time"] == {False, True}
    assert seen["h"] == {False}


def test_bidirectional_rejects_axes_not_being_scanned():
    with pytest.raises(ValueError, match="not in"):
        ScanPlan.cyclic(("h", "w"), n_layers=4, bidirectional=("depth",))


def test_warns_when_layers_are_too_few_for_the_requested_bidirectionality():
    """Four axes in four layers gets each axis exactly one sweep, so no
    schedule can give any of them both directions. Say so rather than
    silently downgrading."""
    with pytest.warns(UserWarning, match="requested bidirectional"):
        ScanPlan.cyclic(("t", "s", "c", "f"), n_layers=4, bidirectional=True)


def test_no_warning_when_the_layer_budget_is_sufficient(recwarn):
    ScanPlan.cyclic(("t", "s", "c", "f"), n_layers=8, bidirectional=True)
    assert len(recwarn) == 0


def test_paired_matches_the_official_mamba_nd_schedule():
    """Upstream advances the ordering every two layers (z = i // 2) while
    flipping direction every layer, so each ordering runs once forward and
    once backward."""
    plan = ScanPlan.paired(("a", "b", "c"), n_layers=6)
    upstream = [("a", False), ("a", True), ("b", False), ("b", True), ("c", False), ("c", True)]
    assert [(s.axis, s.reverse) for s in plan] == upstream


def test_paired_gives_unpaired_axes_a_single_forward_layer():
    plan = ScanPlan.paired(("time", "h"), n_layers=6, bidirectional=("h",))
    assert [(s.axis, s.reverse) for s in plan] == [
        ("time", False),
        ("h", False),
        ("h", True),
        ("time", False),
        ("h", False),
        ("h", True),
    ]


def test_warnings_can_be_suppressed_for_deliberate_shallow_plans(recwarn):
    ScanPlan.cyclic(("a", "b", "c", "d"), n_layers=4, bidirectional=True, warn=False)
    assert len(recwarn) == 0


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


def test_a_plan_cannot_be_mutated_after_construction():
    """It is hashable, so it must be immutable. A mutated plan changes its own
    hash — silently losing it from any dict — and desyncs from the mixers a
    block already built one-per-step from it."""
    plan = ScanPlan.cyclic(("a", "b"), 4)
    with pytest.raises(AttributeError, match="immutable"):
        plan.steps = ()
    with pytest.raises(AttributeError, match="immutable"):
        del plan.steps
    with pytest.raises(AttributeError):
        plan.anything_else = 1


def test_a_plan_survives_use_as_a_dict_key():
    plan = ScanPlan.cyclic(("a", "b"), 4)
    store = {plan: "value"}
    assert store[ScanPlan.cyclic(("a", "b"), 4)] == "value"


# -- algebra ---------------------------------------------------------------


def test_plans_concatenate_and_repeat():
    a = ScanPlan.from_list([("h", False)])
    b = ScanPlan.from_list([("w", True)])
    assert (a + b).steps == (Step("h", False), Step("w", True))
    assert len(a * 3) == 3 and (a * 3).steps == a.steps * 3
    assert 2 * a == a * 2
    with pytest.raises(ValueError, match="at least once"):
        a * 0


def test_reversed_is_layer_order_and_flipped_is_direction():
    plan = ScanPlan.from_list([("h", False), ("w", True)])
    assert plan.reversed().steps == (Step("w", True), Step("h", False))
    assert plan.flipped().steps == (Step("h", True), Step("w", False))
    assert plan.flipped().flipped() == plan


def test_a_plan_plus_its_flip_makes_every_axis_bidirectional():
    """The composition identity the algebra exists for: whatever the schedule,
    `p + p.flipped()` covers both directions everywhere it swept at all."""
    lat = Lattice(shape=(4, 5), names=("h", "w"), time=True)
    plan = ScanPlan.cyclic(("time", "h", "w"), 3, warn=False)
    both = plan + plan.flipped()
    cov = both.coverage(lat)
    assert cov.unswept == () and cov.pinned == ()
    assert all(a.direction == "both" for a in cov.axes)


# -- coverage --------------------------------------------------------------


def test_coverage_counts_directions_and_names_the_untouched_axes():
    lat = Lattice(shape=(3, 4), names=("h", "w"), time=True)
    plan = ScanPlan.from_list([("time", False), ("h", False), ("h", True)])
    cov = plan.coverage(lat)
    assert cov.n_layers == 3
    assert [a.name for a in cov.axes] == ["time", "h", "w"]
    assert cov["h"].forward == 1 and cov["h"].backward == 1
    assert cov["h"].direction == "both" and cov["h"].layers == (1, 2)
    assert cov["time"].direction == "forward"
    assert cov["w"].direction == "none" and cov["w"].n_sweeps == 0
    assert cov.unswept == ("w",)
    assert cov.pinned == ("time",)
    assert cov.directions() == {"time": "forward", "h": "both"}


def test_coverage_reports_without_warning():
    """A report that warns cannot be used to decide whether to warn — the
    constructor's check and the viewer's spec both call this."""
    lat = Lattice(shape=(3, 4), names=("h", "w"))
    plan = ScanPlan.from_list([("h", False)])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cov = plan.coverage(lat)
    assert cov.unswept == ("w",)


def test_coverage_round_trips_to_plain_data():
    lat = Lattice(shape=(2, 2), names=("h", "w"))
    d = ScanPlan.cyclic(("h", "w"), 4, bidirectional=True).coverage(lat).to_dict()
    assert json.loads(json.dumps(d)) == d
    assert d["unswept"] == [] and d["pinned"] == []
    assert d["axes"][0]["direction"] == "both"
