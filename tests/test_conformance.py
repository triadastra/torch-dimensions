"""Phase 4 acceptance for td.testing.check_block. See PLAN.md.

Half of this file is deliberately broken blocks. A conformance suite that only
ever passes proves nothing, so every check gets a block built to violate it and
nothing else.
"""

import pytest
import torch
import torch.nn as nn

import torch_dimensions as td
from torch_dimensions import GRU, LSTM, Lattice, axial_apply


def rnn_factory(cls, n_layers=None):
    def build(lat, d_model, plan=None):
        return cls(d_model, n_layers or len(lat.axis_names), lat, plan=plan)

    return build


def one_layer_reference(block, x):
    """Pre-norm residual around the single mixer — what a 1-layer stack is."""
    return x + block.nd.mixers[0].rnn(block.nd.norms[0](x))[0]


def status(report, fragment):
    for r in report.results:
        if fragment in r.name:
            return r.status
    raise KeyError(f"no check named ~{fragment!r} in:\n{report}")


# -- the library's own blocks conform ----------------------------------------


@pytest.mark.parametrize("cls", [LSTM, GRU])
def test_rnn_family_passes_every_applicable_check(cls):
    report = td.testing.check_block(rnn_factory(cls))
    assert report, str(report)
    assert not report.failed


def test_rank_one_equivalence_passes_when_a_reference_is_given():
    report = td.testing.check_block(rnn_factory(LSTM), reference=one_layer_reference)
    assert status(report, "rank-1") == "pass", str(report)


def test_checks_that_cannot_run_are_reported_as_skipped_not_passed():
    """A skipped check must never read as a passing one."""
    report = td.testing.check_block(rnn_factory(LSTM))
    assert status(report, "rank-1") == "skip"  # no reference supplied
    assert status(report, "Kronecker") == "skip"  # Phase 6
    assert status(report, "compile") == "skip"  # off by default
    assert report, "skips must not make the report falsy"


def test_covariance_is_skipped_when_the_factory_cannot_fix_the_plan():
    def no_plan_arg(lat, d_model):
        return LSTM(d_model, len(lat.axis_names), lat)

    report = td.testing.check_block(no_plan_arg)
    assert status(report, "covariant") == "skip"


# -- deliberately broken blocks ----------------------------------------------


class _Base(nn.Module):
    def __init__(self, lat, d_model):
        super().__init__()
        self.lat, self.lin = lat, nn.Linear(d_model, d_model)


class ShapeBreaker(_Base):
    def forward(self, x):
        return self.lin(x)[..., :1]


class LeakyMask(_Base):
    """Scans without zeroing absent cells, so their values ride the recurrence
    into present ones."""

    def forward(self, x):
        return axial_apply(self.lin(x), self.lat, 0, lambda s: s.cumsum(1))


class HardcodedAxis(_Base):
    """Always sweeps tensor dim 1 instead of resolving the axis, so the result
    depends on which axis happens to be stored first."""

    def forward(self, x):
        return self.lin(x).cumsum(dim=1)


class DeadParameter(_Base):
    def __init__(self, lat, d_model):
        super().__init__(lat, d_model)
        self.unused = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        return self.lin(x)


def _factory(cls):
    def build(lat, d_model, plan=None):
        return cls(lat, d_model)

    return build


def test_shape_check_catches_a_block_that_changes_width():
    report = td.testing.check_block(_factory(ShapeBreaker), raise_on_failure=False)
    assert status(report, "shape") == "fail"
    assert not report


def test_mask_check_catches_absent_cells_leaking_into_the_output():
    report = td.testing.check_block(_factory(LeakyMask), raise_on_failure=False)
    assert status(report, "absent cells") == "fail"


def test_covariance_check_catches_a_hardcoded_axis():
    report = td.testing.check_block(_factory(HardcodedAxis), ranks=(3,), raise_on_failure=False)
    assert status(report, "covariant") == "fail"


def test_gradient_check_catches_a_parameter_that_never_receives_grad():
    report = td.testing.check_block(_factory(DeadParameter), raise_on_failure=False)
    assert status(report, "gradients") == "fail"
    assert "unused" in [r for r in report.results if "gradients" in r.name][0].detail


def test_equivalence_check_catches_a_wrong_rank_one_result():
    report = td.testing.check_block(
        rnn_factory(LSTM),
        reference=lambda block, x: torch.zeros_like(x),
        raise_on_failure=False,
    )
    assert status(report, "rank-1") == "fail"


# -- reporting ---------------------------------------------------------------


def test_failures_raise_by_default_with_the_report_attached():
    with pytest.raises(AssertionError, match="conformance check failed"):
        td.testing.check_block(_factory(ShapeBreaker))


def test_report_renders_every_check():
    text = str(td.testing.check_block(rnn_factory(LSTM)))
    assert text.count("\n") == 6  # seven checks
    assert "[  ok]" in text and "[skip]" in text


def test_report_is_falsy_only_when_something_failed():
    assert td.testing.check_block(rnn_factory(LSTM))
    assert not td.testing.check_block(_factory(ShapeBreaker), raise_on_failure=False)


def test_compile_check_runs_when_asked():
    report = td.testing.check_block(
        rnn_factory(LSTM, n_layers=1), ranks=(2,), check_compile=True, raise_on_failure=False
    )
    assert status(report, "compile") in {"pass", "fail"}


def test_sparse_can_be_turned_off_for_blocks_that_do_not_support_it():
    report = td.testing.check_block(rnn_factory(LSTM), sparse=False)
    assert status(report, "absent cells") == "skip"


def test_time_axis_lattices_are_supported():
    report = td.testing.check_block(rnn_factory(LSTM), time=True, ranks=(1, 2))
    assert report, str(report)


def test_custom_nd_method_can_be_checked():
    """The point of shipping this: a user's own strategy gets the same checks."""

    def reversed_sweep(mixer, plan, lattice, d_model, **kw):
        flipped = td.ScanPlan.from_list([(s.axis, not s.reverse) for s in plan])
        return td.AxialScan(mixer=mixer, plan=flipped, lattice=lattice, d_model=d_model, **kw)

    def build(lat, d_model, plan=None):
        return LSTM(d_model, len(lat.axis_names), lat, plan=plan, nd_method=reversed_sweep)

    assert td.testing.check_block(build)


def test_lattice_helper_builds_a_genuinely_sparse_mask():
    lat = td.testing._lattice(3, sparse=True, seed=0)
    assert 0 < lat.n_valid < lat.n_cells
    assert isinstance(lat, Lattice)
