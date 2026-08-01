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


def test_checks_run_only_at_ranks_the_caller_requested():
    """The gradient check hardcoded rank 2 "for speed", so ranks=(3, 4)
    gradchecked a rank-2 block the factory never claimed to support — and a
    factory valid only at its stated ranks failed a check it should pass."""

    def rank3_only(lat, d_model):
        assert lat.rank in (3, 4), f"built at unrequested rank {lat.rank}"
        plan = td.ScanPlan.cyclic(lat.axis_names, lat.n_axes)
        return td.AxialScan(
            mixer=lambda: torch.nn.Linear(d_model, d_model),
            plan=plan,
            lattice=lat,
            d_model=d_model,
        )

    report = td.testing.check_block(
        rank3_only,
        ranks=(3, 4),
        # A reference forces the equivalence check to decide; it must skip
        # (its claim is rank-1 and rank 1 was not requested), not build a
        # rank-1 lattice the factory refuses.
        reference=lambda block, x: block(x),
        raise_on_failure=False,
    )
    assert not report.failed, str(report)
    assert any("rank 1" in r.detail for r in report.skipped), str(report)


# -- high ranks --------------------------------------------------------------


@pytest.mark.parametrize("rank", [5, 6])
def test_the_scan_family_conforms_at_ranks_five_and_six(rank):
    """The machinery is rank-generic by construction; that is a claim, and this
    is the test of it. Ranks 5-6 were shipped untested for one release — the
    README said so — because "generic" felt like enough. It is not: a rank-6
    lattice is 5,040 cells and every axis bug that hides behind a lucky square
    shape at rank 2 has room to show here."""
    report = td.testing.check_block(rnn_factory(LSTM, n_layers=3), ranks=(rank,), d_model=2)
    assert report, str(report)


@pytest.mark.parametrize("rank", [5, 6])
def test_the_kernel_family_conforms_at_ranks_five_and_six(rank):
    def build(lat, d_model, plan=None):
        return td.AxialKernel(
            mixer=None,
            plan=plan or td.ScanPlan.cyclic(lat.axis_names, len(lat.axis_names)),
            lattice=lat,
            d_model=d_model,
        )

    report = td.testing.check_block(build, ranks=(rank,), d_model=2)
    assert report, str(report)


def test_a_rank_five_model_trains():
    """Conformance says the plumbing is right; this says the thing still learns
    when five axes have to be reached through three layers."""
    lat = td.Lattice(shape=(2, 2, 3, 2, 2), names=("a", "b", "c", "d", "e"))
    model = td.LSTM(d_model=8, n_layers=5, lattice=lat, d_input=1)
    x = torch.randn(4, *lat.shape, 1)
    y = x.cumsum(dim=lat.tensor_dim("c"))
    head = nn.Linear(8, 1)
    opt = torch.optim.Adam([*model.parameters(), *head.parameters()], lr=3e-2)
    first = last = None
    for step in range(60):
        loss = (head(model(x)) - y).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        first = float(loss.detach()) if step == 0 else first
        last = float(loss.detach())
    assert last < first * 0.25, f"rank-5 model did not learn: {first:.4f} -> {last:.4f}"


# -- the debug mixer and the source checker ----------------------------------


def test_recorder_answers_which_axis_each_layer_swept():
    """ "Which axis did layer 3 actually sweep" is the first question every
    integration bug asks; this is the tool that answers it."""
    lat = td.Lattice(shape=(3, 7), names=("h", "w"), time=True)
    plan = td.ScanPlan.from_list([("h", False), ("w", True), ("time", False)])
    model = td.LSTM(4, lattice=lat, plan=plan, mixer=td.testing.Recorder)
    x = torch.randn(2, 5, 3, 7, 4)
    out = model(x)

    assert out.shape == x.shape
    lengths = [m.calls[0].length for m in model.nd.mixers]
    assert lengths == [3, 7, 5], f"layers swept axes of length {lengths}"
    # the folded batch is everything else
    assert model.nd.mixers[0].calls[0].lines == 2 * 5 * 7
    model.nd.mixers[0].reset()
    assert model.nd.mixers[0].calls == []


def test_recorder_leaves_the_data_alone():
    """It has to be the identity, or it cannot be dropped into a real model to
    ask a question about that model."""
    lat = td.Lattice(shape=(4,), names=("a",))
    rec = td.testing.Recorder(6)
    x = torch.randn(3, 4, 6)
    assert torch.equal(rec(x), x)
    assert lat.rank == 1


def test_check_data_source_accepts_the_shipped_source():
    source = td.data.TensorSource(
        torch.randn(9, 3, 4, 2), td.Lattice(shape=(3, 4), names=("h", "w"))
    )
    report = td.testing.check_data_source(source)
    assert report, str(report)


def test_check_data_source_catches_a_source_that_lies_about_its_shape():
    class Liar:
        lattice = td.Lattice(shape=(3, 4), names=("h", "w"))

        def __len__(self):
            return 5

        def __getitem__(self, index):
            return torch.randn(3, 9, 9, 2)  # not the declared lattice

    report = td.testing.check_data_source(Liar(), raise_on_failure=False)
    assert not report
    assert any("declared lattice" in r.name for r in report.failed)


def test_check_data_source_catches_a_source_that_cannot_reach_a_worker():
    """DEBUG.md #9: an unpicklable source does not raise under
    DataLoader(num_workers>0) — it hangs. Finding it here is the whole point."""

    class Handle:
        lattice = td.Lattice(shape=(2,), names=("a",))

        def __init__(self):
            self.data = torch.randn(4, 2, 1)
            self.lock = __import__("threading").Lock()  # unpicklable

        def __len__(self):
            return 4

        def __getitem__(self, index):
            return self.data[index]

    report = td.testing.check_data_source(Handle(), raise_on_failure=False)
    assert not report
    assert any("worker" in r.name for r in report.failed)
