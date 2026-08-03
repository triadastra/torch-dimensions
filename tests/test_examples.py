"""The documented examples, executed.

Every code block in `docs/adding-a-mixer.md` and `docs/adding-a-method.md`
comes from these two files. A guide whose example silently stopped working is
worse than no guide, because the reader assumes the failure is theirs.

These are slower than a unit test — they run the full conformance suite twice
and train a small model — and that is the cost of the guides being true.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from examples import custom_method, custom_mixer  # noqa: E402

import torch_dimensions as td  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_the_method_registry():
    """`register_nd_method` writes to a process-global dict, and the guide in
    docs/adding-a-method.md really does register "pyramid". Without this, every
    test that runs after this file sees a registry the library did not ship —
    which is exactly how it was found: tests/test_matrix.py asserts what the
    registry contains and failed only in a full-suite run.
    """
    before = dict(td.ND_METHODS)
    yield
    td.ND_METHODS.clear()
    td.ND_METHODS.update(before)


def test_the_custom_mixer_guide_runs_end_to_end():
    custom_mixer.run_conformance()
    custom_mixer.use_it()


def test_the_custom_method_guide_runs_end_to_end():
    custom_method.main()


def test_the_example_mixer_is_a_real_sequence_model():
    """An EMA whose decay is learnable must actually depend on order — a mixer
    that ignores position would pass every shape check and teach the reader
    nothing."""
    mixer = custom_mixer.EMAMixer(d_model=4).eval()
    x = torch.randn(2, 6, 4)
    with torch.no_grad():
        forward = mixer(x)
        reversed_ = mixer(x.flip(1)).flip(1)
    assert not torch.allclose(forward, reversed_), "the example mixer is order-blind"


def test_the_example_strategy_prioritizes_the_axis_it_claims_to():
    """The guide says pyramid sweeps the largest axis most. If that stops being
    true the guide is teaching a lie, whatever the conformance report says."""
    lattice = td.Lattice(shape=(3, 12), names=("row", "col"), time=True)
    model = td.LSTM(16, 8, lattice, method=custom_method.pyramid, d_input=1)
    cov = model.nd.plan.coverage(lattice)
    assert cov["col"].n_sweeps > cov["row"].n_sweeps
    assert cov["time"].backward == 0, "time must stay causal"
    assert not cov.unswept, f"axes left unswept: {cov.unswept}"


def test_the_storage_covariance_bug_the_guide_describes_is_real():
    """The guide claims ordering axes by `lattice.axis_names` breaks covariance.
    This reproduces the broken version, so the claim in the prose is checked
    rather than remembered."""

    def by_storage_order(mixer, plan, lattice, d_model, **kwargs):
        order = list(lattice.axis_names)
        steps = [td.Step(order[i % len(order)], False) for i in range(len(plan))]
        return td.axial_scan(mixer, td.ScanPlan.from_list(steps), lattice, d_model, **kwargs)

    def factory(lat, d_model, plan=None):
        return td.LSTM(d_model, len(lat.axis_names), lat, plan=plan, method=by_storage_order)

    report = td.testing.check_block(factory, ranks=(3,), raise_on_failure=False)
    covariance = next(r for r in report.results if "covariant" in r.name)
    assert covariance.status == "fail", (
        "the storage-order strategy passed the covariance check, so either the "
        "check or the guide's explanation of it is wrong"
    )
