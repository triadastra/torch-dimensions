"""The starting weights a cross-device comparison actually shares.

`pretrain.py` and `agreement.py` both said, in a docstring, that building on
CPU under a fixed seed gives bit-identical weights on any machine, and every
number they produced rested on it. For most models it is true — an LSTM built
that way hashes identically on macOS and on Linux.

For S4 and S4D it is false, and not because of a bug. `hippo.nplr` diagonalises
the HiPPO matrix with `torch.linalg.eigh`. Eigen*values* are unique and matched
across the two platforms to every digit printed, which is why `A_imag` looked
fine. Eigen*vectors* are fixed only up to a phase, and macOS Accelerate and
Linux LAPACK are each free to return a different one. `B` and `P` are
projections through those vectors, so they inherit it: measured across a Mac
Studio and an RTX 5090 box, `B` differed by a relative 1.5 and `P` by 0.53
while `A_imag` was identical.

The comparison was therefore reporting a 2.6e-01 output difference for the
vendored S4D — unchanged in float64, which reads exactly like a different
kernel — when the two machines had simply built two different models. With the
weights shared, the same pair agrees at 4e-07.

A test on one machine cannot observe a cross-platform difference. What it can
do is pin the mechanism that now carries the assumption: that weights written
by one run are what a later run gets, exactly, in place of whatever the seed
would have produced.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import torch_dimensions as td

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))

spec = importlib.util.spec_from_file_location(
    "_td_init_weights", ROOT / "benchmarks" / "init_weights.py"
)
init_weights = importlib.util.module_from_spec(spec)
sys.modules["_td_init_weights"] = init_weights
spec.loader.exec_module(init_weights)


LAT = td.Lattice(shape=(4, 5), names=("h", "w"), time=True)


def test_the_first_run_writes_and_the_second_loads(tmp_path):
    a, b = nn.Linear(8, 8), nn.Linear(8, 8)
    assert init_weights.sync(a, tmp_path, "m") == "written"
    assert init_weights.sync(b, tmp_path, "m") == "loaded"
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        assert torch.equal(pa, pb)


def test_loading_overrides_whatever_the_seed_produced(tmp_path):
    """The point of the mechanism: the loaded values win over construction.

    If `sync` returned "loaded" while leaving the model on its own weights,
    every comparison would still be measuring initialisation drift and would
    still report a plausible-looking number.
    """
    torch.manual_seed(0)
    reference = nn.Linear(8, 8)
    init_weights.sync(reference, tmp_path, "m")

    torch.manual_seed(999)  # deliberately a different draw
    other = nn.Linear(8, 8)
    assert not torch.equal(other.weight, reference.weight)

    assert init_weights.sync(other, tmp_path, "m") == "loaded"
    assert torch.equal(other.weight, reference.weight)
    assert torch.equal(other.bias, reference.bias)


def test_no_store_means_no_change_and_says_so(tmp_path):
    """`--init` is opt-in; without it the behaviour is exactly what it was."""
    torch.manual_seed(3)
    model = nn.Linear(8, 8)
    before = model.weight.detach().clone()
    assert init_weights.sync(model, None, "m") == "seed"
    assert torch.equal(model.weight, before)


def test_every_parameter_and_buffer_round_trips_for_a_real_model(tmp_path):
    """A `state_dict` is not just parameters. S4's kernel keeps buffers, and a
    mechanism that restored parameters while leaving buffers to the seed would
    reintroduce the bug in the exact place it came from."""
    pytest.importorskip("einops", reason="the vendored S4 needs the [upstream] extra")
    pytest.importorskip("hydra", reason="the s4 pipeline needs hydra-core")

    torch.manual_seed(0)
    first = td.S4(32, 2, LAT, d_input=1, d_state=16)
    init_weights.sync(first, tmp_path, "s4")

    torch.manual_seed(1)
    second = td.S4(32, 2, LAT, d_input=1, d_state=16)
    assert init_weights.sync(second, tmp_path, "s4") == "loaded"

    sd_a, sd_b = first.state_dict(), second.state_dict()
    assert set(sd_a) == set(sd_b)
    for key in sd_a:
        assert torch.equal(sd_a[key], sd_b[key]), f"{key} did not round trip"


def test_shared_weights_make_two_builds_agree_where_the_seed_would_not(tmp_path):
    """End to end, in the shape the benchmark uses it: build, sync, and the two
    models compute the same thing. On one machine the seed would also have
    achieved this — the value of the test is that it fails loudly if `sync`
    ever stops applying, which is the failure that hid for a whole run."""
    pytest.importorskip("einops", reason="the vendored S4 needs the [upstream] extra")
    pytest.importorskip("hydra", reason="the s4 pipeline needs hydra-core")

    torch.manual_seed(0)
    a = td.S4D(32, 2, LAT, d_input=1, d_state=16).eval()
    init_weights.sync(a, tmp_path, "s4d")
    torch.manual_seed(7)
    b = td.S4D(32, 2, LAT, d_input=1, d_state=16).eval()
    init_weights.sync(b, tmp_path, "s4d")

    x = torch.randn(2, 3, *LAT.shape, 1)
    with torch.no_grad():
        assert torch.equal(a(x), b(x))


def test_the_eigenvalues_are_the_reproducible_part(tmp_path):
    """Why the bug was invisible: the part of the decomposition that *is*
    unique agrees, so `A_imag` matched across platforms to twelve decimals and
    the initialisation looked sound. Pinned here so the diagnosis stays
    attached to the code it explains."""
    pytest.importorskip("einops", reason="the vendored S4 needs the [upstream] extra")

    from torch_dimensions._vendor.s4.src.models.hippo.hippo import nplr

    w1, p1, b1, v1 = nplr("legs", 32)
    w2, p2, b2, v2 = nplr("legs", 32)
    # Same machine, so everything repeats; the eigenvalues are the only part
    # that also repeats across machines.
    assert torch.allclose(w1, w2)
    assert w1.is_complex(), "the eigenvalues are the spectrum of the HiPPO matrix"
