"""The reproductions, at a size CI can afford.

A reproduction script that nobody runs between releases stops working quietly:
an API changes, a shape drifts, and the failure surfaces months later when
somebody tries to reproduce a number. These tests build every reproduction's
*model* and train it for a handful of steps on synthetic data of the right
shape, asserting only that the thing learns something. No downloads, no
accuracy claims — the numbers live in RESULTS.md and come from real runs.

The synthetic stand-in is deliberate: PLAN.md's Phase 9 risk note says dataset
download flakiness must not be able to turn CI red.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

repro = pytest.importorskip("examples.repro.image_nd")
from examples.repro import forecast_sparse, harness, image_nd, smnist  # noqa: E402


def fit(model, batches, lr=3e-3, steps=40):
    """Train briefly; return (first loss, last loss)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    first = last = None
    for i in range(steps):
        x, y = batches[i % len(batches)]
        loss = (
            nn.functional.mse_loss(model(x), y)
            if y.dtype.is_floating_point
            else (nn.functional.cross_entropy(model(x), y))
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        first = float(loss.detach()) if i == 0 else first
        last = float(loss.detach())
    return first, last


def test_the_sequence_classifier_learns():
    """sMNIST's model on a short synthetic sequence task: the class is decided
    by an early token, so a model that cannot carry information forward fails."""
    torch.manual_seed(0)
    model = smnist.SequenceClassifier(d_model=16, n_layers=2, n_classes=2, d_state=8)
    x = torch.randn(16, 32, 1)
    y = (x[:, 0, 0] > 0).long()
    x[:, 0, 0] = 0.0  # the cue is removed from the input it was derived from...
    x[:, 2, 0] = torch.where(y == 1, 3.0, -3.0)  # ...and planted at step 2
    first, last = fit(model, [(x, y)], steps=60)
    assert last < first * 0.6, f"sequence classifier did not learn: {first:.3f} -> {last:.3f}"


@pytest.mark.parametrize("arch", ["mamba_nd", "s4d_nd", "lstm"])
def test_every_image_model_builds_and_learns(arch):
    torch.manual_seed(0)
    model = image_nd.build_model(arch, (4, 4), 1, d_model=16, n_layers=2, n_classes=2)
    x = torch.randn(16, 4, 4, 1)
    y = (x[:, 0, 0, 0] > 0).long()
    first, last = fit(model, [(x, y)], steps=50)
    assert last < first * 0.8, f"{arch} did not learn: {first:.3f} -> {last:.3f}"


@pytest.mark.parametrize("arm", sorted(forecast_sparse.ARMS))
def test_every_forecasting_arm_builds_and_learns(arm):
    """Each arm differs in one thing; all four must at least run and improve,
    or a comparison between them is comparing one working model to one broken
    one."""
    torch.manual_seed(0)
    keep = torch.ones(3, 4, dtype=torch.bool)
    keep[0, 0] = keep[2, 3] = False
    prep = {"shape": (3, 4), "names": ("station", "pollutant"), "keep": keep}
    model = forecast_sparse.build_arm(arm, prep, d_model=16, n_layers=2)
    x = torch.randn(4, 6, 3, 4, 1)
    y = x[:, -1:] * 0.5  # a target the model can reach from its own last input
    first, last = fit(model, [(x, y)], steps=40)
    assert last < first * 0.9, f"arm {arm} did not learn: {first:.4f} -> {last:.4f}"


def test_absent_cells_stay_absent_through_the_masked_arm():
    """The sparse arm's whole claim: values in absent cells cannot reach the
    output. If this fails, the `masked` vs `zeros` comparison measures noise."""
    torch.manual_seed(0)
    keep = torch.ones(3, 4, dtype=torch.bool)
    keep[1, 2] = False
    prep = {"shape": (3, 4), "names": ("station", "pollutant"), "keep": keep}
    model = forecast_sparse.build_arm("masked", prep, d_model=8, n_layers=2).eval()
    x = torch.randn(2, 5, 3, 4, 1)
    mask = keep.reshape(1, 1, 3, 4, 1).float()
    noisy = x.clone()
    noisy[:, :, 1, 2] += 1e3
    with torch.no_grad():
        assert torch.equal(model(x * mask), model(noisy * mask))


def test_the_ledger_row_names_its_own_metric():
    """A results row whose number has no name is a number nobody can check."""
    result = harness.Result(
        config={"task": "t", "model": "m", "epochs": 1, "seed": 0},
        metric=0.9912,
        metric_name="test_acc",
        train_loss=0.01,
        epochs_run=1,
        seconds=60.0,
        machine={"accelerator": "cpu"},
        history=[],
    )
    assert "99.12%" in result.row() and "test_acc" in result.row()
