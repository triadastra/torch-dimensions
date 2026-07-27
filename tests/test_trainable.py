"""The models must actually train, not merely have correct gradients.

"No trainer in the library" is a scope decision. It must not quietly become
"nobody ever checked that it converges" — a block can pass gradcheck and still
never learn. These tests are the guard against that reading.
"""

import warnings

import pytest
import torch
import torch.nn as nn

import torch_dimensions as td


def factory(cls, plan=None):
    def build(lat, d_model):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return cls(d_model, len(lat.axis_names), lat, plan=plan)

    return build


@pytest.mark.parametrize("cls", [td.LSTM, td.GRU])
def test_the_rnn_family_learns_a_task_that_needs_axial_mixing(cls):
    result = td.testing.check_trainable(factory(cls), steps=150)
    assert result["ratio"] >= 3.0, result
    assert result["held_out"] < result["initial"]


def test_a_model_that_never_sweeps_the_needed_axis_cannot_learn_it():
    """The negative that makes the positive mean something. The task is a
    cumulative sum along `w`; a plan that only sweeps time has no path for
    information to travel along `w` at all."""
    blind = factory(td.LSTM, plan=td.ScanPlan.from_list(["time", "time"]))
    result = td.testing.check_trainable(blind, steps=150, raise_on_failure=False)
    assert result["ratio"] < 3.0, f"blind model should not solve this: {result}"


def test_check_trainable_raises_with_an_actionable_message():
    blind = factory(td.LSTM, plan=td.ScanPlan.from_list(["time", "time"]))
    with pytest.raises(AssertionError, match="did not learn"):
        td.testing.check_trainable(blind, steps=100)


def test_check_trainable_reports_the_numbers_it_judged_on():
    result = td.testing.check_trainable(factory(td.LSTM), steps=50, min_ratio=0.0)
    assert set(result) == {"initial", "final", "held_out", "ratio"}
    assert all(isinstance(v, float) for v in result.values())


def test_a_sparse_lattice_model_still_learns():
    """Masking absent cells must not sever the gradient path for present ones."""
    valid = torch.tensor(
        [[True, True, True, False], [True, True, True, True], [True, False, True, True]]
    )
    lat = td.Lattice(shape=(3, 4), names=("h", "w"), valid=valid, time=True)
    torch.manual_seed(0)
    model, head = td.LSTM(16, 3, lat), nn.Linear(16, 1)
    opt = torch.optim.Adam([*model.parameters(), *head.parameters()], lr=1e-2)
    g = torch.Generator().manual_seed(0)

    def draw():
        x = torch.randn(8, 5, 3, 4, 16, generator=g)
        return x, x[..., :1].cumsum(dim=3) * lat.mask().to(x.dtype)

    first = None
    for _ in range(150):
        x, y = draw()
        loss = (head(model(x)) * lat.mask().to(x.dtype) - y).pow(2).mean()
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first / 3, f"{first:.4f} -> {loss.item():.4f}"


def test_training_updates_every_parameter():
    """A parameter that never moves is dead weight the gradient check would
    not catch, since it can receive a gradient of exactly zero forever."""
    lat = td.Lattice(shape=(2, 3), time=True)
    torch.manual_seed(0)
    model = td.LSTM(8, 3, lat)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(5):
        loss = model(torch.randn(4, 3, 2, 3, 8)).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    unmoved = [n for n, p in model.named_parameters() if torch.equal(p, before[n])]
    assert not unmoved, unmoved
