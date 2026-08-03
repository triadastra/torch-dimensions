"""Parameter groups that honour what the upstream models ask for.

`A` controls how fast a state decays and `dt` its timescale, both inside an
exponential — so weight decay on them is not a mild regulariser but a change
to the dynamics, and a learning rate suited to a projection walks them out of
the stable region. Both upstream repos say so in the parameters themselves
(`_optim` in s4's kernel, `_no_weight_decay` on Mamba's `A_log`/`D`/`dt_bias`),
and those tags do nothing unless an optimizer reads them.

Which means the obvious line — `AdamW(model.parameters(), lr=...)` — trains an
S4 the way its authors avoid, silently. These tests pin the reading of the
tags, because the failure mode is a model that trains, converges to something,
and is misconfigured the whole way.
"""

from __future__ import annotations

import pytest
import torch

import torch_dimensions as td

pytest.importorskip("einops", reason="the tagged parameters live in the vendored code")
pytest.importorskip("hydra", reason="the s4 pipeline needs hydra-core")

LAT = td.Lattice(shape=(4, 5), names=("h", "w"), time=True)


def _all_params(groups):
    return [p for g in groups for p in g["params"]]


def test_every_parameter_lands_in_exactly_one_group():
    """A parameter in two groups is stepped twice; one in none is frozen
    without anyone saying so."""
    model = td.S4(32, 2, LAT, d_input=1, d_state=16)
    groups = td.param_groups(model, lr=1e-3)
    listed = _all_params(groups)
    ids = [id(p) for p in listed]
    assert len(ids) == len(set(ids)), "a parameter appears in more than one group"
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    assert set(ids) == expected, "a parameter was dropped or invented"


def test_s4_kernel_parameters_keep_the_settings_upstream_gave_them():
    """s4 attaches `_optim` to its SSM kernel parameters. Those settings are
    the authors', not ours, and must survive verbatim."""
    model = td.S4(32, 2, LAT, d_input=1, d_state=16)
    tagged = [p for p in model.parameters() if getattr(p, "_optim", None)]
    assert tagged, "the vendored s4 kernel no longer tags its parameters"

    groups = td.param_groups(model, lr=1e-2, weight_decay=0.1)
    for param in tagged:
        group = next(g for g in groups if any(p is param for p in g["params"]))
        assert group["weight_decay"] == 0.0
        assert group["lr"] <= td.optim.SSM_MAX_LR


def test_mamba_no_weight_decay_parameters_get_none():
    """`A_log`, `D` and `dt_bias` carry `_no_weight_decay`. Decay on `A_log`
    pulls every state toward the same timescale, which is the opposite of what
    a bank of states is for."""
    model = td.Mamba(32, 2, LAT, d_input=1, d_state=8)
    tagged = [p for p in model.parameters() if getattr(p, "_no_weight_decay", False)]
    assert tagged, "the vendored mamba block no longer tags its parameters"

    groups = td.param_groups(model, lr=1e-3, weight_decay=0.1)
    for param in tagged:
        group = next(g for g in groups if any(p is param for p in g["params"]))
        assert group["weight_decay"] == 0.0


def test_the_ssm_rate_is_capped_not_raised():
    """Upstream fixes the SSM rate at 1e-3. A caller asking for 1e-2 gets it
    for ordinary weights and 1e-3 for the SSM; a caller asking for 1e-5 keeps
    1e-5 everywhere rather than having it raised to the cap."""
    model = td.S4(32, 1, LAT, d_input=1, d_state=16)

    high = td.param_groups(model, lr=1e-2)
    assert max(g["lr"] for g in high) == pytest.approx(1e-2)
    assert min(g["lr"] for g in high) == pytest.approx(1e-3)

    low = td.param_groups(model, lr=1e-5)
    assert all(g["lr"] == pytest.approx(1e-5) for g in low)


def test_norms_and_biases_are_excluded_from_decay_by_default():
    model = td.LSTM(32, 2, LAT, d_input=1)
    groups = td.param_groups(model, lr=1e-3, weight_decay=0.1)
    for group in groups:
        if group["weight_decay"] > 0:
            assert all(p.ndim > 1 for p in group["params"])

    including = td.param_groups(model, lr=1e-3, weight_decay=0.1, decay_1d=True)
    assert any(p.ndim <= 1 for g in including if g["weight_decay"] > 0 for p in g["params"])


def test_frozen_parameters_are_left_out():
    model = td.LSTM(16, 1, LAT, d_input=1)
    frozen = next(iter(model.parameters()))
    frozen.requires_grad_(False)
    listed = _all_params(td.param_groups(model, lr=1e-3))
    assert all(p is not frozen for p in listed)


def test_the_groups_drive_a_real_optimizer_step():
    """The point of the split is that the optimizer applies it — checked by
    stepping and confirming the tagged parameters moved less."""
    torch.manual_seed(0)
    model = td.S4(32, 1, LAT, d_input=1, d_state=16)
    tagged = [p for p in model.parameters() if getattr(p, "_optim", None)]
    before = [p.detach().clone() for p in tagged]

    opt = torch.optim.AdamW(td.param_groups(model, lr=1e-1), lr=1e-1)
    x = torch.randn(2, 3, 4, 5, 1)
    model(x).pow(2).mean().backward()
    opt.step()

    moved = max(float((p.detach() - b).abs().max()) for p, b in zip(tagged, before, strict=True))
    # Adam's step is ~lr in magnitude, so a 100x smaller rate must show.
    assert moved < 1e-2, f"SSM parameters moved by {moved:.3e} despite the capped rate"


# --- the schedule -------------------------------------------------------------


def test_warmup_reaches_the_peak_then_decays():
    model = td.LSTM(16, 1, LAT, d_input=1)
    opt = torch.optim.AdamW(td.param_groups(model, lr=1e-2), lr=1e-2)
    sched = td.warmup_cosine(opt, warmup=10, total=100)

    seen = []
    for _ in range(100):
        seen.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()

    assert seen[0] < seen[5] < seen[9]  # ramping
    assert seen[9] == pytest.approx(1e-2, rel=1e-6)  # peak at the end of warmup
    assert seen[-1] < seen[50] < seen[10]  # decaying thereafter
    # `seen` records before each step, so the last entry is the rate *going
    # into* the final step rather than the one after it: near zero, not zero.
    assert seen[-1] < 1e-5
    opt.step()
    sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-9)


def test_the_schedule_preserves_the_ratio_between_groups():
    """Scaling is multiplicative on purpose: a schedule that flattened every
    group to one rate would undo the separation that is the whole point."""
    model = td.S4(32, 1, LAT, d_input=1, d_state=16)
    opt = torch.optim.AdamW(td.param_groups(model, lr=1e-2), lr=1e-2)
    sched = td.warmup_cosine(opt, warmup=5, total=50)

    for _ in range(25):
        opt.step()
        sched.step()

    rates = [g["lr"] for g in opt.param_groups]
    assert max(rates) == pytest.approx(10 * min(rates), rel=1e-6)


def test_a_floor_keeps_the_rate_above_zero():
    model = td.LSTM(16, 1, LAT, d_input=1)
    opt = torch.optim.AdamW(td.param_groups(model, lr=1e-2), lr=1e-2)
    sched = td.warmup_cosine(opt, warmup=2, total=20, floor=0.1)
    for _ in range(20):
        opt.step()
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-3, rel=1e-6)


def test_an_impossible_schedule_is_refused():
    model = td.LSTM(16, 1, LAT, d_input=1)
    opt = torch.optim.AdamW(td.param_groups(model, lr=1e-3), lr=1e-3)
    with pytest.raises(ValueError, match="total > 0"):
        td.warmup_cosine(opt, warmup=5, total=0)


def test_the_recipe_trains_every_family_without_per_family_tuning():
    """The practical claim. Hand-picked per-family learning rates were needed
    only because the SSM parameters were being trained at the projection rate;
    with the groups and a warmup, one recipe holds for all of them."""
    lat = td.Lattice(shape=(4, 5), names=("h", "w"), time=True)
    builders = {
        "s4": lambda: td.S4(32, 2, lat, d_input=1, d_state=16),
        "mamba": lambda: td.Mamba(32, 2, lat, d_input=1, d_state=8),
        "transformer": lambda: td.Transformer(32, 2, lat, d_input=1),
        "lstm": lambda: td.LSTM(32, 2, lat, d_input=1),
    }
    steps = 60
    for name, build in builders.items():
        torch.manual_seed(0)
        model = build()
        head = torch.nn.Linear(32, 1)
        opt = torch.optim.AdamW(
            td.param_groups(model, lr=3e-3) + [{"params": head.parameters(), "lr": 3e-3}],
            lr=3e-3,
            betas=(0.9, 0.95),
        )
        sched = td.warmup_cosine(opt, warmup=steps // 10, total=steps)

        gen = torch.Generator().manual_seed(1)
        losses = []
        for _ in range(steps):
            x = torch.randn(3, 4, 4, 5, 1, generator=gen)
            loss = (head(model(x)) - x.cumsum(dim=3)).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(float(loss))

        assert all(torch.isfinite(torch.tensor(losses))), f"{name} diverged to non-finite"
        assert losses[-1] < losses[0], (
            f"{name} did not improve: {losses[0]:.3f} -> {losses[-1]:.3f}"
        )
