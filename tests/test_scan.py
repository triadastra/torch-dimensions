"""Phase 3 acceptance for axial_apply / AxialScan / LSTMND. See PLAN.md.

The load-bearing tests use ``cumsum`` and a position-weighting as mixers,
because torch already provides an independent reference for "apply this to
every 1-D line along axis d": ``x.cumsum(dim=d)`` and a broadcast multiply.
Anything wrong with the fold, the permutation, or the flip shows up as a
mismatch against an implementation that shares no code with ours.
"""

import pytest
import torch
import torch.nn as nn

from torch_dimensions import GRUND, LSTMND, AxialScan, Lattice, ScanPlan, axial_apply
from torch_dimensions.mixers import LSTMMixer

RANKS = [1, 2, 3, 4]


def _shape(rank):
    return tuple(range(2, 2 + rank))


def _x(lat, b=2, t=3, h=3, dtype=torch.float64):
    lead = (b, t) if lat.time else (b,)
    return torch.randn(*lead, *lat.shape, h, dtype=dtype)


def cumsum(seq):
    return seq.cumsum(dim=1)


def weighted(seq):
    """Scale position i by (i + 1) — catches misalignment that cumsum, being
    a prefix operation, could tolerate."""
    w = torch.arange(1, seq.shape[1] + 1, dtype=seq.dtype).view(1, -1, 1)
    return seq * w


def weighted_ref(x, dim):
    shape = [1] * x.ndim
    shape[dim] = x.shape[dim]
    w = torch.arange(1, x.shape[dim] + 1, dtype=x.dtype).view(shape)
    return x * w


# -- axial_apply: the axis bookkeeping ---------------------------------------


@pytest.mark.parametrize("rank", RANKS)
@pytest.mark.parametrize("time", [False, True])
def test_applies_the_mixer_to_every_line_of_the_axis(rank, time):
    lat = Lattice(shape=_shape(rank), time=time)
    x = _x(lat)
    for axis in range(lat.n_axes):
        d = lat.tensor_dim(axis)
        assert torch.equal(axial_apply(x, lat, axis, cumsum), x.cumsum(dim=d)), f"axis={axis}"
        assert torch.equal(axial_apply(x, lat, axis, weighted), weighted_ref(x, d))


@pytest.mark.parametrize("rank", RANKS)
def test_reverse_sweeps_the_line_backwards(rank):
    lat = Lattice(shape=_shape(rank))
    x = _x(lat)
    for axis in range(lat.n_axes):
        d = lat.tensor_dim(axis)
        got = axial_apply(x, lat, axis, cumsum, reverse=True)
        assert torch.equal(got, x.flip(d).cumsum(dim=d).flip(d)), f"axis={axis}"


def test_reverse_actually_changes_the_result():
    """Guards against a flip that is silently a no-op."""
    lat = Lattice(shape=(4,))
    x = _x(lat)
    assert not torch.equal(
        axial_apply(x, lat, 0, cumsum), axial_apply(x, lat, 0, cumsum, reverse=True)
    )


def test_rank_one_is_exactly_the_bare_mixer():
    """With one axis there is nothing to fold, so the machinery must vanish."""
    lat = Lattice(shape=(6,))
    x = _x(lat)
    assert torch.equal(axial_apply(x, lat, 0, cumsum), cumsum(x))


def test_axes_can_be_named():
    lat = Lattice(shape=(3, 4), names=("h", "w"))
    x = _x(lat)
    assert torch.equal(axial_apply(x, lat, "w", cumsum), axial_apply(x, lat, 1, cumsum))


@pytest.mark.parametrize("chunk", [1, 2, 3, 1000])
def test_chunking_does_not_change_the_result(chunk):
    lat = Lattice(shape=(3, 4))
    x = _x(lat)
    assert torch.equal(axial_apply(x, lat, 0, cumsum, chunk=chunk), axial_apply(x, lat, 0, cumsum))


def test_a_mixer_that_changes_shape_is_rejected_clearly():
    lat = Lattice(shape=(3, 4))
    with pytest.raises(ValueError, match=r"\(M, A, H\) -> \(M, A, H\)"):
        axial_apply(_x(lat), lat, 0, lambda s: s[..., :1])


# -- AxialScan ---------------------------------------------------------------


def _linear_scan(lat, d_model=3, n_layers=None, **kw):
    # Default to one layer per axis so the plan covers the lattice and the
    # "axis never swept" warning stays meaningful when it does fire.
    plan = ScanPlan.cyclic(lat.axis_names, n_layers or lat.n_axes)
    return AxialScan(
        mixer=lambda: nn.Linear(d_model, d_model),
        plan=plan,
        lattice=lat,
        d_model=d_model,
        **kw,
    ).double()


@pytest.mark.parametrize("rank", RANKS)
@pytest.mark.parametrize("time", [False, True])
def test_scan_preserves_shape(rank, time):
    lat = Lattice(shape=_shape(rank), time=time)
    x = _x(lat)
    assert _linear_scan(lat)(x).shape == x.shape


def test_scan_follows_the_plan_in_order():
    """Distinct axis sizes let a shared mixer report which axis each layer
    swept, via the sequence length it was handed."""

    class Recorder(nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = []

        def forward(self, x):
            self.seen.append(x.shape[1])
            return x

    lat = Lattice(shape=(2, 3, 4), names=("a", "b", "c"))
    rec = Recorder()
    plan = ScanPlan.from_list(["c", "a", "b", "c"])
    AxialScan(mixer=rec, plan=plan, lattice=lat, d_model=3)(_x(lat, dtype=torch.float32))
    assert rec.seen == [4, 2, 3, 4]


def test_scan_gradients_reach_every_parameter():
    lat = Lattice(shape=(2, 3))
    model = _linear_scan(lat, n_layers=3)
    model(_x(lat)).pow(2).mean().backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, missing


def test_scan_is_gradcheck_clean():
    lat = Lattice(shape=(2, 3))
    model = _linear_scan(lat, d_model=2, n_layers=2)
    x = torch.randn(1, 2, 3, 2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(model, (x,), fast_mode=True)


def test_rejects_wrong_feature_width():
    lat = Lattice(shape=(2, 3))
    with pytest.raises(ValueError, match="expected 3 features"):
        _linear_scan(lat)(torch.randn(2, 2, 3, 5, dtype=torch.float64))


def test_unknown_axis_fails_at_construction_not_at_forward():
    lat = Lattice(shape=(2, 3), names=("h", "w"))
    with pytest.raises(KeyError, match="depth"):
        AxialScan(
            mixer=lambda: nn.Identity(),
            plan=ScanPlan.from_list(["depth"]),
            lattice=lat,
            d_model=3,
        )


def test_module_mixer_shares_weights_across_layers():
    lat = Lattice(shape=(2, 3))
    shared = nn.Linear(3, 3)
    plan = ScanPlan.cyclic(("dim0", "dim1"), 4)
    scan = AxialScan(mixer=shared, plan=plan, lattice=lat, d_model=3)
    assert all(m is shared for m in scan.mixers)
    factory = _linear_scan(lat, n_layers=4)
    assert len({id(m) for m in factory.mixers}) == 4


# -- sparse lattices ---------------------------------------------------------


def _sparse(shape=(3, 4), seed=0):
    torch.manual_seed(seed)
    valid = torch.rand(shape) > 0.4
    valid.reshape(-1)[0] = True
    return Lattice(shape=shape, valid=valid)


def test_absent_cell_values_cannot_influence_the_output():
    """Perturb only the absent cells; every output must be bitwise identical.
    Zeroing on entry is what buys this."""
    lat = _sparse()
    model = _linear_scan(lat, n_layers=3)
    x = _x(lat)
    other = x + (~lat.mask()).to(x.dtype) * torch.randn_like(x) * 1e3
    assert torch.equal(model(x), model(other))


def test_absent_cells_are_zero_on_output():
    lat = _sparse()
    out = _linear_scan(lat, n_layers=2)(_x(lat))
    assert out.masked_select(~lat.mask().expand_as(out)).abs().max() == 0


def test_dense_lattice_allocates_no_mask():
    assert _linear_scan(Lattice(shape=(2, 3))).cell_mask is None


# -- LSTMND / GRUND ----------------------------------------------------------


def test_lstmnd_rank_one_single_layer_equals_a_bare_lstm():
    """With one axis, no norm, and no residual, the stack must reduce exactly
    to nn.LSTM — the sharpest check that the fold is transparent."""
    torch.manual_seed(0)
    lat = Lattice(shape=(5,))
    mixer = LSTMMixer(4).double()
    scan = AxialScan(
        mixer=mixer,
        plan=ScanPlan.from_list([0]),
        lattice=lat,
        d_model=4,
        norm=False,
        residual=False,
    )
    x = _x(lat, h=4)
    assert torch.equal(scan(x), mixer.rnn(x)[0])


@pytest.mark.parametrize("model_cls", [LSTMND, GRUND])
@pytest.mark.parametrize("rank", RANKS)
def test_nd_rnn_forward_and_backward(model_cls, rank):
    lat = Lattice(shape=_shape(rank), time=True)
    model = model_cls(d_model=4, n_layers=lat.n_axes, lattice=lat)
    x = torch.randn(2, 3, *lat.shape, 4)
    out = model(x)
    assert out.shape == x.shape
    out.pow(2).mean().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_nd_rnn_uses_a_cyclic_plan_by_default():
    lat = Lattice(shape=(2, 3), names=("h", "w"), time=True)
    assert [s.axis for s in LSTMND(4, 4, lat).plan] == [0, 1, 2, 0]


def test_nd_rnn_accepts_a_custom_plan():
    lat = Lattice(shape=(2, 3), names=("h", "w"))
    plan = ScanPlan.from_list([("w", True), ("h", False)])
    assert [(s.axis, s.reverse) for s in LSTMND(4, 2, lat, plan=plan).plan] == [
        (1, True),
        (0, False),
    ]


def test_nd_rnn_refuses_plan_and_bidirectional_together():
    lat = Lattice(shape=(2, 3))
    with pytest.raises(ValueError, match="not both"):
        LSTMND(4, 2, lat, plan=ScanPlan.from_list([0]), bidirectional=True)


def test_nd_rnn_bidirectional_reaches_the_plan():
    lat = Lattice(shape=(2, 3), names=("h", "w"), time=True)
    plan = LSTMND(4, 6, lat, bidirectional=("h", "w")).plan
    seen = {}
    for s in plan:
        seen.setdefault(s.axis, set()).add(s.reverse)
    assert seen[0] == {False}  # time stays causal
    assert seen[1] == {False, True} and seen[2] == {False, True}
