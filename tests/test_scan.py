"""Phase 3 acceptance for axial_apply / AxialScan / LSTM. See PLAN.md.

The load-bearing tests use ``cumsum`` and a position-weighting as mixers,
because torch already provides an independent reference for "apply this to
every 1-D line along axis d": ``x.cumsum(dim=d)`` and a broadcast multiply.
Anything wrong with the fold, the permutation, or the flip shows up as a
mismatch against an implementation that shares no code with ours.
"""

import pytest
import torch
import torch.nn as nn

from torch_dimensions import GRU, LSTM, AxialScan, Lattice, ScanPlan, axial_apply
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


# -- LSTM / GRU --------------------------------------------------------------


def test_rank_one_single_layer_equals_a_bare_lstm():
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


@pytest.mark.parametrize("model_cls", [LSTM, GRU])
@pytest.mark.parametrize("rank", RANKS)
def test_rnn_forward_and_backward(model_cls, rank):
    lat = Lattice(shape=_shape(rank), time=True)
    model = model_cls(d_model=4, n_layers=lat.n_axes, lattice=lat)
    x = torch.randn(2, 3, *lat.shape, 4)
    out = model(x)
    assert out.shape == x.shape
    out.pow(2).mean().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_rnn_uses_a_cyclic_plan_by_default():
    lat = Lattice(shape=(2, 3), names=("h", "w"), time=True)
    assert [s.axis for s in LSTM(4, 4, lat).plan] == [0, 1, 2, 0]


def test_rnn_accepts_a_custom_plan():
    lat = Lattice(shape=(2, 3), names=("h", "w"))
    plan = ScanPlan.from_list([("w", True), ("h", False)])
    assert [(s.axis, s.reverse) for s in LSTM(4, 2, lat, plan=plan).plan] == [
        (1, True),
        (0, False),
    ]


def test_rnn_refuses_plan_and_bidirectional_together():
    lat = Lattice(shape=(2, 3))
    with pytest.raises(ValueError, match="not both"):
        LSTM(4, 2, lat, plan=ScanPlan.from_list([0]), bidirectional=True)


def test_rnn_bidirectional_reaches_the_plan():
    lat = Lattice(shape=(2, 3), names=("h", "w"), time=True)
    plan = LSTM(4, 6, lat, bidirectional=("h", "w")).plan
    seen = {}
    for s in plan:
        seen.setdefault(s.axis, set()).add(s.reverse)
    assert seen[0] == {False}  # time stays causal
    assert seen[1] == {False, True} and seen[2] == {False, True}


# -- 1-D is the N-D case with nothing to fold --------------------------------


def test_no_lattice_gives_a_plain_sequence_model():
    model = LSTM(d_model=4, n_layers=3)
    assert model.lattice.rank == 0 and model.lattice.n_axes == 1
    x = torch.randn(2, 7, 4)
    assert model(x).shape == x.shape


def test_no_lattice_accepts_varying_sequence_length():
    """The degenerate lattice has no static size, so length stays dynamic."""
    model = LSTM(d_model=4, n_layers=2)
    for t in (1, 5, 50):
        assert model(torch.randn(2, t, 4)).shape == (2, t, 4)


def test_one_dimensional_stack_matches_stacked_lstm_layers():
    """No norm, no residual: the stack must equal applying each nn.LSTM in
    turn, which is what makes the 1-D path free of special-casing.

    Equal to floating point, not bitwise, and the reason is worth knowing.
    The fold reshapes, which needs a contiguous tensor, while nn.LSTM returns
    a transposed view. So from layer two onward the mixer receives contiguous
    input where the reference hands it a view, and torch's RNN kernels are not
    bit-identical across memory layouts. A single layer *is* bitwise exact --
    see the test above -- which is what pins the fold itself.
    """
    torch.manual_seed(0)
    lat = Lattice(shape=(), time=True)
    mixers = [LSTMMixer(4).double() for _ in range(3)]
    it = iter(mixers)
    scan = AxialScan(
        mixer=lambda: next(it),
        plan=ScanPlan.cyclic(("time",), 3),
        lattice=lat,
        d_model=4,
        norm=False,
        residual=False,
    )
    x = torch.randn(2, 6, 4, dtype=torch.float64)
    ref = x
    for m in mixers:
        ref = m(ref)
    assert torch.allclose(scan(x), ref, rtol=0, atol=1e-15)
    assert (scan(x) - ref).abs().max() < 1e-15


def test_time_only_lattice_rejects_a_validity_mask():
    with pytest.raises(ValueError, match="no cells to mark valid"):
        Lattice(shape=(), time=True, valid=torch.ones(1, dtype=torch.bool))


def test_empty_shape_without_time_is_still_an_error():
    with pytest.raises(ValueError, match="time=False"):
        Lattice(shape=())


# -- nd_method ---------------------------------------------------------------


def test_nd_method_accepts_the_exported_function():
    """The documented spelling: a strategy is a function, not a string."""
    import torch_dimensions as td

    lat = Lattice(shape=(2, 3))
    assert isinstance(LSTM(4, 2, lat, nd_method=td.axial_scan).nd, AxialScan)


def test_the_default_strategy_is_that_same_function():
    import torch_dimensions as td

    lat = Lattice(shape=(2, 3))
    explicit = LSTM(4, 2, lat, nd_method=td.axial_scan)
    assert type(LSTM(4, 2, lat).nd) is type(explicit.nd)


def test_nd_method_still_accepts_a_registered_name_for_config():
    """YAML cannot hold a callable, so names keep working."""
    lat = Lattice(shape=(2, 3))
    assert isinstance(LSTM(4, 2, lat, nd_method="axial_scan").nd, AxialScan)


def test_nd_method_accepts_a_user_written_callable():
    """A custom strategy needs no registration — that is the extension point."""
    seen = {}

    def only_first_axis(mixer, plan, lattice, d_model, **kw):
        seen["called"] = True
        return AxialScan(
            mixer=mixer, plan=ScanPlan.from_list([0]), lattice=lattice, d_model=d_model
        )

    lat = Lattice(shape=(2, 3))
    model = LSTM(4, 2, lat, nd_method=only_first_axis)
    assert seen["called"] and len(model.plan) == 1
    assert model(torch.randn(2, 2, 3, 4)).shape == (2, 2, 3, 4)


def test_unknown_nd_method_names_the_registered_ones():
    lat = Lattice(shape=(2, 3))
    with pytest.raises(ValueError, match="axial_scan"):
        LSTM(4, 2, lat, nd_method="cafa")


def test_nd_method_must_be_a_name_or_callable():
    lat = Lattice(shape=(2, 3))
    with pytest.raises(TypeError, match="name or a callable"):
        LSTM(4, 2, lat, nd_method=42)


def test_registering_a_duplicate_nd_method_is_refused():
    import torch_dimensions as td
    from torch_dimensions import register_nd_method

    with pytest.raises(ValueError, match="already registered"):
        register_nd_method("axial_scan", td.axial_scan)
