"""The convolutional family: separability, causality, and dilation schedules.

The centrepiece is :func:`test_separable_stack_equals_a_full_nd_convolution`.
The kernel family's factorization is checked against ``torch.kron``; this is
the same discipline for the other factorization the library performs — a stack
of per-axis 1-D convolutions is claimed to *be* an N-D convolution with a
rank-1 kernel, and that claim is checked against ``F.conv2d``/``F.conv3d``
rather than against another call to itself.
"""

import math

import pytest
import torch
import torch.nn.functional as F

import torch_dimensions as td
from torch_dimensions.mixers.conv import ConvMixer, TCNMixer

MODELS = [td.CNN, td.TCN]


def _factory(cls):
    def build(lat, d_model, plan=None):
        return cls(d_model, lat.n_axes, lat, plan=plan)

    return build


# -- conformance --------------------------------------------------------------


@pytest.mark.parametrize("cls", MODELS)
def test_conv_family_passes_every_applicable_check(cls):
    report = td.testing.check_block(_factory(cls), ranks=(1, 2, 3))
    assert report, str(report)


@pytest.mark.parametrize("cls,mixer_cls", [(td.CNN, ConvMixer), (td.TCN, TCNMixer)])
def test_rank_one_single_layer_equals_the_bare_mixer(cls, mixer_cls):
    lat = td.Lattice(shape=(), time=True)
    torch.manual_seed(0)
    model = cls(6, 1, lat, norm=False, residual=False).double()
    torch.manual_seed(0)
    bare = mixer_cls(6).double()
    x = torch.randn(2, 7, 6, dtype=torch.float64)
    assert torch.equal(model(x), bare(x))


@pytest.mark.parametrize("cls", MODELS)
def test_the_conv_family_learns(cls):
    stats = td.testing.check_trainable(lambda lat, d: cls(d, 6, lat), steps=150)
    assert stats["ratio"] > 3.0, stats


# -- separability: the claim that makes this "N-D" --------------------------


def _joint_kernel(weights):
    """Outer-product the per-axis conv weights into one N-D kernel.

    Each ``(out, in, k)`` weight contracts with the next over the intermediate
    channel, which is what composing the layers does; the spatial indices stay
    separate and become the kernel's axes.
    """
    joint = weights[0]  # (o, i, k0)
    for axis, w in enumerate(weights[1:], start=1):
        # (o, m, k_new) x (m, i, k0..k_{axis-1}) -> (o, i, k0.., k_new)
        letters = "abcdefg"[:axis]
        joint = torch.einsum(f"omz,mi{letters}->oi{letters}z", w, joint)
    return joint


@pytest.mark.parametrize("rank,shape", [(2, (5, 7)), (3, (4, 5, 6))])
def test_separable_stack_equals_a_full_nd_convolution(rank, shape):
    """One sweep per axis, linear, is exactly an N-D convolution.

    Two details make the equality exact rather than approximate, and both are
    load-bearing. ``activation=None`` — with a nonlinearity between the sweeps
    the operators no longer commute and no joint kernel exists (the negative
    control below). And ``bias=False`` — zero-padding along one axis only
    survives a convolution along another axis if that convolution maps zero
    rows to zero rows, which a bias breaks. A separable model *with* a bias is
    still a fine model; it is just no longer this identity.
    """
    names = tuple("hwd"[:rank])
    lat = td.Lattice(shape=shape, names=names)
    torch.manual_seed(0)
    model = td.CNN(
        3,
        rank,
        lat,
        kernel_size=3,
        activation=None,
        norm=False,
        residual=False,
        mixer_kwargs={"bias": False},
    ).double()
    assert [lat.axis_names[s.axis] for s in model.plan] == list(names)

    x = torch.randn(2, *shape, 3, dtype=torch.float64)
    got = model(x)

    joint = _joint_kernel([m.convs[0].weight for m in model.nd.mixers])
    conv = F.conv2d if rank == 2 else F.conv3d
    channels_first = x.permute(0, x.ndim - 1, *range(1, x.ndim - 1))
    want = conv(channels_first, joint, padding=(1,) * rank)
    want = want.permute(0, *range(2, want.ndim), 1)

    diff = (got - want).abs().max().item()
    assert diff < 1e-12, f"separable stack differs from the joint kernel by {diff:.2e}"


def test_the_separability_identity_has_teeth():
    """The negative control: a nonlinearity between sweeps must break it.

    Without this, the test above would pass just as happily against a model
    that had quietly stopped being separable.
    """
    lat = td.Lattice(shape=(5, 7), names=("h", "w"))
    torch.manual_seed(0)
    model = td.CNN(
        3,
        2,
        lat,
        kernel_size=3,
        activation="gelu",
        norm=False,
        residual=False,
        mixer_kwargs={"bias": False},
    ).double()
    x = torch.randn(2, 5, 7, 3, dtype=torch.float64)
    joint = _joint_kernel([m.convs[0].weight for m in model.nd.mixers])
    want = F.conv2d(x.permute(0, 3, 1, 2), joint, padding=(1, 1)).permute(0, 2, 3, 1)
    assert (model(x) - want).abs().max().item() > 1e-3


# -- causality ----------------------------------------------------------------


def test_the_tcn_mixer_is_bitwise_causal():
    torch.manual_seed(0)
    mixer = TCNMixer(6).double()
    x = torch.randn(3, 16, 6, dtype=torch.float64)
    future = x.clone()
    future[:, 9:] += 100.0
    a, b = mixer(x), mixer(future)
    assert torch.equal(a[:, :9], b[:, :9]), "a causal convolution leaked the future"
    assert not torch.equal(a[:, 9:], b[:, 9:])


def test_the_centred_conv_is_deliberately_not_causal():
    """The negative control for the test above.

    A centred window *must* see forward — otherwise the causality test is
    passing on a mixer that happens to look backwards for some other reason.
    """
    torch.manual_seed(0)
    mixer = ConvMixer(6, activation=None).double()
    x = torch.randn(3, 12, 6, dtype=torch.float64)
    future = x.clone()
    future[:, 6:] += 100.0
    assert not torch.equal(mixer(x)[:, :6], mixer(future)[:, :6])


def test_an_even_kernel_is_refused_when_centred():
    with pytest.raises(ValueError, match="odd"):
        ConvMixer(4, kernel_size=4)
    ConvMixer(4, kernel_size=4, causal=True)  # defined side: allowed


# -- the dilation schedule ----------------------------------------------------


def test_dilation_doubles_per_axis_not_per_layer():
    """The N-D correction. Under a cyclic plan over three axes, layer 3 is the
    *second* sweep of axis 0, so it dilates by 2 — not by 8.

    This guards a real regression: ``TCNMixer`` swallowed ``sweep`` into
    ``**kw`` and every layer silently ran at dilation 1, which is a plain
    convolution stack wearing a TCN's name.
    """
    lat = td.Lattice(shape=(6, 8), names=("h", "w"), time=True)
    model = td.TCN(8, 9, lat)
    assert [m.dilation for m in model.nd.mixers] == [1, 1, 1, 2, 2, 2, 4, 4, 4]


def test_dilation_growth_is_opt_out():
    lat = td.Lattice(shape=(6, 8), names=("h", "w"), time=True)
    model = td.TCN(8, 6, lat, dilation_base=1)
    assert {m.dilation for m in model.nd.mixers} == {1}


def test_a_mixer_that_never_asked_for_sweep_is_built_unchanged():
    """The opt-in rule: only a factory whose signature names `sweep` gets one.

    An LSTM mixer has no such argument, and passing it would be a TypeError at
    construction — this is what keeps the feature invisible to every mixer
    that does not want it.
    """
    lat = td.Lattice(shape=(4, 5), names=("h", "w"), time=True)
    td.LSTM(8, 6, lat)  # would raise if `sweep` were passed unconditionally


# -- receptive field ----------------------------------------------------------


def test_receptive_field_reports_coverage_against_the_axis():
    lat = td.Lattice(shape=(6, 40), names=("h", "w"))
    model = td.CNN(8, 4, lat, kernel_size=3)  # two sweeps per axis, span 5
    rf = td.receptive_field(model)
    assert rf["h"] == {"span": 5, "size": 6, "covers": False, "layers": 2}
    assert rf["w"] == {"span": 5, "size": 40, "covers": False, "layers": 2}

    wide = td.CNN(8, 4, lat, kernel_size=7)  # span 13
    assert td.receptive_field(wide)["h"]["covers"] is True


def test_global_mixers_report_an_infinite_span():
    """An RNN spans its axis in one layer; saying "inf" is the honest answer,
    and it keeps the report meaningful across mixed families."""
    lat = td.Lattice(shape=(6, 8), names=("h", "w"))
    rf = td.receptive_field(td.LSTM(8, 2, lat))
    assert rf["h"]["span"] == math.inf and rf["h"]["covers"] is True


def test_time_axis_coverage_is_unanswerable_and_says_so():
    lat = td.Lattice(shape=(4,), names=("h",), time=True)
    rf = td.receptive_field(td.CNN(8, 2, lat))
    assert rf["time"]["size"] is None and rf["time"]["covers"] is None


def test_receptive_field_refuses_the_kernel_family():
    lat = td.Lattice(shape=(4, 5), names=("h", "w"), time=True)
    model = td.CNN(8, 3, lat, method=td.cafa)
    with pytest.raises(TypeError, match="scan-family"):
        td.receptive_field(model)


# -- registry and round-trip --------------------------------------------------


@pytest.mark.parametrize("kind", ["cnn", "tcn"])
def test_build_and_round_trip(kind, tmp_path):
    model = td.build(
        {
            "kind": kind,
            "d_model": 8,
            "n_layers": 4,
            "lattice": {"shape": [4, 5], "names": ["h", "w"]},
        }
    )
    x = torch.randn(2, 4, 5, 8)
    model.eval()
    path = tmp_path / "m.td"
    model.save(path)
    same = td.load(path).eval()
    assert torch.equal(model(x), same(x))


def test_depthwise_separable_is_separable_in_both_senses():
    """Depthwise convolutions separate channels from space; the axial fold
    separates the axes. Both at once is the cheapest corner of the design
    space, and it should still be a working model."""
    lat = td.Lattice(shape=(5, 6), names=("h", "w"))
    dense = td.CNN(16, 4, lat, depthwise=False)
    light = td.CNN(16, 4, lat, depthwise=True)
    n_dense = sum(p.numel() for p in dense.parameters())
    n_light = sum(p.numel() for p in light.parameters())
    assert n_light < n_dense / 2, (n_light, n_dense)
    assert light(torch.randn(2, 5, 6, 16)).shape == (2, 5, 6, 16)
