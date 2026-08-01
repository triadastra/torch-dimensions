"""LTI and non-LTI mixers, and what the difference does to N-D composition.

The library's premise is that any 1-D mixer can be swept over a lattice. True,
but not uniform: *which* mixer decides whether the sweep order is a modelling
choice or a no-op. This file measures the property and then measures the
consequence.

See LTI.md for the table these tests produce and what it means.
"""

from functools import partial

import pytest
import torch

import torch_dimensions as td
from torch_dimensions.mixers.attention import AttentionMixer
from torch_dimensions.mixers.conv import ConvMixer, TCNMixer
from torch_dimensions.mixers.rnn import GRUMixer, LSTMMixer
from torch_dimensions.mixers.ssm import MambaMixer, S4DMixer, S4Mixer

D = 4

# (name, factory, expected verdict). The expectations are the claim; the
# measurement is in check_lti.
CLASSIFICATION = [
    ("ConvMixer(activation=None)", partial(ConvMixer, D, activation=None), "LTI"),
    ("ConvMixer(gelu)", partial(ConvMixer, D), "time-invariant, nonlinear"),
    ("TCNMixer", partial(TCNMixer, D), "time-invariant, nonlinear"),
    ("S4DMixer", partial(S4DMixer, D), "time-invariant, nonlinear"),
    ("S4Mixer", partial(S4Mixer, D), "time-invariant, nonlinear"),
    ("MambaMixer", partial(MambaMixer, D), "neither"),
    ("LSTMMixer", partial(LSTMMixer, D), "neither"),
    ("GRUMixer", partial(GRUMixer, D), "neither"),
    ("AttentionMixer", partial(AttentionMixer, D, 2), "neither"),
]


@pytest.mark.parametrize("name,factory,expected", CLASSIFICATION)
def test_every_shipped_mixer_classifies_as_documented(name, factory, expected):
    report = td.testing.check_lti(factory)
    # "LTI (affine)" is LTI with a bias; the distinction is reported, not
    # asserted here.
    assert report.verdict.startswith(expected), f"{name}: {report}"


def test_linearity_is_measured_not_assumed():
    """A linear convolution is linear to floating point; adding one GELU is
    enough to destroy it. If both came out the same, the measurement would be
    measuring nothing."""
    linear = td.testing.check_lti(partial(ConvMixer, D, activation=None))
    nonlinear = td.testing.check_lti(partial(ConvMixer, D))
    assert linear.additivity < 1e-14
    assert nonlinear.additivity > 1e-3
    # Both remain time-invariant: a pointwise nonlinearity does not care where
    # in the sequence it is applied.
    assert linear.time_invariant and nonlinear.time_invariant


def test_a_recurrence_with_biased_gates_is_not_at_rest():
    """Why the RNN and Mamba families are not time-invariant despite being
    perfectly causal.

    Time invariance is a claim about a system *at rest*: feed it nothing, then
    feed the signal later, and the same thing comes out later. A gated
    recurrence fed zeros does not sit still — its biases drive the state — so
    a delayed signal arrives to a different machine than the original did.
    """
    for factory in (partial(LSTMMixer, D), partial(MambaMixer, D)):
        report = td.testing.check_lti(factory)
        assert not report.time_invariant, report
        # But the deviation is small: the state does settle, so this is a
        # transient rather than a wholesale failure. Worth stating, because
        # "not time-invariant" and "wildly different" are different claims.
        assert report.shift_equivariance < 1e-1, report


def test_attention_is_not_shift_equivariant_because_padding_adds_tokens():
    """Self-attention is permutation-equivariant, which is *stronger* than
    shift-equivariance on a fixed token set — but delaying a signal means
    padding, and padding a sequence adds tokens to attend over. The set
    changes, so the output does."""
    report = td.testing.check_lti(partial(AttentionMixer, D, 2))
    assert not report.time_invariant and report.shift_equivariance > 1e-3, report


# -- the consequence for N-D --------------------------------------------------


def _two_orders(mixer, lat):
    """The same shared mixer, swept in both axis orders.

    A shared instance rather than two builds: the claim is about the *order*,
    so the operators have to be identical objects or the comparison also
    contains a weight difference. `norm=False` because LayerNorm is nonlinear
    and would break commutation on its own; the residual is linear and stays.
    """
    a = td.AxialScan(
        mixer=mixer, plan=td.ScanPlan.from_list(["h", "w"]), lattice=lat, d_model=D, norm=False
    )
    b = td.AxialScan(
        mixer=mixer, plan=td.ScanPlan.from_list(["w", "h"]), lattice=lat, d_model=D, norm=False
    )
    return a.double().eval(), b.double().eval()


def _order_gap(mixer, lat, x):
    a, b = _two_orders(mixer, lat)
    with torch.no_grad():
        return (a(x) - b(x)).abs().max().item() / x.abs().max().item()


def test_a_scalar_valued_lti_filter_commutes_across_axes():
    """**The result this file exists for, and it is narrower than folklore.**

    "Separable convolutions commute, so the order does not matter" is true for
    *scalar* filters and false for the multichannel filters every real network
    uses. The operator commutes across axes exactly when it factorizes as a
    channel matrix times a spatial filter, ``W[out, in, j] = A[out, in]·b[j]``:
    then each axis contributes ``C_axis ⊗ A``, the spatial parts commute
    because they act on different axes, and the channel parts are the same
    matrix.

    Constructed rather than trained, because the point is which *structure*
    commutes — see the next test for what a generic filter does.
    """
    lat = td.Lattice(shape=(4, 5), names=("h", "w"))
    x = torch.randn(2, 4, 5, D, dtype=torch.float64)
    torch.manual_seed(0)
    mixer = ConvMixer(D, activation=None, bias=False).double()
    with torch.no_grad():
        channel = torch.randn(D, D, dtype=torch.float64)
        spatial = torch.randn(3, dtype=torch.float64)
        mixer.convs[0].weight.copy_(channel[:, :, None] * spatial[None, None, :])

    assert _order_gap(mixer, lat, x) < 1e-12


@pytest.mark.parametrize(
    "name,factory",
    [
        ("generic linear conv", partial(ConvMixer, D, activation=None, bias=False)),
        ("depthwise + pointwise", partial(ConvMixer, D, activation=None, depthwise=True)),
        ("gelu conv", partial(ConvMixer, D)),
        ("S4D", partial(S4DMixer, D)),
        ("LSTM", partial(LSTMMixer, D)),
        ("Mamba", partial(MambaMixer, D)),
    ],
)
def test_everything_else_depends_on_the_sweep_order(name, factory):
    """Including a perfectly linear, perfectly time-invariant convolution.

    A generic multichannel convolution is a *matrix-valued* filter: offset
    ``j`` carries its own channel matrix ``W[:, :, j]``, and matrix-valued
    filters commute only when those matrices commute with each other, which
    random ones do not. So the sweep order is a real modelling choice even for
    a linear CNN — and ``ScanPlan`` is not only for the selective models.

    The depthwise case is the instructive one: the depthwise convolution alone
    *would* commute, and the pointwise channel mix that follows it is what
    breaks it.
    """
    lat = td.Lattice(shape=(4, 5), names=("h", "w"))
    x = torch.randn(2, 4, 5, D, dtype=torch.float64)
    torch.manual_seed(0)
    assert _order_gap(factory().double(), lat, x) > 1e-6, name


def test_direction_is_free_for_a_centred_convolution_and_not_for_a_recurrence():
    """The second consequence. A centred convolution swept backwards is the
    same operator with a mirrored kernel, so bidirectionality buys nothing it
    did not already have. A causal mixer genuinely sees a different signal."""
    lat = td.Lattice(shape=(7,), names=("h",))
    x = torch.randn(2, 7, D, dtype=torch.float64)

    def swept(mixer, reverse):
        scan = td.AxialScan(
            mixer=mixer,
            plan=td.ScanPlan.from_list([("h", reverse)]),
            lattice=lat,
            d_model=D,
            norm=False,
        )
        with torch.no_grad():
            return scan.double().eval()(x)

    torch.manual_seed(0)
    # A symmetric-by-construction kernel: forward and reverse must agree
    # exactly, which is the sharp version of "direction is meaningless here".
    conv = ConvMixer(D, activation=None, bias=False).double()
    with torch.no_grad():
        w = conv.convs[0].weight
        w.copy_((w + w.flip(-1)) / 2)
    assert (swept(conv, False) - swept(conv, True)).abs().max().item() < 1e-12

    torch.manual_seed(0)
    lstm = LSTMMixer(D).double()
    assert (swept(lstm, False) - swept(lstm, True)).abs().max().item() > 1e-3
