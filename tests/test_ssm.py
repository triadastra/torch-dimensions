"""Phase 7 acceptance for the portable SSM family. See PLAN.md.

The mixers' mathematics is cross-validated against the upstream reference
implementations out-of-repo (bitwise for the S4D kernel, ~1e-6 float32 for the
selective scan — recorded in PLAN.md Phase 7); those repos are not importable
in CI, so the in-repo tests check the properties that do not need them:
conformance, causality, learning, and the rank-1 reduction.
"""

import pytest
import torch

import torch_dimensions as td
from torch_dimensions.mixers.ssm import MambaMixer, S4DMixer

MODELS = [td.S4D, td.Mamba]
MIXERS = [S4DMixer, MambaMixer]


def _factory(cls):
    def build(lat, d_model, plan=None):
        return cls(d_model, lat.n_axes, lat, plan=plan)

    return build


# -- conformance --------------------------------------------------------------


@pytest.mark.parametrize("cls", MODELS)
def test_ssm_family_passes_every_applicable_check(cls):
    report = td.testing.check_block(_factory(cls), ranks=(1, 2, 3))
    assert report, str(report)


# -- causality ----------------------------------------------------------------


@pytest.mark.parametrize(
    "mixer_cls,atol",
    [
        # S4D's causal convolution runs through an FFT over the whole padded
        # line, so a changed future perturbs the *rounding* of the past by
        # ~1e-15 while the mathematics stays causal. Measured: 2.6e-15 under a
        # magnitude-100 future perturbation, vs 48 after the cut. The Mamba
        # recurrence touches the past not at all, so it is held to bitwise.
        (S4DMixer, 1e-12),
        (MambaMixer, 0.0),
    ],
)
def test_the_mixer_is_causal_along_the_swept_axis(mixer_cls, atol):
    """Output at t must ignore inputs after t — the structural property that
    makes a backward sweep meaningful. An acausal mixer would silently see
    both directions at once and make every bidirectional plan a lie."""
    torch.manual_seed(0)
    mixer = mixer_cls(6).double()
    x = torch.randn(3, 9, 6, dtype=torch.float64)
    future = x.clone()
    future[:, 5:] += 100.0
    a, b = mixer(x), mixer(future)
    leak = (a[:, :5] - b[:, :5]).abs().max().item()
    assert leak <= atol, f"perturbing the future changed the past by {leak:.2e}"
    assert not torch.equal(a[:, 5:], b[:, 5:]), "perturbing the future changed nothing at all"


# -- rank-1 reduction ---------------------------------------------------------


@pytest.mark.parametrize("cls,mixer_cls", zip(MODELS, MIXERS, strict=True))
def test_rank_one_single_layer_equals_the_bare_mixer(cls, mixer_cls):
    lat = td.Lattice(shape=(), time=True)
    torch.manual_seed(0)
    model = cls(6, 1, lat, norm=False, residual=False).double()
    torch.manual_seed(0)
    bare = mixer_cls(6).double()
    x = torch.randn(2, 7, 6, dtype=torch.float64)
    assert torch.equal(model(x), bare(x))


# -- learning -----------------------------------------------------------------


@pytest.mark.parametrize("cls", MODELS)
def test_the_ssm_family_learns_a_task_that_needs_axial_mixing(cls):
    result = td.testing.check_trainable(_factory(cls), steps=150)
    assert result["ratio"] >= 3.0, result


# -- configuration reaches the mixers -----------------------------------------


def test_mixer_options_reach_every_layer():
    lat = td.Lattice(shape=(2, 3))
    model = td.Mamba(8, 2, lat, d_state=4, expand=3)
    for m in model.nd.mixers:
        assert m.d_state == 4 and m.d_inner == 24
    s4 = td.S4D(8, 2, lat, d_state=8)
    for m in s4.nd.mixers:
        assert m.kernel.A_imag.shape[-1] == 4  # d_state // 2 conjugate pairs


def test_s4d_rejects_an_odd_state_size():
    with pytest.raises(ValueError, match="even"):
        S4DMixer(8, d_state=7)
