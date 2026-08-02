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
from torch_dimensions.mixers.ssm import MambaMixer, S4DMixer, S4Mixer

MODELS = [td.S4, td.S4D, td.Mamba]
MIXERS = [S4Mixer, S4DMixer, MambaMixer]


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
        # The convolutional mixers run through an FFT over the whole padded
        # line, so a changed future perturbs the *rounding* of the past by
        # ~1e-15 while the mathematics stays causal. Measured: 2.6e-15 under a
        # magnitude-100 future perturbation, vs 48 after the cut. The Mamba
        # recurrence touches the past not at all, so it is held to bitwise.
        (S4Mixer, 1e-12),
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
    # portable=True: the model class must add nothing around its bare mixer.
    # The default (upstream) build is checked the same way in test_vendored /
    # test_portable_flag; here the pairing is with the portable mixer.
    lat = td.Lattice(shape=(), time=True)
    torch.manual_seed(0)
    model = cls(6, 1, lat, norm=False, residual=False, portable=True).double()
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
    # portable build: our mixers' own attributes
    model = td.Mamba(8, 2, lat, d_state=4, expand=3, portable=True)
    for m in model.nd.mixers:
        assert m.d_state == 4 and m.d_inner == 24
    s4 = td.S4D(8, 2, lat, d_state=8, portable=True)
    for m in s4.nd.mixers:
        assert m.kernel.A_imag.shape[-1] == 4  # d_state // 2 conjugate pairs
    # default (upstream) build: the same options must reach the authors' blocks
    model = td.Mamba(8, 2, lat, d_state=4, expand=3)
    for m in model.nd.mixers:
        assert m.block.d_state == 4 and m.block.d_inner == 24
    s4 = td.S4D(8, 2, lat, d_state=8)
    for m in s4.nd.mixers:
        assert m.block.layer.kernel.N == 4  # their kernel halves N for conjugate pairs


def test_s4d_rejects_an_odd_state_size():
    with pytest.raises(ValueError, match="even"):
        S4DMixer(8, d_state=7)


# -- the S4 kernel against an independent dense reference ----------------------


def test_the_s4_kernel_equals_dense_state_space_powers():
    """The frequency-domain DPLR computation (Cauchy resolvent + Woodbury +
    irfft) against the thing it claims to equal: materialize A = Λ - PP*,
    bilinear-discretize, and take matrix powers. Evaluating the transfer
    function at the L-th roots of unity yields the L-periodized kernel, hence
    the (I - dA^L)^{-1} aliasing factor. Machine-precision equality expected —
    measured 4.6e-16 — because both sides are exact linear algebra."""
    from torch_dimensions.mixers.ssm import _S4Kernel

    torch.manual_seed(0)
    h_width, n_state, length = 3, 8, 24
    kern = _S4Kernel(h_width, d_state=n_state).double()
    with torch.no_grad():
        k_freq = kern(length)
        dt = torch.exp(kern.log_dt)
        lam = -torch.exp(kern.log_A_real) + 1j * kern.A_imag
        b = torch.view_as_complex(kern.B)
        c = torch.view_as_complex(kern.C)
        p = torch.view_as_complex(kern.P)
        rows = []
        for h in range(h_width):
            lam_f = torch.cat([lam[h], lam[h].conj()])
            p_f = torch.cat([p[h], p[h].conj()])
            b_f = torch.cat([b[h], b[h].conj()])
            c_f = torch.cat([c[h], c[h].conj()])
            a_f = torch.diag(lam_f) - torch.outer(p_f, p_f.conj())
            eye = torch.eye(n_state, dtype=a_f.dtype)
            half = dt[h] / 2
            inv = torch.linalg.inv(eye - half * a_f)
            d_a = inv @ (eye + half * a_f)
            d_b = inv @ (dt[h] * b_f)
            alias = torch.linalg.inv(eye - torch.linalg.matrix_power(d_a, length))
            state = alias @ d_b
            row = []
            for _ in range(length):
                row.append((c_f @ state).real)
                state = d_a @ state
            rows.append(torch.stack(row))
        k_dense = torch.stack(rows)
    assert torch.allclose(k_freq, k_dense, atol=1e-12), (
        f"max diff {(k_freq - k_dense).abs().max().item():.2e}"
    )


# -- the explicit N-D names ----------------------------------------------------


@pytest.mark.parametrize(
    "nd_cls,base_name", [(td.S4ND, "S4"), (td.S4DND, "S4D"), (td.MambaND, "Mamba")]
)
def test_the_nd_names_build_their_own_lattice(nd_cls, base_name):
    model = nd_cls(8, 4, dim=2, shape=(3, 4), names=("h", "w"))
    assert model.lattice.rank == 2 and model.lattice.time
    out = model(torch.randn(2, 5, 3, 4, 8))
    assert out.shape == (2, 5, 3, 4, 8)
    assert model.to_spec()["model"]["kind"] == nd_cls.__name__


def test_the_nd_names_refuse_ambiguity_and_absence():
    with pytest.raises(ValueError, match="requires `dim`"):
        td.S4ND(8, 4, shape=(3, 4))  # the N-D name without declaring N
    with pytest.raises(ValueError, match="needs a lattice"):
        td.S4ND(8, 4, dim=2)
    with pytest.raises(ValueError, match="not both"):
        td.S4ND(8, 4, dim=2, lattice=td.Lattice(shape=(3, 4)), shape=(3, 4))
    with pytest.raises(ValueError, match="dim=3"):
        td.MambaND(8, 4, shape=(3, 4), dim=3)
    with pytest.raises(ValueError, match=">= 2"):
        td.S4ND(8, 4, dim=0, shape=(3,))


def test_dim_one_is_refused_by_name():
    """dim=1 under the N-D name would run the plain 1-D model while the code
    reads 'S4ND' — the reader would believe something false. Each refusal
    names the class actually being requested."""
    with pytest.raises(ValueError, match="one spatial axis is just S4[^D]"):
        td.S4ND(8, 4, dim=1, shape=(6,))
    with pytest.raises(ValueError, match="just S4D"):
        td.S4DND(8, 4, dim=1, shape=(6,))
    with pytest.raises(ValueError, match="just Mamba"):
        td.MambaND(8, 4, dim=1, lattice=td.Lattice(shape=(6,), time=True))


def test_dim_that_agrees_is_accepted():
    assert td.S4ND(8, 4, dim=2, shape=(3, 4)).lattice.rank == 2
