"""Phase 6 acceptance for the kernel family. See PLAN.md.

The load-bearing test builds the joint operator explicitly as a Kronecker
product and checks the factorized contraction equals it. That is only possible
while the lattice is small, which is exactly why it happens now rather than
after the attention modules are layered on top.
"""

import pytest
import torch

from torch_dimensions import Lattice
from torch_dimensions.compose.kernel import axial_contract, kron_operator

RANKS = [1, 2, 3, 4]


def _lat(rank, **kw):
    return Lattice(shape=tuple(range(2, 2 + rank)), **kw)


def _kernels(lat, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(s, s, generator=g, dtype=torch.float64) for s in lat.shape]


def _contract_all(x, lat, kernels, valid=None):
    for axis, k in enumerate(kernels):
        x = axial_contract(x, lat, axis, k, valid=valid)
    return x


# -- the identity the whole family rests on ----------------------------------


@pytest.mark.parametrize("rank", RANKS)
def test_sequential_contraction_equals_the_kronecker_product(rank):
    lat = _lat(rank)
    kernels = _kernels(lat)
    x = torch.randn(2, *lat.shape, 3, dtype=torch.float64)

    got = _contract_all(x, lat, kernels)

    # Independent reference: flatten the lattice and apply the joint operator.
    joint = kron_operator(kernels)
    flat = x.reshape(2, lat.n_cells, 3)
    want = (joint @ flat).reshape(x.shape)

    assert torch.allclose(got, want, atol=1e-10), (got - want).abs().max()


def test_the_joint_operator_is_as_large_as_advertised():
    """The reason the factorization exists: the explicit operator is quadratic
    in cells, the factorized one only in axial size."""
    lat = _lat(3)  # (2, 3, 4) -> 24 cells
    joint = kron_operator(_kernels(lat))
    assert joint.shape == (24, 24)
    assert sum(k.numel() for k in _kernels(lat)) == 4 + 9 + 16 < 24 * 24


@pytest.mark.parametrize("rank", RANKS)
def test_contraction_order_does_not_matter_on_a_dense_lattice(rank):
    """Kronecker factors commute across distinct axes; if ours do not, the
    contraction is entangling axes it should not."""
    lat = _lat(rank)
    kernels = _kernels(lat)
    x = torch.randn(2, *lat.shape, 3, dtype=torch.float64)

    forward = _contract_all(x, lat, kernels)
    backward = x
    for axis in reversed(range(rank)):
        backward = axial_contract(backward, lat, axis, kernels[axis])
    assert torch.allclose(forward, backward, atol=1e-10)


def test_identity_kernels_leave_the_input_alone():
    lat = _lat(3)
    eye = [torch.eye(s, dtype=torch.float64) for s in lat.shape]
    x = torch.randn(2, *lat.shape, 3, dtype=torch.float64)
    assert torch.allclose(_contract_all(x, lat, eye), x, atol=1e-12)


def test_a_single_axis_contraction_is_a_plain_matmul():
    lat = _lat(1)
    k = _kernels(lat)[0]
    x = torch.randn(2, 2, 3, dtype=torch.float64)
    assert torch.allclose(axial_contract(x, lat, 0, k), k @ x, atol=1e-12)


def test_contraction_works_with_a_time_axis():
    lat = _lat(2, time=True)
    kernels = _kernels(lat)
    x = torch.randn(2, 4, *lat.shape, 3, dtype=torch.float64)
    out = x
    for axis, k in enumerate(kernels):
        out = axial_contract(out, lat, lat.axis_names[axis + 1], k)
    assert out.shape == x.shape


def test_axes_can_be_named():
    lat = Lattice(shape=(3, 4), names=("h", "w"))
    k = torch.randn(4, 4, dtype=torch.float64)
    x = torch.randn(2, 3, 4, 5, dtype=torch.float64)
    assert torch.equal(axial_contract(x, lat, "w", k), axial_contract(x, lat, 1, k))


def test_a_batched_kernel_broadcasts_over_the_folded_batch():
    lat = _lat(2)
    x = torch.randn(2, *lat.shape, 3, dtype=torch.float64)
    m = x.shape[0] * lat.shape[1]  # folded rows when sweeping axis 0
    k = torch.randn(m, 2, 2, dtype=torch.float64)
    assert axial_contract(x, lat, 0, k).shape == x.shape


# -- sparse renormalization --------------------------------------------------


def _sparse(rank=2, seed=0):
    shape = tuple(range(2, 2 + rank))
    g = torch.Generator().manual_seed(seed)
    valid = torch.rand(shape, generator=g) > 0.4
    valid.reshape(-1)[0] = True
    valid.reshape(-1)[-1] = True
    return Lattice(shape=shape, valid=valid)


def test_renormalization_makes_a_uniform_kernel_average_only_present_cells():
    """With a uniform kernel the contraction is a mean; renormalized, it must
    be the mean over cells that exist, not over all of them."""
    valid = torch.tensor([[True, True, True], [True, False, False]])
    lat = Lattice(shape=(2, 3), valid=valid)
    x = torch.ones(1, 2, 3, 1, dtype=torch.float64) * lat.mask().to(torch.float64)
    ones = torch.ones(3, 3, dtype=torch.float64)

    out = axial_contract(x, lat, 1, ones, valid=lat.mask().to(torch.float64))
    # Row 0 has three present cells all equal to 1 -> mean 1.
    assert torch.allclose(out[0, 0], torch.ones(3, 1, dtype=torch.float64))
    # Row 1 has one present cell equal to 1 -> still 1, not 1/3.
    assert torch.allclose(out[0, 1], torch.ones(3, 1, dtype=torch.float64))


def test_without_renormalization_structural_zeros_dilute_the_result():
    """The control that gives the test above its meaning.

    Needs a *row-stochastic* kernel to say anything: with an unnormalized
    all-ones kernel the contraction is a sum rather than a mean, and a sum has
    no dilution to show.
    """
    valid = torch.tensor([[True, True, True], [True, False, False]])
    lat = Lattice(shape=(2, 3), valid=valid)
    mask = lat.mask().to(torch.float64)
    x = torch.ones(1, 2, 3, 1, dtype=torch.float64) * mask
    uniform = torch.full((3, 3), 1 / 3, dtype=torch.float64)  # rows sum to 1

    plain = axial_contract(x, lat, 1, uniform)
    renormed = axial_contract(x, lat, 1, uniform, valid=mask)
    one = torch.ones(3, 1, dtype=torch.float64)

    # Row 0: all three cells present, so both agree on the true mean of 1.
    assert torch.allclose(plain[0, 0], one)
    assert torch.allclose(renormed[0, 0], one)

    # Row 1: only one cell present. Unrenormalized it is averaged over three
    # slots, two of which are structural zeros -> 1/3. That is the dilution.
    assert torch.allclose(plain[0, 1], one / 3)
    assert torch.allclose(renormed[0, 1], one)


@pytest.mark.parametrize("rank", [2, 3])
def test_absent_cell_values_cannot_influence_present_outputs(rank):
    lat = _sparse(rank)
    kernels = _kernels(lat)
    mask = lat.mask().to(torch.float64)
    x = torch.randn(2, *lat.shape, 3, dtype=torch.float64) * mask
    noise = torch.randn_like(x) * 1e3 * (1 - mask)

    a = _contract_all(x, lat, kernels, valid=mask) * mask
    b = _contract_all(x + noise, lat, kernels, valid=mask) * mask
    assert torch.equal(a, b), "absent cells leaked into present outputs"


def test_a_line_with_no_present_cells_stays_finite():
    """Dead lines divide by clamped zero; they must not produce NaN."""
    valid = torch.tensor([[True, True], [False, False]])
    lat = Lattice(shape=(2, 2), valid=valid)
    mask = lat.mask().to(torch.float64)
    x = torch.randn(1, 2, 2, 3, dtype=torch.float64) * mask
    out = axial_contract(x, lat, 1, torch.randn(2, 2, dtype=torch.float64), valid=mask)
    assert torch.isfinite(out).all()


def test_renormalization_is_a_no_op_on_a_dense_lattice_with_a_stochastic_kernel():
    """When every cell is present and the kernel rows sum to one, the
    denominator is one everywhere and nothing changes."""
    lat = _lat(2)
    ones = torch.ones(*lat.shape, 1, dtype=torch.float64)
    kernels = [torch.softmax(k, dim=-1) for k in _kernels(lat)]
    x = torch.randn(2, *lat.shape, 3, dtype=torch.float64)
    plain = _contract_all(x, lat, kernels)
    renorm = _contract_all(x, lat, kernels, valid=ones)
    assert torch.allclose(plain, renorm, atol=1e-10)


# -- autograd ----------------------------------------------------------------


def test_contraction_is_differentiable_through_both_arguments():
    lat = _lat(2)
    x = torch.randn(1, *lat.shape, 2, dtype=torch.float64, requires_grad=True)
    kernels = [k.clone().requires_grad_(True) for k in _kernels(lat)]
    _contract_all(x, lat, kernels).pow(2).sum().backward()
    assert x.grad is not None
    assert all(k.grad is not None for k in kernels)


def test_gradcheck_passes_through_the_contraction():
    lat = _lat(2)
    kernels = _kernels(lat)

    def fn(x):
        return _contract_all(x, lat, kernels)

    x = torch.randn(1, *lat.shape, 2, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(fn, (x,), fast_mode=True)


def test_a_signed_kernel_does_not_explode_when_the_mass_cancels():
    """`clamp_min` assumes a non-negative mass. A signed kernel — LeakyReLU
    scores, as upstream CaFA uses by default — can cancel to zero while the
    numerator stays nonzero, and clamping to +eps then divides by ~0."""
    lat = Lattice(shape=(2, 4), valid=torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]]).bool())
    mask = lat.mask().to(torch.float64)
    x = torch.randn(1, 2, 4, 3, dtype=torch.float64) * mask
    signed = torch.tensor(
        [
            [1.0, -1.0, 0.5, 0.5],
            [-1.0, 1.0, 0.5, 0.5],
            [0.5, 0.5, 1.0, -1.0],
            [0.5, 0.5, -1.0, 1.0],
        ],
        dtype=torch.float64,
    )
    out = axial_contract(x, lat, 1, signed, valid=mask)
    assert torch.isfinite(out).all()
    # Row 0's mass cancels exactly; the output must stay the same order of
    # magnitude as the input rather than blowing up by ~1e6.
    assert out.abs().max() < 10 * x.abs().max(), out.abs().max().item()


def test_a_genuinely_dead_line_is_still_zero_under_the_guard():
    """Leaving degenerate lines unscaled must not resurrect them: with no
    present cells the numerator is zero, so the output stays zero."""
    lat = Lattice(shape=(2, 2), valid=torch.tensor([[True, True], [False, False]]))
    mask = lat.mask().to(torch.float64)
    x = torch.randn(1, 2, 2, 3, dtype=torch.float64) * mask
    out = axial_contract(x, lat, 1, torch.rand(2, 2, dtype=torch.float64), valid=mask)
    assert torch.isfinite(out).all()
    assert out[0, 1].abs().max() == 0.0


def test_a_nan_in_the_input_is_not_silently_laundered():
    """A `nan_to_num` after the division zeroed NaNs arriving in `x`, hiding a
    diverging model mid-network behind finite numbers. The magnitude guard
    already makes the division itself safe, so the only NaNs reaching that
    point are real upstream failures — and a NaN that arrives must leave."""
    lat = Lattice(shape=(4,), valid=torch.tensor([True, True, True, False]))
    mask = lat.mask().to(torch.float64)
    x = torch.randn(2, 4, 3, dtype=torch.float64) * mask
    x[0, 1, 2] = float("nan")  # a present cell diverged upstream
    out = axial_contract(x, lat, 0, torch.randn(4, 4, dtype=torch.float64), valid=mask)
    assert bool(out.isnan().any()), "an input NaN vanished into finite output"


def test_float32_near_cancellation_does_not_explode():
    """The absolute-epsilon guard waved through a denominator of ~1e-4 —
    small enough to amplify by 1e4, large enough to pass any tiny fixed
    threshold — and float32 outputs blew up ~7000x. Degeneracy is
    cancellation, and cancellation is *relative* to the absolute mass."""
    lat = Lattice(shape=(2, 4), valid=torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]]).bool())
    mask = lat.mask().to(torch.float32)
    x = (torch.randn(1, 2, 4, 3) * 100) * mask
    near_cancel = torch.tensor(
        [
            [1.0, -0.9999, 0.5, 0.5],
            [-1.0, 1.0001, 0.5, 0.5],
            [0.5, 0.5, 1.0, -1.0],
            [0.5, 0.5, -1.0, 1.0],
        ]
    )
    out = axial_contract(x, lat, 1, near_cancel, valid=mask)
    assert torch.isfinite(out).all()
    assert out.abs().max() < 10 * x.abs().max(), out.abs().max().item()


def test_a_genuinely_small_mass_still_renormalizes_exactly():
    """The relative guard must not overreach: a tiny but uncancelled mass
    divides out exactly, because the numerator carries the same factor."""
    lat = Lattice(shape=(3,), valid=torch.tensor([True, False, False]))
    mask = lat.mask().to(torch.float64)
    x = torch.randn(2, 3, 4, dtype=torch.float64) * mask
    tiny = torch.full((3, 3), 1e-6, dtype=torch.float64)  # small, all-positive
    out = axial_contract(x, lat, 0, tiny, valid=mask)
    # one present cell, mass 1e-6, numerator 1e-6 * x -> renormalizes to x
    assert torch.allclose(out[:, 0], x[:, 0], atol=1e-9)


def test_kron_operator_refuses_an_empty_kernel_list():
    with pytest.raises(ValueError, match="at least one kernel"):
        kron_operator([])
