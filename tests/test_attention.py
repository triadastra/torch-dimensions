"""Phase 6 acceptance: the kernel-family methods of multidimensionality.

``td.axial_attention`` and ``td.cafa`` as ``nd_method`` strategies, including
the hybrid form — kernels over the spatial axes, the model's own mixer along
time. The load-bearing property is causality along time: the kernels act
within a timestep, so a causal mixer must keep the whole hybrid causal, and a
CaFA kernel pooled across time would silently break exactly that.
"""

import pytest
import torch

import torch_dimensions as td

STRATEGIES = [td.axial_attention, td.cafa]


def _factory(strategy, **method_kwargs):
    def build(lat, d_model, plan=None):
        return td.LSTM(d_model, lat.n_axes, lat, nd_method=strategy, plan=plan, **method_kwargs)

    return build


# -- conformance ---------------------------------------------------------------


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_kernel_family_passes_every_applicable_check(strategy):
    report = td.testing.check_block(_factory(strategy), ranks=(1, 2, 3), time=True)
    assert report, str(report)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_kernel_family_with_leaky_relu_gate_passes_on_sparse(strategy):
    """The CaFA paper's default gate is signed, which is exactly the case the
    relative-cancellation guard exists for."""
    report = td.testing.check_block(_factory(strategy, gate="leaky_relu"), ranks=(2,), time=True)
    assert report, str(report)


# -- the hybrid stays causal along time ---------------------------------------


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_hybrid_is_causal_along_time(strategy):
    """Spatial kernels act within a timestep and the LSTM is causal, so the
    past must not see a perturbed future. For CaFA this is the test that
    forbids pooling over time when building the kernel."""
    torch.manual_seed(0)
    lat = td.Lattice(shape=(3, 4), names=("h", "w"), time=True)
    model = td.LSTM(8, 3, lat, nd_method=strategy).double().eval()
    x = torch.randn(2, 6, 3, 4, 8, dtype=torch.float64)
    future = x.clone()
    future[:, 4:] += 100.0
    a, b = model(x), model(future)
    assert torch.equal(a[:, :4], b[:, :4]), "the past changed when the future did"
    assert not torch.equal(a[:, 4:], b[:, 4:])


# -- learning ------------------------------------------------------------------


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_kernel_family_learns_a_task_that_needs_axial_mixing(strategy):
    result = td.testing.check_trainable(_factory(strategy), steps=150)
    assert result["ratio"] >= 3.0, result


# -- wiring --------------------------------------------------------------------


def test_a_mixer_without_a_time_axis_is_refused():
    """On a purely spatial lattice the kernels are the whole model; accepting
    a mixer would carry dead parameters that the user believes are running."""
    lat = td.Lattice(shape=(3, 4))
    with pytest.raises(ValueError, match="dead\\s*weight|no time axis"):
        td.LSTM(8, 2, lat, nd_method=td.cafa)


def test_kernel_only_block_works_without_a_mixer():
    lat = td.Lattice(shape=(3, 4))
    block = td.AxialKernel(
        mixer=None, plan=td.ScanPlan.cyclic(lat.axis_names, 2), lattice=lat, d_model=8
    )
    assert block(torch.randn(2, 3, 4, 8)).shape == (2, 3, 4, 8)


def test_the_methods_are_registered_by_name():
    assert td.resolve_nd_method("cafa") is td.cafa
    assert td.resolve_nd_method("axial_attention") is td.axial_attention
    assert td.ND_METHODS["axial_scan"] is td.axial_scan


def test_method_is_the_short_spelling_of_nd_method():
    lat = td.Lattice(shape=(2, 3), time=True)
    model = td.S4D(8, 2, lat, method=td.cafa)
    assert type(model.nd).__name__ == "AxialKernel"
    with pytest.raises(ValueError, match="not both"):
        td.S4D(8, 2, lat, method=td.cafa, nd_method=td.cafa)


def test_an_unknown_gate_is_refused():
    lat = td.Lattice(shape=(2, 3), time=True)
    with pytest.raises(ValueError, match="gate"):
        td.LSTM(8, 2, lat, nd_method=td.cafa, gate="sigmoid")
