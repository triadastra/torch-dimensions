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


# -- the module-level Kronecker claim ------------------------------------------


def cafa_kernels(block, x):
    """Run one CaFA layer's contractions and hand back the operators it used.

    This is the adapter shape `td.testing.check_block(kernels=...)` asks for,
    and it is written here rather than in the library because it necessarily
    knows a block's internals.

    Note which matrices come back: CaFA pools *the current activation*, so the
    kernel for the second axis is built from the output of the first axis's
    contraction. The Kronecker factors are therefore the ones actually applied,
    not the ones a static reading of the module would predict — which is
    exactly why the identity is worth testing rather than asserting.
    """
    from torch_dimensions.compose.kernel import axial_contract

    h = x
    mats = []
    for j, axis in enumerate(block.spatial_axes):
        kernel = block._kernel(0, j, axis, h)
        lines = kernel.reshape(-1, kernel.shape[-2], kernel.shape[-1])
        assert torch.allclose(lines, lines[0].expand_as(lines)), (
            "pooled kernels differ between lines of the same batch element; "
            "the joint-operator comparison assumes one operator per axis"
        )
        mats.append(lines[0])
        h = axial_contract(h, block.lattice, axis, kernel)
    return mats, h


def test_cafa_contraction_is_the_kronecker_product_it_claims_to_be():
    """The kernel family's central claim, checked at module level rather than
    on hand-built matrices: contracting axis by axis equals applying the single
    joint operator `A_0 ⊗ A_1 ⊗ …`. Until now this check was an unconditional
    skip in the conformance report."""

    def build(lat, d_model, plan=None):
        return td.AxialKernel(
            mixer=None,
            plan=plan or td.ScanPlan.cyclic(lat.axis_names, len(lat.axis_names)),
            lattice=lat,
            d_model=d_model,
            per_line=False,
            norm=False,
            residual=False,
        )

    report = td.testing.check_block(build, ranks=(2, 3), kernels=cafa_kernels, d_model=4)
    assert report, str(report)
    assert (
        td.testing.check_block(build, ranks=(2,), kernels=cafa_kernels, raise_on_failure=False)
        .results[3]
        .status
        == "pass"
    )


def test_the_kronecker_check_catches_a_contraction_that_is_not_a_product():
    """Negative control: an adapter that reports the wrong operators must fail
    the check. A conformance check that has never failed is a comment."""

    def build(lat, d_model, plan=None):
        return td.AxialKernel(
            mixer=None,
            plan=plan or td.ScanPlan.cyclic(lat.axis_names, len(lat.axis_names)),
            lattice=lat,
            d_model=d_model,
            per_line=False,
            norm=False,
            residual=False,
        )

    def wrong(block, x):
        mats, out = cafa_kernels(block, x)
        return [m * 1.5 for m in mats], out

    report = td.testing.check_block(
        build, ranks=(2,), kernels=wrong, d_model=4, raise_on_failure=False
    )
    assert not report
    assert "Kronecker" in report.failed[0].name


# -- options taken from the CaFA reference implementation ---------------------


def test_the_new_kernel_options_default_to_the_old_behaviour():
    """Both are off by default, so every existing model and checkpoint is
    numerically unchanged by their arrival."""
    lat = td.Lattice(shape=(4, 5), names=("h", "w"), time=True)
    x = torch.randn(2, 3, 4, 5, 8, dtype=torch.float64)
    torch.manual_seed(0)
    plain = td.LSTM(8, 3, lat, method=td.cafa).double().eval()
    torch.manual_seed(0)
    explicit = (
        td.LSTM(8, 3, lat, method=td.cafa, qk_norm=False, kernel_residual=False).double().eval()
    )
    assert torch.equal(plain(x), explicit(x))


@pytest.mark.parametrize("strategy", [td.cafa, td.axial_attention])
def test_qk_norm_changes_the_kernel_and_keeps_the_block_working(strategy):
    lat = td.Lattice(shape=(4, 5), names=("h", "w"), time=True)
    x = torch.randn(2, 3, 4, 5, 8, dtype=torch.float64)
    torch.manual_seed(0)
    off = td.LSTM(8, 3, lat, method=strategy).double().eval()
    torch.manual_seed(0)
    on = td.LSTM(8, 3, lat, method=strategy, qk_norm=True).double().eval()
    assert not torch.allclose(off(x), on(x))
    assert on(x).shape == x.shape
    # Same parameter count: RMS normalization is learnable-free on purpose.
    assert sum(p.numel() for p in off.parameters()) == sum(p.numel() for p in on.parameters())


def test_the_kernel_residual_starts_the_contraction_near_the_identity():
    """CaFA's `K + gamma*I`. The point is the inductive bias: with the residual
    a freshly built contraction is closer to leaving each cell alone than to
    averaging the axis, and it has to learn to mix."""
    lat = td.Lattice(shape=(6,), names=("h",))
    block = td.AxialKernel(
        mixer=None,
        plan=td.ScanPlan.cyclic(("h",), 1),
        lattice=lat,
        d_model=8,
        gate="softmax",
        kernel_residual=True,
        norm=False,
        residual=False,
    ).double()
    with torch.no_grad():
        block.gamma[0].fill_(50.0)  # a large gamma is a near-perfect identity
        h = torch.randn(1, 6, 8, dtype=torch.float64)
        kernel = block._kernel(0, 0, 0, h)
    eye = torch.eye(6, dtype=torch.float64)
    assert (kernel - eye).abs().max().item() < 1e-6, kernel


def test_the_kernel_residual_is_added_before_the_gate():
    """Order matters and is not a detail: softmax is not additive, so adding
    the identity after it would be a different operator (and would break the
    rows-sum-to-one property the softmax gate exists for)."""
    lat = td.Lattice(shape=(5,), names=("h",))
    block = td.AxialKernel(
        mixer=None,
        plan=td.ScanPlan.cyclic(("h",), 1),
        lattice=lat,
        d_model=8,
        gate="softmax",
        kernel_residual=True,
        norm=False,
        residual=False,
    ).double()
    with torch.no_grad():
        block.gamma[0].fill_(3.0)
        kernel = block._kernel(0, 0, 0, torch.randn(1, 5, 8, dtype=torch.float64))
    rows = kernel.sum(-1)
    assert torch.allclose(rows, torch.ones_like(rows), atol=1e-12), rows


@pytest.mark.parametrize("kw", [{"qk_norm": True}, {"kernel_residual": True}])
def test_the_new_options_pass_conformance_and_learn(kw):
    def factory(lat, d_model, plan=None):
        return td.LSTM(d_model, lat.n_axes, lat, plan=plan, method=td.cafa, **kw)

    report = td.testing.check_block(factory, ranks=(1, 2, 3), time=True)
    assert report, str(report)
    stats = td.testing.check_trainable(
        lambda lat, d: td.LSTM(d, 4, lat, method=td.cafa, **kw), steps=150
    )
    assert stats["ratio"] > 3.0, stats
