"""Every mixer against every method — the claim, checked as a matrix.

The library's thesis is one sentence: *an N-D model is a 1-D mixer plus a plan
for sweeping it over an N-D lattice*. If that is true then any mixer composes
with any method, and the product of the two lists is the product the library
ships. Until now that was spot-checked — each mixer was tested under its usual
composition, each method under a convenient mixer — and the diagonal of a
matrix is not the matrix. 13 mixers x 4 methods is 52 cells, and every one of
them is a configuration a user can write in a single line.

The invariants here are the ones that must hold for *any* combination, and
they are deliberately few, because this file's job is breadth: shape, finite
values, gradients reaching the mixer, absent cells staying inert, and batch
independence. Depth belongs in the per-family files, which know what each
mechanism additionally claims.

All 52 run in about 13 seconds, so the matrix is exhaustive rather than
sampled. A cell that cannot work should be a refusal with a reason, not a
silent absence from the test suite.
"""

from __future__ import annotations

import itertools

import pytest
import torch

import torch_dimensions as td
from torch_dimensions import mixers as M

pytest.importorskip("einops", reason="the vendored mixers need the [upstream] extra")
pytest.importorskip("hydra", reason="the s4 pipeline needs hydra-core")

D_MODEL = 32

MIXERS = [
    "LSTMMixer",
    "GRUMixer",
    "S4DMixer",
    "S4Mixer",
    "MambaMixer",
    "UpstreamS4DMixer",
    "UpstreamS4Mixer",
    "UpstreamMambaMixer",
    "UpstreamMamba2Mixer",
    "Mamba3Mixer",
    "AttentionMixer",
    "ConvMixer",
    "TCNMixer",
]
METHODS = ["axial_scan", "axial_attention", "cafa", "flatten"]
CELLS = list(itertools.product(MIXERS, METHODS))

# The head layout of the Mamba-2/3 blocks constrains d_model; nothing else
# needs an argument, which is itself part of the claim.
EXTRA = {"UpstreamMamba2Mixer": {"headdim": 16}, "Mamba3Mixer": {"headdim": 16}}


def lattice(sparse: bool = True) -> td.Lattice:
    if not sparse:
        return td.Lattice(shape=(4, 5), names=("h", "w"), time=True)
    gen = torch.Generator().manual_seed(5)
    valid = torch.rand(4, 5, generator=gen) > 0.3
    valid[0, 0] = True
    return td.Lattice(shape=(4, 5), names=("h", "w"), valid=valid, time=True)


def build(mixer: str, method: str, lat: td.Lattice):
    cls = getattr(M, mixer)
    kw = EXTRA.get(mixer, {})
    torch.manual_seed(0)
    return td.LSTM(  # the host class only supplies the composition; `mixer=` is the mechanism
        D_MODEL,
        2,
        lat,
        d_input=1,
        mixer=lambda d, _c=cls, _k=kw: _c(d, **_k),
        method=getattr(td, method),
    )


def inputs(lat: td.Lattice, batch: int = 2, t: int = 3) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(batch, t, *lat.shape, 1)


@pytest.mark.parametrize("mixer,method", CELLS, ids=[f"{m}-{s}" for m, s in CELLS])
def test_the_cell_runs_and_keeps_the_lattice_shape(mixer, method):
    lat = lattice()
    model = build(mixer, method, lat).eval()
    x = inputs(lat)
    with torch.no_grad():
        y = model(x)
    assert y.shape[:-1] == x.shape[:-1]
    assert y.shape[-1] == D_MODEL
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("mixer,method", CELLS, ids=[f"{m}-{s}" for m, s in CELLS])
def test_gradients_reach_the_mixer(mixer, method):
    """Not merely that a backward pass runs: that it reaches the *mixer*. A
    composition that dropped the mixer out of the graph — folding it away, or
    detaching somewhere in the axis bookkeeping — would still train the
    projections and still show a falling loss."""
    lat = lattice()
    model = build(mixer, method, lat)
    model(inputs(lat)).pow(2).mean().backward()

    inner = [p for name, p in model.named_parameters() if name.startswith("nd.")]
    assert inner, "the composition exposed no parameters of its own"
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in inner), (
        "no gradient reached any parameter inside the composition"
    )


@pytest.mark.parametrize("mixer,method", CELLS, ids=[f"{m}-{s}" for m, s in CELLS])
def test_absent_cells_stay_inert_in_every_cell_of_the_matrix(mixer, method):
    """The sparse guarantee is a property of the *composition*, so it has to
    survive every mixer put through it — including the ones whose own code
    knows nothing about masks."""
    lat = lattice()
    model = build(mixer, method, lat).eval()
    mask = lat.mask(torch.float32)
    x = inputs(lat) * mask
    noise = torch.randn_like(x) * 50 * (1 - mask)

    with torch.no_grad():
        assert torch.equal(model(x), model(x + noise))


@pytest.mark.parametrize("mixer,method", CELLS, ids=[f"{m}-{s}" for m, s in CELLS])
def test_batch_independence_in_every_cell_of_the_matrix(mixer, method):
    lat = lattice()
    model = build(mixer, method, lat).eval()
    x = inputs(lat)
    poisoned = x.clone()
    poisoned[1:] += 100.0
    with torch.no_grad():
        assert torch.equal(model(x)[0], model(poisoned)[0])


@pytest.mark.parametrize("mixer,method", CELLS, ids=[f"{m}-{s}" for m, s in CELLS])
def test_a_dense_lattice_works_in_every_cell(mixer, method):
    lat = lattice(sparse=False)
    model = build(mixer, method, lat).eval()
    with torch.no_grad():
        assert torch.isfinite(model(inputs(lat))).all()


def test_the_matrix_is_complete():
    """The count itself is the claim. If a mixer or a method is added and this
    file is not, the product silently stops being the product — so the size is
    asserted against the exported lists rather than against a literal."""
    exported_mixers = [n for n in M.__all__ if n.endswith("Mixer") and not n.startswith("Mixer")]
    assert sorted(MIXERS) == sorted(exported_mixers), (
        "the matrix does not cover every exported mixer; "
        f"missing {sorted(set(exported_mixers) - set(MIXERS))}"
    )
    # `ND_METHODS` is a registry users extend at runtime — `register_nd_method`
    # is public API, and the guide in docs/adding-a-method.md adds "pyramid" to
    # it for real. So the claim cannot be equality with the registry; it is
    # equality with the methods the *library* ships, which are exactly those
    # the registry maps to a `td.`-exported callable of the same name.
    shipped = sorted(n for n in td.ND_METHODS if getattr(td, n, None) is td.ND_METHODS[n])
    assert sorted(METHODS) == shipped, (
        f"the matrix does not cover every method the library ships; missing "
        f"{sorted(set(shipped) - set(METHODS))}"
    )
    assert len(CELLS) == len(MIXERS) * len(METHODS)
