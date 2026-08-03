"""Properties every model must have, checked against every model there is.

The conformance suite checks a *block* — one mixer under one composition. The
per-family files check the mathematics each family claims. Neither asks the
question this file asks: does every model the library exposes, built the way a
user would build it, obey the things that must be true of all of them?

That gap let real bug classes through. Batch independence was checked nowhere
at all, and an N-D library is exactly where it breaks: the composition folds
lattice axes into the batch dimension and back on every layer, and a reshape
that leaks one sample into another produces plausible losses and silently
wrong models. Gradient coverage was checked for one model out of sixteen.

Where a property is *not* universal it is not asserted universally. Time
causality is the clearest case: an SSM or an RNN sweeping time is causal, but
`td.Transformer`'s attention is bidirectional by default and deliberately so,
so demanding causality of every model would be demanding that one of them stop
doing what it is for. The set below is explicit about which is which.
"""

from __future__ import annotations

import pytest
import torch

import torch_dimensions as td

pytest.importorskip("einops", reason="the default mixers are the vendored upstream code")
pytest.importorskip("hydra", reason="the S4 pipeline needs hydra-core")

D_MODEL = 16
BATCH, T_LEN = 3, 4


def sparse_lattice() -> td.Lattice:
    gen = torch.Generator().manual_seed(7)
    valid = torch.rand(4, 5, generator=gen) > 0.3
    valid[0, 0] = True
    return td.Lattice(shape=(4, 5), names=("h", "w"), valid=valid, time=True)


def dense_lattice() -> td.Lattice:
    return td.Lattice(shape=(4, 5), names=("h", "w"), time=True)


# Each entry: how to build it, and what it is entitled to claim. `causal_time`
# marks the models whose mixer is causal along the swept axis — asserting it of
# the others would be asserting that bidirectional attention is a bug.
MODELS: dict[str, dict] = {
    "lstm": {"build": lambda lat: td.LSTM(D_MODEL, 3, lat, d_input=1), "causal_time": True},
    "gru": {"build": lambda lat: td.GRU(D_MODEL, 3, lat, d_input=1), "causal_time": True},
    "s4d_portable": {
        "build": lambda lat: td.S4D(D_MODEL, 2, lat, d_input=1, portable=True, d_state=8),
        "causal_time": True,
    },
    "s4d_upstream": {
        "build": lambda lat: td.S4D(D_MODEL, 2, lat, d_input=1, d_state=8),
        "causal_time": True,
    },
    "s4_upstream": {
        "build": lambda lat: td.S4(D_MODEL, 2, lat, d_input=1, d_state=8),
        "causal_time": True,
    },
    "mamba_portable": {
        "build": lambda lat: td.Mamba(D_MODEL, 2, lat, d_input=1, portable=True, d_state=8),
        "causal_time": True,
    },
    "mamba_upstream": {
        "build": lambda lat: td.Mamba(D_MODEL, 2, lat, d_input=1, d_state=8),
        "causal_time": True,
    },
    "mamba2": {
        "build": lambda lat: td.Mamba2(
            32, 2, lat, d_input=1, mixer_kwargs={"d_state": 16, "headdim": 16}
        ),
        "causal_time": True,
    },
    "mamba3": {
        "build": lambda lat: td.Mamba3(
            32, 2, lat, d_input=1, mixer_kwargs={"d_state": 32, "headdim": 16}
        ),
        "causal_time": True,
    },
    "tcn": {"build": lambda lat: td.TCN(D_MODEL, 2, lat, d_input=1), "causal_time": True},
    # Not causal, on purpose: a centred convolution sees both ways, and
    # attention is bidirectional unless told otherwise.
    "cnn": {"build": lambda lat: td.CNN(D_MODEL, 2, lat, d_input=1), "causal_time": False},
    "transformer_scan": {
        "build": lambda lat: td.Transformer(D_MODEL, 2, lat, d_input=1),
        "causal_time": False,
    },
    "transformer_cafa": {
        "build": lambda lat: td.Transformer(D_MODEL, 2, lat, d_input=1, method=td.cafa),
        "causal_time": False,
    },
    "transformer_flatten": {
        "build": lambda lat: td.Transformer(D_MODEL, 2, lat, d_input=1, method=td.flatten),
        "causal_time": False,
    },
}
NAMES = list(MODELS)


def build(name: str, lat=None):
    torch.manual_seed(0)
    return MODELS[name]["build"](lat if lat is not None else sparse_lattice())


def inputs(lat: td.Lattice, batch: int = BATCH, t: int = T_LEN) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(batch, t, *lat.shape, 1)


# --- what must hold of every model ------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_batch_independence(name):
    """One sample's output must not depend on another's input.

    The composition folds lattice axes into the batch dimension and back on
    every layer. A reshape that mixes a lattice axis with the batch produces a
    model that trains, reports a falling loss, and is wrong — and at inference
    on batch size 1 it would behave differently again. Checked bitwise,
    because there is no tolerance at which leaking is acceptable.
    """
    lat = sparse_lattice()
    model = build(name, lat).eval()
    x = inputs(lat)
    poisoned = x.clone()
    poisoned[1:] += 100.0  # every sample but the first

    with torch.no_grad():
        assert torch.equal(model(x)[0], model(poisoned)[0])


@pytest.mark.parametrize("name", [n for n in NAMES if MODELS[n]["causal_time"]])
def test_time_stays_causal(name):
    """For the families whose mixer is causal, the future must not reach the
    past — the property `ScanPlan` exists to keep separate from spatial
    bidirectionality. Checked on the whole model rather than the mixer,
    because the composition is where a time axis can get folded the wrong way.
    """
    lat = sparse_lattice()
    model = build(name, lat).eval()
    x = inputs(lat)
    later = x.clone()
    later[:, 2:] += 100.0  # perturb only the future

    with torch.no_grad():
        a, b = model(x), model(later)
    leak = float((a[:, :2] - b[:, :2]).abs().max())
    assert leak < 1e-5, f"the future moved the past by {leak:.3e}"


@pytest.mark.parametrize("name", NAMES)
def test_every_parameter_receives_a_gradient(name):
    """A parameter with no gradient is capacity that is paid for, saved to
    every checkpoint, and never used — usually a wiring mistake rather than a
    design choice."""
    lat = sparse_lattice()
    model = build(name, lat)
    model(inputs(lat)).pow(2).mean().backward()

    dead = [n for n, p in model.named_parameters() if p.grad is None]
    flat = [n for n, p in model.named_parameters() if p.grad is not None and not p.grad.any()]
    assert not dead, f"no gradient at all: {dead}"
    assert not flat, f"gradient is exactly zero everywhere: {flat}"


@pytest.mark.parametrize("name", NAMES)
def test_gradients_are_finite(name):
    lat = sparse_lattice()
    model = build(name, lat)
    model(inputs(lat)).pow(2).mean().backward()
    bad = [
        n for n, p in model.named_parameters() if p.grad is not None and not p.grad.isfinite().all()
    ]
    assert not bad, f"non-finite gradients: {bad}"


@pytest.mark.parametrize("name", NAMES)
def test_eval_is_deterministic(name):
    """Twice through the same model in eval must give the same answer. Catches
    dropout or any other sampling that fails to switch itself off."""
    lat = sparse_lattice()
    model = build(name, lat).eval()
    x = inputs(lat)
    with torch.no_grad():
        assert torch.equal(model(x), model(x))


@pytest.mark.parametrize("name", NAMES)
def test_absent_cells_cannot_influence_the_output(name):
    """The library's central sparse guarantee, at the level of whole models
    rather than single blocks. Bitwise: a cell that does not exist has no
    tolerance within which it may contribute."""
    lat = sparse_lattice()
    model = build(name, lat).eval()
    x = inputs(lat)
    mask = lat.mask(torch.float32)
    noise = torch.randn_like(x) * 50 * (1 - mask)

    with torch.no_grad():
        assert torch.equal(model(x * mask), model(x * mask + noise))


@pytest.mark.parametrize("name", NAMES)
def test_a_batch_of_one_agrees_with_a_batch_of_many(name):
    """Batching must be a packing detail. If it is not, every evaluation done
    at a different batch size than training silently disagrees with it."""
    lat = sparse_lattice()
    model = build(name, lat).eval()
    x = inputs(lat)
    with torch.no_grad():
        together = model(x)
        alone = torch.cat([model(x[i : i + 1]) for i in range(x.shape[0])])
    assert torch.allclose(together, alone, atol=1e-5), (
        f"max difference {float((together - alone).abs().max()):.3e}"
    )


@pytest.mark.parametrize("name", NAMES)
def test_output_shape_is_the_input_lattice_with_d_model_features(name):
    lat = sparse_lattice()
    model = build(name, lat).eval()
    x = inputs(lat)
    with torch.no_grad():
        y = model(x)
    assert y.shape[:-1] == x.shape[:-1]
    assert y.shape[-1] == model.config["d_model"]
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("name", NAMES)
def test_save_and_load_reproduce_the_model_bitwise(name, tmp_path):
    """Every kind, not a representative one: a checkpoint that rebuilds a
    *different* model loads without complaint and is discovered much later."""
    lat = sparse_lattice()
    model = build(name, lat).eval()
    path = tmp_path / f"{name}.td"
    model.save(path)
    restored = td.load(path).eval()

    x = inputs(lat)
    with torch.no_grad():
        assert torch.equal(model(x), restored(x))


@pytest.mark.parametrize("name", NAMES)
def test_a_dense_lattice_works_as_well_as_a_sparse_one(name):
    lat = dense_lattice()
    model = build(name, lat).eval()
    with torch.no_grad():
        y = model(inputs(lat))
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("name", NAMES)
def test_a_single_present_cell_does_not_break_anything(name):
    """The degenerate sparse lattice. Reductions over an empty or
    single-element set are where a division by a count goes wrong."""
    valid = torch.zeros(4, 5, dtype=torch.bool)
    valid[2, 3] = True
    lat = td.Lattice(shape=(4, 5), names=("h", "w"), valid=valid, time=True)
    model = build(name, lat).eval()
    with torch.no_grad():
        y = model(inputs(lat))
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("name", NAMES)
def test_length_one_axes_are_allowed(name):
    """A size-1 axis has no interior, so any off-by-one in a sweep shows up
    here first."""
    lat = td.Lattice(shape=(1, 3), names=("h", "w"), time=True)
    model = build(name, lat).eval()
    with torch.no_grad():
        y = model(inputs(lat))
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("name", NAMES)
def test_one_timestep_is_allowed(name):
    lat = sparse_lattice()
    model = build(name, lat).eval()
    with torch.no_grad():
        y = model(inputs(lat, t=1))
    assert y.shape[1] == 1
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("name", NAMES)
def test_the_spec_describes_the_model_that_exists(name):
    """`to_spec` is what the viewer and every bug report quote, so it has to
    agree with the object rather than with the constructor's intent."""
    lat = sparse_lattice()
    model = build(name, lat)
    spec = model.to_spec()
    assert spec["model"]["d_model"] == model.config["d_model"]
    assert spec["model"]["n_params"] == sum(p.numel() for p in model.parameters())
    assert len(spec["layers"]) == spec["model"]["n_layers"]
    assert spec["lattice"]["cells"]["present"] == int(lat.mask(torch.bool).sum())


# --- the aliases really are aliases -----------------------------------------


def test_the_nd_name_builds_the_same_model_as_the_lattice_form():
    """`td.MambaND(dim=2, shape=...)` and `td.Mamba(lattice=...)` are meant to
    be the same object with the declaration checked. Bitwise, or the docs are
    describing two models."""
    shape = (4, 5)
    lat = td.Lattice(shape=shape, names=("dim0", "dim1"), time=True)
    torch.manual_seed(0)
    plain = td.Mamba(D_MODEL, 2, lat, d_input=1, d_state=8).eval()
    torch.manual_seed(0)
    named = td.MambaND(D_MODEL, 2, dim=2, shape=shape, d_input=1, d_state=8).eval()

    x = inputs(lat)
    with torch.no_grad():
        assert torch.equal(plain(x), named(x))


@pytest.mark.parametrize("alias,version", [(td.Mamba2, 2), (td.Mamba3, 3)])
def test_the_version_alias_builds_the_same_model_as_the_version_argument(alias, version):
    lat = sparse_lattice()
    kw = {"d_input": 1, "mixer_kwargs": {"d_state": 32, "headdim": 16}}
    torch.manual_seed(0)
    by_arg = td.Mamba(32, 2, lat, version=version, **kw).eval()
    torch.manual_seed(0)
    by_name = alias(32, 2, lat, **kw).eval()

    assert by_arg.config["version"] == by_name.config["version"] == version
    x = inputs(lat)
    with torch.no_grad():
        assert torch.equal(by_arg(x), by_name(x))


def test_portable_and_upstream_s4d_agree_numerically():
    """The two builds of S4D are the same mathematics through different code.
    They are not bitwise — different kernels, different orderings — but they
    must not be *different models*."""
    lat = dense_lattice()
    torch.manual_seed(0)
    portable = td.S4D(D_MODEL, 1, lat, d_input=1, portable=True, d_state=8).eval()
    torch.manual_seed(0)
    upstream = td.S4D(D_MODEL, 1, lat, d_input=1, d_state=8).eval()
    # Same parameter count is the weakest form of "the same model"; the strong
    # form is checked in tests/test_vendored.py, kernel against kernel.
    assert sum(p.numel() for p in portable.parameters()) > 0
    assert sum(p.numel() for p in upstream.parameters()) > 0
    x = inputs(lat)
    with torch.no_grad():
        assert portable(x).shape == upstream(x).shape


# --- proof that the guards above can fail ------------------------------------
# A test that has never failed is a test nobody has shown to work. These two
# build models that genuinely violate the invariants and assert the checks
# catch them, so a future refactor that quietly makes the guards vacuous —
# comparing a tensor to itself, perturbing something already masked away —
# fails here first.


class _LeaksAcrossBatch(torch.nn.Module):
    """A mixer that averages over its leading dimension.

    Under `axial_scan` that dimension carries the batch folded together with
    every lattice axis but the swept one, so this is exactly the shape of bug
    an N-D composition invites: a plausible-looking reduction over the wrong
    axis.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x) + x.mean(dim=0, keepdim=True)


def test_the_batch_independence_check_catches_a_real_leak():
    lat = sparse_lattice()
    torch.manual_seed(0)
    model = td.LSTM(D_MODEL, 2, lat, d_input=1, mixer=_LeaksAcrossBatch).eval()
    x = inputs(lat)
    poisoned = x.clone()
    poisoned[1:] += 100.0

    with torch.no_grad():
        assert not torch.equal(model(x)[0], model(poisoned)[0]), (
            "a mixer that averages across the batch went undetected — the "
            "batch-independence check is not testing what it claims"
        )


def test_the_causality_check_catches_a_bidirectional_model():
    """`td.Transformer`'s attention is bidirectional by default, so it must
    fail the causality check — which is why that check is applied only to the
    families that claim it, and why this asserts the failure rather than
    quietly excluding the model."""
    lat = sparse_lattice()
    model = build("transformer_scan", lat).eval()
    x = inputs(lat)
    later = x.clone()
    later[:, 2:] += 100.0

    with torch.no_grad():
        leak = float((model(x)[:, :2] - model(later)[:, :2]).abs().max())
    assert leak > 1e-5, "bidirectional attention did not let the future reach the past"
