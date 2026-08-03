"""`td.viz.weights` — the parameters, in a form a diagram can draw.

The spec says what the architecture is; this says what its weights hold. The
tests below are mostly about two things the viewer depends on and cannot check
for itself: that every tensor is classified by the *role* it plays — which is
what decides whether a diagram draws a bipartite graph, a receptive field or a
bank of decaying states — and that downsampling is always declared, because a
picture of one corner of a matrix presented as the matrix is worse than no
picture.
"""

from __future__ import annotations

import json

import pytest
import torch

import torch_dimensions as td

LAT = td.Lattice(shape=(4, 5), names=("h", "w"))


def _roles(payload, layer=0):
    return {t["role"] for t in payload["layers"][layer]["tensors"]}


def _by_name(payload, name, layer=0):
    return next(t for t in payload["layers"][layer]["tensors"] if t["name"] == name)


def test_the_payload_is_json_able_and_names_its_format():
    payload = td.viz.weights(td.LSTM(16, 2, LAT))
    assert payload["format"] == "torch-dimensions/weights"
    assert payload["version"] >= 1
    json.dumps(payload)  # must not raise: tensors have to be plain lists by here


def test_one_entry_per_layer_with_the_mixer_named():
    model = td.LSTM(16, 3, LAT)
    payload = td.viz.weights(model)
    assert [entry["layer"] for entry in payload["layers"]] == [0, 1, 2]
    assert {entry["mixer"] for entry in payload["layers"]} == {"LSTMMixer"}


def test_a_convolution_is_labelled_a_convolution_not_a_matrix():
    """Role comes from the owning module, so a conv kernel stays a conv kernel
    however its attribute happens to be named."""
    payload = td.viz.weights(td.CNN(16, 1, LAT))
    assert "conv" in _roles(payload)


def test_ssm_parts_are_told_apart():
    """S4D's decay, output map and skip are three different things in the
    diagram; collapsing them to 'parameter' would draw one picture for all."""
    payload = td.viz.weights(td.S4D(16, 1, LAT, portable=True, d_state=8))
    roles = _roles(payload)
    assert "ssm_decay" in roles  # log_A_real / A_imag
    assert "ssm_out" in roles  # C
    assert "skip" in roles  # D


def test_mamba_carries_conv_ssm_and_projections_together():
    payload = td.viz.weights(td.Mamba(16, 1, LAT, portable=True, d_state=8))
    roles = _roles(payload)
    assert {"conv", "ssm_decay", "linear", "skip"} <= roles


def test_attention_projections_are_linear():
    payload = td.viz.weights(td.Transformer(16, 1, LAT))
    assert "linear" in _roles(payload)
    qkv = _by_name(payload, "qkv.weight")
    assert qkv["shape"][1] == 16


def test_a_small_tensor_is_drawn_whole_and_says_so():
    model = td.CNN(8, 1, LAT, mixer_kwargs={"kernel_size": 3})
    payload = td.viz.weights(model, max_units=64)
    conv = next(t for t in payload["layers"][0]["tensors"] if t["role"] == "conv")
    assert conv["sampled"] is False
    assert conv["stride"] == [1, 1]


def test_a_large_tensor_is_strided_and_the_stride_is_reported():
    payload = td.viz.weights(td.Transformer(64, 1, LAT), max_units=8)
    big = _by_name(payload, "qkv.weight")
    assert big["sampled"] is True
    assert big["rows"] <= 8 and big["cols"] <= 8
    assert big["stride"][0] > 1
    # The original shape survives, so the drawing can say what it is a sample of.
    assert big["shape"] == [192, 64]
    assert big["stats"]["n"] == 192 * 64


def test_drawn_values_are_actually_the_model_s_weights():
    """A strided sample, not a summary: the values must match the tensor at the
    positions they claim to come from."""
    torch.manual_seed(0)
    model = td.Transformer(32, 1, LAT)
    payload = td.viz.weights(model, max_units=8)
    entry = _by_name(payload, "proj.weight")
    real = dict(model.nd.mixers[0].named_parameters())["proj.weight"]
    rs, cs = entry["stride"]
    expected = real[::rs, ::cs].detach()
    got = torch.tensor(entry["values"])
    assert got.shape == expected.shape
    assert torch.allclose(got, expected, atol=1e-4)


def test_conv_kernels_keep_their_taps_as_columns():
    """A (out, in, k) kernel folds to (out*in, k) so every tap survives —
    averaging them away would erase the receptive field the diagram draws."""
    payload = td.viz.weights(td.CNN(8, 1, LAT, mixer_kwargs={"kernel_size": 5}), max_units=64)
    conv = next(t for t in payload["layers"][0]["tensors"] if t["role"] == "conv")
    assert conv["cols"] == 5


def test_stats_describe_the_whole_tensor_not_the_sample():
    payload = td.viz.weights(td.Transformer(64, 1, LAT), max_units=4)
    entry = _by_name(payload, "qkv.weight")
    assert entry["stats"]["n"] == 192 * 64  # every weight, not the 16 drawn
    # The sample cannot contain anything larger than the whole tensor's maximum.
    drawn_max = max(max(abs(v) for v in row) for row in entry["values"])
    assert entry["stats"]["absmax"] >= drawn_max


def test_a_model_without_a_composition_is_refused():
    with pytest.raises(TypeError, match="no `nd` composition"):
        td.viz.weights(torch.nn.Linear(4, 4))


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    """A stand-in for the built viewer.

    The JSON routes are library code and have nothing to do with the JavaScript
    — but `serve` refuses to start without a bundle, so without this the whole
    server went untested wherever the viewer had not been built, which is every
    CI job that is not the viewer job. A directory with an index.html in it is
    all the static handler needs.
    """
    (tmp_path / "index.html").write_text("<!doctype html><title>stub</title>")
    monkeypatch.setattr(td.viz, "BUNDLE", tmp_path)
    return tmp_path


def test_the_server_offers_weights_for_a_model(bundle):
    import urllib.request

    server = td.viz.serve(td.LSTM(16, 2, LAT), port=0)
    try:
        url = f"http://127.0.0.1:{server.server_port}/weights.json"
        payload = json.loads(urllib.request.urlopen(url).read())
        assert payload["format"] == "torch-dimensions/weights"
        assert payload["layers"]
    finally:
        server.shutdown()


def test_the_server_says_so_when_there_are_no_weights_to_serve(bundle):
    """Opened on a spec dict there are no parameters, and 404 is the honest
    answer — the viewer then tells the user to open a model instead."""
    import urllib.error
    import urllib.request

    spec = td.LSTM(16, 1, LAT).to_spec()
    server = td.viz.serve(spec, port=0)
    try:
        url = f"http://127.0.0.1:{server.server_port}/weights.json"
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            urllib.request.urlopen(url)
        assert excinfo.value.code == 404
    finally:
        server.shutdown()


# --- the operator: one picture every family can be drawn in -------------------


def _operator(model, size=12):
    return td.viz.weights(model, operator_size=size)["layers"][0]["operator"]


def test_causal_families_reach_only_forwards():
    """A TCN and an SSM are causal by construction, and the impulse response
    has to show it: nothing above the diagonal at all."""
    for model in (
        td.TCN(32, 1, LAT),
        td.S4D(32, 1, LAT, portable=True, d_state=8),
        td.LSTM(32, 1, LAT),
    ):
        assert _operator(model)["causal"] == pytest.approx(1.0, abs=1e-3)


def test_a_centred_convolution_reaches_both_ways():
    """`ConvMixer` is centred by default, so some influence runs backwards —
    the measurement must not report it as causal."""
    assert _operator(td.CNN(32, 1, LAT))["causal"] < 0.999


def test_a_convolution_is_banded_and_tied():
    """The two properties that make a convolution a convolution rather than a
    dense map: it reaches a fixed distance, and it is the *same* kernel at
    every position."""
    op = _operator(td.CNN(32, 1, LAT, mixer_kwargs={"kernel_size": 3}))
    assert op["bandwidth"] <= 2  # kernel of 3 reaches one step either side
    assert op["tied"] > 0.9  # constant along each diagonal


def test_an_ssm_carries_influence_past_the_diagonal():
    """An SSM's tail is small but it is the whole mechanism: position t reaches
    later positions through a decaying state. `reach` is what keeps a
    threshold-based bandwidth from reporting it as a pointwise map."""
    op = _operator(td.S4D(32, 1, LAT, portable=True, d_state=16), size=16)
    assert op["reach"] > 0.01
    assert op["causal"] == pytest.approx(1.0, abs=1e-3)


def test_attention_shows_no_structure_in_its_parameters():
    """Attention's mixing is computed from the data, so an impulse about zero
    finds no fixed off-diagonal operator — which is the correct answer, not a
    failure, and the viewer says so in those words."""
    op = _operator(td.Transformer(32, 1, LAT))
    assert op["reach"] < 0.05


def test_the_operator_survives_a_mixer_whose_first_parameter_is_not_the_width():
    """MambaMixer's first parameter is `A_log` of shape (d_inner, d_state), so
    inferring the probe width from parameter shapes fed it an 8-wide impulse
    where it wanted 32 — and the probe failed silently into no diagram."""
    assert _operator(td.Mamba(32, 1, LAT, portable=True, d_state=8)) is not None


def test_the_operator_can_be_turned_off():
    payload = td.viz.weights(td.LSTM(16, 1, LAT), operator_size=0)
    assert payload["layers"][0]["operator"] is None


def test_the_digest_shows_the_trained_weights_not_a_snapshot():
    """The diagrams have to move when the model does. `weights()` reads the
    parameters at call time, so a viewer refreshing it during a run draws what
    the model holds now — checked against the live tensors, not just against
    "something changed"."""
    torch.manual_seed(0)
    lat = td.Lattice(shape=(4, 5), names=("h", "w"), time=True)
    model = td.LSTM(16, 2, lat, d_input=1)
    head = torch.nn.Linear(16, 1)
    opt = torch.optim.Adam([*model.parameters(), *head.parameters()], lr=0.05)

    def drawn():
        payload = td.viz.weights(model, max_units=8, operator_size=8)
        entry = _by_name(payload, "rnn.weight_ih_l0")
        return torch.tensor(entry["values"]), entry["stride"]

    before, _ = drawn()
    x = torch.randn(2, 5, 4, 5, 1)
    for _ in range(20):
        loss = (head(model(x)) - x.cumsum(dim=3)).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    after, (rs, cs) = drawn()

    assert not torch.allclose(before, after)  # training moved them
    live = dict(model.nd.mixers[0].named_parameters())["rnn.weight_ih_l0"]
    assert torch.allclose(after, live[::rs, ::cs].detach(), atol=1e-4)


# --- the fallback paths, which are where a digest quietly goes wrong ----------


def test_roles_fall_back_sensibly_for_unusual_names():
    """Role classification has to cope with mixers this library did not write.
    Owner type decides where it can; the name-based rules below it are the
    fallback, and an unrecognised tensor must land on a role rather than crash."""
    from torch_dimensions.viz.weights import _role

    two_d = torch.zeros(3, 3)
    one_d = torch.zeros(3)
    assert _role("something.bias", one_d, None) == "bias"
    assert _role("A_log", one_d, None) == "ssm_decay"
    # `kernel.log_dt` is a timescale, not a decay matrix, and lands on the
    # generic 1-D role — which is what S4D actually reports for it.
    assert _role("kernel.log_dt", one_d, None) == "vector"
    assert _role("D", one_d, None) == "skip"
    assert _role("gamma", one_d, None) == "vector"  # a plain 1-D parameter
    assert _role("kernel.B", two_d, None) == "ssm_in"
    assert _role("kernel.C", two_d, None) == "ssm_out"
    assert _role("A_imag", two_d, None) == "ssm_freq"  # a frequency, not a decay
    assert _role("kernel.log_A_real", two_d, None) == "ssm_decay"
    assert _role("mystery", two_d, None) == "linear"  # 2-D and nothing else fits


def test_a_mixer_that_refuses_the_probe_yields_no_operator_rather_than_raising():
    """The impulse probe feeds a shape the mixer may simply not accept. That is
    a picture that cannot be drawn, not an error that should sink the payload."""
    from torch_dimensions.viz.weights import _operator

    class Picky(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(4, 4))

        def forward(self, x):
            raise RuntimeError("not that shape")

    assert _operator(Picky(), 8, 4) is None
    # A width the composition could not supply is refused before probing.
    assert _operator(Picky(), 8, 0) is None


def test_a_mixer_that_changes_the_shape_yields_no_operator():
    """The operator only means anything if the mixer maps positions to
    positions; one that returns something else has no square to draw."""
    from torch_dimensions.viz.weights import _operator

    class Reshaper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(4, 4))

        def forward(self, x):
            return x[:, :1]

    assert _operator(Reshaper(), 8, 4) is None


def test_a_constant_mixer_has_no_structure_to_report():
    """An all-zero response divides by nothing: the guards have to hold."""
    from torch_dimensions.viz.weights import _operator

    class Zero(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(4, 4))

        def forward(self, x):
            return torch.zeros_like(x)

    op = _operator(Zero(), 6, 4)
    assert op is not None
    assert op["absmax"] == 0.0
    assert op["tied"] == 0.0  # nothing to be tied along


def test_the_kernel_family_reports_its_kernels():
    """`td.cafa` has no `mixers` at all — its per-axis kernels are the whole
    mechanism, and a payload that skipped them would report nothing."""
    lat = td.Lattice(shape=(4, 5), names=("h", "w"), time=True)
    payload = td.viz.weights(td.LSTM(16, 2, lat, method=td.cafa))
    assert payload["layers"], "a kernel-family model must still report parameters"


def test_the_server_also_serves_the_spec_and_the_static_bundle(bundle):
    """The other two routes, which were untested for the same reason."""
    import urllib.request

    server = td.viz.serve(td.LSTM(16, 2, LAT), port=0)
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        spec = json.loads(urllib.request.urlopen(f"{base}/spec.json").read())
        assert spec["layers"]
        assert b"stub" in urllib.request.urlopen(f"{base}/index.html").read()
    finally:
        server.shutdown()


def test_serving_without_a_bundle_says_which_path_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(td.viz, "BUNDLE", tmp_path / "nope")
    with pytest.raises(FileNotFoundError):
        td.viz.serve(td.LSTM(16, 1, LAT), port=0)


# --- findings: measurements a reader can act on ------------------------------


def _health(model):
    return td.viz.weights(model, operator_size=0)["layers"][0]["health"]


def test_an_ssm_reports_its_decay_rates_as_rates():
    """Not as a per-step retention. The decay a state applies is
    exp(-rate * dt) and `dt` is a *different* learned tensor: at unit dt a
    Mamba state with rate 8 reads as instant forgetting, while with its
    learned dt of ~0.01 it retains 92% a step. Reporting the rate is true;
    reporting a retention without dt is not."""
    notes = _health(td.Mamba(32, 1, LAT, portable=True, d_state=8))
    text = " ".join(n["text"] for n in notes)
    assert "decay rates span" in text
    assert "dt" in text  # the other half is named
    assert all(n["level"] == "ok" for n in notes)


def test_uniform_decay_is_not_reported_as_a_fault_when_frequencies_differ():
    """S4D-Lin gives every state the same real part on purpose and separates
    them by frequency. A check that warned here would fire on a correctly
    initialised S4D every time, and a diagnostic that cries wolf on the
    default configuration gets ignored."""
    notes = _health(td.S4D(32, 1, LAT, portable=True, d_state=16))
    assert [n["level"] for n in notes] == ["ok"]
    assert "frequency" in notes[0]["text"]


def test_genuinely_degenerate_states_are_reported():
    """Same decay *and* same frequency is a bank of states doing one state's
    job, and that must be caught — otherwise the check above is just silence."""
    model = td.S4D(32, 1, LAT, portable=True, d_state=16)
    with torch.no_grad():
        for mixer in model.nd.mixers:
            mixer.kernel.A_imag.zero_()
    assert any(n["level"] == "warn" for n in _health(model))


def test_dead_units_are_counted_against_the_tensor_s_own_scale():
    """An absolute threshold means nothing across initialisations, so a dead
    unit is one whose peak weight is negligible relative to its own tensor."""
    model = td.Transformer(32, 1, LAT)
    with torch.no_grad():
        model.nd.mixers[0].proj.weight[:4] = 0.0
    notes = _health(model)
    assert any("4 of 32 output units are dead" in n["text"] for n in notes)


def test_the_frequency_parameter_is_not_mistaken_for_a_decay():
    """`A_imag` sets how fast a state oscillates, not how fast it fades.
    Classifying it as decay made the health check read a frequency as a
    retention and call a healthy S4D instantly-forgetting."""
    payload = td.viz.weights(td.S4D(16, 1, LAT, portable=True, d_state=8))
    assert _by_name(payload, "kernel.A_imag")["role"] == "ssm_freq"
    assert _by_name(payload, "kernel.log_A_real")["role"] == "ssm_decay"


def test_every_tensor_carries_a_histogram():
    payload = td.viz.weights(td.LSTM(16, 1, LAT))
    for tensor in payload["layers"][0]["tensors"]:
        hist = tensor["histogram"]
        assert sum(hist["bins"]) == tensor["stats"]["n"]
        assert hist["lo"] <= hist["hi"]
