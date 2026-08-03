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


def test_the_server_offers_weights_for_a_model():
    pytest.importorskip("torch_dimensions.viz")
    if not td.viz.bundle_exists():
        pytest.skip("viewer bundle not built")
    import urllib.request

    server = td.viz.serve(td.LSTM(16, 2, LAT), port=0)
    try:
        url = f"http://127.0.0.1:{server.server_port}/weights.json"
        payload = json.loads(urllib.request.urlopen(url).read())
        assert payload["format"] == "torch-dimensions/weights"
        assert payload["layers"]
    finally:
        server.shutdown()


def test_the_server_says_so_when_there_are_no_weights_to_serve():
    """Opened on a spec dict there are no parameters, and 404 is the honest
    answer — the viewer then tells the user to open a model instead."""
    if not td.viz.bundle_exists():
        pytest.skip("viewer bundle not built")
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
