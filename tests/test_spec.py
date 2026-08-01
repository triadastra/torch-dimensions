"""Acceptance for td.spec — the architecture document the viewer reads.

See VIEWER.md. The spec is a contract with a separate codebase, so the tests
care about exact keys and lossless encoding, not just plausibility.
"""

import json

import pytest
import torch

import torch_dimensions as td
from torch_dimensions.spec import SPEC_VERSION, _rle


def rle_decode(runs, total):
    """Independent decoder, so the encoder is checked against something that
    does not share its code."""
    out, value = [], False
    for count in runs:
        out.extend([value] * count)
        value = not value
    assert len(out) == total
    return torch.tensor(out, dtype=torch.bool)


def sparse_lattice():
    valid = torch.tensor([[True, True], [True, False], [False, True]])
    return td.Lattice(shape=(3, 2), names=("state", "sku"), valid=valid, time=True)


# -- envelope ----------------------------------------------------------------


def test_spec_is_json_serializable():
    model = td.LSTM(4, 3, td.Lattice(shape=(2, 3), time=True))
    text = json.dumps(td.spec(model))
    assert json.loads(text)["format"] == "torch-dimensions/architecture"


def test_spec_carries_a_version():
    model = td.LSTM(4, 2)
    assert td.spec(model)["version"] == SPEC_VERSION


def test_spec_refuses_a_module_that_does_not_describe_itself():
    with pytest.raises(TypeError, match="does not describe itself"):
        td.spec(torch.nn.Linear(2, 2))


# -- lattice -----------------------------------------------------------------


def test_lattice_axes_mark_time_as_dynamic():
    spec = td.spec(td.LSTM(4, 3, sparse_lattice()))["lattice"]
    assert spec["names"] == ["time", "state", "sku"]
    assert spec["axes"][0] == {"name": "time", "size": None, "dynamic": True}
    assert spec["axes"][1] == {"name": "state", "size": 3, "dynamic": False}
    assert spec["rank"] == 2 and spec["n_axes"] == 3


def test_presence_encoding_is_lossless():
    lat = sparse_lattice()
    cells = td.spec(td.LSTM(4, 3, lat))["lattice"]["cells"]
    assert cells == {
        "total": 6,
        "present": 4,
        "dense": False,
        "present_rle": cells["present_rle"],
    }
    decoded = rle_decode(cells["present_rle"], 6)
    assert torch.equal(decoded, lat.valid.reshape(-1))


def test_a_dense_lattice_encodes_as_one_run():
    lat = td.Lattice(shape=(2, 3), time=True)
    cells = td.spec(td.LSTM(4, 3, lat))["lattice"]["cells"]
    assert cells["dense"] is True
    assert cells["present_rle"] == [0, 6]


@pytest.mark.parametrize(
    "flags",
    [
        [True] * 5,
        [False] * 4 + [True],
        [True, False, True, False, True],
        [False, False, True, True, False],
    ],
)
def test_rle_round_trips_for_arbitrary_masks(flags):
    t = torch.tensor(flags, dtype=torch.bool)
    assert torch.equal(rle_decode(_rle(t), len(flags)), t)


# -- layers ------------------------------------------------------------------


def test_layers_name_the_axis_each_one_sweeps():
    lat = td.Lattice(shape=(2, 3), names=("h", "w"), time=True)
    layers = td.spec(td.LSTM(4, 4, lat))["layers"]
    assert [layer["axis"] for layer in layers] == ["time", "h", "w", "time"]
    assert all(layer["mixer"] == "LSTMMixer" for layer in layers)
    assert all(layer["n_params"] > 0 for layer in layers)


def test_layers_record_direction():
    lat = td.Lattice(shape=(2, 3), names=("h", "w"))
    model = td.LSTM(4, 4, lat, bidirectional=True)
    assert [layer["reverse"] for layer in td.spec(model)["layers"]] == [
        False,
        False,
        True,
        True,
    ]


def test_layer_count_matches_the_model():
    assert td.spec(td.LSTM(4, 7, td.Lattice(shape=(2, 3))))["model"]["n_layers"] == 7


# -- the thing the viewer exists to show -------------------------------------


def test_directions_report_axes_pinned_to_one_way():
    """An axis swept only forward is invisible in code and obvious in a
    picture. The spec makes it data so the picture can show it."""
    lat = td.Lattice(shape=(2, 3), names=("h", "w"), time=True)
    model = td.LSTM(4, 6, lat, bidirectional=("h", "w"))
    directions = td.spec(model)["sweeps"]["directions"]
    assert directions["time"] == "forward"
    assert directions["h"] == "both" and directions["w"] == "both"


def test_directions_distinguish_backward_only():
    lat = td.Lattice(shape=(2, 3), names=("h", "w"))
    plan = td.ScanPlan.from_list([("h", True), ("w", False)])
    directions = td.spec(td.LSTM(4, 2, lat, plan=plan))["sweeps"]["directions"]
    assert directions["h"] == "backward" and directions["w"] == "forward"


def test_unswept_axes_are_listed():
    lat = td.Lattice(shape=(2, 3), names=("h", "w"))
    with pytest.warns(UserWarning):
        model = td.LSTM(4, 1, lat, plan=td.ScanPlan.from_list(["h"]))
    assert td.spec(model)["sweeps"]["unswept_axes"] == ["w"]


def test_nothing_unswept_when_the_plan_covers_the_lattice():
    lat = td.Lattice(shape=(2, 3), names=("h", "w"))
    assert td.spec(td.LSTM(4, 2, lat))["sweeps"]["unswept_axes"] == []


# -- io ----------------------------------------------------------------------


def test_io_shapes_are_symbolic_in_batch_and_time():
    lat = td.Lattice(shape=(3, 2), time=True)
    io = td.spec(td.LSTM(8, 3, lat, d_input=2))["io"]
    assert io["input"] == ["B", "T", 3, 2, 2]
    assert io["output"] == ["B", "T", 3, 2, 8]
    assert io["cells_per_step"] == 6


def test_io_omits_time_when_the_lattice_has_none():
    lat = td.Lattice(shape=(3, 2))
    assert td.spec(td.LSTM(8, 2, lat))["io"]["input"] == ["B", 3, 2, 8]


def test_d_input_defaults_to_d_model_without_a_projection():
    model = td.LSTM(8, 2, td.Lattice(shape=(3, 2)))
    assert td.spec(model)["model"]["d_input"] == 8


def test_the_one_dimensional_model_still_describes_itself():
    io = td.spec(td.LSTM(4, 2))["io"]
    assert io["input"] == ["B", "T", 4] and io["cells_per_step"] == 1


def test_reported_parameter_count_matches_the_module():
    model = td.LSTM(8, 3, td.Lattice(shape=(2, 3)), d_input=2)
    assert td.spec(model)["model"]["n_params"] == sum(p.numel() for p in model.parameters())


def test_gru_is_described_too():
    spec = td.spec(td.GRU(4, 2, td.Lattice(shape=(2, 3))))
    assert spec["model"]["kind"] == "GRU"
    assert spec["layers"][0]["mixer"] == "GRUMixer"


# -- the spec must describe the family that actually runs ---------------------


def test_the_kernel_family_spec_does_not_claim_spatial_sweeps():
    """A kernel-family layer contracts every spatial axis at once and sweeps
    only time. The spec used to describe it with the scan family's schema, so
    a 3-layer CaFA model claimed to sweep time, then h, then w — three sweeps
    that never happen — and the viewer drew them (DEBUG.md #26).
    """
    lat = td.Lattice(shape=(4, 5), names=("h", "w"), time=True)
    s = td.spec(td.LSTM(8, 3, lat, method=td.cafa))

    assert s["nd_method"]["family"] == "kernel"
    assert [layer["kind"] for layer in s["layers"]] == ["kernel"] * 3
    assert {layer["axis"] for layer in s["layers"]} == {"time"}
    assert all(layer["contracted"] == ["h", "w"] for layer in s["layers"])
    assert s["sweeps"]["contracted_axes"] == ["h", "w"]
    # A contraction has no direction; only the swept axis gets one.
    assert s["sweeps"]["directions"] == {"time": "forward"}
    # Contracted axes are mixed, so they are not "unswept" in the sense the
    # viewer warns about.
    assert s["sweeps"]["unswept_axes"] == []


def test_a_kernel_only_block_reports_no_swept_axis_at_all():
    lat = td.Lattice(shape=(4, 5), names=("h", "w"))
    block = td.AxialKernel(
        mixer=None, plan=td.ScanPlan.cyclic(lat.axis_names, 2), lattice=lat, d_model=8
    )

    class Wrapper(torch.nn.Module):
        lattice = lat

        def __init__(self):
            super().__init__()
            self.nd = block

        def to_spec(self):
            from torch_dimensions.spec import scan_model_spec

            return scan_model_spec(self)

    s = td.spec(Wrapper())
    assert [layer["axis"] for layer in s["layers"]] == [None, None]
    assert [layer["mixer"] for layer in s["layers"]] == [None, None]
    assert s["sweeps"]["directions"] == {}
    assert s["sweeps"]["contracted_axes"] == ["h", "w"]


def test_the_scan_family_spec_is_unchanged_in_shape():
    lat = td.Lattice(shape=(4, 5), names=("h", "w"), time=True)
    s = td.spec(td.LSTM(8, 3, lat))
    assert s["nd_method"]["family"] == "scan"
    assert [layer["axis"] for layer in s["layers"]] == ["time", "h", "w"]
    assert [layer["kind"] for layer in s["layers"]] == ["scan"] * 3
    assert s["sweeps"]["contracted_axes"] == []
