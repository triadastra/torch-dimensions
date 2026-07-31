"""Phase 8 acceptance: registry, config, and save/load. See PLAN.md.

The two properties the plan calls non-negotiable are tested hardest: a
checkpoint refuses an incompatible format version rather than loading wrong
silently, and the lattice's validity mask travels with the weights.
"""

import json

import pytest
import torch

import torch_dimensions as td
from torch_dimensions.config import MODELS

MINIMAL = {
    "lstm": {},
    "gru": {},
    "s4": {},
    "s4d": {},
    "mamba": {},
    "s4nd": {"dim": 2, "shape": [3, 4]},
    "s4dnd": {"dim": 2, "shape": [3, 4]},
    "mamband": {"dim": 2, "shape": [3, 4]},
}


def _sparse_cfg():
    valid = [[True, True, True, False], [True, False, True, True], [True, True, False, True]]
    return {
        "kind": "lstm",
        "d_model": 8,
        "n_layers": 3,
        "lattice": {"shape": [3, 4], "names": ["h", "w"], "time": True, "valid": valid},
    }


# -- build ---------------------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(MINIMAL))
def test_every_registered_kind_builds_from_a_dict(kind):
    model = td.build({"kind": kind, "d_model": 8, "n_layers": 2, **MINIMAL[kind]})
    x = torch.randn(2, 5, 3, 4, 8) if kind.endswith("nd") else torch.randn(2, 5, 8)
    assert model(x).shape == x.shape


def test_build_accepts_lattice_plan_and_method_as_plain_data():
    cfg = {
        "kind": "lstm",
        "d_model": 8,
        "n_layers": 4,
        "lattice": {"shape": [3, 4], "names": ["h", "w"], "time": True},
        "plan": {"type": "cyclic", "bidirectional": ["h", "w"]},
        "nd_method": "axial_scan",
    }
    model = td.build(cfg)
    seen = {}
    for s in model.plan:
        seen.setdefault(s.axis, set()).add(s.reverse)
    assert seen[0] == {False}  # time stays causal
    assert model(torch.randn(1, 2, 3, 4, 8)).shape == (1, 2, 3, 4, 8)


def test_build_from_yaml(tmp_path):
    yaml = pytest.importorskip("yaml")
    cfg = {"kind": "s4d", "d_model": 8, "n_layers": 2, "d_state": 8}
    p = tmp_path / "model.yaml"
    p.write_text(yaml.safe_dump(cfg))
    model = td.build(p)
    assert model(torch.randn(2, 5, 8)).shape == (2, 5, 8)


def test_unknown_kind_lists_the_registered_ones():
    with pytest.raises(ValueError, match="mamband.*s4nd"):
        td.build({"kind": "transformer9000", "d_model": 8})


def test_unknown_keys_are_a_hard_error_naming_them():
    """A silently ignored typo is a silently different model."""
    with pytest.raises(ValueError, match=r"\['d_modle'\].*accepted"):
        td.build({"kind": "lstm", "d_model": 8, "d_modle": 16})


def test_config_is_a_fixed_point_of_build():
    """config -> model -> config must not drift."""
    model = td.build(_sparse_cfg())
    again = td.build({"kind": "lstm", **model.config})
    assert again.config == model.config


# -- save / load ---------------------------------------------------------------


@pytest.mark.parametrize(
    "cfg",
    [
        _sparse_cfg(),
        {
            "kind": "s4nd",
            "d_model": 8,
            "n_layers": 2,
            "dim": 2,
            "shape": [3, 4],
            "method": "cafa",
            "gate": "leaky_relu",
        },
        {
            "kind": "mamba",
            "d_model": 8,
            "n_layers": 2,
            "d_state": 4,
            "lattice": {"shape": [3], "names": ["w"], "time": True},
        },
    ],
)
def test_save_load_round_trips_bitwise(cfg, tmp_path):
    torch.manual_seed(0)
    model = td.build(dict(cfg))
    p = tmp_path / "ckpt.td"
    model.save(p)
    twin = td.load(p)
    model.eval(), twin.eval()
    lat = model.lattice
    x = torch.randn(2, 5, *lat.shape, 8) if lat.time else torch.randn(2, *lat.shape, 8)
    assert torch.equal(model(x), twin(x))


def test_training_then_saving_preserves_the_trained_weights(tmp_path):
    torch.manual_seed(0)
    model = td.build(_sparse_cfg())
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = torch.randn(2, 5, 3, 4, 8)
    for _ in range(3):
        loss = model(x).pow(2).mean()
        opt.zero_grad(), loss.backward(), opt.step()
    p = tmp_path / "trained.td"
    model.save(p)
    twin = td.load(p)
    model.eval(), twin.eval()
    assert torch.equal(model(x), twin(x))


def test_the_validity_mask_travels_with_the_checkpoint(tmp_path):
    """A model restored against a different sparsity pattern is a wrong
    model; the mask is architecture, so it lives in the checkpoint."""
    model = td.build(_sparse_cfg())
    p = tmp_path / "sparse.td"
    model.save(p)
    twin = td.load(p)
    assert twin.lattice.valid is not None
    assert torch.equal(twin.lattice.valid, model.lattice.valid)


def test_an_incompatible_checkpoint_version_is_refused(tmp_path):
    model = td.build({"kind": "lstm", "d_model": 4})
    p = tmp_path / "old.td"
    model.save(p)
    ckpt = torch.load(p)
    ckpt["version"] = 999
    torch.save(ckpt, p)
    with pytest.raises(ValueError, match="v999.*refusing"):
        td.load(p)


def test_a_foreign_file_is_refused(tmp_path):
    p = tmp_path / "notours.pt"
    torch.save({"weights": torch.zeros(3)}, p)
    with pytest.raises(ValueError, match="not a torch-dimensions checkpoint"):
        td.load(p)


def test_an_unregistered_nd_method_refuses_to_save(tmp_path):
    def my_method(mixer, plan, lattice, d_model, **kw):
        return td.AxialScan(mixer=mixer, plan=plan, lattice=lattice, d_model=d_model, **kw)

    model = td.LSTM(4, 2, td.Lattice(shape=(2, 3)), nd_method=my_method)
    with pytest.raises(ValueError, match="unregistered nd_method"):
        model.save(tmp_path / "x.td")


def test_an_unregistered_model_class_refuses_to_save(tmp_path):
    class Custom(td.LSTM):
        pass

    model = Custom(4, 2)
    with pytest.raises(ValueError, match="not a registered model kind"):
        td.save(model, tmp_path / "x.td")
    td.register_model("custom_lstm", Custom)
    try:
        td.save(model, tmp_path / "x.td")
        twin = td.load(tmp_path / "x.td")
        assert type(twin) is Custom
    finally:
        del MODELS["custom_lstm"]


def test_registering_a_duplicate_kind_is_refused():
    with pytest.raises(ValueError, match="already registered"):
        td.register_model("lstm", td.LSTM)


def test_configs_are_json_and_registry_is_listable():
    model = td.build(_sparse_cfg())
    assert json.loads(json.dumps(model.config)) == model.config
    assert "s4nd" in td.list_models() and "lstm" in td.list_models()
