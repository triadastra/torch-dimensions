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


# -- safetensors container -----------------------------------------------------

safetensors = pytest.importorskip("safetensors")
# safetensors' torch bindings import numpy internally, so skipping on
# safetensors alone is not enough — that is exactly how these failed in CI.
pytest.importorskip("numpy")


@pytest.mark.parametrize("suffix", [".td", ".safetensors"])
def test_a_checkpoint_rebuilds_the_same_model_in_either_container(tmp_path, suffix):
    lat = td.Lattice(
        shape=(2, 3),
        names=("a", "b"),
        valid=torch.tensor([[True, False, True], [True, True, False]]),
        time=True,
    )
    model = td.S4D(16, 3, lat, d_input=2).eval()
    path = tmp_path / f"model{suffix}"
    td.save(model, path)
    same = td.load(path).eval()

    x = torch.randn(2, 4, 2, 3, 2)
    assert torch.equal(model(x), same(x)), "restored model is not the same model"
    assert same.lattice.valid.tolist() == lat.valid.tolist(), "validity mask did not travel"


def test_the_safetensors_file_holds_no_pickle(tmp_path):
    """The reason to offer this container at all: opening it cannot run code.
    A torch pickle starts with the zip magic and contains `data.pkl`."""
    model = td.LSTM(8, 2, td.Lattice(shape=(2, 2), time=True))
    path = tmp_path / "model.safetensors"
    td.save(model, path)
    raw = path.read_bytes()
    assert b"data.pkl" not in raw and not raw.startswith(b"PK")


def test_safetensors_metadata_carries_the_recipe(tmp_path):
    from safetensors import safe_open

    model = td.MambaND(16, 2, dim=2, shape=(2, 2), time=True)
    path = tmp_path / "m.safetensors"
    td.save(model, path)
    with safe_open(str(path), framework="pt") as fh:
        meta = fh.metadata()
    assert meta["kind"] == "mamband"
    assert json.loads(meta["config"])["d_model"] == 16


def test_a_foreign_safetensors_file_is_refused(tmp_path):
    from safetensors.torch import save_file

    path = tmp_path / "someone_elses.safetensors"
    save_file({"w": torch.zeros(2, 2)}, str(path), metadata={"format": "not-ours"})
    with pytest.raises(ValueError, match="not a torch-dimensions checkpoint"):
        td.load(path)


def test_a_model_with_a_substituted_mixer_refuses_to_save(tmp_path):
    """`mixer=` swaps the 1-D operator for debugging, and a class cannot go
    into a JSON recipe — so a checkpoint would rebuild with the stock mixer and
    return a different model that loads perfectly."""
    model = td.LSTM(8, 2, td.Lattice(shape=(2, 2), time=True), mixer=td.testing.Recorder)
    with pytest.raises(ValueError, match="Recorder"):
        td.save(model, tmp_path / "wrong.td")
    # the stock model still saves
    td.save(td.LSTM(8, 2, td.Lattice(shape=(2, 2), time=True)), tmp_path / "fine.td")


def test_build_from_a_checkpoint_gives_the_architecture_without_the_weights(tmp_path):
    """ "The same model, freshly initialized" is wanted often enough — a second
    seed, a fine-tune baseline — that it should not mean unpacking the file by
    hand. `weights=True` is exactly `load`."""
    lat = td.Lattice(shape=(2, 3), names=("a", "b"), time=True)
    trained = td.S4D(16, 3, lat, d_input=1)
    with torch.no_grad():
        for p in trained.parameters():
            p.add_(1.0)  # make the weights unmistakably not fresh
    path = tmp_path / "run.td"
    td.save(trained, path)

    fresh = td.build(path)
    assert type(fresh) is td.S4D and len(fresh.nd.plan) == 3
    same = [
        torch.equal(a, b) for a, b in zip(fresh.parameters(), trained.parameters(), strict=True)
    ]
    assert not all(same), "build() restored the weights; that is load()'s job"

    restored = td.build(path, weights=True)
    assert all(
        torch.equal(a, b) for a, b in zip(restored.parameters(), trained.parameters(), strict=True)
    )


def test_read_config_names_what_the_checkpoint_claims_to_be(tmp_path):
    lat = td.Lattice(shape=(2, 2), names=("a", "b"), time=True)
    path = tmp_path / "m.safetensors"
    td.save(td.LSTM(8, 2, lat), path)
    cfg = td.read_config(path)
    assert cfg["kind"] == "lstm" and cfg["d_model"] == 8
    assert json.loads(json.dumps(cfg)) == cfg, "a recipe must be plain data"


def test_a_third_party_kind_can_register_itself_by_entry_point(monkeypatch):
    """Plugins register without being imported eagerly — an eager import of
    every installed plugin is how an optional dependency becomes mandatory."""
    from importlib.metadata import EntryPoint

    from torch_dimensions import config as cfgmod

    calls = []

    class Fake(EntryPoint):
        def load(self):
            calls.append(self.name)
            return td.LSTM

    fake = Fake("plugin_lstm", "does.not.exist:Thing", "torch_dimensions.models")
    monkeypatch.setattr(cfgmod, "entry_points", lambda group: [fake])
    monkeypatch.setitem(cfgmod.MODELS, "lstm", td.LSTM)  # registry already populated
    assert calls == [], "the plugin was imported before anyone asked"
    cfgmod._load_entry_points()
    assert cfgmod.MODELS["plugin_lstm"] is td.LSTM and calls == ["plugin_lstm"]
    cfgmod.MODELS.pop("plugin_lstm")


def test_a_broken_plugin_warns_instead_of_breaking_the_import(monkeypatch):
    from importlib.metadata import EntryPoint

    from torch_dimensions import config as cfgmod

    class Broken(EntryPoint):
        def load(self):
            raise ImportError("no such module")

    broken = Broken("broken", "nope:Nope", "torch_dimensions.models")
    monkeypatch.setattr(cfgmod, "entry_points", lambda group: [broken])
    with pytest.warns(UserWarning, match="failed to load"):
        cfgmod._load_entry_points()
    assert "broken" not in cfgmod.MODELS


def test_the_package_reports_the_version_it_was_built_as():
    """`__version__` used to be a literal in `__init__.py`, and it said 0.1.0
    through the 0.2.0 and 0.3.1 releases: a published wheel misreporting its
    own version, which is precisely the string a bug report quotes. It comes
    from the installed metadata now, so there is one source of truth."""
    from importlib.metadata import version

    import torch_dimensions as td

    assert td.__version__ == version("torch-dimensions")
    assert td.__version__ != "0.0.0+source", "the package under test is not installed"
