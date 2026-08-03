"""The published benchmark checkpoints load, and load into the right model.

`MPS bench/` and `CUDA bench/` are shipped in the repository and uploaded to the
Hugging Face repo, so they are part of the published surface: someone will
`torch.load` one and expect it to go into the model the manifest names. That
makes them a compatibility contract, and an untested contract is a promise.

The failure this guards against is quiet. Rename a submodule, change a mixer's
internal layout, add a buffer — the library's own tests all still pass, and the
only thing that breaks is every checkpoint published before the change. Loading
with `strict=True` here turns that into a failing test on the commit that
causes it rather than a bug report from someone who downloaded the weights.

Skipped rather than failed when the directories are absent, since a source
checkout without the artifacts is a legitimate way to work on the library.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))

spec = importlib.util.spec_from_file_location("_td_zoo", ROOT / "benchmarks" / "pretrain.py")
pretrain = importlib.util.module_from_spec(spec)
sys.modules["_td_zoo"] = pretrain
spec.loader.exec_module(pretrain)

RUNS = [d for d in ("MPS bench", "CUDA bench") if (ROOT / d / "manifest.json").exists()]


def entries(run: str):
    manifest = json.loads((ROOT / run / "manifest.json").read_text())
    return [e for e in manifest["models"] if "error" not in e]


CASES = [(run, e["name"]) for run in RUNS for e in entries(run)]


@pytest.mark.skipif(not RUNS, reason="no benchmark artifacts in this checkout")
@pytest.mark.parametrize("run,name", CASES, ids=[f"{r.split()[0]}-{n}" for r, n in CASES])
def test_the_checkpoint_loads_into_the_model_it_names(run, name):
    """`strict=True` on purpose: a checkpoint that only loads with
    `strict=False` has already silently stopped matching the architecture, and
    the missing tensors are whatever the model then leaves at its init."""
    cfg = pretrain.ZOO[name]
    lat = cfg["lat"]()
    blob = torch.load(ROOT / run / name / "weights.pt", map_location="cpu", weights_only=True)

    torch.manual_seed(0)
    model = cfg["build"](lat)
    head = nn.Linear(model.config["d_model"], 1)
    model.load_state_dict(blob["model"])
    head.load_state_dict(blob["head"])

    # The manifest counts *parameters*; a state dict also carries buffers, and
    # S4's DPLR kernel has four elements of them. Comparing the two totals
    # directly reports a mismatch that is only bookkeeping.
    recorded = next(e for e in entries(run) if e["name"] == name)["n_params"]
    assert sum(p.numel() for p in model.parameters()) == recorded


@pytest.mark.skipif(not RUNS, reason="no benchmark artifacts in this checkout")
@pytest.mark.parametrize("run,name", CASES, ids=[f"{r.split()[0]}-{n}" for r, n in CASES])
def test_the_loaded_checkpoint_runs(run, name):
    """Loading is not running: a state dict can be accepted and still leave a
    model that produces NaN on the machine doing the loading, which is the
    whole point of publishing weights trained on other hardware."""
    cfg = pretrain.ZOO[name]
    lat = cfg["lat"]()
    blob = torch.load(ROOT / run / name / "weights.pt", map_location="cpu", weights_only=True)

    torch.manual_seed(0)
    model = cfg["build"](lat)
    head = nn.Linear(model.config["d_model"], 1)
    model.load_state_dict(blob["model"])
    head.load_state_dict(blob["head"])
    model, head = model.eval(), head.eval()

    x = torch.randn(2, 3, *lat.shape, 1)
    with torch.no_grad():
        y = head(model(x))
    assert y.shape == (2, 3, *lat.shape, 1)
    assert torch.isfinite(y).all()


@pytest.mark.skipif(len(RUNS) < 2, reason="needs both runs to compare")
def test_the_two_runs_cover_the_same_models_and_say_what_they_ran_at():
    """A comparison of two runs is only meaningful if they contain the same
    models, and only interpretable if each records the precision it used —
    CUDA's cuDNN TF32 default is on, and a run that does not say so cannot be
    compared with one that had it off."""
    left, right = ({e["name"] for e in entries(r)} for r in RUNS)
    assert left == right, f"the runs differ: {sorted(left ^ right)}"

    for run in RUNS:
        manifest = json.loads((ROOT / run / "manifest.json").read_text())
        assert "precision" in manifest, f"{run} does not record what precision it ran at"
        assert "weights_from" in entries(run)[0], (
            f"{run} does not record where its weights came from"
        )
