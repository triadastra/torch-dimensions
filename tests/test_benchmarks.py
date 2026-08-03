"""The comparison instrument, calibrated.

`benchmarks/compare.py` is what will produce the CUDA-vs-MPS verdict, and
`pretrain.py` is what produces the artifacts it reads. Both were untested,
which is a bad property for an instrument: a wrong answer from a measuring
tool looks exactly like a right one, and the CUDA run is a thing we get to do
once before believing its output.

The load-bearing assumption is the first test below. Everything the comparison
claims rests on two machines starting from bit-identical weights on
bit-identical data — if that quietly stopped being true, every number in the
comparison would be measuring initialisation drift and still look plausible.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str, path: Path):
    """Import a script that is not part of the installed package."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pretrain = _load("_td_pretrain", ROOT / "benchmarks" / "pretrain.py")
compare = _load("_td_compare", ROOT / "benchmarks" / "compare.py")


# --- the assumption the whole comparison rests on ----------------------------


def test_initialisation_is_identical_across_calls():
    """Models are built on CPU under a fixed seed and only then moved, so two
    machines start from the same weights *bitwise*. This is the property that
    makes a cross-device difference mean 'arithmetic' rather than 'different
    starting point', and nothing else in the suite checks it."""
    lat = pretrain.sparse_2d()
    built = []
    for _ in range(2):
        torch.manual_seed(pretrain.SEED)
        built.append(pretrain.ZOO["lstm_2d_sparse"]["build"](lat))
    a, b = (dict(m.named_parameters()) for m in built)
    assert a.keys() == b.keys()
    for key in a:
        assert torch.equal(a[key], b[key]), f"{key} differs between two seeded builds"


def test_the_data_stream_is_identical_across_calls():
    """Same for the batches: drawn on CPU from a seeded generator, so both
    machines see the same numbers in the same order."""

    def draw_all():
        gen = torch.Generator().manual_seed(pretrain.SEED + 1)
        return [torch.randn(4, 6, 6, 8, 1, generator=gen) for _ in range(3)]

    for first, second in zip(draw_all(), draw_all(), strict=True):
        assert torch.equal(first, second)


def test_the_lattice_is_the_same_every_time():
    """The sparse mask is generated, not stored — if it drifted, the two runs
    would be training on different problems."""
    a, b = pretrain.sparse_2d(), pretrain.sparse_2d()
    assert torch.equal(a.valid, b.valid)
    assert a.shape == b.shape and a.names == b.names


@pytest.mark.parametrize("name", list(pretrain.ZOO))
def test_every_model_in_the_zoo_builds_and_runs(name):
    """A broken entry currently surfaces only part-way through a benchmark run
    — after minutes of training other models."""
    cfg = pretrain.ZOO[name]
    lat = cfg["lat"]()
    torch.manual_seed(pretrain.SEED)
    model = cfg["build"](lat)
    x = torch.randn(1, 2, *lat.shape, 1)
    with torch.no_grad():
        out = model(x)
    assert out.shape[:-1] == x.shape[:-1]
    assert torch.isfinite(out).all()


# --- the comparison itself ---------------------------------------------------


def _run_dir(tmp_path: Path, name: str, losses: list[float], weights: dict) -> Path:
    """A minimal `pretrain.py`-shaped output directory."""
    root = tmp_path / name
    (root / "toy").mkdir(parents=True)
    torch.save({"model": weights, "head": {}}, root / "toy" / "weights.pt")
    record = {
        "name": "toy",
        "n_params": sum(v.numel() for v in weights.values()),
        "losses": losses,
        "loss_first": losses[0],
        "loss_final": losses[-1],
        "steps_per_second": 10.0,
    }
    (root / "toy" / "metrics.json").write_text(json.dumps(record))
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "device": "cpu",
                "device_name": name,
                "torch": torch.__version__,
                "torch_dimensions": "test",
                "seed": 1,
                "steps": len(losses),
                "batch": 1,
                "models": [record],
            }
        )
    )
    return root


def test_two_identical_runs_report_no_divergence(tmp_path):
    w = {"a": torch.ones(4, 4)}
    left = _run_dir(tmp_path, "left", [1.0, 0.5, 0.25], w)
    right = _run_dir(tmp_path, "right", [1.0, 0.5, 0.25], dict(w))

    a, b = compare.load(left), compare.load(right)
    assert compare.first_divergence(a["models"][0]["losses"], b["models"][0]["losses"]) is None
    rel, worst = compare.weight_delta(
        compare.weights_of(left, "toy"), compare.weights_of(right, "toy")
    )
    assert rel == 0.0 and worst == 0.0


def test_divergence_is_found_at_the_step_it_happens():
    """Off by one here would misattribute *when* two machines parted company,
    which is the number the comparison leans on most."""
    base = [1.0, 0.5, 0.25, 0.125]
    moved = [1.0, 0.5, 0.2500004, 0.125]  # 1.6e-6 relative, above the 1e-6 tolerance
    assert compare.first_divergence(base, moved) == 2
    assert compare.first_divergence(base, list(base)) is None
    # A difference below tolerance is not a divergence.
    assert compare.first_divergence(base, [1.0, 0.5, 0.25000001, 0.125]) is None


def test_weight_delta_is_the_arithmetic_it_claims():
    """Relative Frobenius distance and the worst single element, checked
    against numbers computed by hand rather than against itself."""
    a = {"model": {"w": torch.tensor([[3.0, 0.0], [0.0, 4.0]])}}  # ‖w‖ = 5
    b = {"model": {"w": torch.tensor([[3.0, 0.0], [0.0, 1.0]])}}  # differs by 3 at one entry
    rel, worst = compare.weight_delta(a, b)
    assert worst == pytest.approx(3.0)
    assert rel == pytest.approx(3.0 / 5.0)


def test_a_model_missing_from_one_run_is_reported_not_crashed(tmp_path, capsys):
    """One side failing to train a model must not take the whole comparison
    down — the other fifteen rows are still the point."""
    left = _run_dir(tmp_path, "left", [1.0, 0.5], {"a": torch.ones(2, 2)})
    right = _run_dir(tmp_path, "right", [1.0, 0.5], {"a": torch.ones(2, 2)})
    manifest = json.loads((right / "manifest.json").read_text())
    manifest["models"].append({"name": "only_here", "error": "boom"})
    (right / "manifest.json").write_text(json.dumps(manifest))

    argv = sys.argv
    sys.argv = ["compare.py", str(left), str(right)]
    try:
        assert compare.main() == 0
    finally:
        sys.argv = argv
    assert "toy" in capsys.readouterr().out


def test_the_cuda_harness_imports_and_skips_without_a_device():
    """`scripts/cuda_check.py` runs its checks at import. Without CUDA every
    CUDA-only check must skip rather than fail, or the report from a machine
    that *has* one cannot be trusted either."""
    if torch.cuda.is_available():
        pytest.skip("this asserts the no-CUDA behaviour")
    module = _load("_td_cuda_check", ROOT / "scripts" / "cuda_check.py")
    statuses = {name: status for name, status, _ in module.RESULTS}
    assert statuses, "the harness recorded nothing at all"
    assert "fail" not in statuses.values(), f"failed without a device: {statuses}"
    assert any(s == "skip" for s in statuses.values())
