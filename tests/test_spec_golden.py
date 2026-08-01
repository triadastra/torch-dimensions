"""Stored specs, diffed on change: the viewer contract as a fixture.

The spec is a published document — the viewer parses it, `td.viz.show` serves
it, and a downstream tool may read it. Its shape can therefore break without
any test failing: every existing spec test asserts particular keys, so a
*silently added, removed, or renamed* field passes them all and reaches the
viewer as a blank panel.

These files are the answer. One stored spec per family, compared verbatim. A
deliberate change is a one-line regeneration and a visible diff in review; an
accidental one is a failing test.

    TD_UPDATE_GOLDEN=1 pytest tests/test_spec_golden.py    # after an intended change
"""

import json
import os
from pathlib import Path

import pytest
import torch

import torch_dimensions as td

GOLDEN = Path(__file__).parent / "golden"


def sparse(shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    valid = torch.rand(shape, generator=g) > 0.35
    valid.reshape(-1)[0] = True
    return valid


MODELS = {
    "lstm_2d_sparse": lambda: td.LSTM(
        16,
        4,
        td.Lattice(shape=(3, 4), names=("h", "w"), valid=sparse((3, 4)), time=True),
        bidirectional=("h", "w"),
    ),
    "s4d_nd_3d": lambda: td.S4DND(24, 6, dim=3, shape=(2, 3, 4), time=True),
    "mamba_nd_paired": lambda: td.Mamba(
        32,
        lattice=td.Lattice(shape=(2, 3), names=("row", "col"), time=True),
        plan=td.ScanPlan.paired(("time", "row", "col"), n_layers=6, bidirectional=("row", "col")),
    ),
    "cafa_hybrid": lambda: td.GRU(
        16, 3, td.Lattice(shape=(3, 3), names=("y", "x"), time=True), method=td.cafa
    ),
}


@pytest.mark.parametrize("name", sorted(MODELS))
def test_spec_matches_its_stored_golden(name):
    torch.manual_seed(0)
    got = td.spec(MODELS[name]())
    path = GOLDEN / f"{name}.json"

    if os.environ.get("TD_UPDATE_GOLDEN"):
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(got, indent=2, sort_keys=True) + "\n")
        pytest.skip(f"regenerated {path.name} — review the diff before committing")

    assert path.exists(), f"no golden file for {name}; run with TD_UPDATE_GOLDEN=1"
    want = json.loads(path.read_text())
    if got != want:
        added = sorted(set(_paths(got)) - set(_paths(want)))
        removed = sorted(set(_paths(want)) - set(_paths(got)))
        raise AssertionError(
            f"{name}: spec differs from the stored contract.\n"
            f"  fields added:   {added or '—'}\n"
            f"  fields removed: {removed or '—'}\n"
            "If this change is intended, regenerate with TD_UPDATE_GOLDEN=1 and bump "
            "SPEC_VERSION when the change is not backwards compatible."
        )


def _paths(node, prefix=""):
    """Every dotted key path in a nested document, for a readable diff."""
    out = []
    if isinstance(node, dict):
        for k, v in node.items():
            out.append(f"{prefix}{k}")
            out.extend(_paths(v, f"{prefix}{k}."))
    elif isinstance(node, list) and node:
        out.extend(_paths(node[0], f"{prefix}[]."))
    return out


def test_the_stored_specs_are_all_at_the_current_version():
    """A golden file left at an older SPEC_VERSION is a contract nobody is
    checking any more."""
    for path in GOLDEN.glob("*.json"):
        spec = json.loads(path.read_text())
        assert spec["version"] == td.spec_version, (
            f"{path.name} was written for spec v{spec['version']}, library emits v{td.spec_version}"
        )
