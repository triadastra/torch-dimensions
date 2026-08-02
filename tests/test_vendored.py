"""The vendored upstream code is verifiably the original.

Three claims, all checked offline on every run:

1. Each ``.orig`` file's sha256 matches MANIFEST.json — the manifest is the
   link to the upstream commit, and ``dossier/verify_vendored.py`` proves the
   hashes against the real repositories.
2. Each working copy differs from its ``.orig`` only in lines tagged
   ``torch-dimensions patch``, and deletes exactly the lines the manifest
   records. Any other edit — however small — fails here.
3. The originals *agree with our portable mixers numerically*, which is the
   point of shipping them: anyone can run this file and watch the reference
   implementation and the rewrite produce the same numbers.
"""

from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path

import pytest
import torch

import torch_dimensions as td

VENDOR = Path(td.__file__).parent / "_vendor"
MANIFEST = json.loads((VENDOR / "MANIFEST.json").read_text())

upstream_deps = pytest.importorskip("einops", reason="vendored modules need the [upstream] extra")

MARKER = "torch-dimensions patch"


def _orig(rel: str) -> Path:
    p = VENDOR / rel
    return p if rel.endswith("LICENSE") else p.with_suffix(p.suffix + ".orig")


@pytest.mark.parametrize("rel", sorted(MANIFEST["files"]))
def test_orig_hash_matches_manifest(rel):
    entry = MANIFEST["files"][rel]
    digest = hashlib.sha256(_orig(rel).read_bytes()).hexdigest()
    assert digest == entry["sha256"], (
        f"{rel}: .orig no longer matches the manifest — the pristine copy was edited, "
        "or the manifest was not regenerated (dossier/verify_vendored.py --write-manifest)"
    )


@pytest.mark.parametrize("rel", sorted(MANIFEST["files"]))
def test_patches_are_exactly_the_documented_ones(rel):
    if rel.endswith("LICENSE"):
        return  # shipped as-is; covered by the hash test
    entry = MANIFEST["files"][rel]
    a = _orig(rel).read_text().splitlines()
    b = (VENDOR / rel).read_text().splitlines()

    added, removed = [], []
    for line in difflib.unified_diff(a, b, lineterm="", n=0):
        if line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:])
        elif line.startswith("-") and not line.startswith("---"):
            removed.append(line[1:])

    # An added line may go untagged only if it *is* one of the removed
    # original lines, re-indented — i.e. the author's own import moved inside
    # a try-guard, textually intact. Everything genuinely new must be tagged.
    removed_stripped = {line.strip() for line in removed}
    unmarked = [
        line for line in added if MARKER not in line and line.strip() not in removed_stripped
    ]
    assert not unmarked, f"{rel}: added lines without the '{MARKER}' tag: {unmarked}"
    assert removed == entry["removed_lines"], (
        f"{rel}: deleted lines differ from what the manifest records.\n"
        f"expected removals: {entry['removed_lines']}\nactual removals:   {removed}"
    )


def test_manifest_covers_every_vendored_module():
    on_disk = {str(p.relative_to(VENDOR)) for p in VENDOR.rglob("*.py") if "__init__" not in p.name}
    assert on_disk == set(MANIFEST["files"]) - {"s4/LICENSE", "mamba/LICENSE"}, (
        "a vendored module exists that MANIFEST.json does not account for (or vice versa)"
    )


# --- the originals against our portable mixers -------------------------------


def test_upstream_s4d_kernel_equals_portable_kernel():
    """Their S4DKernel and our _S4DKernel implement the same formula — copying
    parameters across must give the same kernel to float32 epsilon."""
    from torch_dimensions._vendor.s4.s4d import S4DKernel
    from torch_dimensions.mixers.ssm import _S4DKernel

    torch.manual_seed(0)
    h, n, length = 5, 16, 48
    theirs = S4DKernel(h, N=n)
    ours = _S4DKernel(h, d_state=n)
    with torch.no_grad():
        ours.log_dt.copy_(theirs.log_dt)
        ours.C.copy_(theirs.C)
        ours.log_A_real.copy_(theirs.log_A_real)
        ours.A_imag.copy_(theirs.A_imag)

    k_theirs = theirs(length)
    k_ours = ours(length)
    assert torch.allclose(k_ours, k_theirs, atol=1e-6), (k_ours - k_theirs).abs().max().item()


def test_upstream_mamba_equals_portable_mixer():
    """The authors' Mamba block (running their selective_scan_ref) against our
    portable MambaMixer, parameters copied across."""
    from torch_dimensions.mixers.upstream import UpstreamMambaMixer

    torch.manual_seed(0)
    d_model, d_state, length = 8, 8, 24
    theirs = UpstreamMambaMixer(d_model, d_state=d_state).eval()
    ours = td.mixers.MambaMixer(d_model, d_state=d_state).eval()
    with torch.no_grad():
        ours.in_proj.weight.copy_(theirs.block.in_proj.weight)
        ours.conv.weight.copy_(theirs.block.conv1d.weight)
        ours.conv.bias.copy_(theirs.block.conv1d.bias)
        ours.x_proj.weight.copy_(theirs.block.x_proj.weight)
        ours.dt_proj.weight.copy_(theirs.block.dt_proj.weight)
        ours.dt_proj.bias.copy_(theirs.block.dt_proj.bias)
        ours.A_log.copy_(theirs.block.A_log)
        ours.D.copy_(theirs.block.D)
        ours.out_proj.weight.copy_(theirs.block.out_proj.weight)

    x = torch.randn(2, length, d_model)
    with torch.no_grad():
        diff = (ours(x) - theirs(x)).abs().max().item()
    assert diff < 1e-5, diff


def test_upstream_s4d_runs_as_mixer_on_a_lattice():
    """The verbatim S4D block swept over a 2-D lattice by our composition."""
    from torch_dimensions.mixers.upstream import UpstreamS4DMixer

    torch.manual_seed(0)
    lat = td.Lattice(shape=(4, 5), names=("y", "x"))
    mixer = UpstreamS4DMixer(6, d_state=8).eval()
    grid = torch.randn(2, 4, 5, 6)
    out = td.axial_apply(grid, lat, "x", mixer)
    assert out.shape == grid.shape
    assert torch.isfinite(out).all()


def test_upstream_s4_block_runs():
    """The full DPLR S4Block, verbatim, on CPU."""
    from torch_dimensions.mixers.upstream import UpstreamS4Mixer

    torch.manual_seed(0)
    mixer = UpstreamS4Mixer(6, d_state=8).eval()
    x = torch.randn(2, 32, 6)
    with torch.no_grad():
        y = mixer(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_upstream_mixer_through_model_api():
    """`mixer=` substitution puts the verbatim block inside a full N-D model."""
    from torch_dimensions.mixers.upstream import UpstreamMambaMixer

    torch.manual_seed(0)
    lat = td.Lattice(shape=(3, 4), names=("y", "x"))
    model = td.Mamba(
        8, n_layers=2, lattice=lat, mixer=UpstreamMambaMixer, mixer_kwargs={"d_state": 8}
    ).eval()
    x = torch.randn(2, 3, 4, 8)  # (B, y, x, H)
    with torch.no_grad():
        y = model(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_upstream_extra_missing_message(monkeypatch):
    """Without einops the adapters must say how to get it, not stack-trace."""
    import builtins

    real_import = builtins.__import__

    def no_einops(name, *args, **kwargs):
        if name == "einops":
            raise ImportError("No module named 'einops'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_einops)
    from torch_dimensions.mixers.upstream import UpstreamMambaMixer

    with pytest.raises(ImportError, match=r"torch-dimensions\[upstream\]"):
        UpstreamMambaMixer(8)
