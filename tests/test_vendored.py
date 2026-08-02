"""The vendored upstream code is verifiably the original.

Three claims, all checked offline on every run:

1. Every vendored file's pristine bytes match MANIFEST.json — the manifest is
   the link to the upstream commit, and ``dossier/verify_vendored.py`` proves
   the hashes against the real repositories. For unpatched files the vendored
   file *is* the pristine copy; patched files ship a ``.orig`` beside them.
2. Each patched file differs from its ``.orig`` only in lines tagged
   ``torch-dimensions patch``, and deletes exactly the lines the manifest
   records. Any other edit — however small — fails here.
3. The originals *agree with our portable mixers numerically*, which is the
   point of shipping them: anyone can run this file and watch the reference
   implementation and the rewrite produce the same numbers.

The s4 side is the pipeline upstream's train.py actually runs (S4Block and
everything under it), mounted as the ``src`` package exactly as their repo
convention expects — see ``torch_dimensions._vendor.s4``.
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
pytest.importorskip("hydra", reason="the s4 pipeline needs hydra-core (the [upstream] extra)")

MARKER = "torch-dimensions patch"


def _orig(rel: str) -> Path:
    p = VENDOR / rel
    return p.with_suffix(p.suffix + ".orig") if MANIFEST["files"][rel]["patched"] else p


@pytest.mark.parametrize("rel", sorted(MANIFEST["files"]))
def test_pristine_bytes_match_manifest(rel):
    entry = MANIFEST["files"][rel]
    digest = hashlib.sha256(_orig(rel).read_bytes()).hexdigest()
    assert digest == entry["sha256"], (
        f"{rel}: pristine copy no longer matches the manifest — the file was edited, "
        "or the manifest was not regenerated (dossier/verify_vendored.py --write-manifest)"
    )


@pytest.mark.parametrize(
    "rel", sorted(r for r in MANIFEST["files"] if MANIFEST["files"][r]["patched"])
)
def test_patches_are_exactly_the_documented_ones(rel):
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
        line
        for line in added
        if MARKER not in line and line.strip() not in removed_stripped and line.strip()
    ]
    assert not unmarked, f"{rel}: added lines without the '{MARKER}' tag: {unmarked}"
    assert removed == entry["removed_lines"], (
        f"{rel}: deleted lines differ from what the manifest records.\n"
        f"expected removals: {entry['removed_lines']}\nactual removals:   {removed}"
    )


def test_manifest_covers_every_vendored_module():
    ours = {"__init__.py", "s4/__init__.py", "mamba/__init__.py"}  # our loaders, not upstream's
    on_disk = {
        str(p.relative_to(VENDOR))
        for p in VENDOR.rglob("*.py")
        if str(p.relative_to(VENDOR)) not in ours
    }
    expected = {r for r in MANIFEST["files"] if r.endswith(".py")}
    assert on_disk == expected, (
        "vendored modules and MANIFEST.json disagree.\n"
        f"on disk but not accounted for: {sorted(on_disk - expected)}\n"
        f"in manifest but missing on disk: {sorted(expected - on_disk)}"
    )
    for rel, entry in MANIFEST["files"].items():
        if entry["patched"]:
            assert (VENDOR / (rel + ".orig")).exists(), f"{rel}: patched but no .orig beside it"


# --- the originals against our portable mixers -------------------------------


def _pipeline():
    from torch_dimensions._vendor.s4 import mount

    mount()


def test_pipeline_s4d_kernel_equals_portable_kernel():
    """The pipeline's SSMKernelDiag (init='diag-lin', disc='zoh' — the S4D-Lin
    setup) and our _S4DKernel implement the same formula: copying parameters
    across must give the same kernel to float32 epsilon."""
    _pipeline()
    from src.models.sequence.kernels.ssm import SSMKernelDiag

    from torch_dimensions.mixers.ssm import _S4DKernel

    torch.manual_seed(0)
    h, n, length = 5, 16, 48
    theirs = SSMKernelDiag(d_model=h, d_state=n, init="diag-lin", disc="zoh", dt_transform="exp")
    ours = _S4DKernel(h, d_state=n)
    with torch.no_grad():
        theirs.inv_dt.copy_(ours.log_dt.unsqueeze(-1))
        theirs.A_real.copy_(ours.log_A_real)  # both store log(-Re A) under 'exp'
        theirs.A_imag.copy_(ours.A_imag)  # 'none' transform stores -Im A directly
        # The pipeline keeps the negative-imaginary conjugate half (A = -re - i*im)
        # where ours keeps the positive; 2*Re(sum C exp(dtA t)) is unchanged iff C
        # is conjugated along with A. B is constant ones under diag-lin.
        c = torch.view_as_complex(ours.C.detach().clone())
        theirs.C.copy_(torch.view_as_real(c.conj().resolve_conj()).unsqueeze(0))

    k_theirs, _ = theirs.forward(L=length)  # (channels=1, H, L)
    k_ours = ours(length)  # (H, L)
    diff = (k_ours - k_theirs[0]).abs().max().item()
    assert diff < 1e-6, diff


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


def test_pipeline_s4d_runs_as_mixer_on_a_lattice():
    """The real S4Block (mode='diag'), built through upstream's own hydra
    registry, swept over a 2-D lattice by our composition."""
    from torch_dimensions.mixers.upstream import UpstreamS4DMixer

    torch.manual_seed(0)
    lat = td.Lattice(shape=(4, 5), names=("y", "x"))
    mixer = UpstreamS4DMixer(6, d_state=8).eval()
    grid = torch.randn(2, 4, 5, 6)
    out = td.axial_apply(grid, lat, "x", mixer)
    assert out.shape == grid.shape
    assert torch.isfinite(out).all()


def test_pipeline_s4_dplr_runs():
    """The full DPLR S4Block — the layer upstream's registry calls "s4"."""
    from torch_dimensions.mixers.upstream import UpstreamS4Mixer

    torch.manual_seed(0)
    mixer = UpstreamS4Mixer(6, d_state=8).eval()
    x = torch.randn(2, 32, 6)
    with torch.no_grad():
        y = mixer(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_pipeline_s4nd_is_importable():
    """The real S4ND layer ships too — the module their registry calls
    "s4nd". Constructing it here proves the vendored subtree is complete."""
    _pipeline()
    from src.models.sequence.modules.s4nd import S4ND

    torch.manual_seed(0)
    layer = S4ND(d_model=6, dim=2, l_max=(8, 9), contract_version=1).eval()
    x = torch.randn(2, 6, 8, 9)  # their layout: (B, H, *spatial)
    with torch.no_grad():
        y, _ = layer(x)
    assert y.shape == x.shape


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


def test_mount_refuses_a_foreign_src(monkeypatch):
    """A process that already imported its own `src` package must get a clear
    error, not silent shadowing in either direction."""
    import sys
    import types

    from torch_dimensions._vendor.s4 import mount

    foreign = types.ModuleType("src")
    foreign.__path__ = ["/somewhere/else/src"]
    monkeypatch.setitem(sys.modules, "src", foreign)
    with pytest.raises(ImportError, match="cannot coexist"):
        mount()


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
