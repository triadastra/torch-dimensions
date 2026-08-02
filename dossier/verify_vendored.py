"""Verify the vendored upstream files against the real upstream repositories.

    python dossier/verify_vendored.py                 # check .orig == upstream at pinned commit
    python dossier/verify_vendored.py --write-manifest  # (maintainers) re-pin to the clone's HEAD

The offline half of this scheme lives in tests/test_vendored.py and runs in
CI: it checks that each ``.orig`` file matches the sha256 recorded in
MANIFEST.json and that the working copy differs from its ``.orig`` only in
lines tagged ``torch-dimensions patch``. What CI cannot know is whether the
``.orig`` files are really upstream's bytes — that is this script's job. It
fetches each repository at the *pinned commit* (not HEAD, so upstream moving
on is reported as drift rather than failure) and compares byte-for-byte.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _shims import UPSTREAM, external  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
VENDOR = REPO_ROOT / "src" / "torch_dimensions" / "_vendor"
MANIFEST = VENDOR / "MANIFEST.json"

# The s4 side is the *pipeline* their train.py runs (under src/), vendored
# with its directory structure intact so the code's own `src.*` imports and
# hydra string-targets resolve unchanged; see _vendor/s4/__init__.py.
_S4_PIPELINE = [
    "models/sequence/modules/s4block.py",
    "models/sequence/modules/s4nd.py",
    "models/sequence/kernels/__init__.py",
    "models/sequence/kernels/fftconv.py",
    "models/sequence/kernels/kernel.py",
    "models/sequence/kernels/ssm.py",
    "models/sequence/kernels/dplr.py",
    "models/sequence/__init__.py",
    "models/sequence/base.py",
    "models/nn/__init__.py",
    "models/nn/linear.py",
    "models/nn/activation.py",
    "models/nn/normalization.py",
    "models/nn/dropout.py",
    "models/hippo/hippo.py",
    "models/functional/cauchy.py",
    "models/functional/vandermonde.py",
    "models/functional/krylov.py",
    "models/functional/toeplitz.py",
    "utils/__init__.py",
    "utils/config.py",
    "utils/registry.py",
    "utils/train.py",
]

# vendored path (relative to _vendor) -> (repo key, upstream path, patched?)
# `patched` means the working copy differs from upstream by tagged lines and
# a `.orig` pristine copy ships beside it; unpatched files ARE the pristine
# copy and their own sha256 is the check.
FILES = {
    **{
        f"s4/src/{p}": (
            "s4",
            f"src/{p}",
            # utils/train.py: training-only imports guarded.
            # kernels/ssm.py: exact-hit Nyquist-pole guard (MPS portability;
            # inert on CPU/CUDA — verified bit-for-bit unchanged there).
            p in ("utils/train.py", "models/sequence/kernels/ssm.py"),
        )
        for p in _S4_PIPELINE
    },
    "s4/LICENSE": ("s4", "LICENSE", False),
    "mamba/mamba_simple.py": ("mamba", "mamba_ssm/modules/mamba_simple.py", True),
    "mamba/selective_scan_interface.py": (
        "mamba",
        "mamba_ssm/ops/selective_scan_interface.py",
        True,
    ),
    "mamba/utils_torch.py": ("mamba", "mamba_ssm/utils/torch.py", False),
    "mamba/LICENSE": ("mamba", "LICENSE", False),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def orig_of(rel: str, patched: bool) -> Path:
    """The pristine copy: `.orig` beside each patched module, the file itself
    otherwise (an unpatched vendored file IS upstream's bytes)."""
    p = VENDOR / rel
    return p.with_suffix(p.suffix + ".orig") if patched else p


def removed_lines(rel: str, patched: bool) -> list[str]:
    """Lines of the original that the patched working copy no longer contains,
    recorded in the manifest so the offline test can pin deletions exactly."""
    if not patched:
        return []
    import difflib

    a = orig_of(rel, patched).read_text().splitlines()
    b = (VENDOR / rel).read_text().splitlines()
    return [
        line[1:]
        for line in difflib.unified_diff(a, b, lineterm="", n=0)
        if line.startswith("-") and not line.startswith("---")
    ]


def write_manifest() -> None:
    repos = {}
    for key in {repo for repo, _, _ in FILES.values()}:
        path = external(key)  # clones if absent
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True, check=True
        ).stdout.strip()
        repos[key] = {
            "url": UPSTREAM[key]["url"],
            "license": UPSTREAM[key]["license"],
            "commit": commit,
        }

    files = {}
    for rel, (repo, upstream_path, patched) in FILES.items():
        files[rel] = {
            "repo": repo,
            "upstream_path": upstream_path,
            "patched": patched,
            "sha256": sha256(orig_of(rel, patched)),
            "removed_lines": removed_lines(rel, patched),
        }
    MANIFEST.write_text(json.dumps({"repos": repos, "files": files}, indent=2) + "\n")
    print(f"wrote {MANIFEST}")


def verify() -> int:
    manifest = json.loads(MANIFEST.read_text())
    failures = 0
    for rel, entry in manifest["files"].items():
        repo = entry["repo"]
        clone = external(repo)
        pinned = manifest["repos"][repo]["commit"]
        # Read the file at the *pinned* commit, so upstream moving on is
        # reported as drift, not treated as our error.
        show = subprocess.run(
            ["git", "show", f"{pinned}:{entry['upstream_path']}"],
            cwd=clone,
            capture_output=True,
            check=False,
        )
        if show.returncode != 0:
            # Shallow clone may not have the pinned commit; deepen and retry.
            subprocess.run(
                ["git", "fetch", "--depth", "50", "origin", pinned], cwd=clone, check=False
            )
            show = subprocess.run(
                ["git", "show", f"{pinned}:{entry['upstream_path']}"],
                cwd=clone,
                capture_output=True,
                check=False,
            )
        if show.returncode != 0:
            err = show.stderr.decode().strip()
            print(f"  {rel}: could not read {entry['upstream_path']} at {pinned[:12]} — {err}")
            failures += 1
            continue
        theirs = hashlib.sha256(show.stdout).hexdigest()
        ours = sha256(orig_of(rel, entry["patched"]))
        ok = theirs == ours == entry["sha256"]
        print(f"  {rel}: {'byte-identical to upstream ' + pinned[:12] if ok else 'MISMATCH'}")
        if not ok:
            failures += 1

        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=clone, capture_output=True, text=True, check=True
        ).stdout.strip()
        if head != pinned:
            hd = subprocess.run(
                ["git", "show", f"{head}:{entry['upstream_path']}"],
                cwd=clone,
                capture_output=True,
                check=False,
            )
            if hd.returncode == 0 and hashlib.sha256(hd.stdout).hexdigest() != theirs:
                print(
                    f"    note: upstream HEAD ({head[:12]}) has since changed this file "
                    "— drift, not error"
                )
    return failures


if __name__ == "__main__":
    if "--write-manifest" in sys.argv:
        write_manifest()
        raise SystemExit(0)
    print("vendored .orig files vs upstream at pinned commits\n")
    n = verify()
    print(f"\n{'all verified' if n == 0 else f'{n} FAILURES'}")
    raise SystemExit(1 if n else 0)
