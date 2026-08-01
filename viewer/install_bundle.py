"""Copy the built viewer into the package, where the wheel can carry it.

    cd viewer && npm install && npm run build
    python viewer/install_bundle.py

Run from the repo root or from ``viewer/``; both work. The bundle lands in
``src/torch_dimensions/viz/static`` and is deliberately **not** committed —
it is a build artifact, it changes on every rebuild, and a repo that carries
its own build output accumulates diffs nobody reads. CI builds it before
packaging; ``pyproject.toml`` lists it under ``artifacts`` so hatchling
includes it even though ``.gitignore`` hides it.

The size guard here is the sdist-bloat lesson from DEBUG.md, one directory
over: `node_modules` once reached a released tarball because nothing looked at
what was actually in it.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

MAX_MB = 8.0

root = Path(__file__).resolve().parent.parent
src = root / "viewer" / "dist"
dest = root / "src" / "torch_dimensions" / "viz" / "static"


def main() -> int:
    if not (src / "index.html").exists():
        print(f"no build found at {src}\nrun: cd viewer && npm install && npm run build")
        return 1

    size_mb = sum(f.stat().st_size for f in src.rglob("*") if f.is_file()) / 2**20
    files = sum(1 for f in src.rglob("*") if f.is_file())
    if size_mb > MAX_MB:
        print(f"refusing to install a {size_mb:.1f} MB bundle (limit {MAX_MB} MB)")
        print("something is being bundled that should not be — check for source maps or assets")
        return 1
    if any(p.name == "node_modules" for p in src.rglob("*")):
        print("node_modules is inside the build output; refusing")
        return 1

    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)

    # Vite copies everything in `viewer/public/` into the build, and the live
    # training script writes `viewer/public/run.json` there. Left alone, a
    # wheel would carry whatever run happened to be on the packager's laptop —
    # and the viewer would load it in preference to the model the user passed
    # to `td.viz.show`, which is how this was noticed at all.
    for stray in dest.rglob("run.json"):
        stray.unlink()
        print(f"stripped {stray.relative_to(dest)} (a local run, not part of the viewer)")

    installed = sum(1 for f in dest.rglob("*") if f.is_file())
    print(f"installed {installed} of {files} files, {size_mb:.2f} MB -> {dest.relative_to(root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
