"""Regenerate the viewer's bundled sample specs.

    python viewer/make_samples.py

The samples used to be artifacts with no recipe: someone produced them once,
they were committed, and when the spec format moved they silently described a
library version that no longer existed. Now they are a build step, and the
models below are the source of truth for what the viewer ships with.

Each sample is chosen to exercise a different rendering path — sparse cells,
a paired schedule, dimensional stacking at rank 4, and the kernel family's
simultaneous contraction (which has no travelling wavefront at all).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import torch_dimensions as td

OUT = Path(__file__).parent / "src" / "samples"


def sparse(shape: tuple[int, ...], keep: float, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    valid = torch.rand(shape, generator=g) < keep
    valid.reshape(-1)[0] = True
    return valid


def lstm_2d_sparse():
    lat = td.Lattice(shape=(6, 8), names=("h", "w"), valid=sparse((6, 8), 0.75), time=True)
    return td.LSTM(32, 6, lat, d_input=1, bidirectional=("h", "w"))


def mamba_3d():
    lat = td.Lattice(shape=(4, 5, 6), names=("depth", "row", "col"), time=True)
    plan = td.ScanPlan.paired(lat.axis_names, n_layers=12, bidirectional=("depth", "row", "col"))
    return td.Mamba(48, lattice=lat, plan=plan, d_input=1, d_state=16)


def s4d_4d():
    lat = td.Lattice(
        shape=(3, 4, 5, 4),
        names=("depth", "row", "col", "group"),
        valid=sparse((3, 4, 5, 4), 0.7, seed=3),
        time=True,
    )
    return td.S4D(32, 8, lat, d_input=1, d_state=32)


def cafa_hybrid():
    """The kernel family: every layer contracts both spatial axes at once and
    sweeps only time. There is no per-axis wavefront to draw, and until the
    spec described families honestly the viewer drew one anyway."""
    lat = td.Lattice(
        shape=(5, 7), names=("station", "sensor"), valid=sparse((5, 7), 0.8, seed=1), time=True
    )
    return td.LSTM(32, 5, lat, d_input=1, method=td.cafa)


def vit_joint():
    """The joint family: a ViT's patch grid, where every cell is one token and
    no axis is swept. Nothing travels, so the scene pulses instead."""
    return td.ViT(96, 6, image=(32, 32), patch=4, in_channels=3, n_heads=4, names=("row", "col"))


SAMPLES = {
    "lstm_2d_sparse": lstm_2d_sparse,
    "mamba_3d": mamba_3d,
    "s4d_4d": s4d_4d,
    "cafa_hybrid": cafa_hybrid,
    "vit_joint": vit_joint,
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, build in SAMPLES.items():
        torch.manual_seed(0)
        spec = td.spec(build())
        (OUT / f"{name}.json").write_text(json.dumps(spec, indent=1, sort_keys=True) + "\n")
        n = spec["model"]["n_params"]
        print(f"{name:16s} v{spec['version']}  {spec['nd_method']['family']:7s}  {n:,} params")


if __name__ == "__main__":
    main()
