"""Run official Mamba-ND on MPS, and check its scan against ours.

    python dossier/verify_mamba_nd.py

Mamba-ND is normally CUDA-only. It is not, quite: `Block.__init__` takes
`mixer_cls` as an **argument**, so the selective scan is injected rather than
hardcoded. Give it a portable mixer and the whole N-D scan runs anywhere.
Everything else standing in the way — `mmcv`, `mmengine`, `mmaction`,
`prettytable`, `timm`'s registry, and `mamba_ssm`'s Triton norms — is imported
at module scope for the *backbone*, not for the scan. `_shims.py` stands them
up.

**The upstream repo ships no LICENSE file.** It is therefore not vendored,
not redistributed, and not copied into this project — it stays in your own
clone and this script imports it by path. Nothing here reproduces their code.

**What is being compared.** Mamba-ND's method is not "one axis per layer". It
rearranges the flattened lattice into a per-layer axis *order* and scans the
last `n_dim_pos` axes as one sequence, batching the rest:

    n_dim_pos = 1   ->  scan one axis          == td.axial_scan
    n_dim_pos = rank->  scan every cell at once == td.flatten
    in between      ->  scan a *subset* of axes  (td has no name for this yet)

So their two published extremes are two of our methods, and the middle is a
method this library does not currently express. That is the finding; the
numbers below are the check that the first equivalence is exact.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from _shims import external, stub_mm_and_mamba_cuda  # noqa: E402

import torch_dimensions as td  # noqa: E402
from torch_dimensions.mixers.ssm import MambaMixer  # noqa: E402


def load_mamba_nd():
    """Import upstream's module from its clone, without vendoring it."""
    stub_mm_and_mamba_cuda()
    root = external("Mamba-ND") / "video_pretraining"
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("mamband", root / "models" / "mamband.py")
    module = importlib.util.module_from_spec(spec)
    # timm's @register_model looks the defining module up in sys.modules, so
    # it has to be registered before the body executes.
    sys.modules["mamband"] = module
    spec.loader.exec_module(module)
    return module


class _Injected(nn.Module):
    """Our portable mixer, wearing upstream's calling convention."""

    def __init__(self, mixer: nn.Module) -> None:
        super().__init__()
        self.mixer = mixer

    def forward(self, x, inference_params=None):  # upstream passes this kwarg
        return self.mixer(x)


# Their order strings are over four axes; the last one is what gets scanned
# when n_dim_pos=1.
ORDERS = ["t l h w", "t l w h", "w h t l"]


def compare(device: str, seed: int = 0) -> list[tuple[str, bool, float]]:
    mamband = load_mamba_nd()
    shape = (2, 3, 4, 5)  # t, l, h, w
    names = ("t", "l", "h", "w")
    d_model = 8

    lat = td.Lattice(shape=shape, names=names)
    results = []

    for order in ORDERS:
        for reverse in (False, True):
            torch.manual_seed(seed)
            mixer = MambaMixer(d_model, d_state=8).to(device).eval()

            block = mamband.Block(
                d_model,
                mixer_cls=lambda dim, _m=mixer: _Injected(_m),
                norm_cls=nn.Identity,
                fused_add_norm=False,
                reverse=reverse,
            ).to(device)
            block.eval()

            x = torch.randn(2, shape[0] * shape[1] * shape[2] * shape[3], d_model, device=device)
            with torch.no_grad():
                # skip=False strips their residual so the comparison is about
                # the *scan*, not about a block's residual convention (theirs
                # adds to the normalized tensor, ours to the input).
                theirs = block(x, order=order, shape=shape, skip=False, n_dim_pos=1)

                axis = order.split()[-1]  # n_dim_pos=1 scans the last axis named
                grid = x.reshape(2, *shape, d_model)
                ours = td.axial_apply(grid, lat, axis, mixer, reverse=reverse)
                ours = ours.reshape(2, -1, d_model)

            diff = (ours - theirs).abs().max().item()
            results.append((order, reverse, diff))
            print(
                f"  {device:4s} order '{order}' reverse={str(reverse):5s} "
                f"-> scans '{axis}'   max |ours - upstream| = {diff:.3e}"
            )
    return results


def schedule_check() -> None:
    """Their per-layer schedule against `ScanPlan.paired`.

    Upstream advances the axis ordering every *two* layers (`z = i // 2`,
    `d = z % len(orders)`) while direction alternates per layer. That is the
    claim `ScanPlan.paired` was written from; this prints both so the
    correspondence is visible rather than asserted in a docstring.
    """
    n_layers = 12
    upstream = [(ORDERS[(i // 2) % len(ORDERS)].split()[-1], bool(i % 2)) for i in range(n_layers)]
    axes = [o.split()[-1] for o in ORDERS]
    plan = td.ScanPlan.paired(tuple(axes), n_layers=n_layers, bidirectional=True)
    ours = [(step.axis, step.reverse) for step in plan]

    print("\n  layer  upstream (axis, reverse)   ScanPlan.paired")
    for i, (u, o) in enumerate(zip(upstream, ours, strict=True)):
        mark = "==" if u == o else "!="
        print(f"  {i:>4}   {str(u):<24} {mark} {o}")
    print(f"\n  schedules identical: {upstream == ours}")


def main() -> None:
    devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    if torch.cuda.is_available():
        devices.append("cuda")

    print("Mamba-ND's Block (n_dim_pos=1) vs td.axial_apply, portable mixer injected\n")
    worst = 0.0
    for device in devices:
        for _, _, diff in compare(device):
            worst = max(worst, diff)
    print(f"\nworst case: {worst:.3e}")
    schedule_check()


if __name__ == "__main__":
    main()
