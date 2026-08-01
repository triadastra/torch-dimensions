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

**Exactly what is in the measured path**, because a stub that participates in
a measurement is part of the result. With `skip=False` their block computes
`drop_path(dropout(mixer(x)))` around the rearrange, and:

- `mixer` is **ours**, injected on purpose — that is the experiment;
- `norm` is `nn.Identity`, passed by us;
- `dropout` and `drop_path` come from **our stub**, which returns `Identity()`
  at the default rate of 0.

So two stubbed components do sit in the path. They contribute nothing
numerically *because* they are identities at rate 0, which is why the
comparison can be bitwise rather than approximate — "they are identities" is
the reason, not "they are absent". The one assumption left standing: real
`mmcv`'s `DropPath(drop_prob=0)` is likewise the identity. That is the
definition of stochastic depth, but it is an assumption here rather than a
measurement, since mmcv is not installed.

**What is being compared.** Mamba-ND's method is not "one axis per layer". It
rearranges the flattened lattice into a per-layer axis *order* and scans the
last `n_dim_pos` axes as one sequence, batching the rest:

    n_dim_pos = 1    -> one sequence per line      == td.axial_scan
    n_dim_pos = rank -> one sequence for everything == td.flatten
    in between       -> a subset scanned jointly    (we have no name for it)

The paper calls this choice **scan factorization** (§4.1, Fig. 5b: "no
factorization, there is only 1 sequence" through "factorizing the 3D sequence
into D² 1D sequences"). So their two extremes are our two methods, and the
middle — which their published configs actually use — is one this library
cannot express. That is the finding; the numbers below check that the
maximally-factorized corner is exact.

A note on their notation, from the paper rather than the code: `H+` is not
"scan the H axis". §4.1 defines a scan ordering as a permutation of the axes
flattened whole into one sequence, "the last dimension traversed
continuously", so `H+` names the permutation ending in H.
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
    `d = z % len(orders)`) while direction alternates per layer, which is also
    what the paper specifies in words (Fig. 2: "in 3D space, we use the order
    H+ H- W+ W- T+ T-"). This prints both schedules so the correspondence is
    visible rather than asserted. See NOTICE for what the match does and does
    not establish about provenance.
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
