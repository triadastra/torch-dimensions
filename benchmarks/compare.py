"""Compare two `pretrain.py` runs — one machine against another.

    python benchmarks/compare.py "MPS bench" "CUDA bench" --out COMPARISON.md

Both runs start from bit-identical weights on bit-identical data, so every
number below is a difference the *arithmetic* produced. What that means
depends on the row:

- **the portable models** (``*_portable_*``, RNNs, conv, attention) run the
  same pure-torch code on both machines, so a difference is float
  non-associativity — different reduction orders on different hardware.
  Expect small and growing-with-steps, not zero.
- **the vendored models** run the authors' *fused* kernels on CUDA and their
  reference path on MPS. Those rows are fused-vs-reference, and they are the
  ones worth reading: this is the check PLAN.md fixes as the rule for any
  fast path.

A loss trajectory is chaotic, so late-step agreement is not the right test —
two runs of the same model can separate simply because gradient descent
amplifies. The honest summary is the *early* divergence and whether both runs
landed in the same place, which is why both are reported.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


def load(path: Path) -> dict:
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["_dir"] = path
    return manifest


def weights_of(run: Path, name: str) -> dict | None:
    blob = run / name / "weights.pt"
    if not blob.exists():
        return None
    return torch.load(blob, map_location="cpu", weights_only=True)


def weight_delta(a: dict, b: dict) -> tuple[float, float]:
    """Relative and absolute distance between two trained parameter sets."""
    num = 0.0
    den = 0.0
    worst = 0.0
    for key, va in a["model"].items():
        vb = b["model"][key]
        d = (va.double() - vb.double()).abs()
        num += float((d**2).sum())
        den += float((va.double() ** 2).sum())
        worst = max(worst, float(d.max()))
    return (math.sqrt(num / max(den, 1e-30)), worst)


def first_divergence(la: list[float], lb: list[float], tol: float = 1e-6) -> int | None:
    """The first step whose losses differ by more than `tol` relatively.

    Says when the two machines stopped agreeing, which is far more informative
    than how far apart they ended up: the end of a chaotic trajectory is not a
    measurement of anything.
    """
    for i, (x, y) in enumerate(zip(la, lb, strict=False)):
        scale = max(abs(x), abs(y), 1e-12)
        if abs(x - y) / scale > tol:
            return i
    return None


def _weights_provenance(left: dict, right: dict) -> list[str]:
    """Say — and check — where each run's starting weights came from.

    The comparison used to assert that both machines began from bit-identical
    weights. For S4 and S4D that was false: `torch.linalg.eigh` fixes
    eigenvectors only up to a phase, so their `B` and `P` differ between macOS
    and Linux under the same seed, and the comparison reported a 2.6e-01
    difference that was two different models rather than two devices.

    A claim the instrument cannot check is worth less than one it can, so this
    reads what the runs recorded instead of assuming.
    """

    def sources(run: dict) -> set[str]:
        return {m.get("weights_from", "seed") for m in run["models"] if "error" not in m}

    both = sources(left) | sources(right)
    if both <= {"written", "loaded"}:
        return [
            "Both runs load one shared set of starting weights, so the initial",
            "conditions are bit-identical and every difference below is arithmetic.",
            "",
        ]
    return [
        "> **The two runs did not share starting weights** "
        f"({', '.join(sorted(both))}). The seed alone does not give identical",
        "> S4/S4D weights across platforms — `eigh`'s eigenvectors are fixed only",
        "> up to a phase — so differences below may be initialisation rather than",
        "> arithmetic. Re-run both sides with `--init <dir>` pointing at the same",
        "> directory. See benchmarks/init_weights.py.",
        "",
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("left")
    ap.add_argument("right")
    ap.add_argument("--out", default=None, help="write a markdown table here")
    args = ap.parse_args()

    left, right = load(Path(args.left)), load(Path(args.right))
    by_name = {
        "left": {m["name"]: m for m in left["models"] if "error" not in m},
        "right": {m["name"]: m for m in right["models"] if "error" not in m},
    }
    shared = [n for n in by_name["left"] if n in by_name["right"]]

    lines = [
        "# Device comparison",
        "",
        f"- **{args.left}** — {left['device_name']} · torch {left['torch']}"
        f" · torch-dimensions {left['torch_dimensions']}",
        f"- **{args.right}** — {right['device_name']} · torch {right['torch']}"
        f" · torch-dimensions {right['torch_dimensions']}",
        "",
        f"Both runs: seed {left['seed']}, {left['steps']} steps, batch {left['batch']}.",
        "Data is drawn on CPU from a seeded generator, so both machines see the",
        "same batches in the same order.",
        "",
        *_weights_provenance(left, right),
        "| model | loss (left) | loss (right) | Δ loss | first divergence "
        "| rel Δw | max Δw | speed |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for name in shared:
        a, b = by_name["left"][name], by_name["right"][name]
        wa, wb = weights_of(left["_dir"], name), weights_of(right["_dir"], name)
        if wa and wb:
            rel_w, max_w = weight_delta(wa, wb)
            wcol, mcol = f"{rel_w:.2e}", f"{max_w:.2e}"
        else:
            wcol = mcol = "—"
        step = first_divergence(a["losses"], b["losses"])
        dcol = "identical" if step is None else f"step {step}"
        speed = b["steps_per_second"] / max(a["steps_per_second"], 1e-9)
        lines.append(
            f"| `{name}` | {a['loss_final']:.5f} | {b['loss_final']:.5f} | "
            f"{abs(a['loss_final'] - b['loss_final']):.2e} | {dcol} | {wcol} | {mcol} | "
            f"{speed:.2f}× |"
        )

    missing = sorted(set(by_name["left"]) ^ set(by_name["right"]))
    if missing:
        lines += ["", f"Not in both runs: {', '.join('`' + m + '`' for m in missing)}."]

    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
