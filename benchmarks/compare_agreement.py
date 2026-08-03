"""Compare two Bench A runs — did the two devices compute the same thing?

    python benchmarks/compare_agreement.py "MPS agree" "CUDA agree"

`agreement.py` runs one forward and one backward from fixed weights on fixed
data, with no optimiser anywhere. This reads two of those runs and reports the
difference. Because nothing was trained, a difference here *is* the
arithmetic: there is no accumulated trajectory to blame it on.

The comparison is elementwise on the saved tensors, not on the JSON summaries.
Two runs can agree on a sum while disagreeing everywhere, with the errors
cancelling; a maximum over elements cannot be fooled that way.

**Relative to what.** Differences are scaled by the larger of the two tensors'
magnitudes, so the number is readable against float epsilon: ~1e-7 for
float32, ~1e-16 for float64. That comparison across dtypes is the point of
measuring both. If a difference shrinks by nine orders of magnitude when the
precision does, it is float non-associativity — the two devices summed the
same terms in a different order. If it stays put, the two devices ran
different computations, and for the vendored models on CUDA that is expected
and intended: the fused kernel *is* a different implementation. Which of the
two it is cannot be read off a single-precision run, which is why a run with
no float64 column reports its absence rather than passing quietly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

# The bound this project holds itself to in float32. float32 epsilon is
# ~1.2e-07, so 1e-06 is roughly eight ulps — reachable for a forward pass
# through several layers, and the measured worst output difference across the
# whole zoo is 3.1e-06 (the three cuDNN RNNs; everything else is under).
#
# Gradients are judged by the same number but do not always obey it, and that
# is a property of the quantity rather than of the device: the derivative with
# respect to an SSM frequency is a sum of oscillating terms that nearly cancel,
# so its *relative* error in float32 is large by construction. `A_imag` differs
# by 1.35e-04 in float32 and 4.87e-15 in float64 — eleven orders — which is
# what cancellation looks like and what a different computation does not.
NOTABLE = 1e-6


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    scale = max(float(a.abs().max()), float(b.abs().max()), 1e-12)
    return float((a - b).abs().max()) / scale


def compare_tensors(left: Path, right: Path) -> dict | None:
    """Worst elementwise relative difference, over the output and every
    gradient. Returns None when either side did not save this pair."""
    if not left.exists() or not right.exists():
        return None
    a = torch.load(left, map_location="cpu", weights_only=False)
    b = torch.load(right, map_location="cpu", weights_only=False)

    out = rel(a["output"], b["output"])
    grads = {}
    for name, ga in a["grads"].items():
        gb = b["grads"].get(name)
        if gb is not None and ga.shape == gb.shape:
            grads[name] = rel(ga, gb)

    worst_name = max(grads, key=grads.get) if grads else None
    return {
        "output": out,
        "grad_worst": grads[worst_name] if worst_name else None,
        "grad_worst_name": worst_name,
        "n_grads": len(grads),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("left")
    ap.add_argument("right")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    left, right = Path(args.left), Path(args.right)
    lj = json.loads((left / "agreement.json").read_text())
    rj = json.loads((right / "agreement.json").read_text())

    lines = [
        f"# Agreement — {lj['device_name']} vs {rj['device_name']}",
        "",
        f"- **{left.name}** — {lj['device_name']}, torch {lj['torch']}, "
        f"dtypes {', '.join(lj['dtypes'])}",
        f"- **{right.name}** — {rj['device_name']}, torch {rj['torch']}, "
        f"dtypes {', '.join(rj['dtypes'])}",
        "",
        "One forward and one backward from identical weights on identical data,",
        "with no optimiser in the loop — so what is measured below is the",
        "arithmetic and nothing else. Differences are elementwise maxima,",
        "relative to the larger tensor's magnitude.",
        "",
    ]

    if lj["torch"] != rj["torch"]:
        lines += [
            f"> **Note.** The two runs are on different torch versions "
            f"({lj['torch']} vs {rj['torch']}). That is a second difference "
            "between them besides the device, and a large disagreement cannot "
            "be attributed to the hardware alone.",
            "",
        ]

    shared = [d for d in lj["dtypes"] if d in rj["dtypes"]]
    missing = [d for d in ("float32", "float64") if d not in shared]
    if missing:
        lines += [
            f"> **{', '.join(missing)} is not compared**: not present in both "
            "runs. MPS has no float64, so the precision control — the thing "
            "that separates reassociation from a different computation — is "
            "unavailable for this pair. It is recorded rather than skipped "
            "silently.",
            "",
        ]

    names = [n for n in lj["models"] if n in rj["models"]]
    rows: list[tuple[str, str, dict]] = []
    for name in names:
        for dtype in shared:
            le, re_ = lj["models"][name].get(dtype, {}), rj["models"][name].get(dtype, {})
            if "error" in le or "error" in re_:
                which = left.name if "error" in le else right.name
                rows.append((name, dtype, {"error": f"failed in {which}"}))
                continue
            cmp = compare_tensors(left / f"{name}.{dtype}.pt", right / f"{name}.{dtype}.pt")
            if cmp is None:
                rows.append((name, dtype, {"error": "tensors not saved"}))
                continue
            cmp["loss_left"], cmp["loss_right"] = le["loss"], re_["loss"]
            rows.append((name, dtype, cmp))

    lines += [
        "| model | dtype | output | worst gradient | which gradient | loss "
        f"({left.name}) | loss ({right.name}) |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, dtype, c in rows:
        if "error" in c:
            lines.append(f"| `{name}` | {dtype} | — | — | {c['error']} | — | — |")
            continue
        gw = "—" if c["grad_worst"] is None else f"{c['grad_worst']:.2e}"
        flag = " ⚠" if c["output"] > NOTABLE or (c["grad_worst"] or 0) > NOTABLE else ""
        lines.append(
            f"| `{name}`{flag} | {dtype} | {c['output']:.2e} | {gw} | "
            f"`{c['grad_worst_name'] or '—'}` | {c['loss_left']:.6f} | "
            f"{c['loss_right']:.6f} |"
        )

    ok = [c for _, _, c in rows if "error" not in c]
    flagged = [
        (n, d, c)
        for n, d, c in rows
        if "error" not in c and (c["output"] > NOTABLE or (c["grad_worst"] or 0) > NOTABLE)
    ]
    lines += [
        "",
        f"⚠ marks a difference above {NOTABLE:.0e}, in the output or in any gradient.",
        "",
        "## What this says",
        "",
        f"- {len(ok)} of {len(rows)} model-dtype pairs compared successfully.",
    ]
    if ok:
        worst = max(ok, key=lambda c: c["output"])
        lines.append(f"- Worst **output** difference across everything: {worst['output']:.2e}.")
        wg = max(ok, key=lambda c: c["grad_worst"] or 0)
        lines.append(
            f"- Worst **gradient** difference: {wg['grad_worst']:.2e} on `{wg['grad_worst_name']}`."
        )
    if flagged:
        lines.append(
            f"- {len(flagged)} pair(s) above {NOTABLE:.0e}: "
            + ", ".join(f"`{n}` ({d})" for n, d, _ in flagged)
            + "."
        )
        lines.append(
            "- A row flagged only on a *gradient* of an SSM frequency (`A_imag`, "
            "`A_real`) is cancellation, not disagreement: those gradients sum "
            "oscillating terms that nearly cancel, and the same pair that differs "
            "by 1.35e-04 in float32 differs by 4.87e-15 in float64. A row flagged "
            "on its *output* is worth reading — for a vendored model on CUDA it is "
            "the expected signature of the fused kernel being a different "
            "implementation of the same recurrence."
        )
    else:
        lines.append(
            f"- No pair differs by more than {NOTABLE:.0e}: on this workload every "
            "model computes the same thing on both devices to within float "
            "reassociation."
        )

    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
