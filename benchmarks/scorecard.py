"""Score the models in a benchmark run, on each axis separately.

    python benchmarks/scorecard.py "MPS bench" --out SCORECARD.md

**There is deliberately no single score.** A composite number would need
weights — how much is a point of loss worth in steps per second? — and those
weights would be this file's opinion dressed up as the models' property. What
is here instead is one ranking per question, so a reader can apply their own
weighting to numbers that mean something on their own:

- **how well** — the loss it reached
- **how fast it got there** — steps to cross 90% of its own improvement
- **how steady** — spread over the last tenth of training, which is what tells
  a converged model from one still bouncing
- **how quickly it runs** — steps per second
- **what it cost** — parameters, and improvement per thousand of them

Every number here is one task on one machine, and the task is a cumulative sum
along an axis. That suits sequence models with a causal sweep and is close to
the worst case for a permutation-invariant one, so the ranking is a ranking on
*this* problem and says nothing about images or forecasting. It is reported
this way rather than as "the best model" for that reason.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def convergence_step(losses: list[float], fraction: float = 0.9) -> int | None:
    """First step at which `fraction` of the run's total improvement is done.

    Measures speed of learning independently of where it ended up: a model that
    reaches its plateau in 40 steps and one that takes 300 can finish at the
    same loss and are not the same model.
    """
    if len(losses) < 2:
        return None
    start, best = losses[0], min(losses)
    if start <= best:
        return None  # never improved; the question does not apply
    target = start - fraction * (start - best)
    for i, v in enumerate(losses):
        if v <= target:
            return i
    return None


def tail_spread(losses: list[float], portion: float = 0.1) -> float:
    """Standard deviation over the last `portion` of training, relative to its
    mean — a converged run is flat there and a bouncing one is not."""
    n = max(2, int(len(losses) * portion))
    tail = losses[-n:]
    mean = sum(tail) / len(tail)
    var = sum((v - mean) ** 2 for v in tail) / len(tail)
    return (var**0.5) / max(abs(mean), 1e-12)


def rank(rows: list[dict], key: str, *, lower_is_better: bool = True) -> dict[str, int]:
    """Ordinal rank on one column. Rows with no value are left unranked rather
    than given a default, which would invent a position for them."""
    have = [r for r in rows if r.get(key) is not None]
    have.sort(key=lambda r: r[key], reverse=not lower_is_better)
    return {r["name"]: i + 1 for i, r in enumerate(have)}


def score(run: Path) -> list[dict]:
    manifest = json.loads((run / "manifest.json").read_text())
    rows = []
    for entry in manifest["models"]:
        if "error" in entry:
            continue
        losses = entry["losses"]
        improvement = entry["loss_first"] - min(losses)
        rows.append(
            {
                "name": entry["name"],
                "params": entry["n_params"],
                "loss": entry["loss_final"],
                "best": min(losses),
                "converged_at": convergence_step(losses),
                "spread": tail_spread(losses),
                "steps_per_s": entry["steps_per_second"],
                # Improvement bought per thousand parameters: the only column
                # that asks whether the capacity was worth carrying.
                "per_1k": improvement / max(entry["n_params"] / 1000, 1e-9),
                "learned": improvement > 0.05 * abs(entry["loss_first"]),
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run = Path(args.run)
    manifest = json.loads((run / "manifest.json").read_text())
    rows = score(run)

    ranks = {
        "loss": rank(rows, "loss"),
        "converged_at": rank(rows, "converged_at"),
        "spread": rank(rows, "spread"),
        "speed": rank(rows, "steps_per_s", lower_is_better=False),
        "per_1k": rank(rows, "per_1k", lower_is_better=False),
    }

    lines = [
        f"# Scorecard — {manifest['device_name']}",
        "",
        f"{manifest['steps']} steps, batch {manifest['batch']}, seed {manifest['seed']}, "
        f"torch {manifest['torch']}, torch-dimensions {manifest['torch_dimensions']}.",
        "",
        "One task — a cumulative sum along an axis — on one machine. That suits a",
        "causal sequence model and is close to the worst case for a permutation-",
        "invariant one, so this ranks models *on this problem* and implies nothing",
        "about images or forecasting. There is no combined score on purpose: the",
        "weighting between a point of loss and a step per second is the reader's,",
        "not this file's.",
        "",
        "| model | params | final loss | best | 90% at | tail spread "
        "| steps/s | Δloss per 1k params |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(rows, key=lambda r: r["loss"]):
        conv = "—" if r["converged_at"] is None else f"step {r['converged_at']}"
        flag = "" if r["learned"] else " ⚠"
        lines.append(
            f"| `{r['name']}`{flag} | {r['params']:,} | {r['loss']:.4f} | {r['best']:.4f} | "
            f"{conv} | {r['spread']:.3f} | {r['steps_per_s']:.1f} | {r['per_1k']:.3f} |"
        )

    lines += ["", "⚠ marks a model whose loss never fell by 5% of where it started."]

    lines += ["", "## Best on each question", ""]
    titles = {
        "loss": "lowest final loss",
        "converged_at": "fastest to 90% of its own improvement",
        "spread": "steadiest at the end",
        "speed": "most steps per second",
        "per_1k": "most improvement per 1k parameters",
    }
    for key, title in titles.items():
        winners = [n for n, pos in sorted(ranks[key].items(), key=lambda kv: kv[1])[:3]]
        lines.append(f"- **{title}** — " + ", ".join(f"`{w}`" for w in winners))

    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
