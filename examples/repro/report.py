"""Turn the results ledger into RESULTS.md.

    python -m examples.repro.report --out RESULTS.md

The ledger (``results.json``) is append-only and holds every run including the
ones that disagreed; this renders the latest run per configuration into a
table, and says how many runs a configuration has when it has more than one.
Nothing here computes a number — it only formats numbers that a training run
already produced, which is why the two live in separate files.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

HEADER = """# Results

Reproductions run with this library and nothing else — no model code outside
`torch_dimensions`, no second framework, no vendored datasets. Every row
carries its epoch budget, seed, wall-clock and hardware, because a result
without them is an anecdote.

**These are laptop-scale runs.** The papers referenced below train far longer
on far larger machines. What is being tested here is whether the *construction*
reproduces — whether the portable kernels and the N-D machinery learn the
tasks they are supposed to learn — not whether one Mac can match a cluster.

Regenerate with:

```bash
bash examples/repro/run_all.sh          # runs everything, appends to results.json
python -m examples.repro.report --out RESULTS.md
```
"""

# Each section: (title, task prefixes, what the section is, what it turned out
# to say). The last field is written by hand after reading a run and lives here
# rather than in RESULTS.md because regenerating overwrites that file — the same
# reason BENCHMARKS.md keeps its findings in the generator. An interpretation a
# re-run silently deletes is one nobody rewrites.
SECTIONS = [
    (
        "Sequence tasks — the mixer without the lattice",
        ("sMNIST", "psMNIST"),
        "A sequence is a lattice with no spatial axes, so these isolate the portable S4D "
        "kernel from the N-D machinery entirely. Published reference points: the S4/S4D "
        "papers report ~99.6% on sMNIST and ~98.5% on psMNIST, with longer schedules.",
        "**Reproduced.** Both land inside one point of the published numbers, from a config "
        "and one command, on a laptop, in 21 minutes each. The portable S4D kernel — pure "
        "torch, no CUDA — learns what the paper's does.",
    ),
    (
        "2-D lattices — the N-D machinery on images",
        ("mnist (2-D", "cifar10 (2-D"),
        "The image is a lattice; rows and columns are swept with the paired schedule the "
        "Mamba-ND paper describes. No pixel is flattened into a sequence.",
        "",
    ),
    (
        "Sparse lattices — no published baseline exists",
        ("air quality",),
        "Beijing air quality: 12 stations x 6 pollutants, hourly, with a fraction of cells "
        "made absent. Arms differ in exactly one thing each and are scored on present cells "
        "only. These rows *are* the baseline.",
        "",
    ),
]


def normalize(run: dict) -> dict:
    """Bring a ledger row up to the current schema, in memory only.

    The ledger is append-only — including across changes to the recording
    code, which is how rows written before the metric was given an explicit
    name still sit in it. Rewriting them to match would be the one edit this
    file is not allowed to make, so the reader adapts instead. A row whose
    number is called `test_acc` and nothing else *is* an accuracy; that is the
    only assumption made here, and it is made once.
    """
    if "metric" not in run and "test_acc" in run:
        run = {**run, "metric": run["test_acc"], "metric_name": "test_acc"}
    return run


def latest_per_config(ledger: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for run in map(normalize, ledger):
        c = run["config"]
        groups[(c["task"], c["model"], c["seed"], c["epochs"])].append(run)
    out = []
    for runs in groups.values():
        latest = runs[-1]
        latest["_n_runs"] = len(runs)
        out.append(latest)
    return out


def table(runs: list[dict]) -> str:
    head = (
        "| task | configuration | params | epochs | seed | result | wall clock | hardware |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for run in sorted(runs, key=lambda r: (r["config"]["task"], r["config"]["model"])):
        c = run["config"]
        value = (
            f"{run['metric'] * 100:.2f}%"
            if run["metric_name"] == "test_acc"
            else f"{run['metric']:.4f}"
        )
        label = "test accuracy" if run["metric_name"] == "test_acc" else run["metric_name"]
        note = f" ({run['_n_runs']} runs)" if run.get("_n_runs", 1) > 1 else ""
        capped = " ⚠ capped train set" if c.get("limit_train") else ""
        rows.append(
            f"| {c['task']}{capped} | {c['model']} | {c.get('n_params', 0):,} | "
            f"{c['epochs']} | {c['seed']} | **{value}** {label}{note} | "
            f"{run['seconds'] / 60:.1f} min | {run['machine']['accelerator']} |"
        )
    return head + "\n".join(rows) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger", default=str(Path(__file__).parent / "results.json"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ledger = json.loads(Path(args.ledger).read_text())
    runs = latest_per_config(ledger)

    parts = [HEADER]
    placed: set[int] = set()
    for title, prefixes, blurb, finding in SECTIONS:
        chosen = [
            r
            for i, r in enumerate(runs)
            if i not in placed and any(r["config"]["task"].startswith(p) for p in prefixes)
        ]
        if not chosen:
            continue
        placed.update(runs.index(r) for r in chosen)
        found = f"\n**Finding.** {finding}\n" if finding else ""
        parts.append(f"\n## {title}\n\n{blurb}\n\n{table(chosen)}{found}")

    rest = [r for i, r in enumerate(runs) if i not in placed]
    if rest:
        parts.append(f"\n## Other runs\n\n{table(rest)}")

    machines = {r["machine"]["accelerator"] for r in runs}
    versions = {r["machine"]["torch_dimensions"] for r in runs}
    parts.append(
        f"\n---\n\nAll rows above: torch-dimensions {', '.join(sorted(versions))} on "
        f"{', '.join(sorted(machines))}. Raw history, including runs superseded by a "
        f"later one, is in [`examples/repro/results.json`](examples/repro/results.json).\n"
    )

    text = "".join(parts)
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out} ({len(runs)} configurations)")
    else:
        print(text)


if __name__ == "__main__":
    main()
