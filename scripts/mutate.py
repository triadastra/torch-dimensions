"""Mutation testing: break the code on purpose, check the suite notices.

    python scripts/mutate.py            # run the catalog
    python scripts/mutate.py --list     # show it without running anything

DEBUG.md §B ranks this the highest-yield technique in the project, and it has
been a *manual* discipline the whole time — which means it runs when somebody
remembers. This makes it a machine's job.

Each entry names a real invariant, a one-line edit that violates it, and the
tests that must fail as a result. A mutation that survives is a hole in the
suite: the code can be wrong in that specific way and nothing says so. That is
the finding, and it is reported as a failure of *the tests*, not of the code.

Every mutation here corresponds to a bug that was either found in this project
or deliberately designed against — see the `why` field. Restoring the file is
guaranteed by a finally-block; the script also refuses to start on a dirty
working tree, because a crash mid-mutation must never be confusable with your
own uncommitted work.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "torch_dimensions"


@dataclass
class Mutation:
    name: str
    path: Path
    old: str
    new: str
    tests: str
    """The `-k` expression, or a test path, that must fail once mutated."""
    why: str

    def describe(self) -> str:
        return f"{self.name:34s} {self.path.relative_to(ROOT)}  ->  {self.tests}"


CATALOG = [
    Mutation(
        name="absolute cancellation epsilon",
        path=SRC / "compose" / "kernel.py",
        old="den = torch.where(den.abs() <= _REL * den_abs, torch.ones_like(den), den)",
        new="den = torch.where(den.abs() <= 1e-12, torch.ones_like(den), den)",
        tests="tests/test_kernel.py -k cancel or explode",
        why="DEBUG.md #12: an absolute epsilon has no idea what scale the kernel works at; "
        "float32 blew up 7,000x under one.",
    ),
    Mutation(
        name="mask only on entry, not per layer",
        path=SRC / "compose" / "kernel.py",
        old="    if valid is not None:\n        x = x * valid",
        new="    if valid is not None and False:\n        x = x * valid",
        tests="tests/test_kernel.py -k absent or influence",
        why="Zeroing once at entry is sufficient only at rank 1; a contraction leaves absent "
        "cells holding legitimate scratch that the next axis then counts.",
    ),
    Mutation(
        name="direction flips per layer, not per cycle",
        path=SRC / "plan.py",
        old="Step(axes[i % n], axes[i % n] in bidi and (i // n) % 2 == 1) for i in range(n_layers)",
        new="Step(axes[i % n], axes[i % n] in bidi and i % 2 == 1) for i in range(n_layers)",
        tests="tests/test_plan.py -k alias or direction or cycle",
        why="DEBUG.md #4: with an even axis count the two periods phase-lock and every axis is "
        "silently pinned one way. This bug exists in published research code.",
    ),
    Mutation(
        name="lattice mask returns a view",
        path=SRC / "lattice.py",
        old="return base.reshape(*lead, *self.shape, 1).to(dtype, copy=True)",
        new="return base.reshape(*lead, *self.shape, 1).to(dtype)",
        tests="tests/test_lattice.py -k view or copy or alias",
        why="DEBUG.md #13: a caller writing into 'their' mask reached into the lattice.",
    ),
    Mutation(
        name="flat_idx indexes on the wrong device",
        path=SRC / "lattice.py",
        old="out[..., self.flat_idx.to(x.device), :] = x",
        new="out[..., self.flat_idx, :] = x",
        tests="tests/test_device.py -k device_mismatch or gather_scatter",
        why="DEBUG.md #18: torch tolerates CPU indices into a device tensor but not the "
        "reverse, so one direction raised and the other did not.",
    ),
    Mutation(
        name="conformance skips count as passes",
        path=SRC / "testing.py",
        old='rep.results.append(Result(name, "skip", str(s)))',
        new='rep.results.append(Result(name, "pass", str(s)))',
        tests="tests/test_conformance.py -k skip",
        why="A skipped check that reads as a pass is how a whole family goes unverified while "
        "the report looks green.",
    ),
    Mutation(
        name="plan/n_layers disagreement goes silent",
        path=SRC / "models" / "base.py",
        old="if n_layers != 1 and n_layers != len(plan):",
        new="if False:",
        tests="tests/test_scan.py tests/test_conformance.py -k n_layers or plan_wins",
        why="DEBUG.md #10: a model quietly shallower than requested.",
    ),
]


def run(mutation: Mutation, verbose: bool) -> bool:
    """Apply, run the named tests, restore. True when the mutation was caught."""
    source = mutation.path.read_text()
    if mutation.old not in source:
        print("  STALE: the code no longer contains the mutated line — update the catalog")
        return False
    try:
        mutation.path.write_text(source.replace(mutation.old, mutation.new, 1))
        cmd = [sys.executable, "-m", "pytest", "-x", "-q", *mutation.tests.split()]
        result = subprocess.run(cmd, cwd=ROOT, capture_output=not verbose, text=True)
    finally:
        mutation.path.write_text(source)
    return result.returncode != 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--only", help="substring of a mutation name")
    args = ap.parse_args()

    if args.list:
        for m in CATALOG:
            print(m.describe())
        return 0

    dirty = subprocess.run(
        ["git", "status", "--porcelain", "src"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    if dirty:
        print("refusing to run with uncommitted changes under src/:\n" + dirty)
        print("a crash mid-mutation must not be confusable with your own edits")
        return 2

    chosen = [m for m in CATALOG if not args.only or args.only in m.name]
    survived = []
    for m in chosen:
        print(f"\n== {m.name}\n   {m.why}")
        caught = run(m, args.verbose)
        print(f"   {'caught' if caught else 'SURVIVED — the suite does not notice this'}")
        if not caught:
            survived.append(m)

    print(f"\n{len(chosen) - len(survived)}/{len(chosen)} mutations caught")
    if survived:
        print("\nsurviving mutations — each is a hole in the test suite, not in the code:")
        for m in survived:
            print(f"  - {m.name}: {m.why}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
