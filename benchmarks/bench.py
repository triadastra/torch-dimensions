"""Phase 10 benchmarks: measure the known risks before claiming anything.

Run everything and rewrite the results table::

    python benchmarks/bench.py --out BENCHMARKS.md

Or one suite, printed::

    python benchmarks/bench.py --only permute --device mps

Three rules this file exists to keep. **Nothing here asserts a speed** — the
suites measure, and the numbers land in BENCHMARKS.md with the hardware
attached; a performance claim without a machine name is a rumour. **Every
suite states what would change a decision** (the `question` field), because a
benchmark nobody can act on is a slow way to produce a number. And **the timer
synchronizes**: on MPS and CUDA the dispatch returns long before the work
does, so an unsynchronized loop measures the queue, not the model.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

import torch_dimensions as td

# -- harness -----------------------------------------------------------------


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def pick_device(requested: str | None = None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def timed(fn: Callable[[], Any], device: str, *, repeat: int = 20, warmup: int = 3) -> float:
    """Median wall-clock milliseconds per call, device-synchronized."""
    for _ in range(warmup):
        fn()
    sync(device)
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        sync(device)
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def peak_memory_mb(device: str) -> float | str:
    """Peak *allocation* since the last reset, where torch actually tracks one.

    Only CUDA does. The first version of this reported
    ``torch.mps.driver_allocated_memory()`` on MPS, which is the driver's total
    allocation for the process — it read 18 GB for a 25k-parameter model and
    would have shipped as a memory column in a published table. A number that
    is not the thing its column header claims is worse than a blank.
    """
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 2**20
    return "n/a"


def reset_memory(device: str) -> None:
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    elif device == "mps":
        torch.mps.empty_cache()


@dataclass
class Suite:
    name: str
    question: str
    """What decision these numbers would change. Printed with the table."""
    columns: tuple[str, ...]
    rows: list[list[Any]] = field(default_factory=list)
    finding: str = ""
    """What the numbers turned out to say.

    Written by hand after reading a run, and it lives here rather than in
    BENCHMARKS.md because regenerating that file overwrites it — an
    interpretation that a re-run silently deletes is one nobody will re-write.
    Whoever re-runs on new hardware is expected to re-read this line and edit
    it if the machine disagrees.
    """

    def markdown(self) -> str:
        head = "| " + " | ".join(self.columns) + " |"
        rule = "|" + "|".join("---" for _ in self.columns) + "|"
        body = "\n".join(
            "| " + " | ".join(f"{c:.3g}" if isinstance(c, float) else str(c) for c in row) + " |"
            for row in self.rows
        )
        found = f"\n**Finding.** {self.finding}\n" if self.finding else ""
        return f"### {self.name}\n\n{self.question}\n\n{head}\n{rule}\n{body}\n{found}"


# -- shared builders ---------------------------------------------------------


def cube(rank: int, cells: int, time_axis: bool = True) -> td.Lattice:
    """A rank-`rank` lattice of roughly `cells` cells, as square as it divides."""
    side = max(2, round(cells ** (1 / rank)))
    return td.Lattice(
        shape=(side,) * rank, names=tuple(f"ax{i}" for i in range(rank)), time=time_axis
    )


def inputs(lat: td.Lattice, d_model: int, batch: int, seq: int, device: str) -> torch.Tensor:
    lead = (batch, seq) if lat.time else (batch,)
    return torch.randn(*lead, *lat.shape, d_model, device=device)


# The timed callables bind their arguments here rather than closing over loop
# variables: a lambda inside the sweep reads whatever the loop variable holds
# when it *runs*, which for a benchmark means timing the wrong configuration.


def forward_of(model: torch.nn.Module, x: torch.Tensor) -> Callable[[], Any]:
    return lambda: model(x)


def step_of(model: torch.nn.Module, x: torch.Tensor) -> Callable[[], None]:
    def step() -> None:
        model.zero_grad(set_to_none=True)
        model(x).pow(2).mean().backward()

    return step


def fold_of(lat: td.Lattice, x: torch.Tensor) -> Callable[[], None]:
    axes = list(range(lat.n_axes))

    def fold() -> None:
        for axis in axes:
            seq, restore = lat.to_sequence(x, axis)
            lat.from_sequence(seq, restore)

    return fold


# -- suites ------------------------------------------------------------------


def bench_permute(device: str, quick: bool = False) -> Suite:
    """How much of a step is the fold/unfold bookkeeping rather than the mixer?

    Two `.contiguous()` per layer means a 12-layer model does ~24 full-tensor
    copies. If that is a large fraction of the step, a fused fold is the first
    justified kernel in this library — and if it is not, writing one is
    self-indulgence.
    """
    s = Suite(
        "Fold overhead",
        "**Decides Track C3.** A fused, permute-avoiding fold is only worth writing if the "
        "fold column is a large fraction of the step column.",
        ("rank", "d_model", "cells", "fold+unfold ms", "LSTM step ms", "fold %"),
        finding=(
            "The fold is 10-30% of a step at rank >= 2 and small `d_model`, and its share "
            "*falls* as `d_model` grows because the mixer's work grows faster than the copy's. "
            "So a fused fold is worth roughly a fifth of a step in the small-model corner and "
            "almost nothing in the large-model corner — real, but not the first thing to "
            "optimize. Rank 1 is the odd row: one 4,096-long sequential sweep dominates so "
            "thoroughly that the copy disappears into it."
        ),
    )
    d_models = [32, 128] if quick else [32, 64, 128, 256, 512]
    ranks = [1, 2] if quick else [1, 2, 3, 4]
    for rank in ranks:
        lat = cube(rank, 4096)
        for d_model in d_models:
            x = inputs(lat, d_model, 4, 8, device)
            model = td.LSTM(d_model, lat.n_axes, lat).to(device)
            fold_ms = timed(fold_of(lat, x), device, repeat=10)
            step_ms = timed(forward_of(model, x), device, repeat=10)
            s.rows.append([rank, d_model, lat.n_cells, fold_ms, step_ms, 100 * fold_ms / step_ms])
    return s


def bench_families(device: str, quick: bool = False) -> Suite:
    """Scan vs per-line kernels vs factorized (CaFA) at matched size.

    The kernel family exists because materializing one attention matrix per
    lattice line runs out of memory at rank 4. That is the claim; this is where
    it either shows up in the numbers or does not.
    """
    s = Suite(
        "Composition families",
        "**Decides which method to reach for**, at the small-lattice size most users start "
        "at. Per-line cost grows with the number of lines; factorized cost does not — so the "
        "ordering here should reverse somewhere, and the next table finds where.",
        ("rank", "cells", "method", "params", "fwd ms", "fwd+bwd ms", "peak MB"),
        finding=(
            "At ~1,300 cells the factorized path is *slower* than per-line attention, not "
            "faster: the lattice is small enough that per-line scores fit comfortably, and "
            "CaFA pays for pooling and per-line kernel expansion on top. That is the honest "
            "result at this size and it is not the claim the family exists for — see the next "
            "table, which grows the axis until the crossover shows up."
        ),
    )
    ranks = [2, 3] if quick else [2, 3, 4]
    d_model = 32
    for rank in ranks:
        lat = cube(rank, 1296)
        x = inputs(lat, d_model, 2, 4, device)
        methods = [
            ("axial_scan", dict(nd_method=td.axial_scan)),
            ("axial_attention", dict(nd_method=td.axial_attention)),
            ("cafa", dict(nd_method=td.cafa)),
        ]
        for name, kw in methods:
            reset_memory(device)
            model = td.LSTM(d_model, lat.n_axes, lat, **kw).to(device)
            n_params = sum(p.numel() for p in model.parameters())
            fwd = timed(forward_of(model, x), device, repeat=8)
            both = timed(step_of(model, x), device, repeat=8)
            s.rows.append([rank, lat.n_cells, name, n_params, fwd, both, peak_memory_mb(device)])
    return s


def bench_kernel_scaling(device: str, quick: bool = False) -> Suite:
    """Grow the axis until per-line attention loses to the factorized path.

    This is the kernel family's whole reason to exist, so it gets its own
    table. Per-line materializes one ``A x A`` score matrix per lattice line —
    ``cells/A`` of them, so ``O(cells · A)``. Factorized materializes one per
    *axis*: ``O(A²)`` plus a contraction. The ratio should therefore grow
    linearly with the axis length, and "should" is what a benchmark is for.
    """
    s = Suite(
        "Where factorization starts winning",
        "**The claim the kernel family exists for.** Per-line attention costs O(cells · A); "
        "factorized costs O(A² + cells). Growing the axis at fixed rank must make the ratio "
        "grow, or the family is not earning its complexity.",
        ("rank", "shape", "cells", "axial_attention ms", "cafa ms", "per-line / cafa"),
        finding=(
            "The ratio climbs with axis length as predicted — and the crossover sits far "
            "further out than the library's own documentation implied. Per-line attention is "
            "**faster** everywhere below roughly 50 cells per axis, by 2-3x on small "
            "lattices; the two meet around 48³ and factorization only pulls ahead at 64³ "
            "(262k cells). So the honest rule is: use `td.axial_attention` for anything "
            "grid-shaped and modest — CIFAR-sized images, small volumes — and reach for "
            "`td.cafa` at image-or-volume scale, where the ratio keeps growing and per-line "
            "eventually cannot allocate at all. The family earns its complexity at the top "
            "end, not in the middle, and this table replaces the guess that it earned it "
            "everywhere."
        ),
    )
    d_model = 32
    for rank in [2] if quick else [2, 3]:
        sides = [8, 16, 32, 64, 96] if rank == 2 else [8, 16, 32, 48, 64]
        if quick:
            sides = sides[:2]
        for side in sides:
            lat = td.Lattice(
                shape=(side,) * rank, names=tuple(f"ax{i}" for i in range(rank)), time=True
            )
            # The big lattices are activation-bound: 64³ cells at batch 2 and 4
            # timesteps is a gigabyte of features before any score matrix exists,
            # so the batch shrinks as the lattice grows. Time per step is what is
            # being compared, and both methods see the same batch.
            batch, seq = (2, 4) if lat.n_cells <= 20_000 else (1, 2)
            x = inputs(lat, d_model, batch, seq, device)
            times = []
            for method in (td.axial_attention, td.cafa):
                model = td.LSTM(d_model, lat.n_axes, lat, nd_method=method).to(device)
                times.append(timed(forward_of(model, x), device, repeat=4, warmup=1))
                del model
            del x
            reset_memory(device)
            s.rows.append(
                [
                    rank,
                    "×".join(str(side) for _ in range(rank)),
                    lat.n_cells,
                    times[0],
                    times[1],
                    times[0] / times[1],
                ]
            )
    return s


def bench_rank_scaling(device: str, quick: bool = False) -> Suite:
    """Does cost grow with rank, or with cells? The design says cells."""
    s = Suite(
        "Rank at fixed cell count",
        "**Tests a design claim.** Cost should track cells, not rank; a rank term would mean "
        "the abstraction charges for dimensions it promised were free.",
        ("rank", "shape", "cells", "layers", "fwd ms", "ms / (cell·layer) ×10⁻³"),
        finding=(
            "Cost tracks the **length of the swept axis**, not the cell count and not the "
            "rank. At a fixed ~4,096 cells, rank 1 (one 4,096-step sweep) costs ~95x rank 4 "
            "(sweeps of 8). The design said cost should track cells; the measurement says the "
            "sequential mixer's launch depth dominates everything else, so spreading the same "
            "cells over more axes is a large speedup rather than a cost. Rank does not appear "
            "in the cost model - axis *length* does."
        ),
    )
    d_model = 64
    ranks = [1, 2, 3] if quick else [1, 2, 3, 4, 5]
    for rank in ranks:
        side = round(4096 ** (1 / rank))
        lat = td.Lattice(shape=(side,) * rank, names=tuple(f"ax{i}" for i in range(rank)))
        x = inputs(lat, d_model, 2, 0, device)
        n_layers = 4
        model = td.LSTM(d_model, n_layers, lat).to(device)
        ms = timed(forward_of(model, x), device, repeat=8)
        s.rows.append(
            [
                rank,
                "×".join(str(v) for v in lat.shape),
                lat.n_cells,
                n_layers,
                ms,
                1e3 * ms / (lat.n_cells * n_layers),
            ]
        )
    return s


def bench_chunk(device: str, quick: bool = False) -> Suite:
    """Where does chunked folding start to pay?"""
    s = Suite(
        "Chunked fold",
        "**Feeds the Phase 3 auto-tune item.** `chunk=` splits the folded batch; it trades "
        "peak memory for launch count. The crossover is what auto-tuning would need to know.",
        ("cells", "d_model", "chunk", "fwd ms", "peak MB"),
        finding=(
            "Chunking never won here. `chunk=64` costs 3x by launching 3x the kernels; the "
            "largest chunk is within noise of no chunking at all. On this hardware `chunk=` "
            "is a memory-pressure valve, not a speed knob, and the Phase 3 auto-tuner should "
            "default it off and reach for it only when an allocation actually fails."
        ),
    )
    lat = cube(2, 4096, time_axis=False)
    for d_model in [64] if quick else [64, 256]:
        x = inputs(lat, d_model, 8, 0, device)
        for chunk in [None, 64, 256, 1024]:
            reset_memory(device)
            model = td.LSTM(d_model, 4, lat, chunk=chunk).to(device)
            ms = timed(forward_of(model, x), device, repeat=8)
            s.rows.append([lat.n_cells, d_model, str(chunk), ms, peak_memory_mb(device)])
    return s


def bench_mixers(device: str, quick: bool = False) -> Suite:
    """Per-family step cost on the same lattice — the honest 'what does this cost me' table."""
    s = Suite(
        "Mixer families",
        "**Sets expectations, and the baseline every fast path must beat.** The portable Mamba "
        "scan is a python loop over the swept axis: correct, honest, and the row to watch when "
        "Track C2's fused path lands.",
        ("model", "rank", "cells", "params", "fwd ms", "fwd+bwd ms"),
        finding=(
            "The portable Mamba scan is ~5-8x the cost of the kernel-based SSMs at equal "
            "width, which is exactly what a python-level sequential loop should cost and is "
            "the number Track C2's fused path has to beat. S4 (DPLR) is within ~10% of S4D "
            "(diagonal) despite the Cauchy resolvent and Woodbury correction: the full kernel "
            "is not the expensive one, the sequential scan is."
        ),
    )
    lat = cube(2, 256)
    d_model = 64
    x = inputs(lat, d_model, 2, 16, device)
    families = [("LSTM", td.LSTM), ("GRU", td.GRU), ("S4D", td.S4D), ("S4", td.S4)]
    if not quick:
        families.append(("Mamba", td.Mamba))
    for name, cls in families:
        model = cls(d_model, 4, lat).to(device)
        fwd = timed(forward_of(model, x), device, repeat=6)
        both = timed(step_of(model, x), device, repeat=6)
        s.rows.append(
            [name, lat.rank, lat.n_cells, sum(p.numel() for p in model.parameters()), fwd, both]
        )
    return s


def bench_compile(device: str, quick: bool = False) -> Suite:
    """torch.compile on/off, per family. Correctness is the conformance suite's
    job (`check_compile=True`); this is only the speed half."""
    s = Suite(
        "torch.compile",
        "**Decides whether to document compile as recommended.** Compile time is paid once; "
        "the ratio is what a user gets for it.",
        ("model", "eager ms", "compiled ms", "speedup"),
        finding=(
            "`torch.compile` is a *slowdown* on MPS at this size (~0.8x) - the graphs are "
            "small and the launch overhead it removes was not the bottleneck. It is therefore "
            "not recommended by default; the conformance suite still checks that it is "
            "numerically correct when a user does turn it on. Re-measure on CUDA before "
            "repeating this sentence about any other machine."
        ),
    )
    lat = cube(2, 256)
    d_model = 64
    x = inputs(lat, d_model, 2, 8, device)
    for name, cls in [("LSTM", td.LSTM), ("S4D", td.S4D)][: 1 if quick else 2]:
        model = cls(d_model, 3, lat).to(device)
        eager = timed(forward_of(model, x), device, repeat=6)
        try:
            comp = timed(forward_of(torch.compile(model), x), device, repeat=6, warmup=2)
        except Exception as e:  # noqa: BLE001 — a compile failure is a result
            s.rows.append([name, eager, f"failed: {type(e).__name__}", "—"])
            continue
        s.rows.append([name, eager, comp, f"{eager / comp:.2f}×"])
    return s


SUITES: dict[str, Callable[[str, bool], Suite]] = {
    "permute": bench_permute,
    "families": bench_families,
    "kernels": bench_kernel_scaling,
    "rank": bench_rank_scaling,
    "chunk": bench_chunk,
    "mixers": bench_mixers,
    "compile": bench_compile,
}


# -- reporting ---------------------------------------------------------------


def machine(device: str) -> dict[str, str]:
    name = device
    if device == "cuda":
        name = torch.cuda.get_device_name(0)
    elif device == "mps":
        name = f"Apple Silicon (MPS), {platform.machine()}"
    return {
        "device": device,
        "accelerator": name,
        "platform": f"{platform.system()} {platform.release()}",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_dimensions": td.__version__,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default=None, help="cpu | cuda | mps (default: best available)")
    ap.add_argument("--only", nargs="*", choices=sorted(SUITES), help="run a subset")
    ap.add_argument("--quick", action="store_true", help="a smaller sweep, for smoke runs")
    ap.add_argument("--out", default=None, help="write a markdown report here")
    ap.add_argument("--json", default=None, help="also write raw rows here")
    args = ap.parse_args()

    device = pick_device(args.device)
    torch.manual_seed(0)
    chosen = args.only or list(SUITES)
    info = machine(device)
    print(f"# benchmarks on {info['accelerator']} · torch {info['torch']}\n")

    suites = []
    for key in chosen:
        print(f"running {key} …", flush=True)
        suite = SUITES[key](device, args.quick)
        suites.append(suite)
        print(suite.markdown())

    if args.out:
        header = [
            "# Benchmarks",
            "",
            "Measured, not estimated; regenerate with `python benchmarks/bench.py --out "
            "BENCHMARKS.md`. Every number below comes from one machine, named here, with "
            "seeds fixed — a benchmark whose hardware is unstated is a rumour.",
            "",
            "| | |",
            "|---|---|",
            *(f"| {k} | {v} |" for k, v in info.items()),
            "",
            "Timings are the median of repeated runs after warmup, device-synchronized. "
            "Peak memory reads `n/a` on everything but CUDA, because CUDA is the only "
            "backend where torch tracks an allocation high-water mark — see "
            "`peak_memory_mb` for why a plausible-looking substitute was removed.",
            "",
        ]
        body = "\n".join(s.markdown() for s in suites)
        with open(args.out, "w") as fh:
            fh.write("\n".join(header) + "\n" + body)
        print(f"wrote {args.out}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(
                {
                    "machine": info,
                    "suites": [
                        {"name": s.name, "columns": list(s.columns), "rows": s.rows} for s in suites
                    ],
                },
                fh,
                indent=2,
                default=str,
            )


if __name__ == "__main__":
    main()
