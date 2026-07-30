# torch-dimensions — Build Plan

Companion to [DESIGN.md](DESIGN.md). That document says *what* the library is; this one says *in what order it gets built and how we know each step worked*.

**Sequencing principle:** each phase ends with something runnable and tested, and the riskiest unknowns get resolved before anything is built on top of them. Concretely that means the tensor bookkeeping (`Lattice`) is proven before any model exists, the test suite exists before the hard math, and the config surface is written **last** — a config schema authored before the block signatures settle just freezes a guess.

Sizes are relative (S ≈ half a day, M ≈ 1–2 days, L ≈ 3–5 days) for one person already fluent in this material.

---

## Resolved: the name

**`torch-dimensions`**, importing as `torch_dimensions`, conventionally aliased **`td`**:

```python
import torch_dimensions as td
```

Settled before Phase 0, since the name fixes the import path, the PyPI reservation, and every doc example — mechanical to change now, painful after release. Every example in the docs uses the `td` alias, and the short form is what makes the full package name affordable: `td.Lattice(...)` reads no worse than a terse package name would, without giving up a descriptive one on PyPI.

---

## Phase 0 — Skeleton and packaging  ·  S  ·  no deps

Make the repo installable and enforceable before any logic exists.

- `pyproject.toml` (hatchling), `src/` layout so tests import the installed package rather than the working tree.
- Dependencies: `torch>=2.4` required and **nothing else**. Optional extras: `[mamba]`, `[s4]`, `[fla]`, `[dev]`, `[all]`.
- `ruff` (lint + format), `pytest`, `pytest-cov`. Type hints throughout; `mypy` advisory, not gating.
- CI: CPU matrix on Python 3.10–3.12. GPU tests marked `@pytest.mark.gpu` and skipped.
- `LICENSE` (Apache-2.0 — matches PyTorch, permits the adapter story), `README.md`, `CONTRIBUTING.md`.

**Acceptance:** `pip install -e ".[dev]"` then `pytest` passes on a machine with no CUDA and no optional deps installed. This is a real constraint, not ceremony — it is the promise that a CPU-only user can install and import the library.

---

## Phase 1 — `Lattice`  ·  M  ·  needs Phase 0

The foundation. Pure tensor bookkeeping, zero models, zero learnable parameters. Everything else depends on this being exactly right, which is why it comes first and gets tested harder than its line count suggests.

- `shape`, `names`, `valid`, `time`; axis name → position resolution.
- Permutation generation and inverse for "move axis `k` to sequence position".
- Flat indices, scatter-to-dense, gather-from-dense.
- Broadcastable validity mask, per-axis valid counts for masked pooling.

**Acceptance:**
- Permutation round-trip is the identity for every axis, ranks 1–5.
- `gather(scatter(x)) == x` exactly, dense and sparse.
- Inverse permutation is verified against `torch.argsort`, not hand-derived twice.
- Property test: random shapes and random valid-masks, ranks 1–5.

**Why this first:** every downstream bug in an ND model looks like "the model is bad" and is actually an axis-order bug. Isolating this layer means those bugs are impossible later rather than merely unlikely.

---

## Phase 2 — `ScanPlan`  ·  S  ·  needs Phase 1

Pure data. No tensors, no modules.

- `.cyclic()`, `.paired()`, `.from_list()`; `__repr__`; to/from dict.
- Validation: every axis in the plan exists in the lattice; a warning when the plan does not visit every axis (legal, usually a mistake).

**Acceptance:** plans round-trip through serialization; `cyclic(axes, n)` visits each axis `⌈n/|axes|⌉` or `⌊n/|axes|⌋` times and alternates direction.

---

## Phase 3 — `AxialScan` + LSTM mixer  ·  M  ·  needs Phases 1–2

The first real block. **LSTM first, not Mamba** — `nn.LSTM` needs no optional deps, runs on CPU, and is a known-correct 1-D reference, so this phase tests the permute machinery in isolation from any kernel problem. Leading with Mamba would conflate "my axis logic is wrong" with "the Triton kernel is unhappy," which is the worst debugging position to be in.

- `AxialScan`: permute → fold others into batch → mixer → unfold → inverse permute, pre-norm residual per layer.
- Batch-fold chunking, library-owned and auto-tuned from device limits — never a user-facing constant.
- `LSTM`, `GRU` — one class each, 1-D without a lattice and N-D with one.

**Acceptance:** a single layer on a rank-1 lattice matches `nn.LSTM` **bit-for-bit**. That one test catches essentially every permutation and residual bug. Multi-layer stacks match only to floating-point tolerance, because the fold normalizes memory layout and torch's RNN kernels are layout-sensitive — expected, not a defect.

**Deliverable:** a working N-dimensional LSTM — prior art as MDRNN (Graves et al. 2007) and Grid-LSTM (Kalchbrenner et al. 2015), but with no maintained modern implementation. Shippable on its own.

---

## Phase 4 — Conformance suite  ·  M  ·  needs Phase 3

Built now, before the hard math, so every later block is validated by construction rather than retrofitted. This is the single highest-leverage phase in the plan: it is what keeps an (models × ranks × sparsity) matrix from rotting.

The seven checks are specified in [DESIGN.md §6](DESIGN.md). Packaged as `td.testing.check_block(...)` so downstream users can run it against their own mixers.

**Acceptance:** `LSTM` and `GRU` pass all seven. Suite is parametrized over rank 1–4 and dense/sparse.

---

## Phase 5 — Data: getting real data into lattice layout  ·  M  ·  needs Phase 4

Scoped by one distinction: **building a lattice from data is lattice construction, which this library already owns. Running a training loop is not.** Everything here is the former.

The gap is concrete. A user holding long-format rows — `(coord₀, coord₁, …, t, features…)` — must currently hand-write the coordinate-to-index mapping, infer the shape, build the valid mask, and scatter into `(B, T, *shape, H)`. That is precisely the code that silently produces a *mis-shuffled* lattice: it trains, it converges, the numbers are quietly wrong. Same bug class Phase 1 exists to eliminate, one layer up, currently unowned.

Four pieces, each usable alone:

- **`Lattice.from_coords(coords, names=...)`** — infer shape, valid mask, and categorical vocabularies from observed coordinates. Returns the lattice plus the mapping needed to place rows.
- **`LatticeWindow`** — windowing over time (`input_len`, `horizon`, `stride`, split boundaries). Pure index arithmetic, no I/O, independently testable.
- **`LatticeSource`** — a small protocol, *not* a base class. This is where customization comes from: a memory-mapped array, zarr, HDF5, or a database all batch correctly as long as they satisfy it. Ship the protocol plus two reference implementations (in-memory tensor, long-format table).
- **`collate_lattice`** — stacks windows and keeps the `Lattice` *out* of the batch, since it is static metadata rather than per-sample data.

Customization comes from composition, never from a god-class with forty constructor arguments.

**Deliberately absent:** dataset downloads, normalization policy (a hook, shipping nothing), augmentation, splitting strategy, trainers, Lightning integration.

**Placement rationale:** after Phase 4 so the data API is designed against a real, already-validated consumer rather than a guess — the same reasoning that puts config last. This is also the first phase that makes an end-to-end example runnable: load → model → loss → backward.

**Acceptance:** a long-format table with deliberately missing combinations round-trips to a lattice and back with values landing in the right cells — verified against an independently constructed dense reference, not just a shape check. Windows tile the time axis with no gaps or overlaps beyond the requested stride. A user-supplied source satisfying only the protocol batches correctly.

---

## Phase 6 — `AxialKernel` and axial attention  ·  L  ·  needs Phase 4

The hardest math in the project. Three strictly ordered steps, each independently verifiable:

1. **Dense contraction.** Per-axis kernel, sequential contraction. Verified against the explicitly materialized Kronecker product on a small lattice — a test only possible while the lattice is small, which is exactly why it happens here.
2. **Sparse renormalization.** Masked-mean pooling and per-line renormalization by valid-key softmax mass, for arbitrary N.
3. **Factorized axial attention.** Per-axis Q/K from the pooled 1-D function, one shared V.

Generalizing rank-locked contraction tables to arbitrary N is the largest single piece of work in the library. Use permute+matmul, not generated einsum strings.

**Acceptance:** Kronecker identity holds to `1e-5` on dense; mask-invariance holds exactly on sparse; `AxialTransformer` and the factorized variant pass the full suite. Memory at rank 4 stays quadratic in axial size — the property that makes the factorized path viable where a per-line implementation exhausts memory.

---

## Phase 7 — SSM adapters  ·  L  ·  needs Phase 4  ·  **needs a GPU (Mamba only)**

- Thin adapters over `mamba-ssm` (Mamba-2), `state-spaces/s4`, and Mamba-3.
- Defensive imports: a missing optional dep unregisters that block and leaves everything else importable. A missing real implementation **fails loudly and never silently substitutes a stub**.
- `MambaND`, `S4ND`.

**S4 portability, established empirically (2026-07-30, MPS):** the full
`S4Block` and `S4D` both run on MPS — forward, backward, and CPU-parity to
~1.1e-6 — once three things are handled in the vendored copy, all of which a
derivative under Apache-2.0 may do:

1. Drop the `pytorch_lightning` import (used for one logging decorator).
2. Inline `DropoutNd` in `s4d.py` (it imports a repo-internal path).
3. Fix a **latent upstream numerical bug** in `SSMKernelDPLR._omega`: the
   bilinear transform `z = 2(1-ω)/(1+ω)` divides by a quantity that is
   mathematically zero at the Nyquist point (ω = -1). CPU/CUDA dodge the pole
   by ~1e-7 of rounding error in the complex pow; MPS lands on -1 exactly and
   the whole kernel goes NaN. Guard the denominator (the full expression
   cancels the pole analytically, so an eps nudge reproduces what rounding
   already does elsewhere). This must be fixed in the vendored copy regardless
   of backend — code that works only by accident of rounding is not portable.

So the GPU-only remainder of this phase is `mamba-ssm` (CUDA-only extension);
the S4 family is coverable by the device suite (tests/test_device.py) on MPS.

**Acceptance:** both pass the suite on GPU; `MambaND` on a rank-1 lattice matches a bare `Mamba2` block.

---

## Phase 8 — Registry, config, and save/load  ·  M  ·  needs Phases 6–7

Deliberately late. Now that every block signature has settled, the config schema describes reality instead of predicting it.

- `register()` / `build()` / `list_blocks()`.
- One dataclass schema per block; validation errors name the offending key and list valid ones.
- YAML loader.
- **`model.save(path)` / `td.load(path)`** — architecture *and* weights in one file, so a checkpoint reconstructs its own model without the user re-specifying the config. This belongs here rather than earlier: it is the registry plus the config schema, and it cannot exist before either.

Save/load is where a config system quietly rots, so two properties are non-negotiable. A checkpoint records the library version and refuses to load silently under an incompatible one. And a lattice's validity mask travels *with* the checkpoint — a model restored against a different sparsity pattern is a wrong model, not a warning.

**Acceptance:** every model round-trips config → model → config, and save → load → identical outputs on identical input, verified bitwise. Unknown config keys are a hard error, not a silent ignore.

---

## Phase 9 — Acceptance against a real workload  ·  M  ·  needs Phase 8

**The phase that decides whether the abstraction is right.** Reproduce a published result from one of the source papers — an S4ND or Mamba-ND benchmark number on a public dataset — using only torch-dimensions blocks and config.

If the numbers land, the generalization preserved semantics. If they do not, the library is wrong, and it is far better to learn that here than after release. If expressing the model requires reaching around the API into bespoke Python, the abstraction leaks and Phase 5 needs revisiting.

**Acceptance:** metrics match the paper within seed noise, and the model is expressible in config alone.

---

## Phase 10 — Benchmarks  ·  M  ·  needs Phase 9

Measure the known risk before making any performance claim: two `.contiguous()` calls per layer means a 12-layer ND model does ~24 full-tensor copies, which plausibly dominates the mixer at small `d_model`.

- Permute/copy overhead as a fraction of step time, swept over `d_model` and rank.
- Dense vs factorized memory at ranks 2–4.
- `torch.compile` on/off.

**Acceptance:** published numbers with hardware stated. If permute overhead is material, fusing it becomes the first v0.2 item — and *only then* is writing a kernel justified.

---

## Phase 11 — Docs and v0.1.0  ·  M  ·  needs Phase 10

- README: the unification table, 10-line quickstart, honest scope limits.
- "Adding a mixer" guide — the extension point is the product, so this page matters more than the API reference.
- TestPyPI, then PyPI. Tag `v0.1.0`.

**README must state plainly:** forward-only, no autoregressive stepping; ranks 1–4 tested; fused kernels come from upstream, not from us.

---

## Critical path

```
                            ┌──► 5 ──┐
0 ──► 1 ──► 2 ──► 3 ──► 4 ──┼──► 6 ──┼──► 8 ──► 9 ──► 10 ──► 11
                            └──► 7 ──┘
```

Phases 5, 6, and 7 are independent once the conformance suite exists. Phase 7 is the one to defer if GPU access is intermittent, since everything else is CPU-only. That is deliberate: **the library is useful and shippable without a single CUDA kernel.**

---

## Repo layout

```
torch-dimensions/
├── pyproject.toml                  hatchling; torch>=2.4 only; extras [mamba] [s4] [fla] [dev] [all]
├── README.md                       pitch, unification table, quickstart, scope limits
├── DESIGN.md                       architecture
├── PLAN.md                         this file
├── CONTRIBUTING.md                 how to add a mixer — the extension point is the product
├── CHANGELOG.md
├── LICENSE                         Apache-2.0
│
├── src/torch_dimensions/
│   ├── __init__.py                 public API only; defensive optional imports
│   ├── lattice.py                  Lattice: axes, names, valid mask, permute/scatter/gather
│   ├── plan.py                     ScanPlan + constructors
│   ├── registry.py                 register / build / list_blocks
│   ├── config.py                   dataclass schemas + YAML loader + save/load
│   ├── testing.py                  check_block() — the conformance suite, public
│   │
│   ├── data/                       BUILDS lattices; never trains them
│   │   ├── source.py               LatticeSource protocol + in-memory reference
│   │   ├── table.py                long-format rows -> Lattice + dense tensor
│   │   ├── window.py               LatticeWindow — time windowing, pure index math
│   │   └── collate.py              collate_lattice
│   │
│   ├── compose/
│   │   ├── scan.py                 AxialScan — sequential 1-D passes
│   │   └── kernel.py               AxialKernel — Kronecker contraction + sparse renorm
│   │
│   ├── mixers/                     ADAPTERS, not reimplementations
│   │   ├── base.py                 Mixer protocol: (M, A, H) -> (M, A, H)
│   │   ├── rnn.py                  LSTM, GRU        (torch.nn)
│   │   ├── attn.py                 self-attn, cross-attn, factorized axial kernel
│   │   └── ssm.py                  Mamba-2/3, S4, S5   [optional deps]
│   │
│   └── models/
│       ├── rnn.py                  LSTM, GRU  (1-D without a lattice, N-D with one)
│       ├── transformer_nd.py       AxialTransformer, factorized axial attention
│       └── ssm_nd.py               MambaND, S4ND
│
├── tests/
│   ├── test_lattice.py             permutation + scatter/gather round-trips, property-based
│   ├── test_plan.py
│   ├── test_conformance.py         parametrized: every block × rank 1–4 × dense/sparse
│   ├── test_kronecker.py           factorized == explicit ⊗
│   ├── test_equivalence.py         rank-1 == the underlying 1-D module
│   ├── test_data.py                coords -> lattice -> back, against a dense reference
│   ├── test_save_load.py           checkpoint reconstructs its own model, bitwise
│   └── gpu/                        @pytest.mark.gpu — SSM adapters
│
├── benchmarks/
│   ├── permute_overhead.py         the known risk, measured
│   └── memory_scaling.py           dense vs factorized, ranks 2–4
│
├── examples/
│   ├── 01_lattice_basics.py
│   ├── 02_sparse_lattice.py        the differentiator, demonstrated
│   └── 03_config_driven.py
│
└── .github/workflows/ci.yml        ruff + pytest, py3.10–3.12, CPU only
```

Three properties of this layout are load-bearing:

- **`mixers/` are adapters.** That directory is where the composition-layer decision either holds or quietly collapses into a reimplementation.
- **`testing.py` is public API**, not `tests/`. Users adding their own mixer get the same verification the library uses on itself.
- **`data/` builds lattices; it does not train them.** The dividing line is that constructing a lattice from real data is this library's own object being constructed, whereas optimizers, losses, schedules, and training loops belong to the caller. No trainers, ever.
