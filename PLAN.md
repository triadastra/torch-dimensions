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
- `LSTMND`, `GRUND`.

**Acceptance:** on a rank-1 lattice, `LSTMND` matches `nn.LSTM` **bit-for-bit**. That single test catches essentially every permutation and residual bug.

**Deliverable:** a working N-dimensional LSTM — prior art as MDRNN (Graves et al. 2007) and Grid-LSTM (Kalchbrenner et al. 2015), but with no maintained modern implementation. Shippable on its own.

---

## Phase 4 — Conformance suite  ·  M  ·  needs Phase 3

Built now, before the hard math, so every later block is validated by construction rather than retrofitted. This is the single highest-leverage phase in the plan: it is what keeps an (models × ranks × sparsity) matrix from rotting.

The seven checks are specified in [DESIGN.md §6](DESIGN.md). Packaged as `td.testing.check_block(...)` so downstream users can run it against their own mixers.

**Acceptance:** `LSTMND` and `GRUND` pass all seven. Suite is parametrized over rank 1–4 and dense/sparse.

---

## Phase 5 — `AxialKernel` and axial attention  ·  L  ·  needs Phase 4

The hardest math in the project. Three strictly ordered steps, each independently verifiable:

1. **Dense contraction.** Per-axis kernel, sequential contraction. Verified against the explicitly materialized Kronecker product on a small lattice — a test only possible while the lattice is small, which is exactly why it happens here.
2. **Sparse renormalization.** Masked-mean pooling and per-line renormalization by valid-key softmax mass, for arbitrary N.
3. **Factorized axial attention.** Per-axis Q/K from the pooled 1-D function, one shared V.

Generalizing rank-locked contraction tables to arbitrary N is the largest single piece of work in the library. Use permute+matmul, not generated einsum strings.

**Acceptance:** Kronecker identity holds to `1e-5` on dense; mask-invariance holds exactly on sparse; `AxialTransformer` and the factorized variant pass the full suite. Memory at rank 4 stays quadratic in axial size — the property that makes the factorized path viable where a per-line implementation exhausts memory.

---

## Phase 6 — SSM adapters  ·  L  ·  needs Phase 4  ·  **needs a GPU**

- Thin adapters over `mamba-ssm` (Mamba-2), `state-spaces/s4`, and Mamba-3.
- Defensive imports: a missing optional dep unregisters that block and leaves everything else importable. A missing real implementation **fails loudly and never silently substitutes a stub**.
- `MambaND`, `S4ND`.

**Risk:** CI cannot cover this. GPU tests run locally and are marked. Budget real time for kernel-integration friction — grid limits, dtype constraints, and version pinning are where this phase actually goes over.

**Acceptance:** both pass the suite on GPU; `MambaND` on a rank-1 lattice matches a bare `Mamba2` block.

---

## Phase 7 — Registry and config  ·  M  ·  needs Phases 5–6

Deliberately late. Now that every block signature has settled, the config schema describes reality instead of predicting it.

- `register()` / `build()` / `list_blocks()`.
- One dataclass schema per block; validation errors name the offending key and list valid ones.
- YAML loader.

**Acceptance:** every model in the library round-trips config → model → config. Unknown keys are a hard error, not a silent ignore.

---

## Phase 8 — Acceptance against a real workload  ·  M  ·  needs Phase 7

**The phase that decides whether the abstraction is right.** Reproduce a published result from one of the source papers — an S4ND or Mamba-ND benchmark number on a public dataset — using only torch-dimensions blocks and config.

If the numbers land, the generalization preserved semantics. If they do not, the library is wrong, and it is far better to learn that here than after release. If expressing the model requires reaching around the API into bespoke Python, the abstraction leaks and Phase 5 needs revisiting.

**Acceptance:** metrics match the paper within seed noise, and the model is expressible in config alone.

---

## Phase 9 — Benchmarks  ·  M  ·  needs Phase 8

Measure the known risk before making any performance claim: two `.contiguous()` calls per layer means a 12-layer ND model does ~24 full-tensor copies, which plausibly dominates the mixer at small `d_model`.

- Permute/copy overhead as a fraction of step time, swept over `d_model` and rank.
- Dense vs factorized memory at ranks 2–4.
- `torch.compile` on/off.

**Acceptance:** published numbers with hardware stated. If permute overhead is material, fusing it becomes the first v0.2 item — and *only then* is writing a kernel justified.

---

## Phase 10 — Docs and v0.1.0  ·  M  ·  needs Phase 9

- README: the unification table, 10-line quickstart, honest scope limits.
- "Adding a mixer" guide — the extension point is the product, so this page matters more than the API reference.
- TestPyPI, then PyPI. Tag `v0.1.0`.

**README must state plainly:** forward-only, no autoregressive stepping; ranks 1–4 tested; fused kernels come from upstream, not from us.

---

## Critical path

```
0 ──► 1 ──► 2 ──► 3 ──► 4 ──┬──► 5 ──┬──► 7 ──► 8 ──► 9 ──► 10
                            └──► 6 ──┘
```

Phases 5 and 6 are independent once the suite exists — 6 is the one to defer if GPU access is intermittent, since everything through Phase 5 is CPU-only. That is deliberate: **the library is useful and shippable without a single CUDA kernel.**

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
│   ├── config.py                   dataclass schemas + YAML loader
│   ├── testing.py                  check_block() — the conformance suite, public
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
│       ├── rnn_nd.py               LSTMND, GRUND
│       ├── transformer_nd.py       AxialTransformer, factorized axial attention
│       └── ssm_nd.py               MambaND, S4ND
│
├── tests/
│   ├── test_lattice.py             permutation + scatter/gather round-trips, property-based
│   ├── test_plan.py
│   ├── test_conformance.py         parametrized: every block × rank 1–4 × dense/sparse
│   ├── test_kronecker.py           factorized == explicit ⊗
│   ├── test_equivalence.py         rank-1 == the underlying 1-D module
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
- **No `training/`, no `data/`, no trainers — ever.** This is a layer library.
