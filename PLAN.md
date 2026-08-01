# torch-dimensions — Build Plan

Companion to [DESIGN.md](DESIGN.md). That document says *what* the library is; this one says *in what order it gets built, how we know each step worked, and how much of the surface each step actually covers*.

**Sequencing principle:** each phase ends with something runnable and tested, and the riskiest unknowns get resolved before anything is built on top of them. Concretely that means the tensor bookkeeping (`Lattice`) is proven before any model exists, the test suite exists before the hard math, and the config surface is written **last** — a config schema authored before the block signatures settle just freezes a guess.

**Legend.** `[x]` done and verified (a test or a shipped artifact backs the tick — a tick with no evidence is not a tick). `[ ]` open. `[~]` partially done, with the gap stated inline. Sizes are relative (S ≈ half a day, M ≈ 1–2 days, L ≈ 3–5 days, XL ≈ 1–2 weeks) for one person already fluent in this material.

**Coverage** lines state what fraction of the surface a phase's ticks actually exercise — ranks, dtypes, devices, densities, model families — because "phase done" and "surface covered" are different claims, and conflating them is how silent gaps ship.

---

## Current position (2026-08-01, second pass)

**Track A:** Phases 0–11 complete except where noted. **9 and 10 closed this pass** — reproductions with numbers (sMNIST 99.53%, psMNIST 97.79%, a sparse-lattice baseline that had no published prior) and benchmarks that contradicted two of the design's own predictions. v0.1.0 is live on PyPI. Phase 12 (autoregressive stepping) is designed and is now the largest single piece of unbuilt core.
**Track B (viewer):** V1, V1.5 and **V2 done** — `td.viz.show(model)` ships inside the wheel. V3 (shape flow) is next and unstarted.
**Track C (performance):** the checklist exists; **no CUDA kernel has ever run**. This is the biggest gap between what the library claims and what it has demonstrated.
**Track D (ecosystem):** both extension-point guides written and executed by tests; no docs site, no notebooks, no tech report.
**Track E (research):** parked by design.

Next up, in order of value: **run the CUDA checklist** (fifteen minutes, and it converts the project's largest unverified claim), **three seeds per RESULTS.md row** (every row is currently n=1), **Phase 12 AR stepping**, **Viewer V3**.

What this pass changed, in one line each: 23 bugs now documented (four new, two found by *looking at what shipped* rather than at the build); ranks 5–6 tested so the README caveat could be deleted; the Kronecker conformance check runs for the first time; `td.Transformer` completes the family the README table always claimed; coverage floored at 95% (measured 97%).

---

# Track A — Core library

## Phase 0 — Skeleton and packaging · S · no deps

Make the repo installable and enforceable before any logic exists.

- [x] `pyproject.toml` (hatchling), `src/` layout so tests import the installed package rather than the working tree.
- [x] Dependencies: `torch>=2.4` required and **nothing else**. Optional extras: `[mamba]`, `[fla]`, `[dev]`, `[all]`. (An `[s4]` extra existed until the portable S4D landed in-tree and made it pointless.)
- [x] `ruff` (lint + format), `pytest`, `pytest-cov`. Type hints throughout; `mypy` advisory — *and executed in CI*, because "advisory" once quietly meant "never run" (DEBUG.md #7).
- [x] CI: CPU matrix on Python 3.10–3.12, full-history checkout (DEBUG.md's citation test needs it).
- [x] `LICENSE` (Apache-2.0), `README.md`, `CONTRIBUTING.md`, `CHANGELOG.md`.
- [x] Publishing workflow: tag-triggered, PyPI trusted-publisher OIDC, sdist-size guard (the 35 MB node_modules incident, now a CI check).
- [x] sdist scoped to the library (`src`, `tests`, `examples`, docs) — the sdist is the library, not the repo.
- [ ] Python 3.13 in the CI matrix once torch ships stable wheels for it.
- [ ] A `nightly` CI job against torch nightly, so upstream breakage is our alarm and not our users'.
- [ ] Windows runner (even one smoke job) — path handling and MPS-vs-CUDA device pick are the risks.

**Coverage:** packaging is fully exercised (clean-venv wheel install is tested; PyPI install is live). CI covers Linux CPU only — macOS/MPS runs locally, Windows never.

**Acceptance:** `pip install -e ".[dev]"` then `pytest` passes on a machine with no CUDA and no optional deps installed. ✅ Also now: `pip install torch-dimensions` from PyPI works cold.

---

## Phase 1 — `Lattice` · M · needs Phase 0

The foundation. Pure tensor bookkeeping, zero models, zero learnable parameters.

- [x] `shape`, `names`, `valid`, `time`; axis name → position resolution; degenerate `shape=()` + `time=True` sequence lattice.
- [x] Permutation generation and inverse for "move axis `k` to sequence position"; inverse checked against `torch.argsort`, not hand-derived twice.
- [x] Flat indices, scatter-to-dense, gather-from-dense; broadcast validity mask; per-axis valid counts for masked pooling.
- [x] Immutable after construction (`__setattr__` refused) — a value object that is hashed from and cached from must be a value (DEBUG.md #2).
- [x] Defensive copies: `valid` cloned on entry, `mask()` returns a fresh tensor, never a view (DEBUG.md #13).
- [x] Device correctness: `flat_idx` indexes on the data tensor's device, both mismatch directions (DEBUG.md #18, proven on MPS).
- [x] Property/fuzz tests: random shapes and masks, ranks 1–4, fold/scatter/permutation round-trips against independent references.
- [x] Extend the fuzz envelope to ranks 5–6 — done, and both families now pass **full conformance at ranks 5 and 6** with a rank-5 model trained end to end. The README caveat is deleted because it stopped being true.
- [x] Stress shapes: every axis length 1, a single valid cell, a 10,000-long axis — explicit cases rather than fuzz luck (`test_stress_shapes_that_fuzz_would_have_to_be_lucky_to_draw`).
- [x] `Lattice.sliced(**axes)` / `Lattice.merge` — sub-lattice views for splits over *space*. `sliced` returns the lattice **and** the tensor selection as one object, so the two cannot drift apart; integer indices are refused because a rank that changes with a train/test split is not a rank.
- [ ] Serialization guarantee documented: a lattice pickled/JSON-round-tripped on one torch version loads on the next (add a stored-fixture test).

**Coverage:** ranks 1–6 fuzz-tested against independent references; dense and sparse; CPU + MPS; float32/float64; degenerate and extreme aspect ratios as explicit cases. Not covered: pickling across torch versions, ranks ≥ 7 (untested, and now merely untried rather than claimed).

---

## Phase 2 — `ScanPlan` · S · needs Phase 1

Pure data. No tensors, no modules.

- [x] `.cyclic()`, `.paired()`, `.from_list()`; `__repr__`; to/from dict; name-or-index axes resolved against a lattice; warning when a plan leaves an axis unswept.
- [x] Per-axis `bidirectional` (time stays causal while space does not); the `set("time")` string-iteration trap guarded.
- [x] Direction flips per *cycle*, not per layer — the even-axis-count aliasing that silently pins every axis one way (DEBUG.md #4; exists in published research code).
- [x] `_warn_if_pinned`: a layer budget that cannot deliver the requested bidirectionality warns instead of silently downgrading.
- [x] Immutable and hashable together (DEBUG.md #1).
- [x] Plan algebra: `+`, `* k`, `.reversed()` (layer order) and `.flipped()` (sweep direction), named apart because calling either one "reversed" alone is how they get confused. The identity that earns it: `p + p.flipped()` is bidirectional everywhere `p` swept at all.
- [x] `plan.coverage(lattice)` — per-axis sweep counts, directions, layer indices, `unswept` and `pinned`. It never warns, because a report that warns cannot be used to decide whether to warn; `spec()` now derives its sweeps section from it and gained `pinned_axes` for free.
- [ ] Schedule catalog: named constructors for the published schemes beyond Mamba-ND's paired (e.g. S4ND's simultaneous-separable as a degenerate plan, zigzag variants) with citations.
- [ ] Property test: for every constructor, every axis mentioned is swept ≥ 1 time or the warning fires — fuzz over axis counts 1–8 and layer counts 1–64.

**Coverage:** constructors, resolution, serialization, algebra, coverage reporting, and the aliasing signature are fully tested (even *and* odd axis counts — one parity finds nothing). Not covered: exotic published schedules.

---

## Phase 3 — `AxialScan` + RNN family · M · needs Phases 1–2

- [x] `axial_apply`: permute → fold → mixer → unfold → inverse permute; `reverse`; `chunk` (validated ≥ 1); shape-contract error naming `(M, A, H)`.
- [x] `AxialScan`: pre-norm residual per layer, per-layer or shared mixers, absent cells zeroed on entry **and** after every layer.
- [x] `LSTM`, `GRU` — one class each, 1-D without a lattice, N-D with one; no `LSTMND`.
- [x] Single layer on a rank-1 lattice matches `nn.LSTM` **bit-for-bit**; multi-layer matches to float tolerance (fold normalizes memory layout; torch RNN kernels are layout-sensitive — expected, not a defect, and documented).
- [x] Verified against independent references: `cumsum`, position-weighting — mixers that torch itself can check.
- [ ] Batch-fold chunking auto-tuned from device limits rather than user-supplied. Phase 10 now says what the tuner should do: **chunking never won on time** here — `chunk=64` cost 3× by launching 3× the kernels — so it is a memory-pressure valve, and the default should be off until an allocation actually fails.
- [x] `td.testing.Recorder` — a mixer that computes nothing and records every call, droppable into a real model via the new `mixer=` argument. That argument needed a guard: a substituted mixer cannot go into the recipe, so `save()` refuses rather than silently rebuilding a *different* model that loads without complaint.
- [ ] Gradient-checkpointing option per layer (rank-4 lattices at real `d_model` will need it; measure in Phase 10 first).

**Coverage:** ranks 1–4, dense + sparse, time on/off, forward + reverse + chunked, float64 gradcheck, MPS parity. Not covered: activation checkpointing, mixed precision (Track C), very long folded batches near memory limits.

---

## Phase 4 — Conformance suite · M · needs Phase 3

The single highest-leverage phase: it is what keeps a (models × ranks × sparsity × devices) matrix from rotting.

- [x] `td.testing.check_block` — the seven checks: shape, gradients + gradcheck, rank-1 equivalence vs the bare mixer, Kronecker identity (kernel family), absent-cell inertia, axis-storage covariance, `torch.compile` (opt-in).
- [x] Skips are recorded, never silently passed (`_Skip` is the mechanism; "we never ran that one" can never read as "that one passed").
- [x] Checks run only at ranks the caller requested (DEBUG.md #16).
- [x] `check_trainable`: fresh data per step, held-out scoring, and a negative control that must fail — a learning test without one measures capacity, not learning (DEBUG.md #5).
- [x] Fuzz suite (`test_fuzz.py`): seeded, checked against slow independent references, mutation-verified.
- [x] Device suite (`test_device.py`): runs against whatever accelerator exists (MPS locally, CUDA elsewhere), skips visibly otherwise.
- [x] Check #4 — the Kronecker identity **runs now**. It was an unconditional skip since it was written; `check_block(kernels=...)` takes an adapter returning the operators a block actually used, and CaFA passes at ranks 2–3 with a negative control that fails.
- [ ] Check #8 — parallel/sequential equivalence (lands with Phase 12; specified there).
- [ ] Mutation-testing harness as a scheduled CI job (weekly, not per-push): auto-apply the catalog of mutations from DEBUG.md §B1, fail if any survive that previously died. Today mutation testing is a manual discipline; make it a machine's.
- [x] Coverage floor in CI at 95%, measured at **97%** across 447 tests. `pytest-cov` had been installed and unused since Phase 0 — DEBUG.md #7's shape, one tool over.
- [x] `td.testing.check_data_source(source)` — five checks including slice-seam consistency and picklability, both with negative controls. The pickle check matters most: DEBUG.md #9's version of that failure *hung* rather than raised.
- [x] Golden-file specs, one per family, compared verbatim with a field-path diff and a `TD_UPDATE_GOLDEN=1` regeneration path. Every previous spec test asserted particular keys, so a silently added or renamed field passed them all and reached the viewer as a blank panel.

**Coverage:** every shipped block passes all applicable checks at ranks 1–3 (RNN + SSM + kernel + attention families) and ranks 5–6 for the scan and kernel families, dense and sparse, CPU float64 + accelerator float32; data sources and spec documents have their own checkers. Not covered: automated mutation runs, fp16/bf16 (Track C1).

---

## Phase 5 — Data · M · needs Phase 4

Building a lattice from data is lattice construction and belongs here; running a training loop is not and never will be.

- [x] `from_coords` — vocabularies, shape, valid mask inferred from observed tuples; unknown values raise; encode/decode round-trip fuzz-tested.
- [x] `from_table` — long-format rows to `(T, *shape, F)`; duplicate `(time, cell)` rows refused (a join bug must not become a plausible dataset).
- [x] `LatticeWindow` — pure index arithmetic; split drops straddling windows on both sides; unsorted timestamps refused (a silent nonsense split is the quietest possible leakage bug — DEBUG.md #15).
- [x] `LatticeSource` protocol + `TensorSource`; `LatticeDataset`; `collate_lattice` (ragged refused; mixed target presence refused — DEBUG.md #14; lattice kept out of the batch).
- [x] Multiprocessing-safe: samples and batches pickle; `DataLoader(num_workers>0)` proven end-to-end (DEBUG.md #9 — it used to *hang*).
- [x] `td.data.MemmapSource` — the on-disk reference, written to fail the way on-disk sources fail: the handle opens lazily and is dropped on pickling, because a live mmap in a `DataLoader` worker hangs rather than raises. Passes `check_data_source`.
- [ ] A zarr source behind an extra — one on-disk implementation is a demonstration; two would be a pattern.
- [x] `td.data.masked_stats` / `Normalizer` — statistics over present cells only. A plain `series.mean()` on a 30%-sparse lattice is dragged 30% toward zero by structural zeros, the scale with it, and nothing about the resulting model looks wrong. NaN is treated as absent too.
- [ ] Ragged-time policy documented and tested: what a source with per-cell history lengths should do (today: build the union lattice and mask; write the recipe down with a test).
- [ ] Streaming/windowed iteration for series too large for memory (windows over a source that only supports sequential reads).
- [ ] An end-to-end "CSV to trained model" example with a real public dataset (feeds Phase 9 and Track D).

**Coverage:** in-memory and memory-mapped paths tested including worker processes; property-tested windowing; masked normalization with its failure mode as an explicit test. Not covered: zarr, ragged series, >memory scale.

---

## Phase 6 — Kernel family · M/L · needs Phase 4

- [x] `axial_contract` + `kron_operator`; the factorization checked against the explicit Kronecker product, not against itself.
- [x] Sparse renormalization with the **relative** cancellation guard — degeneracy is cancellation, and cancellation is relative to the absolute mass (DEBUG.md #3 → #12: the first fix repeated the bug one level up; float32 blew up 7,000× under an absolute epsilon).
- [x] No NaN laundering: an input NaN must leave (DEBUG.md #11).
- [x] `AxialKernel`: per-line scores (axial attention) or pooled per-axis kernels (CaFA); learned relative-position bias per axis; softmax or leaky_relu gate; per-line renormalization *is* masked softmax for the softmax gate.
- [x] The hybrid form: kernels own space, the model's mixer owns time; CaFA pools over *other spatial axes only* — a kernel at time t built from the future would leak through a causal model, and the causality test holds the hybrid to bitwise on the past.
- [x] A mixer on a time-less lattice refused as dead weight.
- [x] `td.axial_attention`, `td.cafa` as strategies; registered by name; `method=` as the short spelling of `nd_method=`.
- [x] Module-level Kronecker equivalence for CaFA — closed, via the `kernels=` adapter in `check_block`. Worth recording *what* the adapter returns: CaFA pools the current activation, so the second axis's kernel is built from the first axis's output. The factors are the ones actually applied, not the ones a static reading predicts.
- [ ] Multi-head kernels (currently single-head per axis) — heads are the difference between this and what a Transformer person expects to configure.
- [x] `td.Transformer` / `AttentionMixer` — attention as the swept mixer, which makes "N-D Transformer" literal and gives both constructions in the literature a name. Non-causal by default (a mixer is not told which axis it sweeps, so masking "the future" of a spatial axis is meaningless); the causal mask is tested bitwise in both directions.
- [x] Cost model, measured rather than derived: BENCHMARKS.md's "Where factorization starts winning". The answer was **not** what this line assumed — per-line attention is faster everywhere below ~50 cells per axis, and factorization only leads at 64³.
- [ ] `gate="softmax"` temperature / learned-per-axis option (CaFA paper ablates this).

**Coverage:** both strategies pass full conformance at ranks 1–3 and 5–6, dense + sparse, both gates, learnability, hybrid causality, the module-level Kronecker identity, MPS. Not covered: multi-head kernels, gate temperature.

---

## Phase 7 — SSM family · L · portable core done; fast paths open

**Portable core (done):**

- [x] `S4DMixer` — diagonal kernel, **bitwise identical** to upstream's with copied weights.
- [x] `S4Mixer` — the full DPLR kernel: HiPPO-LegS NPLR init, Cauchy resolvent, rank-1 Woodbury, bilinear transform with the **Nyquist-pole guard built in** (upstream survives that pole only by rounding luck; on MPS it NaNs — recorded below). Matches upstream at 3e-8; matches a dense state-space matrix-power reference at 4.6e-16 **in CI**.
- [x] `MambaMixer` — gated conv + selective scan, mirroring the reference non-fused path; scan matches `selective_scan_ref` at 1e-6.
- [x] `td.S4`, `td.S4D`, `td.Mamba` via the shared `LatticeModel` base; explicit N-D names `S4ND`/`S4DND`/`MambaND` with mandatory, checked `dim` (`dim=1` refused by name — one spatial axis is the 1-D model, and code reading "S4ND" must not be running S4).
- [x] Causality tests (S4 causal to 2.6e-15 through the FFT — mathematically causal, not bitwise, and the test says exactly that; Mamba held to bitwise).
- [x] Full conformance, learnability, device parity for all three.

**S4/Mamba portability dossier (established empirically, 2026-07-30, MPS):** the upstream repos run on Mac once vendored copies drop `pytorch_lightning`/hub/`transformers` imports and fix the `_omega` Nyquist pole — full S4Block at 1.13e-6 CPU↔MPS parity, Mamba v1 at 2.98e-8, Mamba2 via an `ssd_minimal` adapter at 4.77e-7 (adapter verified against a sequential recurrence at 8.9e-15 over 40 ragged-length cases). Import shims: stub `selective_scan_cuda`, stub triton *after* torch imports with `.configs`-bearing autotune, stub the hub mixin. **Mamba3 ships no reference implementation** — every path is a 712-line triton kernel; a portable version means writing the rotary trapezoidal recurrence from the paper.

**Fast paths and breadth (open):**

- [ ] `mamba-ssm` fast-path adapter: when installed and on CUDA, `MambaMixer` delegates to the fused kernel; the conformance suite runs *both* paths and asserts they agree — the shim harness from the portability work is the seed.
- [ ] `causal-conv1d` fast path for the conv (same pattern, same agreement test).
- [ ] `Mamba2Mixer` (SSD): portable chunked implementation — the verified `chunk_scan_ref` adapter *is* this, one packaging step away — plus the triton fast path when available.
- [ ] `Mamba3Mixer`: write the pure-torch reference from the paper (own task, XL); verify on a CUDA box against upstream triton; then it joins the family.
- [ ] S4 options parity where they earn their keep: rank-2 low-rank correction, `legt`/`fourier` measures, learnable-vs-fixed `dt` per the paper's ablations — each lands only with a test against the dense reference.
- [ ] `fla` (flash-linear-attention) adapter behind the existing `[fla]` extra — currently the extra installs a dependency nothing uses; that is a small dishonesty with a deadline.
- [ ] Sequential-scan speedup for the portable Mamba path: block-parallel (associative) scan in pure torch — the O(A) python loop is correct and honest, and a log-depth scan would make the portable path usable for long 1-D sequences too.

**Coverage:** the three shipped mixers are verified against upstream references *and* independent in-repo references, on CPU/MPS, ranks 1–3, all conformance checks. Not covered: any fused kernel, Mamba-2/3 as mixers, CUDA execution anywhere.

---

## Phase 8 — Registry, config, save/load · M · done

- [x] `model.config` — the construction recipe recorded at build, plain JSON-able types, `n_layers` recorded as the plan's true depth (a recipe that could disagree with itself is not a recipe).
- [x] `td.build(dict | yaml path)` — registry by kind (`lstm`, `gru`, `s4`, `s4d`, `mamba`, + ND names); unknown keys are a hard error naming the offender.
- [x] `model.save(path)` / `td.load(path)` — one checkpoint file: format version, config, state dict; rebuilt model is the same model (outputs bitwise equal; validity mask included); incompatible format versions refused.
- [x] The round-trip proven for every model family and both composition families.
- [x] `safetensors` as an optional container, chosen by file extension, with the config carried in its metadata — still one file, and one that cannot execute code when opened. A test asserts the written file contains no pickle.
- [ ] Checkpoint migration policy written down: what version N promises to load from version N−1, and a stored-fixture test per released version (the fixtures directory *is* the compatibility contract).
- [ ] `td.build` accepting a checkpoint path directly (config-with-weights vs config-only is a flag, not two APIs).
- [ ] Registry entry points (`importlib.metadata`) so third-party packages can register model kinds without importing them eagerly.

**Coverage:** every shipped model round-trips config and weights bitwise. Not covered: cross-version loading (no released fixture set yet), third-party registration.

---

## Phase 9 — Reproduce a published result · L · needs Phases 5–8 · **DONE for the sequence and sparse rows**

The make-or-break test of the abstraction: a published N-D result, reproduced from a config file alone, with no code written outside this library. Until this passes, "the unification holds" is a design claim, not an empirical one.

Candidates, in order of feasibility on available hardware:

- [x] **sMNIST / psMNIST with `td.S4D`** — **99.53%** and **97.79%** test accuracy, 20 epochs each, 201k parameters, ~21 min per run on one Apple Silicon laptop. The S4D paper reports ~99.6% and ~98.5% with far longer schedules, so both land inside the one-point target. This isolates the mixer from the N-D machinery: the lattice is the degenerate one.
- [x] **2-D lattice image classification** (`examples/repro/image_nd.py`) — MambaND with the paired schedule, S4DND, and LSTM over a real 2-D lattice, no pixel flattened into a sequence. Runs on MNIST-as-image today; CIFAR-10 is wired and checksummed and waits only on a download.
- [x] **Sparse-lattice forecasting** on UCI Beijing air quality — 12 stations × 6 pollutants, hourly, a genuinely 2-D lattice. `masked` vs `zeros` arms, same data and budget, scored on present cells only so the comparison is about representation and not about which cells count. Sparsity is **induced and said to be induced**: a naturally sparse dataset confounds "is missing" with "is different", and removing cells at random isolates the variable.
- [x] **Method comparison as one flag** — `cafa` and `axial_attention` arms of the same forecasting experiment, differing in exactly one argument.
- [x] Each ships as a `python -m` runnable, an append-only ledger, a RESULTS.md row with hardware and wall-clock, and a CI smoke variant on synthetic data (`tests/test_repro.py`) — including the check that makes the sparse comparison mean anything: a 1e3 perturbation in an absent cell must not move the output by one bit.
- [ ] Seed variance: every row is one seed. Three seeds per row is the next honest increment, and until then no row should be read as a mean.
- [ ] CIFAR-10 row (the download stalled repeatedly on this connection; the code path is tested, the number is not taken).

**Acceptance:** met — RESULTS.md carries reproducible rows, each one command. **Coverage:** the sequence and sparse-lattice claims have numbers; the image rows are laptop-scale and say so; nothing here has run on CUDA.

**Risk:** dataset licensing and download flakiness — vendor nothing; download scripts with checksums, and CI smoke uses synthetic stand-ins.

---

## Phase 10 — Benchmarks · M · **DONE on MPS; CUDA open**

Measure the known risks before making any performance claim.

- [x] Fold overhead: **10–30% of a step** at rank ≥ 2 and small `d_model`, falling as width grows. Real, and not the first thing to optimize.
- [x] Dense vs factorized, and then the sweep that matters: the crossover is at **64³**, and below ~50 cells per axis per-line attention is 2–3× *faster*. The family earns its complexity at the top end, not everywhere.
- [x] Portable mixer costs, MPS: the Mamba scan is 5–8× the kernel-based SSMs — the number a fused path must beat.
- [ ] The same table against a fused path and against CUDA (Track C).
- [x] `torch.compile` is a **0.8× slowdown** on MPS at these sizes, so it is not recommended by default. Re-measure before repeating that about any other machine.
- [x] Chunked fold: never won here; `chunk=64` cost 3×. It is a memory valve, not a speed knob.
- [x] Rank 1→5 at fixed cells — and the design claim was **wrong**. Cost tracks the *length of the swept axis*, not the cell count and not the rank: rank 1 costs ~95× rank 4 at the same 4,096 cells, because a sequential mixer's cost is its launch depth.
- [x] Published as BENCHMARKS.md with the machine named and the scripts in-repo. Interpretations live in the script as a `finding` field, because regenerating overwrites the markdown and an interpretation a re-run deletes is one nobody rewrites.
- [ ] A CI perf-smoke: one tiny timed run with a generous regression threshold (2×) — catches the accidental O(n²) without flaking on runner noise.

**Acceptance:** met on MPS. **Coverage:** one machine, float32, forward and forward+backward. Not covered: CUDA, half precision, multi-GPU, memory (only CUDA tracks an allocation peak — see DEBUG.md #20 for the plausible substitute that was removed).

---

## Phase 11 — Docs and release · M · **v0.1.0 SHIPPED; docs open**

- [x] README: the unification table, verified quickstart (every snippet executed before written), honest scope limits, "Correct on purpose" section.
- [x] CHANGELOG discipline (Keep-a-Changelog, real 0.1.0 entry).
- [x] PyPI: trusted publishing, tag-triggered, live — `pip install torch-dimensions` works.
- [x] DEBUG.md as a living practice document, with its citations enforced by a test.
- [ ] Docs site (mkdocs-material): API reference from docstrings, the design docs rendered, versioned with releases.
- [x] **[Adding a mixer](docs/adding-a-mixer.md)** — a real mixer end to end, with its conformance report, the table of what each check protects against, and the mistakes to expect. The example is executed by `tests/test_examples.py`.
- [x] **[Adding an nd_method](docs/adding-a-method.md)** — same treatment, and its best material was an accident: the conformance suite failed the guide's own example twice (DEBUG.md #21, #22), so the page documents what it caught and a test reproduces the broken version.
- [ ] Tutorial notebooks: (1) forecasting a real CSV end-to-end through `td.data`, (2) the method-of-multidimensionality comparison on one task, (3) sparse lattices — why masking matters, with the Phase 9 numbers.
- [ ] Docstring pass with a doctest runner in CI (examples in docstrings that execute are examples that stay true).
- [ ] Release automation niceties: CHANGELOG section extracted into the GitHub release notes by the publish workflow.

---

## Phase 12 — Autoregressive stepping · L · needs Track C fast paths · v0.2

Generation over a lattice: process one timestep at a time with carried state, so the same models that trained in parallel can roll forward step by step.

**The shape of it.** Autoregression is a property of the *time* axis only — the spatial axes are fully materialized at every step. So the primitive is

```python
state = model.init_state(batch)             # per-layer, opaque
y_t, state = model.step(x_t, state)         # x_t: (B, *shape, H) — one timestep
```

and a layer's behavior at step time follows from what it sweeps: a layer sweeping **time** consumes and updates its slice of the state; a layer sweeping a **spatial** axis runs exactly its normal forward on the single timestep (nothing to carry). The kernel family needs no new mechanism at all — its spatial kernels are already timestep-local, which the causality test proves today.

- [ ] `Mixer.step(x, state) -> (y, state)` as an optional protocol method; a mixer without it cannot be stepped and the model refuses AR mode loudly at `init_state`, never at step 500.
- [ ] `LSTM`/`GRU` state: `(h, c)` — free, `nn.LSTM` already does this.
- [ ] `Mamba` state: conv ring buffer (`d_conv−1` inputs) + selective-scan state — mirrors upstream's inference cache, and must be *the same design* as the fused path's cache (why this phase follows Track C).
- [ ] `S4`/`S4D` state: the recurrent view of the same SSM — materialize `(dA, dB)` from the learned parameters with the *same* discretization as the kernel; DPLR steps in the diagonalized basis, O(N) per channel per step. The dense-reference test already proves the recurrence equals the kernel.
- [ ] Refusals: a plan sweeping time backward (bidirectional time says the future is visible — generation cannot honor it); any step-less mixer. Constructor-time, pointed messages.
- [ ] Conformance check #8: parallel `forward` equals sequential `step`ping — bitwise for RNNs, float tolerance for SSMs. The single check that catches the classic AR bugs (off-by-one state, conv buffer misalignment, leaked normalization statistics).
- [ ] `save`/`load` round-trips mid-generation state (a checkpoint that cannot resume generation is half a checkpoint).
- [ ] Viewer: step mode in the run panel (Track B V5) — generate cell-by-cell along time and watch the lattice fill.

**Deliberately out of scope, permanently:** sampling policy, beam search, KV caches for attention over time, readout heads — the caller's, same as training loops. `step` is the primitive; generation loops are five lines of user code around it.

---

# Track B — Viewer / GUI mode

The library emits a versioned JSON spec (`td.spec`, SPEC_VERSION 1); the viewer consumes it. The boundary is a document, never a socket into the library.

## V1 — Static architecture viewer · **done**

- [x] React + Vite + react-three-fiber; lattice rendered as cubes, absent cells genuinely absent; sweep wavefront animated per layer; per-layer timeline; sample specs for 2-D sparse LSTM, 3-D paired Mamba-ND, 4-D S4D.
- [x] Rank ≥ 4 via **dimensional stacking**: a cube of cubes, spaced groups along the fourth axis (the "16-pack" idea).
- [x] Open-a-spec-JSON from disk; sidebar with model card, directions, unswept-axis warning.

## V1.5 — Live runs and controls · **done**

- [x] `examples/viewer_live.py`: state to `run.json` after every step; viewer polls, auto-switches, loss chart (train + held-out, log scale), progress bar, status lifecycle.
- [x] Run controls: the script **waits** — nothing trains until Start is pressed; pause froze the counter (proven 5255 → 5255), resume, stop; control server with CORS; state machine.
- [x] Presets: `lstm2d`, `mamba4d` (18-layer Mamba-ND over a sparse 4-D lattice — trained live to held-out 9e-5).

## V2 — Ship it in the wheel · M · **done**

- [x] `td.viz.show(model)` — the bundle rides inside the wheel (387 KB), so there is no node, no network and no build step at install time. `serve()` is the non-blocking half.
- [x] The publish workflow builds the bundle and greps the built **wheel** for `viz/static/index.html` rather than trusting the build log; CI has a viewer job.
- [x] `show(spec_dict)`, `show(path.json)`, `show(checkpoint)`. Anything else raises — an empty viewer is a far worse error message than an exception.
- [x] A stale local training run got baked into the first bundle and the viewer loaded it *in preference to* the model passed in (DEBUG.md #19). Stripped by the installer, guarded by a test.
- [ ] Playwright smoke in CI: the three samples render, the sidebar populates. The accessibility-tree assertions are proven by hand; making them a script is what remains.

## V3 — Shape flow · M

- [ ] Per-layer tensor-shape inspector: hover a layer, see `(B, [T,] *shape, H)` → fold → `(M, A, H)` → restore, with the actual numbers for a chosen batch size.
- [ ] The fold visualized: which axes collapse into `M` for the hovered layer (this is the single hardest thing to explain in prose; it is one animation).
- [ ] Parameter breakdown per layer (bar, hover for tensor names) from the spec's `n_params`.

## V4 — Metrics dashboard · L

- [ ] Multi-run compare: `run.json` grows a run id; the panel lists past runs (ring buffer on disk), overlays loss curves.
- [ ] More channels: learning rate, gradient norm, per-axis gradient norms (does the `w` sweep learn faster than `h`? — a genuinely novel N-D diagnostic), step time.
- [ ] Live lattice heatmap: per-cell loss contribution at eval time, painted on the 3-D lattice — *where* on the grid the model is wrong, over training.
- [ ] Export: PNG of the scene, CSV of metrics.

## V5 — Full GUI mode (the original vision) · XL

- [ ] Config editor in the panel: edit d_model/layers/method/plan with validation via `td.build`'s schema errors, see the architecture update live *before* any training.
- [ ] Launch from the panel: the control server grows a `POST /launch` that spawns the training script with the edited config (opt-in flag, localhost only, explicitly not a deployment tool).
- [ ] Expected-output preview: run one forward pass on synthetic data at build time; show output shapes and ranges ("press continue to train").
- [ ] Plan editor: drag layers, flip directions, see the coverage report (Phase 2's `plan.coverage`) recompute — the aliasing bug rendered as a picture would have caught DEBUG.md #4 in a viewer.
- [ ] AR step mode (with Phase 12): generate along time, watch the lattice fill cell by cell.
- [ ] Sparse-mask painter: click cells on/off, export the `valid` mask — the fastest way to build a demo lattice.

---

# Track C — Performance & hardware

## C1 — CUDA verification · M · **procedure written, not yet run**

- [x] The procedure exists as [docs/cuda-checklist.md](docs/cuda-checklist.md): what to run, what each item protects, and what it deliberately does not cover. The README now says CUDA is a design claim rather than a test result, which it should have said from the start.
- [ ] Run it. Fifteen minutes on a free Colab T4; the device suite was built to run there unchanged. **Nothing in this project has executed a single CUDA kernel.**
- [ ] The bitwise rank-1 LSTM claim re-checked under cuDNN (expected: holds per single layer; document whatever is true).
- [ ] fp16/bf16/AMP conformance additions: the relative-cancellation guard was *designed* dtype-aware; prove it at fp16 where the accidental-safety finding (rounding to exact zero) came from.
- [ ] A `gpu` CI lane (GitHub GPU runners are paid — decide when Phase 7 fast paths land; until then, a documented manual Colab checklist per release).

## C2 — Fast paths · L (see Phase 7 open items)

- [ ] The adapter pattern: portable path is the reference; fused path must agree with it in the conformance suite on the same machine. Never ship a fast path whose only test is "it ran".

## C3 — Throughput engineering · L · gated on Phase 10 numbers

- [ ] Fused/permute-avoiding fold if Phase 10 says permute dominates (the only justified kernel).
- [ ] `chunk` auto-tuning from device properties.
- [ ] Associative-scan (log-depth) portable selective scan.
- [ ] Activation checkpointing option; memory profile at rank 4.
- [ ] `torch.compile` graph-break audit per family (the conformance hook exists; make the graphs clean).

## C4 — Distributed sanity · M · v0.3+

- [ ] DDP: one multi-process CPU test that gradients sync correctly (buffers like `cell_mask` must not desync — the `persistent=False` choice needs a test under DDP broadcast).
- [ ] FSDP smoke with the SSM families (complex-as-real parameters interact with flat-param sharding; find out now, not in a user issue).

---

# Track D — Ecosystem & community

- [ ] Docs site + guides (Phase 11 items; listed here because they gate community, not release).
- [ ] Example gallery in-repo: one runnable file per (family × method) pair — the matrix is the product; show it as a matrix.
- [ ] GitHub issue templates that ask for `td.spec(model)` output — the spec doubles as a bug report's reproduction seed.
- [ ] `CITATION.cff` + a short tech report (arXiv) once Phase 9 rows exist — the sparse-lattice baseline is the publishable nugget.
- [ ] Hugging Face Hub integration via Phase 8 checkpoints (config + safetensors maps cleanly; do it after safetensors lands, not before).
- [ ] "State of the lattice" doc: honest comparison table vs mamba-ssm, s4, FLA — what to use when, including "not us".
- [ ] Conformance badges: a downstream mixer repo can run `check_block` in its CI and claim conformance — write the two paragraphs and the badge that make that a thing.

---

# Track E — Research extensions (parked deliberately; each is a design, not a stub)

- [ ] **Space-filling traversals** (Hilbert/Morton) as a *second step kind* carrying a full cell permutation — rejected from `ScanPlan` v1 because a step is `(axis, reverse)` and a curve is not axis-aligned; the design doc for step-kind-2 must handle sparse lattices (curve over present cells only?) before any code.
- [ ] **Dynamic-shape lattices**: `dim` without `shape` — rank known, sizes bound at first forward. The scan family needs only rank; masks and kernel-family bias tables need sizes. Doable for dense scan-only models; the design question is what `spec()` and `save()` mean before binding.
- [ ] **Learned schedules**: differentiable relaxation over sweep order (soft mixture of axis sweeps annealed to a hard plan). Research-grade; the `ScanPlan`-as-data design is what makes it even expressible.
- [ ] **Hierarchical / nested lattices** (multi-resolution grids, lattice-of-lattices): the S4ND paper's multi-scale appendix and weather models both want this; `Lattice.merge/slice` (Phase 1) is the substrate.
- [ ] **Masked-pretraining utilities**: random cell masking as a training-time lattice transform — the `valid` machinery already guarantees inertia; the utility is three functions and a tutorial, but only after Phase 9 proves the supervised story.
- [ ] **Continuous-time lattices**: irregular timestamps per cell (the `dt` in SSMs is *built* for this — S4's continuous-time parameterization applied per-observation). The biggest research swing in the list; would make td the only library doing irregular N-D series natively.

---

## Critical path

```
                             ┌──► 5 ✅ ┐
0 ──► 1 ──► 2 ──► 3 ──► 4 ──┼──► 6 ✅ ┼──► 8 ✅──► 9 ✅──► 10 ✅──► 11 (docs: guides ✅, site open)
       (all done)            └──► 7 ✅ ┘                │
                                   │                    └──► Track D (has 9's numbers now)
                                   └──► [C1 CUDA] ──► C2 fast paths ──► 12 AR (v0.2)
Track B (viewer): V1 ✅ ── V1.5 ✅ ── V2 ✅ ── [V3] ── V4 ── V5 (V5 wants 12)
```

The critical path now runs through **C1**, and it is fifteen minutes of
someone else's GPU. Everything downstream of it — the fused fast paths, the AR
state design that must match the fused cache, the half-precision claims — is
blocked behind a machine this project has never had. That is the single
highest-leverage hour available to anyone reading this.

Phases 5, 6, 7 were independent once the conformance suite existed, and that bet paid: all three landed without touching each other. The same principle now says **9 and C1 are independent** — reproduction runs on MPS while CUDA verification happens on Colab — and both feed 10. The library remains useful and shippable without a single CUDA kernel; that stays deliberate.

---

## Repo layout

```
torch-dimensions/
├── pyproject.toml                  hatchling; torch>=2.4 only; extras [mamba] [fla] [dev] [all]
├── README.md · DESIGN.md · PLAN.md · VIEWER.md · DEBUG.md · CHANGELOG.md · CONTRIBUTING.md
├── .github/workflows/              ci.yml (full-history checkout) · publish.yml (OIDC, size guard)
├── src/torch_dimensions/
│   ├── lattice.py · plan.py · spec.py · config.py · testing.py
│   ├── compose/                    scan.py · kernel.py · attention.py · __init__ (strategies, registry)
│   ├── mixers/                     base.py · rnn.py · ssm.py
│   ├── models/                     base.py · rnn.py · ssm.py
│   └── data/                       coords.py · table.py · window.py · source.py · collate.py
├── tests/                          one file per module + conformance, device, fuzz, trainable, debug_md
├── examples/                       train_nd.py · viewer_live.py
└── viewer/                         Vite + React + react-three-fiber (excluded from sdist)
```
