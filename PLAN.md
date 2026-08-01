# torch-dimensions — Build Plan

Companion to [DESIGN.md](DESIGN.md). That document says *what* the library is; this one says *in what order it gets built, how we know each step worked, and how much of the surface each step actually covers*.

**Sequencing principle:** each phase ends with something runnable and tested, and the riskiest unknowns get resolved before anything is built on top of them. Concretely that means the tensor bookkeeping (`Lattice`) is proven before any model exists, the test suite exists before the hard math, and the config surface is written **last** — a config schema authored before the block signatures settle just freezes a guess.

**Legend.** `[x]` done and verified (a test or a shipped artifact backs the tick — a tick with no evidence is not a tick). `[ ]` open. `[~]` partially done, with the gap stated inline. Sizes are relative (S ≈ half a day, M ≈ 1–2 days, L ≈ 3–5 days, XL ≈ 1–2 weeks) for one person already fluent in this material.

**Coverage** lines state what fraction of the surface a phase's ticks actually exercise — ranks, dtypes, devices, densities, model families — because "phase done" and "surface covered" are different claims, and conflating them is how silent gaps ship.

---

## Current position (2026-08-01)

**Track A:** Phases 0–8 complete. Phase 11 executed early — **v0.1.0 is live on PyPI** with trusted publishing. Phases 9 (reproductions) and 10 (benchmarks) are the open core work. Phase 12 (autoregressive stepping) is designed, targeted v0.2.
**Track B (viewer):** V1 static viewer done; live-run mode with run controls done (the V4/V5 seed). V2 (ship in wheel) next.
**Track C (performance):** unstarted except the MPS device suite; CUDA still unverified anywhere.
**Track D (ecosystem):** README/CHANGELOG/DEBUG.md strong; no docs site, no tutorials.
**Track E (research):** parked by design.

Next up, in order of value: **Phase 9 reproduction**, **Track C CUDA verification**, **Viewer V2**, **Phase 10 benchmarks**.

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
- [ ] Extend the fuzz envelope to ranks 5–6 (the machinery is rank-generic; the claim should be tested, then the README's "ranks ≥ 5 untested" line gets deleted).
- [ ] Stress shapes: one-cell axes everywhere, a single valid cell, a 1×1×…×1 lattice, an axis of length 10⁴ — each as an explicit edge test rather than fuzz luck.
- [ ] `Lattice.merge` / `Lattice.slice` utilities (sub-lattice views for train/eval splits over *space*, not just time) — needed by Phase 9's real-data reproductions.
- [ ] Serialization guarantee documented: a lattice pickled/JSON-round-tripped on one torch version loads on the next (add a stored-fixture test).

**Coverage:** ranks 1–4 fuzz-tested against independent references; dense and sparse; CPU + MPS; float32/float64. Not covered: ranks ≥ 5, extreme aspect ratios, pickling across torch versions.

---

## Phase 2 — `ScanPlan` · S · needs Phase 1

Pure data. No tensors, no modules.

- [x] `.cyclic()`, `.paired()`, `.from_list()`; `__repr__`; to/from dict; name-or-index axes resolved against a lattice; warning when a plan leaves an axis unswept.
- [x] Per-axis `bidirectional` (time stays causal while space does not); the `set("time")` string-iteration trap guarded.
- [x] Direction flips per *cycle*, not per layer — the even-axis-count aliasing that silently pins every axis one way (DEBUG.md #4; exists in published research code).
- [x] `_warn_if_pinned`: a layer budget that cannot deliver the requested bidirectionality warns instead of silently downgrading.
- [x] Immutable and hashable together (DEBUG.md #1).
- [ ] Plan algebra: `plan + plan` (concatenate), `plan * k` (repeat), `plan.reversed()` — composition is currently manual list surgery.
- [ ] A `plan.coverage(lattice)` report object (per-axis sweep counts and directions, machine-readable) — the viewer and the docs both want it; today each recomputes it.
- [ ] Schedule catalog: named constructors for the published schemes beyond Mamba-ND's paired (e.g. S4ND's simultaneous-separable as a degenerate plan, zigzag variants) with citations.
- [ ] Property test: for every constructor, every axis mentioned is swept ≥ 1 time or the warning fires — fuzz over axis counts 1–8 and layer counts 1–64.

**Coverage:** constructors, resolution, serialization, and the aliasing signature are fully tested (even *and* odd axis counts — one parity finds nothing). Not covered: plan composition, exotic schedules.

---

## Phase 3 — `AxialScan` + RNN family · M · needs Phases 1–2

- [x] `axial_apply`: permute → fold → mixer → unfold → inverse permute; `reverse`; `chunk` (validated ≥ 1); shape-contract error naming `(M, A, H)`.
- [x] `AxialScan`: pre-norm residual per layer, per-layer or shared mixers, absent cells zeroed on entry **and** after every layer.
- [x] `LSTM`, `GRU` — one class each, 1-D without a lattice, N-D with one; no `LSTMND`.
- [x] Single layer on a rank-1 lattice matches `nn.LSTM` **bit-for-bit**; multi-layer matches to float tolerance (fold normalizes memory layout; torch RNN kernels are layout-sensitive — expected, not a defect, and documented).
- [x] Verified against independent references: `cumsum`, position-weighting — mixers that torch itself can check.
- [ ] Batch-fold chunking auto-tuned from device limits rather than user-supplied — currently `chunk` is manual; the fused-kernel adapters (Track C) will force the issue.
- [ ] A `Recorder`-style debug mixer in `td.testing` (currently a private test helper) — "which axis did layer 3 actually sweep" is the first question every integration bug asks.
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
- [ ] Check #8 — parallel/sequential equivalence (lands with Phase 12; specified there).
- [ ] Mutation-testing harness as a scheduled CI job (weekly, not per-push): auto-apply the catalog of mutations from DEBUG.md §B1, fail if any survive that previously died. Today mutation testing is a manual discipline; make it a machine's.
- [ ] Coverage floor in CI (`pytest-cov` is installed and unused — DEBUG.md #7's shape, one tool over).
- [ ] A public `check_data_source(source)` — the `LatticeSource` protocol has reference implementations but no conformance checker for *user* sources; the extension point deserves the same treatment mixers got.
- [ ] Golden-file tests for `spec()` output (the viewer contract): a stored spec per model family, diffed on change, so a spec-format break is a loud diff instead of a viewer that quietly renders nonsense.

**Coverage:** every shipped block passes all applicable checks at ranks 1–3 (RNN + SSM + kernel family), dense and sparse, CPU float64 + accelerator float32. Not covered: user-side data sources, spec golden files, automated mutation runs.

---

## Phase 5 — Data · M · needs Phase 4

Building a lattice from data is lattice construction and belongs here; running a training loop is not and never will be.

- [x] `from_coords` — vocabularies, shape, valid mask inferred from observed tuples; unknown values raise; encode/decode round-trip fuzz-tested.
- [x] `from_table` — long-format rows to `(T, *shape, F)`; duplicate `(time, cell)` rows refused (a join bug must not become a plausible dataset).
- [x] `LatticeWindow` — pure index arithmetic; split drops straddling windows on both sides; unsorted timestamps refused (a silent nonsense split is the quietest possible leakage bug — DEBUG.md #15).
- [x] `LatticeSource` protocol + `TensorSource`; `LatticeDataset`; `collate_lattice` (ragged refused; mixed target presence refused — DEBUG.md #14; lattice kept out of the batch).
- [x] Multiprocessing-safe: samples and batches pickle; `DataLoader(num_workers>0)` proven end-to-end (DEBUG.md #9 — it used to *hang*).
- [ ] Reference `LatticeSource` for memory-mapped `.npy` and for zarr (behind an extra) — the protocol's promise is "a database cursor batches correctly"; two on-disk implementations would make that promise tested rather than asserted.
- [ ] A normalization *hook* (protocol, ships nothing): per-cell statistics computed over present cells only — masked normalization is exactly the kind of thing users will get subtly wrong, and a wrong mean over absent zeros is invisible.
- [ ] Ragged-time policy documented and tested: what a source with per-cell history lengths should do (today: build the union lattice and mask; write the recipe down with a test).
- [ ] Streaming/windowed iteration for series too large for memory (windows over a source that only supports sequential reads).
- [ ] An end-to-end "CSV to trained model" example with a real public dataset (feeds Phase 9 and Track D).

**Coverage:** in-memory paths thoroughly tested including worker processes; property-tested windowing. Not covered: on-disk sources, normalization, ragged series, >memory scale.

---

## Phase 6 — Kernel family · M/L · needs Phase 4

- [x] `axial_contract` + `kron_operator`; the factorization checked against the explicit Kronecker product, not against itself.
- [x] Sparse renormalization with the **relative** cancellation guard — degeneracy is cancellation, and cancellation is relative to the absolute mass (DEBUG.md #3 → #12: the first fix repeated the bug one level up; float32 blew up 7,000× under an absolute epsilon).
- [x] No NaN laundering: an input NaN must leave (DEBUG.md #11).
- [x] `AxialKernel`: per-line scores (axial attention) or pooled per-axis kernels (CaFA); learned relative-position bias per axis; softmax or leaky_relu gate; per-line renormalization *is* masked softmax for the softmax gate.
- [x] The hybrid form: kernels own space, the model's mixer owns time; CaFA pools over *other spatial axes only* — a kernel at time t built from the future would leak through a causal model, and the causality test holds the hybrid to bitwise on the past.
- [x] A mixer on a time-less lattice refused as dead weight.
- [x] `td.axial_attention`, `td.cafa` as strategies; registered by name; `method=` as the short spelling of `nd_method=`.
- [ ] Module-level Kronecker equivalence test for CaFA (single layer, identity out-proj, gate off): the pooled per-axis kernels contracted sequentially vs `kron_operator` of the same kernels — closes the one conformance skip that remains for this family.
- [ ] Multi-head kernels (currently single-head per axis) — heads are the difference between this and what a Transformer person expects to configure.
- [ ] Attention **along time** as a mixer (`AttentionMixer`, causal mask) — the missing entry that makes "N-D Transformer" literal: kernel family across space + attention along time, or axial_scan with attention as the swept mixer.
- [ ] Cost model in docs: per-axis O(A²) vs per-line O(M·A²) vs scan O(M·A), with the rank-4 memory cliff worked out — the *reason this family exists* deserves numbers (Phase 10 measures them).
- [ ] `gate="softmax"` temperature / learned-per-axis option (CaFA paper ablates this).

**Coverage:** both strategies pass full conformance at ranks 1–3, dense + sparse, both gates, learnability, hybrid causality, MPS. Not covered: multi-head, attention-as-mixer, module-level Kronecker check.

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
- [ ] `safetensors` as the weights container (optional dependency) — torch pickles are a supply-chain liability the ecosystem is moving away from.
- [ ] Checkpoint migration policy written down: what version N promises to load from version N−1, and a stored-fixture test per released version (the fixtures directory *is* the compatibility contract).
- [ ] `td.build` accepting a checkpoint path directly (config-with-weights vs config-only is a flag, not two APIs).
- [ ] Registry entry points (`importlib.metadata`) so third-party packages can register model kinds without importing them eagerly.

**Coverage:** every shipped model round-trips config and weights bitwise. Not covered: cross-version loading (no released fixture set yet), third-party registration.

---

## Phase 9 — Reproduce a published result · L · needs Phases 5–8 · **OPEN — the current frontier**

The make-or-break test of the abstraction: a published N-D result, reproduced from a config file alone, with no code written outside this library. Until this passes, "the unification holds" is a design claim, not an empirical one.

Candidates, in order of feasibility on available hardware:

- [ ] **Sequential/permuted-image classification with the portable SSMs** (sMNIST/psMNIST with `td.S4D`): small, CPU/MPS-feasible, published baselines abundant. Target: within 1 point of the S4D paper's small-model number. This is the *first* reproduction because it isolates the mixer from the N-D machinery.
- [ ] **2-D: Mamba-ND-style image classification at toy scale** (CIFAR-10 with `td.MambaND(dim=2)`, paired schedule): validates the schedule machinery against the construction the paper describes. Accept a stated-hardware small-config number, not the paper's A100 number.
- [ ] **Sparse-lattice forecasting on a public dataset** (e.g. a public demand/traffic dataset with genuinely absent series): no published baseline exists for the sparse case — *we* set the baseline, which is the differentiator's first citable artifact. Compare dense-with-zeros vs masked-sparse handling; the claim "masking absent cells matters" gets a number.
- [ ] **Hybrid check:** the same forecasting task with `method=td.cafa` vs `td.axial_scan` at matched parameter count — the library's whole point is that this comparison is one flag.
- [ ] Each reproduction ships as: a config file, a `python -m` runnable, a RESULTS.md row with hardware + wall-clock + seed variance (≥ 3 seeds), and a CI smoke variant (tiny config, asserts learning happened, minutes not hours).

**Acceptance:** at least two rows in RESULTS.md that someone else could reproduce with `pip install torch-dimensions` and one command each.

**Risk:** dataset licensing and download flakiness — vendor nothing; download scripts with checksums, and CI smoke uses synthetic stand-ins.

---

## Phase 10 — Benchmarks · M · needs Phase 9 · OPEN

Measure the known risks before making any performance claim.

- [ ] Permute/copy overhead as a fraction of step time — two `.contiguous()` per layer means a 12-layer model does ~24 full-tensor copies; sweep over `d_model` ∈ {32…512} and rank ∈ {1…4}. If it is material, fusing becomes the first v0.2 item — and *only then* is writing a kernel justified.
- [ ] Dense vs factorized (kernel-family) memory and time at ranks 2–4 — the O(A²)-vs-O(cells²) claim with axes on it.
- [ ] Portable Mamba scan vs (Track C) fused path, CPU vs MPS vs CUDA — the honest "when do you need the extra" table.
- [ ] `torch.compile` on/off across families (the conformance hook exists; the numbers do not).
- [ ] Chunked fold (`chunk=`) sweep — find where chunking wins, feed the auto-tune item in Phase 3.
- [ ] Scaling curve rank 1→5 at fixed cell count (does the machinery's cost grow with rank or with cells? design says cells; prove it).
- [ ] Published as BENCHMARKS.md with hardware stated, seeds fixed, and the benchmark scripts in-repo (`benchmarks/`, excluded from the sdist).
- [ ] A CI perf-smoke: one tiny timed run with a generous regression threshold (2×) — catches the accidental O(n²) without flaking on runner noise.

**Acceptance:** published numbers with hardware stated, and every performance sentence in README/docs traceable to a row.

---

## Phase 11 — Docs and release · M · **v0.1.0 SHIPPED; docs open**

- [x] README: the unification table, verified quickstart (every snippet executed before written), honest scope limits, "Correct on purpose" section.
- [x] CHANGELOG discipline (Keep-a-Changelog, real 0.1.0 entry).
- [x] PyPI: trusted publishing, tag-triggered, live — `pip install torch-dimensions` works.
- [x] DEBUG.md as a living practice document, with its citations enforced by a test.
- [ ] Docs site (mkdocs-material): API reference from docstrings, the design docs rendered, versioned with releases.
- [ ] **"Adding a mixer" guide** — the extension point is the product; this page matters more than the API reference. Walk one real example end-to-end: implement, `check_block`, `check_trainable`, register, config.
- [ ] "Adding an `nd_method`" guide (same treatment; the strategy contract in prose with the hybrid example).
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

## V2 — Ship it in the wheel · M · **next in track**

- [ ] `td.viz.show(model, port=…)`: build the bundle, ship it inside the wheel (`[viz]` extra for the server bits if any; the bundle itself is static files), serve locally, open browser. Zero node required at install time.
- [ ] The publish workflow builds the viewer bundle and verifies its size (the sdist guard pattern).
- [ ] `td.viz.show(spec_dict)` and `show(path.json)` variants; `show(checkpoint)` via Phase 8 load.
- [ ] Playwright smoke in CI: the three samples render, the sidebar populates (the accessibility-tree assertions already proven by hand become a script).

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

## C1 — CUDA verification · M · **blocked on hardware access, unblockable by Colab**

- [ ] Run the full suite on a CUDA box (free Colab T4 suffices: `pip install torch-dimensions && pytest tests/test_device.py` plus the marked-slow full run) — the device suite was built to run there unchanged.
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
                             ┌──► 5 ──┐
0 ──► 1 ──► 2 ──► 3 ──► 4 ──┼──► 6 ──┼──► 8 ──► [9] ──► [10] ──► docs
        (all done)           └──► 7 ──┘            │
                                  │                └──► Track D (needs 9's numbers)
                                  └──► C1 CUDA ──► C2 fast paths ──► 12 AR (v0.2)
Track B (viewer): V1 ✅ ── V1.5 ✅ ── [V2] ── V3 ── V4 ── V5 (V5 wants 12)
```

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
