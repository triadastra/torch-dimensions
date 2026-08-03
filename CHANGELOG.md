# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Verified
- **CUDA, on real hardware, for the first time.** An NVIDIA RTX 5090
  (Blackwell, sm_120; torch 2.12.1+cu130) ran the full suite — **1211 passed,
  6 skipped, 0 failed** — and `scripts/cuda_check.py` reported **13 passed, 0
  failed, 1 skipped**. `prefer_upstream` returns True on a GPU for the first
  time; the vendored S4 DPLR agrees with CPU at **1.9e-07 including L=64**,
  where MPS lands exactly on the Nyquist pole, so the guard is provably inert
  on CUDA; the rank-1 LSTM is still bitwise identical under cuDNN. Results in
  [docs/cuda-checklist.md](docs/cuda-checklist.md), raw output in
  `CUDA bench/cuda_check.txt`.
- **The device comparison, corrected twice and then meaningful.** Worst float32
  output difference between the RTX 5090 and an M1 Ultra is **3.11e-06**, with
  13 of 16 models at or below 1e-06 and float64 agreement at **2.2e-16**. See
  [AGREEMENT.md](AGREEMENT.md) and [COMPARISON.md](COMPARISON.md).

### Fixed
- **Mamba-2 chose its fused path from the box, not the tensor.**
  `use_mem_eff_path` defaulted to `torch.cuda.is_available()`, so on a machine
  with a GPU every CPU-resident Mamba-2 asked for a kernel it could not reach
  and raised `NotImplementedError` on its first forward — every CPU sanity
  check, every CPU test, and the CPU half of any device comparison. It now
  dispatches per call, gated on the `mamba-ssm` entry point the path actually
  calls rather than on Triton alone, and refuses float64 where the fused kernel
  has no instantiation.
- **The vendored gated RMSNorm did the same thing**, dispatching on
  `_HAS_TRITON` and sending a CPU tensor into a Triton kernel on any box with
  both a GPU and Triton.
- **`agreement.py` saved every tensor as `.float()`**, putting a float32 floor
  (~1.2e-07) under the float64 comparison whose only purpose is to measure
  beneath it. The float64 column had been reporting the saving code.
- **`agreement.py --only` rewrote `agreement.json`** with just the subset it
  ran, silently dropping every other model's result.
- **`scripts/cuda_check.py` had three bugs of its own**, unrunnable until a GPU
  existed: it fed a 16-wide input to every mixer regardless of `d_model`, and
  compared a full four-layer model against a bare `nn.LSTM`. Widths are now
  required rather than defaulted.
- **The examples' `register_nd_method` leaked into the global registry** for
  every test that ran afterwards.

### Changed
- **A fixed seed does not give identical S4/S4D weights across platforms, and
  the benchmarks no longer pretend it does.** `hippo.nplr` diagonalises with
  `torch.linalg.eigh`: eigenvalues are unique and matched across macOS and
  Linux to twelve decimals — which is exactly why this hid, since `A_imag`
  looked perfect — but eigenvectors are fixed only up to a phase, so `B` and
  `P` differed by a relative 1.5 and 0.53. The comparison was reporting a
  2.6e-01 difference for the vendored S4D, unchanged in float64, that was two
  different models rather than two devices. Both benchmarks now take `--init`
  and share one set of starting weights; the same pair agrees at 4e-07, and
  after 300 optimiser steps their losses agree to 8.9e-07. Every run records
  `weights_from` per model, and `compare.py` refuses to describe two runs as
  comparable when they did not share weights.
- **CUDA does not run float32 by default, and the benchmarks now say which
  precision they used.** `cudnn.allow_tf32` ships as `True`, so cuDNN
  convolutions and RNNs execute in TF32 — 10 mantissa bits against float32's
  23 — which accounted for the *entire* original 1.96e-04 device gap and
  explains why only LSTM, GRU, CNN and TCN were affected while attention,
  Mamba and S4 sat at ~2e-07. With it off, `tcn_2d_sparse` improves 1024x and
  `cnn_2d_sparse` 839x. `agreement.py` defaults to `--tf32 off` because a
  numerical comparison must compare like with like; `pretrain.py` defaults to
  `--tf32 torch`, which touches nothing, so its speed numbers describe what a
  user actually gets. Both record the settings *as observed*, since on a
  machine without CUDA they do nothing and a manifest claiming otherwise would
  assert a control that was never applied.

### Added
- **`benchmarks/compare_agreement.py`** — the cross-device numerical report,
  elementwise on the saved tensors rather than on summary statistics, with
  output and gradient reported separately. A gradient of an SSM frequency is
  cancellation-dominated and cannot meet a float32 bound by construction
  (`A_imag`: 1.35e-04 in float32, 4.87e-15 in float64); the report names that
  case instead of flagging an ill-conditioned quantity as disagreement.
- **`benchmarks/init_weights.py`** and **`benchmarks/precision.py`** — the two
  controls above, each documenting the measurement it exists to protect.
- **`tests/test_matrix.py`** — 13 mixers x 4 methods, all 52 cells, five
  invariants each. The library's thesis is that any mixer composes with any
  method; the diagonal of a matrix is not the matrix.
- **`tests/test_published_checkpoints.py`** — every shipped checkpoint loads
  with `strict=True` into the model its manifest names, and runs. The published
  weights are a compatibility contract, and an untested contract is a promise.
- **`tests/test_init_weights.py`**, **`tests/test_precision.py`**, and Mamba-2
  dispatch tests.


## [0.3.1] - 2026-08-03

### Added
- **`td.viz.weights(model)` and a weights view in the viewer** — the
  parameters themselves, drawn as the mechanism rather than as a heatmap of
  everything. Every tensor is classified by the *role* it plays (linear, conv,
  ssm_decay, ssm_in, ssm_out, skip, bias), which is what lets a linear map be
  drawn as a matrix, a convolution as a receptive field of taps, and an SSM as
  a bank of states with a retention loop each. Downsampling is always
  declared: each entry carries its original shape, the stride used, and the
  statistics of the *whole* tensor, because a picture of one corner of a
  matrix presented as the matrix is worse than no picture. Served at
  `/weights.json` by `td.viz.show(model)`, and 404 with a reason when the
  viewer was opened on a spec, which has no parameters to read.
- **The impulse-response operator: one picture every family can be drawn in.**
  A mixer is a map over positions along the swept axis; families differ in the
  structure that map is *forced* to have. Probing each mixer with a unit
  impulse per position measures it directly, and the payload reports what it
  finds — causality, bandwidth, weight-tying, and reach past the diagonal. The
  numbers separate the families exactly as the theory says they should: a
  convolution is banded and tied (the same kernel repeated at every position),
  a TCN and an SSM are strictly causal, an SSM carries real influence past the
  diagonal while attention carries none — because attention's mixing is
  computed from the data and is not in the parameters to draw at all. That
  last case is reported in those words rather than as an empty diagram.
- **Findings, histograms, and a frequency told apart from a decay.** Each
  layer reports what is actually wrong with its weights, as measurements
  rather than a score: how many output units are dead (judged against the
  tensor's own scale, since an absolute threshold means nothing across
  initialisations), and what the SSM decay rates are. The decay is reported
  as a *rate*, not as a per-step retention, because the retention is
  `exp(-rate·dt)` and `dt` is a different learned tensor — at unit dt a Mamba
  state with rate 8 reads as instant forgetting where its learned dt of ~0.01
  makes it 92% a step, the opposite conclusion. Uniform decay is not flagged
  when the states are separated by frequency instead, which is exactly how
  S4D-Lin initialises: a check that fires on the default configuration is a
  check people learn to ignore. Every tensor also carries a histogram, since
  min/max/std describe a healthy spread and a spike at zero with two outliers
  identically.
- **Weights stream while training.** `examples/viewer_live.py` refreshes the
  digest on the eval cadence, so the diagrams show what the model holds now
  rather than what it held when the page was opened. On the eval cadence and
  not every step, because the operator probe is forward passes and a viewer
  that changes the training cost is measuring itself.
- **`td.data.sparsity(...)`** — a pre-run over the data that answers "is this
  lattice actually sparse", because that is a property of the data rather than
  a setting to declare. Takes a `Lattice`, a `LatticeTable`, a boolean mask, or
  a raw tensor whose non-finite entries mark absence (`missing=` for a sentinel
  such as 0.0). `report.percent_sparse` is the headline; `report.summary()`
  prints it with a per-axis breakdown that separates scattered gaps from one
  whole slice missing — the second is usually a join that went wrong upstream.
  An ambiguous lattice placement inside a tensor is refused rather than
  guessed, since silently picking is how a transposed axis survives to
  training.
- **The viewer splits in half: the lattice on the left, the model on the
  right.** The model screen draws the layer stack as a flow — input shape into
  the model into output shape — with one node per layer carrying its axis and
  direction, its mixer, and its weight count as a bar scaled against the
  heaviest layer. The mechanisms actually in play are named (composition
  method, mixer types, families present), and when the layer weights do not add
  up to the model's total, the remainder is labelled "outside layers" rather
  than left to quietly disagree. Selecting a layer on either side selects it on
  both.
- **`data_show`** — a toggle that labels every cell with its axis names and
  indices, tinted by which side of the wavefront it is on. Labels are drawn
  from a fixed pool assigned to the nearest cells and then thinned in *screen*
  space, so none of them overlap and zooming in reveals more; a label per cell
  across a rank-4 lattice is thousands of text meshes and an unreadable thicket
  besides.
- **Live cell contents.** With a run attached, `examples/viewer_live.py`
  streams the prediction and target for every present cell each step — reusing
  the training forward pass, since a viewer that changes the training cost is
  measuring itself — and the labels show them under the coordinates. The
  viewer checks the array length against the spec before using it: numbers
  against the wrong cells would be worse than no numbers.
- **The sweep runs at training speed.** One pass over every layer is one
  training step, so the wavefront is clocked to the measured step rate rather
  than a decorative constant, clamped at both ends because a fast run is a
  strobe and a slow one a still image. The measured rate is shown in the dock
  as steps/s.
- **An analytics dock** across the bottom of both halves: status, step, losses,
  throughput, run controls, and a loss curve that now has room to be a curve
  with a labelled log axis. It sizes itself to the space available — a
  hard-coded chart width made the dock wider than its container and pushed the
  sidebar off the left edge of the window.

### Changed
- **The sidebar is tabbed** (model / layers / data) instead of one column
  holding everything at once, with the run metrics moved out to the dock.
- **The viewer reads as motion instead of a slideshow.** The wavefront is a
  gaussian that travels and swells the cells it passes, rather than a hard
  stripe switching cells between two flat states; behind it a wake decays
  onto a floor, so "which part of the axis is done" stays legible however
  far back the front has gone. Fog, a ground grid and a gradient sky give
  the rank-4 dimensional stacking the depth it needs to stop reading as a
  wall of cubes. The direction arrow now rides the front — derived from the
  same `screenOffset` the cells use, because computing it from the screen
  extent put the arrow at one end while the glow was at the other, two
  lattice axes sharing one screen dimension above rank 3. The camera drifts
  on its own until you touch it, then stops for good. Axis labels scale with
  the model, so they are the same apparent size on a 5x7 lattice and a
  rank-4 stack. The active layer's row fills with its own progress, driven
  by the same animation ref the scene uses so nothing re-renders at 60fps.
  The directionless families keep their single uniform pulse: a travelling
  ripple would look better and would imply an ordering they do not have.
- **Mamba-2, from the authors' code, running on CPU and MPS.**
  `td.Mamba(..., version=2)` and `td.Mamba2(...)` are the same model — the
  vendored upstream `Mamba2` block (SSD: multi-head, gated RMSNorm). Its
  fused path needs Triton and CUDA, so off GPU the block takes upstream's own
  `use_mem_eff_path=False` route and the chunked scan is computed by *their*
  reference `ssd_minimal_discrete` ("the same as Listing 1 from the paper"),
  with the gated norm falling back to their `rms_norm_ref`.
  `mixers/mamba2_compat.py` is the adapter that presents the fused kernels'
  calling convention on top of that reference — ours, and verified against an
  independent naive recurrence in float64 at **3e-16** across non-chunk-
  aligned lengths, grouped B/C, dt bias, D skip and z gate. MPS-vs-CPU 6e-07,
  gradients finite.
- **Mamba-3 — the authors' block, with their Triton scan transcribed into
  PyTorch.** `td.Mamba(..., version=3)`, `td.Mamba3(...)` and `td.Mamba3ND`
  build the vendored upstream `Mamba3` block (rotary state, trapezoidal
  discretization, heavy-tail `A`). Unlike Mamba-1 and Mamba-2, upstream
  ships **no pure-torch reference** for Mamba-3 — the recurrence exists only
  as ~6,300 lines of Triton — so off GPU it is computed by
  `mixers/mamba3_compat.py`, **our transcription**, and the mixer is called
  `Mamba3Mixer` rather than `Upstream…` for exactly that reason. What is
  established: the chunked and recurrent forms, written independently, agree
  to **3e-15** in float64; a third direct O(L^2) sum agrees at **2e-16** in
  the `trap -> 1` limit; `trap -> 0` removes the current step as the
  trapezoid claims; chunk size does not change the answer; gradcheck passes
  (the 1,788-line Triton backward is not ported — autograd differentiates
  the forward). What is *not* established: equality with their kernel, which
  no CUDA-less machine can check. MPS-vs-CPU 1.4e-06.

### Changed
- **Fused kernels are now chosen per call, not per install.** A CUDA tensor
  plus an importable Triton kernel runs the authors' kernel; anything else
  (no Triton, a failed import, CPU or MPS tensors) runs the torch path.
  Deciding at import time got the in-between case wrong: Triton installs
  fine on a CUDA-less box, so the kernel was picked and then died on the
  first CPU tensor. `TD_FORCE_TORCH_KERNELS=1` forces the torch path, which
  is how the fallback gets exercised on a CUDA machine.
- **The default S4/S4D/Mamba implementation is now the original authors'
  code.** `td.S4`, `td.S4D`, and `td.Mamba` construct the vendored upstream
  blocks by default; `portable=True` selects the pure-torch mixers (the old
  default). The flag is recorded in the model config, so checkpoints rebuild
  what they were trained with — and checkpoints written before the flag
  existed rebuild `portable=True`, bitwise. The originals' dependencies are
  auto-installed on the first call that needs them (announced first;
  `TD_NO_AUTO_INSTALL=1` turns that into an error with the manual command;
  `portable=True` never installs anything). Passing both `portable=True` and
  `mixer=` is refused. Golden specs now name the upstream mixers in the
  default models' layers.

## [0.2.0] - 2026-08-03

### Added
- **The original authors' S4/S4D/Mamba code, shipped verbatim.**
  `torch_dimensions/_vendor/` redistributes, from state-spaces/s4, the
  pipeline subtree their `train.py` actually runs — `S4Block` (the layer
  their registry calls `"s4"`), the S4ND layer, the kernel modules
  (`fftconv`, `ssm`, `dplr`, `kernel`), HiPPO, the functional kernels, and
  the `nn`/`utils` modules they import (23 files) — and, from
  state-spaces/mamba, `mamba_simple.py`, `selective_scan_interface.py`, and
  `utils/torch.py`. Both Apache-2.0, byte-identical to pinned upstream
  commits; the s4 tree keeps its directory structure and is mounted as the
  `src` package (their own repo convention), so even hydra's string-target
  registry resolves unchanged and **zero s4 import lines are patched**. The
  only patched files: `utils/train.py` (training-only imports guarded) and
  the two mamba files (import paths plus one dispatch line: off GPU,
  `selective_scan_fn` runs the authors' own `selective_scan_ref`). Every
  changed line is tagged `torch-dimensions patch`, pristine `.orig` copies
  ship beside patched files, `tests/test_vendored.py` fails on any untagged
  difference, and `dossier/verify_vendored.py` proves the bytes against the
  real repositories.
- **The vendored pipeline runs on MPS.** The DPLR kernel's bilinear
  transform divides by `1 + ω` at the Nyquist node; MPS's power op lands
  exactly on ω = −1 at some lengths (L=64 reproduced NaN, forward and
  backward). A tagged guard in the vendored `ssm.py` nudges only an *exact*
  pole hit — never true on CPU/CUDA, where outputs are verified bit-for-bit
  unchanged — after which CPU-vs-MPS agrees ≤ 1e-6 for L=16..256 with finite
  gradients. Measured parity for the rest: vendored Mamba 3e-8, S4D 1.2e-7,
  S4ND (16×16) 8e-6.
- **`UpstreamS4DMixer`, `UpstreamS4Mixer`, `UpstreamMambaMixer`** — the
  vendored originals wrapped in the `(M, A, H)` mixer contract, so the exact
  upstream blocks sweep over any lattice:
  `td.Mamba(64, 6, lattice=lat, mixer=UpstreamMambaMixer)`. The S4 mixers
  construct upstream's real `S4Block` through their own hydra registry
  (`mode='diag'` for S4D, `mode='dplr'` for S4). CI checks the originals
  against the portable mixers: the pipeline's S4D kernel matches ours
  **bitwise** with shared parameters; Mamba to 1e-5. Needs the new
  `[upstream]` extra (einops, numpy, scipy, hydra-core, omegaconf — the
  originals' own imports).
- **Kernel-family options from the CaFA reference implementation**:
  `qk_norm=` (RMS-normalize query and key; no new parameters) and
  `kernel_residual=` (a learnable `gamma * I` added to the kernel *before* the
  gate, so a contraction starts near the identity and has to learn to mix).
  Both default to off, so existing models are bit-for-bit unchanged.
- **Convolutional family — `td.CNN` and `td.TCN`.** The first mixers that are
  not sequence models at all: no state, no direction, no "so far". Sweeping a
  1-D convolution per axis is a *separable* convolution, exactly equal (linear,
  no bias) to one N-D convolution whose kernel is the outer product of the
  per-axis kernels — checked against `F.conv2d`/`F.conv3d` at 1e-12, with a
  nonlinearity as the negative control. `td.TCN`'s dilation doubles with the
  number of times **its own axis** has been swept, so receptive field grows
  exponentially per axis; a scan layer passes `sweep=` only to mixer factories
  whose signature names it.
- **`td.receptive_field(model)`** — per-axis span against axis size, so "can
  this model see across the lattice at all" is answerable before training.
  `inf` for RNNs, SSMs and attention; a real constraint for convolutions.
- **`td.ViT` and `td.PatchEmbed`** — rank-generic patch embedding (a 3-D volume
  patchifies with the same code) plus the transformer stack over the patch
  lattice. Factorized positional embedding by default (`r·A` parameters
  instead of `A^r`), ViT's full table one argument away. Per-patch features
  out; no class token and no head, the same boundary every other model keeps.
- **`td.flatten`** — the fourth method of multidimensionality and the honest
  baseline for the others: every axis folded into one sequence, no
  factorization, which is what ViT actually does. The one composition where a
  sparse lattice is genuinely cheaper — absent cells are dropped from the
  sequence rather than masked within it.
- **`td.testing.check_lti`** and **LTI.md** — measure whether a mixer is linear
  and time-invariant, and what that does to N-D composition. Never raises: no
  mixer is supposed to be LTI. The document corrects a piece of folklore this
  project repeated — LTI does **not** imply the sweep order is free. A
  multichannel convolution is a matrix-valued filter and its per-offset channel
  matrices do not commute (measured order gap 3e-01); only a scalar-valued
  filter `A ⊗ b` commutes (3e-16). What LTI does buy is exact separability.
- **`td.Transformer`** — attention as the swept mixer, which makes "N-D
  Transformer" literal and gives both constructions in the literature a name:
  `td.Transformer` sweeps attention along one axis per layer, while
  `method=td.axial_attention` builds per-axis kernels and contracts them.
  `AttentionMixer` is non-causal by default (a mixer is never told which axis
  it sweeps, so masking "the future" of a spatial axis is meaningless); the
  causal mask, when asked for, is tested bitwise in both directions.
- **`td.viz.show(model)`** — the architecture viewer, shipped inside the
  wheel. No node, no network, no build step at install time. Accepts a model,
  a spec dict, a spec JSON, or a checkpoint.
- **Sub-lattices:** `Lattice.sliced(**axes)` returns the sub-lattice *and* the
  tensor selection as one object, so the two cannot drift apart;
  `Lattice.merge` is the inverse. Integer indices are refused — a rank that
  changes with a train/test split is not a rank.
- **Plan algebra and coverage:** `+`, `* k`, `.reversed()` (layer order) and
  `.flipped()` (sweep direction), plus `plan.coverage(lattice)` — per-axis
  sweep counts, directions, `unswept` and `pinned`, machine-readable. `spec()`
  now derives its sweeps section from it and gained `pinned_axes`.
- **`safetensors` checkpoints** (optional extra), chosen by file extension,
  with the config in the container's metadata: still one file, and one that
  cannot execute code when it is opened.
- **`td.testing.Recorder`** — a mixer that computes nothing and records every
  call, droppable into a real model with the new `mixer=` argument.
- **`td.testing.check_data_source`** — the `LatticeSource` protocol gets the
  treatment mixers already had, including the picklability check whose failure
  mode is a hang rather than an error.
- **`td.data.MemmapSource`** and **`td.data.masked_stats`** — an on-disk
  source that survives `DataLoader` workers, and statistics computed over
  present cells only (a mean over a sparse lattice's structural zeros is
  dragged toward zero, invisibly).
- **[RESULTS.md](RESULTS.md)** — reproductions with hardware attached: sMNIST
  99.53% and psMNIST 97.79% with `td.S4D`, 2-D lattice image classification,
  and a sparse-lattice forecasting baseline that has no published prior.
- **[BENCHMARKS.md](BENCHMARKS.md)** — measured costs, including two results
  that contradicted this project's own predictions: cost tracks the *length of
  the swept axis* rather than the cell count (rank 1 costs ~95× rank 4 at equal
  cells), and the factorized kernel family only overtakes per-line attention at
  64³ rather than everywhere.
- **Guides** for both extension points ([adding a mixer](docs/adding-a-mixer.md),
  [adding an nd_method](docs/adding-a-method.md)), with examples executed by the
  test suite, and a [CUDA verification checklist](docs/cuda-checklist.md).

- **`td.build(checkpoint)`** returns the architecture a checkpoint records with
  fresh weights (`weights=True` is `load`), plus `td.read_config(path)`.
- **Entry-point registration** (`torch_dimensions.models` group) so third-party
  packages can add model kinds without being imported eagerly.
- **`scripts/mutate.py`** — mutation testing as a weekly CI job. Seven
  mutations, each one a bug this project actually had; all seven caught. A
  survivor is reported as a hole in the tests, not a bug in the code.
- **`scripts/check.sh`** — exactly what CI runs, in the same order, because
  "I ran the linter" and "I ran the check that gates the build" turned out to
  be different sentences three times.
- **`tests/test_perf.py`** — a perf smoke on *ratios between configurations*
  rather than absolute times, so machine speed cancels and complexity does not.

### Changed
- The conformance suite's **Kronecker check runs** — it had been an
  unconditional skip since it was written. Pass `check_block(kernels=...)`.
- Ranks 5 and 6 are tested (full conformance, both families), so the README's
  "ranks ≥ 5 untested" caveat is gone.
- The README no longer claims CUDA: it has never been run, and that is now
  stated with a checklist beside it.
- Coverage floor in CI at 95% (measured 97%); `pytest-cov` had been installed
  and unused since the first commit.

### Fixed
- **The spec described a model the library never runs.** For a kernel-family
  model (`method=td.cafa` / `td.axial_attention`) it claimed one spatial sweep
  per layer — sweeps that do not happen, since every layer contracts *all* the
  spatial axes and the mixer runs along time only — and `"family"` was the
  hardcoded string `"scan"`. The viewer already compensated by sniffing the
  class name, which is a bug report about the producer. Spec version 1 → 2:
  layers carry `kind`, the `axes` they actually mix, and `contracted`; sweeps
  gained `contracted_axes` and list directions only for axes a mixer genuinely
  sweeps (DEBUG.md #26).
- A stale local training run was baked into the viewer bundle and loaded *in
  preference to* the model passed to `show()` (DEBUG.md #19).
- `check_block`'s gradient step built at `d_model=2` regardless of the caller,
  so a width-constrained mixer failed a gradient check with an error about
  head counts (DEBUG.md #23).
- numpy was assumed present — it is not a torch dependency, and both the
  memmap source and `safetensors`' torch bindings need it. The extras declare
  it, the error says what to install, and the tests skip visibly without it
  (DEBUG.md #24).
- The reproductions' dry-run timer did not synchronize the accelerator, so it
  measured the dispatch queue and underestimated a run by 40x (DEBUG.md #25).

## [0.1.0] — 2026-07-31

The first release: the full portable core.

### Added
- **Models, 1-D and N-D under one name:** `LSTM`, `GRU`, `S4` (full DPLR),
  `S4D`, `Mamba` — plus the explicit N-D names `S4ND`, `S4DND`, `MambaND`,
  which make `dim` mandatory and refuse `dim=1` as the 1-D model in disguise.
  The SSM mixers are pure torch (CPU/CUDA/MPS), verified against the upstream
  reference kernels: S4D bitwise, S4 at 3e-8 (and machine-precision against a
  dense state-space reference in CI), the Mamba scan at 1e-6.
- **Methods of multidimensionality:** `td.axial_scan`, `td.axial_attention`,
  and `td.cafa` (Kronecker-factorized, per-axis kernels), selectable per model
  via `method=` / `nd_method=`, including the hybrid form — kernels across the
  lattice, the model's own mixer along time, causal along time by construction.
- **Phase 3–4:** `AxialScan` with per-layer pre-norm residuals; the conformance
  suite as public API (`td.testing.check_block`, `td.testing.check_trainable`
  with a negative control that must fail).
- **Phase 5 data layer:** `from_coords`/`from_table`, `LatticeWindow`,
  the `LatticeSource` protocol, `collate_lattice`.
- **Phase 8:** `td.build` from dict or YAML with hard errors on unknown keys;
  the model registry; `save`/`load` checkpoints that rebuild their own model —
  validity mask included, outputs bitwise equal, incompatible format versions
  refused.
- **Device suite** running against whatever accelerator exists (MPS or CUDA),
  and a seeded fuzz suite checking core invariants against independent slow
  references.
- [DEBUG.md](DEBUG.md): all 18 bugs found while building this, what caught
  each, and the recurring patterns — with its citations enforced by a test.
- Architecture design ([DESIGN.md](DESIGN.md)) and phased build plan ([PLAN.md](PLAN.md)).
- Phase 0 packaging skeleton: `src/` layout, `torch>=2.4` as the only required
  dependency, kernel backends behind optional extras, ruff + pytest + CPU-only CI.
- Phase 1 `Lattice`: axis naming and resolution, permutation to and from folded
  1-D sequences, scatter/gather for lattices whose cells are not all populated,
  broadcast validity masks, and per-axis valid counts for masked pooling.

- Phase 2 `ScanPlan`: per-layer sweep schedules as plain data — `cyclic`,
  `paired`, and `from_list` constructors, name-or-index axes resolved against a
  lattice, dict round-trip, and a warning when a plan leaves an axis unswept.

### Changed
- Data construction (`Lattice.from_coords`, windowing, the `LatticeSource`
  protocol, `collate_lattice`) is now in scope as Phase 5, and models gain
  `save`/`load` in Phase 8. Training loops remain permanently out of scope.

### Removed
- `ScanPlan.hilbert()` from the planned constructor list. A step is
  `(axis, reverse)` and a space-filling curve is not axis-aligned, so it needs
  a different step kind entirely. Deferred rather than stubbed.

- Per-axis bidirectionality in `ScanPlan`: `bidirectional=` accepts a collection
  of axes, so time can stay causal while spatial axes are swept both ways.
  Construction warns when the layer budget cannot deliver what was requested —
  bidirectional coverage of *k* axes needs roughly *2k* layers.

- Phase 3 `axial_apply` and `AxialScan`: sweep any 1-D mixer along one lattice
  axis at a time, with pre-norm residual layers driven by a `ScanPlan`. Absent
  cells are zeroed on entry and after each layer, so outputs at present cells
  are invariant to whatever values sat in absent ones.
- `LSTM` and `GRU` — one class each covering 1-D and N-D, plus `LSTMMixer` /
  `GRUMixer` adapters over `torch.nn`. Direction is the schedule's job, so
  neither adapter sets `bidirectional=True` on the underlying RNN.
- `nd_method`: how a model's extra axes are handled. Strategies are plain
  functions exported at top level (`td.axial_scan`, with `td.axial_attention`
  and `td.cafa` to follow), and a user-written function is on the same footing.
  Names still resolve, but only because YAML cannot hold a callable.
- `Lattice(shape=(), time=True)` — a lattice that is only a sequence. Its
  permutation is the identity, so the 1-D path needs no special-casing and
  `td.LSTM(d_model, n_layers)` with no lattice is an ordinary sequence model.

### Changed (breaking, pre-alpha)
- `LSTMND` / `GRUND` are now `LSTM` / `GRU`, taking an optional `lattice=`.

- Phase 4 `td.testing.check_block` — the shared conformance suite, shipped as
  public API so a user's own mixer or `nd_method` gets the same verification the
  library runs on itself. Checks shape, gradients and `gradcheck`, rank-1
  equivalence, absent-cell invariance, covariance with axis storage order, and
  `torch.compile` numerics. A check that cannot run is reported as **skipped**,
  never as passed.

- Phase 5 `td.data` — long-format rows to lattice layout. `from_coords` infers
  the lattice, its vocabularies, and which cells are absent; `from_table` builds
  the `(T, *shape, F)` series; `LatticeWindow` handles time windowing as pure
  index arithmetic; `LatticeSource` is a protocol so any storage backend works;
  `LatticeDataset` and `collate_lattice` plug into `torch.utils.data.DataLoader`.
  No trainer, no normalization policy, no downloads.
- `d_input=` on the RNN models, adding a single input projection when the data
  is not already `d_model` wide.

- `td.spec(model)` — a versioned JSON description of a model's N-D
  architecture: the lattice with a run-length-encoded presence mask, the
  resolved per-layer sweep schedule, mixer identities and parameter counts, and
  symbolic input/output shapes. Derived without a forward pass. It also reports
  which directions each axis is *actually* swept in, and which axes a plan never
  sweeps. Foundation for the viewer ([VIEWER.md](VIEWER.md)).

- `td.testing.check_trainable` — fits a small task that genuinely needs axial
  mixing and checks the block learns it. Separate from `check_block`, which asks
  whether a block is *correct*; this asks whether it *converges*, which can fail
  independently through initialization, masking, or activation scale.
- `examples/train_nd.py` — the full path from long-format rows to a trained
  model, in plain PyTorch.

- Phase 6 (part 1) `axial_contract` and `kron_operator` — per-axis kernels
  contracted into a joint operator that, on a dense lattice, is exactly the
  Kronecker product. Verified against the explicitly materialized operator at
  ranks 1-4. Sparse lattices renormalize each output line by the kernel mass
  landing on present cells, generalized to arbitrary rank rather than keyed to
  one.

The attention modules land next.
