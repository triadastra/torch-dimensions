# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
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
