# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
