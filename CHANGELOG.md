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

`AxialScan` and `LSTMND` land in Phase 3.
