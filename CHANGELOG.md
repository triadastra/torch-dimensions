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

`ScanPlan` lands in Phase 2.
