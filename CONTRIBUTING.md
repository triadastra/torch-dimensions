# Contributing

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

`torch` is the only required dependency. Kernel backends live behind extras (`[mamba]`, `[fla]`, `[s4]`); install them only if you are working on those adapters.

```bash
ruff check . && ruff format .
pytest -v                  # GPU tests are deselected by default
pytest -m gpu              # requires CUDA
```

## Adding a mixer

The mixer is the extension point, and keeping it trivial is the whole design. A mixer is any `nn.Module` with the signature:

```
(M, A, H) -> (M, A, H)
```

where `A` is the length of the axis currently being swept and `M` is every other lattice axis folded into the batch. It sees a plain batch of 1-D sequences and needs to know nothing about lattices, ranks, or axis order — that is `AxialScan`'s job, not yours.

Once written, register it and run the shared conformance suite:

```python
td.testing.check_block(MyBlock, ranks=(1, 2, 3), sparse=True)
```

That suite is public API, not test-directory scaffolding: it is the same set of checks the library runs against its own blocks. A mixer that passes it works at every rank, dense and sparse, under `torch.compile`, with correct gradients.

## Ground rules

- **Adapters, not reimplementations.** Fused kernels come from upstream. If a change starts reimplementing a selective scan or an FFT conv, it belongs upstream instead.
- **Never silently substitute a stub.** A missing optional dependency unregisters its block and leaves everything else importable. It must never fall back to a different architecture under the original name — a benchmark that reports an LSTM's numbers as an SSM's is worse than a crash.
- **No device-dependent constants in user-facing signatures.** Chunk sizes and grid limits are the library's problem to auto-tune.
- **The library never imports a training loop, a dataset, or an optimizer.**

## Tests

Every new block must pass `check_block`. Beyond that, the tests worth writing are the ones that catch axis bugs: rank-1 equivalence against the underlying 1-D module, permutation round-trips, and mask invariance on sparse lattices. See [PLAN.md](PLAN.md) for what each phase must prove before the next one starts.
