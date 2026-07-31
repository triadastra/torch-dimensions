# torch-dimensions

N-dimensional sequence models for PyTorch — state-space models, RNNs, and axial
attention over arbitrary lattices, behind one API.

```python
import torch_dimensions as td

model = td.S4ND(d_model=64, n_layers=12, dim=2, shape=(32, 32))
loss = model(x).pow(2).mean()  # x: (B, T, 32, 32, 64)
loss.backward()  # ordinary autograd; nothing custom to call
```

> **Status: 0.1.** Built and tested: `Lattice`, `ScanPlan`, the scan and kernel
> composition families, `LSTM`/`GRU`, `S4`/`S4D`/`Mamba` (portable, pure torch,
> verified against the upstream reference kernels), the data layer, the
> conformance suite, config/save/load, and the device suite (CPU, CUDA, MPS).
> See [DESIGN.md](DESIGN.md) for the architecture, [PLAN.md](PLAN.md) for what
> remains, and [DEBUG.md](DEBUG.md) for every bug found on the way and what
> caught it.

## Why

`torch.nn` ships `LSTM`, `GRU`, and `Transformer`, and nothing for state-space
models. The N-dimensional corner is emptier still: S4ND, Mamba-ND, and axial
attention each ship their own incompatible axis bookkeeping, hardcoded to one
rank. They do not need to be separate implementations — every one of them is
the same object:

> **a 1-D mixer, plus a plan for sweeping it over an N-D lattice.**

| Model | 1-D mixer | Method of multidimensionality |
|---|---|---|
| Mamba-ND | selective scan | `td.axial_scan` — one axis per layer, alternating direction |
| MDRNN / Grid-LSTM | `nn.LSTM` / `nn.GRU` | `td.axial_scan` |
| S4ND | S4 kernel conv | `td.axial_scan` over the axes |
| Axial Transformer | attention | `td.axial_attention` — per-axis kernels, contracted |
| CaFA / factorized axial | pooled attention | `td.cafa` — per-axis kernels, Kronecker contraction |
| Forecasting hybrids | any of the above along time | kernel across the lattice, mixer along time |

Write that abstraction once and N-D RNNs, N-D transformers, and N-D SSMs all
fall out of it — including on lattices where **not every cell exists**, which
no source implementation handles.

## The API in five lines each

**One class per model, 1-D and N-D.** No `LSTMND`: a lattice with no spatial
axes folds to the identity, so the 1-D case is the N-D case with nothing to do.

```python
td.Mamba(64, n_layers=12)  # a sequence model
td.Mamba(64, 12, lattice=lat)  # the same class, N-dimensional

# the explicit N-D name: dim is mandatory and checked against the lattice,
# and dim=1 is refused — one spatial axis is just Mamba, and code reading
# "MambaND" must not be running Mamba
td.MambaND(64, 12, dim=2, shape=(32, 32))
```

**The method of multidimensionality is one argument.** Registered names or your
own function, on equal footing:

```python
td.LSTM(64, 12, lattice=lat, method=td.axial_scan)  # the RNN sweeps every axis
td.LSTM(64, 12, lattice=lat, method=td.cafa)  # CaFA across space, RNN along time
td.S4ND(64, 12, dim=3, shape=s, method=td.axial_attention)
td.Mamba(64, 12, lat, method=my_traversal)  # yours, no registration needed
```

**Direction is a schedule, not a flag.** `ScanPlan` is data — printable,
serializable, testable — and bidirectionality is per-axis, so time stays causal
while space does not:

```python
plan = td.ScanPlan.cyclic(("time", "h", "w"), n_layers=12, bidirectional=("h", "w"))
td.Mamba(64, lattice=lat, plan=plan)  # .paired() is official Mamba-ND's schedule
```

**Lattices with absent cells are first-class.** Mark which cells exist and
their values can never influence any output — tested bitwise:

```python
lat = td.Lattice(shape=(14, 1263), names=("port", "commodity"), valid=observed, time=True)
```

**Real data in, one call.** Long-format rows to a lattice-shaped series, plus
windowing, a source protocol, and a collate that keeps metadata out of batches:

```python
table = td.data.from_table(coords, times, values, names=("store", "sku"))
ds = td.data.LatticeDataset(
    td.data.TensorSource(table.series, table.lattice),
    td.data.LatticeWindow(len(table), input_len=28, horizon=7),
)
```

**Config in, checkpoint out.** Everything above as plain data, and checkpoints
that rebuild their own model — validity mask included, outputs bitwise equal:

```python
model = td.build("model.yaml")  # or a dict; unknown keys are a hard error
model.save("run.td")
same = td.load("run.td")
```

## Correct on purpose

The test suite is the product as much as the models are:

- **Conformance suite as public API.** `td.testing.check_block(factory)` runs
  the same seven checks the library runs on itself — shapes, gradcheck, rank-1
  equivalence against the bare mixer, absent-cell inertia, axis-storage
  covariance — against *your* mixer or method.
- **Trainability is tested, not asserted.** `td.testing.check_trainable`
  fits a task that genuinely needs axial mixing, on fresh data, scored held-out,
  with a negative control that must fail.
- **Verified against the sources.** The S4D kernel is bitwise-identical to
  upstream's; the full S4 (DPLR) kernel matches upstream at 3e-8 and a dense
  state-space reference at machine precision; the Mamba scan matches
  `selective_scan_ref` at 1e-6.
- **Portable by construction.** Pure torch throughout: CPU, CUDA, and Apple
  Silicon (MPS) — the device suite runs against whatever accelerator exists.
  Fused CUDA kernels are a planned fast path, not a requirement.
- Every bug found in building this — 18 so far — is documented in
  [DEBUG.md](DEBUG.md) with what caught it, and the citations in that file are
  themselves tested.

## Install

```bash
pip install torch-dimensions
```

`torch>=2.4` is the only required dependency. A CPU-only machine gets a fully
working library; `pip install "torch-dimensions[mamba]"` adds the fused CUDA
kernels where available.

## Scope

**This is a composition layer.** What it owns: the N-D structure, the sparse
lattices, the schedules, the registry, and the `nn.Module` contract. Training
loops, optimizers, and losses are yours, permanently — `td.data` builds
lattices from data, it never trains on them.

**Not yet:** autoregressive stepping (forward-only for now), ranks ≥ 5
(machinery is rank-generic but untested there), fused-kernel fast paths.

## License

Apache-2.0. The S4/S4D/Mamba mixer mathematics derives from
[state-spaces/s4](https://github.com/state-spaces/s4) and
[state-spaces/mamba](https://github.com/state-spaces/mamba) (both Apache-2.0);
see `mixers/ssm.py` for what was carried over and what deliberately was not.
