# torch-dimensions

N-dimensional sequence models for PyTorch — state-space models, RNNs, and transformers over arbitrary lattices, behind one API.

> **Status: pre-alpha.** The N-D machinery and the RNN family are built and tested — `Lattice`, `ScanPlan`, `AxialScan`, `LSTMND`, `GRUND`. SSM and attention families are not yet implemented. See [PLAN.md](PLAN.md) for build order and [DESIGN.md](DESIGN.md) for the architecture.

## Why

`torch.nn` ships `LSTM`, `GRU`, and `Transformer`, and nothing for state-space models. Anyone doing SSM work vendors code out of research repos written to reproduce one paper's table. The N-dimensional corner is emptier still: S4ND, Mamba-ND, and axial attention each ship their own incompatible axis bookkeeping, hardcoded to one rank.

They do not need to be separate implementations. Every one of them is the same object:

> **a 1-D mixer, plus a plan for sweeping it over an N-D lattice.**

| Model | 1-D mixer | Sweep strategy |
|---|---|---|
| Mamba-ND | Mamba-2 / Mamba-3 selective scan | sequential, one axis per layer, alternating direction |
| MDRNN / Grid-LSTM | `nn.LSTM` / `nn.GRU` | sequential, one axis per layer |
| S4ND | S4 FFT conv | separable — per-axis kernel, outer product |
| Axial Transformer | attention | per-axis kernel, contracted in turn |
| Factorized axial attention | factorized axial cross-attn | per-axis kernel, Kronecker contraction |

Write that abstraction once and N-D RNNs, N-D transformers, and N-D SSMs all fall out of it.

## Planned API

```python
import torch_dimensions as td

lattice = td.Lattice(shape=(32, 64, 64), names=("depth", "height", "width"))

model = td.MambaND(d_model=128, n_layers=12, lattice=lattice, time=True)
loss = model(x).pow(2).mean()
loss.backward()
```

Swap `MambaND` for `S4ND`, `LSTMND`, or `AxialTransformer` without changing anything else. Drop to `td.AxialScan(mixer=..., plan=...)` to compose your own, or drive the whole thing from YAML.

## Scope

**This is a composition layer.** Fused kernels come from `mamba-ssm`, `flash-linear-attention`, and `state-spaces/s4` — we adapt them, we do not reimplement them. What this library owns is the N-D structure, the sparse-lattice handling, the registry, and the `nn.Module` contract.

`torch` is the only required dependency; every kernel is an optional extra, so a CPU-only install gets a working library.

**Not in v0.1:** autoregressive stepping (forward-only), ranks ≥ 5 (untested), and training loops or datasets of any kind (permanently out of scope — this is a layer library).

## License

Apache-2.0
