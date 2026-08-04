---
license: apache-2.0
library_name: pytorch
tags:
  - state-space-models
  - s4
  - mamba
  - sequence-modeling
  - pytorch
  - benchmark
---

<!--
  This file is uploaded as the *root* README.md of
  https://huggingface.co/Celsia/torch-dimensions, so every relative path below
  is resolved from the Hub repo root, not from docs/. That is why the image
  reference reads docs/device-comparison.png and appears broken when this file
  is viewed here. Edit it here; publish with:
      hf upload Celsia/torch-dimensions docs/hf-card.md README.md
-->

# torch-dimensions

**An N-dimensional model is a 1-D mixer plus a plan for sweeping it over an
N-D lattice.** State-space models, RNNs, transformers and convolutions over
arbitrary lattices, behind one API.

- **Code:** https://github.com/triadastra/torch-dimensions
- **PyPI:** `pip install torch-dimensions`
- **License:** Apache-2.0, © 2026 Celsia Juilyn Fan

```python
import torch_dimensions as td

model = td.S4ND(d_model=64, n_layers=12, dim=2, shape=(32, 32))
loss = model(x).pow(2).mean()   # x: (B, T, 32, 32, 64)
loss.backward()                 # ordinary autograd; nothing custom to call
```

`td.S4`, `td.S4D`, `td.Mamba`, `td.Mamba2` and `td.Mamba3` run the **original
authors' code**, redistributed verbatim under `torch_dimensions/_vendor/` with
every patched line tagged and a manifest pinning upstream commits and sha256s.
`portable=True` selects a pure-torch build with a smaller dependency footprint.

---

## What is in this repository

Sixteen models trained identically on three devices, plus the artifacts needed
to check that the comparison means anything.

| directory | contents |
|---|---|
| `CUDA bench/` | 16 checkpoints trained on an **NVIDIA RTX 5090** (Blackwell, sm_120; torch 2.12.1+cu130), plus `cuda_check.txt` |
| `MPS bench/` | the same 16 on an **Apple Mac Studio, M1 Ultra** (Metal / MPS; torch 2.13.0) |
| `CPU bench/` | the same 16 on that Mac's **CPU** (arm64, torch 2.13.0) — the same-machine control for MPS |
| `CUDA agree/`, `MPS agree/`, `CPU agree/` | one forward and one backward from fixed weights, **no optimiser** — outputs and every gradient, saved as tensors |
| `init weights/` | the single set of starting weights every device loads |
| `docs/device-comparison.png` | the figure below, and its `.svg` |
| `AGREEMENT.md` | do the two devices compute the same thing? |
| `COMPARISON.md` | do they train to the same place, and how fast? |
| `*/SCORECARD.md` | per-machine model ranking, one column per question, no combined score |
| `BENCH-README.md`, `BENCHMARK-DESIGN.md` | the design, and what the comparison cannot show |

Each checkpoint is `<model>/weights.pt` (a `{"model": ..., "head": ...}` state
dict) beside `<model>/metrics.json`.

### Loading a checkpoint

```python
import json, torch, torch.nn as nn, torch_dimensions as td

lat = td.Lattice(shape=(6, 8), names=("h", "w"), valid=..., time=True)
model = td.S4D(32, 4, lat, d_input=1, d_state=16)
head = nn.Linear(32, 1)

blob = torch.load("CUDA bench/s4d_upstream_2d/weights.pt", weights_only=True)
model.load_state_dict(blob["model"])   # strict; tested in CI
head.load_state_dict(blob["head"])
```

The exact lattice and constructor for every entry is in
[`benchmarks/pretrain.py`](https://github.com/triadastra/torch-dimensions/blob/main/benchmarks/pretrain.py)'s
`ZOO`, and `tests/test_published_checkpoints.py` loads all 32 with
`strict=True` on every commit — these weights are a compatibility contract, not
a snapshot.

**These are not pretrained models to fine-tune.** They are small models
(12k–141k parameters) trained 300 steps on one synthetic task — a cumulative
sum along a lattice axis — for the purpose of comparing devices and
mechanisms. Use them to reproduce the comparison, not as a starting point.

---

## Evaluation and Device Comparison

![Evaluation and device comparison across CUDA, CPU and MPS](docs/device-comparison.png)

Sixteen models, three devices, two benchmarks, from one shared set of starting
weights on identical data. Regenerate with `python benchmarks/figure.py` — every
panel reads artifact directories that already exist, so the figure cannot drift
from `AGREEMENT.md` and `COMPARISON.md`.

| device | hardware | torch | artifacts |
|---|---|---|---|
| **CUDA** | NVIDIA RTX 5090 (Blackwell, sm_120) | 2.12.1+cu130 | `CUDA bench/`, `CUDA agree/` |
| **MPS** | Apple Mac Studio, M1 Ultra (Metal) | 2.13.0 | `MPS bench/`, `MPS agree/` |
| **CPU** | Apple M1 Ultra (arm64) | 2.13.0 | `CPU bench/`, `CPU agree/` |

**CPU and MPS are the same machine and the same torch build**, so a difference
between them is the device and nothing else. CPU-vs-CUDA crosses machines and
torch versions. Reading the pairs together separates what the hardware did from
what the software version did — neither pair alone can.

### What the throughput panel shows

Three findings, none of which is "the GPU is faster":

| model | CPU | MPS | CUDA | CUDA vs CPU |
|---|---|---|---|---|
| `mamba_upstream_3d` | 0.1 | 26.8 | 23.7 | **455×** |
| `mamba2_2d` | 0.1 | 5.6 | 24.3 | **285×** |
| `mamba_upstream_2d` | 0.2 | 40.1 | 23.8 | 95× |
| `tcn_2d_sparse` | 5.2 | 90.5 | 87.3 | 17× |
| `cnn_2d_sparse` | 12.7 | 106.5 | 104.4 | 8× |
| `mamba3_2d` | 5.3 | 14.6 | 21.0 | 3.9× |
| `s4_upstream_2d` | 39.4 | 39.0 | 28.3 | **0.7×** |
| `s4d_upstream_2d` | 42.7 | 46.9 | 32.7 | **0.8×** |

*training steps per second, 300 steps, batch 4*

**The upstream reference scans are the bottleneck, not the hardware.** Mamba-1
and Mamba-2 fall to 0.1–0.2 steps/s on CPU because `selective_scan_ref` and
`ssd_minimal_discrete` are sequential Python loops over sequence length. That
is what the fused CUDA kernels exist to avoid, and it is why a CPU pass over
this zoo takes hours while a GPU pass takes minutes.

**Our Mamba-3 transcription is 30–50× faster on CPU than the authors' Mamba-1
and Mamba-2 references** (5.3 steps/s against 0.1–0.2), because
`mixers/mamba3_compat.py` computes the recurrence as a chunked, vectorised
scan rather than a Python loop. Mamba-3 ships Triton-only upstream, so there
was no reference loop to copy — the transcription had to be written, and it
was written vectorised.

**The RTX 5090 loses to an M1 Ultra CPU on the small S4 models** (0.7–0.8×).
These are 13k–17k-parameter models on a 6×8 lattice: the axial sweep issues
many small sequential kernels, launch overhead dominates every one of them, and
the GPU never fills. That is a property of this benchmark's size rather than of
the hardware — a defensible throughput number needs the warmup-and-repeats
protocol designed in `BENCHMARK-DESIGN.md` and not yet
implemented.

## Results

### Do the two devices compute the same thing?

**Worst float32 output difference across every device pair: 3.11e-06.** Thirteen of sixteen models are
at or below **1e-06**. In float64 the devices agree to **2.2e-16** — the last
bit. Full per-model table in `AGREEMENT.md`.

Two corrections had to happen before that number meant anything, and both
generalise to anyone benchmarking across devices:

**CUDA does not run float32 by default.** `torch.backends.cudnn.allow_tf32`
ships as `True`, so cuDNN convolutions and RNNs execute in TF32 — 10 mantissa
bits against float32's 23. That accounted for the *entire* original 1.96e-04
gap, and explains why only LSTM, GRU, CNN and TCN were affected while
attention, Mamba and S4 sat at ~2e-07 throughout (`matmul.allow_tf32` ships as
`False`).

| model | TF32 on (torch's default) | TF32 off | + cuDNN off |
|---|---|---|---|
| `tcn_2d_sparse` | 1.16e-04 | **1.13e-07** | — |
| `cnn_2d_sparse` | 1.96e-04 | **2.34e-07** | — |
| `gru_2d_sparse` | 1.82e-04 | 3.13e-06 | **1.50e-07** |
| `lstm_2d_sparse` | 1.21e-04 | 2.66e-06 | **1.73e-07** |

The residual for the three RNNs is not precision but *algorithm* — cuDNN's
fused LSTM/GRU is a different implementation. That is left visible rather than
configured away, because the fused kernel is what runs.

**A fixed seed does not give identical S4 weights across platforms.**
`hippo.nplr` diagonalises with `torch.linalg.eigh`. Eigenvalues are unique and
matched across macOS and Linux to twelve decimals — which is exactly why this
hid, since `A_imag` looked perfect. Eigenvectors are fixed only up to a phase,
so `B` and `P` differed by a relative **1.5** and **0.53**: two valid S4
initialisations, two different models. The comparison was reporting 2.6e-01 for
the vendored S4D, unchanged in float64. With shared weights the same pair
agrees at 4e-07, and after 300 optimiser steps their losses agree to 8.9e-07.

**One gradient cannot meet a float32 bound, by construction.** The derivative
with respect to an SSM frequency sums oscillating terms that nearly cancel:
`A_imag` differs by **1.35e-04 in float32 and 4.87e-15 in float64**. Eleven
orders between the two is what cancellation looks like; a genuinely different
computation does not shrink when mantissa bits are added.

**MPS has no float64 at all**, so on Apple silicon float32 is the only column
there is.

### CUDA verification

`scripts/cuda_check.py` runs every CUDA claim the library makes as one file.
On the RTX 5090: **13 passed, 0 failed, 1 skipped**, and the full test suite
**1211 passed, 0 failed**. Highlights:

- `prefer_upstream` returns **True** on real hardware — the per-call dispatch
  between the authors' fused kernels and the portable path, verified for the
  first time.
- The vendored **S4 (DPLR)** agrees with CPU at **1.9e-07 including L=64**,
  where MPS lands exactly on the Nyquist pole — so the guard that makes S4 work
  on Metal is provably inert on CUDA.
- The rank-1 LSTM is still **bitwise identical** to its pre-norm residual under
  cuDNN.

**Still not established:** Mamba-3's PyTorch transcription has never been
compared against the Triton kernel it was transcribed from. `mamba-ssm` has no
sm_120 wheel and does not build against CUDA 13, so even on the 5090 the fused
Mamba entry points were never importable. The transcription is validated three
other ways — against an independently written recurrent form (3.6e-15 in
float64), a direct sum in the `trap → 1` limit (2.2e-16), and gradcheck — but
not against its own kernel. An Ampere or Ada card would close it.

### Reproducing

```bash
pip install "torch-dimensions[dev,upstream]"

# on each machine, once
python benchmarks/agreement.py --out "MPS agree" --init "init weights"
python benchmarks/pretrain.py  --out "MPS bench" --init "init weights"

# then, anywhere
python benchmarks/compare_agreement.py "MPS agree" "CUDA agree" --out AGREEMENT.md
python benchmarks/compare.py "MPS bench" "CUDA bench" --out COMPARISON.md
python benchmarks/figure.py --out docs/device-comparison.png

python scripts/cuda_check.py
```

Pass the **same** `--init` directory on every machine: the seed alone is not
enough, for the reason above.

---

## Attribution

Full third-party attribution is in
[`NOTICE`](https://github.com/triadastra/torch-dimensions/blob/main/NOTICE).
In short: the files under `_vendor/` are redistributed verbatim from
[state-spaces/s4](https://github.com/state-spaces/s4) and
[state-spaces/mamba](https://github.com/state-spaces/mamba) (both Apache-2.0)
with their licenses alongside and every patched line tagged; the portable
mixer mathematics derives from the same two repositories;
[CaFA](https://github.com/BaratiLab/CaFA) (MIT) contributed two kernel ideas,
independently implemented. `ScanPlan.paired()` produces the same schedule as
[Mamba-ND](https://github.com/jacklishufan/Mamba-ND)'s reference and was
verified against it — that repository states **no license**, so no code from it
is used or redistributed here.
