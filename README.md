# torch-dimensions

N-dimensional models for PyTorch — state-space models, RNNs, transformers and
convolutions over arbitrary lattices, behind one API.

```python
import torch_dimensions as td

model = td.S4ND(d_model=64, n_layers=12, dim=2, shape=(32, 32))
loss = model(x).pow(2).mean()  # x: (B, T, 32, 32, 64)
loss.backward()  # ordinary autograd; nothing custom to call
```

> **Status: 0.3.** Built and tested: `Lattice`, `ScanPlan`, the scan, kernel
> and joint composition families, `LSTM`/`GRU`, `S4`/`S4D`/`Mamba` (portable,
> pure torch, verified against the upstream reference kernels — plus the
> authors' originals vendored verbatim, see below),
> `Transformer`/`ViT`, `CNN`/`TCN`, the data layer, the conformance suite,
> config/save/load, the architecture viewer, and the device suite (CPU, MPS;
> CUDA is untested — see [docs/cuda-checklist.md](docs/cuda-checklist.md)).
> [RESULTS.md](RESULTS.md) has the reproductions, [BENCHMARKS.md](BENCHMARKS.md)
> the measured costs, [LTI.md](LTI.md) which mixers commute across axes and
> which do not, [DESIGN.md](DESIGN.md) the architecture, [PLAN.md](PLAN.md)
> what remains, and [DEBUG.md](DEBUG.md) every bug found on the way and what
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
| Axial Transformer | attention | `td.axial_scan` — attention sweeps each axis (`td.Transformer`) |
| Axial Transformer (factorized) | attention | `td.axial_attention` — per-axis kernels, contracted |
| CaFA / factorized axial | pooled attention | `td.cafa` — per-axis kernels, Kronecker contraction |
| ViT | attention | `td.flatten` — every cell attends to every cell, no factorization |
| Separable CNN | 1-D convolution | `td.axial_scan` — provably one N-D conv with a rank-1 kernel |
| TCN, N-D | dilated causal conv | `td.axial_scan` — dilation doubles per *axis* |
| Forecasting hybrids | any of the above along time | kernel across the lattice, mixer along time |

The ViT and convolution rows are what make that claim more than a slogan. A
convolutional network is not a sequence model — no state, no direction, no
"so far" — and a Vision Transformer's whole trick is *not* factorizing the
grid at all. Both fit without either side bending, which is the evidence that
the abstraction is about lattices rather than about sequences.

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
td.Transformer(64, 12, lat)  # attention as the swept mixer — the other axial transformer
td.Transformer(64, 12, lat, method=td.flatten)  # every cell attends to every cell
```

**Convolutions and vision transformers, same abstraction.** Two families that
had to be added without special cases, or the claim above was empty:

```python
td.CNN(64, 6, lat)  # separable N-D conv: provably one conv with a rank-1 kernel
td.TCN(64, 6, lat)  # causal, dilated — and the dilation doubles per *axis*
td.receptive_field(td.TCN(64, 6, lat))
# {'h': {'span': 13, 'size': 32, 'covers': False, 'layers': 2}, ...}  <- it cannot
# see across the lattice; the question no N-D paper answers, answered before training

vit = td.ViT(192, 6, image=(32, 32), patch=4, in_channels=3)  # joint attention
axial_vit = td.ViT(192, 6, image=(32, 32), patch=4, method=td.axial_scan)
# identical parameter counts; one argument apart
```

Which mixers this composition is *safe* to reorder is a measured question, not
an assumed one — [LTI.md](LTI.md) has the table, and the answer is narrower
than the folklore.

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
lat = td.Lattice(shape=(12, 6), names=("station", "pollutant"), valid=observed, time=True)
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
model.save("run.td")  # or "run.safetensors", which cannot execute code
same = td.load("run.td")
```

**Look at it.** The viewer ships inside the wheel — no node, no build step:

```python
td.viz.show(model)  # a model, a spec dict, a spec JSON, or a checkpoint
```

The lattice is rendered as cubes with absent cells genuinely absent, the sweep
wavefront animates per layer, and rank ≥ 4 stacks dimensionally. Train under it
with `examples/viewer_live.py` and the panel gets run controls and a loss
curve. See [VIEWER.md](VIEWER.md).

## Correct on purpose

The test suite is the product as much as the models are:

- **Conformance suite as public API.** `td.testing.check_block(factory)` runs
  the same seven checks the library runs on itself — shapes, gradcheck, rank-1
  equivalence against the bare mixer, absent-cell inertia, axis-storage
  covariance — against *your* mixer or method.
- **Trainability is tested, not asserted.** `td.testing.check_trainable`
  fits a task that genuinely needs axial mixing, on fresh data, scored held-out,
  with a negative control that must fail.
- **The N-D composition is verified against a published N-D method.** S4ND
  does not sweep — it outer-products one kernel per axis and applies a single
  N-D FFT. Our one-axis-per-layer sweep reproduces that operator to **1.8e-17**
  at ranks 2 and 3, with a negative control. A plan of sweeps reproducing
  exactly a model never written as a sweep is the strongest evidence the
  premise gets.
- **Verified against the sources.** The S4D kernel is bitwise-identical to
  upstream's; the full S4 (DPLR) kernel matches upstream at 3e-8 and a dense
  state-space reference at machine precision; the Mamba scan matches
  `selective_scan_ref` at 1e-6.
- **Portable by construction, and now measured on all three.** Pure torch
  throughout, so the device suite runs against whatever accelerator exists —
  verified on CPU, Apple Silicon (MPS) **and CUDA**: the full suite passes on an
  RTX 5090 (1211 passed, 0 failed) and every CUDA claim the library makes runs
  as one file, [`scripts/cuda_check.py`](scripts/cuda_check.py), reporting 13
  passed / 0 failed. The two devices agree to **3.11e-06 in float32** and
  2.2e-16 in float64. See [docs/cuda-checklist.md](docs/cuda-checklist.md) for
  the results and the one gap that remains.
- **Reproductions and benchmarks, with hardware attached.** sMNIST at 99.53%
  with `td.S4D` ([RESULTS.md](RESULTS.md)); the fold, the families and the
  factorization crossover measured rather than asserted
  ([BENCHMARKS.md](BENCHMARKS.md)), including the two results that came out
  against the design's own predictions.
- Every bug found in building this — 25 so far, including several found in this
  library's own examples and shipped artifacts — is documented in
  [DEBUG.md](DEBUG.md) with what caught it, and the citations in that file are
  themselves tested.

## Guides

- **[Adding a mixer](docs/adding-a-mixer.md)** — the extension point that
  matters most: a 1-D sequence model becomes an N-D one by satisfying one shape
  contract. Worked example, conformance report, mistakes to expect.
- **[Adding an nd_method](docs/adding-a-method.md)** — changing *how* the axes
  are handled, including the two bugs the conformance suite found in this
  project's own example strategy.
- **[CUDA verification checklist](docs/cuda-checklist.md)** — every CUDA claim
  as one runnable file, and what it reported on an RTX 5090.

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

**Not yet:** autoregressive stepping (forward-only for now — designed as
[PLAN.md](PLAN.md) Phase 12 for v0.2: a `step`/`init_state` protocol with a
parallel-vs-sequential equivalence check), fused-kernel fast paths, multi-head
axial kernels.

**Rank** is not a limit: both composition families pass the full conformance
suite at ranks 5 and 6 (a rank-6 lattice is 5,040 cells), and the fold
round-trips are fuzzed there too. Nothing in the machinery counts axes.

## The default IS the original authors' code

The reference implementations of S4/S4D and Mamba ship inside the package,
**byte-for-byte** — and they are what `td.S4`, `td.S4D`, and `td.Mamba` run
by default. For S4 that means the pipeline their `train.py` actually runs
(`S4Block`, its kernels, HiPPO, the S4ND layer), not a standalone re-export;
for Mamba, the reference block and the authors' own pure-torch selective
scan:

```python
model = td.Mamba(64, n_layers=6, lattice=lat)  # the authors' Mamba block, verbatim
s4d = td.S4D(64, 8, lattice=lat)  # their S4Block(mode='diag'), verbatim

td.Mamba2(64, 6, lattice=lat)  # or td.Mamba(..., version=2) — the SSD block
td.Mamba3(64, 6, lattice=lat)  # or version=3 — rotary state, trapezoidal rule

light = td.S4D(64, 8, lattice=lat, portable=True)  # our pure-torch build, zero extra deps
```

**Which implementation runs is decided per call.** A CUDA tensor plus an
importable Triton kernel runs the authors' fused kernel; anything else — no
Triton, a failed import, CPU or MPS tensors — runs the torch path, so a
machine that merely *has* the CUDA packages installed never sends a CPU
tensor into a CUDA kernel. `TD_FORCE_TORCH_KERNELS=1` forces the torch path.

Mamba-3 is the one model whose *scan* is not the authors' code off GPU: they
ship it only as Triton, with no pure-torch reference, so
`mixers/mamba3_compat.py` transcribes the recurrence — which is why that
mixer is `Mamba3Mixer` and not `Upstream…`. It is checked against a second
independently written form (3e-15, float64), a third direct sum in a limiting
case (2e-16), and gradcheck; it is *not* checked against their kernel, and
the package says so rather than implying otherwise.

The originals' own dependencies (einops, numpy, scipy, hydra-core, omegaconf)
are **installed on first use** — never at import, never for `portable=True`,
announced before installing, and disabled by `TD_NO_AUTO_INSTALL=1` (which
turns the moment into an error with the manual command,
`pip install 'torch-dimensions[upstream]'`). The flag is recorded in each
model's config, so checkpoints rebuild the implementation they were trained
with — including pre-flag checkpoints, which rebuild portable.

`torch_dimensions/_vendor/` holds the vendored subtree with its directory
structure intact and a manifest pinning the upstream commits and hashes.
Exactly two s4 files are patched (`utils/train.py`, guarding training-only
imports; `kernels/ssm.py`, a Nyquist-pole guard that makes the DPLR kernel
run on MPS and is verified bit-for-bit inert on CPU/CUDA) and two mamba
files (import paths plus one dispatch line); every changed line is tagged
`torch-dimensions patch`, pristine `.orig` copies ship beside patched files —
and CI fails if any untagged difference exists, so "identical to the
original" is a checked property, not a promise. The S4 side even constructs
itself through upstream's own hydra registry, exactly as their configs do.
Off GPU, the Mamba block runs the authors' own `selective_scan_ref`; on CUDA
with `mamba-ssm` installed it picks up the fused kernels exactly as upstream
intends. **Everything vendored runs on CPU, CUDA and MPS** (CPU-vs-MPS ≤ 1e-6
across the components; Mamba at 3e-8). Needs
`pip install 'torch-dimensions[upstream]'` (einops, numpy, scipy, hydra-core,
omegaconf — the originals' own imports).

The portable mixers (`td.S4D`, `td.Mamba` defaults) remain pure-torch, and
the two agree numerically — the pipeline's S4D kernel matches ours **bitwise**
with shared parameters, and that agreement is itself a CI test
(`tests/test_vendored.py`).

## Evaluation and Device Comparison

![Evaluation and device comparison across CUDA, CPU and MPS](docs/device-comparison.png)

*Regenerate with `python benchmarks/figure.py`. Every panel reads the artifact
directories in this repository — nothing in the figure is computed for it, so
the picture cannot drift from [AGREEMENT.md](AGREEMENT.md) and
[COMPARISON.md](COMPARISON.md).*

Sixteen models, three devices, two benchmarks, from one shared set of starting
weights on identical data.

| device | hardware | torch | artifacts |
|---|---|---|---|
| **CUDA** | NVIDIA RTX 5090 (Blackwell, sm_120) | 2.12.1+cu130 | [`CUDA bench/`](CUDA%20bench), [`CUDA agree/`](CUDA%20agree) |
| **MPS** | Apple Mac Studio, M1 Ultra (Metal) | 2.13.0 | [`MPS bench/`](MPS%20bench), [`MPS agree/`](MPS%20agree) |
| **CPU** | Apple M1 Ultra (arm64) | 2.13.0 | [`CPU bench/`](CPU%20bench), [`CPU agree/`](CPU%20agree) |

**CPU and MPS are the same machine and the same torch build**, so a difference
between them is the device and nothing else. CPU-vs-CUDA crosses machines and
torch versions. Reading the pairs together is what separates what the hardware
did from what the software version did — neither pair alone can.

### What it shows

**Worst float32 output difference across every device pair: 3.11e-06.**
Thirteen of sixteen models are at or below **1e-06**. In float64, measured in
one process so only the device varies, they agree to **2.2e-16** — the last
bit. The devices compute the same thing; float32 has nowhere to put the
agreement.

The three models above the bound are the cuDNN RNNs, and their residual is
*algorithm*, not precision: cuDNN's fused LSTM/GRU is a different
implementation, and disabling it brings them to 1.5–1.8e-07 like everything
else. That is left visible because the fused kernel is what actually runs.

One class of gradient cannot meet a float32 bound and should not be expected
to. The derivative with respect to an SSM frequency sums oscillating terms
that nearly cancel: `A_imag` differs by **1.35e-04 in float32 and 4.87e-15 in
float64**. Eleven orders between the two is what cancellation looks like — a
genuinely different computation does not shrink when mantissa bits are added.

### Two corrections were needed first, and both generalise

**CUDA does not run float32 by default.** `torch.backends.cudnn.allow_tf32`
ships as `True`, so cuDNN convolutions and RNNs execute in TF32 — 10 mantissa
bits against float32's 23. That accounted for the *entire* original 1.96e-04
gap, and explains why only LSTM, GRU, CNN and TCN were affected while
attention, Mamba and S4 sat at ~2e-07 throughout. Turning it off moves
`tcn_2d_sparse` by 1024x and `cnn_2d_sparse` by 839x.

**A fixed seed does not give identical S4 weights across platforms.**
`hippo.nplr` diagonalises with `torch.linalg.eigh`: eigenvalues are unique and
matched across macOS and Linux to twelve decimals — which is why this hid,
since `A_imag` looked perfect — but eigenvectors are fixed only up to a phase,
so `B` and `P` differed by a relative **1.5** and **0.53**. Two valid S4
initialisations; two different models. Both benchmarks now load one shared set
of weights via `--init`.

### On the throughput panel

The 5090 wins where there is real matmul work (`mamba2_2d` 4.35x, CaFA
attention 2.21x) and *loses* on the small recurrent models. These are
12k–141k-parameter models on a 6x8 lattice: the axial sweep issues many small
sequential kernels, launch overhead dominates, and the GPU never fills. That
is a property of this benchmark's size, not of the hardware — a defensible
throughput number needs the warmup-and-repeats protocol designed in
[BENCHMARK-DESIGN.md](BENCHMARK-DESIGN.md) and not yet implemented.

### Every artifact

| artifact | what it is |
|---|---|
| [`CUDA bench/`](CUDA%20bench) | 16 trained checkpoints — RTX 5090 · [`cuda_check.txt`](CUDA%20bench/cuda_check.txt) |
| [`MPS bench/`](MPS%20bench), [`CPU bench/`](CPU%20bench) | the same 16 on the Mac Studio's GPU and its CPU |
| [`CUDA agree/`](CUDA%20agree), [`MPS agree/`](MPS%20agree), [`CPU agree/`](CPU%20agree) | one forward + backward from fixed weights, no optimiser |
| [`init weights/`](init%20weights) | the shared starting weights every machine loads |
| [`docs/device-comparison.png`](docs/device-comparison.png) | the figure above · [`.svg`](docs/device-comparison.svg) · [`benchmarks/figure.py`](benchmarks/figure.py) |
| [**AGREEMENT.md**](AGREEMENT.md) | do the two devices compute the same thing? |
| [**COMPARISON.md**](COMPARISON.md) | do they train to the same place, and how fast? |
| `*/SCORECARD.md` | per-machine model ranking, one column per question, no combined score |
| [**BENCH-README.md**](BENCH-README.md) | the design, and what the comparison cannot show |
| [**BENCHMARK-DESIGN.md**](BENCHMARK-DESIGN.md) | why three benchmarks rather than one |

All of it is mirrored on **Hugging Face** at
[`Celsia/torch-dimensions`](https://huggingface.co/Celsia/torch-dimensions),
where the checkpoints can be pulled individually:

```python
from huggingface_hub import hf_hub_download
import torch
path = hf_hub_download("Celsia/torch-dimensions", "CUDA bench/s4d_upstream_2d/weights.pt")
blob = torch.load(path, weights_only=True)   # {"model": ..., "head": ...}
```

The card there is [docs/hf-card.md](docs/hf-card.md) in this repository, so the
two stay in step.

### Reproduce

Pass the **same** `--init` directory on every machine — the seed alone is not
enough, for the reason above. The first run writes it; the rest load it.

```bash
# on each machine, once
python benchmarks/agreement.py --out "MPS agree" --init "init weights"
python benchmarks/pretrain.py  --out "MPS bench" --init "init weights"

# then, anywhere
python benchmarks/compare_agreement.py "MPS agree" "CUDA agree" --out AGREEMENT.md
python benchmarks/compare.py "MPS bench" "CUDA bench" --out COMPARISON.md
python benchmarks/scorecard.py "CUDA bench" --out "CUDA bench/SCORECARD.md"
python benchmarks/figure.py --out docs/device-comparison.png
```

Every CUDA claim the library makes runs as one file:

```bash
python scripts/cuda_check.py
```

On the RTX 5090 that reports **13 passed, 0 failed, 1 skipped** — the skip is
Mamba-3's Triton kernel, which needs `mamba-ssm` and has no sm_120 wheel. The
full output is checked in at [`CUDA bench/cuda_check.txt`](CUDA%20bench/cuda_check.txt),
and the reasoning behind each check is in
[docs/cuda-checklist.md](docs/cuda-checklist.md).

## License

Apache-2.0, © 2026 Celsia Juilyn Fan. [NOTICE](NOTICE) carries the full
third-party attribution; the short version:

- The reference files under `torch_dimensions/_vendor/` are **redistributed
  verbatim** from [state-spaces/s4](https://github.com/state-spaces/s4) and
  [state-spaces/mamba](https://github.com/state-spaces/mamba), both
  Apache-2.0, with their licenses alongside and every patched line tagged.
- The portable S4/S4D/Mamba mixer mathematics **derives from** the same two
  repositories. `mixers/ssm.py` states what was carried over and what
  deliberately was not.
- Two kernel options (`qk_norm`, `kernel_residual`) are **ideas taken from**
  [BaratiLab/CaFA](https://github.com/BaratiLab/CaFA) (MIT), independently
  implemented.
- `ScanPlan.paired()` produces the same per-layer schedule as
  [Mamba-ND](https://github.com/jacklishufan/Mamba-ND)'s reference
  implementation, verified against it. That repository states **no license**,
  so no code from it is used or redistributed here — see [NOTICE](NOTICE),
  which states precisely what that verification does and does not establish.
