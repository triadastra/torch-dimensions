# Device benchmarks — CUDA vs MPS

> Three benchmarks, not one, and a corrected training recipe. The reasoning is
> in [BENCHMARK-DESIGN.md](BENCHMARK-DESIGN.md); the short version is that
> "do the devices agree?" and "how fast is it?" and "does it train the same?"
> are different questions, and one run answering all three answers none of
> them cleanly.
>
> | bench | command | question |
> |---|---|---|
> | **A — agreement** | `benchmarks/agreement.py` | do the devices compute the same thing? No optimiser, so nothing amplifies |
> | **B — training** | `benchmarks/pretrain.py` | does it train to the same place? |
> | **scorecard** | `benchmarks/scorecard.py` | how do the models compare on this task |
>
> Training now uses the recipe the upstream authors specify — AdamW,
> `betas=(0.9, 0.95)`, `td.param_groups` reading their own `_optim` and
> `_no_weight_decay` tags, linear warmup into cosine decay, gradient clipping.
> The earlier run used plain Adam and ignored both sets of tags, which is why
> it needed hand-picked per-family learning rates; one rate now trains all
> sixteen.

Two runs of the same sixteen models, trained identically on two machines:

| directory | hardware | what runs there |
|---|---|---|
| **`MPS bench`** | Apple Mac Studio, M1 Ultra (Metal / MPS) | the portable path everywhere; the vendored models take upstream's own *reference* implementations, because Triton needs CUDA |
| **`CUDA bench`** | NVIDIA RTX 5090 | the portable path where it is asked for, and the authors' **fused** kernels for the vendored models |

## What makes them comparable

Every model is built on **CPU** under one fixed seed and only then moved to
the device, and every batch is drawn on CPU from a seeded generator, so both
machines see bit-identical data.

**The seed is not enough for the weights, and this was found the hard way.**
S4 and S4D diagonalise the HiPPO matrix with `torch.linalg.eigh`. Eigenvalues
are unique and agreed across the two machines to every digit printed — which
is exactly why the problem hid, since `A_imag` looked perfect. Eigenvectors are
fixed only up to a phase, and macOS Accelerate and Linux LAPACK each return a
different one. `B` and `P` are projections through those vectors, so on the
Mac Studio and the 5090 they differed by a relative **1.5** and **0.53**.

Both initialisations are valid S4 models. They are not the *same* model, and
the first CUDA run therefore reported a 2.6e-01 output difference for the
vendored S4D — unchanged in float64, which is precisely the signature of a
different kernel — when nothing about the kernel was involved.

So one set of starting weights is written once and loaded by every machine
(`--init`, see [benchmarks/init_weights.py](benchmarks/init_weights.py)). With
that in place the same pair agrees at **4e-07**. Pass the same directory on
both sides:

```bash
python benchmarks/pretrain.py --out "MPS bench"  --init "init weights"   # writes
python benchmarks/pretrain.py --out "CUDA bench" --init "init weights"   # loads
python benchmarks/compare.py "MPS bench" "CUDA bench" --out COMPARISON.md
```

The manifest records `weights_from` as `written`, `loaded` or `seed` for every
model, so a run can always say which of the two assumptions its numbers rest
on — the difference is invisible in the losses otherwise.

## How close can two devices actually get?

Three separate questions hide inside "do they agree", and separating them
moves the answer by nine orders of magnitude. Measured **in one process on the
5090**, so the torch version and the platform are held fixed and only the
device varies:

| model | float32 | float64 |
|---|---|---|
| `lstm_2d_sparse` | 1.21e-04 | **2.69e-16** |
| `cnn_2d_sparse` | 1.96e-04 | **2.83e-16** |
| `s4d_upstream_2d` | 1.77e-07 | **2.20e-16** |
| `mamba_upstream_2d` | 1.28e-07 | 1.48e-09 |

- **In float64 the devices agree to about one part in 1e16** — the last bit.
  So the differences are float non-associativity and nothing else: the same
  terms summed in a different order.
- **In float32, 1e-7 is not reachable and should not be asked for.** float32
  machine epsilon is 1.19e-07, so "under 1e-7" means bit-identical output from
  cuDNN, Metal and a pure-torch loop. Note that `s4d_upstream` and
  `mamba_upstream` are already at 1.3–1.8e-07 — they are *at* the floor, not
  above it. The ~1e-04 rows are four to six layers of that floor compounding.
- **`mamba_upstream` stops at 1.5e-09** rather than 1e-16, because its scan
  casts internally (`A_log.float()`) — a deliberate upstream choice for fp16
  stability, not a device difference.

Two consequences worth stating plainly:

**MPS cannot take this route.** Metal has no float64 at all, so on Apple
silicon the float32 column is the only column, and 1e-04 is where it ends. The
sub-1e-16 numbers above are CPU-vs-CUDA.

**The float32 result is not contaminated by the version skew.** The two
machines run torch 2.13.0 and 2.12.1. Same-process CPU-vs-CUDA gives 1.21e-04
and 1.96e-04 for the LSTM and the CNN — the same three digits as the
cross-machine MPS-vs-CUDA comparison. At float32 there is no headroom for the
version to show; it only becomes visible in float64, where the cross-machine
comparison flattens out at ~5e-07 while the same-process one reaches 2e-16.
That residual is the torch version, not the hardware.

## What the comparison can and cannot show

It **cannot** show bitwise agreement, and a run that claimed to would be
suspect. Different devices reduce in different orders; cuDNN, Metal and a
pure-torch loop are three implementations of one formula. Float addition is
not associative, so a difference at the last bit is expected and grows as
training proceeds.

It **can** show three things worth knowing:

1. **Do the two machines agree to the precision the arithmetic allows?**
   Reported as the first step at which the losses diverge, and as the relative
   distance between the two sets of trained weights.
2. **Do they train to the same place?** Final losses side by side. Two runs
   can diverge numerically and still land together, which is the outcome that
   says the model is well conditioned.
3. **How much faster is the 5090?** Steps per second, per model.

Learning rates are chosen so that *every* model converges. That is a
requirement rather than a nicety: a diverging run turns a 1e-7 arithmetic
difference into an arbitrary one within a few steps, and comparing two chaotic
trajectories measures the chaos, not the devices.

## The rows that matter most

The vendored models — `s4d_upstream_2d`, `s4_upstream_2d`,
`mamba_upstream_2d`, `mamba_upstream_3d`, `mamba2_2d`, `mamba3_2d` — are not
really testing CUDA against MPS. On CUDA they take the authors' fused kernels
and on MPS they take the reference path, so those rows are **fused against
reference**. That is the comparison PLAN.md fixes in advance as the rule for
any fast path: *the portable path is the reference, and the fused path must
agree with it.*

`mamba3_2d` carries the most weight of all. Mamba-3 ships upstream as Triton
kernels alone, so `torch_dimensions/mixers/mamba3_compat.py` is **our**
transcription of the recurrence. On CPU and MPS it has been checked against a
second independently written form (3e-15 in float64), a third direct
summation (2e-16) and gradcheck — but never against the kernel it was
transcribed from, because no machine here can run one. The CUDA run is the
first time those two meet.

Expect that row to agree less tightly than the others, and for a stated
reason: their kernel computes in bfloat16 with PTX `cos`/`sin`/`tanh`
approximations, ours in float32 with exact library functions. Agreement at
bfloat16's own resolution is the most that can be asked; agreement at float
epsilon would mean something was wrong with the test.

## One row that does not learn, on purpose

`transformer_flatten_2d` never improves — it starts at 2.70 and ends there,
at every learning rate tried. That is not a broken run: `td.flatten` supplies
no positional information, so joint attention over the flattened lattice is a
**set function**, and the cells are indistinguishable to it. A cumulative sum
along an axis depends on *where* each cell is, so this configuration cannot
learn the task in principle.

Measured rather than argued: on a dense lattice, permuting the `w` axis of the
input and un-permuting the output changes the result by 9.5e-07 — float noise.
(On the sparse lattice the mask breaks that symmetry, which is why the effect
is only visible with the mask removed; the pattern of present cells is
information, just not useful information.) `td.ViT` is the flatten-family
model meant for real use and carries a positional embedding of its own.

The row is kept because it is still a valid *device* comparison: both machines
run the same non-learning model on the same data, so the numerical columns
mean exactly what they mean elsewhere. Only its loss column should be read as
"this model cannot do this task" rather than "this device trained worse".

## Contents of each directory

```
<dir>/manifest.json          device, versions, seed, and every model's summary
<dir>/<model>/metrics.json   loss curve, timings, weight norms, probe outputs
<dir>/<model>/weights.pt     trained weights (CPU tensors, loadable anywhere)
```

`weights.pt` holds CPU tensors regardless of the training device, so a
checkpoint from either machine loads on either machine.
