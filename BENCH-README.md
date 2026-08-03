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

The headline number was 1.96e-04 and it was **not the hardware**. CUDA does
not run float32 by default: `torch.backends.cudnn.allow_tf32` ships as `True`,
so every cuDNN convolution and RNN runs in TF32 — 10 mantissa bits where
float32 has 23. (`matmul.allow_tf32` ships as `False`, which is exactly why
attention, Mamba and S4 were never affected and sat at ~2e-07 all along.)

Comparing MPS float32 against CUDA TF32 and calling the difference a device
difference is a category error. Measured CPU-against-CUDA on the 5090, so only
the device varies:

| model | TF32 on (torch's default) | TF32 off | + cuDNN off |
|---|---|---|---|
| `tcn_2d_sparse` | 1.16e-04 | **1.13e-07** | — |
| `cnn_2d_sparse` | 1.96e-04 | **2.34e-07** | — |
| `gru_2d_sparse` | 1.82e-04 | 3.13e-06 | **1.50e-07** |
| `lstm_2d_sparse` | 1.21e-04 | 2.66e-06 | **1.73e-07** |
| `lstm_3d` | 1.44e-04 | 2.57e-06 | **1.77e-07** |
| the other eleven | ~2e-07 | ~2e-07 | — |

`agreement.py` therefore runs with **`--tf32 off` by default**: a numerical
comparison has to compare like with like. `pretrain.py` keeps TF32 **on**,
because that is what a user gets and what the speed numbers should reflect.
Both record the setting in their JSON.

### Where that leaves the float32 bound

**Worst output difference, MPS vs CUDA: 3.11e-06** — down from 1.96e-04.
Thirteen of the sixteen models are at or below **1e-06**; the three at
2.5–3.1e-06 are the cuDNN RNNs, and that residual is not precision but
*algorithm* — cuDNN's fused LSTM/GRU is a different implementation, and
disabling it brings them to 1.5–1.8e-07 like everything else. That is left
visible rather than configured away, since the fused kernel is what runs.

**Gradients are judged by the same bound and one class does not meet it.** The
derivative with respect to an SSM frequency sums oscillating terms that nearly
cancel, so its *relative* error in float32 is large by construction:
`A_imag` differs by **1.35e-04 in float32 and 4.87e-15 in float64**. Eleven
orders between the two is what cancellation looks like; a genuinely different
computation does not shrink when you add mantissa bits. No setting fixes this,
and none should — the quantity is ill-conditioned, not the arithmetic.

**In float64 the devices agree to the last bit**: 2.2e-16 to 2.8e-16, and
4.9e-15 for the cancellation-heavy SSM gradients. Mamba stops at 1.5e-09
because its scan casts internally (`A_log.float()`) for fp16 stability — an
upstream choice, not a device difference.

**MPS cannot take the float64 route at all.** Metal has no float64, so on
Apple silicon float32 is the only column there is, and 3.11e-06 is the answer.

**The float32 result is not contaminated by the torch version skew.** The two
machines run torch 2.13.0 and 2.12.1. Same-process CPU-vs-CUDA reproduces the
cross-machine MPS-vs-CUDA numbers to three digits, so at float32 there is no
headroom for the version to show. It appears only in float64, where the
cross-machine comparison flattens at ~5e-07 while the same-process one reaches
2e-16 — that residual is the torch version, not the hardware.

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
