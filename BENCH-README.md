# Device benchmarks — CUDA vs MPS

Two runs of the same sixteen models, trained identically on two machines:

| directory | hardware | what runs there |
|---|---|---|
| **`MPS bench`** | Apple Mac Studio, M1 Ultra (Metal / MPS) | the portable path everywhere; the vendored models take upstream's own *reference* implementations, because Triton needs CUDA |
| **`CUDA bench`** | NVIDIA RTX 5090 | the portable path where it is asked for, and the authors' **fused** kernels for the vendored models |

## What makes them comparable

Every model is built on **CPU** under one fixed seed and only then moved to
the device, and every batch is drawn on CPU from a seeded generator. So both
machines start from bit-identical weights and see bit-identical data, and
every difference in the results is produced by the arithmetic rather than by
initialisation or data order.

Reproduce either side with:

```bash
python benchmarks/pretrain.py --out "MPS bench"     # on the Mac Studio
python benchmarks/pretrain.py --out "CUDA bench"    # on the 5090
python benchmarks/compare.py "MPS bench" "CUDA bench" --out COMPARISON.md
```

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
