# A better set of benchmarks

Notes from reading what the upstream authors actually do, and what the current
benchmark gets wrong as a result. The short version: the existing run measures
*a misconfigured model on an unmeasured clock*, and those are two separable
problems that want two separate benchmarks rather than one.

## What the research says

**Optimizer.** State-space parameters are not ordinary weights, and every
upstream implementation says so in the parameters themselves:

| repo | tag | on what |
|---|---|---|
| `state-spaces/s4` | `param._optim = {"lr": 1e-3, "weight_decay": 0.0}` | the SSM kernel's `A`, `B`, `C`, `dt` |
| `state-spaces/mamba` | `param._no_weight_decay = True` | `A_log`, `D`, `dt_bias` |

Both are in this repository already, in the vendored files. Neither does
anything unless the optimizer reads them, and
`AdamW(model.parameters(), lr=...)` — the obvious line — does not. `A` sets a
decay rate and `dt` a timescale, both inside an exponential; weight decay on
them is not regularisation but a change to the dynamics, and a rate tuned for
a projection matrix walks them out of the stable region.

s4's own `example.py` builds the optimizer by splitting on exactly this tag,
and schedules with `CosineAnnealingLR`. The Mamba-family papers report AdamW
with `betas=(0.9, 0.95)`, weight decay 0.1, gradient clipping at 1.0, and a
linear warmup into cosine decay, with the SSM rate capped separately at 1e-3.

**Timing.** Published methodology is warmup iterations that are not counted,
many measured repeats, the **median** rather than the mean, a reported
coefficient of variation, and device-side synchronisation before the clock
stops — without a `synchronize()` the CPU timer stops while the GPU is still
working, and the number is fiction. Memory should be recorded alongside, since
a fast model that saturates memory is not fast at the next size up.

**MPS caveat.** Several operations have no Metal implementation and fall back
to CPU silently. A device comparison has to record *which* models did that, or
it attributes a framework gap to the hardware.

## What the current benchmark gets wrong

1. **It trains every model with plain `Adam(lr=1e-2)`**, ignoring both sets of
   upstream tags. The S4 and Mamba rows are therefore measuring models trained
   the way their authors avoid.
2. **Per-family learning rates were hand-picked** to stop divergence. That was
   treating the symptom: with the parameter groups and a warmup, one rate
   trains all four families (now a test). Hand-tuned rates are also a confound
   — a device comparison should not vary anything between rows that it is not
   measuring.
3. **No scheduler at all**, where every published recipe warms up and decays.
4. **The clock is one untimed pass**: no warmup, no repeats, mean not median,
   and `synchronize()` only at the ends. Fine for a rough ratio, not for a
   number anyone should quote.
5. **No memory measurement**, and no record of MPS fallbacks.
6. **Numerics and optimisation are entangled.** When two devices' losses
   differ, the current run cannot say whether the arithmetic differed or the
   optimiser amplified a difference the arithmetic made ten steps earlier.

## The proposed set

Three benchmarks, because the questions are genuinely different and one run
answering all three answers none of them cleanly.

### A — agreement (no training at all)

Fixed weights, one forward and one backward, compare outputs and gradients
between devices. **This is the honest CUDA-vs-MPS numerical comparison**: with
no optimiser in the loop there is nothing to amplify a difference, so what is
measured is the arithmetic and only the arithmetic.

Report: relative error on the output, on each gradient, in float32 and
float64, per model. Expect ~1e-6 for portable models and looser for the
vendored ones on CUDA, where the fused kernel is a different implementation.

This is the benchmark that answers "do the fused kernels agree with the
reference?" — the rule PLAN.md fixes for fast paths — and it is currently
mixed into a training run that cannot answer it.

### B — training fidelity (the recipe, correctly)

One recipe for every model: AdamW, `betas=(0.9, 0.95)`, weight decay 0.1,
`td.param_groups` for the SSM tags, linear warmup into cosine decay, gradient
clipping at 1.0. Same seed, same data, identical initial weights.

Report: loss curves, the step at which two devices' curves first separate, and
final weight distance. Because no per-family tuning is left, a difference
between two rows is a difference between two *models* rather than between two
learning rates.

### C — throughput and memory

Warmup iterations discarded, ≥50 measured, median and coefficient of
variation, `synchronize()` around each, forward and backward timed separately,
peak memory recorded, and a flag for any model that fell back off the GPU.

Report: median step time with CV, peak memory, and steps/s — with the CV
present so a reader can tell a real 1.3× from noise.

## What stays as it is

The zoo of sixteen models, the CPU-side initialisation and data generation
(the property that makes any cross-device comparison meaningful), and the
scorecard's refusal to produce a single combined number.

## Sources

- [state-spaces/s4](https://github.com/state-spaces/s4) — the `_optim` tag and
  its use in `example.py`'s optimizer construction and `CosineAnnealingLR`
- [s4/example.py](https://github.com/state-spaces/s4/blob/main/example.py)
- [MambaByte: Token-free Selective State Space Model](https://arxiv.org/pdf/2401.13660)
  — AdamW `betas=(0.9, 0.95)`, wd 0.1, clip 1.0, warmup + cosine
- [Stuffed Mamba: Oversized States Lead to the Inability to Forget](https://arxiv.org/pdf/2410.07145)
  — SSM learning rate capped separately from the rest
- [PyTorch Benchmark — Lei Mao](https://leimao.github.io/blog/PyTorch-Benchmark/)
  — synchronisation, warmup, why a CPU timer alone is wrong
- [How to Accurately Time CUDA Kernels in PyTorch — Speechmatics](https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch)
- [MLX vs MPS vs CUDA: a Benchmark](https://towardsdatascience.com/mlx-vs-mps-vs-cuda-a-benchmark-c5737ca6efc9/)
  — MPS operation coverage gaps and silent CPU fallback
