# CUDA verification checklist

**Status: run, on an NVIDIA RTX 5090 (Blackwell, sm_120), 4 August 2026.**
torch 2.12.1+cu130, Triton 3.7.1, Linux. Results below, and the raw output is
checked in at [`CUDA bench/cuda_check.txt`](../CUDA%20bench/cuda_check.txt).

```
13 passed · 0 failed · 1 skipped · 1 recorded
```

The page used to say "not yet run", and that mattered: since 0.3.1 the library
ships the original authors' fused kernels and chooses between them and a
portable path per call, and none of that choice had ever executed on a GPU.
What the run established:

| claim | result |
|---|---|
| `prefer_upstream` is True for a CUDA tensor | **True** — the fused path is reachable, verified for the first time |
| `prefer_upstream` is False for a CPU tensor on a CUDA box | False |
| `TD_FORCE_TORCH_KERNELS` overrides CUDA | honoured |
| vendored **S4 (DPLR)**, CUDA vs CPU, L=32/64/128 | **1.90e-07** — including L=64, where MPS lands exactly on the Nyquist pole; the guard is provably inert on CUDA |
| vendored **S4D** | 2.85e-07 |
| vendored **Mamba-1** | 2.30e-07 |
| vendored **Mamba-2** | 4.38e-07 |
| **Mamba-3** block end to end on CUDA | runs, gradients finite |
| rank-1 LSTM vs the pre-norm residual, under cuDNN | **bitwise identical** (0.0) |
| device placement refused in both directions | raises, as designed |
| autocast fp16 / bf16 through the kernel family | finite, output stays float32 |
| absent cells stay inert on CUDA | 0.0 |

And the full suite on the same machine: **1211 passed, 6 skipped, 0 failed** —
the skips are the MPS-only tests, which is what should skip on an NVIDIA box.

## What is still not established

**Mamba-3's Triton kernel has never been compared against our PyTorch
transcription.** That is the one comparison no CPU or MPS machine can make, and
it did not happen here either: `mamba-ssm` and `causal-conv1d` have no wheel for
sm_120 and fail to build from source against CUDA 13, so the fused Mamba
entry points were never importable. Every vendored Mamba number above is
therefore *reference path on CUDA*, not *fused kernel on CUDA*.

Our transcription remains validated three other ways — against an
independently written recurrent form (3.6e-15 in float64), against a direct
sum in the `trap -> 1` limit (2.2e-16), and by gradcheck — but not against the
kernel it was transcribed from. Anyone with an Ampere or Ada card, where
`mamba-ssm` installs cleanly, can close this by running the same one command.

## Two bugs this run found

Both were invisible on a machine without a GPU, and both were the same mistake:
dispatching on a property of the **box** instead of the **tensor**.

1. `UpstreamMamba2Mixer` defaulted `use_mem_eff_path` to
   `torch.cuda.is_available()`, so on a CUDA machine every CPU-resident Mamba-2
   asked for a kernel it could not reach and raised on its first forward — a
   CPU sanity check, a CPU test, the CPU half of any device comparison.
2. The vendored gated RMSNorm dispatched on `_HAS_TRITON`, sending a CPU tensor
   into a Triton kernel on any box that had both a GPU and Triton.

---

## The short version

To re-run it anywhere — a free Colab T4 is enough — two commands in a GPU
runtime:

```python
!git clone --depth 1 https://github.com/triadastra/torch-dimensions
%cd torch-dimensions
!pip install -q -e ".[dev,upstream]" mamba-ssm causal-conv1d

!python scripts/cuda_check.py     # every CUDA claim, with the number behind each
!pytest tests/ -q                 # the suite itself
```

`scripts/cuda_check.py` is the whole of this page as one runnable file: it
prints `pass`, `fail`, `skip` or `info` per claim and exits non-zero if
anything failed. It runs on CPU too, reporting `skip` with the reason, so the
harness can be verified without a GPU — which is how it was written.

`mamba-ssm` and `causal-conv1d` are what make the *fused* paths reachable.
Without them the dispatch still works but always chooses the portable side,
and the most valuable checks below turn into skips.

The device suite (`tests/test_device.py`) picks up CUDA automatically and
skips *visibly* when there is none, so a green run with no CUDA present proves
nothing — check that the device tests report `cuda` and not `skipped`.

## What is newly at stake since 0.3.1

The library now ships the original authors' S4, Mamba-1, Mamba-2 and Mamba-3
and picks per call between their fused CUDA kernels and a portable path. None
of that has ever run on CUDA:

- [ ] **`prefer_upstream` has never returned True.** The dispatch decides on
      the tensor's device at call time. If it is wrong, *every* CUDA user gets
      the portable path silently — or worse, a CPU tensor reaches a CUDA
      kernel. Both directions are checked.

- [ ] **No fused kernel has ever executed.** The vendored Mamba-1/2 blocks
      take upstream's fused route when `mamba_ssm` imports; the check compares
      each against its own CPU reference on the same weights.

- [ ] **Mamba-3's transcription has never met the kernel it came from.**
      Mamba-3 ships upstream as Triton alone, so
      `mixers/mamba3_compat.py` is *our* rewrite of the recurrence. On CPU and
      MPS it is checked against a second independent form (3e-15), a third
      direct sum (2e-16) and gradcheck — but never against their kernel,
      because no machine here can run one. This is the check that closes that
      gap, and the only one on this page that cannot be approximated
      elsewhere. Expect agreement at bfloat16's own resolution rather than
      float epsilon: their kernel runs bf16 with PTX `cos/sin/tanh`
      approximations, ours runs fp32 with exact functions.

- [ ] **`TD_FORCE_TORCH_KERNELS=1` exists to make the fallback testable on a
      CUDA box** and has never been used on one.

## What each item is actually checking

- [ ] **The full suite passes.** Not a smoke test: the whole thing. Most of it
      is device-agnostic, and the point is that nothing in the N-D bookkeeping
      assumed a device by accident.

- [ ] **Device-placement, both directions.** DEBUG.md #18: a lattice on one
      device indexing a tensor on another raises in one direction and silently
      works in the other. The suite moves models and lattices in both
      directions; it was written against MPS, and CUDA has different tolerance
      for the mismatch.

- [ ] **The Nyquist-pole guard still matters.** DEBUG.md records that the full
      S4 (DPLR) kernel divides by `1 + ω`, which is exactly zero at the
      Nyquist frequency; upstream survives it only by floating-point luck, and
      on MPS it lands exactly and NaNs. Confirm CUDA agrees with CPU on
      `tests/test_ssm.py` — the guard is in the shipped kernel, so this is
      checking the guard, not looking for the bug.

- [ ] **The bitwise rank-1 LSTM claim under cuDNN.** A single layer on a
      rank-1 lattice is bitwise identical to `nn.LSTM` on CPU. cuDNN may
      reorder reductions; whatever is true, write it down. Expected: still
      exact per single layer. If it is not, the claim in the README needs a
      device qualifier.

- [ ] **fp16 / bf16 / AMP.** The relative-cancellation guard in
      `compose/kernel.py` was designed dtype-aware, and DEBUG.md #12 found
      that float32 was the *vulnerable* dtype while fp16 was accidentally safe
      (values round to exact zero). CUDA is where half precision is actually
      used, so:

      ```python
      with torch.autocast("cuda", dtype=torch.float16):
          model(x)
      ```

      Check the sparse kernel-family path specifically — that is where the
      guard lives.

- [ ] **Benchmarks, for the table that does not exist yet.**

      ```bash
      python benchmarks/bench.py --out BENCHMARKS-cuda.md
      ```

      Two rows in the current BENCHMARKS.md explicitly say "re-measure on
      CUDA": `torch.compile` is a 0.8x *slowdown* on MPS at these sizes, and
      the factorized-vs-per-line crossover sits at 64³. Both are plausibly
      hardware-specific and neither should be repeated about CUDA until
      measured there.

- [ ] **A reproduction, to make the numbers comparable.**

      ```bash
      python -m examples.repro.smnist --epochs 20 --device cuda
      ```

      RESULTS.md rows carry their hardware, so a CUDA row sits beside the MPS
      one rather than replacing it.

## What this does *not* cover

Training to convergence on CUDA, and any performance claim beyond the
benchmark rows above. The rule for fast paths is unchanged and is what the
comparisons above implement: **the portable path is the reference, and the
fused path must agree with it on the same machine.** A fast path whose only
test is "it ran" does not ship.

## Reporting back

Open an issue with the output of:

```python
import torch, torch_dimensions as td

print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
print(td.__version__)
```

plus the pytest summary. If something fails, that failure is worth more than
the rest of this page — it is the first CUDA data point the project has.
