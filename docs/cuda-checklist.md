# CUDA verification checklist

**Status: not yet run.** Everything in this library is pure torch and the
device suite was written to run on whatever accelerator exists — but it has
only ever *executed* on CPU and Apple Silicon (MPS). Until somebody runs this
page on a CUDA machine, "works on CUDA" is a design claim here, not a tested
one, and the README says so.

This is deliberately a checklist rather than a CI job: GitHub's GPU runners
are paid, and the honest interim is a documented procedure that takes about
fifteen minutes on a free Colab T4.

---

## The short version

```python
!pip install torch-dimensions safetensors pytest
!python -c "import torch; print(torch.cuda.get_device_name(0))"
!git clone --depth 1 https://github.com/triadastra/torch-dimensions
!cd torch-dimensions && pytest tests/ -v
```

The device suite (`tests/test_device.py`) picks up CUDA automatically and
skips *visibly* when there is none, so a green run with no CUDA present proves
nothing — check that the device tests report `cuda` and not `skipped`.

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

Fused kernels. `mamba-ssm` and `causal-conv1d` are CUDA-only and are not
wired in yet (PLAN.md Track C2). When they are, the rule is fixed in advance:
**the portable path is the reference, and the fused path must agree with it in
the conformance suite on the same machine.** A fast path whose only test is
"it ran" does not ship.

## Reporting back

Open an issue with the output of:

```python
import torch, torch_dimensions as td

print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
print(td.__version__)
```

plus the pytest summary. If something fails, that failure is worth more than
the rest of this page — it is the first CUDA data point the project has.
