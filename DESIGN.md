# torch-dimensions — Design

**Status:** design draft, nothing implemented.
**Positioning:** composition layer. We depend on `mamba-ssm` / `flash-linear-attention` / `state-spaces/s4` for 1-D kernels. We own the N-D structure, the registry, the config surface, and the `nn.Module` contract.

---

## 1. The claim

Every model in scope is the same object:

> **a 1-D mixer, plus a plan for sweeping it over an N-D lattice.**

| Model | 1-D mixer | Sweep strategy |
|---|---|---|
| Mamba-ND | Mamba-2 / Mamba-3 selective scan | sequential, one axis per layer, alternating direction |
| MDRNN / Grid-LSTM | `nn.LSTM` / `nn.GRU` | sequential, one axis per layer |
| S4ND | S4 FFT conv | separable — per-axis kernel, outer product |
| Axial Transformer | attention | per-axis kernel, contracted in turn |
| Factorized axial attention | factorized axial cross-attn | per-axis kernel, Kronecker contraction |

Nobody has written this down as one abstraction. Every repo above hardcodes its own axis bookkeeping. That is the entire product: **N-D RNNs, N-D transformers, and N-D SSMs fall out of one mechanism**, which is why the user-facing API can be as small as `S4(dim=2, layers=12)`.

The mechanism splits into exactly two composition strategies:

**`AxialScan` — sequential.** Permute one lattice axis to the sequence position, fold every other axis into batch, run the 1-D mixer, permute back. Residual + pre-norm per layer. This is what makes ND tractable: *we never write an N-D kernel.* Mamba-ND's real insight is that alternating 1-D scans over permuted axis orderings recover N-D context, so the existing 1-D CUDA kernel is reused unchanged.

**`AxialKernel` — contraction.** Build one kernel `A_ax ∈ R^{S_ax × S_ax}` per axis, contract them into the value tensor one axis at a time. On a dense lattice the joint operator is exactly the Kronecker product `A_0 ⊗ A_1 ⊗ … ⊗ A_{n-1}`, so cost is quadratic in *axial* size, not in `prod(shape)`.

Everything else in the library is a mixer, a plan, or a readout.

---

## 2. Public API

Three levels, each a strict superset of the one above.

### Level 1 — drop-in modules

```python
import torch
import torch_dimensions as td

lattice = td.Lattice(shape=(32, 64, 64), names=("depth", "height", "width"))

model = td.MambaND(d_model=128, n_layers=12, lattice=lattice, time=True)
y = model(x)  # (B, T, *shape, d_model)
loss = y.pow(2).mean()
loss.backward()  # plain autograd; nothing custom to call
```

Autograd is free. Composed torch ops give backward automatically; the upstream kernels already ship their own `autograd.Function`. There is no `model.backwards()` — you call `.backward()` on the loss, as with any torch model.

`LSTMND` and `AxialTransformer` take the same constructor shape. That is the point.

### Level 2 — mixer + plan

```python
import torch_dimensions as td

plan = td.ScanPlan.cyclic(axes=("time", "depth", "height"), n_layers=12, bidirectional=True)
block = td.AxialScan(
    mixer=td.mixers.Mamba2Mixer(d_model=128, d_state=64), plan=plan, lattice=lattice
)
```

`ScanPlan` is **data, not control flow** — a list of `(axis, reverse)` steps that is printable, serializable, diffable, and unit-testable independent of any mixer. In every existing ND implementation this schedule is inline list comprehensions welded to the module, which is why none of them can be inspected or swapped without editing the model. Constructors: `.cyclic()`, `.paired()`, `.hilbert()`, `.from_list()`.

Any `nn.Module` mapping `(M, A, H) -> (M, A, H)` is a valid mixer. That is the extension point — a user drops in a new SSM from a paper published next month without touching the library.

### Level 3 — config

```yaml
model:
  kind: mamba_nd
  d_model: 128
  n_layers: 12
  d_state: 64
  lattice: {shape: [32, 64, 64], names: [depth, height, width], time: true}
  plan:   {type: cyclic, bidirectional: true}
```

```python
model = td.build(cfg)  # or td.build_from_yaml(path)
```

One dataclass schema per block, validated at construction, with the error naming the offending key. No stringly-typed variant names.

---

## 3. `Lattice`

`Lattice` is the object that carries all N-D structure, so that no block ever hardcodes a rank or an axis meaning.

```python
@dataclass
class Lattice:
    shape: tuple[int, ...]
    names: tuple[str, ...] | None = None
    valid: torch.Tensor | None = None  # bool, `shape` — sparse support
    time: bool = False  # prepend a scanned, non-lattice time axis
```

It owns: axis-name → position resolution, flat indices for scatter/gather, the broadcast validity mask, per-axis valid counts for masked pooling, and permutation/inverse-permutation generation.

Two failure modes it exists to prevent, both endemic to existing ND implementations:

1. **Conflating an encoding strategy with a rank.** Names of the form `<encoder>_4d` mean "this encoder" *and* "four axes" at once, which forces string-rewriting hacks the moment a new encoder appears. Here they are orthogonal arguments: `kind=`, `lattice=`, `encoder=`.
2. **Rank-locked contraction tables.** Hand-written einsum strings keyed by axis index fix the rank at three (or two, or four) forever. `Lattice` generates permutations for arbitrary N instead. Prefer `permute → reshape → matmul → permute back` over generated einsum strings: same FLOPs, no per-call path planning, and `torch.compile`-friendly.

Device-dependent limits — chunk sizes for the folded batch dimension, kernel grid bounds — are library-owned and auto-tuned, never user-facing constants.

---

## 4. Sparse lattices — the differentiator

S4ND, Mamba-ND, and factorized axial attention all assume a **dense** grid. Real N-D data frequently is not: not every (sensor, channel, band) triple is instrumented, not every (patient, visit, assay) exists, meshes are irregular, modalities go missing.

Support is mechanical once `Lattice.valid` exists, and different per family:

- **Scan family:** scatter to dense, zero invalid cells after each layer, gather back.
- **Kernel family:** masked-mean pooling over valid cells only, then per-line renormalization by the softmax mass that landed on valid keys, so every output stays a convex combination of valid values.

That per-line rescale is the one departure from a strict Kronecker product, and it costs `O(N · S_ax)` elementwise work rather than `O(N · S_ax)` score *memory*. This is what keeps the factorized path alive at rank 4, where materializing one attention matrix per lattice line runs out of memory.

Making this a first-class feature — automatic for every mixer, tested once — is the strongest reason for this library to exist rather than for users to keep vendoring kernels and re-deriving the masking themselves.

---

## 5. Layout

```
torch_dimensions/
  lattice.py          Lattice, permutation + scatter/gather + mask machinery
  plan.py             ScanPlan and its constructors
  compose/
    scan.py           AxialScan
    kernel.py         AxialKernel, Kronecker contraction, sparse renormalization
  mixers/
    ssm.py            Mamba2, Mamba3, S4, S5   (thin adapters over upstream kernels)
    rnn.py            LSTM, GRU                (adapters over torch.nn)
    attn.py           self-attn, cross-attn, factorized axial kernel
  models/             MambaND, S4ND, AxialTransformer, LSTMND, GRUND
  registry.py         register / build / list
  config.py           dataclass schemas + YAML loader
  testing.py          shared conformance suite
```

`mixers/` are **adapters, not reimplementations** — that is the composition-layer decision made concrete. Optional deps are imported defensively: a missing `mamba-ssm` unregisters `mamba_nd` and leaves the rest importable. A missing real implementation must **fail loudly rather than silently register a stub**, so that a benchmark can never accidentally report an LSTM's numbers under an SSM's name.

---

## 6. Conformance suite

One parametrized suite every registered block must pass. This is what keeps an N-model × N-dim matrix from rotting.

1. **Shape** — `(B, *shape, H)` in, same out, for ranks 1–4.
2. **Gradient** — `gradcheck` in float64 on a tiny instance; every parameter receives non-`None` grad.
3. **Equivalence** — a rank-1 lattice must match the underlying 1-D module bit-for-bit. Catches permutation bugs immediately.
4. **Kronecker identity** — for `AxialKernel` on a dense lattice, sequential contraction must equal the explicit `⊗` operator to numerical tolerance.
5. **Mask invariance** — invalid cells must not influence valid outputs. Perturb invalid cells, assert valid outputs are bitwise unchanged. This is the sparse-lattice guarantee, and the test most likely to catch a real bug.
6. **Permutation covariance** — permuting lattice axes and the plan together permutes the output correspondingly.
7. **Compile** — `torch.compile` produces matching numerics.

---

## 7. v0.1 scope

**In:** `Lattice` (dense + sparse), `ScanPlan`, `AxialScan`, `AxialKernel`, mixers for LSTM/GRU/attention (pure torch, no optional deps) and Mamba-2/S4 (adapters), models `LSTMND`/`GRUND`/`AxialTransformer`/`MambaND`/`S4ND`, registry, config, conformance suite. Ranks 1–4.

**Deferred, deliberately:**

- **Stateful stepping.** Autoregressive decode needs per-axis recurrent state caching, and along a *non-time* axis "state" is not well-defined — you would be caching a cross-section, not a prefix. This is a genuine open research question, not an implementation gap. v0.1 is forward-only; the README must say so.
- **Custom kernels.** The composition-layer decision. Revisit only when profiling shows permute/contiguous dominating, at which point the fix is a fused permute+scan kernel, not a reimplemented SSM.
- **Ranks ≥ 5.** Nothing forbids them; nothing tests them.
- **Training loops, datasets, trainers.** Out of scope permanently. This is a layer library.

**Known risk.** The permutes are not free: `AxialScan` needs two `.contiguous()` calls per layer, so a 12-layer ND model does ~24 full-tensor copies. Benchmark this before advertising performance numbers — it may well dominate the mixer at small `d_model`.

---

## 8. Open questions

1. **Axis order when `time=True`.** Is time axis 0 of the lattice, or a separate leading dim? Existing implementations special-case it into the scan schedule. Cleaner: time is a normal named axis, and causality is a property of the *mixer*, not the axis.
2. **Does `AxialKernel` need `AxialScan`'s per-layer alternation**, or is one pass over all axes sufficient? Factorized attention does one pass; Mamba-ND alternates across 12 layers. Probably a plan-level choice rather than two mechanisms.
3. ~~**Name.**~~ Resolved: `torch-dimensions`, importing as `torch_dimensions`, aliased `td`. See [PLAN.md](PLAN.md).
