# Upstream verification dossier

Numbers from checking this library's **N-D composition** against the original
implementations, running on this machine. Not part of the package: the sdist
ships `src`, `tests`, `examples` and the top-level docs, so nothing here is
distributed, and neither is any upstream code.

Every mixer in `torch-dimensions` was already verified against its source
(S4D bitwise, S4 at 3e-8, Mamba at 1e-6 — PLAN.md Phase 7). What had never
been checked is the part this library claims as its own: **how the axes are
composed.** That is what these scripts do.

## Running them

The upstream repos stay in your own clone. Point `TD_EXTERNAL` at the
directory holding them (default `~/Desktop/Safe/code/github/external`):

```bash
pip install hydra-core einops timm        # genuine imports, dev-only
python dossier/verify_s4nd.py
python dossier/verify_mamba_nd.py
```

Neither repo is vendored. `_shims.py` makes them *importable*; it copies
nothing.

## Results, 2026-08-02, Apple Silicon (MPS) + CPU, float32

### S4ND — simultaneous N-D kernel vs our sequential sweep

S4ND does not sweep. It builds one 1-D SSM kernel per axis, outer-products
them in Fourier space, and applies a single N-D FFT convolution. We apply one
axis per layer. Feeding **upstream's own kernels** and adding **upstream's own
`D` skip**, so the only variable is the composition:

| device | rank | shape | max &#124;ours − upstream&#124; | output scale |
|---|---|---|---|---|
| CPU | 2 | (6, 7) | 2.384e-07 | 5.24 |
| CPU | 3 | (4, 5, 6) | **0.000e+00** | 6.53 |
| MPS | 2 | (6, 7) | 5.960e-08 | 5.30 |
| MPS | 3 | (4, 5, 6) | 4.657e-10 | 6.74 |

Bitwise at rank 3 on CPU, float32 epsilon elsewhere. **A plan of
one-axis-per-layer sweeps reproduces a model that was never written as a
sweep.**

The rank-2 rows are the *less* exact ones, which looks backwards until you
read the code: their `contract_version=0` path is hardcoded to two axes — it
indexes `k_f[0], k_f[1]` literally — and carries an extra reduce-and-divide
that rank 3 (`contract_version=1`, which builds its einsum string
programmatically) does not. The default does not survive rank 3. That is the
"hardcoded to one rank" problem this library exists to remove, in the source
of the paper that introduced N-D SSMs.

**What made it run off-GPU:** `src/utils/train.py` imports `pytorch_lightning`
at module scope for a logger no forward pass touches, and the Cauchy/
Vandermonde CUDA extension is absent so the kernel takes its pure-torch
fallback. Nothing about the mathematics needed a GPU.

### Mamba-ND — per-layer axis order vs `ScanPlan.paired`

| device | order | reverse | scans | max &#124;ours − upstream&#124; |
|---|---|---|---|---|
| CPU | `t l h w` | False / True | `w` | **0.0 / 0.0** |
| CPU | `t l w h` | False / True | `h` | **0.0 / 0.0** |
| CPU | `w h t l` | False / True | `l` | **0.0 / 0.0** |
| MPS | (same three) | False / True | — | **0.0 across all six** |

Bitwise, on both devices, in both directions. And the schedule:

```
layer   upstream (axis, reverse)   ScanPlan.paired
   0    ('w', False)            == ('w', False)
   1    ('w', True)             == ('w', True)
   2    ('h', False)            == ('h', False)
   ...
   schedules identical: True      (12 layers)
```

So `ScanPlan.paired` is not "inspired by" Mamba-ND's schedule; it *is* that
schedule, layer for layer. The docstring claiming so is now backed by a
comparison against the code rather than against the paper's prose.

**What made it run off-GPU:** `Block.__init__` takes `mixer_cls` as an
argument, so the selective scan is injected. Give it our portable
`MambaMixer` and the entire N-D scan runs on a Mac. Everything else in the
way — `mmcv`, `mmengine`, `mmaction`, `prettytable`, `timm`'s registry,
`mamba_ssm`'s Triton norms — is imported for the *backbone*, not the scan.
**Mamba-ND's schedule needs no CUDA; its packaging does.**

## A method we do not have

Reading their code answered a question the paper's prose leaves open.
Mamba-ND rearranges the flattened lattice into a per-layer axis order and
scans the **last `n_dim_pos` axes as one sequence**, batching the rest:

| `n_dim_pos` | what it does | our name |
|---|---|---|
| 1 | scan one axis | `td.axial_scan` |
| rank | scan every cell in one sequence | `td.flatten` |
| 2 of 4 | scan a *subset* of axes jointly | **none** |

Their published configs use the middle setting (`factorization='hw_t'` gives
`n_dim_pos = (2, 2, 4)`; `'h_w_t'` gives `(1, 1, 2)`), so it is not a corner
case — it is how the model is actually run. Both of our existing methods are
its endpoints, and the interpolation between them is expressible in this
library's terms but is not currently expressible in its API. Logged as a
Track E item.

## Licensing

- **state-spaces/s4** — Apache-2.0. A derivation with attribution already
  lives in `src/torch_dimensions/mixers/ssm.py`.
- **Mamba-ND** — **no LICENSE file at the commit cloned.** Provenance and
  usage rights unconfirmed, so it is neither vendored nor redistributed here,
  and the comparison imports it from a local clone only. Resolve with the
  authors before any use beyond private study.
