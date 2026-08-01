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

```bash
pip install hydra-core einops timm        # genuine imports, dev-only
python dossier/verify_s4nd.py
python dossier/verify_mamba_nd.py
```

Each script clones what it needs on first run, into
`~/.cache/torch-dimensions/upstream` (override with `TD_EXTERNAL`), prints the
repository's license before fetching, and reports the commit it landed on so
every number below is attributable to a revision.

The clones are **sparse**: `--filter=blob:none --sparse` plus a checkout of
only the directories a comparison reads. For s4 that is `src/models` and
`src/utils` — **1.7 MB instead of 39 MB**, producing identical numbers.

That is deliberately not the same as vendoring a trimmed copy, which was
considered and rejected. A distilled copy is a fork we would maintain forever,
and worse, it would put *our* judgement about which modules matter inside the
very check whose value is that it runs *their* code. Sparse checkout gets the
whole size benefit and keeps the code genuinely theirs. (Licensing would have
permitted vendoring for the Apache-2.0 and MIT repositories; it was not the
licence that decided this.)

**The clone directory may not be inside this repository, and that is enforced
rather than advised.** `_shims.py` raises if `TD_EXTERNAL` resolves to this
tree or below it. The reason is narrow and worth stating: an upstream clone
sitting in the working tree is one `git add -A` from being committed, and one
of these repositories grants no license to redistribute. Refusing the path
outright is the only version of that rule that survives a hurried afternoon.

Nothing upstream is vendored: no third-party code is in our tree, our git
history, our sdist or our wheel. `_shims.py` fetches and makes importable; it
copies nothing.

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

### CaFA — factorized attention

[BaratiLab/CaFA](https://github.com/BaratiLab/CaFA) (Li, Zhou, Patil, Barati
Farimani, [arXiv:2405.07395](https://arxiv.org/abs/2405.07395)), **MIT** — so
unlike Mamba-ND this one could be vendored; it still is not, because it does
not need to be.

Their contraction (`FABlockS2`) is two sequential per-axis einsums:

```python
u = einsum("bhij,bhjmc->bhimc", k_lat, u) * pi / n_lat
u = einsum("bhlm,bhimc->bhilc", k_long, u) * 2 * pi / n_lon
```

That is `axial_contract` applied axis by axis. Checked directly, float64:
**2.0e-16 relative** — machine precision. Their `PoolingReducer` also pools
"all spatial dimensions but the first", which is the same pooling rule ours
uses.

The differences are all *features*, not composition:

| | CaFA | torch-dimensions |
|---|---|---|
| rank | **2, hardcoded** (`to_lat`, `to_long`) | any |
| heads | multi-head | single — still open |
| q/k normalization | RMSNorm option | added (`qk_norm=`) |
| kernel residual | `K + gamma·I`, on by default | added (`kernel_residual=`) |
| positional | RoPE + Bessel radial basis | learned relative-position bias |
| row normalization | `normalize_to_one` — **crashes** | per-line renormalization with a relative cancellation guard |
| sparse lattices | none | masked and renormalized |
| domain | sphere (quadrature weights, area weighting) | domain-agnostic |

Two things worth stating plainly. Their `normalize_to_one=True` path raises
`AttributeError: 'LowRankKernel' object has no attribute 'use_softmax'` —
`__init__` guards on an attribute that is never assigned, so a documented
option is unreachable. And `FABlockS2` names its two axes in its fields, so
CaFA is a 2-D model by construction; the rank-generality is ours.

What we took: `qk_norm` and `kernel_residual`, both off by default so existing
models are bit-for-bit unchanged. What we did not: RoPE (we have a learned
bias table) and the spherical quadrature weights, which are a property of the
sphere rather than of the method.

## Scan factorization — a method we do not have, and its proper name

The code has a `n_dim_pos` parameter that scans the last *k* axes of a
per-layer ordering as one sequence and batches the rest. The paper names this
directly (§4.1, Fig. 5b, "Scan-Factorization policies"):

> Col 1: No factorization, there is only 1 sequence. Col 2: Factorizing the 3D
> sequence into D 2D sequences [...] Col 3: Factorizing the 3D sequence into
> D² 1D sequences.

So it is a spectrum, and our two methods are its endpoints:

| paper's policy | `n_dim_pos` | our name |
|---|---|---|
| no factorization — 1 sequence | rank | `td.flatten` |
| factorize into D 2D sequences | in between | **none** |
| factorize into D² 1D sequences | 1 | `td.axial_scan` |

Their published configs use the middle (`factorization='hw_t'` →
`n_dim_pos = (2, 2, 4)`; `'h_w_t'` → `(1, 1, 2)`), so it is how the model is
actually run rather than a corner case. Logged as a Track E item.

**One correction from reading the paper rather than only the code.** In their
notation `H+` is not "scan the H axis". §4.1: a scan ordering permutes the
axes and flattens *all* of them into one 1D sequence, forward or reversed,
with "the last dimension traversed continuously" — so `H+` names the
permutation that ends in H. Scanning a single axis is the maximally factorized
policy, not a different kind of object. The equivalence measured above
(`n_dim_pos=1` ≡ `td.axial_apply`) is that corner of the spectrum.

## Licensing audit

Checked from the files on disk on 2026-08-02, to a depth of four directories,
for `LICENSE*`, `NOTICE*` and `COPYING*`:

| repo | LICENSE | NOTICE | what we take |
|---|---|---|---|
| state-spaces/s4 | Apache-2.0 | none | S4/S4D kernel mathematics — a **derivation**, attributed in `mixers/ssm.py` |
| state-spaces/mamba | Apache-2.0 | none | selective-scan recurrence — a **derivation**, same file |
| BaratiLab/CaFA | MIT (© 2024 BaratiLab) | none | two **ideas** (`qk_norm`, `kernel_residual`), independently implemented |
| jacklishufan/Mamba-ND | **none** | none | an **algorithm** described in the paper (`ScanPlan.paired`); no code |

Neither Apache-2.0 upstream ships a NOTICE, so section 4(d) creates nothing to
propagate. Our own `NOTICE` carries the attributions above and ships with the
package.

**Mamba-ND is the one to be careful with.** It states no license anywhere, and
eight of its files carry `Copyright (c) OpenMMLab` headers — OpenMMLab code is
Apache-2.0, which asks a redistributor to ship the license with it. Whatever
the explanation, the practical consequence for this project is the same one it
already followed: no code from that repository is copied, vendored or
redistributed here, and `verify_mamba_nd.py` imports it from a user's own
clone for comparison only. Anyone building on it should settle the question
with its authors.

*(A false positive worth noting so the next person does not chase it: `s4`
contains several files named `copying.py` / `copying.yaml`. That is the
synthetic **copying task** from the long-range-arena literature, not a
license.)*
