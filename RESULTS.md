# Results

Reproductions run with this library and nothing else — no model code outside
`torch_dimensions`, no second framework, no vendored datasets. Every row
carries its epoch budget, seed, wall-clock and hardware, because a result
without them is an anecdote.

**These are laptop-scale runs.** The papers referenced below train far longer
on far larger machines. What is being tested here is whether the *construction*
reproduces — whether the portable kernels and the N-D machinery learn the
tasks they are supposed to learn — not whether one Mac can match a cluster.

Regenerate with:

```bash
bash examples/repro/run_all.sh          # runs everything, appends to results.json
python -m examples.repro.report --out RESULTS.md
```

## Sequence tasks — the mixer without the lattice

A sequence is a lattice with no spatial axes, so these isolate the portable S4D kernel from the N-D machinery entirely. Published reference points: the S4/S4D papers report ~99.6% on sMNIST and ~98.5% on psMNIST, with longer schedules.

| task | configuration | params | epochs | seed | result | wall clock | hardware |
|---|---|---|---|---|---|---|---|
| psMNIST | td.S4D d_model=128 n_layers=4 d_state=64 | 201,482 | 20 | 0 | **97.79%** test accuracy | 20.7 min | Apple Silicon (MPS) arm64 |
| sMNIST | td.S4D d_model=128 n_layers=4 d_state=64 | 201,482 | 20 | 0 | **99.53%** test accuracy | 20.7 min | Apple Silicon (MPS) arm64 |

**Finding.** **Reproduced.** Both land inside one point of the published numbers, from a config and one command, on a laptop, in 21 minutes each. The portable S4D kernel — pure torch, no CUDA — learns what the paper's does.

## Sparse lattices — no published baseline exists

Beijing air quality: 12 stations x 6 pollutants, hourly, with a fraction of cells made absent. Arms differ in exactly one thing each and are scored on present cells only. These rows *are* the baseline.

| task | configuration | params | epochs | seed | result | wall clock | hardware |
|---|---|---|---|---|---|---|---|
| air quality 12×6 lattice, 30% cells absent | attention (sparse lattice + td.axial_attention across space), d_model=48 n_layers=4 | 123,169 | 8 | 0 | **0.1840** test_mse | 3.2 min | Apple Silicon (MPS) arm64 |
| air quality 12×6 lattice, 30% cells absent | cafa (sparse lattice + td.cafa across space), d_model=48 n_layers=4 | 123,169 | 8 | 0 | **0.1089** test_mse | 3.6 min | Apple Silicon (MPS) arm64 |
| air quality 12×6 lattice, 30% cells absent | masked (sparse lattice, absent cells marked), d_model=48 n_layers=4 | 75,793 | 8 | 0 | **0.0907** test_mse | 0.9 min | Apple Silicon (MPS) arm64 |
| air quality 12×6 lattice, 30% cells absent | masked (sparse lattice, absent cells marked), d_model=48 n_layers=4 | 75,793 | 1 | 0 | **0.1032** test_mse | 0.1 min | Apple Silicon (MPS) arm64 |
| air quality 12×6 lattice, 30% cells absent | zeros (dense lattice, absent cells zero-filled), d_model=48 n_layers=4 | 75,793 | 8 | 0 | **0.0904** test_mse | 0.8 min | Apple Silicon (MPS) arm64 |
| air quality 12×6 lattice, 30% cells absent | zeros (dense lattice, absent cells zero-filled), d_model=48 n_layers=4 | 75,793 | 1 | 0 | **0.1042** test_mse | 0.1 min | Apple Silicon (MPS) arm64 |
| air quality 12×6 lattice, 60% cells absent | masked (sparse lattice, absent cells marked), d_model=48 n_layers=4 | 75,793 | 8 | 0 | **0.0996** test_mse | 0.9 min | Apple Silicon (MPS) arm64 |
| air quality 12×6 lattice, 60% cells absent | zeros (dense lattice, absent cells zero-filled), d_model=48 n_layers=4 | 75,793 | 8 | 0 | **0.0988** test_mse | 0.8 min | Apple Silicon (MPS) arm64 |

**Finding.** **Masking absent cells did not improve accuracy, and that is the result.** At 30% absent the sparse and dense-with-zeros arms land at 0.0907 and 0.0904; at 60% absent, 0.0996 and 0.0988. The dense arm is marginally *ahead* both times, by less than one percent relative — well inside what a single seed can distinguish. The one place the ordering flips is the 1-epoch run (0.1032 masked vs 0.1042 dense), which is consistent with masking being a prior the dense model can otherwise learn, and is far too small a difference to claim from one seed.

The honest reading: **with a mask that is fixed across training and evaluation, a dense model learns the mask.** What `valid=` buys is not accuracy here — it is a *guarantee*, tested bitwise: absent cells provably cannot influence any output, whatever the data does. That matters when the mask varies between train and inference, when a cell's absence must not be learnable from correlations, and when you need the invariance to hold rather than to have been approximated. None of those is what this experiment measured, and the next one should measure the first of them.

On the method comparison: `axial_scan` (0.0907) beat `cafa` (0.1089) and `axial_attention` (0.1840) on this task, despite the kernel arms carrying 62% more parameters — consistent with BENCHMARKS.md, where factorization only starts paying at lattices two orders of magnitude larger than 12x6. Being able to run that comparison by changing one argument is the point; the answer being unflattering to the fancier method is why it was worth running.

---

All rows above: torch-dimensions 0.1.0 on Apple Silicon (MPS) arm64. Raw history, including runs superseded by a later one, is in [`examples/repro/results.json`](examples/repro/results.json).
