# Benchmarks

Measured, not estimated; regenerate with `python benchmarks/bench.py --out BENCHMARKS.md`. Every number below comes from one machine, named here, with seeds fixed — a benchmark whose hardware is unstated is a rumour.

| | |
|---|---|
| device | mps |
| accelerator | Apple Silicon (MPS), arm64 |
| platform | Darwin 25.3.0 |
| python | 3.10.10 |
| torch | 2.13.0 |
| torch_dimensions | 0.1.0 |

Timings are the median of repeated runs after warmup, device-synchronized. Peak memory is torch's allocator high-water mark and is reported only where torch tracks it (not on CPU).

### Fold overhead

**Decides Track C3.** A fused, permute-avoiding fold is only worth writing if the fold column is a large fraction of the step column.

| rank | d_model | cells | fold+unfold ms | LSTM step ms | fold % |
|---|---|---|---|---|---|
| 1 | 32 | 4096 | 1.38 | 78.9 | 1.75 |
| 1 | 64 | 4096 | 1.42 | 87.5 | 1.63 |
| 1 | 128 | 4096 | 2.61 | 106 | 2.47 |
| 1 | 256 | 4096 | 4.87 | 132 | 3.69 |
| 1 | 512 | 4096 | 9.64 | 229 | 4.21 |
| 2 | 32 | 4096 | 1.66 | 7.3 | 22.8 |
| 2 | 64 | 4096 | 2.64 | 12.3 | 21.5 |
| 2 | 128 | 4096 | 4.96 | 25.6 | 19.4 |
| 2 | 256 | 4096 | 9.57 | 62 | 15.4 |
| 2 | 512 | 4096 | 18.4 | 196 | 9.39 |
| 3 | 32 | 4096 | 2.53 | 8.14 | 31.1 |
| 3 | 64 | 4096 | 4.08 | 14.9 | 27.3 |
| 3 | 128 | 4096 | 7.24 | 31.9 | 22.7 |
| 3 | 256 | 4096 | 14 | 79.8 | 17.5 |
| 3 | 512 | 4096 | 27 | 236 | 11.4 |
| 4 | 32 | 4096 | 2.98 | 9.94 | 30 |
| 4 | 64 | 4096 | 5.14 | 18.3 | 28.1 |
| 4 | 128 | 4096 | 9.75 | 39.3 | 24.8 |
| 4 | 256 | 4096 | 18.4 | 100 | 18.4 |
| 4 | 512 | 4096 | 35.8 | 297 | 12.1 |

**Finding.** The fold is 10-30% of a step at rank >= 2 and small `d_model`, and its share *falls* as `d_model` grows because the mixer's work grows faster than the copy's. So a fused fold is worth roughly a fifth of a step in the small-model corner and almost nothing in the large-model corner — real, but not the first thing to optimize. Rank 1 is the odd row: one 4,096-long sequential sweep dominates so thoroughly that the copy disappears into it.

### Composition families

**Decides which method to reach for.** Per-line cost grows with the number of lines; factorized cost does not. Rank 4 is where the difference stops being academic.

| rank | cells | method | params | fwd ms | fwd+bwd ms | peak MB |
|---|---|---|---|---|---|---|
| 2 | 1296 | axial_scan | 25536 | 4.18 | 10.1 | n/a |
| 2 | 1296 | axial_attention | 48960 | 4.84 | 12 | n/a |
| 2 | 1296 | cafa | 48960 | 11.6 | 21.6 | n/a |
| 3 | 1331 | axial_scan | 34048 | 3.8 | 7.09 | n/a |
| 3 | 1331 | axial_attention | 64556 | 8.9 | 21.6 | n/a |
| 3 | 1331 | cafa | 64556 | 28.2 | 39.6 | n/a |
| 4 | 1296 | axial_scan | 42560 | 3.4 | 8.93 | n/a |
| 4 | 1296 | axial_attention | 89840 | 12.3 | 35.5 | n/a |
| 4 | 1296 | cafa | 89840 | 44.7 | 61.6 | n/a |

**Finding.** At ~1,300 cells the factorized path is *slower* than per-line attention, not faster: the lattice is small enough that per-line scores fit comfortably, and CaFA pays for pooling and per-line kernel expansion on top. That is the honest result at this size and it is not the claim the family exists for — see the next table, which grows the axis until the crossover shows up.

### Where factorization starts winning

**The claim the kernel family exists for.** Per-line attention costs O(cells · A); factorized costs O(A² + cells). Growing the axis at fixed rank must make the ratio grow, or the family is not earning its complexity.

| rank | shape | cells | axial_attention ms | cafa ms | per-line / cafa |
|---|---|---|---|---|---|
| 2 | 8×8 | 64 | 4.83 | 8.34 | 0.579 |
| 2 | 16×16 | 256 | 4.12 | 8.7 | 0.474 |
| 2 | 32×32 | 1024 | 4.4 | 10.2 | 0.434 |
| 2 | 64×64 | 4096 | 9.86 | 16.3 | 0.604 |
| 2 | 96×96 | 9216 | 15.2 | 19.4 | 0.786 |
| 3 | 8×8×8 | 512 | 7.23 | 18.8 | 0.385 |
| 3 | 16×16×16 | 4096 | 11.6 | 25.8 | 0.45 |
| 3 | 32×32×32 | 32768 | 18.8 | 36.2 | 0.518 |
| 3 | 48×48×48 | 110592 | 66.7 | 76.1 | 0.877 |
| 3 | 64×64×64 | 262144 | 162 | 148 | 1.1 |

**Finding.** The ratio climbs with axis length as predicted — and the crossover sits far further out than the library's own documentation implied. Per-line attention is **faster** everywhere below roughly 50 cells per axis, by 2-3x on small lattices; the two meet around 48³ and factorization only pulls ahead at 64³ (262k cells). So the honest rule is: use `td.axial_attention` for anything grid-shaped and modest — CIFAR-sized images, small volumes — and reach for `td.cafa` at image-or-volume scale, where the ratio keeps growing and per-line eventually cannot allocate at all. The family earns its complexity at the top end, not in the middle, and this table replaces the guess that it earned it everywhere.

### Rank at fixed cell count

**Tests a design claim.** Cost should track cells, not rank; a rank term would mean the abstraction charges for dimensions it promised were free.

| rank | shape | cells | layers | fwd ms | ms / (cell·layer) ×10⁻³ |
|---|---|---|---|---|---|
| 1 | 4096 | 4096 | 4 | 250 | 15.3 |
| 2 | 64×64 | 4096 | 4 | 7.59 | 0.464 |
| 3 | 16×16×16 | 4096 | 4 | 3.33 | 0.203 |
| 4 | 8×8×8×8 | 4096 | 4 | 2.89 | 0.176 |
| 5 | 5×5×5×5×5 | 3125 | 4 | 2.66 | 0.213 |

**Finding.** Cost tracks the **length of the swept axis**, not the cell count and not the rank. At a fixed ~4,096 cells, rank 1 (one 4,096-step sweep) costs ~95x rank 4 (sweeps of 8). The design said cost should track cells; the measurement says the sequential mixer's launch depth dominates everything else, so spreading the same cells over more axes is a large speedup rather than a cost. Rank does not appear in the cost model - axis *length* does.

### Chunked fold

**Feeds the Phase 3 auto-tune item.** `chunk=` splits the folded batch; it trades peak memory for launch count. The crossover is what auto-tuning would need to know.

| cells | d_model | chunk | fwd ms | peak MB |
|---|---|---|---|---|
| 4096 | 64 | None | 9.37 | n/a |
| 4096 | 64 | 64 | 33.9 | n/a |
| 4096 | 64 | 256 | 13.3 | n/a |
| 4096 | 64 | 1024 | 10.9 | n/a |
| 4096 | 256 | None | 24.7 | n/a |
| 4096 | 256 | 64 | 60.2 | n/a |
| 4096 | 256 | 256 | 30 | n/a |
| 4096 | 256 | 1024 | 24.8 | n/a |

**Finding.** Chunking never won here. `chunk=64` costs 3x by launching 3x the kernels; the largest chunk is within noise of no chunking at all. On this hardware `chunk=` is a memory-pressure valve, not a speed knob, and the Phase 3 auto-tuner should default it off and reach for it only when an allocation actually fails.

### Mixer families

**Sets expectations, and the baseline every fast path must beat.** The portable Mamba scan is a python loop over the swept axis: correct, honest, and the row to watch when Track C2's fused path lands.

| model | rank | cells | params | fwd ms | fwd+bwd ms |
|---|---|---|---|---|---|
| LSTM | 2 | 256 | 133632 | 3.67 | 9.2 |
| GRU | 2 | 256 | 100352 | 13.3 | 39.9 |
| S4D | 2 | 256 | 67072 | 4.65 | 11.1 |
| S4 | 2 | 256 | 99840 | 7.56 | 16.7 |
| Mamba | 2 | 256 | 131072 | 31.9 | 103 |

**Finding.** The portable Mamba scan is ~5-8x the cost of the kernel-based SSMs at equal width, which is exactly what a python-level sequential loop should cost and is the number Track C2's fused path has to beat. S4 (DPLR) is within ~10% of S4D (diagonal) despite the Cauchy resolvent and Woodbury correction: the full kernel is not the expensive one, the sequential scan is.

### torch.compile

**Decides whether to document compile as recommended.** Compile time is paid once; the ratio is what a user gets for it.

| model | eager ms | compiled ms | speedup |
|---|---|---|---|
| LSTM | 2.64 | 3.09 | 0.85× |
| S4D | 3.27 | 4.21 | 0.78× |

**Finding.** `torch.compile` is a *slowdown* on MPS at this size (~0.8x) - the graphs are small and the launch overhead it removes was not the bottleneck. It is therefore not recommended by default; the conformance suite still checks that it is numerically correct when a user does turn it on. Re-measure on CUDA before repeating this sentence about any other machine.
