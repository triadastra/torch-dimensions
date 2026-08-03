# Device comparison

- **MPS bench** — Apple arm64 (MPS) · torch 2.13.0 · torch-dimensions 0.3.1
- **CUDA bench** — NVIDIA GeForce RTX 5090 · torch 2.12.1+cu130 · torch-dimensions 0.3.1

Both runs: seed 20260803, 300 steps, batch 4.
Data is drawn on CPU from a seeded generator, so both machines see the
same batches in the same order.

Both runs load one shared set of starting weights, so the initial
conditions are bit-identical and every difference below is arithmetic.

| model | loss (left) | loss (right) | Δ loss | first divergence | rel Δw | max Δw | speed |
|---|---|---|---|---|---|---|---|
| `lstm_2d_sparse` | 0.72047 | 0.72259 | 2.13e-03 | step 1 | 3.43e-02 | 4.05e-02 | 0.72× |
| `gru_2d_sparse` | 0.72084 | 0.71882 | 2.03e-03 | step 0 | 2.71e-02 | 2.75e-02 | 1.84× |
| `s4d_portable_2d` | 0.73560 | 0.73450 | 1.09e-03 | step 84 | 2.57e-04 | 5.21e-03 | 0.78× |
| `s4d_upstream_2d` | 0.73867 | 0.73867 | 8.94e-07 | step 86 | 2.13e-06 | 1.50e-03 | 0.73× |
| `s4_upstream_2d` | 0.73359 | 0.73277 | 8.28e-04 | step 89 | 3.58e-05 | 1.73e-03 | 0.79× |
| `mamba_portable_2d` | 0.73149 | 0.73126 | 2.26e-04 | step 93 | 1.29e-03 | 5.29e-03 | 0.54× |
| `mamba_upstream_2d` | 0.73244 | 0.73441 | 1.97e-03 | step 74 | 1.74e-03 | 7.64e-03 | 0.69× |
| `mamba2_2d` | 0.74520 | 0.74476 | 4.44e-04 | step 21 | 4.34e-02 | 5.67e-02 | 3.86× |
| `mamba3_2d` | 0.72130 | 0.72066 | 6.39e-04 | step 26 | 6.35e-03 | 1.32e-02 | 1.57× |
| `transformer_scan_2d` | 1.63615 | 1.63248 | 3.67e-03 | step 2 | 5.53e-02 | 8.12e-02 | 0.94× |
| `transformer_cafa_2d` | 1.45857 | 1.45871 | 1.44e-04 | step 42 | 4.64e-02 | 5.41e-02 | 2.03× |
| `transformer_flatten_2d` | 2.31866 | 2.31973 | 1.07e-03 | step 24 | 2.81e-02 | 5.61e-02 | 1.10× |
| `cnn_2d_sparse` | 1.86875 | 1.87058 | 1.83e-03 | step 0 | 2.75e-02 | 3.09e-02 | 1.02× |
| `tcn_2d_sparse` | 1.05559 | 1.05888 | 3.28e-03 | step 0 | 1.04e-01 | 1.35e-01 | 1.03× |
| `mamba_upstream_3d` | 0.00608 | 0.00599 | 9.04e-05 | step 55 | 3.05e-04 | 3.72e-03 | 0.88× |
| `lstm_3d` | 0.00822 | 0.00598 | 2.24e-03 | step 0 | 5.10e-02 | 7.38e-02 | 0.81× |
