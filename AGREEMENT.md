# Agreement — Apple arm64 (MPS) vs NVIDIA GeForce RTX 5090

- **MPS agree** — Apple arm64 (MPS), torch 2.13.0, dtypes float32
- **CUDA agree** — NVIDIA GeForce RTX 5090, torch 2.12.1+cu130, dtypes float32, float64

One forward and one backward from identical weights on identical data,
with no optimiser in the loop — so what is measured below is the
arithmetic and nothing else. Differences are elementwise maxima,
relative to the larger tensor's magnitude.

> **Note.** The two runs are on different torch versions (2.13.0 vs 2.12.1+cu130). That is a second difference between them besides the device, and a large disagreement cannot be attributed to the hardware alone.

> **float64 is not compared**: not present in both runs. MPS has no float64, so the precision control — the thing that separates reassociation from a different computation — is unavailable for this pair. It is recorded rather than skipped silently.

| model | dtype | output | worst gradient | which gradient | loss (MPS agree) | loss (CUDA agree) |
|---|---|---|---|---|---|---|
| `lstm_2d_sparse` ⚠ | float32 | 1.21e-04 | 5.19e-04 | `nd.norms.3.weight` | 0.657720 | 0.657708 |
| `gru_2d_sparse` ⚠ | float32 | 1.82e-04 | 4.25e-04 | `nd.norms.3.bias` | 1.002386 | 1.002378 |
| `s4d_portable_2d` ⚠ | float32 | 3.88e-07 | 1.36e-04 | `nd.mixers.0.kernel.A_imag` | 0.633561 | 0.633561 |
| `s4d_upstream_2d` | float32 | 3.99e-07 | 2.31e-05 | `nd.mixers.3.block.layer.kernel.A_real` | 0.560619 | 0.560619 |
| `s4_upstream_2d` | float32 | 4.11e-07 | 2.24e-05 | `nd.mixers.1.block.layer.kernel.A_imag` | 0.550349 | 0.550349 |
| `mamba_portable_2d` | float32 | 2.87e-07 | 9.40e-07 | `nd.mixers.2.out_proj.weight` | 0.486116 | 0.486116 |
| `mamba_upstream_2d` | float32 | 2.87e-07 | 9.40e-07 | `nd.mixers.2.block.out_proj.weight` | 0.486116 | 0.486116 |
| `mamba2_2d` | float32 | 8.75e-07 | 5.13e-07 | `nd.mixers.1.block.A_log` | 1.445458 | 1.445458 |
| `mamba3_2d` | float32 | 9.97e-07 | 2.76e-06 | `nd.mixers.1.block.C_bias` | 1.139720 | 1.139720 |
| `transformer_scan_2d` | float32 | 6.65e-07 | 6.44e-07 | `nd.mixers.1.proj.weight` | 19.463812 | 19.463810 |
| `transformer_cafa_2d` | float32 | 5.00e-07 | 1.36e-06 | `nd.bias.0` | 19.429951 | 19.429951 |
| `transformer_flatten_2d` | float32 | 6.83e-07 | 5.78e-07 | `nd.mixers.1.mlp.2.weight` | 19.420815 | 19.420813 |
| `cnn_2d_sparse` ⚠ | float32 | 1.96e-04 | 2.12e-04 | `nd.norms.3.weight` | 0.920592 | 0.920584 |
| `tcn_2d_sparse` ⚠ | float32 | 1.16e-04 | 8.93e-03 | `nd.mixers.3.convs.0.weight` | 0.606166 | 0.606155 |
| `mamba_upstream_3d` | float32 | 4.33e-07 | 1.05e-06 | `nd.mixers.3.block.in_proj.weight` | 0.678931 | 0.678931 |
| `lstm_3d` ⚠ | float32 | 1.44e-04 | 4.94e-04 | `nd.norms.0.weight` | 0.899200 | 0.899185 |

⚠ marks a difference above 1e-04.

## What this says

- 16 of 16 model-dtype pairs compared successfully.
- Worst output difference across everything: 1.96e-04.
- 6 pair(s) above 1e-04: `lstm_2d_sparse` (float32), `gru_2d_sparse` (float32), `s4d_portable_2d` (float32), `cnn_2d_sparse` (float32), `tcn_2d_sparse` (float32), `lstm_3d` (float32). For a vendored model on CUDA this is the expected signature of the fused kernel being a different implementation of the same recurrence, not evidence that either side is wrong.
