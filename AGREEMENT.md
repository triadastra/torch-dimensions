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
| `lstm_2d_sparse` ⚠ | float32 | 2.66e-06 | 3.05e-06 | `nd.mixers.5.rnn.weight_hh_l0` | 0.657720 | 0.657720 |
| `gru_2d_sparse` ⚠ | float32 | 3.11e-06 | 1.67e-06 | `nd.mixers.5.rnn.weight_hh_l0` | 1.002386 | 1.002386 |
| `s4d_portable_2d` ⚠ | float32 | 3.88e-07 | 1.36e-04 | `nd.mixers.0.kernel.A_imag` | 0.633561 | 0.633561 |
| `s4d_upstream_2d` ⚠ | float32 | 3.99e-07 | 2.31e-05 | `nd.mixers.3.block.layer.kernel.A_real` | 0.560619 | 0.560619 |
| `s4_upstream_2d` ⚠ | float32 | 4.11e-07 | 2.24e-05 | `nd.mixers.1.block.layer.kernel.A_imag` | 0.550349 | 0.550349 |
| `mamba_portable_2d` | float32 | 2.87e-07 | 9.40e-07 | `nd.mixers.2.out_proj.weight` | 0.486116 | 0.486116 |
| `mamba_upstream_2d` | float32 | 2.87e-07 | 9.40e-07 | `nd.mixers.2.block.out_proj.weight` | 0.486116 | 0.486116 |
| `mamba2_2d` | float32 | 8.75e-07 | 5.13e-07 | `nd.mixers.1.block.A_log` | 1.445458 | 1.445458 |
| `mamba3_2d` ⚠ | float32 | 9.97e-07 | 2.76e-06 | `nd.mixers.1.block.C_bias` | 1.139720 | 1.139720 |
| `transformer_scan_2d` | float32 | 6.65e-07 | 6.44e-07 | `nd.mixers.1.proj.weight` | 19.463812 | 19.463810 |
| `transformer_cafa_2d` ⚠ | float32 | 5.00e-07 | 1.36e-06 | `nd.bias.0` | 19.429951 | 19.429951 |
| `transformer_flatten_2d` | float32 | 6.83e-07 | 5.78e-07 | `nd.mixers.1.mlp.2.weight` | 19.420815 | 19.420813 |
| `cnn_2d_sparse` | float32 | 3.51e-07 | 3.58e-07 | `nd.mixers.0.convs.0.weight` | 0.920592 | 0.920592 |
| `tcn_2d_sparse` | float32 | 2.79e-07 | 2.06e-07 | `nd.mixers.1.convs.0.weight` | 0.606166 | 0.606166 |
| `mamba_upstream_3d` ⚠ | float32 | 4.33e-07 | 1.05e-06 | `nd.mixers.3.block.in_proj.weight` | 0.678931 | 0.678931 |
| `lstm_3d` ⚠ | float32 | 2.54e-06 | 1.73e-06 | `nd.mixers.5.rnn.weight_hh_l0` | 0.899200 | 0.899200 |

⚠ marks a difference above 1e-06, in the output or in any gradient.

## What this says

- 16 of 16 model-dtype pairs compared successfully.
- Worst **output** difference across everything: 3.11e-06.
- Worst **gradient** difference: 1.36e-04 on `nd.mixers.0.kernel.A_imag`.
- 9 pair(s) above 1e-06: `lstm_2d_sparse` (float32), `gru_2d_sparse` (float32), `s4d_portable_2d` (float32), `s4d_upstream_2d` (float32), `s4_upstream_2d` (float32), `mamba3_2d` (float32), `transformer_cafa_2d` (float32), `mamba_upstream_3d` (float32), `lstm_3d` (float32).
- A row flagged only on a *gradient* of an SSM frequency (`A_imag`, `A_real`) is cancellation, not disagreement: those gradients sum oscillating terms that nearly cancel, and the same pair that differs by 1.35e-04 in float32 differs by 4.87e-15 in float64. A row flagged on its *output* is worth reading — for a vendored model on CUDA it is the expected signature of the fused kernel being a different implementation of the same recurrence.
