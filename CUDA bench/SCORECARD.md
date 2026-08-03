# Scorecard — NVIDIA GeForce RTX 5090

300 steps, batch 4, seed 20260803, torch 2.12.1+cu130, torch-dimensions 0.3.1.

One task — a cumulative sum along an axis — on one machine. That suits a
causal sequence model and is close to the worst case for a permutation-
invariant one, so this ranks models *on this problem* and implies nothing
about images or forecasting. There is no combined score on purpose: the
weighting between a point of loss and a step per second is the reader's,
not this file's.

| model | params | final loss | best | 90% at | tail spread | steps/s | Δloss per 1k params |
|---|---|---|---|---|---|---|---|
| `lstm_3d` | 51,136 | 0.0060 | 0.0034 | step 36 | 0.390 | 52.0 | 0.077 |
| `mamba_upstream_3d` | 50,752 | 0.0060 | 0.0051 | step 50 | 0.085 | 23.7 | 0.054 |
| `gru_2d_sparse` | 38,464 | 0.7188 | 0.4305 | step 45 | 0.114 | 62.5 | 0.084 |
| `mamba3_2d` | 141,472 | 0.7207 | 0.4295 | step 19 | 0.114 | 21.0 | 0.025 |
| `lstm_2d_sparse` | 51,136 | 0.7226 | 0.4261 | step 47 | 0.113 | 53.5 | 0.059 |
| `mamba_portable_2d` | 33,856 | 0.7313 | 0.4421 | step 72 | 0.112 | 24.7 | 0.086 |
| `s4_upstream_2d` | 17,216 | 0.7328 | 0.4466 | step 85 | 0.114 | 28.3 | 0.151 |
| `mamba_upstream_2d` | 33,856 | 0.7344 | 0.4437 | step 71 | 0.110 | 23.8 | 0.086 |
| `s4d_portable_2d` | 13,120 | 0.7345 | 0.4507 | step 86 | 0.116 | 42.3 | 0.204 |
| `s4d_upstream_2d` | 15,168 | 0.7387 | 0.4461 | step 74 | 0.115 | 32.7 | 0.173 |
| `mamba2_2d` | 111,920 | 0.7448 | 0.4487 | step 102 | 0.115 | 24.3 | 0.018 |
| `tcn_2d_sparse` | 25,152 | 1.0589 | 0.7618 | step 55 | 0.066 | 87.3 | 0.099 |
| `transformer_cafa_2d` | 55,376 | 1.4587 | 1.2720 | step 31 | 0.065 | 26.6 | 0.203 |
| `transformer_scan_2d` | 34,112 | 1.6325 | 1.3047 | step 35 | 0.065 | 68.6 | 0.050 |
| `cnn_2d_sparse` | 12,736 | 1.8706 | 1.5456 | step 45 | 0.069 | 104.4 | 0.141 |
| `transformer_flatten_2d` | 34,112 | 2.3197 | 1.9498 | step 164 | 0.073 | 41.3 | 0.022 |

⚠ marks a model whose loss never fell by 5% of where it started.

## Best on each question

- **lowest final loss** — `lstm_3d`, `mamba_upstream_3d`, `gru_2d_sparse`
- **fastest to 90% of its own improvement** — `mamba3_2d`, `transformer_cafa_2d`, `transformer_scan_2d`
- **steadiest at the end** — `transformer_scan_2d`, `transformer_cafa_2d`, `tcn_2d_sparse`
- **most steps per second** — `cnn_2d_sparse`, `tcn_2d_sparse`, `transformer_scan_2d`
- **most improvement per 1k parameters** — `s4d_portable_2d`, `transformer_cafa_2d`, `s4d_upstream_2d`
