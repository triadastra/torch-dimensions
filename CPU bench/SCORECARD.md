# Scorecard — arm

300 steps, batch 4, seed 20260803, torch 2.13.0, torch-dimensions 0.3.1.

One task — a cumulative sum along an axis — on one machine. That suits a
causal sequence model and is close to the worst case for a permutation-
invariant one, so this ranks models *on this problem* and implies nothing
about images or forecasting. There is no combined score on purpose: the
weighting between a point of loss and a step per second is the reader's,
not this file's.

| model | params | final loss | best | 90% at | tail spread | steps/s | Δloss per 1k params |
|---|---|---|---|---|---|---|---|
| `mamba_upstream_3d` | 50,752 | 0.0063 | 0.0054 | step 50 | 0.086 | 0.1 | 0.054 |
| `lstm_3d` | 51,136 | 0.0082 | 0.0046 | step 36 | 0.371 | 21.1 | 0.077 |
| `lstm_2d_sparse` | 51,136 | 0.7211 | 0.4268 | step 47 | 0.113 | 36.0 | 0.059 |
| `mamba3_2d` | 141,472 | 0.7220 | 0.4318 | step 19 | 0.114 | 5.3 | 0.025 |
| `gru_2d_sparse` | 38,464 | 0.7237 | 0.4320 | step 45 | 0.115 | 42.6 | 0.084 |
| `mamba_portable_2d` | 33,856 | 0.7313 | 0.4426 | step 72 | 0.111 | 0.2 | 0.086 |
| `s4_upstream_2d` | 17,216 | 0.7328 | 0.4462 | step 85 | 0.114 | 39.4 | 0.151 |
| `mamba_upstream_2d` | 33,856 | 0.7329 | 0.4442 | step 71 | 0.110 | 0.2 | 0.086 |
| `s4d_portable_2d` | 13,120 | 0.7351 | 0.4501 | step 86 | 0.116 | 45.8 | 0.204 |
| `s4d_upstream_2d` | 15,168 | 0.7387 | 0.4461 | step 74 | 0.115 | 42.7 | 0.173 |
| `mamba2_2d` | 111,920 | 0.7486 | 0.4522 | step 101 | 0.115 | 0.1 | 0.018 |
| `tcn_2d_sparse` | 25,152 | 1.0549 | 0.7640 | step 55 | 0.065 | 5.2 | 0.098 |
| `transformer_cafa_2d` | 55,376 | 1.4630 | 1.2607 | step 31 | 0.065 | 20.1 | 0.203 |
| `transformer_scan_2d` | 34,112 | 1.6310 | 1.3772 | step 35 | 0.065 | 36.0 | 0.047 |
| `cnn_2d_sparse` | 12,736 | 1.8688 | 1.5487 | step 45 | 0.069 | 12.7 | 0.141 |
| `transformer_flatten_2d` | 34,112 | 2.3200 | 1.9627 | step 47 | 0.073 | 31.2 | 0.022 |

⚠ marks a model whose loss never fell by 5% of where it started.

## Best on each question

- **lowest final loss** — `mamba_upstream_3d`, `lstm_3d`, `lstm_2d_sparse`
- **fastest to 90% of its own improvement** — `mamba3_2d`, `transformer_cafa_2d`, `transformer_scan_2d`
- **steadiest at the end** — `tcn_2d_sparse`, `transformer_cafa_2d`, `transformer_scan_2d`
- **most steps per second** — `s4d_portable_2d`, `s4d_upstream_2d`, `gru_2d_sparse`
- **most improvement per 1k parameters** — `s4d_portable_2d`, `transformer_cafa_2d`, `s4d_upstream_2d`
