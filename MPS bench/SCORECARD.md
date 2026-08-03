# Scorecard — Apple arm64 (MPS)

300 steps, batch 4, seed 20260803, torch 2.13.0, torch-dimensions 0.3.1.

One task — a cumulative sum along an axis — on one machine. That suits a
causal sequence model and is close to the worst case for a permutation-
invariant one, so this ranks models *on this problem* and implies nothing
about images or forecasting. There is no combined score on purpose: the
weighting between a point of loss and a step per second is the reader's,
not this file's.

| model | params | final loss | best | 90% at | tail spread | steps/s | Δloss per 1k params |
|---|---|---|---|---|---|---|---|
| `mamba_upstream_3d` | 50,752 | 0.0061 | 0.0052 | step 50 | 0.085 | 26.8 | 0.054 |
| `lstm_3d` | 51,136 | 0.0082 | 0.0046 | step 36 | 0.371 | 66.0 | 0.077 |
| `lstm_2d_sparse` | 51,136 | 0.7205 | 0.4254 | step 47 | 0.113 | 70.0 | 0.059 |
| `gru_2d_sparse` | 38,464 | 0.7208 | 0.4314 | step 45 | 0.114 | 32.5 | 0.084 |
| `mamba3_2d` | 141,472 | 0.7213 | 0.4304 | step 19 | 0.113 | 14.6 | 0.025 |
| `mamba_portable_2d` | 33,856 | 0.7315 | 0.4421 | step 72 | 0.112 | 44.9 | 0.086 |
| `mamba_upstream_2d` | 33,856 | 0.7324 | 0.4440 | step 71 | 0.110 | 40.1 | 0.086 |
| `s4_upstream_2d` | 17,216 | 0.7336 | 0.4468 | step 85 | 0.115 | 39.0 | 0.151 |
| `s4d_portable_2d` | 13,120 | 0.7356 | 0.4496 | step 86 | 0.117 | 61.1 | 0.204 |
| `s4d_upstream_2d` | 15,168 | 0.7387 | 0.4461 | step 74 | 0.115 | 46.9 | 0.173 |
| `mamba2_2d` | 111,920 | 0.7452 | 0.4538 | step 101 | 0.115 | 5.6 | 0.018 |
| `tcn_2d_sparse` | 25,152 | 1.0556 | 0.7630 | step 55 | 0.065 | 90.5 | 0.099 |
| `transformer_cafa_2d` | 55,376 | 1.4586 | 1.2864 | step 31 | 0.064 | 12.0 | 0.203 |
| `transformer_scan_2d` | 34,112 | 1.6361 | 1.3025 | step 35 | 0.065 | 55.7 | 0.050 |
| `cnn_2d_sparse` | 12,736 | 1.8688 | 1.5487 | step 45 | 0.069 | 106.5 | 0.141 |
| `transformer_flatten_2d` | 34,112 | 2.3187 | 1.9696 | step 47 | 0.073 | 40.9 | 0.022 |

⚠ marks a model whose loss never fell by 5% of where it started.

## Best on each question

- **lowest final loss** — `mamba_upstream_3d`, `lstm_3d`, `lstm_2d_sparse`
- **fastest to 90% of its own improvement** — `mamba3_2d`, `transformer_cafa_2d`, `transformer_scan_2d`
- **steadiest at the end** — `transformer_cafa_2d`, `tcn_2d_sparse`, `transformer_scan_2d`
- **most steps per second** — `cnn_2d_sparse`, `tcn_2d_sparse`, `lstm_2d_sparse`
- **most improvement per 1k parameters** — `s4d_portable_2d`, `transformer_cafa_2d`, `s4d_upstream_2d`
