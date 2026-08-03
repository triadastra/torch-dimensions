# Scorecard — Apple arm64 (MPS)

400 steps, batch 4, seed 20260803, torch 2.13.0, torch-dimensions 0.3.1.

One task — a cumulative sum along an axis — on one machine. That suits a
causal sequence model and is close to the worst case for a permutation-
invariant one, so this ranks models *on this problem* and implies nothing
about images or forecasting. There is no combined score on purpose: the
weighting between a point of loss and a step per second is the reader's,
not this file's.

| model | params | final loss | best | 90% at | tail spread | steps/s | Δloss per 1k params |
|---|---|---|---|---|---|---|---|
| `mamba_upstream_3d` | 50,752 | 0.0042 | 0.0035 | step 53 | 0.112 | 25.6 | 0.054 |
| `lstm_3d` | 51,136 | 0.0045 | 0.0011 | step 39 | 0.655 | 59.2 | 0.077 |
| `mamba3_2d` | 141,472 | 0.6098 | 0.4131 | step 22 | 0.122 | 12.0 | 0.026 |
| `gru_2d_sparse` | 38,464 | 0.6109 | 0.4073 | step 47 | 0.127 | 30.4 | 0.084 |
| `mamba_upstream_2d` | 33,856 | 0.6111 | 0.4048 | step 78 | 0.126 | 35.2 | 0.087 |
| `lstm_2d_sparse` | 51,136 | 0.6116 | 0.4030 | step 47 | 0.131 | 70.5 | 0.059 |
| `mamba_portable_2d` | 33,856 | 0.6124 | 0.4054 | step 78 | 0.126 | 39.9 | 0.087 |
| `mamba2_2d` | 111,920 | 0.6143 | 0.4340 | step 91 | 0.119 | 5.4 | 0.019 |
| `s4_upstream_2d` | 17,216 | 0.6227 | 0.4246 | step 85 | 0.124 | 35.5 | 0.152 |
| `s4d_upstream_2d` | 15,168 | 0.6252 | 0.4232 | step 85 | 0.123 | 41.3 | 0.175 |
| `s4d_portable_2d` | 13,120 | 0.6378 | 0.4188 | step 91 | 0.125 | 56.7 | 0.206 |
| `tcn_2d_sparse` | 25,152 | 1.0288 | 0.7823 | step 45 | 0.110 | 81.3 | 0.098 |
| `transformer_cafa_2d` | 55,376 | 1.4103 | 1.0995 | step 25 | 0.082 | 10.1 | 0.206 |
| `transformer_scan_2d` | 34,112 | 1.6962 | 1.3074 | step 74 | 0.080 | 49.8 | 0.049 |
| `cnn_2d_sparse` | 12,736 | 2.2291 | 1.5628 | step 45 | 0.106 | 97.7 | 0.140 |
| `transformer_flatten_2d` | 34,112 | 2.7320 | 1.9946 | step 164 | 0.104 | 35.7 | 0.021 |

⚠ marks a model whose loss never fell by 5% of where it started.

## Best on each question

- **lowest final loss** — `mamba_upstream_3d`, `lstm_3d`, `mamba3_2d`
- **fastest to 90% of its own improvement** — `mamba3_2d`, `transformer_cafa_2d`, `lstm_3d`
- **steadiest at the end** — `transformer_scan_2d`, `transformer_cafa_2d`, `transformer_flatten_2d`
- **most steps per second** — `cnn_2d_sparse`, `tcn_2d_sparse`, `lstm_2d_sparse`
- **most improvement per 1k parameters** — `s4d_portable_2d`, `transformer_cafa_2d`, `s4d_upstream_2d`
