#!/usr/bin/env bash
# Every reproduction, in the order RESULTS.md lists them.
#
#   bash examples/repro/run_all.sh            # the full set
#   TD_EPOCHS=1 bash examples/repro/run_all.sh   # a fast smoke of the same commands
#
# Datasets download once into ~/.cache/torch_dimensions (override TD_DATA_DIR)
# and are checksummed on every read. Results append to results.json; nothing is
# overwritten, so a re-run adds a row rather than replacing the one that
# disagreed. Regenerate the table with `python -m examples.repro.report`.
set -euo pipefail

cd "$(dirname "$0")/../.."
PY=${PY:-python}
E=${TD_EPOCHS:-}

echo "== sequence: the mixer without the lattice"
$PY -m examples.repro.smnist --epochs "${E:-20}"
$PY -m examples.repro.smnist --permuted --epochs "${E:-20}"

echo "== 2-D lattice: the N-D machinery on images"
$PY -m examples.repro.image_nd --model mamba_nd --epochs "${E:-6}"
$PY -m examples.repro.image_nd --model s4d_nd --epochs "${E:-6}"

echo "== sparse lattice: the claim with no published baseline"
$PY -m examples.repro.forecast_sparse --arm masked --arm zeros --arm cafa \
  --arm attention --epochs "${E:-8}"

$PY -m examples.repro.report --out RESULTS.md
