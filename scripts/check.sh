#!/usr/bin/env bash
# Exactly what CI runs, in the same order, with the same commands.
#
#   bash scripts/check.sh
#
# This file exists because "I ran the linter" and "I ran what CI runs" have
# been different things three times now, and the difference is always found by
# a red build rather than by a person. DEBUG.md #7 is the original: a tool that
# was configured and never executed. The variants since have been subtler —
# running `ruff check src tests` when CI runs `ruff check .` (the repo's own
# markdown code blocks are formatted too), or watching a configured pipeline
# instead of running the gate locally.
#
# If CI grows a step, it goes here in the same commit. If this passes and CI
# fails, that is a bug in this file.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== ruff check ."
ruff check .

echo "== ruff format --check ."
ruff format --check .

echo "== mypy (advisory in CI, so it does not gate here either)"
mypy || echo "   ^ advisory: not failing the run"

echo "== pytest with the coverage floor"
pytest -q --cov --cov-report=term-missing

echo
echo "all green — this is what CI will run"
