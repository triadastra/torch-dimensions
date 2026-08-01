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
# If CI grows a step, it goes here in the same commit.
#
# Tools are invoked as `python -m <tool>`, never as bare `ruff`/`mypy`/`pytest`.
# That is not style. A bare `ruff` resolves against PATH, and on the machine
# this was written on PATH led to a conda ruff 0.6.7 while the project's dev
# extra installs 0.16.1 — two versions whose rule sets genuinely differ. The
# old one failed this script on UP038, a rule the new one has *removed*. A
# gate that reports failures CI cannot have is worse than no gate: it teaches
# you to ignore it.
#
# What this CANNOT do: make your environment match CI's. It makes the commands
# identical, not the interpreter, the installed extras, or the transitive
# dependencies that happen to be lying around. DEBUG.md #24 is exactly that
# gap — numpy is not a torch dependency, this laptop had it, CI did not, and
# this script passed while the build went red. A green run here means "the
# checks pass in this environment"; only CI can say "in a clean one".
set -euo pipefail
cd "$(dirname "$0")/.."

# The project's interpreter, not whatever is first on PATH. Override with
# PYTHON=... for a different environment.
if [ -z "${PYTHON:-}" ]; then
  if [ -x ".venv/bin/python" ]; then PYTHON=".venv/bin/python"; else PYTHON="python3"; fi
fi

echo "== environment (a mismatch here is the usual cause of a surprise red build)"
echo "   python: $($PYTHON -c 'import sys; print(sys.executable)')"
echo "   ruff:   $($PYTHON -m ruff --version 2>/dev/null || echo MISSING)"
echo "   pytest: $($PYTHON -m pytest --version 2>&1 | head -1)"
echo

echo "== ruff check ."
$PYTHON -m ruff check .

echo "== ruff format --check ."
$PYTHON -m ruff format --check .

echo "== mypy (advisory in CI, so it does not gate here either)"
$PYTHON -m mypy || echo "   ^ advisory: not failing the run"

echo "== pytest with the coverage floor"
$PYTHON -m pytest -q --cov --cov-report=term-missing

echo
echo "all green — this is what CI will run"
