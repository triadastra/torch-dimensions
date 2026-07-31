"""DEBUG.md cites tests and commits by name. Citations rot; this fails when they do.

A document asserting "guarded by ``test_x``" is a comment stating an invariant
(DEBUG.md §A1), so the citations themselves get a test. Renaming a guard,
deleting a test file, or rewriting history now breaks the build instead of
silently orphaning the document.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DEBUG = ROOT / "DEBUG.md"


def _text() -> str:
    if not DEBUG.exists():
        pytest.fail("DEBUG.md is linked from CONTRIBUTING.md but does not exist")
    return DEBUG.read_text(encoding="utf-8")


def test_every_cited_test_name_exists():
    """Every ``test_*`` token in DEBUG.md is a real test function or file."""
    cited = set(re.findall(r"\btest_[a-z0-9_]+\b", _text()))
    assert cited, "DEBUG.md cites no tests; if that became deliberate, delete this test"

    defined: set[str] = set()
    stems: set[str] = set()
    for f in (ROOT / "tests").glob("test_*.py"):
        stems.add(f.stem)
        defined |= set(re.findall(r"def (test_[a-z0-9_]+)", f.read_text(encoding="utf-8")))

    missing = sorted(cited - defined - stems)
    assert not missing, f"DEBUG.md cites tests that no longer exist: {missing}"


def test_every_cited_commit_resolves():
    """Every commit hash in DEBUG.md resolves in this repository.

    A hash must mix digits and letters to count — every hash the file actually
    cites does, and the constraint keeps 7-digit *numbers* from bug output
    (e.g. ``1201025.71``) from being mistaken for one.
    """
    cited = set(re.findall(r"\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*[0-9])[0-9a-f]{7,40}\b", _text()))
    assert cited, "DEBUG.md cites no commits; if that became deliberate, delete this test"

    if not (ROOT / ".git").exists():
        pytest.skip("not a git checkout (sdist or archive export)")
    shallow = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--is-shallow-repository"],
        capture_output=True,
        text=True,
    )
    if shallow.stdout.strip() == "true":
        # A shallow clone is missing old commits by design, not by rewrite.
        # Skipping visibly beats failing wrongly — and beats passing silently,
        # which is why CI fetches full history to run this for real.
        pytest.skip("shallow clone: cited commits predate the fetch depth")
    for sha in sorted(cited):
        proc = subprocess.run(
            ["git", "-C", str(ROOT), "cat-file", "-t", sha],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0 and proc.stdout.strip() == "commit", (
            f"DEBUG.md cites commit {sha}, which does not resolve to a commit; "
            "either history was rewritten or the hash is a typo"
        )
