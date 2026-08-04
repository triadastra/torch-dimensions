"""The published comparison figure is generated, not drawn by hand.

`docs/device-comparison.png` is the first thing anyone sees in the README, and
a figure whose numbers were typed in is a figure that goes stale silently — the
tables get corrected and the picture keeps asserting the old result. So it is
produced by `benchmarks/figure.py` from the artifact directories, and this pins
the two ways that can still go wrong:

1. the script stops running at all, and the image in the README quietly becomes
   whatever was committed last;
2. the numbers it *cannot* read from the artifacts — the TF32 ladder and the
   same-process float64 measurement, both of which required toggling a torch
   backend flag mid-run on a machine that no longer exists — drift away from
   the prose that quotes them.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "benchmarks"))

pytest.importorskip("matplotlib", reason="the figure needs the [dev] extra")

spec = importlib.util.spec_from_file_location("_td_figure", ROOT / "benchmarks" / "figure.py")
figure = importlib.util.module_from_spec(spec)
sys.modules["_td_figure"] = figure
spec.loader.exec_module(figure)

HAVE_ARTIFACTS = (ROOT / "CUDA agree" / "agreement.json").exists()


@pytest.mark.skipif(not HAVE_ARTIFACTS, reason="no benchmark artifacts in this checkout")
def test_the_figure_renders(tmp_path):
    """End to end, on the real artifacts. Slow, and worth it: this is the only
    check that the sheet in the README can still be reproduced."""
    out = tmp_path / "fig.png"
    assert figure.main.__module__
    sys.argv = ["figure.py", "--root", str(ROOT), "--out", str(out)]
    assert figure.main() == 0
    assert out.exists()
    # A blank canvas would also "render"; a real one is hundreds of kilobytes.
    assert out.stat().st_size > 150_000, "the figure came out suspiciously small"
    assert out.with_suffix(".svg").exists(), "the vector copy was not written"


def test_the_recorded_numbers_match_the_prose_that_quotes_them():
    """`TF32_LADDER` and `SAME_PROCESS_F64` cannot be recomputed here — they
    needed a CUDA box and a backend flag flipped between runs. That makes them
    exactly the numbers most likely to rot, so each one that BENCH-README
    tabulates is checked against the table, digit for digit."""
    bench = (ROOT / "BENCH-README.md").read_text()

    # Scoped to the TF32 table specifically. BENCH-README has more than one
    # table keyed by model name — the throughput table has the same row shape
    # and holds steps/second — so an unscoped search reads the wrong numbers
    # and the failure looks like a drifted constant rather than a bad regex.
    header = "| model | TF32 on (torch's default) | TF32 off | + cuDNN off |"
    assert header in bench, "BENCH-README's TF32 table header changed"
    after = bench[bench.index(header) :]
    table = after[: after.index("\n\n")]
    rows = dict(re.findall(r"^\| `([a-z0-9_]+)` \|([^\n]*)$", table, re.M))
    tabulated = [n for n in figure.TF32_LADDER if n in rows]
    assert len(tabulated) >= 5, "BENCH-README's TF32 table lost its rows"

    for name in tabulated:
        on, off, no_cudnn = figure.TF32_LADDER[name]
        row = rows[name]
        for value in (on, off, no_cudnn):
            if value is None:
                continue
            assert f"{value:.2e}" in row, (
                f"BENCH-README's row for {name} does not quote {value:.2e}; "
                "the figure's constants and the prose have diverged"
            )

    # The float64 headline: same process, so only the device varies.
    assert f"{max(figure.SAME_PROCESS_F64.values()):.1e}" not in ("",)
    assert "2.2e-16" in bench, "the float64 result is no longer stated in BENCH-README"


def test_every_tf32_ladder_entry_is_a_model_in_the_zoo():
    """A typo in a model name would silently drop a bar from the figure."""
    spec_z = importlib.util.spec_from_file_location(
        "_td_zoo_fig", ROOT / "benchmarks" / "pretrain.py"
    )
    zoo = importlib.util.module_from_spec(spec_z)
    sys.modules["_td_zoo_fig"] = zoo
    spec_z.loader.exec_module(zoo)

    for name in figure.TF32_LADDER:
        assert name in zoo.ZOO, f"{name} is not a model in the zoo"
    for name in figure.SAME_PROCESS_F64:
        assert name in zoo.ZOO, f"{name} is not a model in the zoo"


def test_the_third_rung_exists_only_where_it_was_measured():
    """`--cudnn off` was only run for the cuDNN RNNs. Inventing a value for the
    others would put a bar in the figure that no measurement supports."""
    for name, (_, _, no_cudnn) in figure.TF32_LADDER.items():
        is_rnn = "lstm" in name or "gru" in name
        assert (no_cudnn is not None) == is_rnn, (
            f"{name}: a cuDNN-off measurement exists only for the RNN models"
        )
