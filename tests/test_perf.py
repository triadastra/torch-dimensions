"""A perf smoke that cannot flake on a shared runner.

PLAN.md asks for "one tiny timed run with a generous regression threshold" to
catch an accidental O(n²) without flaking on runner noise. An *absolute*
threshold does exactly the flaking it is meant to avoid: a loaded CI box is
5x slower than a quiet laptop and the number means nothing on either.

So these assert on **ratios between two configurations measured in the same
process, seconds apart**. Machine speed cancels; complexity does not. A model
whose cost went quadratic in the cell count fails here on any hardware, and a
runner that is merely slow fails nothing.

The thresholds are deliberately loose — 4x headroom over the expected ratio —
because the failure being hunted is an order of magnitude, not a regression of
twenty percent. Twenty percent is what BENCHMARKS.md is for.
"""

import time

import pytest
import torch

import torch_dimensions as td


def median_ms(fn, repeat=5, warmup=2):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return sorted(samples)[len(samples) // 2] * 1e3


def forward_of(model, x):
    def run():
        with torch.no_grad():
            model(x)

    return run


def test_cost_is_linear_ish_in_cells_not_quadratic():
    """Four times the cells must not cost sixteen times as much.

    The scan family is O(cells) by construction: each layer sweeps one axis and
    every cell is visited once. If a fold ever materialized a cell-by-cell
    matrix, this ratio would jump from ~4 to ~16 and nothing else in the suite
    would notice."""
    torch.manual_seed(0)
    small = td.Lattice(shape=(8, 8), names=("h", "w"))
    large = td.Lattice(shape=(16, 16), names=("h", "w"))  # 4x the cells
    a = td.LSTM(32, 4, small)
    b = td.LSTM(32, 4, large)
    x_small = torch.randn(4, *small.shape, 32)
    x_large = torch.randn(4, *large.shape, 32)

    ratio = median_ms(forward_of(b, x_large)) / max(median_ms(forward_of(a, x_small)), 1e-6)
    assert ratio < 16, f"4x the cells cost {ratio:.1f}x the time; expected ~4, quadratic is ~16"


def test_the_factorized_kernel_does_not_scale_with_cells_squared():
    """CaFA's whole justification: one kernel per *axis*, not per cell pair.
    A regression to a dense joint operator would be O(cells²) and would show up
    here long before it showed up as an out-of-memory on somebody's rank-4
    lattice."""
    torch.manual_seed(0)
    small = td.Lattice(shape=(6, 6), names=("h", "w"), time=True)
    large = td.Lattice(shape=(12, 12), names=("h", "w"), time=True)  # 4x cells
    a = td.LSTM(16, 3, small, method=td.cafa)
    b = td.LSTM(16, 3, large, method=td.cafa)
    x_small = torch.randn(2, 3, *small.shape, 16)
    x_large = torch.randn(2, 3, *large.shape, 16)

    ratio = median_ms(forward_of(b, x_large)) / max(median_ms(forward_of(a, x_small)), 1e-6)
    assert ratio < 24, f"4x the cells cost {ratio:.1f}x; a dense joint operator would be ~16-64x"


def test_depth_is_linear_in_layers():
    """Twice the layers, about twice the work. A quadratic here would mean a
    layer is seeing the whole stack's activations."""
    torch.manual_seed(0)
    lat = td.Lattice(shape=(8, 8), names=("h", "w"))
    x = torch.randn(4, *lat.shape, 32)
    four = median_ms(forward_of(td.LSTM(32, 4, lat), x))
    eight = median_ms(forward_of(td.LSTM(32, 8, lat), x))
    ratio = eight / max(four, 1e-6)
    assert ratio < 4, f"2x the layers cost {ratio:.1f}x the time; expected ~2"


@pytest.mark.parametrize("rank", [2, 3])
def test_spreading_cells_over_more_axes_does_not_cost_more(rank):
    """BENCHMARKS.md's finding, pinned as a property: cost tracks the length of
    the swept axis, so redistributing the same cells over more axes should be
    no worse. If this ever inverts, the fold has started charging per axis."""
    torch.manual_seed(0)
    flat = td.Lattice(shape=(64,), names=("a",))
    side = round(64 ** (1 / rank))
    spread = td.Lattice(shape=(side,) * rank, names=tuple(f"a{i}" for i in range(rank)))
    x_flat = torch.randn(4, *flat.shape, 32)
    x_spread = torch.randn(4, *spread.shape, 32)

    flat_ms = median_ms(forward_of(td.LSTM(32, 4, flat), x_flat))
    spread_ms = median_ms(forward_of(td.LSTM(32, 4, spread), x_spread))
    assert spread_ms < flat_ms * 3, (
        f"rank {rank} over {spread.n_cells} cells cost {spread_ms / flat_ms:.1f}x the "
        f"rank-1 sweep over {flat.n_cells}; the fold should not charge per axis"
    )
