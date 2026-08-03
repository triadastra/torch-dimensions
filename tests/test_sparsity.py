"""`td.data.sparsity` — how full the lattice is, measured rather than declared.

The headline is one number, so the tests are mostly about the number being
right for each way of asking, and about the per-axis breakdown separating
"scattered gaps" from "one whole slice missing" — the second is usually a join
that went wrong upstream, and it is the thing worth catching before training.
"""

from __future__ import annotations

import pytest
import torch

import torch_dimensions as td


def _checkerboard(shape):
    idx = torch.arange(int(torch.tensor(shape).prod())).reshape(shape)
    return (idx % 2) == 0


def test_dense_lattice_reports_zero_percent():
    report = td.data.sparsity(td.Lattice(shape=(4, 5), names=("a", "b")))
    assert report.dense
    assert report.present == report.total == 20
    assert report.percent_sparse == 0.0
    assert report.absent == 0
    assert "dense" in report.summary()


def test_sparse_lattice_counts_and_percentage():
    valid = _checkerboard((6, 8))
    report = td.data.sparsity(td.Lattice(shape=(6, 8), names=("h", "w"), valid=valid))
    assert report.total == 48
    assert report.present == int(valid.sum())
    assert report.percent_sparse == pytest.approx(50.0)
    assert not report.dense


def test_the_reported_shape_is_the_lattice_not_the_broadcast_mask():
    """`Lattice.mask()` is broadcast-shaped `(1, 1, *shape, 1)` for multiplying
    against data. Reporting that would name singleton batch and feature axes as
    lattice axes and push the real names off the end."""
    lat = td.Lattice(shape=(6, 8), names=("h", "w"), valid=_checkerboard((6, 8)), time=True)
    report = td.data.sparsity(lat)
    assert report.shape == (6, 8)
    assert report.names == ("h", "w")
    assert set(report.per_axis) == {"h", "w"}
    assert len(report.per_axis["h"]) == 6
    assert len(report.per_axis["w"]) == 8


def test_an_entirely_missing_slice_is_singled_out():
    valid = torch.ones(6, 8, dtype=torch.bool)
    valid[:, 3] = False  # one column never observed
    report = td.data.sparsity(td.Lattice(shape=(6, 8), names=("h", "w"), valid=valid))
    assert report.empty_slices["w"] == [3]
    assert report.empty_slices["h"] == []
    assert "⚠" in report.summary()


def test_per_axis_counts_sum_consistently():
    valid = _checkerboard((4, 5))
    report = td.data.sparsity(td.Lattice(shape=(4, 5), names=("a", "b"), valid=valid))
    for counts in report.per_axis.values():
        assert sum(counts) == report.present


def test_a_data_tensor_with_holes_is_measured_directly():
    """The pre-run case: raw data, absence marked by non-finite values."""
    values = torch.randn(10, 6, 8, 2)
    values[:, :, 3, :] = float("nan")  # a column with no observation at all
    values[0:5, 2, 5, :] = float("nan")  # a cell observed only some of the time
    report = td.data.sparsity(values, shape=(6, 8), names=("h", "w"))

    assert report.total == 48
    assert report.present == 42  # the whole column is gone; the partial cell stays
    assert report.percent_sparse == pytest.approx(12.5)
    assert report.empty_slices["w"] == [3]
    # `observed` counts entries, not cells, so the partially-seen cell shows up
    # here even though it is present: 10*6*2 entries for the dead column plus
    # 5*2 for the half-seen cell, out of 10*6*8*2.
    assert report.observed == pytest.approx((960 - 120 - 10) / 960)


def test_a_sentinel_can_stand_in_for_missing():
    values = torch.ones(4, 5, 3)
    values[:, 2, :] = 0.0
    report = td.data.sparsity(values, shape=(4, 5), names=("a", "b"), missing=0.0)
    assert report.empty_slices["b"] == [2]
    assert report.present == 16


def test_a_boolean_mask_is_accepted_as_is():
    mask = _checkerboard((3, 4))
    report = td.data.sparsity(mask, names=("x", "y"))
    assert report.present == int(mask.sum())
    assert report.shape == (3, 4)


def test_a_table_reports_both_structure_and_observation():
    coords = [("a", "x"), ("a", "y"), ("b", "x")]
    times = [0, 0, 0]
    values = torch.tensor([[1.0], [2.0], [3.0]])
    table = td.data.from_table(coords, times, values, names=("s", "k"))
    report = td.data.sparsity(table)
    # 2 x 2 combinations, three of them observed
    assert report.total == 4
    assert report.present == 3
    assert report.percent_sparse == pytest.approx(25.0)
    assert report.observed is not None


def test_a_data_tensor_needs_its_shape_stated():
    with pytest.raises(ValueError, match="needs the lattice shape"):
        td.data.sparsity(torch.randn(10, 6, 8))


def test_an_ambiguous_placement_is_refused_rather_than_guessed():
    """A (6, 6) lattice inside a (6, 6, 6) tensor has no single right reading,
    and silently picking one is how a transposed axis survives to training."""
    with pytest.raises(ValueError, match="more than once"):
        td.data.sparsity(torch.randn(6, 6, 6), shape=(6, 6))


def test_a_shape_that_is_not_there_is_refused():
    with pytest.raises(ValueError, match="does not appear"):
        td.data.sparsity(torch.randn(10, 6, 8), shape=(5, 5))


def test_report_repr_leads_with_the_number_asked_for():
    report = td.data.sparsity(td.Lattice(shape=(2, 2), names=("a", "b")))
    assert "percent_sparse" in repr(report)
