"""Phase 5 acceptance for td.data. See PLAN.md.

The load-bearing test is that values land in the *right cells*, checked against
an independently built reference. A shape check would pass just as happily on a
mis-shuffled lattice, which is the exact failure this layer exists to prevent.
"""

import pytest
import torch
from torch.utils.data import DataLoader

import torch_dimensions as td
from torch_dimensions.data import (
    LatticeDataset,
    LatticeSource,
    LatticeWindow,
    TensorSource,
    collate_lattice,
    from_coords,
    from_table,
)

STATES = ("CA", "NY", "TX")
SKUS = ("a", "b")


def rows(n_time=8, skip=(("NY", "b"),)):
    """Long-format rows whose value encodes its own coordinates, so a
    misplacement is detectable rather than merely plausible."""
    coords, times, values = [], [], []
    for t in range(n_time):
        for s in STATES:
            for k in SKUS:
                if (s, k) in skip:
                    continue
                coords.append((s, k))
                times.append(2000 + t)
                values.append([t * 100.0 + STATES.index(s) * 10.0 + SKUS.index(k)])
    return coords, times, values


# -- from_coords -------------------------------------------------------------


def test_vocabularies_are_sorted_and_shape_follows():
    cm = from_coords([("TX", "b"), ("CA", "a"), ("CA", "b")], names=("state", "sku"))
    assert cm.vocabs == (("CA", "TX"), ("a", "b"))
    assert cm.lattice.shape == (2, 2)
    assert cm.lattice.axis_names == ("time", "state", "sku")


def test_unobserved_combinations_are_marked_absent():
    cm = from_coords([("CA", "a"), ("CA", "b"), ("TX", "a")])
    assert cm.lattice.n_valid == 3 and cm.lattice.n_cells == 4
    assert not cm.lattice.valid[1, 1]  # TX/b never seen


def test_a_fully_observed_grid_is_dense_not_a_mask_of_all_true():
    cm = from_coords([(s, k) for s in STATES for k in SKUS])
    assert cm.lattice.is_dense


def test_dense_flag_overrides_the_inference():
    cm = from_coords([("CA", "a"), ("TX", "b")], dense=True)
    assert cm.lattice.is_dense and cm.lattice.n_cells == 4


def test_time_axis_is_optional():
    assert not from_coords([("CA", "a")], time=False).lattice.time


def test_index_round_trips_through_decode():
    cm = from_coords([(s, k) for s in STATES for k in SKUS], names=("state", "sku"))
    pairs = [(s, k) for s in STATES for k in SKUS]
    for row, flat in zip(pairs, cm.index.tolist(), strict=True):
        assert cm.decode(flat) == row


def test_encode_matches_the_index_built_at_construction():
    coords = [("TX", "b"), ("CA", "a"), ("NY", "b")]
    cm = from_coords(coords)
    assert torch.equal(cm.encode(coords), cm.index)


def test_encode_refuses_unknown_values_rather_than_folding_them_to_zero():
    cm = from_coords([("CA", "a")], names=("state", "sku"))
    with pytest.raises(KeyError, match="'state'"):
        cm.encode([("ZZ", "a")])
    with pytest.raises(ValueError, match="expected 2 coordinates"):
        cm.encode([("CA",)])


def test_integer_coordinates_work_too():
    cm = from_coords(torch.tensor([[0, 1], [2, 0]]))
    assert cm.lattice.shape == (2, 2)


@pytest.mark.parametrize(
    ("coords", "match"),
    [([], "at least one"), ([("a", "b"), ("c",)], "same length")],
)
def test_from_coords_errors(coords, match):
    with pytest.raises(ValueError, match=match):
        from_coords(coords)


# -- from_table --------------------------------------------------------------


def test_values_land_in_the_cells_their_coordinates_name():
    """The whole point of the layer. Checked against the encoding baked into
    each value, not against another call to our own mapping."""
    coords, times, values = rows()
    table = from_table(coords, times, values, names=("state", "sku"))
    for t in range(len(table.times)):
        for si, s in enumerate(STATES):
            for ki, k in enumerate(SKUS):
                got = table.series[t, si, ki, 0].item()
                if (s, k) == ("NY", "b"):
                    assert got == 0.0, "absent cell must be exactly zero"
                else:
                    assert got == t * 100.0 + si * 10.0 + ki


def test_absent_cells_are_zero_and_marked():
    table = from_table(*rows(), names=("state", "sku"))
    assert table.lattice.n_valid == 5 and table.lattice.n_cells == 6
    assert table.series.masked_select(~table.lattice.valid.reshape(1, 3, 2, 1)).abs().max() == 0


def test_time_axis_is_the_sorted_distinct_timestamps():
    coords, times, values = rows(n_time=4)
    table = from_table(coords, times, values)
    assert table.times == (2000, 2001, 2002, 2003)
    assert len(table) == 4 and table.series.shape[0] == 4


def test_shape_and_feature_count():
    table = from_table(*rows(), names=("state", "sku"))
    assert table.series.shape == (8, 3, 2, 1)
    assert table.n_features == 1
    assert "5/6" in repr(table)


def test_duplicate_time_cell_rows_are_refused():
    """Keeping the last silently would turn a join bug into a plausible
    dataset."""
    with pytest.raises(ValueError, match="duplicate"):
        from_table([("CA", "a"), ("CA", "a")], [2000, 2000], [[1.0], [2.0]])


def test_the_same_cell_at_different_times_is_not_a_duplicate():
    table = from_table([("CA", "a"), ("CA", "a")], [2000, 2001], [[1.0], [2.0]])
    assert table.series.flatten().tolist() == [1.0, 2.0]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"values": [1.0, 2.0]}, r"\(N, F\)"),
        ({"times": [2000]}, "same number of rows"),
    ],
)
def test_from_table_errors(kwargs, match):
    base = {
        "coords": [("CA", "a"), ("CA", "b")],
        "times": [2000, 2000],
        "values": [[1.0], [2.0]],
    }
    with pytest.raises(ValueError, match=match):
        from_table(**{**base, **kwargs})


# -- windowing ---------------------------------------------------------------


def test_windows_tile_the_axis_with_targets_after_inputs():
    w = LatticeWindow(n_time=10, input_len=3, horizon=2)
    assert len(w) == 6
    assert w[0] == (0, 3, 3, 5)
    assert w[-1] == (5, 8, 8, 10)
    for win in w:
        assert win.x1 == win.y0, "a window must never see its own target"


def test_stride_controls_the_step():
    assert [win.x0 for win in LatticeWindow(10, 3, 0, stride=3)] == [0, 3, 6]


def test_horizon_zero_gives_no_target_range():
    w = LatticeWindow(6, 3)
    assert all(win.y0 == win.y1 for win in w)


def test_split_drops_windows_straddling_the_boundary():
    """The gap is deliberate: a straddling window would put post-cut timesteps
    inside a training input."""
    w = LatticeWindow(12, 3, 1)
    before, after = w.split(6)
    assert all(win.y1 <= 6 for win in before)
    assert all(win.x0 >= 6 for win in after)
    assert len(before) + len(after) < len(w)


def test_split_at_time_uses_timestamps():
    w = LatticeWindow(6, 2)
    times = [2000, 2001, 2002, 2003, 2004, 2005]
    by_index = w.split(3)
    by_time = w.split_at_time(times, 2003)
    assert [list(x) for x in by_time] == [list(x) for x in by_index]


def test_split_at_a_time_past_the_end_puts_everything_before():
    w = LatticeWindow(6, 2)
    before, after = w.split_at_time([2000, 2001, 2002, 2003, 2004, 2005], 2099)
    assert len(before) == len(w) and len(after) == 0


def test_slicing_preserves_the_window_type():
    w = LatticeWindow(10, 3)
    assert isinstance(w[:2], LatticeWindow) and len(w[:2]) == 2


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"input_len": 0}, "input_len"),
        ({"horizon": -1}, "horizon"),
        ({"stride": 0}, "stride"),
        ({"input_len": 9, "horizon": 5}, "exceeds"),
    ],
)
def test_window_errors(kwargs, match):
    with pytest.raises(ValueError, match=match):
        LatticeWindow(**{"n_time": 10, "input_len": 3, **kwargs})


# -- source / dataset / collate ----------------------------------------------


def test_tensor_source_satisfies_the_protocol():
    table = from_table(*rows(), names=("state", "sku"))
    src = TensorSource(table.series, table.lattice)
    assert isinstance(src, LatticeSource)
    assert len(src) == 8 and src[0:3].shape == (3, 3, 2, 1)


def test_source_rejects_a_series_that_does_not_match_the_lattice():
    table = from_table(*rows(), names=("state", "sku"))
    with pytest.raises(ValueError, match="lattice dims"):
        TensorSource(torch.zeros(8, 9, 9, 1), table.lattice)


def test_a_custom_source_needs_no_inheritance():
    """The protocol is the customization story — this class knows nothing
    about the library."""
    table = from_table(*rows(), names=("state", "sku"))

    class Custom:
        def __init__(self, series, lattice):
            self._s, self._l = series, lattice

        @property
        def lattice(self):
            return self._l

        def __len__(self):
            return self._s.shape[0]

        def __getitem__(self, sl):
            return self._s[sl]

    ds = LatticeDataset(Custom(table.series, table.lattice), LatticeWindow(8, 3, 1))
    assert ds[0]["x"].shape == (3, 3, 2, 1)


def test_dataset_yields_inputs_and_targets_without_the_lattice():
    table = from_table(*rows(), names=("state", "sku"))
    ds = LatticeDataset(TensorSource(table.series, table.lattice), LatticeWindow(8, 3, 1))
    sample = ds[0]
    assert set(sample) == {"x", "y", "window"}, "the lattice is static, not per-sample"
    assert sample.x.shape == (3, 3, 2, 1) and sample.y.shape == (1, 3, 2, 1)
    assert ds.lattice is table.lattice


def test_dataset_refuses_windows_that_run_past_the_source():
    table = from_table(*rows(n_time=4), names=("state", "sku"))
    with pytest.raises(ValueError, match="past the end"):
        LatticeDataset(TensorSource(table.series, table.lattice), LatticeWindow(8, 3, 1))


def test_dataset_refuses_an_empty_window_set():
    table = from_table(*rows(), names=("state", "sku"))
    empty = LatticeWindow(8, 3, 1).split(0)[0]
    with pytest.raises(ValueError, match="empty"):
        LatticeDataset(TensorSource(table.series, table.lattice), empty)


def test_collate_stacks_and_keeps_the_lattice_out():
    table = from_table(*rows(), names=("state", "sku"))
    ds = LatticeDataset(TensorSource(table.series, table.lattice), LatticeWindow(8, 3, 1))
    batch = collate_lattice([ds[0], ds[1], ds[2]])
    assert batch.x.shape == (3, 3, 3, 2, 1)
    assert batch.y.shape == (3, 1, 3, 2, 1)
    assert "lattice" not in batch and len(batch["windows"]) == 3


def test_collate_refuses_ragged_windows_rather_than_padding():
    table = from_table(*rows(), names=("state", "sku"))
    ds = LatticeDataset(TensorSource(table.series, table.lattice), LatticeWindow(8, 3, 1))
    a = ds[0]
    b = dict(a)
    b["x"] = a["x"][:2]
    with pytest.raises(ValueError, match="differing input lengths"):
        collate_lattice([a, b])


def test_collate_refuses_an_empty_batch():
    with pytest.raises(ValueError, match="empty"):
        collate_lattice([])


# -- end to end --------------------------------------------------------------


def test_table_to_dataloader_to_model_to_backward():
    table = from_table(*rows(n_time=16), names=("state", "sku"))
    windows = LatticeWindow(len(table), input_len=4, horizon=1)
    train, _ = windows.split(10)
    ds = LatticeDataset(TensorSource(table.series, table.lattice), train)
    dl = DataLoader(ds, batch_size=2, shuffle=True, collate_fn=collate_lattice)

    model = td.LSTM(d_model=6, n_layers=3, lattice=table.lattice, d_input=table.n_features)
    batch = next(iter(dl))
    out = model(batch.x)
    assert out.shape == (2, 4, 3, 2, 6)
    out.pow(2).mean().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_d_input_is_optional_when_the_data_is_already_d_model_wide():
    lat = td.Lattice(shape=(2, 3), time=True)
    model = td.LSTM(d_model=5, n_layers=3, lattice=lat)
    assert model(torch.randn(1, 4, 2, 3, 5)).shape == (1, 4, 2, 3, 5)


# -- multiprocessing safety ---------------------------------------------------


def _small_dataset():
    lat = td.Lattice(shape=(2, 3), time=True)
    source = TensorSource(torch.randn(8, 2, 3, 4), lat)
    return LatticeDataset(source, LatticeWindow(8, input_len=3, horizon=1))


def test_samples_and_batches_survive_pickling():
    """DataLoader workers send every Sample — and the collated Batch — through
    a pickled queue. `__getattr__ = dict.__getitem__` broke that: pickle probes
    optional dunders with getattr and tolerates AttributeError, not the
    KeyError a dict lookup raises. num_workers>0 crashed outright."""
    import pickle

    sample = _small_dataset()[0]
    back = pickle.loads(pickle.dumps(sample))
    assert torch.equal(back.x, sample.x) and torch.equal(back.y, sample.y)

    batch = collate_lattice([sample, sample])
    back = pickle.loads(pickle.dumps(batch))
    assert torch.equal(back.x, batch.x)


def test_a_missing_field_reads_as_absent_not_as_a_keyerror():
    """`getattr(sample, "y", None)` and `hasattr` must behave; a horizon-0
    sample simply has no target."""
    lat = td.Lattice(shape=(2, 3), time=True)
    source = TensorSource(torch.randn(8, 2, 3, 4), lat)
    sample = LatticeDataset(source, LatticeWindow(8, input_len=3, horizon=0))[0]
    assert getattr(sample, "y", None) is None
    assert not hasattr(sample, "y")
    with pytest.raises(AttributeError):
        _ = sample.y


def test_dataloader_with_worker_processes():
    """The end-to-end form of the pickling guarantee: real worker processes,
    real queues. This is the configuration every user with a large dataset
    reaches for first."""
    dl = DataLoader(_small_dataset(), batch_size=2, num_workers=2, collate_fn=collate_lattice)
    batch = next(iter(dl))
    assert batch.x.shape == (2, 3, 2, 3, 4) and batch.y.shape == (2, 1, 2, 3, 4)


def test_collate_refuses_mixed_target_presence():
    """Keying off samples[0] silently dropped every target whenever the first
    sample happened to lack one."""
    from torch_dimensions.data.source import Sample

    a = Sample(x=torch.zeros(3, 2), window=None)
    b = Sample(x=torch.zeros(3, 2), y=torch.ones(1, 2), window=None)
    with pytest.raises(ValueError, match="mixed-horizon"):
        collate_lattice([a, b])
    with pytest.raises(ValueError, match="mixed-horizon"):
        collate_lattice([b, a])


def test_split_at_time_refuses_unsorted_times():
    """An unsorted timestamp list used to produce a silently nonsensical
    split — the quietest possible leakage bug."""
    w = LatticeWindow(6, input_len=2, horizon=1)
    with pytest.raises(ValueError, match="sorted"):
        w.split_at_time([3, 1, 2, 5, 4, 6], 4)


# -- on-disk source and masked normalization ----------------------------------


def test_memmap_source_passes_the_source_conformance_check(tmp_path):
    lat = td.Lattice(shape=(3, 4), names=("h", "w"))
    series = torch.randn(20, 3, 4, 2)
    path = td.data.MemmapSource.write(tmp_path / "series.npy", series)
    source = td.data.MemmapSource(path, lat)
    report = td.testing.check_data_source(source)
    assert report, str(report)
    assert torch.allclose(source[0:20], series, atol=1e-6)


def test_a_memmap_source_reaches_a_worker_process(tmp_path):
    """The failure this class exists to demonstrate: a live mmap handle either
    fails to pickle or pickles into something invalid in the child, and under
    DataLoader(num_workers>0) that presents as a hang (DEBUG.md #9)."""
    import pickle

    lat = td.Lattice(shape=(2, 2), names=("a", "b"))
    series = torch.randn(6, 2, 2, 1)
    path = td.data.MemmapSource.write(tmp_path / "s.npy", series)
    source = td.data.MemmapSource(path, lat)
    _ = source[0:2]  # force the handle open *before* pickling
    revived = pickle.loads(pickle.dumps(source))
    assert revived._array is None, "the mmap handle travelled into the pickle"
    assert torch.allclose(revived[0:6], source[0:6])


def test_a_memmap_source_refuses_a_file_that_is_not_its_lattice(tmp_path):
    path = td.data.MemmapSource.write(tmp_path / "s.npy", torch.randn(5, 9, 9, 1))
    with pytest.raises(ValueError, match="lattice dims"):
        td.data.MemmapSource(path, td.Lattice(shape=(3, 4), names=("h", "w")))


def test_masked_stats_ignore_absent_cells():
    """A mean over a sparse lattice's structural zeros is dragged toward zero
    in proportion to the sparsity, and nothing about the model then looks
    wrong."""
    valid = torch.tensor([[True, False], [True, True]])
    lat = td.Lattice(shape=(2, 2), names=("a", "b"), valid=valid)
    series = torch.full((10, 2, 2, 1), 5.0)
    series[:, 0, 1] = 0.0  # the absent cell, zero as the library guarantees

    naive = series.mean()
    stats = td.data.masked_stats(series, lat)
    assert abs(float(naive) - 3.75) < 1e-5, "the naive mean is dragged toward zero"
    present = stats.mean.reshape(-1)[[0, 2, 3]]
    assert torch.allclose(present, torch.full((3,), 5.0)), stats.mean


def test_normalizer_round_trips():
    lat = td.Lattice(shape=(2, 3), names=("a", "b"))
    series = torch.randn(30, 2, 3, 2) * 4 + 7
    stats = td.data.masked_stats(series, lat)
    assert torch.allclose(stats.invert(stats.apply(series)), series, atol=1e-4)
    normalized = stats.apply(series)
    assert normalized.mean().abs() < 0.1 and abs(float(normalized.std()) - 1) < 0.2


def test_masked_stats_treat_nan_as_absent():
    lat = td.Lattice(shape=(2,), names=("a",))
    series = torch.full((8, 2, 1), 3.0)
    series[0:4, 0] = float("nan")
    stats = td.data.masked_stats(series, lat)
    assert torch.isfinite(stats.mean).all()
    assert abs(float(stats.mean.reshape(-1)[0]) - 3.0) < 1e-5
