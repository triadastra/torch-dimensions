"""Phase 1 acceptance for Lattice. See PLAN.md.

These tests are deliberately heavier than the module's line count justifies:
an axis-order bug here is invisible at this layer and presents as a bad model
three phases later.
"""

import math

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from torch_dimensions import Lattice

RANKS = [1, 2, 3, 4, 5]


def _shape(rank: int) -> tuple[int, ...]:
    return tuple(range(2, 2 + rank))  # (2,), (2,3), (2,3,4), ...


def _tensor(lat: Lattice, batch: int = 2, t: int = 3, h: int = 4) -> torch.Tensor:
    lead = (batch, t) if lat.time else (batch,)
    return torch.randn(*lead, *lat.shape, h)


# -- structure --------------------------------------------------------------


def test_defaults_and_names():
    lat = Lattice(shape=(4, 5))
    assert lat.rank == 2 and lat.n_axes == 2
    assert lat.axis_names == ("dim0", "dim1")
    assert lat.tensor_ndim == 4
    assert lat.is_dense and lat.n_valid == lat.n_cells == 20


def test_time_is_a_normal_axis_but_has_no_static_size():
    lat = Lattice(shape=(4, 5), names=("h", "w"), time=True)
    assert lat.n_axes == 3
    assert lat.axis_names == ("time", "h", "w")
    assert lat.tensor_ndim == 5
    assert lat.axis_index("time") == 0
    assert lat.tensor_dim("h") == 2
    assert lat.axis_size("w") == 5
    with pytest.raises(ValueError, match="not a lattice axis"):
        lat.axis_size("time")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"shape": ()}, "at least one axis"),
        ({"shape": (0, 3)}, "must be positive"),
        ({"shape": (2, 3), "names": ("a",)}, "names for"),
        ({"shape": (2, 3), "names": ("a", "a")}, "unique"),
        ({"shape": (2, 3), "names": ("time", "b"), "time": True}, "reserved"),
    ],
)
def test_construction_errors(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Lattice(**kwargs)


def test_valid_mask_errors():
    with pytest.raises(ValueError, match="expected"):
        Lattice(shape=(2, 3), valid=torch.ones(3, 2, dtype=torch.bool))
    with pytest.raises(ValueError, match="no cells"):
        Lattice(shape=(2, 3), valid=torch.zeros(2, 3, dtype=torch.bool))


def test_unknown_axis_names_itself():
    lat = Lattice(shape=(4, 5), names=("h", "w"))
    with pytest.raises(KeyError, match="nope"):
        lat.axis_index("nope")
    with pytest.raises(IndexError):
        lat.axis_index(2)
    assert lat.axis_index(-1) == 1


# -- permutation ------------------------------------------------------------


@pytest.mark.parametrize("rank", RANKS)
@pytest.mark.parametrize("time", [False, True])
def test_inverse_permutation_matches_argsort(rank, time):
    """The inverse is hand-built in the module; argsort is the independent check."""
    lat = Lattice(shape=_shape(rank), time=time)
    for axis in range(lat.n_axes):
        perm, inv = lat.permutation(axis)
        expected = tuple(torch.argsort(torch.tensor(perm)).tolist())
        assert inv == expected, f"axis={axis}, perm={perm}"


@pytest.mark.parametrize("rank", RANKS)
@pytest.mark.parametrize("time", [False, True])
def test_sequence_round_trip_is_the_identity(rank, time):
    lat = Lattice(shape=_shape(rank), time=time)
    x = _tensor(lat)
    for axis in range(lat.n_axes):
        seq, restore = lat.to_sequence(x, axis)
        assert torch.equal(lat.from_sequence(seq, restore), x), f"axis={axis}"


@pytest.mark.parametrize("rank", RANKS)
def test_folded_shape_is_batch_times_every_other_axis(rank):
    lat = Lattice(shape=_shape(rank), time=True)
    b, t, h = 2, 3, 4
    x = _tensor(lat, b, t, h)
    sizes = (t, *lat.shape)
    for axis in range(lat.n_axes):
        seq, _ = lat.to_sequence(x, axis)
        a = sizes[axis]
        assert seq.shape == (b * math.prod(sizes) // a, a, h)


def test_sequence_preserves_values_along_the_swept_axis():
    """Round-tripping can hide a transposition that cancels itself. Check the
    swept axis actually carries the data it should."""
    lat = Lattice(shape=(3, 4), names=("h", "w"))
    x = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
    seq, _ = lat.to_sequence(x, "h")
    assert seq.shape == (2 * 4, 3, 5)
    # first folded row is batch 0, w 0 -> x[0, :, 0, :]
    assert torch.equal(seq[0], x[0, :, 0, :])


def test_shape_validation():
    lat = Lattice(shape=(3, 4))
    with pytest.raises(ValueError, match="expected a 4-D tensor"):
        lat.to_sequence(torch.randn(2, 3, 4), 0)
    with pytest.raises(ValueError, match="lattice dims"):
        lat.to_sequence(torch.randn(2, 3, 9, 5), 0)


# -- sparse cells -----------------------------------------------------------


@pytest.mark.parametrize("rank", RANKS)
def test_scatter_gather_round_trip_dense(rank):
    lat = Lattice(shape=_shape(rank))
    x = torch.randn(2, lat.n_valid, 4)
    assert torch.equal(lat.gather(lat.scatter(x)), x)


@pytest.mark.parametrize("rank", [1, 2, 3, 4])
@pytest.mark.parametrize("time", [False, True])
def test_scatter_gather_round_trip_sparse(rank, time):
    shape = _shape(rank)
    torch.manual_seed(rank)
    valid = torch.rand(shape) > 0.4
    valid.reshape(-1)[0] = True  # never empty
    lat = Lattice(shape=shape, valid=valid, time=time)
    lead = (2, 3) if time else (2,)
    x = torch.randn(*lead, lat.n_valid, 4)
    dense = lat.scatter(x)
    assert dense.shape == (*lead, *shape, 4)
    assert torch.equal(lat.gather(dense), x)


def test_scatter_zeroes_cells_that_do_not_exist():
    valid = torch.tensor([[True, False], [False, True]])
    lat = Lattice(shape=(2, 2), valid=valid)
    dense = lat.scatter(torch.ones(1, 2, 3))
    assert torch.equal(dense[0, 0, 1], torch.zeros(3))
    assert torch.equal(dense[0, 1, 0], torch.zeros(3))
    assert torch.equal(dense[0, 0, 0], torch.ones(3))


def test_scatter_rejects_wrong_cell_count():
    lat = Lattice(shape=(2, 2), valid=torch.tensor([[True, False], [False, True]]))
    with pytest.raises(ValueError, match="expected 2 cells"):
        lat.scatter(torch.ones(1, 4, 3))


def test_mask_broadcasts_over_batch_and_features():
    valid = torch.tensor([[True, False], [False, True]])
    lat = Lattice(shape=(2, 2), valid=valid, time=True)
    m = lat.mask(torch.float32)
    assert m.shape == (1, 1, 2, 2, 1)
    x = torch.randn(2, 3, 2, 2, 4)
    assert (x * m)[:, :, 0, 1, :].abs().sum() == 0


def test_dense_mask_is_all_true():
    lat = Lattice(shape=(2, 3))
    assert lat.mask().all() and lat.mask().shape == (1, 2, 3, 1)


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_valid_counts_matches_manual_sum(rank):
    shape = _shape(rank)
    torch.manual_seed(7)
    valid = torch.rand(shape) > 0.3
    valid.reshape(-1)[0] = True
    lat = Lattice(shape=shape, valid=valid)
    for i in range(rank):
        others = tuple(j for j in range(rank) if j != i)
        expected = valid.sum(dim=others) if others else valid.to(torch.long)
        assert torch.equal(lat.valid_counts(i), expected.float().clamp_min(1.0))


def test_valid_counts_dense_is_the_product_of_other_axes():
    lat = Lattice(shape=(2, 3, 4))
    assert torch.equal(lat.valid_counts(0), torch.full((2,), 12.0))
    assert torch.equal(lat.valid_counts(2), torch.full((4,), 6.0))


def test_valid_counts_never_zero_so_division_is_safe():
    valid = torch.tensor([[True, True], [False, False]])
    lat = Lattice(shape=(2, 2), valid=valid)
    assert torch.equal(lat.valid_counts(0), torch.tensor([2.0, 1.0]))


# -- misc -------------------------------------------------------------------


def test_to_is_a_noop_for_dense_and_copies_for_sparse():
    dense = Lattice(shape=(2, 3))
    assert dense.to("cpu") is dense
    sparse = Lattice(shape=(2, 2), valid=torch.tensor([[True, False], [True, True]]))
    moved = sparse.to("cpu")
    assert moved is not sparse and moved.n_valid == 3


def test_repr_reports_sparsity():
    lat = Lattice(shape=(2, 2), valid=torch.tensor([[True, False], [True, True]]), time=True)
    r = repr(lat)
    assert "3/4" in r and "time=True" in r


# -- property-based ---------------------------------------------------------


@settings(deadline=None, max_examples=40, suppress_health_check=[HealthCheck.too_slow])
@given(
    sizes=st.lists(st.integers(min_value=1, max_value=4), min_size=1, max_size=5),
    time=st.booleans(),
    density=st.floats(min_value=0.2, max_value=1.0),
    seed=st.integers(min_value=0, max_value=2**16),
)
def test_round_trips_hold_for_arbitrary_lattices(sizes, time, density, seed):
    shape = tuple(sizes)
    torch.manual_seed(seed)
    valid = torch.rand(shape) < density
    valid.reshape(-1)[0] = True
    lat = Lattice(shape=shape, valid=valid, time=time)

    lead = (2, 3) if time else (2,)
    sparse = torch.randn(*lead, lat.n_valid, 2)
    dense = lat.scatter(sparse)
    assert torch.equal(lat.gather(dense), sparse)

    for axis in range(lat.n_axes):
        perm, inv = lat.permutation(axis)
        assert inv == tuple(torch.argsort(torch.tensor(perm)).tolist())
        seq, restore = lat.to_sequence(dense, axis)
        assert torch.equal(lat.from_sequence(seq, restore), dense)
