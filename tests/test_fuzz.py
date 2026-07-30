"""Seeded fuzz over the library's invariants, checked against slow references.

Targeted tests check configurations someone thought of; these check the ones
nobody did. Every case is seeded, so a failure reproduces exactly — paste the
printed config into a targeted test and it stays failed until fixed.
"""

import torch

import torch_dimensions as td
from torch_dimensions.compose.kernel import axial_contract
from torch_dimensions.compose.scan import axial_apply
from torch_dimensions.data.coords import from_coords
from torch_dimensions.data.window import LatticeWindow

_REL = 1e-3  # mirror of the kernel module's cancellation threshold


def _rand_lattice(g, rank, time, sparse):
    shape = tuple(int(torch.randint(1, 5, (1,), generator=g)) for _ in range(rank))
    valid = None
    if sparse and rank > 0:
        valid = torch.rand(shape, generator=g) > 0.5
        if not valid.any():
            valid.reshape(-1)[int(torch.randint(0, valid.numel(), (1,), generator=g))] = True
    return td.Lattice(shape=shape, valid=valid, time=time)


def test_fold_scatter_and_permutation_round_trip_on_random_lattices():
    g = torch.Generator().manual_seed(0)
    for i in range(60):
        rank = int(torch.randint(1, 5, (1,), generator=g))
        time = bool(torch.randint(0, 2, (1,), generator=g))
        sparse = bool(torch.randint(0, 2, (1,), generator=g))
        lat = _rand_lattice(g, rank, time, sparse)
        lead = (2, 3) if time else (2,)
        x = torch.randn(*lead, *lat.shape, 4, generator=g)
        for axis in range(lat.n_axes):
            seq, restore = lat.to_sequence(x, axis)
            assert torch.equal(lat.from_sequence(seq, restore), x), f"[{i}] axis {axis} {lat}"
            perm, inv = lat.permutation(axis)
            assert list(torch.argsort(torch.tensor(perm))) == list(inv), f"[{i}] {lat}"
        xm = x * lat.mask().to(x.dtype)
        assert torch.equal(lat.scatter(lat.gather(xm)), xm), f"[{i}] {lat}"
        assert lat.flat_idx.numel() == lat.n_valid, f"[{i}] {lat}"


def test_axial_contract_matches_a_per_line_loop_on_random_sparse_lattices():
    """Independent reference: an explicit loop over every line, including the
    relative-cancellation rule for degenerate denominators."""
    g = torch.Generator().manual_seed(1)
    for i in range(30):
        rank = int(torch.randint(1, 4, (1,), generator=g))
        lat = _rand_lattice(g, rank, False, True)
        axis = int(torch.randint(0, rank, (1,), generator=g))
        a_len = lat.axis_size(axis)
        mask = lat.mask().to(torch.float64)
        x = torch.randn(2, *lat.shape, 3, dtype=torch.float64, generator=g) * mask
        kernel = torch.randn(a_len, a_len, dtype=torch.float64, generator=g)  # signed
        got = axial_contract(x, lat, axis, kernel, valid=mask)

        seq, restore = lat.to_sequence(x, axis)
        mseq, _ = lat.to_sequence(mask.expand(*x.shape[:-1], 1), axis)
        out = torch.zeros_like(seq)
        for m in range(seq.shape[0]):
            pres = mseq[m, :, 0]
            for q in range(a_len):
                den = float((kernel[q] * pres).sum())
                den_abs = float((kernel[q].abs() * pres).sum())
                num = (kernel[q].unsqueeze(-1) * seq[m] * pres.unsqueeze(-1)).sum(0)
                out[m, q] = num if abs(den) <= _REL * den_abs else num / den
        want = lat.from_sequence(out, restore)
        assert torch.allclose(got, want, atol=1e-10), f"[{i}] {lat} axis {axis}"


def test_axial_apply_matches_cumsum_on_random_configs():
    g = torch.Generator().manual_seed(2)
    for i in range(40):
        rank = int(torch.randint(1, 5, (1,), generator=g))
        time = bool(torch.randint(0, 2, (1,), generator=g))
        lat = _rand_lattice(g, rank, time, False)
        lead = (2, 3) if time else (2,)
        x = torch.randn(*lead, *lat.shape, 3, generator=g)
        axis = int(torch.randint(0, lat.n_axes, (1,), generator=g))
        rev = bool(torch.randint(0, 2, (1,), generator=g))
        chunk = [None, 1, 7][int(torch.randint(0, 3, (1,), generator=g))]
        d = lat.tensor_dim(axis)
        want = x.flip(d).cumsum(dim=d).flip(d) if rev else x.cumsum(dim=d)
        got = axial_apply(x, lat, axis, lambda s: s.cumsum(dim=1), reverse=rev, chunk=chunk)
        assert torch.equal(got, want), f"[{i}] rank {rank} axis {axis} rev {rev} chunk {chunk}"


def test_window_tiling_properties_on_random_configs():
    g = torch.Generator().manual_seed(3)
    for i in range(120):
        n = int(torch.randint(2, 40, (1,), generator=g))
        il = int(torch.randint(1, n + 1, (1,), generator=g))
        hz = int(torch.randint(0, n - il + 1, (1,), generator=g))
        st = int(torch.randint(1, 6, (1,), generator=g))
        w = LatticeWindow(n, input_len=il, horizon=hz, stride=st)
        for win in w:
            assert 0 <= win.x0 < win.x1 <= win.y0 <= win.y1 <= n, f"[{i}] {win} n={n}"
            assert win.x1 - win.x0 == il and win.y1 - win.y0 == hz, f"[{i}] {win}"
        at = int(torch.randint(0, n + 1, (1,), generator=g))
        train, test = w.split(at)
        assert all(win.y1 <= at for win in train), f"[{i}] train crosses cut at {at}"
        assert all(win.x0 >= at for win in test), f"[{i}] test crosses cut at {at}"


def test_coords_encode_decode_round_trip_on_random_tables():
    g = torch.Generator().manual_seed(4)
    for i in range(40):
        k = int(torch.randint(1, 4, (1,), generator=g))
        n_rows = int(torch.randint(1, 30, (1,), generator=g))
        rows = [
            tuple(f"v{int(torch.randint(0, 4, (1,), generator=g))}" for _ in range(k))
            for _ in range(n_rows)
        ]
        cm = from_coords(rows, time=False)
        for row, flat in zip(rows, cm.index.tolist(), strict=True):
            assert cm.decode(flat) == row, f"[{i}] {row} -> {cm.decode(flat)}"
        assert torch.equal(cm.encode(rows), cm.index), f"[{i}] encode != index"
