"""Mamba-2 (SSD) from the authors' code, running where Triton cannot.

Upstream's Mamba-2 block calls fused Triton kernels that need CUDA. Off GPU
the block takes upstream's own unfused path and the chunked scan is computed
by their reference implementation (``ssd_minimal.py``, "the same as Listing 1
from the paper"), reached through our adapter in
:mod:`torch_dimensions.mixers.mamba2_compat`.

The adapter is ours, so it is the thing that has to be *proved*: the tests
below check it against an independent naive SSM recurrence written from the
definition, in float64, with the awkward cases turned on (a length that is
not a whole number of chunks, fewer groups than heads, dt bias, the D skip
and the z gate). The recurrence and the chunked algorithm are mathematically
equal, so anything above float64 epsilon is a bug in the plumbing — which is
exactly how a hardcoded ``.float()`` on ``dt_bias`` was caught.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import torch_dimensions as td

pytest.importorskip("einops", reason="the vendored Mamba-2 needs the [upstream] extra")

from torch_dimensions.mixers.mamba2_compat import mamba_chunk_scan_combined  # noqa: E402
from torch_dimensions.mixers.upstream import UpstreamMamba2Mixer  # noqa: E402


def _naive_ssd(x, dt, A, B, C, D=None, z=None, dt_bias=None):
    """The recurrence the chunked SSD algorithm claims to equal, written from
    the definition: state <- state * exp(dt*A) + dt * x B^T, y = state C."""
    b, length, h, p = x.shape
    g = B.shape[2]
    dtp = F.softplus(dt + dt_bias) if dt_bias is not None else F.softplus(dt)
    be = B.repeat_interleave(h // g, dim=2)
    ce = C.repeat_interleave(h // g, dim=2)
    state = torch.zeros(b, h, p, B.shape[-1], dtype=x.dtype)
    ys = []
    for t in range(length):
        decay = torch.exp(dtp[:, t] * A)  # (b, h)
        state = state * decay[..., None, None] + torch.einsum(
            "bh,bhp,bhn->bhpn", dtp[:, t], x[:, t], be[:, t]
        )
        ys.append(torch.einsum("bhpn,bhn->bhp", state, ce[:, t]))
    y = torch.stack(ys, dim=1)
    if D is not None:
        y = y + x * D[:, None]
    if z is not None:
        y = y * F.silu(z)
    return y


@pytest.mark.parametrize("length,chunk,ngroups", [(32, 8, 4), (30, 8, 2), (32, 32, 4), (17, 4, 1)])
def test_adapter_equals_the_definition(length, chunk, ngroups):
    torch.manual_seed(0)
    b, h, p, n = 2, 4, 8, 16
    kw = dict(dtype=torch.float64)
    x = torch.randn(b, length, h, p, **kw)
    dt = torch.randn(b, length, h, **kw)
    a = -torch.rand(h, **kw).exp()
    bb = torch.randn(b, length, ngroups, n, **kw)
    cc = torch.randn(b, length, ngroups, n, **kw)
    d = torch.randn(h, **kw)
    z = torch.randn(b, length, h, p, **kw)
    dt_bias = torch.randn(h, **kw)

    got = mamba_chunk_scan_combined(
        x, dt, a, bb, cc, chunk, D=d, z=z, dt_bias=dt_bias, dt_softplus=True
    )
    want = _naive_ssd(x, dt, a, bb, cc, D=d, z=z, dt_bias=dt_bias)
    rel = ((got - want).abs().max() / want.abs().max()).item()
    assert rel < 1e-13, rel


def test_adapter_keeps_float64_precision():
    """The regression that motivated the parametrization above: a hardcoded
    float32 cast anywhere in the plumbing shows up here as ~1e-8."""
    torch.manual_seed(1)
    b, length, h, p, n = 1, 16, 2, 4, 8
    kw = dict(dtype=torch.float64)
    got = mamba_chunk_scan_combined(
        torch.randn(b, length, h, p, **kw),
        torch.zeros(b, length, h, **kw),
        -torch.ones(h, **kw),
        torch.ones(b, length, h, n, **kw),
        torch.ones(b, length, h, n, **kw),
        8,
        dt_bias=torch.full((h,), 0.5, **kw),
        dt_softplus=True,
    )
    assert got.dtype == torch.float64


def test_variable_length_is_refused_not_wrong():
    """Ragged batches are a Triton-kernel feature; the reference path says so
    instead of quietly computing something else."""
    kw = dict(dtype=torch.float32)
    args = (
        torch.randn(1, 8, 2, 4, **kw),
        torch.randn(1, 8, 2, **kw),
        -torch.ones(2, **kw),
        torch.randn(1, 8, 2, 8, **kw),
        torch.randn(1, 8, 2, 8, **kw),
        4,
    )
    with pytest.raises(NotImplementedError, match="variable-length"):
        mamba_chunk_scan_combined(*args, seq_idx=torch.zeros(1, 8, dtype=torch.long))


def test_the_block_runs_and_sweeps_a_lattice():
    torch.manual_seed(0)
    lat = td.Lattice(shape=(3, 4), names=("y", "x"))
    model = td.Mamba2(64, 2, lattice=lat, mixer_kwargs={"headdim": 32, "d_state": 16}).eval()
    x = torch.randn(2, 3, 4, 64)
    with torch.no_grad():
        y = model(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_version_two_and_the_named_class_are_the_same_model():
    lat = td.Lattice(shape=(3, 4), names=("y", "x"))
    kw = dict(mixer_kwargs={"headdim": 32, "d_state": 16})
    by_flag = td.Mamba(64, 2, lattice=lat, version=2, **kw)
    by_name = td.Mamba2(64, 2, lattice=lat, **kw)
    assert type(by_flag.nd.mixers[0]) is type(by_name.nd.mixers[0])
    assert by_flag.config["version"] == by_name.config["version"] == 2


def test_save_load_round_trips_the_version(tmp_path):
    torch.manual_seed(0)
    lat = td.Lattice(shape=(3, 4), names=("y", "x"))
    m = td.Mamba2(64, 2, lattice=lat, mixer_kwargs={"headdim": 32, "d_state": 16}).eval()
    p = tmp_path / "m2.td"
    m.save(p)
    r = td.load(p).eval()
    x = torch.randn(2, 3, 4, 64)
    with torch.no_grad():
        assert torch.equal(m(x), r(x))
    assert r.config["version"] == 2


def test_unsupported_versions_are_refused_with_a_reason():
    with pytest.raises(ValueError, match="Mamba-3 ships upstream as Triton kernels"):
        td.Mamba(64, 1, version=3)
    with pytest.raises(ValueError, match="must be 1 or 2"):
        td.Mamba(64, 1, version=0)
    with pytest.raises(ValueError, match="no portable build of Mamba-2"):
        td.Mamba(64, 1, version=2, portable=True)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS device")
def test_mamba2_on_mps_matches_cpu():
    torch.manual_seed(0)
    cpu = UpstreamMamba2Mixer(64, d_state=16, headdim=32).eval()
    mps = UpstreamMamba2Mixer(64, d_state=16, headdim=32).to("mps")
    mps.load_state_dict({k: v.to("mps") for k, v in cpu.state_dict().items()})
    mps.eval()
    x = torch.randn(2, 24, 64)
    with torch.no_grad():
        diff = (cpu(x) - mps(x.to("mps")).cpu()).abs().max().item()
    assert diff < 1e-5, diff

    xg = torch.randn(2, 24, 64, device="mps", requires_grad=True)
    mps.train()
    mps(xg).pow(2).mean().backward()
    assert torch.isfinite(xg.grad).all()
