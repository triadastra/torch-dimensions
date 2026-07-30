"""Device-placement conformance: everything that works on CPU works on an
accelerator, and CPU/device mismatches either work or fail loudly at the API
boundary — never three frames deep in an indexing kernel.

Runs against whatever accelerator the machine has — MPS on Apple Silicon,
CUDA elsewhere — and skips (visibly, per the suite's skips-are-not-passes
rule) when there is none. Device-placement bugs are indifferent to *which*
second device exists; what they need is any second device at all. What this
file cannot vouch for on MPS: CUDA-specific kernel numerics, torch.compile
backends, and float64 (unsupported on MPS — everything here is float32).
"""

import pytest
import torch

import torch_dimensions as td
from torch_dimensions.compose.kernel import axial_contract

if torch.cuda.is_available():
    DEV = "cuda"
elif torch.backends.mps.is_available():
    DEV = "mps"
else:
    DEV = None

pytestmark = pytest.mark.skipif(DEV is None, reason="no accelerator (cuda or mps) available")


def _sparse_lattice():
    valid = torch.tensor(
        [[True, True, True, False], [True, False, True, True], [True, True, False, True]]
    )
    return td.Lattice(shape=(3, 4), names=("h", "w"), valid=valid, time=True)


def test_a_sparse_model_trains_on_the_device():
    lat = _sparse_lattice()
    torch.manual_seed(0)
    model = td.LSTM(8, 3, lat).to(DEV)
    x = torch.randn(2, 5, 3, 4, 8, device=DEV)
    out = model(x)
    assert out.device.type == DEV
    out.pow(2).mean().backward()
    assert all(p.grad is not None and p.grad.device.type == DEV for p in model.parameters())


def test_absent_cells_are_still_inert_on_the_device():
    lat = _sparse_lattice()
    torch.manual_seed(0)
    model = td.LSTM(8, 3, lat).to(DEV)
    x = torch.randn(2, 5, 3, 4, 8, device=DEV)
    noise = torch.randn_like(x) * 1e3 * (~lat.mask().to(DEV)).to(x.dtype)
    assert torch.equal(model(x), model(x + noise))


@pytest.mark.parametrize("lattice_on_device", [False, True])
def test_gather_scatter_round_trip_across_device_mismatches(lattice_on_device):
    """A CPU lattice must handle device tensors and a device lattice must
    handle CPU tensors: `flat_idx` lives wherever `valid` lives, and indexing
    must not require the caller to know that."""
    lat = _sparse_lattice()
    if lattice_on_device:
        lat = lat.to(DEV)
    for x_device in ("cpu", DEV):
        x = torch.randn(2, 5, 3, 4, 8, device=x_device) * lat.mask().to(x_device)
        g = lat.gather(x)
        assert g.device.type == torch.device(x_device).type
        assert torch.equal(lat.scatter(g), x)


def test_axial_contract_renormalizes_on_the_device():
    lat = td.Lattice(shape=(2, 4), valid=torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]]).bool())
    mask = lat.mask().to(torch.float32).to(DEV)
    x = torch.ones(1, 2, 4, 3, device=DEV) * mask
    row_stochastic = torch.full((4, 4), 0.25, device=DEV)
    out = axial_contract(x, lat, 1, row_stochastic, valid=mask)
    # constant-input invariant: a convex combination of ones is one at every
    # present position, on any device
    present = out.masked_select(mask.bool().expand_as(out))
    assert torch.allclose(present, torch.ones_like(present), atol=1e-6)


def test_spec_survives_a_device_model():
    lat = _sparse_lattice()
    model = td.GRU(6, 3, lat).to(DEV)
    spec = model.to_spec()
    assert spec["lattice"]["cells"]["present"] == 9
    assert spec["model"]["kind"] == "GRU"


@pytest.mark.parametrize("cls", [td.S4D, td.Mamba])
def test_the_ssm_family_runs_and_agrees_on_the_device(cls):
    lat = _sparse_lattice()
    torch.manual_seed(0)
    cpu = cls(8, 3, lat)
    dev = cls(8, 3, lat)
    dev.load_state_dict(cpu.state_dict())
    dev = dev.to(DEV)
    x = torch.randn(2, 5, 3, 4, 8)
    with torch.no_grad():
        yc = cpu(x)
        ym = dev(x.to(DEV))
    assert not bool(ym.isnan().any())
    diff = (yc - ym.cpu()).abs().max().item()
    assert diff < 1e-4, f"cpu-vs-device diff {diff:.2e}"
    dev(x.to(DEV)).pow(2).mean().backward()
    assert all(p.grad is not None for p in dev.parameters() if p.requires_grad)
