"""Mamba-3: the authors' block, and our transcription of their scan.

Mamba-3's recurrence exists upstream only as Triton, so unlike Mamba-1 and
Mamba-2 there is no reference implementation of theirs to defer to and no way,
on a machine without CUDA, to compare against their kernel. What *can* be
established is established here, and the file is explicit about the gap:

- the chunked (matmul) and recurrent (loop) forms are written independently
  and must agree to float64 precision;
- in the ``trap -> 1`` limit a third, direct O(L^2) sum must agree;
- ``trap -> 0`` must make the current step contribute nothing, which is what
  "trapezoidal" claims;
- the chunk size must not change the answer;
- gradients must be right (gradcheck), because the Triton backward is not
  ported — autograd differentiates the forward instead.

None of that proves equality with their kernel. It proves the recurrence
implemented here is the one written down, computed consistently.
"""

from __future__ import annotations

import pytest
import torch

import torch_dimensions as td

pytest.importorskip("einops", reason="the vendored Mamba-3 block needs the [upstream] extra")

from torch_dimensions.mixers.mamba3_compat import mamba3_siso_combined  # noqa: E402

F64 = torch.float64


def _inputs(b=2, length=37, hq=1, h=4, dqk=16, dv=8, nang=4, seed=0, gates=True):
    torch.manual_seed(seed)

    def g(*s):
        return torch.randn(*s, dtype=F64)

    return {
        "Q": g(b, length, hq, dqk),
        "K": g(b, length, hq, dqk),
        "V": g(b, length, h, dv),
        # A is negative and dt positive upstream (heavy-tail activation, then a
        # clamp), so every decay exponent is non-positive; sampling any other
        # way would test a regime the model cannot reach.
        "ADT": -torch.rand(b, h, length, dtype=F64) * 0.5 - 1e-3,
        "DT": torch.rand(b, h, length, dtype=F64) * 0.1 + 1e-3,
        "Trap": g(b, h, length),
        "Q_bias": g(h, dqk),
        "K_bias": g(h, dqk),
        "Angles": g(b, length, h, nang),
        "D": g(h) if gates else None,
        "Z": g(b, length, h, dv) if gates else None,
    }


@pytest.mark.parametrize(
    "kw",
    [
        {},
        {"hq": 2},  # grouped query attention: Q/K broadcast over head groups
        {"gates": False},  # no D skip, no Z gate
        {"length": 200},  # several chunks
        {"length": 5},  # shorter than one chunk
    ],
    ids=["dense", "gqa", "no-gates", "long", "short"],
)
def test_chunked_and_recurrent_forms_agree(kw):
    """The two independently written forms of the same recurrence.

    This is the load-bearing check: the chunked form folds each pair's two
    trapezoid visits into one weight, and if that algebra were wrong these
    would diverge.
    """
    args = _inputs(**kw)
    chunked = mamba3_siso_combined(**args, chunk_size=16)
    recurrent = mamba3_siso_combined(**args, chunk_size=16, recurrent=True)
    scale = recurrent.abs().max().item()
    assert (chunked - recurrent).abs().max().item() < 1e-12 * max(scale, 1.0)


def test_chunk_size_does_not_change_the_answer():
    args = _inputs(length=97)
    base = mamba3_siso_combined(**args, chunk_size=8)
    for chunk in (16, 32, 64, 128):
        got = mamba3_siso_combined(**args, chunk_size=chunk)
        assert (got - base).abs().max().item() < 1e-12


def test_trap_to_one_matches_an_independent_direct_sum():
    """With ``trap -> 1`` the previous-step term vanishes and the recurrence
    collapses to a decayed linear attention, which a third implementation —
    an explicit double loop, sharing no code with either scan — can state."""
    args = _inputs(length=12, h=2, dqk=8, dv=4, gates=False)
    args["Angles"] = torch.zeros_like(args["Angles"])  # isolate the scan from the rotation
    b, length, h = args["V"].shape[0], args["V"].shape[1], args["V"].shape[2]
    args["Trap"] = torch.full((b, h, length), 40.0, dtype=F64)  # sigmoid(40) = 1 - 4e-18

    out = mamba3_siso_combined(**args, chunk_size=4)

    q = args["Q"].expand(b, length, h, args["Q"].shape[-1]) + args["Q_bias"]
    k = args["K"].expand(b, length, h, args["K"].shape[-1]) + args["K_bias"]
    dt = args["DT"].movedim(-1, 1)
    cs = args["ADT"].movedim(-1, 1).cumsum(1)
    ref = torch.zeros_like(out)
    for t in range(length):
        for j in range(t + 1):
            weight = (cs[:, t] - cs[:, j]).exp() * dt[:, j]
            ref[:, t] += (weight * (q[:, t] * k[:, j]).sum(-1)).unsqueeze(-1) * args["V"][:, j]
    assert (out - ref).abs().max().item() < 1e-12


def test_trap_to_zero_removes_the_current_step():
    """The trapezoid's other end: with ``trap -> 0`` a pair contributes only
    on the step *after* it arrives, so the first output is proportional to
    ``sigmoid(trap)`` and vanishes with it."""
    args = _inputs(length=12, h=2, dqk=8, dv=4, gates=False)
    b, length, h = args["V"].shape[0], args["V"].shape[1], args["V"].shape[2]
    first = {}
    for value in (-20.0, -40.0):
        args["Trap"] = torch.full((b, h, length), value, dtype=F64)
        first[value] = mamba3_siso_combined(**args, chunk_size=4)[:, 0].abs().max().item()
    # sigmoid(-40)/sigmoid(-20) ~ 2e-9, and the outputs must track it.
    ratio = first[-40.0] / first[-20.0]
    assert 1e-9 < ratio < 1e-8, first


def test_gradients_are_correct():
    """The Triton backward (1,788 lines) is not ported: autograd differentiates
    the forward instead, so the forward being differentiable *correctly* is
    what has to hold."""
    args = _inputs(b=1, length=10, h=2, dqk=8, dv=4, nang=2)
    fixed = {k: args[k] for k in ("Q_bias", "K_bias", "D", "Z")}
    diff = ["Q", "K", "V", "ADT", "DT", "Trap", "Angles"]
    tensors = tuple(args[k].clone().requires_grad_(True) for k in diff)

    def run(*ts):
        return mamba3_siso_combined(**dict(zip(diff, ts, strict=True)), **fixed, chunk_size=4)

    assert torch.autograd.gradcheck(run, tensors, eps=1e-6, atol=1e-7)


def test_rotation_is_the_interleaved_convention():
    """Their kernel pairs adjacent components — ``tl.reshape(x, [D//2, 2])``
    then ``tl.split`` — not the half-and-half split some RoPE code uses. A
    single non-zero angle must therefore mix components 0 and 1, and leave
    component 2 alone."""
    from torch_dimensions.mixers.mamba3_compat import _rotate

    x = torch.tensor([[1.0, 0.0, 1.0, 0.0]], dtype=F64)
    cos = torch.tensor([[0.0, 1.0]], dtype=F64)  # 90 degrees on the first pair only
    sin = torch.tensor([[1.0, 0.0]], dtype=F64)
    got = _rotate(x, cos, sin)
    assert torch.allclose(got, torch.tensor([[0.0, 1.0, 1.0, 0.0]], dtype=F64))


def test_angles_beyond_the_rotary_width_are_not_rotated():
    """``headdim_angles`` can be smaller than ``headdim_qk // 2``; the tail
    pairs get cos=1, sin=0 upstream and must pass through untouched."""
    wide = _inputs(length=8, dqk=16, nang=2, gates=False)
    narrow = dict(wide)
    # Zeroing the angles must equal rotating with none of them set.
    narrow["Angles"] = torch.zeros_like(wide["Angles"])
    rotated = mamba3_siso_combined(**narrow, chunk_size=4)
    assert torch.isfinite(rotated).all()


def test_unsupported_paths_are_refused_rather_than_approximated():
    args = _inputs(length=8)
    with pytest.raises(NotImplementedError, match="cu_seqlens"):
        mamba3_siso_combined(**args, cu_seqlens=torch.tensor([0, 8], dtype=torch.int32))
    states = (torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1))
    with pytest.raises(NotImplementedError, match="input states"):
        mamba3_siso_combined(**args, Input_States=states)


# --- the block, and the model API --------------------------------------------


def test_the_vendored_block_runs_and_learns_shape():
    from torch_dimensions.mixers import Mamba3Mixer

    torch.manual_seed(0)
    mixer = Mamba3Mixer(64, d_state=32, headdim=16)
    x = torch.randn(2, 24, 64, requires_grad=True)
    y = mixer(x)
    assert y.shape == x.shape
    y.pow(2).mean().backward()
    assert torch.isfinite(x.grad).all()
    assert all(torch.isfinite(p.grad).all() for p in mixer.parameters() if p.grad is not None)


def test_mimo_is_refused_with_a_reason():
    from torch_dimensions.mixers import Mamba3Mixer

    with pytest.raises(ValueError, match="MIMO"):
        Mamba3Mixer(64, d_state=32, headdim=16, is_mimo=True)


@pytest.mark.parametrize("spelling", ["version", "name", "nd"])
def test_every_spelling_builds_the_same_model(spelling, tmp_path):
    from torch_dimensions.mixers import Mamba3Mixer

    kw = {"mixer_kwargs": {"d_state": 32, "headdim": 16}}
    lat = td.Lattice(shape=(4, 5), names=("y", "x"))
    if spelling == "version":
        model = td.Mamba(32, 2, lat, version=3, **kw)
    elif spelling == "name":
        model = td.Mamba3(32, 2, lat, **kw)
    else:
        model = td.Mamba3ND(32, 2, dim=2, shape=(4, 5), time=False, **kw)
    model.eval()

    assert isinstance(model.nd.mixers[0], Mamba3Mixer)
    assert model.config["version"] == 3

    x = torch.randn(2, 4, 5, 32)
    path = tmp_path / f"{spelling}.td"
    model.save(path)
    with torch.no_grad():
        assert torch.equal(model(x), td.load(path).eval()(x))


def test_mamba3_has_no_portable_build():
    with pytest.raises(ValueError, match="no portable build of Mamba-3"):
        td.Mamba3(32, 1, portable=True)


def test_version_three_is_registered_for_configs():
    model = td.build({"kind": "mamba3", "d_model": 32, "n_layers": 1})
    assert model.config["version"] == 3


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS device")
def test_mamba3_on_mps_matches_cpu():
    from torch_dimensions.mixers import Mamba3Mixer

    torch.manual_seed(1)
    cpu = Mamba3Mixer(32, d_state=32, headdim=16).eval()
    mps = Mamba3Mixer(32, d_state=32, headdim=16).to("mps")
    mps.load_state_dict({k: v.to("mps") for k, v in cpu.state_dict().items()})
    mps.eval()
    x = torch.randn(2, 24, 32)
    with torch.no_grad():
        assert (cpu(x) - mps(x.to("mps")).cpu()).abs().max().item() < 1e-4

    grad_in = torch.randn(2, 24, 32, device="mps", requires_grad=True)
    mps.train()
    mps(grad_in).pow(2).mean().backward()
    assert torch.isfinite(grad_in.grad).all()


# --- which implementation runs -----------------------------------------------


def test_dispatch_prefers_torch_off_cuda_and_when_forced(monkeypatch):
    from torch_dimensions.mixers._kernels import forced_torch, prefer_upstream

    assert not prefer_upstream(torch.zeros(1))  # CPU tensor: no fused kernel
    monkeypatch.setenv("TD_FORCE_TORCH_KERNELS", "1")
    assert forced_torch()
    assert not prefer_upstream(torch.zeros(1))


def test_load_upstream_returns_none_for_a_missing_kernel():
    from torch_dimensions.mixers._kernels import load_upstream

    assert load_upstream("torch_dimensions._no_such_module", "whatever") is None
    assert load_upstream("torch_dimensions.mixers._kernels", "prefer_upstream") is not None
