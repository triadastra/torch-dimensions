"""Run upstream's Mamba-2 where its Triton kernels cannot go.

Mamba-2's block calls two fused entry points from
``mamba_ssm.ops.triton.ssd_combined``. Triton has no Metal backend and the
kernels need CUDA, so on CPU and MPS they are simply unavailable — but the
*mathematics* is not: the authors ship
``mamba_ssm/modules/ssd_minimal.py::ssd_minimal_discrete``, described in their
own file as "the same as Listing 1 from the paper", and
``layernorm_gated.py::rms_norm_ref``, both pure torch.

**This module is ours, not upstream's.** It is the adapter that presents the
fused kernels' calling convention (dt bias/softplus/limit, grouped B and C,
the ``D`` skip, the ``z`` gate, chunk padding) on top of the authors'
reference SSD. The recurrence itself — every einsum that decides the numbers
— is theirs, called from here. Nothing in this file reimplements the scan.

The split is deliberate: an adapter that got the plumbing wrong would be
caught by tests, whereas a reimplemented recurrence could be subtly wrong in
a way no test on a CUDA-less machine could see.
"""

from __future__ import annotations

import types

import torch
import torch.nn.functional as F

from torch_dimensions.mixers._kernels import load_upstream, prefer_upstream

__all__ = ["mamba_chunk_scan_combined", "mamba_split_conv1d_scan_combined", "triton_stub"]

_SSD = "mamba_ssm.ops.triton.ssd_combined"


def triton_stub() -> tuple[types.ModuleType, types.ModuleType]:
    """A no-op stand-in for ``triton`` / ``triton.language``.

    Upstream's ``layernorm_gated.py`` decorates its kernels with
    ``@triton.jit`` at module scope, so the file cannot even be imported
    without triton installed. These stubs let the module import — the
    decorated kernels are then never *called*, because the vendored file's
    tagged patch routes the forward pass to the authors' own
    ``rms_norm_ref``. Stubbing an import is not the same as replacing a
    computation: no stubbed function participates in any number produced
    here.
    """
    triton = types.ModuleType("triton")
    triton.jit = lambda fn=None, **kw: fn if fn is not None else (lambda f: f)
    triton.heuristics = lambda *a, **k: lambda f: f
    triton.autotune = lambda *a, **k: lambda f: f
    triton.Config = lambda *a, **k: None
    triton.cdiv = lambda a, b: -(-a // b)
    tl = types.ModuleType("triton.language")
    tl.constexpr = object
    for name in (
        "program_id",
        "arange",
        "load",
        "store",
        "sum",
        "sqrt",
        "exp",
        "log",
        "where",
        "zeros",
        "full",
        "maximum",
        "minimum",
        "sigmoid",
        "num_programs",
    ):
        setattr(tl, name, lambda *a, **k: None)
    for name in ("float32", "float16", "bfloat16", "int32", "int64"):
        setattr(tl, name, None)
    triton.language = tl
    return triton, tl


def _compute_dtype(x: torch.Tensor) -> torch.dtype:
    """float32 at least — upstream's kernels accumulate the scan in fp32 — but
    never *downcast*: a float64 input asks for float64 arithmetic, and silently
    halving its precision would make the exactness tests below meaningless."""
    return torch.promote_types(x.dtype, torch.float32)


def _dt_prepare(dt, dt_bias, dt_softplus, dt_limit):
    dt = dt.to(_compute_dtype(dt))
    if dt_bias is not None:
        # .to(dt.dtype), not .float(): a hardcoded float32 cast here silently
        # halved a float64 caller's precision and showed up as a 1e-8 relative
        # error against the reference recurrence (DEBUG.md).
        dt = dt + dt_bias.to(dt.dtype)
    if dt_softplus:
        dt = F.softplus(dt)
    if dt_limit is not None and dt_limit != (0.0, float("inf")):
        dt = dt.clamp(min=dt_limit[0], max=dt_limit[1])
    return dt


def mamba_chunk_scan_combined(
    x,
    dt,
    A,
    B,
    C,
    chunk_size,
    D=None,
    z=None,
    dt_bias=None,
    initial_states=None,
    seq_idx=None,
    cu_seqlens=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    return_final_states=False,
    return_varlen_states=False,
    **_unused,
):
    """The fused chunked-scan entry point, computed by the authors' reference.

    Shapes follow upstream: ``x`` (b, l, h, p), ``dt`` (b, l, h), ``A`` (h,),
    ``B``/``C`` (b, l, g, n), ``D`` (h,) or (h, p), ``z`` (b, l, h, p).

    On CUDA the real fused kernel runs instead — decided here, per call, so
    that a machine which merely *has* ``mamba_ssm`` installed does not send
    CPU tensors into a CUDA kernel.
    """
    fused = load_upstream(_SSD, "mamba_chunk_scan_combined") if prefer_upstream(x) else None
    if fused is not None:
        return fused(
            x,
            dt,
            A,
            B,
            C,
            chunk_size,
            D=D,
            z=z,
            dt_bias=dt_bias,
            initial_states=initial_states,
            seq_idx=seq_idx,
            cu_seqlens=cu_seqlens,
            dt_softplus=dt_softplus,
            dt_limit=dt_limit,
            return_final_states=return_final_states,
            return_varlen_states=return_varlen_states,
        )
    if seq_idx is not None or cu_seqlens is not None or return_varlen_states:
        raise NotImplementedError(
            "variable-length sequences (seq_idx / cu_seqlens) are only supported by "
            "upstream's Triton kernels, which need CUDA; this reference path handles "
            "fixed-length batches"
        )

    from torch_dimensions._vendor.mamba.ssd_minimal import ssd_minimal_discrete

    batch, length, nheads, headdim = x.shape
    dt = _dt_prepare(dt, dt_bias, dt_softplus, dt_limit)  # (b, l, h)

    # B and C are per *group*; the reference wants them per head. Upstream's
    # kernels broadcast the same way (heads // ngroups heads share a group).
    ngroups = B.shape[2]
    if ngroups != nheads:
        if nheads % ngroups:
            raise ValueError(f"nheads={nheads} is not a multiple of ngroups={ngroups}")
        rep = nheads // ngroups
        B = B.repeat_interleave(rep, dim=2)
        C = C.repeat_interleave(rep, dim=2)

    # The reference requires the length to be a whole number of chunks. Pad
    # with zeros: a padded step has dt=0, so its decay is exp(0)=1 and its
    # input contributes nothing, and the padded outputs are dropped below.
    pad = (-length) % chunk_size
    if pad:
        x = F.pad(x, (0, 0, 0, 0, 0, pad))
        dt = F.pad(dt, (0, 0, 0, pad))
        B = F.pad(B, (0, 0, 0, 0, 0, pad))
        C = F.pad(C, (0, 0, 0, 0, 0, pad))

    dtype = x.dtype
    acc = _compute_dtype(x)
    xd = x.to(acc) * dt.unsqueeze(-1).to(acc)
    ad = A.to(acc) * dt.to(acc)
    y, final_state = ssd_minimal_discrete(
        xd, ad, B.to(acc), C.to(acc), chunk_size, initial_states=initial_states
    )

    if pad:
        y = y[:, :length]
    if D is not None:
        d = D.to(acc)
        y = y + (x[:, :length].to(acc) * (d if d.dim() == 2 else d.unsqueeze(-1)))
    if z is not None:
        y = y * F.silu(z.to(acc))
    y = y.to(dtype)
    return (y, final_state) if return_final_states else y


def mamba_split_conv1d_scan_combined(*args, **kwargs):
    """Upstream's fully fused path (conv + scan + norm + out-projection in one
    kernel). There is no reference implementation of it, and unpicking it here
    would be our arithmetic rather than theirs — so instead the adapter passes
    ``use_mem_eff_path=False``, which is upstream's own switch for taking the
    unfused path through the very same module. On CUDA the real kernel runs."""
    fused = load_upstream(_SSD, "mamba_split_conv1d_scan_combined")
    if fused is not None and prefer_upstream(*args):
        return fused(*args, **kwargs)
    raise NotImplementedError(
        "the fully fused Mamba-2 path needs Triton and CUDA; construct the mixer with "
        "use_mem_eff_path=False (what torch-dimensions does off-GPU) to take upstream's "
        "own unfused path"
    )
