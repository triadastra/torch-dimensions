"""Our composition against the composition the papers actually specify.

Every other verification in this library checks a **1-D mixer** against its
source: the S4D kernel is bitwise identical to upstream's, the S4 DPLR kernel
matches at 3e-8, the Mamba scan at 1e-6. None of them checks the **N-D
composition**, which is the part this library claims as its own contribution —
so until now the central claim rested on internal consistency (the Kronecker
identity, the separable-conv identity) rather than on agreement with a
published N-D method.

This file closes that. The method under test is S4ND (Nguyen et al., 2022),
whose composition is not a sweep at all:

    one 1-D SSM kernel per axis
      -> outer product of their Fourier transforms
      -> a single N-D FFT convolution over all axes at once

That is "simultaneous separable". Ours is one axis per layer, sequentially.
The two are *supposed* to be the same operator, and the test is whether they
are — measured, not assumed.

**Why the reference is written here rather than imported.** The upstream repo
needs `hydra` to import a module that computes a kernel, and its N-D model
carries a config framework, a Lightning trainer and a rank-specific einsum
table. Vendoring that to run twenty lines of arithmetic would trade a testable
claim for a dependency and a license review. So the *composition rule* is
transcribed here as an explicit oracle — exactly what `kron_operator` is for
the kernel family, and what the dense state-space matrix-power reference is
for S4 — and the kernels it is fed come from our own `_S4DKernel`, which is
already proven bitwise equal to theirs. Cross-checking against the real repo
belongs in the portability dossier (PLAN.md Phase 7), not in CI.
"""

import pytest
import torch

import torch_dimensions as td
from torch_dimensions.mixers.ssm import _S4DKernel


def causal_fft_conv(seq: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """The bare convolution inside `_KernelConvMixer`, with no skip or gate.

    `(M, A, H)` in and out. `n = 2A` makes it a linear rather than a circular
    convolution, which is what lets a truncated result still be exact.
    """
    a = seq.shape[1]
    n = 2 * a
    xt = seq.transpose(1, 2)
    y = torch.fft.irfft(torch.fft.rfft(xt, n=n) * torch.fft.rfft(k, n=n), n=n)[..., :a]
    return y.transpose(1, 2)


def s4nd_simultaneous(x: torch.Tensor, kernels: list[torch.Tensor]) -> torch.Tensor:
    """S4ND's composition: outer-product the per-axis kernels, convolve once.

    Transcribed from `src/models/sequence/modules/s4nd.py` (`contract_version=0`):
    every axis but the last is transformed with `fft`, the last with `rfft`,
    the results are outer-producted into one N-D kernel, and the input goes
    through a single `rfftn` of the padded shape.

    `x` is `(B, *shape, H)`; each kernel is `(H, axis_length)`.
    """
    rank = len(kernels)
    sizes = [k.shape[-1] for k in kernels]
    padded = [2 * s for s in sizes]
    dims = tuple(range(-rank, 0))

    u = x.permute(0, x.ndim - 1, *range(1, x.ndim - 1))  # (B, H, *shape)
    u_f = torch.fft.rfftn(u, s=tuple(padded), dim=dims)

    # Outer product over the axis dimensions, broadcasting on the channel.
    k_f = None
    for axis, (k, n) in enumerate(zip(kernels, padded, strict=True)):
        t = torch.fft.rfft(k, n=n) if axis == rank - 1 else torch.fft.fft(k, n=n)
        # (H, ..., n) with a singleton for every axis placed so far
        shaped = t.reshape(t.shape[0], *([1] * axis), t.shape[-1])
        k_f = shaped if k_f is None else k_f.unsqueeze(-1) * shaped

    y = torch.fft.irfftn(u_f * k_f, s=tuple(padded), dim=dims)
    y = y[(..., *(slice(0, s) for s in sizes))]
    return y.permute(0, *range(2, y.ndim), 1)


@pytest.mark.parametrize("shape", [(6, 7), (4, 5, 6)])
def test_our_sequential_sweep_equals_s4nds_simultaneous_kernel(shape):
    """**The N-D claim, checked against a published N-D method.**

    S4ND applies every axis at once through one N-D kernel; we apply one axis
    per layer. If the library's premise is right these are the same operator,
    and they are — to machine precision, at rank 2 and rank 3.

    This is the strongest single piece of evidence for "an N-D model is a 1-D
    mixer plus a plan for sweeping it": the plan reproduces, exactly, a model
    that was never written as a sweep.
    """
    torch.manual_seed(0)
    h, rank = 3, len(shape)
    names = tuple("hwd"[:rank])
    lat = td.Lattice(shape=shape, names=names)

    kernels = [_S4DKernel(h, d_state=16).double()(size) for size in shape]
    x = torch.randn(2, *shape, h, dtype=torch.float64)

    ours = x
    for name, k in zip(names, kernels, strict=True):
        ours = td.axial_apply(ours, lat, name, lambda s, k=k: causal_fft_conv(s, k))

    theirs = s4nd_simultaneous(x, kernels)

    diff = (ours - theirs).abs().max().item()
    scale = theirs.abs().max().item()
    assert diff / scale < 1e-12, f"rank {rank}: sequential differs from simultaneous by {diff:.2e}"


def test_the_equivalence_needs_a_channel_diagonal_kernel():
    """The negative control, and the reason LTI.md's correction matters.

    S4ND's kernels are diagonal in channels — one scalar filter per channel per
    axis — which is exactly the "scalar-valued filter" case where per-axis
    operators commute and a sequential sweep collapses to one joint kernel.
    Give each axis a filter that *mixes channels* and the equivalence dies:
    sequential and simultaneous are then different models, because matrix-
    valued filters do not commute.

    Without this control the test above would pass just as happily against an
    implementation that had quietly stopped being separable.
    """
    torch.manual_seed(0)
    h, shape = 3, (5, 6)
    lat = td.Lattice(shape=shape, names=("h", "w"))
    x = torch.randn(2, *shape, h, dtype=torch.float64)

    kernels = [_S4DKernel(h, d_state=16).double()(size) for size in shape]
    mix = torch.randn(h, h, dtype=torch.float64)

    def mixing_conv(seq, k):
        return causal_fft_conv(seq, k) @ mix

    ours = x
    for name, k in zip(("h", "w"), kernels, strict=True):
        ours = td.axial_apply(ours, lat, name, lambda s, k=k: mixing_conv(s, k))

    # The simultaneous form can only carry a per-channel kernel, so the honest
    # comparison applies the same channel mixing twice and asks whether the
    # orders agree. They do not.
    theirs = s4nd_simultaneous(x, kernels) @ mix @ mix

    diff = (ours - theirs).abs().max().item()
    assert diff / theirs.abs().max().item() > 1e-3, (
        "a channel-mixing filter must break the sequential/simultaneous identity"
    )
