"""Mamba-3's SISO scan in PyTorch, for machines Triton cannot reach.

Mamba-1 and Mamba-2 could be run off-GPU without writing any mathematics:
their authors ship pure-torch reference paths (``selective_scan_ref``,
``ssd_minimal_discrete``) and the vendored modules simply call them. **Mamba-3
ships no such reference.** Its scan exists only as ~6,300 lines of Triton
across forward, backward and step kernels, so running it anywhere else means
writing the recurrence out.

This file is therefore **our code, not upstream's** — a transcription of the
recurrence their kernels implement, not a copy of their implementation. Where
the Mamba-1/2 adapters are plumbing around the authors' own numerics, here the
numerics are ours. That distinction is why the mixer built on this is called
``Mamba3Mixer`` and not ``UpstreamMamba3Mixer``, and why the checks below
matter more.

**The recurrence**, transcribed from ``mamba3_siso_step.py`` (their clearest
statement of it) and ``angle_dt.py``::

    theta_t = (cumsum_t tanh(angle_t) * pi * dt_t) mod 2*pi   # rotary phase
    q_t, k_t = rotate(Q_t + Q_bias, theta_t), rotate(K_t + K_bias, theta_t)

    alpha_t = exp(A_t dt_t)                    # decay
    gamma_t = dt_t * sigmoid(trap_t)           # current-step (trapezoid) weight
    beta_t  = alpha_t * dt_t * (1 - sigmoid(trap_t))   # previous-step weight

    S_t = alpha_t S_{t-1} + beta_t v_{t-1} k_{t-1}^T + gamma_t v_t k_t^T
    out_t = S_t q_t  (+ D_t v_t) (* silu(z_t))

The trapezoidal rule is what makes this more than linear attention: each
(k, v) pair enters the state twice, once when it arrives and once on the
following step, so a pair's total weight straddles a step boundary.

**Two implementations, and why.** :func:`_scan_recurrent` is the loop above,
line for line — obviously correct, O(L) sequential steps.
:func:`_scan_chunked` is the matmul form used by default, which folds each
pair's two visits into one weight::

    scale_j = gamma_j + dt_{j+1} (1 - sigmoid(trap_{j+1}))
    out_t   = sum_{j<t} exp(cumsum_t - cumsum_j) scale_j (q_t . k_j) v_j
              + gamma_t (q_t . k_t) v_t

The algebra that merges them is in the tests, not just in this docstring: the
two forms are checked against each other, and the fold is what a test would
catch if it were wrong. Absent a CUDA device, agreement between two
independently written forms is the strongest available evidence, and the
package says so rather than implying a comparison it has not run.

**Deliberate differences from the Triton kernels**, both in the direction of
accuracy: their kernels compute in bfloat16 and use PTX ``cos.approx.f32`` /
``sin.approx.f32`` / ``tanh.approx.f32``, while this runs in float32 with
exact library functions. Numbers will therefore differ slightly from a CUDA
run — by more than float noise, and in this code's favour.

On CUDA with Triton installed the authors' kernels are called instead; see
:mod:`torch_dimensions.mixers._kernels`.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from torch_dimensions.mixers._kernels import load_upstream, prefer_upstream

__all__ = ["mamba3_siso_combined"]

_TWO_PI = 2.0 * math.pi


def _compute_dtype(x: torch.Tensor) -> torch.dtype:
    """float32 at least — their kernels accumulate in fp32 — but never
    *downcast*: a float64 input asks for float64, and silently halving its
    precision would make the agreement checks below meaningless."""
    return torch.promote_types(x.dtype, torch.float32)


def _angle_cumsum(angles: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """Cumulative rotary phase: ``cumsum(tanh(angle) * pi * dt) mod 2*pi``.

    ``angles`` is ``(B, L, H, A)``, ``dt`` is ``(B, L, H)``; the result is
    ``(B, L, H, A)``. Upstream applies the modulo per chunk *and* to the
    carried state (``angle_dt.py``); doing it once at the end is the same
    angle, and cos/sin do not care which representative they are handed.
    """
    step = torch.tanh(angles.to(dt.dtype)) * math.pi * dt.unsqueeze(-1)
    return torch.remainder(step.cumsum(dim=1), _TWO_PI)


def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """The interleaved rotation their kernel applies to Q and K.

    Pairs are adjacent — ``(x[0], x[1])``, ``(x[2], x[3])``, ... — which is
    ``tl.reshape(x, [D // 2, 2])`` followed by ``tl.split``, not the
    half-and-half convention some RoPE implementations use. Pairs beyond the
    angle width get cos=1, sin=0 and pass through unrotated.
    """
    x0, x1 = x[..., 0::2], x[..., 1::2]
    return torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1).flatten(-2)


def _scan_recurrent(q, k, v, alpha, beta, gamma):
    """The recurrence exactly as ``mamba3_siso_step.py`` writes it.

    Kept as the reference the chunked form is checked against — and used
    directly for sequences shorter than one chunk, where the matmul form has
    nothing to win.

    ``q``/``k``: ``(B, L, H, Dqk)``, ``v``: ``(B, L, H, Dv)``, the three
    weights ``(B, L, H)``. Returns ``(out, S, k_last, v_last)``.
    """
    b, length, h, dqk = q.shape
    dv = v.shape[-1]
    state = q.new_zeros(b, h, dv, dqk)
    k_prev = q.new_zeros(b, h, dqk)
    v_prev = q.new_zeros(b, h, dv)
    out = []
    for t in range(length):
        state = (
            state * alpha[:, t, :, None, None]
            + beta[:, t, :, None, None] * v_prev.unsqueeze(-1) * k_prev.unsqueeze(-2)
            + gamma[:, t, :, None, None] * v[:, t].unsqueeze(-1) * k[:, t].unsqueeze(-2)
        )
        out.append(torch.einsum("bhed,bhd->bhe", state, q[:, t]))
        k_prev, v_prev = k[:, t], v[:, t]
    return torch.stack(out, dim=1), state, k_prev, v_prev


def _scan_chunked(q, k, v, adt, scale, gamma, chunk: int):
    """The matmul form: intra-chunk quadratic, inter-chunk recurrent.

    Every decay exponent here is a sum of ``adt`` values, which are negative
    (upstream clamps ``A <= -A_floor``), so every exponential is of a
    non-positive number and cannot overflow.
    """
    b, length, h, dqk = q.shape
    dv = v.shape[-1]
    pad = (-length) % chunk
    if pad:
        q, k, v = (F.pad(t, (0, 0, 0, 0, 0, pad)) for t in (q, k, v))
        adt, scale, gamma = (F.pad(t, (0, 0, 0, pad)) for t in (adt, scale, gamma))
    n = (length + pad) // chunk

    def blocks(t):
        return t.reshape(b, n, chunk, *t.shape[2:])

    qb, kb, vb = blocks(q), blocks(k), blocks(v)
    adtb, scaleb, gammab = blocks(adt), blocks(scale), blocks(gamma)

    cs = adtb.cumsum(dim=2)  # (B, n, C, H) — decay from chunk start, inclusive
    # exponent[t, j] = cumsum_t - cumsum_j, kept only strictly below the
    # diagonal; the diagonal carries the current-step weight gamma_t instead.
    expo = cs.unsqueeze(3) - cs.unsqueeze(2)  # (B, n, t, j, H)
    lower = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device).tril(-1)
    expo = expo.masked_fill(~lower[None, None, :, :, None], -float("inf"))
    weight = expo.exp() * scaleb.unsqueeze(2)  # (B, n, t, j, H)
    eye = torch.eye(chunk, dtype=q.dtype, device=q.device)
    weight = weight + eye[None, None, :, :, None] * gammab.unsqueeze(3)

    qk = torch.einsum("bnthd,bnjhd->bntjh", qb, kb)
    y_intra = torch.einsum("bntjh,bnjhe->bnthe", qk * weight, vb)

    # Inter-chunk: one sequential pass over chunks, carrying (Dv, Dqk) states.
    states = []
    state = q.new_zeros(b, h, dv, dqk)
    tail = cs[:, :, -1]  # (B, n, H) — decay across a whole chunk
    for c in range(n):
        states.append(state)
        to_end = (tail[:, c].unsqueeze(1) - cs[:, c]).exp() * scaleb[:, c]  # (B, C, H)
        contrib = torch.einsum("bjh,bjhe,bjhd->bhed", to_end, vb[:, c], kb[:, c])
        state = state * tail[:, c][:, :, None, None].exp() + contrib
    prior = torch.stack(states, dim=1)  # (B, n, H, Dv, Dqk)
    y_inter = torch.einsum("bnthd,bnhed->bnthe", qb * cs.exp().unsqueeze(-1), prior)

    out = (y_intra + y_inter).reshape(b, n * chunk, h, dv)[:, :length]
    k_last, v_last = k[:, length - 1], v[:, length - 1]
    return out, state, k_last, v_last


def mamba3_siso_combined(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    ADT: torch.Tensor,
    DT: torch.Tensor,
    Trap: torch.Tensor,
    Q_bias: torch.Tensor,
    K_bias: torch.Tensor,
    Angles: torch.Tensor,
    D: torch.Tensor | None = None,
    Z: torch.Tensor | None = None,
    Input_States=None,
    chunk_size: int = 64,
    return_final_states: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    recurrent: bool = False,
):
    """Upstream's ``mamba3_siso_combined`` entry point, on any device.

    Same arguments and shapes as the Triton version. On CUDA the real kernel
    is called; elsewhere the recurrence above runs. ``recurrent=True`` forces
    the sequential reference, which is how the two forms are compared.
    """
    if not recurrent and prefer_upstream(Q, K, V):
        fused = load_upstream(
            "mamba_ssm.ops.triton.mamba3.mamba3_siso_combined", "mamba3_siso_combined"
        )
        if fused is not None:
            return fused(
                Q,
                K,
                V,
                ADT,
                DT,
                Trap,
                Q_bias,
                K_bias,
                Angles,
                D,
                Z,
                Input_States,
                chunk_size,
                return_final_states,
                cu_seqlens,
            )

    if cu_seqlens is not None:
        raise NotImplementedError(
            "variable-length sequences (cu_seqlens) need upstream's Triton kernel; "
            "this torch path handles the dense case only"
        )
    if Input_States is not None and any(s is not None for s in Input_States):
        raise NotImplementedError(
            "carrying input states into the torch Mamba-3 scan is not implemented; "
            "it is only reachable from upstream's step path, which needs CUDA anyway"
        )

    batch, length, nheads_qk, dqk = Q.shape
    nheads = V.shape[2]
    if nheads % nheads_qk:
        raise ValueError(f"nheads ({nheads}) must be divisible by nheads_qk ({nheads_qk})")

    # float32 at least: their kernel works in bfloat16, and matching that
    # would trade away accuracy for a speed this path cannot have anyway.
    dtype = _compute_dtype(V)
    q, k, v = Q.to(dtype), K.to(dtype), V.to(dtype)
    if nheads != nheads_qk:  # grouped query attention: one Q/K group per head group
        q = q.repeat_interleave(nheads // nheads_qk, dim=2)
        k = k.repeat_interleave(nheads // nheads_qk, dim=2)
    # The bias is per *output* head and lands before the rotation, as in their
    # step kernel (q_bias is indexed by pid_head, Q by the GQA head).
    q = q + Q_bias.to(dtype)
    k = k + K_bias.to(dtype)

    dt = DT.to(dtype).movedim(-1, 1)  # (B, L, H)
    adt = ADT.to(dtype).movedim(-1, 1)
    trap = torch.sigmoid(Trap.to(dtype).movedim(-1, 1))

    theta = _angle_cumsum(Angles, dt)  # (B, L, H, A)
    pairs = dqk // 2
    if theta.shape[-1] < pairs:  # dims past the angle width stay unrotated
        theta = F.pad(theta, (0, pairs - theta.shape[-1]))
    cos, sin = torch.cos(theta), torch.sin(theta)
    q, k = _rotate(q, cos, sin), _rotate(k, cos, sin)

    gamma = dt * trap
    if recurrent:
        alpha = adt.exp()
        beta = alpha * dt * (1 - trap)
        out, state, k_last, v_last = _scan_recurrent(q, k, v, alpha, beta, gamma)
    else:
        # scale_j folds a pair's two visits — arriving at j, then again at
        # j+1 — into one weight. The final step has no successor, which is
        # what the carried k/v states are for. Chunking never exceeds the
        # sequence: no branch on length, so one code path is always the one
        # under test.
        shifted = F.pad((dt * (1 - trap))[:, 1:], (0, 0, 0, 1))
        out, state, k_last, v_last = _scan_chunked(
            q, k, v, adt, gamma + shifted, gamma, min(chunk_size, length)
        )

    if D is not None:
        out = out + D.to(dtype).unsqueeze(-1) * v  # one scalar per head, over Dv
    if Z is not None:
        out = out * F.silu(Z.to(dtype))
    out = out.to(V.dtype)

    if return_final_states:
        angle_state = theta[:, -1, :, : Angles.shape[-1]]
        return out, angle_state, state, k_last, v_last
    return out
