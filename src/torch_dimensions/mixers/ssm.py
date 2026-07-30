"""SSM mixers — portable implementations of the S4D and Mamba recurrences.

Pure torch, so they run anywhere torch runs: CPU, CUDA, and MPS. The kernel
mathematics is derived from the reference implementations in `state-spaces/s4`
(``models/s4/s4d.py``) and `state-spaces/mamba` (``modules/mamba_simple.py``),
both Apache-2.0, Copyright the state-spaces authors (Gu, Dao, et al.). What is
deliberately *not* carried over: the fused CUDA/triton kernels (a speed
optimization, not different math), the ``pytorch_lightning`` / hub imports,
and the transposed-layout plumbing — a mixer here always sees ``(M, A, H)``.

Both mixers are causal along the swept axis. Bidirectionality belongs to the
:class:`~torch_dimensions.ScanPlan`, which hands a backward sweep to the mixer
already flipped — same contract as the RNN mixers.

``MambaMixer``'s selective scan is a sequential loop over the axis. That is
O(A) Python steps per call: correct everywhere and fast enough for the short
axes of an N-D lattice, which is the regime this library sweeps. For long 1-D
sequences on CUDA, the fused upstream kernel remains the right tool; wiring it
in as a fast path is adapter work, not different semantics.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MambaMixer", "S4DMixer", "S4Mixer"]


def _legs_dplr(d_state: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """HiPPO-LegS in DPLR form: ``A = Λ - P P*`` up to unitary equivalence.

    Returns ``(Λ, P, B)``, each ``(d_state/2,)`` complex — one of each
    conjugate pair, exactly upstream's ``nplr("legs", N)`` construction: build
    the LegS transition, add the rank-1 correction to get a (nearly) normal
    matrix, diagonalize its skew part with ``eigh``, keep the negative-
    imaginary half, and rotate ``B`` and ``P`` into the eigenbasis.
    """
    n = d_state
    q = torch.arange(n, dtype=torch.float64)
    col, row = torch.meshgrid(q, q, indexing="xy")
    m = -(torch.where(row >= col, 2 * col + 1, torch.zeros(())) - torch.diag(q))
    t = torch.sqrt(2 * q + 1)
    a = t[:, None] * m / t[None, :]
    b = t.clone()
    p = torch.sqrt(q + 0.5)

    ap = a + p[:, None] * p[None, :]
    w_re = torch.diagonal(ap).mean()
    # eigh reads only the lower triangle and assumes a real diagonal, which is
    # exactly what makes this diagonalize the skew part alone — same trick as
    # upstream, kept for bit-level fidelity of the init.
    w_im, v = torch.linalg.eigh(ap * -1j)
    w = w_re + 1j * w_im
    idx = torch.argsort(w.imag)
    v = v[:, idx][:, : n // 2]  # negative-imaginary half of each pair
    w = w[idx][: n // 2]
    v_inv = v.conj().mT
    b_half = v_inv @ b.to(v.dtype)
    p_half = v_inv @ p.to(v.dtype)
    b_half = b_half.real + 1j * b_half.imag.clamp(-2.0, 2.0)  # upstream B_clip
    return w, p_half, b_half


class _S4DKernel(nn.Module):
    """The diagonal SSM convolution kernel (S4D-Lin initialization, ZOH-style
    discretization), exactly the reference formula::

        A = -exp(log_A_real) + i·A_imag          # (H, N/2), left half-plane
        K[h, t] = 2·Re Σ_n C'[h, n]·exp(dt[h]·A[h, n])^t
        C' = C·(exp(dt·A) - 1)/A
    """

    def __init__(self, d_model: int, d_state: int = 64, dt_min: float = 1e-3, dt_max: float = 0.1):
        super().__init__()
        if d_state % 2:
            raise ValueError(f"d_state must be even (conjugate pairs); got {d_state}")
        h, n = d_model, d_state // 2
        self.log_dt = nn.Parameter(
            torch.rand(h) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        # Complex parameters live as (..., 2) real views so optimizers, state
        # dicts, and .double() all treat them as ordinary tensors.
        self.C = nn.Parameter(torch.view_as_real(torch.randn(h, n, dtype=torch.cfloat)))
        self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(h, n)))
        self.A_imag = nn.Parameter(math.pi * torch.arange(n).expand(h, n).clone())

    def forward(self, length: int) -> torch.Tensor:
        dt = torch.exp(self.log_dt)  # (H,)
        c = torch.view_as_complex(self.C)  # (H, N/2)
        a = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (H, N/2)
        dta = a * dt.unsqueeze(-1)  # (H, N/2)
        powers = dta.unsqueeze(-1) * torch.arange(length, device=dta.device)  # (H, N/2, L)
        c = c * (torch.exp(dta) - 1.0) / a
        return 2 * torch.einsum("hn, hnl -> hl", c, torch.exp(powers)).real  # (H, L)


class _S4Kernel(nn.Module):
    """The full S4 convolution kernel: diagonal *plus low-rank* (DPLR).

    ``A = Λ - P P*`` with the HiPPO-LegS initialization, discretized by the
    bilinear transform and evaluated in frequency space: Cauchy resolvent of
    the diagonal part, rank-1 Woodbury correction for the low-rank part, then
    an inverse real FFT. ``C`` here is upstream's C̃ — the truncation-corrected
    output map — learned directly, which is what upstream's own ``_setup_C``
    reduces the problem to.

    The bilinear transform ``z = 2(1-ω)/(1+ω)`` has a genuine pole at the
    Nyquist point ω = -1. Upstream survives it only because floating-point
    rounding misses -1 by ~1e-7; on backends whose complex exp lands on it
    exactly (MPS does) the whole kernel goes NaN. The denominator is guarded
    here, which reproduces what rounding does elsewhere — see PLAN.md Phase 7.
    """

    def __init__(self, d_model: int, d_state: int = 64, dt_min: float = 1e-3, dt_max: float = 0.1):
        super().__init__()
        if d_state % 2:
            raise ValueError(f"d_state must be even (conjugate pairs); got {d_state}")
        h = d_model
        w, p, b = _legs_dplr(d_state)
        self.log_dt = nn.Parameter(
            torch.rand(h) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        self.log_A_real = nn.Parameter(torch.log(-w.real.float()).expand(h, -1).clone())
        self.A_imag = nn.Parameter(w.imag.float().expand(h, -1).clone())
        self.B = nn.Parameter(torch.view_as_real(b.to(torch.cfloat).expand(h, -1).clone()))
        self.P = nn.Parameter(torch.view_as_real(p.to(torch.cfloat).expand(h, -1).clone()))
        self.C = nn.Parameter(torch.view_as_real(torch.randn(h, d_state // 2, dtype=torch.cfloat)))

    def forward(self, length: int) -> torch.Tensor:
        dt = torch.exp(self.log_dt)  # (H,)
        a = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (H, N/2)
        b = torch.view_as_complex(self.B)
        c = torch.view_as_complex(self.C)
        p = torch.view_as_complex(self.P)

        w = a * dt.unsqueeze(-1)  # dt folded into A, as upstream does
        freqs = torch.arange(length // 2 + 1, device=w.device)
        omega = torch.exp(-2j * torch.pi * freqs.to(w.real.dtype) / length)
        den = 1 + omega
        eps = torch.finfo(w.real.dtype).eps
        den = torch.where(den.abs() < eps, torch.full_like(den, eps), den)
        z = 2 * (1 - omega) / den  # (F,)

        def cauchy(v: torch.Tensor) -> torch.Tensor:
            # sum over states of v/(z - w), plus the conjugate pair
            r = (v.unsqueeze(-1) / (z - w.unsqueeze(-1))).sum(-2)
            rc = (v.conj().unsqueeze(-1) / (z - w.conj().unsqueeze(-1))).sum(-2)
            return r + rc  # (H, F)

        dtv = dt.unsqueeze(-1)
        r00 = cauchy(b * c * dtv)
        r01 = cauchy(b * p.conj() * dtv)
        r10 = cauchy(p * c * dtv)
        r11 = cauchy(p * p.conj() * dtv)
        k_f = (r00 - r01 * r10 / (1 + r11)) * 2 / den
        return torch.fft.irfft(k_f, n=length)  # (H, L)


class _KernelConvMixer(nn.Module):
    """Shared block around a convolution-kernel SSM: causal FFT conv, learned
    skip ``D``, GELU, GLU out. ``(M, A, H) -> (M, A, H)``.

    The convolution is causal because the kernel is one-sided; position ``t``
    of the output sees inputs ``0..t`` only.
    """

    _kernel: type[nn.Module]

    def __init__(self, d_model: int, d_state: int = 64, dt_min: float = 1e-3, dt_max: float = 0.1):
        super().__init__()
        self.kernel = self._kernel(d_model, d_state=d_state, dt_min=dt_min, dt_max=dt_max)
        self.D = nn.Parameter(torch.randn(d_model))
        self.act = nn.GELU()
        self.out = nn.Linear(d_model, 2 * d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[1]
        k = self.kernel(length)  # (H, L)
        xt = x.transpose(1, 2)  # (M, H, L)
        n = 2 * length  # linear (not circular) convolution
        y = torch.fft.irfft(torch.fft.rfft(xt, n=n) * torch.fft.rfft(k, n=n), n=n)[..., :length]
        y = y + xt * self.D.unsqueeze(-1)
        y = self.act(y).transpose(1, 2)
        return F.glu(self.out(y), dim=-1)


class S4DMixer(_KernelConvMixer):
    """S4D as a mixer: the diagonal kernel inside the shared conv block."""

    _kernel = _S4DKernel


class S4Mixer(_KernelConvMixer):
    """The full S4 (DPLR) as a mixer: diagonal plus low-rank kernel inside
    the shared conv block. Same interface as :class:`S4DMixer`; the extra
    capacity is the rank-1 correction that makes the HiPPO-LegS dynamics
    exact rather than diagonally approximated."""

    _kernel = _S4Kernel


class MambaMixer(nn.Module):
    """A portable Mamba (v1) block: gated depthwise conv + selective scan.

    ``(M, A, H) -> (M, A, H)``. Mirrors the reference module's non-fused path:
    in-projection to ``2·expand·H``, causal depthwise conv with SiLU, input-
    dependent ``dt``/``B``/``C``, the selective state recurrence, a learned
    skip ``D``, SiLU gating, and the out-projection.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | None = None,
        dt_min: float = 1e-3,
        dt_max: float = 0.1,
    ):
        super().__init__()
        d_inner = expand * d_model
        self.d_inner, self.d_state = d_inner, d_state
        self.dt_rank = dt_rank if dt_rank is not None else max(1, math.ceil(d_model / 16))

        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv = nn.Conv1d(
            d_inner, d_inner, d_conv, groups=d_inner, padding=d_conv - 1, bias=True
        )
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)

        # dt initialization from the reference: softplus(bias) lands in
        # [dt_min, dt_max], sampled log-uniformly.
        with torch.no_grad():
            std = self.dt_rank**-0.5
            self.dt_proj.weight.uniform_(-std, std)
            dt = torch.exp(
                torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
            ).clamp_min(1e-4)
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus

        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
            .expand(d_inner, d_state)
            .clone()
        )
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m, length, _ = x.shape
        xi, z = self.in_proj(x).chunk(2, dim=-1)  # (M, L, d_inner) each
        xi = self.conv(xi.transpose(1, 2))[..., :length].transpose(1, 2)
        xi = F.silu(xi)

        dt, b, c = torch.split(self.x_proj(xi), [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))  # (M, L, d_inner)
        a = -torch.exp(self.A_log)  # (d_inner, N)

        state = xi.new_zeros(m, self.d_inner, self.d_state)
        ys = []
        for t in range(length):
            state = state * torch.exp(dt[:, t, :, None] * a) + (
                dt[:, t, :, None] * b[:, t, None, :] * xi[:, t, :, None]
            )
            ys.append((state * c[:, t, None, :]).sum(-1))  # (M, d_inner)
        y = torch.stack(ys, dim=1) + xi * self.D
        y = y * F.silu(z)
        return self.out_proj(y)
