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

__all__ = ["MambaMixer", "S4DMixer"]


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


class S4DMixer(nn.Module):
    """S4D as a mixer: causal kernel convolution via FFT, skip, GELU, GLU out.

    ``(M, A, H) -> (M, A, H)``. The convolution is causal because the kernel
    is one-sided; position ``t`` of the output sees inputs ``0..t`` only.
    """

    def __init__(self, d_model: int, d_state: int = 64, dt_min: float = 1e-3, dt_max: float = 0.1):
        super().__init__()
        self.kernel = _S4DKernel(d_model, d_state=d_state, dt_min=dt_min, dt_max=dt_max)
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
