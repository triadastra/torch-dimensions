"""Mixers that run the original authors' code, redistributed verbatim.

``torch_dimensions._vendor`` ships the reference implementations of S4/S4D
(state-spaces/s4) and Mamba (state-spaces/mamba) byte-for-byte, Apache-2.0,
with only import-path patches — every changed line is tagged and CI proves
there are no others (see ``tests/test_vendored.py``). These adapters wrap
those originals in the ``(M, A, H)`` mixer contract, so the exact upstream
blocks can be swept over an N-D lattice by any :class:`~torch_dimensions.ScanPlan`.

Use these when you want to be certain you are training *their* model; use the
portable mixers in :mod:`torch_dimensions.mixers.ssm` when you want a smaller
dependency footprint (pure torch) or the MPS-safe DPLR kernel. The two agree
numerically — that agreement is itself a CI test.

Needs the ``[upstream]`` extra: ``pip install 'torch-dimensions[upstream]'``
(einops for all three, numpy and scipy for the S4 family).
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["UpstreamMambaMixer", "UpstreamS4DMixer", "UpstreamS4Mixer"]


def _require_upstream(*, scientific: bool = False) -> None:
    """Fail with an installation hint instead of a stack trace three frames deep."""
    missing = []
    try:
        import einops  # noqa: F401
    except ImportError:
        missing.append("einops")
    if scientific:
        for mod in ("numpy", "scipy"):
            try:
                __import__(mod)
            except ImportError:
                missing.append(mod)
    if missing:
        raise ImportError(
            f"the upstream mixers run the original authors' code, which needs "
            f"{', '.join(missing)}: pip install 'torch-dimensions[upstream]'"
        )


class UpstreamS4DMixer(nn.Module):
    """The authors' standalone S4D block, exactly as shipped.

    Wraps ``_vendor/s4/s4d.py``'s ``S4D`` (kernel + GELU + GLU output, the
    pedagogical standalone the authors publish for reuse) with
    ``transposed=False`` so it sees ``(M, A, H)`` directly.
    """

    def __init__(self, d_model: int, d_state: int = 64, dropout: float = 0.0, **kernel_args):
        super().__init__()
        _require_upstream()
        from torch_dimensions._vendor.s4.s4d import S4D

        self.block = S4D(d_model, d_state=d_state, dropout=dropout, transposed=False, **kernel_args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.block(x)
        return y


class UpstreamS4Mixer(nn.Module):
    """The authors' full S4 block (DPLR kernel), exactly as shipped.

    Wraps ``_vendor/s4/s4.py``'s ``S4Block`` with ``transposed=False``. All of
    upstream's options pass through: ``mode='diag'`` turns it into their S4D,
    ``init='legt'``/``'fourier'`` select other HiPPO measures, ``rank=2`` the
    higher-rank correction, ``bidirectional=True`` their two-sided kernel.

    A faithfulness note rather than a bug: upstream's bilinear transform has a
    genuine pole at the Nyquist frequency and survives on CPU/CUDA only
    because float rounding misses it. On MPS the rounding lands exactly on the
    pole and the kernel goes NaN. That behaviour ships as-is — the portable
    :class:`~torch_dimensions.mixers.S4Mixer` carries the guard (PLAN.md
    Phase 7); the vendored original is deliberately unmodified.
    """

    def __init__(self, d_model: int, d_state: int = 64, **layer_args):
        super().__init__()
        _require_upstream(scientific=True)
        from torch_dimensions._vendor.s4.s4 import S4Block

        self.block = S4Block(d_model, transposed=False, d_state=d_state, **layer_args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.block(x)
        return y


class UpstreamMambaMixer(nn.Module):
    """The authors' Mamba (v1) block, exactly as shipped.

    Wraps ``_vendor/mamba/mamba_simple.py``'s ``Mamba``. Off GPU the fused
    path is absent, so the module takes its own slow path and the selective
    scan dispatches to the authors' ``selective_scan_ref`` — still their code,
    end to end. On CUDA with ``mamba-ssm`` installed the fused kernels are
    picked up exactly as upstream intends.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        **mamba_args,
    ):
        super().__init__()
        _require_upstream()
        from torch_dimensions._vendor.mamba.mamba_simple import Mamba

        self.block = Mamba(d_model, d_state=d_state, d_conv=d_conv, expand=expand, **mamba_args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
