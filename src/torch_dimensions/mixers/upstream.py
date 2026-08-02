"""Mixers that run the original authors' code, redistributed verbatim.

``torch_dimensions._vendor`` ships the reference implementations byte-for-byte
— for S4/S4D the subtree that upstream's **train.py actually runs** (the
``S4Block`` layer their registry calls ``"s4"``, its kernels, HiPPO, and
utils), not a standalone re-export; for Mamba the reference block and the
authors' own pure-torch selective scan. Every patched line is tagged and CI
proves there are no others (``tests/test_vendored.py``). These adapters wrap
those originals in the ``(M, A, H)`` mixer contract, so the exact upstream
blocks can be swept over an N-D lattice by any
:class:`~torch_dimensions.ScanPlan`.

Use these when you want to be certain you are training *their* model; use the
portable mixers in :mod:`torch_dimensions.mixers.ssm` when you want a smaller
dependency footprint (pure torch) or the MPS-safe DPLR kernel. The two agree
numerically — that agreement is itself a CI test.

Needs the ``[upstream]`` extra: ``pip install 'torch-dimensions[upstream]'``.
einops for everything; numpy, scipy, hydra-core and omegaconf for the S4
pipeline (hydra because upstream's own ``S4Block`` builds its inner layer
through their hydra-backed registry — that is the real construction path, so
it is the one that runs).

One caveat inherited from upstream's repo layout: the pipeline imports itself
as ``src.*``, so constructing an S4 mixer mounts the vendored tree as the
``src`` package (see ``torch_dimensions._vendor.s4.mount``). A process that
already imported its own top-level ``src`` gets a clear ``ImportError``
rather than a silent shadowing.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys

import torch
import torch.nn as nn

__all__ = ["UpstreamMambaMixer", "UpstreamS4DMixer", "UpstreamS4Mixer"]

_S4_DEPS = ("einops", "numpy", "scipy", "hydra", "omegaconf")
_PIP_NAMES = {"hydra": "hydra-core"}


def _missing(modules: tuple[str, ...]) -> list[str]:
    out = []
    for mod in modules:
        try:
            __import__(mod)
        except ImportError:
            out.append(mod)
    return out


def _require_upstream(modules: tuple[str, ...] = ("einops",)) -> None:
    """Make the originals' own dependencies available, installing on first use.

    The upstream mixers are the *default* implementation, but their
    dependencies are not `torch-dimensions` dependencies: someone who never
    constructs an upstream-backed model, or always passes ``portable=True``,
    never needs them and never gets them. On the first call that does need
    them, the missing ones are pip-installed into the running interpreter's
    environment — announced on stderr first, disabled entirely by setting
    ``TD_NO_AUTO_INSTALL=1``, in which case (and whenever installation fails)
    this raises with the manual command instead.
    """
    missing = _missing(modules)
    if not missing:
        return
    pips = [_PIP_NAMES.get(m, m) for m in missing]
    manual = "pip install 'torch-dimensions[upstream]'"
    if os.environ.get("TD_NO_AUTO_INSTALL"):
        raise ImportError(
            f"the upstream reference code (the default mixers) needs {', '.join(pips)}. "
            f"Auto-install is disabled by TD_NO_AUTO_INSTALL; run {manual}, or pass "
            "portable=True for the pure-torch build."
        )
    print(
        f"torch-dimensions: the default mixers run the original authors' code, which "
        f"needs {', '.join(pips)}; installing with {sys.executable} -m pip "
        "(set TD_NO_AUTO_INSTALL=1 to disable, or pass portable=True to avoid entirely)",
        file=sys.stderr,
    )
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", *pips], check=True)
    except (subprocess.CalledProcessError, OSError) as e:
        raise ImportError(
            f"auto-installing {', '.join(pips)} failed; run {manual} yourself, "
            "or pass portable=True for the pure-torch build"
        ) from e
    importlib.invalidate_caches()
    still = [_PIP_NAMES.get(m, m) for m in _missing(modules)]
    if still:
        raise ImportError(
            f"{', '.join(still)} still not importable after installation; run {manual} "
            "in the right environment, or pass portable=True"
        )


def _s4_block(d_model: int, **kwargs) -> nn.Module:
    """Construct upstream's real S4Block, mounting the pipeline first."""
    _require_upstream(_S4_DEPS)
    from torch_dimensions._vendor.s4 import mount

    mount()
    from src.models.sequence.modules.s4block import S4Block  # noqa: PLC0415 — their name

    return S4Block(d_model, transposed=False, **kwargs)


class UpstreamS4DMixer(nn.Module):
    """The S4D configuration of upstream's real ``S4Block``, exactly as
    their pipeline runs it: ``S4Block(..., mode='diag')``, which builds
    ``FFTConv`` around ``SSMKernelDiag`` through their own registry.

    Kernel options pass through: ``init='diag-lin'`` (the S4D-Lin paper
    setup), ``disc='zoh'|'bilinear'``, ``d_state``, ``dt_min``/``dt_max`` …
    """

    def __init__(self, d_model: int, d_state: int = 64, **layer_args):
        super().__init__()
        self.block = _s4_block(d_model, mode="diag", d_state=d_state, **layer_args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.block(x)
        return y


class UpstreamS4Mixer(nn.Module):
    """Upstream's real ``S4Block`` with the full DPLR kernel — the layer their
    registry calls ``"s4"`` and their train.py instantiates.

    All of upstream's options pass through: ``init='legt'``/``'fourier'``
    select other HiPPO measures, ``rank=2`` the higher-rank correction,
    ``bidirectional=True`` their two-sided kernel, ``gate``/``bottleneck``
    the GSS variants.

    A portability note: upstream's bilinear transform has a genuine pole at
    the Nyquist frequency and survives on CPU/CUDA only because float
    rounding misses it; MPS's power op lands on it exactly at some lengths
    (L=64 did) and the kernel went NaN. The vendored ``ssm.py`` carries a
    tagged guard that nudges only an *exact* pole hit — never true on
    CPU/CUDA, where results are verified bit-for-bit unchanged — so this
    layer now runs on CPU, CUDA and MPS alike (CPU-vs-MPS ≤ 1e-6 across
    L=16..256, gradients finite).
    """

    def __init__(self, d_model: int, d_state: int = 64, **layer_args):
        super().__init__()
        self.block = _s4_block(d_model, mode="dplr", d_state=d_state, **layer_args)

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
