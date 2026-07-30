"""N-D composition strategies, and the registry that names them.

An **nd_method** decides how a model's extra axes are handled. It receives the
model's 1-D mixer and the lattice, and returns a module. Signature::

    nd_method(mixer, plan, lattice, d_model, *, dropout, chunk, **kw) -> nn.Module

Three strategies fit that contract, and they differ in *who handles which
axis*:

``td.axial_scan``
    The mixer sweeps every axis, one per layer. Mamba-ND and the N-D RNNs.

``td.axial_attention`` / ``td.cafa`` (Phase 6)
    Per-axis kernels contracted together — the joint operator is a Kronecker
    product. The operator *is* the kernel, so there is no mixer slot: axial
    attention and CaFA are their own thing, not an LSTM wearing a hat.

hybrid — the same two names, given a mixer (Phase 6)
    The mixer owns the sequence axis; a kernel-family operator owns the
    lattice axes. Attention or CaFA mixes across the grid at each timestep,
    then the RNN or SSM runs along time. This is the shape of most real
    forecasting models over a categorical lattice, and it is why
    ``LSTM(nd_method="cafa")`` is meaningful — CaFA never consumes the LSTM,
    it just handles the axes the LSTM does not.

Strategies are plain functions, exported at top level: ``td.axial_scan`` today,
``td.axial_attention`` and ``td.cafa`` when the kernel family lands. A
user-supplied function is a first-class strategy on exactly the same footing;
the string registry exists only because YAML cannot hold a callable.
"""

from __future__ import annotations

from collections.abc import Callable

import torch.nn as nn

from torch_dimensions.lattice import Lattice
from torch_dimensions.plan import ScanPlan

from torch_dimensions.compose.attention import AxialKernel  # isort: skip
from torch_dimensions.compose.kernel import axial_contract, kron_operator  # isort: skip
from torch_dimensions.compose.scan import AxialScan, axial_apply  # isort: skip

__all__ = [
    "ND_METHODS",
    "AxialKernel",
    "AxialScan",
    "axial_apply",
    "axial_contract",
    "axial_attention",
    "axial_scan",
    "cafa",
    "kron_operator",
    "register_nd_method",
    "resolve_nd_method",
]


def axial_scan(
    mixer: Callable[[], nn.Module] | nn.Module,
    plan: ScanPlan,
    lattice: Lattice,
    d_model: int,
    **kwargs,
) -> nn.Module:
    """Sweep the mixer along one axis per layer — the default strategy.

    ``td.LSTM(..., nd_method=td.axial_scan)``. This is the Mamba-ND / MDRNN
    shape: the model's own 1-D operator handles every axis, and the schedule
    decides which axis and which direction each layer gets.

    A strategy is a plain function, not a class, because not all of them wrap
    a single module — a hybrid strategy composes two operators over different
    axes. Passing your own function here needs no registration.
    """
    return AxialScan(mixer=mixer, plan=plan, lattice=lattice, d_model=d_model, **kwargs)


def axial_attention(
    mixer: Callable[[], nn.Module] | nn.Module | None,
    plan: ScanPlan,
    lattice: Lattice,
    d_model: int,
    **kwargs,
) -> nn.Module:
    """Per-line attention kernels over the spatial axes; the mixer runs along
    time — the hybrid form. ``td.LSTM(..., nd_method=td.axial_attention)``.

    Each layer contracts every spatial axis with a per-line softmax kernel
    (plus a learned relative-position bias), then the model's own 1-D mixer
    sweeps the time axis. The attention never consumes the mixer; it handles
    the axes the mixer does not.
    """
    return AxialKernel(
        mixer=mixer, plan=plan, lattice=lattice, d_model=d_model, per_line=True, **kwargs
    )


def cafa(
    mixer: Callable[[], nn.Module] | nn.Module | None,
    plan: ScanPlan,
    lattice: Lattice,
    d_model: int,
    **kwargs,
) -> nn.Module:
    """Factorized attention (CaFA): pooled per-axis kernels, Kronecker-
    structured, with the mixer along time. ``td.LSTM(..., nd_method=td.cafa)``.

    Cheaper than :func:`axial_attention` — one kernel per axis per (batch,
    timestep) instead of per line — and more structured: the joint operator
    is exactly a Kronecker product of the per-axis kernels. ``gate=`` selects
    ``"softmax"`` (default) or ``"leaky_relu"`` (the CaFA paper's default).
    """
    return AxialKernel(
        mixer=mixer, plan=plan, lattice=lattice, d_model=d_model, per_line=False, **kwargs
    )


ND_METHODS: dict[str, Callable[..., nn.Module]] = {
    "axial_attention": axial_attention,
    "axial_scan": axial_scan,
    "cafa": cafa,
}


def register_nd_method(name: str, factory: Callable[..., nn.Module]) -> None:
    """Make a composition strategy addressable by name.

    Only needed for config files, which cannot hold a Python callable. In
    Python, pass the function itself.
    """
    if name in ND_METHODS:
        raise ValueError(f"nd_method {name!r} is already registered")
    ND_METHODS[name] = factory


def resolve_nd_method(method: str | Callable[..., nn.Module]) -> Callable[..., nn.Module]:
    """Accept either a registered name or any callable with the strategy
    signature. Passing a callable directly is the point — a user's own
    traversal needs no registration."""
    if isinstance(method, str):
        if method not in ND_METHODS:
            raise ValueError(
                f"unknown nd_method {method!r}; registered: {sorted(ND_METHODS)}. "
                "Pass a callable to use one that is not registered."
            )
        return ND_METHODS[method]
    if not callable(method):
        raise TypeError(f"nd_method must be a name or a callable; got {type(method).__name__}")
    return method
