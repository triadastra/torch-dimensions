"""N-D composition strategies, and the registry that names them.

An **nd_method** decides how a model's extra axes are handled. It receives the
model's 1-D mixer and the lattice, and returns a module. Signature::

    nd_method(mixer, plan, lattice, d_model, *, dropout, chunk, **kw) -> nn.Module

Three strategies fit that contract, and they differ in *who handles which
axis*:

``axial_scan``
    The mixer sweeps every axis, one per layer. Mamba-ND and the N-D RNNs.

``axial_kernel`` (Phase 6)
    Per-axis kernels contracted together — the joint operator is a Kronecker
    product. The operator *is* the kernel, so there is no mixer slot: axial
    attention and CaFA are their own thing, not an LSTM wearing a hat.

``hybrid`` (Phase 6)
    The mixer owns the sequence axis; a kernel-family operator owns the
    lattice axes. Attention or CaFA mixes across the grid at each timestep,
    then the RNN or SSM runs along time. This is the shape of most real
    forecasting models over a categorical lattice, and it is why
    ``LSTM(nd_method="cafa")`` is meaningful — CaFA never consumes the LSTM,
    it just handles the axes the LSTM does not.

A user-supplied callable is a first-class strategy; registration only exists
so a name can be written in a config file.
"""

from __future__ import annotations

from collections.abc import Callable

import torch.nn as nn

from torch_dimensions.compose.scan import AxialScan, axial_apply

__all__ = ["ND_METHODS", "AxialScan", "axial_apply", "register_nd_method", "resolve_nd_method"]

ND_METHODS: dict[str, Callable[..., nn.Module]] = {
    "axial_scan": AxialScan,
}


def register_nd_method(name: str, factory: Callable[..., nn.Module]) -> None:
    """Make a composition strategy addressable by name, so it can be selected
    from config as well as from Python."""
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
