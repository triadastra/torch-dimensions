"""State-space models, 1-D and N-D under one name.

``td.S4D(d_model, n_layers)`` with no lattice is a sequence model; give it a
lattice and it is S4ND. ``td.Mamba`` likewise is Mamba-ND with a lattice —
alternating 1-D selective scans over the axes, which is exactly the Mamba-ND
construction. The mixers are the portable implementations in
:mod:`torch_dimensions.mixers.ssm`: pure torch, verified against the upstream
reference kernels, and runnable on CPU, CUDA, and MPS alike.
"""

from __future__ import annotations

from torch_dimensions.mixers.ssm import MambaMixer, S4DMixer
from torch_dimensions.models.base import LatticeModel

__all__ = ["S4D", "Mamba"]


class S4D(LatticeModel):
    """Diagonal state-space model (S4D) over a sequence or an N-D lattice.

    Args:
        d_model: feature width.
        n_layers: sweeps; with a lattice, layers cycle through its axes.
        lattice: omit for an ordinary 1-D sequence model.
        d_state: state dimension of the diagonal SSM (even; conjugate pairs).

    Extra mixer options (``dt_min``, ``dt_max``) go in ``mixer_kwargs``.
    """

    _mixer = S4DMixer

    def __init__(self, d_model: int, n_layers: int = 1, lattice=None, *, d_state: int = 64, **kw):
        mixer_kwargs = {"d_state": d_state, **kw.pop("mixer_kwargs", {})}
        super().__init__(d_model, n_layers, lattice, mixer_kwargs=mixer_kwargs, **kw)


class Mamba(LatticeModel):
    """Mamba (selective SSM) over a sequence or an N-D lattice.

    With a lattice this is the Mamba-ND construction: each layer runs the
    selective scan along one axis, and the :class:`~torch_dimensions.ScanPlan`
    decides which axis and direction — including the paired schedule of the
    official Mamba-ND implementation via ``ScanPlan.paired``.

    Args:
        d_state: SSM state size per channel.
        d_conv: width of the causal depthwise convolution.
        expand: inner width multiplier.
    """

    _mixer = MambaMixer

    def __init__(
        self,
        d_model: int,
        n_layers: int = 1,
        lattice=None,
        *,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        **kw,
    ):
        mixer_kwargs = {
            "d_state": d_state,
            "d_conv": d_conv,
            "expand": expand,
            **kw.pop("mixer_kwargs", {}),
        }
        super().__init__(d_model, n_layers, lattice, mixer_kwargs=mixer_kwargs, **kw)
