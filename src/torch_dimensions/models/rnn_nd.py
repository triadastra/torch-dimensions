"""N-dimensional RNNs.

Prior art exists — multi-dimensional RNNs (Graves et al., 2007) and Grid LSTM
(Kalchbrenner et al., 2015) — but no maintained modern implementation. Here
they are not a special construction at all: an N-D LSTM is
:class:`~torch_dimensions.AxialScan` with ``nn.LSTM`` as the mixer, which is
the point the whole library is making.
"""

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn

from torch_dimensions.compose.scan import AxialScan
from torch_dimensions.lattice import AxisSpec, Lattice
from torch_dimensions.mixers.rnn import GRUMixer, LSTMMixer
from torch_dimensions.plan import ScanPlan

__all__ = ["GRUND", "LSTMND"]


class _RNNND(nn.Module):
    _mixer: type[nn.Module]

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        lattice: Lattice,
        *,
        plan: ScanPlan | None = None,
        bidirectional: bool | AxisSpec | list[AxisSpec] = False,
        dropout: float = 0.0,
        chunk: int | None = None,
    ) -> None:
        super().__init__()
        if plan is None:
            plan = ScanPlan.cyclic(lattice.axis_names, n_layers, bidirectional=bidirectional)
        elif bidirectional is not False:
            raise ValueError("pass either `plan` or `bidirectional`, not both")
        self.scan = AxialScan(
            mixer=partial(self._mixer, d_model),
            plan=plan,
            lattice=lattice,
            d_model=d_model,
            dropout=dropout,
            chunk=chunk,
        )

    @property
    def plan(self) -> ScanPlan:
        return self.scan.plan

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, [T,] *shape, d_model)`` in, same shape out."""
        return self.scan(x)


class LSTMND(_RNNND):
    """N-dimensional LSTM.

    ``bidirectional`` accepts a collection of axes, so a time axis can stay
    causal while spatial axes are swept both ways. It is off by default: an
    implicit direction schedule is exactly the kind of thing that should be
    stated rather than assumed.
    """

    _mixer = LSTMMixer


class GRUND(_RNNND):
    """N-dimensional GRU. See :class:`LSTMND`."""

    _mixer = GRUMixer
