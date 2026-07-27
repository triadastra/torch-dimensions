"""What counts as a mixer.

The contract is deliberately tiny — one shape in, the same shape out — because
it is the library's extension point. A new sequence model from next month's
paper becomes usable at every rank, dense and sparse, by satisfying it.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch

__all__ = ["Mixer"]


@runtime_checkable
class Mixer(Protocol):
    """Any callable mapping ``(M, A, H) -> (M, A, H)``.

    ``M`` is the batch times every lattice axis except the swept one, ``A`` is
    the swept axis length, ``H`` is the feature width. A mixer is never told
    which axis it is sweeping, what rank the lattice is, or whether the lattice
    has absent cells — :class:`~torch_dimensions.AxialScan` owns all of that.

    Directionality is likewise not a mixer's concern. A backward sweep arrives
    pre-flipped, so mixers stay unidirectional and the schedule decides
    direction. That is why the RNN adapters do not set ``bidirectional=True``.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...
