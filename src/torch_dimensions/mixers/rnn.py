"""RNN mixers — thin adapters over ``torch.nn``.

These exist to normalize the interface, not to reimplement anything.
``nn.LSTM`` returns ``(output, (h_n, c_n))``; a mixer returns just the output.

Neither adapter sets ``bidirectional=True``. Direction belongs to the scan
schedule, which hands a backward sweep to the mixer already flipped. Letting
the RNN do it too would double the feature width and give the schedule nothing
to control.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["GRUMixer", "LSTMMixer"]


class _RNNMixer(nn.Module):
    _cls: type[nn.RNNBase]

    def __init__(self, d_model: int, num_layers: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.rnn = self._cls(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return out


class LSTMMixer(_RNNMixer):
    """``nn.LSTM`` as a mixer. Hidden size is tied to ``d_model`` so the stack
    never changes feature width."""

    _cls = nn.LSTM


class GRUMixer(_RNNMixer):
    """``nn.GRU`` as a mixer."""

    _cls = nn.GRU
