"""Recurrent models, 1-D and N-D under one name.

``td.LSTM(d_model, n_layers)`` with no lattice is an ordinary sequence model.
Give it a ``lattice`` and the same class becomes N-dimensional. There is no
separate ``LSTMND`` because there is no separate mechanism: a lattice with no
spatial axes has an identity permutation, so the 1-D case is the N-D case with
nothing to fold.

How the extra axes are handled is ``nd_method``'s business, not this class's.
The default sweeps them with the RNN itself; a kernel-family method instead
mixes across the lattice and leaves the RNN to run along time.

A multi-layer stack here is pre-norm and residual, which ``nn.LSTM(num_layers=k)``
is not. One layer with ``norm=False, residual=False`` reproduces ``nn.LSTM``
exactly; beyond that these are modern defaults, not a drop-in reimplementation.
"""

from __future__ import annotations

from torch_dimensions.mixers.rnn import GRUMixer, LSTMMixer
from torch_dimensions.models.base import LatticeModel

__all__ = ["GRU", "LSTM"]


class LSTM(LatticeModel):
    """LSTM over a sequence, or over an N-D lattice.

    Args:
        d_model: feature width the model works in, and the output width.
        d_input: width of the incoming features, when it differs from
            ``d_model``. Adds a single input projection; omit it and the input
            is expected to already be ``d_model`` wide.
        n_layers: how many sweeps. With a lattice, layers cycle through its
            axes unless ``plan`` says otherwise.
        lattice: omit for an ordinary 1-D sequence model.
        nd_method: how the extra axes are handled — a registered name or any
            callable with the strategy signature. Defaults to
            :func:`~torch_dimensions.axial_scan`, which sweeps every axis with
            the RNN. Supplying your own function is the supported way to add a
            traversal the library has never heard of.
        bidirectional: ``True``/``False``, or the axes to sweep both ways, so a
            time axis can stay causal while spatial axes do not. Off by
            default: an implicit direction schedule should be stated, not
            assumed.

    ``bidirectional`` is a property of the *schedule*, not of the underlying
    ``nn.LSTM`` — a backward sweep arrives pre-flipped, and the feature width
    never doubles.
    """

    _mixer = LSTMMixer


class GRU(LatticeModel):
    """GRU over a sequence, or over an N-D lattice. See :class:`LSTM`."""

    _mixer = GRUMixer
