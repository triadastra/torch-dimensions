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

from collections.abc import Callable
from functools import partial
from typing import cast

import torch
import torch.nn as nn

from torch_dimensions.compose import axial_scan, resolve_nd_method
from torch_dimensions.lattice import AxisSpec, Lattice
from torch_dimensions.mixers.rnn import GRUMixer, LSTMMixer
from torch_dimensions.plan import ScanPlan

__all__ = ["GRU", "LSTM"]


class _RNNFamily(nn.Module):
    _mixer: type[nn.Module]

    def __init__(
        self,
        d_model: int,
        n_layers: int = 1,
        lattice: Lattice | None = None,
        *,
        d_input: int | None = None,
        nd_method: str | Callable[..., nn.Module] = axial_scan,
        plan: ScanPlan | None = None,
        bidirectional: bool | AxisSpec | list[AxisSpec] = False,
        dropout: float = 0.0,
        chunk: int | None = None,
        **method_kwargs,
    ) -> None:
        super().__init__()
        # No lattice means a single dynamic axis: an ordinary sequence.
        self.lattice = lattice if lattice is not None else Lattice(shape=(), time=True)

        if plan is None:
            plan = ScanPlan.cyclic(self.lattice.axis_names, n_layers, bidirectional=bidirectional)
        elif bidirectional is not False:
            raise ValueError("pass either `plan` or `bidirectional`, not both")

        # An input projection only when the data is not already d_model wide.
        # Without it every caller writes the same nn.Linear, which is friction
        # for no gain in purity.
        self.in_proj = nn.Linear(d_input, d_model) if d_input is not None else nn.Identity()

        self.nd = resolve_nd_method(nd_method)(
            mixer=partial(self._mixer, d_model),
            plan=plan,
            lattice=self.lattice,
            d_model=d_model,
            dropout=dropout,
            chunk=chunk,
            **method_kwargs,
        )

    @property
    def plan(self) -> ScanPlan:
        return cast(ScanPlan, self.nd.plan)

    def to_spec(self) -> dict:
        """A JSON-able description of this model's N-D architecture.

        Derived without a forward pass, so it can be taken before any data
        exists. See VIEWER.md.
        """
        from torch_dimensions.spec import scan_model_spec

        return scan_model_spec(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, [T,] *shape, d_input or d_model)`` in, ``d_model`` out.

        With no lattice that is just ``(B, T, ...)``.
        """
        return self.nd(self.in_proj(x))


class LSTM(_RNNFamily):
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


class GRU(_RNNFamily):
    """GRU over a sequence, or over an N-D lattice. See :class:`LSTM`."""

    _mixer = GRUMixer
