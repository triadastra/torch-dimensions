"""The shared shape of every model in the library.

A model here is nothing but: an optional input projection, a 1-D mixer type,
and an ``nd_method`` deciding how that mixer covers the lattice. One class
holds that recipe; ``LSTM``, ``GRU``, ``S4D``, and ``Mamba`` differ only in
``_mixer``. This is the design's central claim made literal — an N-D model is
a 1-D mixer plus a plan for sweeping it.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from functools import partial
from typing import cast

import torch
import torch.nn as nn

from torch_dimensions.compose import axial_scan, resolve_nd_method
from torch_dimensions.lattice import AxisSpec, Lattice
from torch_dimensions.plan import ScanPlan

__all__ = ["LatticeModel"]


class LatticeModel(nn.Module):
    """Base for the model family. Subclasses set ``_mixer``.

    With no lattice this is an ordinary sequence model — a lattice with no
    spatial axes has an identity permutation, so the 1-D case is the N-D case
    with nothing to fold.
    """

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
        mixer_kwargs: dict | None = None,
        **method_kwargs,
    ) -> None:
        super().__init__()
        # No lattice means a single dynamic axis: an ordinary sequence.
        self.lattice = lattice if lattice is not None else Lattice(shape=(), time=True)

        if plan is None:
            plan = ScanPlan.cyclic(self.lattice.axis_names, n_layers, bidirectional=bidirectional)
        else:
            if bidirectional is not False:
                raise ValueError("pass either `plan` or `bidirectional`, not both")
            # A plan *is* the layer schedule, so it fixes the depth, and it
            # wins. But winning *silently* over a disagreeing n_layers would
            # ship a model shallower (or deeper) than requested — the same
            # silent-downgrade failure the schedule machinery exists to make
            # loud. A warning rather than an error because generic builders
            # legitimately fill n_layers unconditionally and add a plan only
            # sometimes; n_layers=1 is the default and passes untouched.
            if n_layers != 1 and n_layers != len(plan):
                warnings.warn(
                    f"n_layers={n_layers} is ignored: the given plan has {len(plan)} steps "
                    "and a plan determines the depth",
                    UserWarning,
                    stacklevel=2,
                )

        # An input projection only when the data is not already d_model wide.
        # Without it every caller writes the same nn.Linear, which is friction
        # for no gain in purity.
        self.in_proj = nn.Linear(d_input, d_model) if d_input is not None else nn.Identity()

        self.nd = resolve_nd_method(nd_method)(
            mixer=partial(self._mixer, d_model, **(mixer_kwargs or {})),
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
