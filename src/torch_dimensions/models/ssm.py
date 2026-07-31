"""State-space models, 1-D and N-D under one name.

``td.S4(d_model, n_layers)`` with no lattice is a sequence model; give it a
lattice and it is S4ND. Same for ``td.S4D`` and ``td.Mamba``. The explicit N-D
names — ``td.S4ND``, ``td.S4DND``, ``td.MambaND`` — are the same classes with
``dim`` and the lattice made mandatory: taking the N-D name means declaring
what N is, the declaration is checked, and ``dim=1`` is refused outright —
one spatial axis is the 1-D model, and code reading "S4ND" must not be
running S4. Pass ``lattice=...`` or just ``shape=(32, 32)`` and the lattice
is constructed for you.

The mixers are the portable implementations in
:mod:`torch_dimensions.mixers.ssm`: pure torch, verified against the upstream
reference kernels, and runnable on CPU, CUDA, and MPS alike. How the axes are
composed stays ``nd_method``'s business (default :func:`~torch_dimensions.
axial_scan`), exactly as for the RNN family.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from torch_dimensions.lattice import Lattice
from torch_dimensions.mixers.ssm import MambaMixer, S4DMixer, S4Mixer
from torch_dimensions.models.base import LatticeModel

__all__ = ["S4", "S4D", "S4DND", "S4ND", "Mamba", "MambaND"]


class S4(LatticeModel):
    """The full S4 (DPLR: diagonal plus low-rank) over a sequence or lattice.

    Args:
        d_model: feature width.
        n_layers: sweeps; with a lattice, layers cycle through its axes.
        lattice: omit for an ordinary 1-D sequence model.
        d_state: full SSM state size (even; stored as conjugate pairs).

    The kernel carries the rank-1 HiPPO-LegS correction that distinguishes S4
    from its diagonal approximation :class:`S4D`.
    """

    _mixer = S4Mixer

    def __init__(self, d_model: int, n_layers: int = 1, lattice=None, *, d_state: int = 64, **kw):
        mixer_kwargs = {"d_state": d_state, **kw.pop("mixer_kwargs", {})}
        super().__init__(d_model, n_layers, lattice, mixer_kwargs=mixer_kwargs, **kw)


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


def _nd_lattice(
    cls_name: str,
    lattice: Lattice | None,
    shape: Sequence[int] | None,
    names: Sequence[str] | None,
    valid: torch.Tensor | None,
    time: bool,
    dim: int | None,
) -> Lattice:
    """Resolve the N-D classes' lattice sugar, refusing the ambiguous cases."""
    # dim is mandatory on the explicit N-D names: taking the N-D name means
    # declaring what N is, and the declaration is checked against the lattice.
    # Redundant next to `shape` on purpose — that redundancy is the check, and
    # it is what catches "I thought this lattice was 3-D".
    if dim is None:
        raise ValueError(
            f"{cls_name} requires `dim` — declare the number of spatial axes, "
            f"e.g. td.{cls_name}(64, 8, dim=2, shape=(32, 32))"
        )
    base = cls_name.removesuffix("ND")
    if dim == 1:
        # Refused loudly rather than accepted quietly: a model built this way
        # would *be* the 1-D model, and someone reading "S4ND" in their code
        # or their spec would believe they are running something they are not.
        raise ValueError(
            f"dim=1 is not {cls_name} — one spatial axis is just {base}. "
            f"Use td.{base}(..., lattice=...) so the code says what is actually running."
        )
    if dim < 1:
        raise ValueError(f"the N-D names need dim >= 2; got {dim}")
    if lattice is None:
        if shape is None:
            raise ValueError(
                f"{cls_name} needs a lattice — pass `lattice=...` or let it build one: "
                f"td.{cls_name}(64, 8, shape=(32, 32))"
            )
        lattice = Lattice(
            shape=tuple(shape), names=tuple(names) if names else None, valid=valid, time=time
        )
    elif shape is not None or names is not None or valid is not None:
        raise ValueError("pass either `lattice` or `shape`/`names`/`valid`, not both")
    if lattice.rank < 1:
        raise ValueError(
            f"{cls_name} is the N-D name and this lattice has no spatial axes; "
            f"for a plain sequence use td.{cls_name.removesuffix('ND')}"
        )
    if lattice.rank != dim:
        raise ValueError(f"dim={dim}, but the lattice has {lattice.rank} spatial axes")
    return lattice


def _nd_variant(base: type[LatticeModel], cls_name: str) -> type[LatticeModel]:
    class ND(base):  # type: ignore[valid-type, misc]
        def __init__(
            self,
            d_model: int,
            n_layers: int = 1,
            lattice: Lattice | None = None,
            *,
            shape: Sequence[int] | None = None,
            names: Sequence[str] | None = None,
            valid: torch.Tensor | None = None,
            time: bool = True,
            dim: int | None = None,
            **kw,
        ):
            lat = _nd_lattice(cls_name, lattice, shape, names, valid, time, dim)
            super().__init__(d_model, n_layers, lat, **kw)
            # The N-D name's declaration is part of its recipe: a rebuild
            # must satisfy the same mandatory-dim contract it was built under.
            self.config["dim"] = lat.rank

    ND.__name__ = ND.__qualname__ = cls_name
    ND.__doc__ = (
        f"{base.__name__} with `dim` and a lattice mandatory — the explicit N-D name.\n\n"
        f"    td.{cls_name}(64, 8, dim=2, shape=(32, 32))       # builds the lattice\n"
        f"    td.{cls_name}(64, 8, dim=2, lattice=my_lattice)   # or bring your own\n\n"
        "Taking the N-D name means declaring what N is; ``dim`` is checked\n"
        "against the lattice's spatial rank. ``time=True`` by default.\n"
        f"Identical to ``td.{base.__name__}`` with a lattice in every other way."
    )
    return ND


S4ND = _nd_variant(S4, "S4ND")
S4DND = _nd_variant(S4D, "S4DND")
MambaND = _nd_variant(Mamba, "MambaND")
