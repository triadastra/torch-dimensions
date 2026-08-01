"""A JSON description of a model's N-D architecture.

The viewer reads this; the library never imports anything JS-adjacent and never
opens a socket. Keeping the boundary at a versioned document also makes the
spec useful on its own — for diffing two configs, for debugging a schedule, and
for documenting what a model actually does along each axis.

    import torch_dimensions as td
    spec = td.spec(model)          # plain dict, json.dumps-able

Everything here is derived without running a forward pass, so a spec can be
taken before any data exists.
"""

from __future__ import annotations

import math
from typing import Any, cast

import torch
import torch.nn as nn

from torch_dimensions.lattice import Lattice
from torch_dimensions.plan import ScanPlan

__all__ = ["SPEC_VERSION", "spec"]

SPEC_FORMAT = "torch-dimensions/architecture"
# v2: layers describe what their *family* actually does. v1 assumed every
# model was a scan, so a kernel-family model's spec claimed one spatial sweep
# per layer — sweeps that never happen — and the viewer drew them. See
# DEBUG.md #26.
SPEC_VERSION = 2


def _rle(flags: torch.Tensor) -> list[int]:
    """Run-length encode a flattened bool mask, starting with a False run.

    Lossless and compact, which matters because the viewer needs every cell's
    presence to render and a coordinate list would be enormous on a large
    lattice. A fully present lattice encodes as ``[0, n]``.
    """
    runs: list[int] = []
    current = False
    count = 0
    for value in flags.reshape(-1).tolist():
        if bool(value) == current:
            count += 1
        else:
            runs.append(count)
            current, count = bool(value), 1
    runs.append(count)
    return runs


def lattice_spec(lat: Lattice) -> dict[str, Any]:
    """Describe a lattice, including which cells exist."""
    axes: list[dict[str, Any]] = []
    if lat.time:
        # Time has no static size; saying so beats emitting a fake one.
        axes.append({"name": "time", "size": None, "dynamic": True})
    for name, size in zip(lat.names or (), lat.shape, strict=True):
        axes.append({"name": name, "size": size, "dynamic": False})

    present = torch.ones(lat.shape, dtype=torch.bool) if lat.valid is None else lat.valid
    return {
        "shape": list(lat.shape),
        "names": list(lat.axis_names),
        "time": lat.time,
        "rank": lat.rank,
        "n_axes": lat.n_axes,
        "axes": axes,
        "cells": {
            "total": lat.n_cells,
            "present": lat.n_valid,
            "dense": lat.is_dense,
            # RLE over the flattened lattice in row-major order.
            "present_rle": _rle(present),
        },
    }


def plan_spec(plan: ScanPlan, lat: Lattice) -> list[dict[str, Any]]:
    """Per-layer sweep schedule, with axes named rather than indexed.

    The scan family's layer description: one axis, one direction, per layer.
    The other families do something else and say so — see :func:`layers_spec`.
    """
    resolved = plan.resolve(lat) if not plan.is_resolved() else plan
    return [
        {
            "layer": i,
            "kind": "scan",
            "axis": lat.axis_names[cast(int, step.axis)],
            "axis_index": cast(int, step.axis),
            "reverse": step.reverse,
            "axes": [lat.axis_names[cast(int, step.axis)]],
        }
        for i, step in enumerate(resolved)
    ]


def _family(nd: nn.Module) -> str:
    """Which composition family this model uses.

    Was hardcoded to ``"scan"``, which made every kernel-family spec claim to
    be something it is not (DEBUG.md #26).
    """
    from torch_dimensions.compose.attention import AxialKernel
    from torch_dimensions.compose.scan import AxialScan

    if isinstance(nd, AxialScan):
        return "scan"
    if isinstance(nd, AxialKernel):
        return "kernel"
    return type(nd).__name__


def kernel_layers_spec(nd: Any, lat: Lattice) -> list[dict[str, Any]]:
    """Per-layer description for the kernel family.

    Every layer contracts **all** the spatial axes — not one per layer — and
    then, in the hybrid form, sweeps the mixer along time. Describing this with
    the scan family's schema produced a document claiming layer 1 swept ``h``
    with an LSTM, which is not what runs and is what the viewer drew.
    """
    spatial = [lat.axis_names[a] for a in nd.spatial_axes]
    has_mixer = getattr(nd, "mixers", None) is not None
    out = []
    for i in range(len(nd.plan)):
        mixer = nd.mixers[i] if has_mixer else None
        out.append(
            {
                "layer": i,
                "kind": "kernel",
                # The axis actually *swept*, which for this family is time or
                # nothing at all.
                "axis": "time" if has_mixer else None,
                "axis_index": 0 if has_mixer else None,
                "reverse": False,
                "axes": [*spatial, *(["time"] if has_mixer else [])],
                "contracted": spatial,
                "mixer": type(mixer).__name__ if mixer is not None else None,
                "n_params": _n_params(mixer) if mixer is not None else 0,
            }
        )
    return out


def _n_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def sweeps_spec(plan: ScanPlan, lat: Lattice) -> dict[str, Any]:
    """Which directions each axis is actually swept in, and which are missed.

    Surfaced explicitly because "every layer sweeps this axis the same way" is
    invisible in code and obvious in a picture — it is the failure the viewer
    exists to make loud. Derived from :meth:`ScanPlan.coverage`, the one place
    that computation lives.
    """
    cov = plan.coverage(lat)
    return {
        "directions": cov.directions(),
        "unswept_axes": list(cov.unswept),
        "pinned_axes": list(cov.pinned),
        "coverage": cov.to_dict(),
    }


def spec(model: nn.Module) -> dict[str, Any]:
    """Build the architecture spec for a model.

    Works on any model exposing ``.lattice`` and ``.nd`` — the shape every
    model in the scan family has. Anything else raises rather than emitting a
    half-filled document.
    """
    describe = getattr(model, "to_spec", None)
    if callable(describe):
        return cast(dict, describe())
    raise TypeError(
        f"{type(model).__name__} does not describe itself; implement to_spec() "
        "or pass one of the library's models"
    )


def scan_model_spec(model: nn.Module) -> dict[str, Any]:
    """The spec for a composed model. Used by the models' ``to_spec``.

    Named for the scan family because that is all there was when it was
    written; it now describes whichever family the model actually uses.
    """
    lat = cast(Lattice, model.lattice)
    nd: Any = model.nd  # Module.__getattr__ erases the type
    plan: ScanPlan = nd.plan
    family = _family(nd)

    if family == "kernel":
        layers = kernel_layers_spec(nd, lat)
        spatial = [lat.axis_names[a] for a in nd.spatial_axes]
        has_mixer = getattr(nd, "mixers", None) is not None
        mixed = {*spatial, *(["time"] if has_mixer else [])}
        sweeps: dict[str, Any] = {
            # Only the axis a mixer actually sweeps has a direction. The
            # kernels are not directional at all — a contraction has no
            # forward or backward — so listing them here would invent a
            # property the model does not have.
            "directions": {"time": "forward"} if has_mixer else {},
            "contracted_axes": spatial,
            "unswept_axes": [n for n in lat.axis_names if n not in mixed],
            "pinned_axes": ["time"] if has_mixer else [],
            "coverage": None,
        }
    else:
        layers = plan_spec(plan, lat)
        mixers = [
            {"layer": i, "type": type(m).__name__, "n_params": _n_params(m)}
            for i, m in enumerate(nd.mixers)
        ]
        for layer, mixer in zip(layers, mixers, strict=True):
            layer.update({"mixer": mixer["type"], "n_params": mixer["n_params"]})
        sweeps = {**sweeps_spec(plan, lat), "contracted_axes": []}

    d_model: int = nd.d_model
    in_proj = getattr(model, "in_proj", None)
    d_input = in_proj.in_features if isinstance(in_proj, nn.Linear) else d_model

    lead = ["B", "T"] if lat.time else ["B"]
    return {
        "format": SPEC_FORMAT,
        "version": SPEC_VERSION,
        "model": {
            "kind": type(model).__name__,
            "d_model": d_model,
            "d_input": d_input,
            "n_layers": len(layers),
            "n_params": _n_params(model),
        },
        "nd_method": {
            "name": type(nd).__name__,
            "family": family,
        },
        "lattice": lattice_spec(lat),
        "layers": layers,
        "sweeps": sweeps,
        "io": {
            "input": [*lead, *lat.shape, d_input],
            "output": [*lead, *lat.shape, d_model],
            "cells_per_step": math.prod(lat.shape),
        },
    }
