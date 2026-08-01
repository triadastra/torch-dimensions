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
SPEC_VERSION = 1


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
    """Per-layer sweep schedule, with axes named rather than indexed."""
    resolved = plan.resolve(lat) if not plan.is_resolved() else plan
    return [
        {
            "layer": i,
            "axis": lat.axis_names[cast(int, step.axis)],
            "axis_index": cast(int, step.axis),
            "reverse": step.reverse,
        }
        for i, step in enumerate(resolved)
    ]


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
    """The spec for a scan-family model. Used by the models' ``to_spec``."""
    lat = cast(Lattice, model.lattice)
    nd: Any = model.nd  # Module.__getattr__ erases the type
    plan: ScanPlan = nd.plan

    mixers = [
        {"layer": i, "type": type(m).__name__, "n_params": _n_params(m)}
        for i, m in enumerate(nd.mixers)
    ]
    layers = plan_spec(plan, lat)
    for layer, mixer in zip(layers, mixers, strict=True):
        layer.update({"mixer": mixer["type"], "n_params": mixer["n_params"]})

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
            "family": "scan",
        },
        "lattice": lattice_spec(lat),
        "layers": layers,
        "sweeps": sweeps_spec(plan, lat),
        "io": {
            "input": [*lead, *lat.shape, d_input],
            "output": [*lead, *lat.shape, d_model],
            "cells_per_step": math.prod(lat.shape),
        },
    }
