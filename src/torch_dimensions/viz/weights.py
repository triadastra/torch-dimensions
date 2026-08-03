"""The weights themselves, small enough to draw.

``spec(model)`` says what the architecture *is*; this says what its parameters
currently hold, in a form a diagram can render: every tensor of every mixer,
classified by the role it plays in that mixer's arithmetic, downsampled to
something a screen can show, with the full statistics kept alongside.

    payload = td.viz.weights(model)      # JSON-able
    td.viz.show(model)                   # serves it at /weights.json

**Downsampling is stated, never silent.** A 512x512 projection cannot be drawn
edge by edge; what is drawn is a strided sample of it, and every tensor entry
carries ``sampled``, its original ``shape``, and the stride used. A diagram
that quietly showed one corner of a matrix, or a mean of it, would be a
picture of something other than the model.

The role names are what let a viewer draw the right *kind* of diagram rather
than a heatmap for everything: a linear map is a bipartite graph of units, a
convolution is a set of taps over a receptive field, and an SSM is a decay per
state with an input and an output map. Those are different pictures because
they are different mechanisms.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

__all__ = ["WEIGHTS_FORMAT", "WEIGHTS_VERSION", "weights"]

WEIGHTS_FORMAT = "torch-dimensions/weights"
WEIGHTS_VERSION = 1

# Enough to see structure in, few enough to draw as individual edges. 24x24 is
# 576 lines, which reads as a pattern; 64x64 is 4096, which reads as a smear.
MAX_UNITS = 24


def _role(name: str, param: torch.Tensor, owner: nn.Module | None) -> str:
    """What part this tensor plays, from its owner and its name.

    Owner first: an ``nn.Conv1d`` weight is a convolution kernel whatever it is
    called, and a name-only rule would mislabel it the moment a mixer names a
    linear layer ``conv_proj``.
    """
    lower = name.lower()
    if param.ndim == 1:
        if "bias" in lower:
            return "bias"
        if "_log" in lower or lower.endswith("a_log"):
            return "ssm_decay"
        if lower.endswith(".d") or lower == "d":
            return "skip"
        return "vector"
    if isinstance(owner, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return "conv"
    if isinstance(owner, nn.Linear):
        return "linear"
    # SSM parameters live on plain Modules, so they are named rather than typed.
    if "a_log" in lower or "a_real" in lower or "a_imag" in lower or lower.endswith("a"):
        return "ssm_decay"
    if lower.endswith("b") or ".b" in lower:
        return "ssm_in"
    if lower.endswith("c") or ".c" in lower:
        return "ssm_out"
    if param.ndim >= 2:
        return "linear"
    return "other"


def _sample(param: torch.Tensor, max_units: int) -> dict[str, Any]:
    """A drawable 2-D view of a tensor, plus how it was reduced to one.

    Conv kernels keep their taps: a ``(out, in, k)`` kernel becomes
    ``(out*in sampled, k)`` so every tap survives, because the taps *are* the
    receptive field and averaging them away would erase the thing being drawn.
    """
    t = param.detach().to(torch.float32).cpu()
    original = tuple(t.shape)

    if t.ndim == 1:
        t = t.reshape(-1, 1)
    elif t.ndim > 2:
        # (out, in, *kernel) -> (out*in, prod(kernel)): rows are channel pairs,
        # columns are taps.
        t = t.reshape(original[0] * original[1], -1)

    rows, cols = t.shape
    r_stride = max(1, -(-rows // max_units))
    c_stride = max(1, -(-cols // max_units))
    view = t[::r_stride, ::c_stride]

    return {
        "shape": list(original),
        "rows": int(view.shape[0]),
        "cols": int(view.shape[1]),
        "stride": [r_stride, c_stride],
        "sampled": bool(r_stride > 1 or c_stride > 1),
        "values": [[round(float(v), 4) for v in row] for row in view],
    }


def _stats(param: torch.Tensor) -> dict[str, float]:
    t = param.detach().to(torch.float32).cpu()
    return {
        "n": int(t.numel()),
        "min": round(float(t.min()), 5) if t.numel() else 0.0,
        "max": round(float(t.max()), 5) if t.numel() else 0.0,
        "mean": round(float(t.mean()), 5) if t.numel() else 0.0,
        "std": round(float(t.std(unbiased=False)), 5) if t.numel() > 1 else 0.0,
        "absmax": round(float(t.abs().max()), 5) if t.numel() else 0.0,
    }


def _tensors(module: nn.Module, max_units: int) -> list[dict[str, Any]]:
    owners = dict(module.named_modules())
    out = []
    for name, param in module.named_parameters():
        owner = owners.get(name.rsplit(".", 1)[0]) if "." in name else module
        out.append(
            {
                "name": name,
                "role": _role(name, param, owner),
                "owner": type(owner).__name__ if owner is not None else None,
                **_sample(param, max_units),
                "stats": _stats(param),
            }
        )
    return out


@torch.no_grad()
def _operator(mixer: nn.Module, length: int, width: int) -> dict[str, Any] | None:
    """How much position *j* reaches position *i*, measured by impulse.

    This is the one picture every family can be drawn in without lying about
    any of them. A mixer is a map over positions along the swept axis, and the
    families differ in what *structure* that map is forced to have — not in
    whether they have one:

    - a dense linear map fills the square;
    - a convolution is banded, and the band repeats, because the same kernel
      slides along the axis;
    - an SSM is lower-triangular with stripes that fade, because position t
      reaches later positions only through a state that decays;
    - attention fills the square too, but its entries come from the data
      rather than from parameters, so the picture changes with the input.

    Measured rather than derived: feed a unit impulse at each position, in one
    batch, and read the responses. For a linear mixer that *is* the operator.
    For a nonlinear one it is the response about a zero input — a linearization
    at one point, which is why the payload calls it an impulse response and not
    a weight matrix.
    """
    # The width comes from the composition, which knows it, rather than from a
    # parameter shape, which does not: MambaMixer's first parameter is `A_log`
    # of shape (d_inner, d_state), so guessing from it fed the mixer an
    # 8-wide impulse where it wanted 32 — and the probe then failed into the
    # except below and drew nothing at all, silently.
    device = next(mixer.parameters(), torch.zeros(1)).device
    if not width or width < 1:
        return None
    try:
        x = torch.zeros(length, length, int(width), device=device)
        for j in range(length):
            x[j, j, :] = 1.0
        was_training = mixer.training
        mixer.eval()
        y = mixer(x)
        base = mixer(torch.zeros_like(x))
        if was_training:
            mixer.train()
        if not isinstance(y, torch.Tensor) or y.shape[:2] != (length, length):
            return None
        # Subtract the zero-input response so a bias does not read as coupling.
        resp = (y - base).to(torch.float32).mean(dim=-1).cpu()  # (impulse j, position i)
        grid = resp.transpose(0, 1)  # [i][j]
    except Exception:  # noqa: BLE001 - a mixer that will not take this shape simply has no picture
        return None

    absmax = float(grid.abs().max())
    mag = grid.abs()
    total = float(mag.sum()) or 1.0

    # The three properties that actually distinguish the families, measured off
    # the operator rather than assumed from the class name.
    causal = 1.0 - float(mag.triu(1).sum()) / total  # mass at or below the diagonal
    band = length - 1
    for b in range(length):
        keep = sum(float(mag.diagonal(d).sum()) for d in range(-b, b + 1))
        if keep >= 0.95 * total:
            band = b
            break
    # Weight tying: a convolution repeats the same kernel down every diagonal,
    # so each diagonal is constant. Attention and a dense map do not.
    spreads, weights_ = [], []
    for d in range(-(length - 1), length):
        diag = grid.diagonal(d)
        if diag.numel() < 2:
            continue
        w = float(diag.abs().sum())
        if w <= 0:
            continue
        spreads.append(float(diag.std(unbiased=False)) / (absmax or 1.0))
        weights_.append(w)
    if weights_:
        spread = sum(s * w for s, w in zip(spreads, weights_, strict=True)) / sum(weights_)
        tied = 1.0 - spread
    else:
        tied = 0.0

    # How much influence travels off the diagonal at all, relative to the local
    # term. An SSM's tail is real but small — a couple of percent — so a
    # threshold-based bandwidth alone reports "diagonal" and hides the very
    # thing that makes it a state-space model rather than a pointwise map.
    diag_mass = float(mag.diagonal().sum()) or 1.0
    reach = (total - diag_mass) / diag_mass

    return {
        "size": length,
        "absmax": round(absmax, 6),
        "causal": round(causal, 4),
        "bandwidth": band,
        "reach": round(reach, 5),
        "tied": round(max(0.0, min(1.0, tied)), 4),
        "values": [[round(float(v), 5) for v in row] for row in grid],
    }


def weights(
    model: nn.Module, *, max_units: int = MAX_UNITS, operator_size: int = 16
) -> dict[str, Any]:
    """A JSON-able digest of every mixer's parameters.

    Args:
        model: a composed model (one with an ``nd`` composition).
        max_units: the largest number of rows or columns drawn per tensor;
            anything larger is strided down to it and flagged ``sampled``.
        operator_size: positions to probe for each layer's impulse response —
            the position-to-position picture that puts every family on the same
            axes. Zero skips it.

    Returns:
        ``{"format", "version", "layers": [{"layer", "mixer", "tensors": [...]}]}``.
        Each tensor carries its role, a drawable 2-D sample, and full stats.
    """
    nd = getattr(model, "nd", None)
    d_model = getattr(nd, "d_model", None) if nd is not None else None
    if nd is None:
        raise TypeError(
            f"{type(model).__name__} has no `nd` composition; weights() describes "
            "the mixers of a composed model"
        )

    mixers = getattr(nd, "mixers", None)
    layers: list[dict[str, Any]] = []
    if mixers is not None:
        for i, mixer in enumerate(mixers):
            if mixer is None:
                continue
            layers.append(
                {
                    "layer": i,
                    "mixer": type(mixer).__name__,
                    "tensors": _tensors(mixer, max_units),
                    "operator": (
                        _operator(mixer, operator_size, int(d_model or 0))
                        if operator_size
                        else None
                    ),
                }
            )

    # The kernel family's per-axis kernels are parameters too, and they are the
    # whole mechanism there — a model with no `mixers` would otherwise report
    # nothing at all.
    kernels = getattr(nd, "kernels", None)
    if kernels is not None:
        for i, kernel in enumerate(kernels):
            if kernel is None:
                continue
            layers.append(
                {
                    "layer": len(layers) if mixers is None else i,
                    "mixer": type(kernel).__name__,
                    "tensors": _tensors(kernel, max_units),
                    "operator": None,
                }
            )

    return {
        "format": WEIGHTS_FORMAT,
        "version": WEIGHTS_VERSION,
        "max_units": max_units,
        "d_model": d_model,
        "layers": layers,
    }
