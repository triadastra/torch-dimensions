"""Convolutional mixers — the local operators, causal and not.

Two things live here that the rest of the library does not have, and both are
deliberate:

**A convolution is not a sequence model.** An RNN and a selective scan carry
state along the axis; a convolution has no state and no notion of "so far". It
looks at a fixed window and it is finished. That makes ``ConvMixer`` the
library's first genuinely non-sequential mixer, and it is what makes ``td.CNN``
— an axially-separable convolutional network — expressible in the same
abstraction as ``td.Mamba`` without either one bending.

**A convolution is LTI.** Linear (with ``activation=None``) and
shift-equivariant, which is exactly the property that decides how a mixer
behaves under N-D composition: per-axis LTI operators *commute*, so a stack of
them is one separable N-D convolution and the sweep order does not matter. The
selective and recurrent mixers have no such property, which is why their
schedules are an architectural choice and a CNN's is not. That contrast is
measured, not asserted — see :func:`torch_dimensions.testing.check_lti` and
LTI.md.

Dilation grows with **how many times this mixer's axis has already been
swept**, not with the layer index. On a rank-3 lattice under a cyclic plan,
layer 6 is the third sweep of axis 0 and the first sweep of nothing else, so it
gets dilation 4 along that axis rather than 64. Receptive field then grows
exponentially *per axis*, which is what a TCN does in 1-D and what nothing in
the N-D literature bothers to state.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ConvMixer", "TCNMixer"]

_ACTIVATIONS = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU, "tanh": nn.Tanh}


class ConvMixer(nn.Module):
    """1-D convolution along the swept axis.

    Args:
        d_model: feature width, unchanged in and out.
        kernel_size: window width. Must be odd when ``causal=False`` — an even
            window has no centre, so "same" padding would have to guess which
            side to favour, and a silent half-step shift along one axis is
            exactly the class of bug this library exists to prevent.
        causal: pad on the left only, so position ``t`` sees no position after
            it. Off by default: a mixer is not told which axis it is sweeping,
            and "the future" of a spatial axis is not a meaningful idea.
        dilation: spacing between taps, before any per-sweep growth.
        dilation_base: multiply dilation by this each time the swept axis is
            swept again. ``1`` (the default) means fixed dilation; ``2`` is the
            TCN's doubling. Combined with ``sweep``.
        sweep: how many earlier layers already swept this axis.
            :class:`~torch_dimensions.AxialScan` supplies it — a factory that
            accepts the argument gets it, and one that does not is built
            exactly as before.
        n_conv: convolutions stacked inside the block. The TCN's residual block
            uses two.
        depthwise: separate the spatial convolution (one filter per channel)
            from the channel mixing (a pointwise linear). This is the
            depthwise-separable convolution of MobileNet and ConvNeXt, and on a
            lattice it composes with the axial fold into a *doubly* separable
            operator: separable across channels and across axes.
        activation: pointwise nonlinearity after each convolution, or ``None``
            for a bare linear operator. ``None`` is what makes this mixer LTI
            in the strict sense; with an activation it stays shift-equivariant
            but stops being linear, and :func:`~torch_dimensions.testing.
            check_lti` will say so.

    The block carries no residual connection and no normalization of its own:
    :class:`~torch_dimensions.AxialScan` owns those, the same as for the RNN
    and SSM mixers. (:class:`~torch_dimensions.mixers.attention.AttentionMixer`
    is the deliberate exception, because a transformer block is *defined* as
    attention-plus-MLP with its own residuals.)
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 3,
        *,
        causal: bool = False,
        dilation: int = 1,
        dilation_base: int = 1,
        sweep: int = 0,
        n_conv: int = 1,
        depthwise: bool = False,
        activation: str | None = "gelu",
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError(f"kernel_size must be >= 1; got {kernel_size}")
        if not causal and kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size={kernel_size} is even and causal=False: a centred window needs an "
                "odd width, or the output is shifted half a step along this axis. Use an odd "
                "kernel_size, or causal=True where left-padding is the point."
            )
        if n_conv < 1:
            raise ValueError(f"n_conv must be >= 1; got {n_conv}")
        if dilation < 1 or dilation_base < 1:
            raise ValueError(
                f"dilation and dilation_base must be >= 1; got {dilation}, {dilation_base}"
            )
        if activation is not None and activation not in _ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}; known: {sorted(_ACTIVATIONS)}")

        self.d_model = d_model
        self.kernel_size = kernel_size
        self.causal = causal
        self.n_conv = n_conv
        self.depthwise = depthwise
        # The dilation this layer actually runs at, after per-sweep growth.
        # Recorded as an attribute because "what dilation did layer 7 get"
        # is a question the schedule makes genuinely non-obvious.
        self.dilation = dilation * dilation_base**sweep
        self.sweep = sweep

        convs: list[nn.Module] = []
        points: list[nn.Module] = []
        for _ in range(n_conv):
            convs.append(
                nn.Conv1d(
                    d_model,
                    d_model,
                    kernel_size,
                    dilation=self.dilation,
                    groups=d_model if depthwise else 1,
                    bias=bias,
                )
            )
            points.append(nn.Conv1d(d_model, d_model, 1, bias=bias) if depthwise else nn.Identity())
        self.convs = nn.ModuleList(convs)
        self.points = nn.ModuleList(points)
        self.act = _ACTIVATIONS[activation]() if activation is not None else nn.Identity()
        self.drop = nn.Dropout(dropout)

    @property
    def receptive_field(self) -> int:
        """How many positions along the axis one output can depend on."""
        return 1 + self.n_conv * (self.kernel_size - 1) * self.dilation

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        """Pad so the length survives. Left-only when causal."""
        total = (self.kernel_size - 1) * self.dilation
        if total == 0:
            return x
        return F.pad(x, (total, 0) if self.causal else (total // 2, total - total // 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (M, A, H)
        if x.shape[-1] != self.d_model:
            raise ValueError(f"expected {self.d_model} features, got {x.shape[-1]}")
        h = x.transpose(1, 2)  # (M, H, A) — torch convolutions are channel-second
        for conv, point in zip(self.convs, self.points, strict=True):
            h = point(conv(self._pad(h)))
            h = self.drop(self.act(h))
        return h.transpose(1, 2)

    def extra_repr(self) -> str:
        kind = "causal" if self.causal else "centred"
        return (
            f"d_model={self.d_model}, kernel_size={self.kernel_size}, {kind}, "
            f"dilation={self.dilation}, receptive_field={self.receptive_field}"
        )


class TCNMixer(ConvMixer):
    """The temporal convolutional network's block: causal, dilated, doubling.

    A :class:`ConvMixer` with the TCN defaults from Bai, Kolter & Koltun
    (2018) — two causal convolutions per block, ReLU, and dilation that doubles
    each time this axis comes round again. Everything is inherited; only the
    defaults differ, which is the honest way to say "this is that model" when
    it genuinely is.

    Causality here is real and bitwise: position ``t`` is padded only on the
    left, so no arithmetic involving position ``t+1`` reaches it. The test
    suite holds it to that, not to a tolerance.
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 3,
        *,
        causal: bool = True,
        dilation_base: int = 2,
        n_conv: int = 2,
        activation: str | None = "relu",
        sweep: int = 0,
        **kw,
    ) -> None:
        # `sweep` is named here rather than left to `**kw` on purpose: the
        # scan layer passes it only to factories whose signature spells it
        # out, so a subclass that swallows it into `**kw` would silently get
        # dilation 1 on every layer and be a TCN in name only. That is not
        # hypothetical — it is what this class did until the first test of the
        # dilation schedule was written.
        super().__init__(
            d_model,
            kernel_size,
            causal=causal,
            dilation_base=dilation_base,
            n_conv=n_conv,
            activation=activation,
            sweep=sweep,
            **kw,
        )


def axis_receptive_field(model: nn.Module) -> dict[str, dict[str, object]]:
    """Per-axis receptive field of a scan-family model, against the axis size.

    The question a convolutional N-D model raises and no N-D paper answers:
    *can this model see across the lattice at all?* A 4-layer TCN over a 32×32
    image with kernel 3 covers 22 of 32 rows — the top-left pixel and the
    bottom-right one are, structurally, in different models. Attention, RNNs
    and SSMs span their whole axis in one layer, so for them this reports
    ``inf`` and the answer is always yes; for convolutions it is a real
    constraint that should be checked before training rather than diagnosed
    after.

    Layers compose the usual way: stacking two windows of width ``w`` sees
    ``2w − 1``, so spans add as ``1 + Σ(rf_i − 1)`` over the layers sweeping
    that axis.

    Returns one entry per swept axis::

        {"row": {"span": 22, "size": 32, "covers": False, "layers": 4}, ...}

    ``size`` is ``None`` for the time axis, whose length is dynamic; ``covers``
    is then ``None`` too, because whether 22 steps is enough depends on the
    sequence you feed it.
    """
    from torch_dimensions.compose.scan import AxialScan

    nd = getattr(model, "nd", model)
    # An explicit family check, not duck typing. AxialKernel exposes `plan`,
    # `mixers` and `lattice` too, but its mixers run along *time* while the
    # plan names spatial axes — pairing them up would attribute a time mixer's
    # window to a spatial axis and report a confident wrong number.
    if not isinstance(nd, AxialScan):
        raise TypeError(
            f"receptive field along an axis is a scan-family idea, and this model composes "
            f"with {type(nd).__name__}. The kernel family mixes across a whole axis in one "
            "layer by construction, so its span is the entire lattice; there is nothing to "
            "accumulate."
        )
    mixers, plan, lat = nd.mixers, nd.plan, nd.lattice

    spans: dict[str, float] = {}
    layers: dict[str, int] = {}
    for step, mixer in zip(plan, mixers, strict=True):
        name = lat.axis_names[int(step.axis)]  # resolved by AxialScan at construction
        rf = getattr(mixer, "receptive_field", math.inf)
        spans[name] = spans.get(name, 1.0) + (float(rf) - 1.0)
        layers[name] = layers.get(name, 0) + 1

    out: dict[str, dict[str, object]] = {}
    for name, span in spans.items():
        size = None if (lat.time and name == "time") else lat.axis_size(name)
        out[name] = {
            "span": math.inf if math.isinf(span) else int(span),
            "size": size,
            "covers": None if size is None else (math.isinf(span) or span >= size),
            "layers": layers[name],
        }
    return out
