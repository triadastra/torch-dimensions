"""Convolutional models: the separable CNN, and the temporal CNN.

These are the library's first models that are not sequence models at all.
``td.CNN`` has no state, no direction, and no notion of "so far" — it is a
stack of local windows. That it fits the same abstraction as ``td.Mamba``
without either bending is the strongest evidence the abstraction is about
*lattices* rather than about sequences:

    td.CNN(64, 6, lattice)     # separable N-D convolution
    td.TCN(64, 6, lattice)     # causal, dilated, doubling per axis
    td.Mamba(64, 6, lattice)   # selective scan

**What "N-D" means for a CNN, exactly.** Sweeping a 1-D convolution along each
axis in turn is a *separable* convolution: with linear mixers it equals one
N-D convolution whose kernel is the outer product of the per-axis kernels. That
is not an approximation and not a claim — it is checked against ``F.conv2d``
and ``F.conv3d`` in ``tests/test_conv.py``, exactly as the kernel family's
factorization is checked against ``torch.kron``. The cost is what separability
always costs: a rank-1 kernel, ``r·k`` parameters instead of ``k^r``, so a
separable stack cannot represent a diagonal edge detector that a full kernel
can. Depth plus the pointwise channel mixing is what buys most of it back, and
that is the same bargain MobileNet and ConvNeXt make in 2-D.

**Why this one composes so cleanly and Mamba does not.** A convolution is LTI:
per-axis operators commute, so the sweep order is irrelevant and direction is
nearly free. A selective scan is not, so for it the order and direction are
architectural choices with real consequences — which is why ``ScanPlan``
exists. LTI.md measures both claims.
"""

from __future__ import annotations

from torch_dimensions.mixers.conv import ConvMixer, TCNMixer
from torch_dimensions.models.base import LatticeModel

__all__ = ["CNN", "TCN", "CNNND", "TCNND"]


class CNN(LatticeModel):
    """An axially-separable convolutional network over a sequence or lattice.

    Args:
        d_model: feature width, and the output width.
        n_layers: how many sweeps. With a lattice, layers cycle through its
            axes unless ``plan`` says otherwise — so a rank-2 lattice with
            ``n_layers=6`` gets three passes over each of the two axes.
        lattice: omit for an ordinary 1-D convolutional stack.
        kernel_size: window width per axis. Must be odd; see
            :class:`~torch_dimensions.mixers.conv.ConvMixer`.
        depthwise: depthwise-separable convolutions (one filter per channel
            plus a pointwise mix). Separable across channels *and* across
            axes — the cheap corner of the design space.
        activation: pointwise nonlinearity, or ``None`` for a strictly linear
            (and therefore strictly LTI) stack.
        dilation_base: grow the dilation each time an axis is swept again.
            ``1`` keeps it fixed; use :class:`TCN` for the doubling schedule.

    Direction is meaningless for a centred convolution — a backward sweep
    arrives flipped and the kernel is symmetric in its own frame, so
    ``bidirectional`` buys a mirrored filter and nothing else. That is not a
    limitation to work around; it is what "no notion of order" means, and
    LTI.md measures it as ``forward − reverse`` at the noise floor.
    """

    _mixer = ConvMixer

    def __init__(
        self,
        d_model: int,
        n_layers: int = 1,
        lattice=None,
        *,
        kernel_size: int = 3,
        depthwise: bool = False,
        activation: str | None = "gelu",
        dilation_base: int = 1,
        **kw,
    ):
        mixer_kwargs = {
            "kernel_size": kernel_size,
            "depthwise": depthwise,
            "activation": activation,
            "dilation_base": dilation_base,
            **kw.pop("mixer_kwargs", {}),
        }
        super().__init__(d_model, n_layers, lattice, mixer_kwargs=mixer_kwargs, **kw)


class TCN(LatticeModel):
    """Temporal convolutional network — causal, dilated, doubling per axis.

    The 1-D model of Bai, Kolter & Koltun (2018), and its N-D generalization
    for free: each layer runs a causal dilated convolution along one axis, and
    the dilation doubles every time that axis comes round again. On a rank-3
    lattice under a cyclic plan the dilation along each axis is
    ``1, 2, 4, …`` independently — the receptive field grows exponentially
    *per axis*, which is the thing the 1-D TCN is famous for and which nothing
    in the N-D literature states.

    Causality is bitwise, not approximate: padding is left-only, so no
    arithmetic involving a later position reaches an earlier one. The test
    suite holds it to equality.

    Check the model can actually see across the lattice before training it::

        td.receptive_field(td.TCN(64, 6, lattice))
        # {'h': {'span': 29, 'size': 32, 'covers': False, 'layers': 3}, ...}

    Args:
        kernel_size: window width; may be even, since causal padding has a
            defined side.
        dilation_base: ``2`` is the published schedule. ``1`` disables growth.
        n_conv: convolutions per block; ``2`` is the published block.
    """

    _mixer = TCNMixer

    def __init__(
        self,
        d_model: int,
        n_layers: int = 1,
        lattice=None,
        *,
        kernel_size: int = 3,
        dilation_base: int = 2,
        n_conv: int = 2,
        activation: str | None = "relu",
        **kw,
    ):
        mixer_kwargs = {
            "kernel_size": kernel_size,
            "dilation_base": dilation_base,
            "n_conv": n_conv,
            "activation": activation,
            **kw.pop("mixer_kwargs", {}),
        }
        super().__init__(d_model, n_layers, lattice, mixer_kwargs=mixer_kwargs, **kw)


CNNND = CNN
TCNND = TCN
"""Aliases. Unlike ``S4ND``/``MambaND`` these need no separate class with a
mandatory ``dim``: those names denote specific published models, so the library
refuses to let ``S4ND(dim=1)`` quietly be S4. "CNND" denotes nothing, and a
2-D CNN is not at risk of being mistaken for a 1-D one."""
