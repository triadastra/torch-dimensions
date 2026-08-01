"""The Vision Transformer, and why it is an N-D model at all.

ViT is usually described as "images as sequences of patches", which hides the
lattice: the patches *are* a grid, and every design choice that distinguishes
ViT variants is a choice about how that grid is handled.

    td.ViT(d_model=192, n_layers=6, image=(32, 32), patch=4)     # joint, as published
    td.ViT(..., method=td.axial_scan)                            # axial ViT

The first is the published model: patchify, flatten the grid to one sequence,
attend over all of it. The second is the axial variant — one argument apart,
which is the entire point of the library. On an 8×8 patch grid the joint form
attends over 64 tokens; the axial form does two passes of 8. Same patches,
same parameter count, different method of multidimensionality, and
BENCHMARKS.md says which is cheaper where.

The kernel family (``td.cafa``, ``td.axial_attention``) is deliberately *not*
available here, and the refusal is informative rather than a gap: those
methods own every spatial axis themselves and leave the mixer to run along
time, so on a time-less patch grid the transformer blocks would be dead
weight. A factorized-attention model over a patch grid is
``td.AxialKernel(mixer=None, ...)`` — the kernels are the model. Give the
lattice a time axis (video) and the hybrid form applies again.

**What this ships and what it does not.** Patch embedding, positional
embedding, and the transformer stack over the patch lattice, returning
per-patch features ``(B, *grid, d_model)``. No class token, no pooling, no
classification head — the same boundary every other model in the library
keeps. A head is three lines of caller code and it is the caller's three
lines. Concretely::

    vit = td.ViT(192, 6, image=(32, 32), patch=4, in_channels=3)
    head = nn.Linear(192, 10)
    logits = head(vit(images).mean(dim=(1, 2)))    # mean-pool the grid

**Positional embedding is where the lattice shows.** ViT learns one embedding
per patch position — a table the size of the grid, which cannot transfer to a
different image size. The factorized alternative learns one table per *axis*
and adds them, which is ``r·A`` parameters instead of ``A^r`` and extends to a
new grid size by interpolating one axis at a time. Both are here; factorized
is the default because on a lattice it is the natural one, and the published
choice is one argument away.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from torch_dimensions.compose import flatten
from torch_dimensions.lattice import Lattice
from torch_dimensions.mixers.attention import AttentionMixer
from torch_dimensions.models.base import LatticeModel

__all__ = ["PatchEmbed", "ViT"]


class PatchEmbed(nn.Module):
    """Cut an image into patches and embed each one — image to lattice.

    ``(B, *image, C)`` in, ``(B, *grid, d_model)`` out, where
    ``grid[i] = image[i] // patch[i]``. Rank-generic: a 2-D image, a 3-D
    volume, and a 4-D spatio-temporal block all work, because the patching is
    a reshape and a linear map rather than a ``Conv2d`` with a rank baked in.

    Args:
        image: size of each input axis.
        patch: patch size per axis; an int applies to every axis. Must divide
            the image exactly — a partial patch at the edge is a silent crop,
            and cropping the user's data without saying so is not this
            module's decision to make.
        in_channels: channels of the input (3 for RGB, 1 for greyscale).
        d_model: embedding width per patch.
    """

    def __init__(
        self,
        image: Sequence[int],
        patch: Sequence[int] | int,
        in_channels: int,
        d_model: int,
    ) -> None:
        super().__init__()
        self.image = tuple(int(s) for s in image)
        rank = len(self.image)
        self.patch = (
            (int(patch),) * rank if isinstance(patch, int) else tuple(int(p) for p in patch)
        )
        if len(self.patch) != rank:
            raise ValueError(f"patch {self.patch} has {len(self.patch)} axes, image has {rank}")
        bad = [(s, p) for s, p in zip(self.image, self.patch, strict=True) if p < 1 or s % p]
        if bad:
            raise ValueError(
                f"patch size must divide the image exactly; got image={self.image}, "
                f"patch={self.patch}. A partial patch at the edge would silently crop the "
                "input — pad or resize before this point, deliberately."
            )
        self.grid = tuple(s // p for s, p in zip(self.image, self.patch, strict=True))
        self.in_channels = in_channels
        self.d_model = d_model
        self.n_patch_features = in_channels
        for p in self.patch:
            self.n_patch_features *= p
        self.proj = nn.Linear(self.n_patch_features, d_model)

    def lattice(self, *, names: Sequence[str] | None = None, time: bool = False) -> Lattice:
        """The patch grid as a lattice — what the model actually operates on."""
        return Lattice(shape=self.grid, names=tuple(names) if names else None, time=time)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rank = len(self.image)
        if x.ndim != rank + 2:
            raise ValueError(
                f"expected a {rank + 2}-D tensor (B, *{self.image}, {self.in_channels}); "
                f"got shape {tuple(x.shape)}"
            )
        if tuple(x.shape[1:-1]) != self.image:
            raise ValueError(f"expected image dims {self.image}, got {tuple(x.shape[1:-1])}")
        if x.shape[-1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} channels, got {x.shape[-1]}")

        b = x.shape[0]
        # Split each image axis into (grid, patch), then move every patch axis
        # next to the channel axis and flatten them together. Written as one
        # rank-generic reshape+permute rather than einops or a per-rank table,
        # for the same reason the fold is: a rank-4 case must not need new code.
        split: list[int] = [b]
        for g, p in zip(self.grid, self.patch, strict=True):
            split += [g, p]
        split.append(self.in_channels)
        h = x.reshape(*split)

        grid_dims = [1 + 2 * i for i in range(rank)]
        patch_dims = [2 + 2 * i for i in range(rank)]
        h = h.permute(0, *grid_dims, *patch_dims, h.ndim - 1).contiguous()
        h = h.reshape(b, *self.grid, self.n_patch_features)
        return self.proj(h)

    def extra_repr(self) -> str:
        return (
            f"image={self.image}, patch={self.patch}, grid={self.grid}, "
            f"in_channels={self.in_channels}, d_model={self.d_model}"
        )


class _PosEmbed(nn.Module):
    """Learned positional embedding over a patch grid.

    ``factorized`` learns one table per axis and adds them (``r·A``
    parameters); ``full`` learns one per cell (``A^r``), which is what ViT
    publishes. Factorized is the default: on a lattice it is the natural
    parameterization, it is what makes a 3-D or 4-D grid affordable, and the
    axial models in this library already assume per-axis structure everywhere
    else.
    """

    def __init__(self, grid: tuple[int, ...], d_model: int, kind: str) -> None:
        super().__init__()
        if kind not in ("factorized", "full", "none"):
            raise ValueError(f"pos_embed must be factorized|full|none; got {kind!r}")
        self.kind = kind
        self.grid = grid
        if kind == "factorized":
            self.tables = nn.ParameterList(nn.Parameter(torch.zeros(n, d_model)) for n in grid)
        elif kind == "full":
            self.tables = nn.ParameterList([nn.Parameter(torch.zeros(*grid, d_model))])
        else:
            self.tables = nn.ParameterList()
        for t in self.tables:
            nn.init.trunc_normal_(t, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "none":
            return x
        if self.kind == "full":
            return x + self.tables[0]
        rank = len(self.grid)
        for axis, table in enumerate(self.tables):
            # Broadcast the axis's table along every other grid axis.
            shape = [1] * rank
            shape[axis] = self.grid[axis]
            x = x + table.reshape(*shape, -1)
        return x

    def extra_repr(self) -> str:
        n = sum(p.numel() for p in self.tables)
        return f"{self.kind}, grid={self.grid}, {n} parameters"


class ViT(LatticeModel):
    """Vision Transformer over a patch lattice, at any rank.

    Args:
        d_model: embedding width.
        n_layers: transformer blocks.
        image: input size per axis — ``(32, 32)`` for CIFAR, ``(16, 64, 64)``
            for a volume.
        patch: patch size per axis, or one int for all.
        in_channels: input channels.
        pos_embed: ``"factorized"`` (default), ``"full"`` (ViT's own), or
            ``"none"``.
        method: the method of multidimensionality. Defaults to
            :func:`~torch_dimensions.flatten` — attention over all patches at
            once, which is what ViT does. ``td.axial_scan`` gives the axial
            variant. The kernel family needs a time axis; see the module
            docstring.
        mixer_kwargs: forwarded to
            :class:`~torch_dimensions.mixers.attention.AttentionMixer`, e.g.
            ``{"n_heads": 12}``.

    Returns per-patch features ``(B, *grid, d_model)``. Pool and classify
    yourself; see the module docstring.
    """

    _mixer = AttentionMixer

    def __init__(
        self,
        d_model: int,
        n_layers: int = 1,
        *,
        image: Sequence[int],
        patch: Sequence[int] | int = 16,
        in_channels: int = 3,
        pos_embed: str = "factorized",
        names: Sequence[str] | None = None,
        n_heads: int = 4,
        **kw,
    ) -> None:
        # The lattice is derived, so accepting one would be accepting a second
        # answer to a question already answered. A checkpoint records
        # `lattice: None` for exactly this reason, and rebuilding hands it back.
        if kw.pop("lattice", None) is not None:
            raise ValueError(
                "ViT builds its lattice from `image` and `patch`; passing `lattice` too "
                "would let the two disagree. Pass image/patch, or use td.Transformer "
                "directly if you already have a lattice of tokens."
            )
        embed = PatchEmbed(image, patch, in_channels, d_model)
        lat = embed.lattice(names=names)
        # Only default when the caller named neither spelling: setting
        # `nd_method` unconditionally would collide with a caller's `method=`
        # and the base class refuses both, as it should.
        if "method" not in kw and "nd_method" not in kw:
            kw["nd_method"] = flatten
        mixer_kwargs = {"n_heads": n_heads, **kw.pop("mixer_kwargs", {})}
        super().__init__(d_model, n_layers, lat, mixer_kwargs=mixer_kwargs, **kw)
        # Registered after super().__init__ so the base class's parameter
        # accounting and the recorded config are unaffected by their presence.
        self.patch_embed = embed
        self.pos = _PosEmbed(embed.grid, d_model, pos_embed)
        self.config.update(
            {
                "image": list(embed.image),
                "patch": list(embed.patch),
                "in_channels": in_channels,
                "pos_embed": pos_embed,
                "n_heads": n_heads,
                # The lattice is derived from image/patch, so recording it too
                # would let a checkpoint hold two answers to one question.
                "lattice": None,
            }
        )

    @property
    def grid(self) -> tuple[int, ...]:
        """Patch-grid shape — the lattice the transformer runs over."""
        return self.patch_embed.grid

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, *image, C)`` in, ``(B, *grid, d_model)`` out."""
        return self.nd(self.pos(self.patch_embed(x)))
