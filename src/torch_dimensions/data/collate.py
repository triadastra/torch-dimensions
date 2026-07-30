"""Stacking windows into batches.

The only thing this does that ``torch``'s default collate would not is keep the
lattice out of the batch. A lattice is static metadata; collating it would
allocate an identical copy per sample per step for no benefit, and worse, would
invite treating it as per-sample data when it is a property of the whole
dataset.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from torch_dimensions.data.source import Sample

__all__ = ["Batch", "collate_lattice"]


class Batch(dict):
    """A stacked batch. ``batch.x`` is ``(B, T, *shape, F)``."""

    # See Sample: KeyError from attribute lookup breaks hasattr and pickling,
    # and worker processes pickle the collated Batch on its way back.
    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


def collate_lattice(samples: Sequence[Sample]) -> Batch:
    """Stack samples along a new leading batch dimension.

    Windows of differing length are rejected rather than padded: a ragged
    time axis in an N-D batch is far more likely to be a windowing bug than an
    intention, and padding it silently would hide that.
    """
    if not samples:
        raise ValueError("cannot collate an empty list of samples")

    lengths = {s["x"].shape[0] for s in samples}
    if len(lengths) != 1:
        raise ValueError(f"samples have differing input lengths {sorted(lengths)}")

    # Presence of a target is checked across *all* samples, not just the
    # first: keying off samples[0] silently dropped every target whenever the
    # first sample happened to be a horizon-0 one — a batch that trains on
    # nothing and says nothing.
    with_y = sum("y" in s for s in samples)
    if 0 < with_y < len(samples):
        raise ValueError(
            f"{with_y} of {len(samples)} samples have a target and the rest do not; "
            "mixed-horizon samples cannot share a batch"
        )

    batch = Batch(x=torch.stack([s["x"] for s in samples]))
    if with_y:
        y_lengths = {s["y"].shape[0] for s in samples}
        if len(y_lengths) != 1:
            raise ValueError(f"samples have differing target lengths {sorted(y_lengths)}")
        batch["y"] = torch.stack([s["y"] for s in samples])
    batch["windows"] = tuple(s["window"] for s in samples)
    return batch
