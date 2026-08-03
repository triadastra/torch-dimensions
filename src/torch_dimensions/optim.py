"""Optimizer parameter groups that honour what the models ask for.

State-space parameters are not ordinary weights. ``A`` controls how fast a
state decays and ``dt`` its timescale; both sit inside an exponential, so a
weight-decay term pulling them toward zero is not a mild regulariser but a
change to the dynamics, and a learning rate suited to a projection matrix will
walk them straight out of the stable region. Every upstream implementation
knows this and says so **in the parameters themselves**:

- ``state-spaces/s4`` attaches ``param._optim = {"lr": ..., "weight_decay": 0.0}``
  to the SSM kernel parameters (``kernel.py``'s ``register``);
- ``state-spaces/mamba`` attaches ``param._no_weight_decay = True`` to
  ``A_log``, ``D`` and ``dt_bias``.

Those tags travel with the vendored code but do nothing on their own — an
optimizer has to read them. Anyone who writes the obvious thing::

    torch.optim.AdamW(model.parameters(), lr=1e-3)

is training an S4 or a Mamba in the way its authors explicitly avoid, and
nothing complains. This module reads the tags::

    opt = torch.optim.AdamW(td.param_groups(model, lr=1e-3), lr=1e-3)

The published recipes agree on the shape of the rest: AdamW with
``betas=(0.9, 0.95)``, weight decay ~0.1 on ordinary weights, gradient
clipping at 1.0, and a linear warmup into a cosine decay. Norms and biases are
conventionally excluded from weight decay too, which :func:`param_groups`
does — a one-dimensional parameter has no direction for decay to shrink
meaningfully.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

__all__ = ["param_groups", "warmup_cosine"]

# Upstream's own ceiling for SSM parameters. s4 fixes it at 1e-3 in its
# configs; Mamba-family papers cap the SSM learning rate at the same place.
SSM_MAX_LR = 1e-3


def param_groups(
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float = 0.1,
    ssm_lr: float | None = None,
    decay_1d: bool = False,
) -> list[dict[str, Any]]:
    """Split a model's parameters into groups an optimizer can take directly.

    Args:
        model: any module; tags are read per parameter, so this works on the
            vendored upstream blocks and on our portable ones alike.
        lr: the learning rate for ordinary weights.
        weight_decay: decay for ordinary weights. Never applied to a parameter
            that asks not to have it.
        ssm_lr: learning rate for tagged state-space parameters. Defaults to
            ``min(lr, 1e-3)`` — upstream's ceiling, and it lowers a too-high
            ``lr`` rather than raising a deliberately low one.
        decay_1d: apply weight decay to one-dimensional parameters (norms,
            biases). Off by default, as is conventional.

    Returns:
        A list of parameter-group dicts. Groups with no members are dropped, so
        the result can be handed to any optimizer without special-casing.

    The tags come from the upstream authors and are respected exactly: a
    parameter carrying ``_optim`` gets those settings, one carrying
    ``_no_weight_decay`` gets ``weight_decay=0``, and anything else is an
    ordinary weight.
    """
    if ssm_lr is None:
        ssm_lr = min(lr, SSM_MAX_LR)

    ordinary: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    ssm: list[nn.Parameter] = []
    # Parameters carrying an explicit `_optim` dict are grouped by that dict,
    # since two of them may ask for different things.
    special: dict[tuple, list[nn.Parameter]] = {}

    for param in model.parameters():
        if not param.requires_grad:
            continue
        explicit = getattr(param, "_optim", None)
        if explicit:
            key = tuple(sorted(explicit.items()))
            special.setdefault(key, []).append(param)
        elif getattr(param, "_no_weight_decay", False):
            ssm.append(param)
        elif param.ndim <= 1 and not decay_1d:
            no_decay.append(param)
        else:
            ordinary.append(param)

    groups: list[dict[str, Any]] = []
    if ordinary:
        groups.append({"params": ordinary, "lr": lr, "weight_decay": weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
    if ssm:
        # Tagged `_no_weight_decay` by upstream: no decay, and the SSM rate.
        groups.append({"params": ssm, "lr": ssm_lr, "weight_decay": 0.0})
    for key, params in special.items():
        settings = dict(key)
        groups.append(
            {
                "params": params,
                "lr": settings.get("lr", ssm_lr),
                "weight_decay": settings.get("weight_decay", 0.0),
            }
        )
    return groups


def warmup_cosine(
    optimizer: torch.optim.Optimizer, *, warmup: int, total: int, floor: float = 0.0
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup into cosine decay — the schedule the papers use.

    Args:
        warmup: steps spent ramping linearly from zero to the group's own
            learning rate.
        total: total training steps; the cosine completes over what is left.
        floor: fraction of the peak rate to end at, rather than zero.

    Scaling is multiplicative, so each parameter group keeps its own rate —
    the SSM group stays below the others throughout instead of being flattened
    to one schedule, which is the point of having separated them.
    """
    if warmup < 0 or total <= 0:
        raise ValueError(f"need total > 0 and warmup >= 0; got {total=}, {warmup=}")

    def scale(step: int) -> float:
        if step < warmup:
            # +1 so the first step is not exactly zero, which would waste it.
            return (step + 1) / max(warmup, 1)
        if total <= warmup:
            return 1.0
        progress = (step - warmup) / (total - warmup)
        import math

        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return floor + (1.0 - floor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
