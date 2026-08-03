"""One set of starting weights, shared by every machine in a comparison.

Both benchmarks assumed that building on CPU under a fixed seed gives
bit-identical weights on any machine. For most models it does — an LSTM built
this way hashes identically on macOS and Linux. **For S4 and S4D it does not**,
and the reason is not a bug in anyone's code:

`hippo.nplr` diagonalises the HiPPO matrix with `torch.linalg.eigh`. The
eigen*values* are unique, and they match across platforms to every digit
printed — which is why `A_imag` agreed exactly. The eigen*vectors* are only
defined up to a phase, and macOS Accelerate and Linux LAPACK are free to
return different ones. `B` and `P` are projections through those vectors, so
they inherit the phase, and two machines end up with two different — both
perfectly valid — S4 initialisations. Measured on this zoo: `B` differed by a
relative 1.5, `P` by 0.53, while `A_imag` was identical.

A cross-device comparison built on that assumption is not measuring the
device. It was reporting a 2.6e-01 output difference for the vendored S4D,
unchanged in float64, which reads exactly like a different kernel and was in
fact a different model.

So the weights are written once, by whichever run goes first, and every later
run loads them. `torch.load` of the same file cannot be platform-dependent,
which the seed could not promise.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


def sync(model: nn.Module, store: Path | None, name: str) -> str:
    """Load this model's starting weights from `store`, or write them if absent.

    Returns a one-word description of what happened, for the manifest — a run
    must be able to say whether its numbers rest on shared weights or on the
    seed, since the two are not equivalent and the difference is invisible in
    the losses.
    """
    if store is None:
        return "seed"

    store.mkdir(parents=True, exist_ok=True)
    path = store / f"{name}.pt"
    if path.exists():
        # `assign=False` keeps the model's own dtypes and devices; the file is
        # the source of the values only.
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        return "loaded"

    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path)
    return "written"


def add_argument(ap) -> None:
    ap.add_argument(
        "--init",
        default=None,
        metavar="DIR",
        help="directory of shared starting weights; written if empty, loaded if not. "
        "Use the same directory on every machine in a comparison — the seed alone "
        "does not give identical S4/S4D weights across platforms.",
    )
