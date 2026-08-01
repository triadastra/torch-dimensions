"""A complete new mixer, end to end — the worked example for docs/adding-a-mixer.md.

    python examples/custom_mixer.py

An exponentially-weighted moving average with a learned, per-channel decay:
small enough to read in one sitting, real enough to be a genuine sequence
model (it is the diagonal SSM with the state dimension set to one), and it has
the two properties that make an extension interesting — a parameter that must
receive gradient, and an output that depends on order, so an axis bug cannot
hide.

What this file demonstrates, in order:

1. the mixer itself — ``(M, A, H) -> (M, A, H)``, nothing else;
2. running the library's own conformance suite against it;
3. registering it as a model kind so configs and checkpoints can name it;
4. using it at rank 3, sparse, with a schedule.

`tests/test_examples.py` runs all of it, so this file cannot rot into a
plausible-looking snippet that no longer works.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import torch_dimensions as td
from torch_dimensions.models.base import LatticeModel


class EMAMixer(nn.Module):
    """Per-channel exponential moving average: ``y_t = a * y_{t-1} + (1-a) * x_t``.

    The entire mixer contract is the shape: ``(M, A, H)`` in, the same out,
    where ``M`` is the batch times every unswept axis and ``A`` is the swept
    axis. A mixer is never told which axis it is on, what rank the lattice is,
    or which cells are absent — the composition layer owns all of that, which
    is exactly why one implementation works at every rank.

    Direction is not a mixer's business either: a backward sweep arrives
    already flipped. So this is written causally and gets bidirectionality from
    the schedule, for free.
    """

    def __init__(self, d_model: int, init_halflife: float = 4.0) -> None:
        super().__init__()
        # Parameterize the *logit* of the decay so that `a` stays in (0, 1)
        # under unconstrained gradient descent. Clamping instead would give
        # zero gradient exactly where the model most wants to move.
        a0 = 0.5 ** (1.0 / init_halflife)
        self.decay_logit = nn.Parameter(
            torch.full((d_model,), float(torch.logit(torch.tensor(a0))))
        )
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.sigmoid(self.decay_logit)
        # The honest sequential form. A cumulative-product trick is faster and
        # is what a real mixer would use; this stays a loop because the point
        # of the example is the contract, not the kernel.
        out = []
        state = torch.zeros_like(x[:, 0])
        for t in range(x.shape[1]):
            state = a * state + (1 - a) * x[:, t]
            out.append(state)
        return self.out(torch.stack(out, dim=1))


class EMA(LatticeModel):
    """The model: the library's composition layer plus the mixer above.

    Subclassing `LatticeModel` is what supplies `nd_method=`, `plan=`,
    `lattice=`, `d_input=`, `.config`, `.save()`, and `.to_spec()`. A mixer
    author writes the class body and inherits the rest.
    """

    _mixer = EMAMixer


def run_conformance() -> None:
    """The same seven checks the library runs on itself.

    This is the point of `td.testing` being public API rather than test
    scaffolding: a new mixer gets held to the identical standard, including
    the checks that catch axis bugs — rank-1 equivalence against the bare
    mixer, absent-cell inertia, and covariance with axis storage order.
    """

    def factory(lattice, d_model, plan=None):
        # One layer per axis, so that at rank 1 the model *is* a single layer —
        # which is what makes the equivalence check below a real comparison
        # rather than a comparison against a different model.
        return EMA(d_model, len(lattice.axis_names), lattice, plan=plan)

    def reference(block, x):
        """What one pre-norm residual layer around the bare mixer computes.

        Supplying this turns the rank-1 equivalence check from a skip into a
        real comparison: on a lattice with one axis, the whole N-D apparatus
        must reduce to exactly the 1-D model, bitwise.
        """
        return x + block.nd.mixers[0](block.nd.norms[0](x))

    report = td.testing.check_block(factory, reference=reference)
    print(report)
    assert report, "the example mixer does not conform"

    trained = td.testing.check_trainable(factory, d_model=16, steps=120)
    print(trained)
    assert trained, "the example mixer does not learn the axial task"


def use_it() -> None:
    """Register it, then use it exactly like a built-in model."""
    if "ema" not in td.list_models():
        td.register_model("ema", EMA)

    lattice = td.Lattice(
        shape=(4, 5, 3),
        names=("depth", "row", "col"),
        valid=torch.rand(4, 5, 3) > 0.3,
        time=True,
    )
    plan = td.ScanPlan.paired(lattice.axis_names, n_layers=8, bidirectional=("depth", "row", "col"))
    model = EMA(d_model=32, lattice=lattice, plan=plan, d_input=2)

    x = torch.randn(2, 6, *lattice.shape, 2)
    print("output:", tuple(model(x).shape))
    print("coverage:\n", plan.coverage(lattice))

    # Because it is registered, it round-trips through config and checkpoints
    # with no extra code.
    rebuilt = td.build({"kind": "ema", **model.config})
    print("rebuilt from config:", type(rebuilt).__name__, rebuilt.config["n_layers"], "layers")


if __name__ == "__main__":
    run_conformance()
    use_it()
