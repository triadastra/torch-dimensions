"""A complete new nd_method — the worked example for docs/adding-a-method.md.

    python examples/custom_method.py

Where a *mixer* decides what happens along one axis, an **nd_method** decides
how the axes are handled at all: which layer gets which axis, whether they are
swept one at a time or contracted together, whether some axes are handled by a
different operator entirely.

The strategy here — "pyramid" — sweeps the largest axis every layer and rotates
the smaller axes through the remaining slots, on the theory that the axis with
the most positions needs the most mixing. Whether that theory is any good is
not the point; the point is that it is thirty lines and needs no changes to
the library.

`tests/test_examples.py` runs this file.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import torch_dimensions as td


def pyramid(mixer, plan, lattice, d_model, **kwargs) -> nn.Module:
    """An nd_method: build the module that handles every axis.

    The signature is fixed by the contract::

        nd_method(mixer, plan, lattice, d_model, **kwargs) -> nn.Module

    ``mixer`` is a *factory* (call it to build one), not a module, because a
    strategy decides how many mixers exist and whether they are shared.

    This one rewrites the schedule and then delegates. Rewriting the plan is
    the cheapest useful thing a strategy can do, and it is possible at all only
    because a plan is data rather than control flow — the reason `ScanPlan`
    is a value object and not a pair of list comprehensions welded into a
    model.

    Two details here are not decoration; both are bugs the conformance suite
    found in the first version of this function, and they are the two mistakes
    a schedule-rewriting strategy is most likely to make.

    *Order axes by the plan, not by the lattice.* Taking the axis order from
    ``lattice.axis_names`` makes the schedule depend on the order the axes
    happen to be **stored** in, so the same model on the same data laid out
    differently computes something different. `check_block`'s covariance check
    is exactly this and it failed. The plan's own order is storage-independent.

    *A rank-1 lattice has no "other" axes.* The first version divided by
    ``len(others)``, which is zero when the only axis is the dominant one —
    caught by the shape check at rank 1, the cheapest check in the suite.
    """
    # Axis names in the order the *plan* mentions them; a plan may hold names
    # or indices, so normalize through the lattice either way.
    order: list[str] = []
    for step in plan:
        name = lattice.axis_names[lattice.axis_index(step.axis)]
        if name not in order:
            order.append(name)

    spatial = [a for a in order if a != "time"]
    if not spatial:
        return td.axial_scan(mixer, plan, lattice, d_model, **kwargs)

    biggest = max(spatial, key=lattice.axis_size)
    others = [a for a in order if a != biggest]

    steps = []
    for i in range(len(plan)):
        if i % 2 == 0 or not others:
            # The dominant axis, alternating direction each time it comes up.
            steps.append(td.Step(biggest, (i // 2) % 2 == 1))
        else:
            axis = others[(i // 2) % len(others)]
            # Time stays causal no matter what a schedule would prefer.
            steps.append(td.Step(axis, axis != "time" and (i // 2) % 2 == 1))
    return td.axial_scan(mixer, td.ScanPlan.from_list(steps), lattice, d_model, **kwargs)


def main() -> None:
    lattice = td.Lattice(shape=(3, 12), names=("row", "col"), time=True)

    # A strategy is a plain callable: pass it directly, no registration.
    model = td.LSTM(d_model=32, n_layers=8, lattice=lattice, method=pyramid, d_input=1)
    print("output:", tuple(model(torch.randn(2, 5, 3, 12, 1)).shape))

    cov = model.nd.plan.coverage(lattice)
    print(cov)
    assert cov["col"].n_sweeps > cov["row"].n_sweeps, "the strategy did not prioritize the big axis"
    assert cov["time"].backward == 0, "time was swept backwards; that is not causal"

    # The conformance suite takes a strategy exactly as it takes a mixer.
    def factory(lat, d_model, plan=None):
        return td.LSTM(d_model, len(lat.axis_names) + 1, lat, plan=plan, method=pyramid)

    report = td.testing.check_block(factory)
    print(report)
    assert report, "the custom strategy does not conform"

    # Registration is only needed so that a *config file* can name it — YAML
    # cannot hold a callable. In Python the function itself is enough.
    if "pyramid" not in td.ND_METHODS:
        td.register_nd_method("pyramid", pyramid)
    from_config = td.build(
        {
            "kind": "lstm",
            "d_model": 16,
            "n_layers": 6,
            "method": "pyramid",
            "lattice": {"shape": [3, 12], "names": ["row", "col"], "time": True},
        }
    )
    print("from config:", type(from_config).__name__, len(from_config.nd.plan), "layers")


if __name__ == "__main__":
    main()
