# Adding an nd_method

A [mixer](adding-a-mixer.md) decides what happens *along one axis*. An
**nd_method** decides how the axes are handled at all: which layer gets which
axis, whether axes are swept one at a time or contracted together, and whether
some axes belong to a different operator entirely.

This is the argument the library exists to make one flag:

```python
td.LSTM(64, 12, lattice, method=td.axial_scan)       # sweep every axis
td.LSTM(64, 12, lattice, method=td.cafa)             # kernels across space, RNN along time
td.LSTM(64, 12, lattice, method=my_strategy)         # yours, no registration
```

The worked example is [`examples/custom_method.py`](../examples/custom_method.py),
executed by `tests/test_examples.py`.

---

## The contract

```python
def nd_method(mixer, plan, lattice, d_model, **kwargs) -> nn.Module: ...
```

- `mixer` is a **factory**, not a module — call it to build one. A strategy
  decides how many mixers exist and whether they are shared, so it cannot be
  handed a single instance. It may be `None`: the kernel family accepts no
  mixer on a lattice with no time axis, because there would be nothing for it
  to sweep.
- `plan` is the requested schedule, as data. You may use it, ignore it, or
  rewrite it.
- `lattice` and `d_model` are what you are working over and how wide.
- `**kwargs` carries whatever the model was constructed with (`dropout`,
  `chunk`, and anything strategy-specific like `gate=`).

A strategy is a plain function rather than a class because not all of them wrap
a single module — the hybrid strategies compose two different operators over
different axes.

## Three shapes a strategy can take

**1. Rewrite the schedule, delegate the rest.** The cheapest useful thing, and
only possible because a plan is a value:

```python
def pyramid(mixer, plan, lattice, d_model, **kwargs):
    """Sweep the largest axis every other layer; rotate the rest through."""
    order = [lattice.axis_names[lattice.axis_index(s.axis)] for s in plan]
    biggest = max((a for a in order if a != "time"), key=lattice.axis_size)
    ...
    return td.axial_scan(mixer, td.ScanPlan.from_list(steps), lattice, d_model, **kwargs)
```

**2. Compose two operators over different axes.** This is what `td.cafa` and
`td.axial_attention` do: per-axis kernels own the spatial axes, the model's
mixer owns time. Nothing stops a strategy from using attention on one axis, a
scan on another, and nothing at all on a third.

**3. Build something entirely your own.** The return value only has to be an
`nn.Module` mapping `(B, [T,] *shape, H)` to the same shape. `axial_apply` and
`axial_contract` are exported so a new strategy can reuse the fold rather than
reimplement it.

## Run the conformance suite — it will find things

`check_block` takes a strategy exactly as it takes a mixer:

```python
def factory(lat, d_model, plan=None):
    return td.LSTM(d_model, len(lat.axis_names) + 1, lat, plan=plan, method=pyramid)

report = td.testing.check_block(factory)
```

**Both bugs in the example above were found this way, not by reading it.** The
first version of `pyramid` failed two checks:

```
[FAIL] shape is preserved — ZeroDivisionError: integer division or modulo by zero
[FAIL] output is covariant with axis storage order — output depends on the order
       axes happen to be stored in, not just on the sweep order
```

- **The division by zero** was `others[(i // 2) % len(others)]` on a rank-1
  lattice, where the only axis *is* the dominant one and `others` is empty.
  Caught by the cheapest check in the suite, at the rank most people skip.
- **The covariance failure** was subtler and is the classic N-D bug in one
  line: the strategy took its axis order from `lattice.axis_names`, so the same
  model over the same data laid out differently produced a *different
  schedule*. A model whose behaviour depends on storage order rather than on
  the sweep order it was asked for is wrong in a way that no loss curve will
  ever show you. The fix is to order axes by the plan, which is
  storage-independent.

If you write a strategy that rewrites schedules, assume you have made one of
these two mistakes until the report says otherwise.

## Inspect what your strategy actually did

```python
print(model.nd.plan.coverage(lattice))
# Coverage(8 layers)
#   time    2→   0←  forward
#   row     0→   2←  backward
#   col     2→   2←  both
```

`coverage` is the machine-readable answer to "did every axis get swept, and in
both directions". Here `row` came out **backward only** — a real property of
this toy schedule, invisible in its source, and exactly the class of bug
recorded as DEBUG.md #4 (a published implementation whose every axis was
silently pinned to one direction). Assert on coverage in your own tests:

```python
assert cov["time"].backward == 0          # time must stay causal
assert not cov.unswept                    # every axis gets mixing
```

## Register it (only for config files)

```python
td.register_nd_method("pyramid", pyramid)
td.build({"kind": "lstm", "d_model": 16, "n_layers": 6, "method": "pyramid", ...})
```

Registration exists because YAML cannot hold a callable. In Python, pass the
function. A checkpoint of a model built with an *unregistered* strategy refuses
to save, with a message saying why — a checkpoint that cannot name its own
strategy could never rebuild itself.

## Next

- [Adding a mixer](adding-a-mixer.md)
- [DESIGN.md](../DESIGN.md) — why mixer and method are separate concepts
- [BENCHMARKS.md](../BENCHMARKS.md) — what each shipped strategy costs, and
  where the factorized one starts to win
