# Adding a mixer

This is the page that matters most, because the extension point *is* the
product. A mixer is a 1-D sequence model; everything N-dimensional — the
lattice, the schedule, the absent cells, the permutations — is the library's
job, and you never write any of it.

The worked example lives in
[`examples/custom_mixer.py`](../examples/custom_mixer.py) and is executed by
`tests/test_examples.py`, so nothing on this page is a snippet that used to
work.

---

## 1. The contract

```python
def __call__(self, x: torch.Tensor) -> torch.Tensor:   # (M, A, H) -> (M, A, H)
```

That is all of it. `A` is the length of the axis being swept, `H` is the
feature width, and `M` is the batch times *every other axis* folded together.

Three things a mixer is deliberately never told:

- **which axis it is sweeping.** Rows, columns, time, or "commodity" all
  arrive as the same `(M, A, H)`. This is why one implementation works at
  every rank.
- **what rank the lattice is, or which cells are absent.** Absent cells are
  zeroed before you see them and after every layer, so a mixer never masks.
- **which direction it is going.** A backward sweep arrives already flipped.
  Write it causally; the schedule gives you bidirectionality for free, and
  setting `bidirectional=True` on an inner RNN would double the width and
  leave the schedule nothing to control.

If your model can process a batch of sequences, it is already a mixer.

## 2. Write it

```python
class EMAMixer(nn.Module):
    """y_t = a * y_{t-1} + (1 - a) * x_t, with a learned per channel."""

    def __init__(self, d_model: int, init_halflife: float = 4.0) -> None:
        super().__init__()
        a0 = 0.5 ** (1.0 / init_halflife)
        self.decay_logit = nn.Parameter(torch.full((d_model,), logit(a0)))
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x):  # (M, A, H)
        a = torch.sigmoid(self.decay_logit)
        state, out = torch.zeros_like(x[:, 0]), []
        for t in range(x.shape[1]):
            state = a * state + (1 - a) * x[:, t]
            out.append(state)
        return self.out(torch.stack(out, dim=1))
```

The decay is parameterized as a *logit* rather than clamped into `(0, 1)`,
because clamping gives exactly zero gradient at the boundary the model most
wants to move along. That kind of detail is the mixer author's business; the
axis bookkeeping is not.

**The one signature requirement:** the first constructor argument is
`d_model`. The composition layer builds one mixer per layer by calling
`Mixer(d_model, **mixer_kwargs)`.

## 3. Get a model class for free

```python
from torch_dimensions.models.base import LatticeModel


class EMA(LatticeModel):
    _mixer = EMAMixer
```

Two lines, and you have `nd_method=`, `plan=`, `lattice=`, `d_input=`,
`dropout=`, `chunk=`, `.config`, `.save()`, `.to_spec()` and viewer support —
the same surface `td.LSTM` and `td.Mamba` have, because they are written the
same way.

## 4. Run the conformance suite

This is the step to not skip. `td.testing` is public API precisely so that a
new mixer is held to the standard the built-ins are held to.

```python
def factory(lattice, d_model, plan=None):
    return EMA(d_model, len(lattice.axis_names), lattice, plan=plan)


report = td.testing.check_block(factory, reference=reference)
print(report)
```

```
[  ok] shape is preserved — ranks (1, 2, 3)
[  ok] gradients flow and gradcheck passes — 10 tensors, gradcheck clean
[  ok] rank-1 equals the bare 1-D module — bitwise identical
[skip] Kronecker identity (kernel family) — no `kernels` adapter given
[  ok] absent cells cannot influence the output — 2 sparse lattices
[  ok] output is covariant with axis storage order — rank 3, storage order rotated
[skip] torch.compile matches eager — check_compile=False
```

What each check is actually protecting you from:

| check | the bug it catches |
|---|---|
| shape | a mixer that changes width, or a fold that does not invert |
| gradients + gradcheck | a parameter that never learns; a wrong backward |
| rank-1 equivalence | the N-D machinery doing *anything* on a 1-D lattice |
| absent-cell inertia | values from cells that do not exist reaching an output |
| storage covariance | an output that depends on axis storage order, not sweep order |

**Supply `reference=`.** Without it that check is *skipped*, and the report
says so — a skip is recorded, never silently passed. The reference is what one
pre-norm residual layer around your bare mixer computes:

```python
def reference(block, x):
    return x + block.nd.mixers[0](block.nd.norms[0](x))
```

On a rank-1 lattice the whole apparatus must reduce to exactly that, bitwise.
It is the fastest way to discover that a fold or permutation is subtly wrong.

Then check that it *learns*:

```python
td.testing.check_trainable(factory, d_model=16, steps=120)
# {'initial': 2.64, 'final': 0.37, 'held_out': 0.32, 'ratio': 8.4}
```

`check_trainable` fits a task that genuinely requires axial mixing, on fresh
data every step, scored held out, with a negative control that must fail. A
learning test without a negative control measures capacity, not learning.

## 5. Register it (optional)

```python
td.register_model("ema", EMA)
model = td.build(
    {"kind": "ema", "d_model": 32, "n_layers": 8, "lattice": {"shape": [4, 5, 3], "time": True}}
)
```

Registration is what lets a config file or a checkpoint *name* your model, so
it can rebuild itself. Everything else works without it.

## 6. Use it at any rank

```python
lattice = td.Lattice(shape=(4, 5, 3), names=("depth", "row", "col"), valid=observed, time=True)
plan = td.ScanPlan.paired(lattice.axis_names, n_layers=8, bidirectional=("depth", "row", "col"))
model = EMA(d_model=32, lattice=lattice, plan=plan, d_input=2)

print(plan.coverage(lattice))
# Coverage(8 layers)
#   time     2→   0←  forward
#   depth    1→   1←  both
#   row      1→   1←  both
#   col      1→   1←  both
```

A sparse 3-D lattice with a time axis and a bidirectional schedule, from a
mixer that knows about none of those things.

---

## Common mistakes

**Masking inside the mixer.** Absent cells are already zero when you get them,
and re-masking with a mask you derived yourself is how the two disagree.

**Reading `x.shape[0]` as the batch.** It is the batch times every unswept
axis. Anything per-example must come through the feature dimension.

**Making it bidirectional internally.** Set the schedule, not the module.

**Depending on `A` at construction.** The swept axis length varies by axis and
by call — the time axis has no static length at all. Learn per-*channel*
parameters, not per-position ones. (If you genuinely need per-position
parameters, that is the kernel family, not a mixer.)

**Skipping `check_block` because it "obviously works".** Every bug in
[DEBUG.md](../DEBUG.md) obviously worked first.

## Next

- [Adding an nd_method](adding-a-method.md) — changing *how* the axes are
  handled, rather than what happens along one.
- [DESIGN.md](../DESIGN.md) — why the boundary is where it is.
