# Debug log

Every bug found in this library, what caused it, how it was found, and what now
prevents it from coming back.

Kept because the *patterns* are worth more than the individual fixes. Four of
the eight below are the same two mistakes wearing different clothes, and the
techniques that caught them (§B) caught them in code that was already passing a
green test suite.

Every citation here is live. The pre-fix code for each fixed bug is one
`git checkout` away, so each **Reproduce** block below lets you watch the guard
fire — and each was *run before being written down*, which is how two citations
that were not actually guards got caught (see #1 and #3). A "guarded by" line
is itself a comment stating an invariant (§A1), so the citations get a test:
[tests/test_debug_md.py](tests/test_debug_md.py) fails if any test name or
commit hash cited here stops existing.

**Scope note.** This file covers torch-dimensions only. Two further bugs were
found in separate private research repositories during the same work; they are
documented on their own fix branches and are deliberately not detailed here.
Their generalizable lessons are folded into §A without identifying details.

---

## Summary

| # | Where | Severity | Found by | Status |
|---|---|---|---|---|
| 1 | `plan.py` — `ScanPlan` hashable but mutable | high | mypy | fixed, `7c3f812` |
| 2 | `lattice.py` — cached `flat_idx` with a mutable lattice | high | mypy | fixed, `7c3f812` |
| 3 | `compose/kernel.py` — `clamp_min` on a signed denominator | high | targeted probe | fixed, `7c3f812` |
| 4 | `plan.py` — bidirectional schedule aliasing | high | design review | never shipped, `0fba849` |
| 5 | `testing.py` — `check_trainable` trained on a fixed batch | high | negative control | fixed before commit, `e5edbfb` |
| 6 | `tests/test_kernel.py` — a test that could not fail | medium | audit | replaced, `7c3f812` |
| 7 | CI — mypy was never executed | medium | audit | fixed, `7c3f812` |
| 8 | invariant script — batch dims folded into the cell index | low | crash | fixed immediately |

Severity is "what would this have cost if it reached a user", not "how hard was
it to fix". Every one of #1–#5 is silent: no exception, no NaN, just wrong
numbers or a model quietly weaker than requested.

---

## 1. `ScanPlan` was hashable but mutable

```python
plan = td.ScanPlan.cyclic(("a", "b"), 4)
store = {plan: "value"}
plan.steps = (td.Step("z", True),)  # succeeded
store.get(plan)  # None — lost from its own dict
```

`__hash__` was defined over `self.steps`, so mutating a plan changed its hash
and dropped it out of any dict or set holding it. The worse consequence is
downstream: `AxialScan` builds **one mixer per step** at construction, then
`zip`s the plan against that list every forward pass. A plan edited afterwards
would pair new steps with old mixers — no error, just a model that no longer
matches its own description.

The class already used `object.__setattr__` in `__init__`, an idiom that
signals immutability, but nothing enforced it.

**Cause.** Borrowed the frozen-dataclass idiom without the frozen-dataclass
guarantee.

**Fix.** `__setattr__` and `__delattr__` raise after construction.
**Guarded by** `test_a_plan_cannot_be_mutated_after_construction`.
`test_a_plan_survives_use_as_a_dict_key` documents the value-semantics contract
but **passes even on the pre-fix code** — it never mutates anything — so it is
not a guard for this bug. An earlier revision of this file cited it as one;
running both tests against the pre-fix file is what corrected that.

**Reproduce.**

```bash
git checkout 7c3f812~1 -- src/torch_dimensions/plan.py
pytest tests/test_plan.py -k cannot_be_mutated        # 1 failed
git checkout HEAD -- src/torch_dimensions/plan.py
```

---

## 2. `Lattice` cached derived tensors but allowed mutation

```python
lat = Lattice(shape=(2, 2), valid=...)  # 3 cells present
lat.valid = ...  # now 1 cell present
lat.n_valid  # 1   — recomputed
lat.flat_idx  # [0, 1, 2]  — cached, stale
```

`n_valid` recomputes on every access; `flat_idx` is memoized. After a mutation
they disagree, and `scatter` indexes with `flat_idx` — so data lands in cells
that no longer exist, silently. This is precisely the failure mode `Lattice`
exists to make impossible, one level up.

Blocks also `register_buffer` the cell mask at construction, so mutating a
lattice desyncs an already-built module regardless of the cache.

**Cause.** Same as #1: a value object that isn't a value.

**Fix.** Immutable after `__post_init__`. `to()` already returned a new
instance rather than mutating.
**Guarded by** `test_a_lattice_cannot_be_mutated_after_construction`.

**Reproduce.**

```bash
git checkout 7c3f812~1 -- src/torch_dimensions/lattice.py
pytest tests/test_lattice.py -k cannot_be_mutated     # 1 failed
git checkout HEAD -- src/torch_dimensions/lattice.py
```

---

## 3. The sparse renormalizer divided by a clamped signed denominator

```python
out = out / (kernel @ mass).clamp_min(1e-6)
```

`clamp_min` assumes the denominator is a non-negative *mass*. That holds for a
softmax kernel. It does not hold for a signed one — and LeakyReLU-gated scores
are the default in the reference implementation this follows. A signed kernel
can cancel the denominator to **exactly zero** while the numerator stays
nonzero:

```
denominator values : [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
input  max |x|     : 1.85
output max |x|     : 1201025.71          ← 647,502x blow-up
```

The code comment asserted "lines with no present cells have a zero numerator,
the clamp keeps them finite" — true only for non-negative kernels. The comment
documented an assumption instead of checking it.

**Cause.** A guard written against one failure mode (a genuinely empty line)
that silently mis-handles a different one (cancellation).

**Fix.** Guard the *magnitude* and leave degenerate lines unscaled. A genuinely
dead line still has a zero numerator, so it stays zero.
**Guarded by** `test_a_signed_kernel_does_not_explode_when_the_mass_cancels`.
`test_a_genuinely_dead_line_is_still_zero_under_the_guard` **passes even on the
pre-fix code** — a dead line's numerator is zero either way — so it pins the
fix against overcorrection rather than catching the bug. Both roles are worth
having; they are different roles, and this file originally conflated them.

**Reproduce.**

```bash
git checkout 7c3f812~1 -- src/torch_dimensions/compose/kernel.py
pytest tests/test_kernel.py -k signed_kernel          # 1 failed
git checkout HEAD -- src/torch_dimensions/compose/kernel.py
```

---

## 4. Bidirectional schedules aliased against the axis cycle

Caught in design, never shipped — but it is the sharpest bug of the set and it
exists in published research code, so it is recorded here in full.

The obvious way to build a bidirectional sweep schedule is to cycle the axis
each layer and flip the direction each layer:

```python
axis = axes[i % n]  # period n
reverse = bool(i % 2)  # period 2
```

With an **even** number of axes the two periods phase-lock:

| layer | axis | direction |
|---|---|---|
| 0 | h | forward |
| 1 | w | **backward** |
| 2 | h | forward |
| 3 | w | **backward** |

`h` is forward forever and `w` is backward forever, at *any* depth. Adding
layers never fixes it. You asked for bidirectional and got a model with half
the receptive field on every axis — and the loss still goes down.

With **three** axes it accidentally works, because 3 and 2 are coprime. Three
axes is also the case anyone would eyeball first, so the bug hides exactly
where it would be looked for.

**Fix.** Flip after each full *cycle*, giving period `2n`, which cannot share a
factor with `n`.
**Guarded by** `test_bidirectional_cyclic_gives_every_axis_both_directions`,
which fails on exactly the even axis counts and passes on odd — the signature
of the aliasing, and evidence the test is testing the right thing.

**Reproduce.** The bug never shipped, so the reproduction is a mutation: in
`ScanPlan.cyclic`, change the direction term `(i // n) % 2 == 1` to
`i % 2 == 1`, then

```bash
pytest tests/test_plan.py::test_bidirectional_cyclic_gives_every_axis_both_directions
```

fails at 2 and 4 axes and passes at 1 and 3 — the parity signature above,
observed rather than asserted.

---

## 5. `check_trainable` proved nothing on its first version

The first version trained on one fixed batch of eight examples. Result:

```
sweeps w                   loss 2.0812 -> 0.0003        (8073x)
never sweeps w (control)   loss 2.1109 -> 0.0000  (1950510207371x)
```

The **control model — which cannot see the axis the task is defined along —
scored better than the real one.** It had memorized eight examples without
performing any axial mixing at all. Every plan would have passed.

**Fix.** Draw fresh data every step and score on a held-out batch. The same
comparison then reads:

```
sweeps w                   held-out 0.157
never sweeps w (control)   held-out 1.376
```

**Cause.** A learning test without held-out data measures capacity, not
learning.
**Guarded by** `test_a_model_that_never_sweeps_the_needed_axis_cannot_learn_it`,
which asserts the control *fails*.

**Reproduce.** Not reproducible from history: the fixed-batch version was
replaced before `e5edbfb` was committed. The numbers above are from the
session in which it was caught.

---

## 6. A test that could not fail

```python
plain = axial_contract(x, lat, 1, ones)  # `ones` is unnormalized
assert torch.allclose(plain[0, 1] / 3, torch.full((3, 1), 1 / 3))
```

Named "structural zeros dilute the result", but an unnormalized all-ones kernel
computes a *sum*, not a mean — there is no dilution to demonstrate. The final
assertion divides a known value by 3 and asserts it equals that value over 3.

**Fix.** Use a row-stochastic kernel, where the effect is real and visible:
without renormalization a line with one present cell out of three averages to
`1/3`; with it, to `1`.

---

## 7. mypy was in the plan but not in CI

[PLAN.md](PLAN.md) specifies mypy as "advisory, not gating". It was read as
optional and left out of the workflow entirely, so it had never run. First
execution: **23 errors across 5 files**, two of which were bugs #1 and #2.

**Fix.** Runs in CI with `continue-on-error: true`. Advisory means "does not
block merge", not "never executed".

---

## 8. Batch dimensions folded into the cell index

```python
absent = out.reshape(-1, H)[~valid]  # IndexError: mask [12] vs tensor [72, 8]
```

`reshape(-1, H)` collapses batch and time into the cell axis, so a
cell-indexed mask no longer lines up. Caught immediately by the crash. Recorded
only because the correct form — index with the broadcast mask rather than
flattening — is the same operation the library gets right internally, and it
was still easy to get wrong when writing a one-off script.

---

## Not bugs

Two things that look like defects and are not. Recorded so they are not
"fixed" later.

**A one-ULP drift in multi-layer stacks.** A single layer on a rank-1 lattice
is bitwise identical to the bare 1-D module. A *stack* differs by ~1e-16. The
fold reshapes, which requires contiguity, while `nn.LSTM` returns a transposed
view — so from layer two the mixer receives contiguous input where a bare stack
receives a view, and torch's RNN kernels are not bit-identical across memory
layouts. This is why the conformance suite claims bitwise equality for one
layer only.

**A surviving mutant.** Reversing the order in which non-swept axes fold into
the batch changes nothing, and no test catches it. Correct: the mixer contract
requires rows to be independent, so their order within the folded batch is
unobservable. A semantically equivalent mutant, not a coverage hole.

---

## A. Recurring patterns

Four classes account for all eight.

**A1 — An assumption that holds in one configuration.** (#3, and both
private-repo findings.) A masking or normalization step correct for one axis,
or one kernel sign, or one density, silently wrong outside it. All three
instances were *documented in a comment* asserting the property rather than
tested for it. **A comment stating an invariant is a place to put a test.**

**A2 — Periodicity aliasing.** (#4.) Two independent cycles whose periods share
a factor lock together. Anywhere a schedule combines "which thing" with "which
variant", check whether the two periods are coprime, and test at even *and* odd
counts — testing one parity finds nothing.

**A3 — Value objects that aren't.** (#1, #2.) An object that is hashed, cached
from, or used to build derived state at construction must be immutable. The
`object.__setattr__` idiom announces this without delivering it.

**A4 — Tests that cannot fail.** (#5, #6.) A test with no reachable failing
input is worse than no test, because it reads as coverage. Every assertion
should have an input that breaks it; if you cannot name one, the test is
decoration.

---

## B. What actually found these

Ranked by yield in this project.

1. **Mutation testing.** Break the code deliberately, confirm the suite
   notices. Found #6 and validated #4. Cheap: revert with `git checkout`.
   The same trick runs backwards through history —
   `git checkout <fix>~1 -- <file>`, run the guard, restore — which is how
   every **Reproduce** block above was verified, and how two cited "guards"
   were exposed as tests that pass on the buggy code (#1, #3).
2. **Independent references.** Check against something sharing no code —
   `x.cumsum(dim=d)` for an axial sweep, `torch.kron` for the factorization,
   `torch.argsort` for an inverse permutation, a hand-written decoder for an
   encoder. A round-trip against your own implementation proves only
   self-consistency.
3. **Constant-input invariants.** If a convex combination is claimed, feed a
   constant: any convex combination of identical values is that value, so any
   deviation is mass that the denominator did not count. One line, no
   tolerance-tuning, and it localizes the error immediately.
4. **Negative controls.** For anything measuring capability, include a variant
   that *must* fail. #5 was invisible until the control outscored the model.
5. **Running the tools you already configured.** #7. Twenty-three errors were
   sitting behind a command nobody had typed.
6. **Adversarial edge cases on guards.** Ask what a clamp, an epsilon, or a
   `nan_to_num` is protecting against, then construct the case it *isn't*.
   That is exactly how #3 was found.
