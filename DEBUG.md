# Debug log

Every bug found in this library, what caused it, how it was found, and what now
prevents it from coming back.

Kept because the *patterns* are worth more than the individual fixes. Four of
the first eight below are the same two mistakes wearing different clothes, and
the techniques that caught them (§B) caught them in code that was already
passing a green test suite — as did the later entries, found by re-running
those same techniques against code the suite already blessed.

The last four (#19–#22) came from a different direction and are worth reading
together: two were found by *looking at what shipped* rather than at what the
build said, and two were found by running this library's own conformance suite
against this library's own documentation, where it failed twice.

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
| 9 | `data/` — `Sample`/`Batch` unpicklable, `DataLoader(num_workers>0)` hangs | high | targeted probe | fixed |
| 10 | `models/rnn.py` — `plan` silently overrode `n_layers` | medium | targeted probe | fixed |
| 11 | `compose/kernel.py` — `nan_to_num` laundered input NaNs | medium | targeted probe | fixed |
| 12 | `compose/kernel.py` — absolute epsilon vs relative cancellation | high | targeted probe | fixed |
| 13 | `lattice.py` — `mask()` was a view; `valid` aliased the caller's tensor | high | targeted probe | fixed |
| 14 | `data/collate.py` — targets dropped when the first sample lacked one | medium | targeted probe | fixed |
| 15 | `data/window.py` — `split_at_time` on unsorted times: silent nonsense | medium | targeted probe | fixed |
| 16 | `testing.py` — conformance checks ran at ranks the caller excluded | medium | audit | fixed |
| 17 | `compose/scan.py` — `chunk=0` errored from deep inside `range()` | low | targeted probe | fixed |
| 18 | `lattice.py` — device lattice could not index CPU tensors | medium | device probe (MPS) | fixed |
| 19 | viewer bundle — a stale local training run shipped inside the wheel | medium | looking at the artifact | fixed |
| 20 | `benchmarks/bench.py` — a memory column that measured the driver, not the model | medium | reading the output | fixed before publishing |
| 21 | `examples/custom_method.py` — a strategy that indexed an empty axis list at rank 1 | low | conformance suite | fixed |
| 22 | `examples/custom_method.py` — a schedule derived from *storage* order, not sweep order | high | conformance suite | fixed |
| 23 | `testing.py` — gradcheck ran at a width the caller never asked for | low | new mixer | fixed |
| 24 | `data/memmap.py` + `[safetensors]` — assumed numpy is always there | medium | CI (a leaner environment than the laptop) | fixed |
| 25 | `examples/repro/*.py` — a dry-run timer that measured the dispatch queue | low | a run that took 40x its estimate | fixed |

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

## 9. `Sample`/`Batch` could not be pickled, so worker loading hung

```python
class Sample(dict):
    __getattr__ = dict.__getitem__  # sample.x — neat, and broken
```

A missing attribute raised ``KeyError`` where Python promises
``AttributeError``. That breaks ``hasattr`` and ``getattr(s, "y", None)`` — and
breaks **pickling**, because pickle probes for optional dunders like
``__getstate__`` with ``getattr`` and only tolerates ``AttributeError``. Every
``DataLoader(num_workers>0)`` pickles each sample through the worker queue, so
multiprocessing loading did not fail cleanly: the worker died mid-pickle and
the main process **hung forever waiting on the queue**.

**Cause.** A shortcut that changes an exception type across a protocol
boundary. The one-liner reads as equivalent to a real ``__getattr__`` and is
not.
**Fix.** A ``__getattr__`` that translates ``KeyError`` to ``AttributeError``.
**Guarded by** `test_samples_and_batches_survive_pickling`,
`test_a_missing_field_reads_as_absent_not_as_a_keyerror`, and the end-to-end
`test_dataloader_with_worker_processes`. Note the first is the canonical guard:
on the broken code the worker test *hangs* rather than fails, which is exactly
why a fast direct test must sit in front of a slow end-to-end one.

---

## 10. A `plan` silently overrode `n_layers`

`td.LSTM(8, n_layers=6, lattice=lat, plan=two_step_plan)` built a 2-layer
model. No error, no warning — a model quietly shallower than requested, the
same silent-downgrade class as #4.

**Fix.** The plan still wins, but the downgrade warns. A warning rather than an
error because the first attempt at a hard error immediately broke this suite's
own generic factories — builders that fill ``n_layers`` unconditionally and add
a plan only sometimes are legitimate, and the default ``n_layers=1`` passes
untouched either way.
**Guarded by** `test_rnn_warns_when_a_plan_disagrees_with_n_layers`.

---

## 11. `nan_to_num` laundered upstream NaNs into finite output

After #3's magnitude guard, the division in the sparse renormalizer cannot
create a fresh NaN — so the ``nan_to_num`` wrapped around it could only ever
fire on NaNs already present in the *input*, zeroing them. A diverging model's
NaNs vanished mid-network into plausible finite numbers.

**Cause.** A guard kept after the failure it guarded against was fixed
properly. Same shape as #3: ask what a ``nan_to_num`` is for, and if the
answer is "nothing anymore", it is hiding something else.
**Fix.** Removed. A NaN that arrives must leave.
**Guarded by** `test_a_nan_in_the_input_is_not_silently_laundered`.

---

## 12. The renormalizer's epsilon was absolute; cancellation is relative

The #3 fix guarded ``|den| < 1e-6``. A signed float32 kernel row of
``[1.0, -0.9999]`` over two present cells leaves a denominator of ~1e-4 —
small enough to amplify by 1e4, large enough to sail past any tiny fixed
threshold. Measured: input max 252, output max 1.8M, a ~7,000x blow-up in the
library's *default* dtype. The prediction that half precision would be the
vulnerable dtype was exactly backwards: fp16/bf16 round the near-cancellation
to exact zero, which the old guard caught, and came out bounded.

**Cause.** The fix for #3 repeated #3's mistake one level up: it guarded a
threshold instead of the phenomenon. The phenomenon is cancellation, and
cancellation is relative.
**Fix.** Compare the signed mass against the absolute mass that went into it
(``|kernel| @ mass``); a line is degenerate when almost everything cancelled.
A genuinely *small* mass still renormalizes exactly, because the numerator
carries the same factor.
**Guarded by** `test_float32_near_cancellation_does_not_explode`.
`test_a_genuinely_small_mass_still_renormalizes_exactly` passes pre-fix — it
pins the new guard against overreach, in the sense #1/#3 taught us to state.

---

## 13. `mask()` returned a view of `valid`, and `valid` aliased the caller

Two directions of the same aliasing. ``mask(torch.bool)`` was
``valid.reshape(...).to(bool)`` — a **view**, so writing into "your" mask
corrupted the lattice through it. And ``valid.to(torch.bool)`` returns the
caller's own tensor when it is already bool, so a caller who reused or edited
that tensor after construction desynced the cached ``flat_idx`` — measured:
``flat_idx`` still listing 3 cells while ``n_valid`` said 2, which is
``scatter`` misplacing data, the exact failure #2 was fixed to prevent.

**Cause.** #2 froze the *attributes* and left the *tensors* shared. A value
object holding mutable buffers is only a value if it owns them.
**Fix.** Clone ``valid`` at construction; ``mask()`` builds a fresh tensor per
call (callers buffer it at module construction — there was nothing worth
caching).
**Guarded by** `test_mask_is_a_copy_not_a_view_of_valid`,
`test_the_callers_valid_tensor_is_not_aliased`.

---

## 14. `collate_lattice` keyed target presence off `samples[0]`

A batch mixing horizon-0 samples with targeted ones either silently dropped
every target (first sample lacked one) or crashed mid-stack. Under a shuffled
loader, *which* of the two you get changes per batch.

**Fix.** Target presence must be unanimous; a mixed batch raises.
**Guarded by** `test_collate_refuses_mixed_target_presence`, both orderings.

---

## 15. `split_at_time` trusted its times to be sorted

Unsorted timestamps produced a silently nonsensical train/test split — the
quietest possible leakage bug, in the function whose entire job is preventing
leakage. ``LatticeTable.times`` happens to be sorted, which is exactly the
assumption-holds-in-one-configuration shape of §A1: the function is public and
takes any sequence.

**Fix.** Sortedness is checked; unsorted input raises.
**Guarded by** `test_split_at_time_refuses_unsorted_times`.

---

## 16. The conformance suite tested ranks the caller never requested

``check_block(factory, ranks=(3, 4))`` gradchecked a **rank-2** lattice —
hardcoded "for speed" — and would have run the rank-1 equivalence check on a
rank-1 one. A factory that is only valid at its stated ranks failed checks it
should pass, and the passing checks partly measured a configuration nobody
asked about.

**Fix.** The gradient check uses a requested rank; the equivalence check skips
with a reason when rank 1 was not requested, per the suite's own
skips-are-not-passes rule.
**Guarded by** `test_checks_run_only_at_ranks_the_caller_requested`.

---

## 17. `chunk=0` failed as `range() arg 3 must not be zero`

Not wrong, just useless: the contract violation surfaced three frames deep
with no mention of ``chunk``. Boundaries state their contracts;
``axial_apply`` now validates ``chunk >= 1`` itself.
**Guarded by** `test_chunk_must_be_positive`.

---

## 18. `gather`/`scatter` broke in one device direction

``lat.to(device).gather(x_cpu)`` raised from three frames inside an indexing
kernel, while the mirror case — CPU lattice, device tensor — happened to work,
because torch tolerates CPU indices on a device tensor but not the reverse.
The cached ``flat_idx`` lives wherever ``valid`` lives, and callers should not
have to know that.

Found **without CUDA**: device-placement bugs need *a* second device, not a
specific one, and this machine's MPS backend is one. The whole class is now
pinned by [tests/test_device.py](tests/test_device.py), which runs against
whatever accelerator exists (MPS here, CUDA elsewhere) and skips visibly on
CPU-only machines. What MPS cannot vouch for — CUDA kernel numerics,
``torch.compile`` backends, float64 on device — is stated in that file's
docstring rather than silently unclaimed.

**Fix.** Index with ``flat_idx.to(x.device)``; a no-op when they agree.
**Guarded by** `test_gather_scatter_round_trip_across_device_mismatches`,
both mismatch directions.

---

## 19. A stale training run shipped inside the viewer bundle

`td.viz.show` serves a static bundle built from `viewer/`. Vite copies
everything in `viewer/public/` into that build, and the live-training script
writes `viewer/public/run.json` there — so the first bundle carried a training
run from this laptop, and the viewer loaded **it in preference to the model
passed to `show()`**. The feature's central promise ("show me *this* model")
was broken by a file nobody thought of as part of the feature.

Found by opening the served page and reading the sidebar: the model card said
`Mamba, 18 layers, 4x5x6x4` for a model built as `S4DND, 8 layers, 4x5x6`.
Every automated check passed — the bundle built, the server served, the tests
(as written at that moment) were green.

This is the 35 MB sdist (`node_modules` in a released tarball) one directory
over, and the same lesson: **the thing that ships is the thing to inspect.**
`twine check` passes bloated tarballs happily and a build log says nothing
about what a page will render.

**Fix.** `viewer/install_bundle.py` strips `run.json` from the copied bundle
and says so. **Guarded by**
`test_no_local_training_run_rides_along_in_the_bundle`, which asserts the
served `/run.json` is a 404, plus a publish-workflow step that greps the
built *wheel* for `viz/static/index.html` rather than trusting the build.

---

## 20. A benchmark column that measured something else entirely

The Phase 10 benchmark table had a `peak MB` column reading
`torch.mps.driver_allocated_memory()` on MPS — which is the whole process's
driver allocation, not the model's. It reported **18 GB for a 25,000-parameter
model**, and would have been published in BENCHMARKS.md as a memory
measurement.

Nothing failed. The number was plausible in shape (a float, in MB, varying
between rows) and absurd only if you knew what it should be. It was caught by
reading the generated table and asking why two models three orders of
magnitude apart in size used the same memory.

**Fix.** Only CUDA tracks an allocation high-water mark, so the column reads
`n/a` everywhere else, and `peak_memory_mb`'s docstring records why the
plausible substitute was removed. **A number that is not what its header
claims is worse than a blank** — a blank invites a question, a wrong number
answers it.

---

## 21 & 22. The conformance suite found both bugs in its own documentation's example

While writing `docs/adding-a-method.md`, the example strategy — thirty lines
that rewrite a schedule — was run through `check_block`. Two failures:

```
[FAIL] shape is preserved — ZeroDivisionError: integer division or modulo by zero
[FAIL] output is covariant with axis storage order
```

**#21** was `others[(i // 2) % len(others)]` on a rank-1 lattice, where the
only axis *is* the dominant one and `others` is empty. Low severity, instant
diagnosis, and caught by the cheapest check in the suite at the rank people
skip because "rank 1 is trivial".

**#22** is the interesting one, and it is the canonical N-D bug in a single
line: the strategy took its axis order from `lattice.axis_names`, so the same
model over the same data *laid out differently* produced a different schedule.
No exception, no NaN, no loss curve that looks wrong — just a model whose
behaviour depends on storage order rather than on the sweep order it was
asked for. This is the same shape as #4 (bidirectional aliasing) and it is why
the covariance check exists at all.

Both are fixed in the example, and both are now *described in the guide* as
what the suite caught, with a test that reproduces the broken version to prove
the check still fails it. The most useful thing a conformance suite can do for
a documentation page is embarrass it.

---

## 23. The gradient check substituted its own width

`check_block`'s gradient step built the block at `d_model=2` regardless of what
the caller passed, for speed. That went unnoticed for as long as every mixer
accepted every width — and surfaced the moment `AttentionMixer` arrived, whose
head count must divide `d_model`: a check about *gradients* failed with
`n_heads=4 does not divide d_model=2`, about a model the caller never asked
for.

This is DEBUG.md #16 with a different argument. That entry was about ranks the
caller excluded; this is a width the caller excluded. The pattern is the
general one in §A: **a test harness that quietly substitutes its own
parameters is testing something other than what it reports on** — and the
failure mode is not always a confusing error, it can equally be a silent pass
for a configuration nobody runs.

**Fix.** Use the caller's `d_model`. **Guarded by**
`tests/test_attention_mixer.py::test_conformance`, which cannot pass under
the old behaviour.

---

## 24. numpy is not a torch dependency, and the laptop lied about it

`MemmapSource` reads and writes `.npy`; `safetensors`' torch bindings import
numpy internally. Both worked perfectly on the development machine and both
failed in CI with `ModuleNotFoundError: No module named 'numpy'` from three
frames inside somebody else's writer — because torch does not require numpy,
and the CI install is a genuinely minimal one.

Two fixes, because there are two faults. The extras now declare numpy where
the feature needs it, so CI actually exercises those paths instead of skipping
them and reporting green. And `MemmapSource` raises an error that names the
package and points at `TensorSource` as the in-memory alternative, rather than
surfacing an import failure from a stack the user did not write.

**The process finding is the larger one.** This landed one commit after
`scripts/check.sh` was added *specifically* to stop CI surprises, and the
script had passed. It makes the **commands** identical to CI's; it cannot make
the **environment** identical, and this failure lived entirely in the
difference. A local gate can prove "the checks pass here". Only CI can prove
"the checks pass in a clean environment", and the honest conclusion is that
the two answer different questions — so the script now says so in its own
docstring rather than implying it is a substitute.

**Guarded by** `[dev]` carrying numpy (so the paths run in CI at all) plus
`pytest.importorskip("numpy")` in both test modules, so an environment without
it skips *visibly* instead of failing.

---

## 25. The timer measured the queue, and I had already written that down

The reproduction scripts have a `--dry-run` that times one training step and
extrapolates to minutes-per-epoch, so a long run can be scheduled with some
idea of its cost. For the 2-D Mamba model it reported **0.03 s/step, 0.5
min/epoch**. The real run took **1.19 s/step, ~19 min/epoch** — 40x more — and
a six-epoch job was queued on the strength of the wrong number and had to be
killed an hour later.

The cause is one missing line. On MPS and CUDA a `.backward()` returns as soon
as the work is *queued*, so an unsynchronized timer measures dispatch. This is
stated, in bold, in the second paragraph of `benchmarks/bench.py`:

> **the timer synchronizes**: on MPS and CUDA the dispatch returns long before
> the work does, so an unsynchronized loop measures the queue, not the model.

Written by the same hand, the same day, in the file next door — and then not
applied to the second timer, because the second timer did not look like a
benchmark. It looked like a convenience.

**The pattern (§A):** a rule written down in the place it was learned does not
transfer to the next place it applies. What transfers is a shared function.
`sync(device)` now lives in `harness.py` and is called by both dry-run paths,
which is the only version of "remember to synchronize" that survives contact
with a second author or a tired one.

**Guarded by** nothing automatic, and that is honest: this is a measurement
helper, and a test that asserts a timing is the flake this project has already
refused to write once (`tests/test_perf.py`). The mitigation is that the
timing code exists in exactly one place now.

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

## 26. The spec described a model the library never runs

`td.spec(model)` emits the document the viewer renders, `td.viz.show` serves,
and a downstream tool may parse. For a kernel-family model — `method=td.cafa`
or `td.axial_attention` — it said this:

```
layer 0: axis time, mixer LSTMMixer
layer 1: axis h,    mixer LSTMMixer
layer 2: axis w,    mixer LSTMMixer
```

None of layers 1 and 2 happens. `AxialKernel.forward` contracts **every**
spatial axis with a kernel on **every** layer, and applies the mixer along
time only. The document was describing a scan model that was never built, and
`"nd_method": {"family": "scan"}` was a hardcoded string in a function whose
name — `scan_model_spec` — was the only thing about it that was still true
after the kernel family landed.

The viewer, meanwhile, was *already right*: `Scene.jsx` sniffed
`nd_method.name === "AxialKernel"` and drew a simultaneous flash instead of a
travelling wavefront. So the renderer had a special case that the document it
renders did not, and the sidebar — which reads `layer.axis` directly — printed
the three sweeps anyway. Half the system knew.

**Cause.** A schema written when there was one family, extended by adding a
family rather than by extending the schema. The per-layer record had exactly
the fields a scan needs (`axis`, `axis_index`, `reverse`) and no way to say
"this layer contracts a set of axes and sweeps none of them", so the kernel
family was serialized through the only vocabulary available.

**Found by** writing a third family. Asking "what will a `flatten` layer put
in the `axis` field?" has no answer, and the same question asked of the
existing kernel family turned out to have a wrong one already shipped.

**Fixed** by giving layers a `kind` (`scan` | `kernel`), an `axes` list of what
the layer actually mixes, and `contracted` for the axes handled by kernels;
`sweeps` gains `contracted_axes`, and `directions` now lists only axes a mixer
genuinely sweeps, because a contraction has no direction. `nd_method.family`
is derived from the composition class. Spec version 1 to 2.

**Guarded by** `tests/test_spec.py::test_the_kernel_family_spec_does_not_claim_spatial_sweeps`
and the golden fixtures, which is how the blast radius was visible at all: the
regeneration diff showed four stored specs changing, and the cafa one changing
in exactly the way the fix intends.

**The pattern (§A):** the renderer's special case was documentation of a defect
in the data. A consumer that has to compensate for a producer is evidence the
producer is wrong, and it is worth reading such a special case as a bug report
rather than as a feature.

---

## A. Recurring patterns

Four classes account for the first eight — and the later finds keep landing in
them: #10 is another silent downgrade like #4, #11 another guard aimed at the
wrong failure like #3, and #9 another object whose neat idiom does not deliver
what it announces, like A3.

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
   notices. Found #6 and validated #4. Cheap: revert with `git checkout` —
   and now automatic: [`scripts/mutate.py`](scripts/mutate.py) holds a catalog
   of seven mutations, each one a bug from this list, and runs weekly in CI.
   All seven are currently caught. A survivor would be a hole in the tests
   rather than a bug in the code, which is the distinction that makes this
   worth automating at all.
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
7. **Looking at the artifact rather than the process.** #19 and #20 were both
   invisible to every automated check and obvious within seconds of *reading
   the output*: a served page whose sidebar described the wrong model, a table
   claiming 18 GB for a 25k-parameter model. Build logs, green suites and
   `twine check` all report on the process. Open the page, read the table,
   list the wheel.
8. **Running the conformance suite on the examples.** #21 and #22 were in this
   project's own documentation, in code written to *demonstrate* correctness.
   The suite found both in one run.
