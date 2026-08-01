# LTI and non-LTI mixers: what changes in N dimensions

The library's premise is that any 1-D mixer can be swept over an N-D lattice.
That is true, and it is not uniform. *Which* mixer you sweep decides whether
the sweep order is a modelling decision or a no-op, whether direction buys
anything, and whether the whole stack collapses into a single N-D operator.

This document states those consequences and measures them. Every number here
comes from `tests/test_lti.py` and `td.testing.check_lti`, and every claim that
turned out to be folklore rather than fact is marked as such — there is one,
and it is the interesting part.

## The two properties

A mixer is a map from `(M, A, H)` to `(M, A, H)` — a batch of length-`A`
sequences. Two properties matter:

**Linear.** `f(x + y) = f(x) + f(y)` and `f(αx) = αf(x)`. Measured as relative
error; a bias makes a mixer *affine*, which is reported separately because
subtracting the constant recovers linearity.

**Time-invariant.** Delay the input and the output is delayed identically.
Measured by zero-padding the front of the signal rather than by rolling it,
because time invariance is a claim about a system **at rest**: feed it nothing,
then feed the signal later, and the same thing should come out later. Rolling
would wrap a different prefix into place and measure memory decay, which is a
different question with a different answer.

```python
td.testing.check_lti(lambda: td.mixers.ConvMixer(64, activation=None))
# ConvMixer: LTI (affine)
#     additivity          1.6e-16
#     homogeneity         1.4e-16
#     shift equivariance  0.0e+00
#     response to zero    3.1e-02
```

It never raises. No mixer is *supposed* to be LTI; this is a classification,
not a check.

## Every mixer this library ships

Measured at `d_model=4`, float64, `eval` mode. "Order gap" is the relative
difference between sweeping `h` then `w` and sweeping `w` then `h`, with a
**shared** mixer instance and no LayerNorm, so the only difference is order.

| mixer | linear | time-invariant | verdict | order gap |
|---|---|---|---|---|
| *scalar-valued filter* (`A ⊗ b`) | **yes** | **yes** | LTI | 3e-16 |
| `ConvMixer(activation=None)` | **yes** | **yes** | LTI | 3e-01 |
| `ConvMixer()` (GELU) | 4e-01 | **yes** | time-invariant, nonlinear | 1e-01 |
| `TCNMixer` | 4e-01 | **yes** | time-invariant, nonlinear | 4e-02 |
| `S4DMixer` | 5e-01 | **yes** | time-invariant, nonlinear | 2e-02 |
| `S4Mixer` | 6e-01 | **yes** | time-invariant, nonlinear | 4e-02 |
| `MambaMixer` | 5e-01 | 4e-04 | neither | 1e-02 |
| `LSTMMixer` | 6e-01 | 3e-03 | neither | 2e-02 |
| `GRUMixer` | 6e-01 | 3e-03 | neither | 5e-02 |
| `AttentionMixer` | 8e-02 | 2e-02 | neither | 2e-01 |

Read the first two rows together. They are the same class, both LTI, and their
order gaps differ by fifteen orders of magnitude.

## The correction: LTI does not mean order-free

The folklore — and my own first draft of this file — says: *per-axis LTI
operators commute, so for an LTI mixer the sweep order carries no information.*
The first half is true for scalar filters and the conclusion is false for every
multichannel network anyone actually builds.

A 1-D convolution with `groups=1` is a **matrix-valued** filter: offset `j`
carries its own channel matrix `W[:, :, j]`. Composing along two axes gives

```
(F_h F_w x)[c] = Σ W[c,m,i] W[m,n,j] x[n, h+i, w+j]
(F_w F_h x)[c] = Σ W[c,m,j] W[m,n,i] x[n, h+i, w+j]
```

which agree only when `W[:,:,i] W[:,:,j] = W[:,:,j] W[:,:,i]` for every pair of
offsets. Random matrices do not commute, so the order gap is `3e-01` — not a
rounding artifact, a different model.

The operator commutes across axes exactly when it factorizes as a channel
matrix times a spatial filter, `W[out, in, j] = A[out, in] · b[j]`. Then each
axis contributes `C_axis ⊗ A`, the spatial parts commute because they act on
different axes, and the channel parts are the same matrix. Constructed that
way, the gap is `3e-16` — machine precision.

The depthwise row is the instructive one. A depthwise convolution *is* a scalar
filter per channel and would commute; the pointwise mix that follows it is what
breaks it. Separability across channels and commutation across axes are not the
same property, and one does not imply the other.

**So `ScanPlan` is not only for the selective models.** A linear CNN swept over
a lattice has an order-dependent result, and any implementation that reorders
axes for convenience is changing the model.

## What LTI *does* buy: exact separability

The property that survives is stronger and more useful than commutation.

For a linear mixer with no bias, a stack of one sweep per axis is **exactly** an
N-D convolution whose kernel is the outer product of the per-axis kernels —
contracted in the order the sweeps happened. Not an approximation:

```
tests/test_conv.py::test_separable_stack_equals_a_full_nd_convolution
    rank 2 and rank 3, checked against F.conv2d / F.conv3d, max diff < 1e-12
```

This is the convolutional twin of the kernel family's Kronecker identity, and
it is what "S4ND applies its axes simultaneously" rests on: with an LTI mixer
the sequential sweeps and the joint kernel are the same operator, so a single
FFT over all axes is available. The joint kernel still depends on the order —
separability and order-independence are different claims, and only the first
one holds.

**Checked against S4ND itself, not just against the theory.** S4ND does not
sweep: it builds one 1-D kernel per axis, outer-products them in Fourier
space, and applies a single N-D FFT convolution. Transcribing that composition
as an oracle and feeding it our own kernels:

```
tests/test_published_composition.py
    our sequential per-axis sweep vs S4ND's simultaneous N-D kernel
    rank 2 and rank 3, max relative difference 1.8e-17
```

So a plan of one-axis-per-layer sweeps reproduces, exactly, a model that was
never written as a sweep. That is the strongest single piece of evidence for
the library's premise, and it is available *because* S4ND's kernels are
diagonal in channels — the scalar-filter case above. The negative control in
that file gives the axes a channel-mixing filter and the identity dies, which
is the same correction stated in the other direction.

Two things follow for free:

- **Parameter cost.** A separable stack is `r·k` parameters where a full N-D
  kernel is `k^r`. That is the whole bargain, and it is why the rank-1
  restriction is worth stating: a separable stack cannot represent a diagonal
  edge detector, and depth plus channel mixing is what buys most of it back.
- **Direction is nearly free.** For a *centred* convolution, sweeping backwards
  is the same operator with a mirrored kernel. Constructed symmetric, forward
  and reverse agree to `1e-12`; an LSTM disagrees by `1e-3` or more. So
  `bidirectional=` earns its keep for causal and recurrent mixers and buys a
  mirrored filter for a CNN.

## Why the non-LTI models are the way they are

**Recurrences are not time-invariant, despite being perfectly causal.** A gated
recurrence fed zeros does not sit still: its biases drive the state, so a
delayed signal arrives to a machine in a different configuration than the
original did. The deviation is small (`3e-03` for `nn.LSTM`) because the state
settles — it is a transient, not a wholesale failure — but it is real, and it
is the reason "causal" and "time-invariant" are separate columns in the table.

**Attention is permutation-equivariant, which is stronger than shift-equivariant
on a fixed token set, and still fails this test.** Delaying a signal means
padding it, and padding adds tokens to attend over. The token *set* changes, so
the output does. This is not a defect; it is the reason positional embeddings
exist at all.

**Mamba's selectivity is the point.** Its `A`, `B`, `C`, `dt` are functions of
the input, so there is no fixed operator to commute with anything, no joint
kernel to build, and no simultaneous application. Every axis must be swept in
sequence, the order matters, and the direction matters — which is exactly why
Mamba-ND needed a schedule and S4ND did not. `ScanPlan.paired()` exists because
of the `1e-02` in that row.

## Practical consequences

| if your mixer is… | then… |
|---|---|
| a scalar-valued LTI filter | sweep order is genuinely irrelevant; pick any plan |
| LTI, multichannel | the stack is one N-D kernel, but the order changes which one |
| time-invariant, nonlinear (S4, TCN, conv) | order matters; direction matters for causal ones |
| neither (Mamba, RNN, attention) | order *and* direction are architectural choices — measure them |

The library does not enforce any of this. It measures it, and it gives the
schedule a name so the choice is visible in a config file instead of buried in
a loop.

## Reproducing

```bash
pytest tests/test_lti.py -q
```

```python
import torch_dimensions as td

print(td.testing.check_lti(lambda: td.mixers.MambaMixer(64)))
```
