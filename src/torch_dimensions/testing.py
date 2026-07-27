"""The shared conformance suite.

Public API rather than test-directory scaffolding, because the extension point
*is* the product: anyone writing a mixer or an ``nd_method`` should be able to
run exactly the checks the library runs on itself.

    import torch_dimensions as td

    td.testing.check_block(lambda lat, d: td.LSTM(d, 3, lat))

The checks are ordered so the cheapest and most diagnostic run first. An axis
bug in an N-D model presents as "the model trains badly"; these turn it into a
specific failing assertion instead.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from torch_dimensions.lattice import Lattice
from torch_dimensions.plan import ScanPlan

__all__ = ["Report", "Result", "check_block", "check_trainable"]

Factory = Callable[..., nn.Module]


@dataclass
class Result:
    name: str
    status: str  # "pass" | "fail" | "skip"
    detail: str = ""

    def __str__(self) -> str:
        mark = {"pass": "ok", "fail": "FAIL", "skip": "skip"}[self.status]
        return f"[{mark:>4}] {self.name}{f' — {self.detail}' if self.detail else ''}"


@dataclass
class Report:
    results: list[Result] = field(default_factory=list)

    @property
    def failed(self) -> list[Result]:
        return [r for r in self.results if r.status == "fail"]

    @property
    def skipped(self) -> list[Result]:
        return [r for r in self.results if r.status == "skip"]

    def __bool__(self) -> bool:
        return not self.failed

    def __str__(self) -> str:
        return "\n".join(str(r) for r in self.results)


def _lattice(rank: int, *, sparse: bool = False, time: bool = False, seed: int = 0) -> Lattice:
    shape = tuple(range(2, 2 + rank))
    valid = None
    if sparse:
        g = torch.Generator().manual_seed(seed)
        valid = torch.rand(shape, generator=g) > 0.4
        valid.reshape(-1)[0] = True
        valid.reshape(-1)[-1] = True
    return Lattice(shape=shape, valid=valid, time=time)


def _input(lat: Lattice, d_model: int, batch: int, seq: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    lead = (batch, seq) if lat.time else (batch,)
    return torch.randn(*lead, *lat.shape, d_model, generator=g, dtype=torch.float64)


def _build(factory: Factory, lat: Lattice, d_model: int, seed: int, **kw) -> nn.Module:
    torch.manual_seed(seed)
    block = factory(lat, d_model, **kw)
    return block.double().eval()


def _accepts_plan(factory: Factory) -> bool:
    try:
        return "plan" in inspect.signature(factory).parameters
    except (TypeError, ValueError):  # builtins, C callables
        return False


def check_block(
    factory: Factory,
    *,
    d_model: int = 4,
    ranks: Sequence[int] = (1, 2, 3),
    sparse: bool = True,
    time: bool = False,
    batch: int = 2,
    seq: int = 3,
    reference: Callable[[nn.Module, torch.Tensor], torch.Tensor] | None = None,
    check_compile: bool = False,
    seed: int = 0,
    raise_on_failure: bool = True,
) -> Report:
    """Run the conformance checks against a block factory.

    Args:
        factory: ``(lattice, d_model) -> nn.Module``. Accepting a keyword
            ``plan`` additionally enables the permutation-covariance check,
            which needs to hold the sweep order fixed while the lattice's axis
            *storage* order changes.
        ranks: lattice ranks to exercise. Rank 1 is the one that catches
            permutation bugs fastest.
        sparse: also build on a lattice with absent cells and verify that their
            values cannot influence any output.
        reference: ``(block, x) -> expected`` for the rank-1 equivalence check.
            Omit and that check is skipped rather than silently passed.
        check_compile: compare ``torch.compile`` numerics against eager. Off by
            default because it is slow, not because it is unimportant.
        raise_on_failure: raise ``AssertionError`` with the full report when
            any check fails. The report is returned either way.

    Returns:
        A :class:`Report`, falsy if anything failed.
    """
    rep = Report()

    def record(name, fn):
        try:
            detail = fn()
        except _Skip as s:
            rep.results.append(Result(name, "skip", str(s)))
        except Exception as e:  # noqa: BLE001 — a failing check is data, not a crash
            rep.results.append(Result(name, "fail", f"{type(e).__name__}: {e}"))
        else:
            rep.results.append(Result(name, "pass", detail or ""))

    # 1. shape ---------------------------------------------------------------
    def _shapes():
        for r in ranks:
            lat = _lattice(r, time=time)
            x = _input(lat, d_model, batch, seq, seed)
            out = _build(factory, lat, d_model, seed)(x)
            if out.shape != x.shape:
                raise AssertionError(f"rank {r}: got {tuple(out.shape)}, expected {tuple(x.shape)}")
        return f"ranks {tuple(ranks)}"

    record("shape is preserved", _shapes)

    # 2. gradients -----------------------------------------------------------
    def _grads():
        lat = _lattice(min(ranks) if len(ranks) == 1 else 2, time=time)
        block = _build(factory, lat, 2, seed)
        x = _input(lat, 2, 1, seq, seed).requires_grad_(True)
        block(x).pow(2).mean().backward()
        dead = [n for n, p in block.named_parameters() if p.grad is None]
        if dead:
            raise AssertionError(f"parameters received no gradient: {dead}")
        if not torch.autograd.gradcheck(block, (x.detach().requires_grad_(True),), fast_mode=True):
            raise AssertionError("gradcheck failed")
        return f"{sum(1 for _ in block.parameters())} tensors, gradcheck clean"

    record("gradients flow and gradcheck passes", _grads)

    # 3. rank-1 equivalence --------------------------------------------------
    def _equivalence():
        if reference is None:
            raise _Skip("no `reference` given")
        lat = _lattice(1, time=time)
        block = _build(factory, lat, d_model, seed)
        x = _input(lat, d_model, batch, seq, seed)
        got, want = block(x), reference(block, x)
        if not torch.equal(got, want):
            raise AssertionError(
                f"rank-1 output differs from the reference by "
                f"{(got - want).abs().max().item():.3e} (must be exact)"
            )
        return "bitwise identical"

    record("rank-1 equals the bare 1-D module", _equivalence)

    # 4. Kronecker identity --------------------------------------------------
    record(
        "Kronecker identity (kernel family)",
        lambda: (_ for _ in ()).throw(_Skip("kernel family lands in Phase 6")),
    )

    # 5. mask invariance -----------------------------------------------------
    def _mask():
        if not sparse:
            raise _Skip("sparse=False")
        checked = 0
        for r in ranks:
            if r < 1:
                continue
            lat = _lattice(r, sparse=True, time=time, seed=seed)
            if lat.n_valid == lat.n_cells:
                continue
            block = _build(factory, lat, d_model, seed)
            x = _input(lat, d_model, batch, seq, seed)
            noise = torch.randn_like(x) * 1e3 * (~lat.mask()).to(x.dtype)
            if not torch.equal(block(x), block(x + noise)):
                raise AssertionError(
                    f"rank {r}: perturbing absent cells changed the output; they must be "
                    "zeroed before the mixer sees them"
                )
            checked += 1
        if not checked:
            raise _Skip("no sparse lattice was generated")
        return f"{checked} sparse lattices"

    record("absent cells cannot influence the output", _mask)

    # 6. permutation covariance ----------------------------------------------
    def _covariance():
        if not _accepts_plan(factory):
            raise _Skip("factory does not accept `plan`")
        r = max(ranks)
        if r < 2:
            raise _Skip("needs rank >= 2")
        names = tuple(f"ax{i}" for i in range(r))
        shape = tuple(range(2, 2 + r))
        order = tuple(range(1, r)) + (0,)  # rotate the storage order

        plan = ScanPlan.from_list(list(names))
        a = Lattice(shape=shape, names=names, time=time)
        b = Lattice(
            shape=tuple(shape[i] for i in order),
            names=tuple(names[i] for i in order),
            time=time,
        )
        x = _input(a, d_model, batch, seq, seed)
        # move each lattice dim of x into b's storage order
        lead = 2 if time else 1
        perm = (*range(lead), *(lead + i for i in order), x.ndim - 1)

        out_a = _build(factory, a, d_model, seed, plan=plan)(x)
        out_b = _build(factory, b, d_model, seed, plan=plan)(x.permute(*perm))
        if not torch.allclose(out_b, out_a.permute(*perm), rtol=0, atol=1e-12):
            raise AssertionError(
                "output depends on the order axes happen to be stored in, not just on "
                "the sweep order"
            )
        return f"rank {r}, storage order rotated"

    record("output is covariant with axis storage order", _covariance)

    # 7. compile -------------------------------------------------------------
    def _compile():
        if not check_compile:
            raise _Skip("check_compile=False")
        lat = _lattice(max(ranks), time=time)
        block = _build(factory, lat, d_model, seed)
        x = _input(lat, d_model, batch, seq, seed)
        eager = block(x)
        got = torch.compile(block)(x)
        if not torch.allclose(got, eager, rtol=1e-9, atol=1e-9):
            raise AssertionError(f"max diff {(got - eager).abs().max().item():.3e}")
        return "matches eager"

    record("torch.compile matches eager", _compile)

    if raise_on_failure and rep.failed:
        raise AssertionError("conformance check failed:\n" + str(rep))
    return rep


class _Skip(Exception):
    """Raised inside a check to record it as skipped rather than passed.

    Deliberately not silent: a skipped check appears in the report, so
    "we never ran that one" can never read as "that one passed".
    """


def check_trainable(
    factory: Factory,
    *,
    d_model: int = 16,
    steps: int = 200,
    lr: float = 1e-2,
    batch: int = 8,
    seq: int = 5,
    min_ratio: float = 3.0,
    seed: int = 0,
    raise_on_failure: bool = True,
) -> dict[str, float]:
    """Fit a small task that genuinely needs N-D mixing, and check it learns.

    Separate from :func:`check_block` on purpose. That one asks *is this
    correct* — deterministic, exact, fast. This one asks *does this learn*,
    which is stochastic, slower, and catches a different failure entirely: a
    block can have flawless gradients, pass ``gradcheck``, and still never
    converge because of initialization, masking that kills the signal, or
    activations that blow up. "No trainer in the library" must not quietly
    become "nobody ever checked that it trains".

    The task is a cumulative sum along the **last lattice axis**, so a model
    that never sweeps that axis cannot solve it — the check has a meaningful
    negative, not just a number that goes down.

    Fresh data is drawn every step and the reported score is on a held-out
    batch. With a fixed training set this test is worthless: a model with
    enough capacity memorizes eight examples without doing any axial mixing at
    all, and every plan passes.

    Returns a dict of ``initial``, ``final``, ``held_out`` and ``ratio``.
    """
    lat = Lattice(shape=(3, 4), names=("h", "w"), time=True)
    torch.manual_seed(seed)
    block = factory(lat, d_model)
    head = nn.Linear(d_model, 1)
    opt = torch.optim.Adam([*block.parameters(), *head.parameters()], lr=lr)

    def draw(g):
        x = torch.randn(batch, seq, *lat.shape, d_model, generator=g)
        return x, x[..., :1].cumsum(dim=lat.tensor_dim("w"))

    g = torch.Generator().manual_seed(seed)
    initial = final = 0.0
    for i in range(steps):
        x, y = draw(g)
        loss = (head(block(x)) - y).pow(2).mean()
        if i == 0:
            initial = loss.item()
        final = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()

    block.eval()
    with torch.no_grad():
        x, y = draw(torch.Generator().manual_seed(seed + 9973))
        held_out = (head(block(x)) - y).pow(2).mean().item()

    ratio = initial / max(held_out, 1e-12)
    result = {
        "initial": initial,
        "final": final,
        "held_out": held_out,
        "ratio": ratio,
    }
    if raise_on_failure and ratio < min_ratio:
        raise AssertionError(
            f"block did not learn: held-out loss {held_out:.4f} vs initial {initial:.4f} "
            f"({ratio:.1f}x, needed {min_ratio}x). Gradients can be correct and the "
            "block still not converge."
        )
    return result
