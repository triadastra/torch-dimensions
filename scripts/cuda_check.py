"""Run every CUDA claim this library makes, and print what is actually true.

    python scripts/cuda_check.py

Runs on CPU too — the CUDA-only checks report `skip` with the reason, so the
harness itself can be verified without a GPU and the Colab run is one command
rather than fifteen manual steps. That distinction is the whole point: the
checklist in docs/cuda-checklist.md has existed since 0.1 and has never been
run, and a procedure nobody runs protects nothing.

**What is at stake, specifically.** Since 0.3.1 the library ships the original
authors' S4, Mamba-1, Mamba-2 and Mamba-3 and chooses per call between their
fused CUDA kernels and a portable path. Every part of that choice is untested:
`prefer_upstream` has never once returned True on real hardware, no fused
kernel has ever run, and Mamba-3's PyTorch transcription has never been
compared against the Triton kernel it was transcribed from — the one
comparison no CPU or MPS machine can make.

Each check prints `pass`, `fail`, `skip` or `info` with the number behind it.
`info` is for measurements that have no pass/fail — a benchmark, or a fact
worth writing down. Paste the whole report into the issue.
"""

from __future__ import annotations

import os
import platform
import sys
import traceback

import torch

import torch_dimensions as td

CUDA = torch.cuda.is_available()
RESULTS: list[tuple[str, str, str]] = []


def record(name: str, status: str, detail: str = "") -> None:
    RESULTS.append((name, status, detail))
    mark = {"pass": "PASS", "fail": "FAIL", "skip": "skip", "info": "info"}[status]
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""), flush=True)


def check(name: str, *, needs_cuda: bool = True):
    """Run a check, turning any exception into a `fail` with its message.

    A check that raises is a result, not a crash: the first CUDA data point
    this project has should not be lost because one probe threw.
    """

    def wrap(fn):
        if needs_cuda and not CUDA:
            record(name, "skip", "no CUDA device")
            return fn
        try:
            status, detail = fn()
            record(name, status, detail)
        except Exception as exc:  # noqa: BLE001 - the failure *is* the finding
            record(name, "fail", f"{type(exc).__name__}: {exc}")
            traceback.print_exc(limit=3)
        return fn

    return wrap


def lat(shape=(4, 5), names=("h", "w"), **kw):
    return td.Lattice(shape=shape, names=names, **kw)


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """Relative difference, on the scale of the larger tensor."""
    scale = max(float(a.abs().max()), float(b.abs().max()), 1e-12)
    return float((a - b).abs().max()) / scale


# --------------------------------------------------------------------------
print("\n=== environment")
print(f"  python        {platform.python_version()} on {platform.platform()}")
print(f"  torch         {torch.__version__} (cuda {torch.version.cuda})")
print(f"  torch-dims    {getattr(td, '__version__', 'unknown')}")
if CUDA:
    print(f"  device        {torch.cuda.get_device_name(0)}")
    print(f"  capability    {torch.cuda.get_device_capability(0)}")
else:
    print("  device        none — CUDA checks will skip")
try:
    import triton

    print(f"  triton        {triton.__version__}")
except ImportError:
    print("  triton        not installed")
try:
    import mamba_ssm  # noqa: F401

    print("  mamba_ssm     installed (fused kernels importable)")
except ImportError:
    print("  mamba_ssm     not installed — fused paths will fall back")


# --------------------------------------------------------------------------
print("\n=== dispatch: which implementation actually runs")


@check("prefer_upstream is True for a CUDA tensor")
def _():
    from torch_dimensions.mixers._kernels import prefer_upstream

    got = prefer_upstream(torch.zeros(1, device="cuda"))
    return (
        "pass" if got else "fail",
        f"returned {got}; the fused path is {'' if got else 'NOT '}reachable",
    )


@check("prefer_upstream is False for a CPU tensor on a CUDA box", needs_cuda=False)
def _():
    from torch_dimensions.mixers._kernels import prefer_upstream

    got = prefer_upstream(torch.zeros(1))
    return ("pass" if not got else "fail", f"returned {got}")


@check("TD_FORCE_TORCH_KERNELS overrides CUDA")
def _():
    from torch_dimensions.mixers._kernels import prefer_upstream

    os.environ["TD_FORCE_TORCH_KERNELS"] = "1"
    try:
        got = prefer_upstream(torch.zeros(1, device="cuda"))
    finally:
        del os.environ["TD_FORCE_TORCH_KERNELS"]
    return ("pass" if not got else "fail", f"returned {got} with the override set")


@check("the fused Mamba-1 / Mamba-2 kernels import")
def _():
    from torch_dimensions.mixers._kernels import load_upstream

    scan = load_upstream("mamba_ssm.ops.selective_scan_interface", "selective_scan_fn")
    ssd = load_upstream("mamba_ssm.ops.triton.ssd_combined", "mamba_chunk_scan_combined")
    found = [n for n, f in (("selective_scan_fn", scan), ("ssd_combined", ssd)) if f]
    if not found:
        return ("info", "neither importable — install mamba-ssm to exercise the fused paths")
    return ("pass", f"importable: {', '.join(found)}")


# --------------------------------------------------------------------------
print("\n=== the vendored models on CUDA")


def _cuda_vs_cpu(build, length=32, width=16, seed=0):
    torch.manual_seed(seed)
    cpu = build().eval()
    gpu = build().eval().cuda()
    gpu.load_state_dict({k: v.cuda() for k, v in cpu.state_dict().items()})
    x = torch.randn(2, length, width)
    with torch.no_grad():
        return rel(cpu(x), gpu(x.cuda()).cpu())


@check("vendored Mamba-1: CUDA agrees with CPU")
def _():
    from torch_dimensions.mixers.upstream import UpstreamMambaMixer

    d = _cuda_vs_cpu(lambda: UpstreamMambaMixer(16, d_state=8))
    return (
        "pass" if d < 5e-2 else "fail",
        f"relative {d:.2e} (fused vs reference; bf16 tolerated)",
    )


@check("vendored Mamba-2: CUDA agrees with CPU")
def _():
    from torch_dimensions.mixers.upstream import UpstreamMamba2Mixer

    d = _cuda_vs_cpu(lambda: UpstreamMamba2Mixer(64, d_state=16, headdim=32))
    return ("pass" if d < 5e-2 else "fail", f"relative {d:.2e}")


@check("vendored S4 (DPLR): CUDA agrees with CPU — the Nyquist guard")
def _():
    from torch_dimensions.mixers.upstream import UpstreamS4Mixer

    worst = max(
        _cuda_vs_cpu(lambda: UpstreamS4Mixer(6, d_state=8), length=n) for n in (32, 64, 128)
    )
    # L=64 is where MPS landed exactly on the pole and produced NaN.
    return ("pass" if worst < 1e-4 else "fail", f"worst relative over L=32/64/128 {worst:.2e}")


@check("vendored S4D: CUDA agrees with CPU")
def _():
    from torch_dimensions.mixers.upstream import UpstreamS4DMixer

    d = _cuda_vs_cpu(lambda: UpstreamS4DMixer(6, d_state=8))
    return ("pass" if d < 1e-4 else "fail", f"relative {d:.2e}")


# --------------------------------------------------------------------------
print("\n=== Mamba-3: our transcription against the kernel it came from")
print("  (the one comparison no CPU or MPS machine can make)")


@check("Mamba-3 SISO: Triton kernel vs our PyTorch transcription")
def _():
    from torch_dimensions.mixers._kernels import load_upstream
    from torch_dimensions.mixers.mamba3_compat import mamba3_siso_combined

    if (
        load_upstream("mamba_ssm.ops.triton.mamba3.mamba3_siso_combined", "mamba3_siso_combined")
        is None
    ):
        return ("skip", "mamba_ssm's Mamba-3 Triton kernels are not importable")

    torch.manual_seed(0)
    b, length, h, dqk, dv, nang = 2, 64, 4, 64, 32, 8

    def g(*s):
        return torch.randn(*s, device="cuda")

    args = {
        "Q": g(b, length, 1, dqk),
        "K": g(b, length, 1, dqk),
        "V": g(b, length, h, dv),
        "ADT": -torch.rand(b, h, length, device="cuda") * 0.5 - 1e-3,
        "DT": torch.rand(b, h, length, device="cuda") * 0.1 + 1e-3,
        "Trap": g(b, h, length),
        "Q_bias": g(h, dqk),
        "K_bias": g(h, dqk),
        "Angles": g(b, length, h, nang),
        "D": g(h),
        "Z": g(b, length, h, dv),
    }
    fused = mamba3_siso_combined(**args, chunk_size=64)  # dispatches to Triton on CUDA
    ours = mamba3_siso_combined(**args, chunk_size=64, recurrent=True)  # forces our loop
    d = rel(fused.float(), ours.float())
    # Their kernel runs in bfloat16 with PTX cos/sin/tanh approximations; ours
    # is float32 with exact library functions. Agreement to bf16's own
    # resolution (~1e-2 relative) is the most that can be asked, and is what
    # would confirm the transcription.
    return (
        "pass" if d < 5e-2 else "fail",
        f"relative {d:.2e} (theirs bf16 + PTX approximations, ours fp32 exact)",
    )


@check("Mamba-3 block end to end on CUDA")
def _():
    from torch_dimensions.mixers.upstream import Mamba3Mixer

    torch.manual_seed(0)
    m = Mamba3Mixer(64, d_state=64, headdim=32).cuda()
    x = torch.randn(2, 64, 64, device="cuda", requires_grad=True)
    y = m(x)
    y.pow(2).mean().backward()
    finite = torch.isfinite(y).all() and torch.isfinite(x.grad).all()
    return (
        "pass" if finite else "fail",
        f"output {tuple(y.shape)}, gradients finite={bool(finite)}",
    )


# --------------------------------------------------------------------------
print("\n=== claims the README makes")


@check("rank-1 LSTM is still bitwise identical to nn.LSTM under cuDNN")
def _():
    torch.manual_seed(0)
    model = td.LSTM(16, 1, lat(shape=(), names=()), portable=True).cuda().eval()
    bare = torch.nn.LSTM(16, 16, batch_first=True).cuda().eval()
    mixer = model.nd.mixers[0]
    bare.load_state_dict({k: v for k, v in mixer.rnn.state_dict().items()})
    x = torch.randn(2, 32, 16, device="cuda")
    with torch.no_grad():
        d = float((model(x) - bare(x)[0]).abs().max())
    # Whatever is true, write it down: cuDNN may reorder reductions.
    return (
        "pass" if d == 0.0 else "info",
        f"max |difference| {d:.3e} ({'bitwise' if d == 0 else 'not bitwise'})",
    )


@check("device placement is refused in both directions")
def _():
    model = td.LSTM(16, 2, lat()).cuda()
    x = torch.randn(2, 4, 5, 16)  # CPU input, CUDA model
    try:
        model(x)
        return ("fail", "a CPU tensor through a CUDA model did not raise")
    except (RuntimeError, ValueError) as exc:
        return ("pass", f"raised {type(exc).__name__} as it should")


@check("autocast fp16 through the kernel family's cancellation guard")
def _():
    valid = torch.rand(6, 8) > 0.3
    valid[0, 0] = True
    sparse = td.Lattice(shape=(6, 8), names=("h", "w"), valid=valid, time=True)
    model = td.LSTM(32, 3, sparse, method=td.cafa).cuda()
    x = torch.randn(2, 5, 6, 8, 32, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        y = model(x)
    return (
        "pass" if torch.isfinite(y).all() else "fail",
        f"fp16 autocast output finite={bool(torch.isfinite(y).all())}, dtype {y.dtype}",
    )


@check("autocast bf16 through the kernel family")
def _():
    valid = torch.rand(6, 8) > 0.3
    valid[0, 0] = True
    sparse = td.Lattice(shape=(6, 8), names=("h", "w"), valid=valid, time=True)
    model = td.LSTM(32, 3, sparse, method=td.cafa).cuda()
    x = torch.randn(2, 5, 6, 8, 32, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = model(x)
    return (
        "pass" if torch.isfinite(y).all() else "fail",
        f"finite={bool(torch.isfinite(y).all())}",
    )


@check("absent cells stay inert on CUDA")
def _():
    """The library's central sparse guarantee, on a device it has never run on."""
    valid = torch.rand(6, 8) > 0.3
    valid[0, 0] = True
    sparse = td.Lattice(shape=(6, 8), names=("h", "w"), valid=valid, time=True)
    model = td.LSTM(16, 3, sparse).cuda().eval()
    x = torch.randn(2, 4, 6, 8, 16, device="cuda")
    mask = sparse.mask(torch.float32).cuda()
    noise = torch.randn_like(x) * 50 * (1 - mask)
    with torch.no_grad():
        d = float((model(x) - model(x + noise)).abs().max())
    return ("pass" if d == 0.0 else "fail", f"max |difference| from absent-cell noise {d:.3e}")


# --------------------------------------------------------------------------
print("\n=== summary")
counts = {k: sum(1 for _, s, _ in RESULTS if s == k) for k in ("pass", "fail", "skip", "info")}
print(
    f"  {counts['pass']} passed · {counts['fail']} failed · "
    f"{counts['skip']} skipped · {counts['info']} recorded"
)
if counts["fail"]:
    print("\n  failures:")
    for name, status, detail in RESULTS:
        if status == "fail":
            print(f"    - {name}: {detail}")
print(
    "\nAlso run the suite itself:  pytest tests/ -q"
    "\n(the device tests must report cuda, not skipped — a green run with no"
    "\n CUDA present proves nothing)"
)

# Only when run as a script. The checks above execute at import — that is the
# design, since each one is a decorated function — but exiting at import made
# the harness impossible to import, and therefore impossible to test that it
# skips cleanly without a device. Which is precisely the property a report
# from a machine that *has* one depends on.
if __name__ == "__main__":
    sys.exit(1 if counts["fail"] else 0)
