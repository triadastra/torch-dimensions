"""Bench A — do two devices compute the same thing? No training involved.

    python benchmarks/agreement.py --out "MPS agree"
    python benchmarks/compare_agreement.py "MPS agree" "CUDA agree"

The training benchmark cannot answer this. When two devices' losses differ at
step 300, the optimiser has been amplifying whatever the arithmetic did at
step 1 for 299 steps, and nothing in the run separates the two. Here there is
no optimiser: fixed weights, one forward, one backward, and the numbers are
the arithmetic.

That makes this the benchmark for the rule PLAN.md fixes for fast paths —
*the portable path is the reference and the fused path must agree with it* —
because on CUDA the vendored models take the authors' fused kernels while the
same weights on MPS take the reference path.

Both float32 and float64 are measured where the device supports them. float64
is the control: it has ~1e-16 of headroom instead of ~1e-7, so a difference
that shrinks with precision is float non-associativity, and one that does not
is a different computation. MPS has no float64 at all, which is itself
recorded rather than skipped silently.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch

import torch_dimensions as td

SEED = 20260803


def pick_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_name(device: str) -> str:
    if device == "cuda":
        return torch.cuda.get_device_name(0)
    if device == "mps":
        return f"Apple {platform.machine()} (MPS)"
    return platform.processor() or platform.machine()


def probe(name: str, build, lat, device: str, dtype: torch.dtype) -> dict:
    """One forward and one backward from fixed weights, recorded exactly.

    Everything is created on CPU and moved, so the *inputs* to the arithmetic
    are identical on every device by construction and only the arithmetic can
    differ.
    """
    torch.manual_seed(SEED)
    model = build(lat)
    torch.manual_seed(SEED + 1)
    x = torch.randn(2, 3, *lat.shape, 1, dtype=torch.float32)

    model = model.to(device=device, dtype=dtype)
    x = x.to(device=device, dtype=dtype).requires_grad_(True)

    out = model(x)
    # A fixed, non-random scalar: a random loss weighting would introduce a
    # second source of difference between runs.
    loss = out.square().mean()
    loss.backward()

    grads = {
        n: p.grad.detach().float().cpu() for n, p in model.named_parameters() if p.grad is not None
    }
    return {
        "output": out.detach().float().cpu(),
        "loss": float(loss.detach()),
        "input_grad": x.grad.detach().float().cpu(),
        "grads": grads,
    }


def summarise(result: dict) -> dict:
    """JSON-able fingerprints. Full tensors go to a side file; these are what a
    human reads, and are enough to notice a gross disagreement on their own."""
    return {
        "loss": result["loss"],
        "output_sum": float(result["output"].double().sum()),
        "output_absmax": float(result["output"].abs().max()),
        "input_grad_sum": float(result["input_grad"].double().sum()),
        "grad_norm": float(
            torch.sqrt(sum((g.double() ** 2).sum() for g in result["grads"].values()))
        )
        if result["grads"]
        else 0.0,
        "n_grads": len(result["grads"]),
    }


if __name__ == "__main__":
    # The zoo lives in pretrain.py; import it by path so the two benchmarks
    # cannot drift apart on which models they cover.
    import importlib.util
    import sys

    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("_pretrain", here / "pretrain.py")
    pretrain = importlib.util.module_from_spec(spec)
    sys.modules["_pretrain"] = pretrain
    spec.loader.exec_module(pretrain)

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--only", default=None)
    args = ap.parse_args()

    device = pick_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dtypes = [("float32", torch.float32)]
    if device != "mps":
        dtypes.append(("float64", torch.float64))
    else:
        print("note: MPS has no float64; the precision control is unavailable here")

    names = args.only.split(",") if args.only else list(pretrain.ZOO)
    print(f"device: {device} ({device_name(device)})\nmodels: {len(names)}\n")

    records: dict[str, dict] = {}
    for i, name in enumerate(names, 1):
        cfg = pretrain.ZOO[name]
        print(f"[{i:2d}/{len(names)}] {name:26s} ", end="", flush=True)
        entry: dict = {}
        for label, dtype in dtypes:
            try:
                result = probe(name, cfg["build"], cfg["lat"](), device, dtype)
                entry[label] = summarise(result)
                torch.save(
                    {"output": result["output"], "grads": result["grads"]},
                    out / f"{name}.{label}.pt",
                )
            except Exception as exc:  # noqa: BLE001 - a dtype a model cannot take is a result
                entry[label] = {"error": f"{type(exc).__name__}: {exc}"}
        records[name] = entry
        f32 = entry.get("float32", {})
        print(f"loss {f32.get('loss', float('nan')):.6f}" if "loss" in f32 else "failed")

    (out / "agreement.json").write_text(
        json.dumps(
            {
                "device": device,
                "device_name": device_name(device),
                "torch": torch.__version__,
                "torch_dimensions": getattr(td, "__version__", "unknown"),
                "seed": SEED,
                "dtypes": [label for label, _ in dtypes],
                "models": records,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {out}/agreement.json")
