"""Train the model matrix identically on one device, for comparison on another.

    python benchmarks/pretrain.py --out "MPS bench"      # on the Mac Studio
    python benchmarks/pretrain.py --out "CUDA bench"     # on the 5090

Every model is built on **CPU** under a fixed seed and only then moved to the
device, and every batch is drawn on CPU from a seeded generator. That is the
whole design: identical initial weights and identical data on both machines,
so the only thing that differs between two runs is the arithmetic. Compare the
two directories with ``benchmarks/compare.py``.

**What a CUDA-vs-MPS comparison can and cannot show.** It cannot show bitwise
agreement — different devices reduce in different orders, and cuDNN, Metal and
a pure-torch loop are three different implementations of the same formula. It
*can* show whether they agree to the precision the arithmetic allows, whether
they train to the same place, and how much faster one is.

The interesting rows are the vendored models. On CUDA they take the authors'
fused kernels; on MPS they take the reference path. So those rows are not
really "CUDA vs MPS" — they are **fused vs reference**, which is the
comparison PLAN.md fixes as the rule for any fast path: the portable path is
the reference, and the fused path must agree with it.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import init_weights
import precision
import torch
import torch.nn as nn

import torch_dimensions as td

SEED = 20260803


def device_name(device: str) -> str:
    if device == "cuda":
        return torch.cuda.get_device_name(0)
    if device == "mps":
        return f"Apple {platform.machine()} (MPS)"
    return platform.processor() or platform.machine()


def pick_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


# One recipe for every model, which is only possible because `td.param_groups`
# reads the tags the upstream authors put on their own parameters: s4's
# `_optim` and Mamba's `_no_weight_decay`. Training an SSM's `A` and `dt` at
# the projection rate with weight decay is what made the earlier runs diverge
# at 1e-2 and forced hand-picked per-family rates — which were themselves a
# confound, since a device comparison should hold everything it is not
# measuring fixed. See BENCHMARK-DESIGN.md.
LR = 3e-3
BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
WARMUP_FRACTION = 0.1


# --- the matrix ------------------------------------------------------------
# Small on purpose: the point is coverage of every family and composition, not
# a leaderboard. Each model has to fit the same task so the losses are
# comparable, and stay small enough that fifteen of them are minutes rather
# than hours — and that the checkpoints are a sane upload.


def sparse_2d() -> td.Lattice:
    gen = torch.Generator().manual_seed(11)
    valid = torch.rand(6, 8, generator=gen) > 0.25
    valid[0, 0] = True
    return td.Lattice(shape=(6, 8), names=("h", "w"), valid=valid, time=True)


def dense_3d() -> td.Lattice:
    return td.Lattice(shape=(4, 5, 6), names=("d", "h", "w"), time=True)


ZOO: dict[str, dict] = {
    # --- recurrent
    "lstm_2d_sparse": {"build": lambda lat: td.LSTM(32, 6, lat, d_input=1), "lat": sparse_2d},
    "gru_2d_sparse": {"build": lambda lat: td.GRU(32, 6, lat, d_input=1), "lat": sparse_2d},
    # --- state space, portable vs the authors' own code
    "s4d_portable_2d": {
        "build": lambda lat: td.S4D(32, 4, lat, d_input=1, portable=True, d_state=16),
        "lat": sparse_2d,
    },
    "s4d_upstream_2d": {
        "build": lambda lat: td.S4D(32, 4, lat, d_input=1, d_state=16),
        "lat": sparse_2d,
    },
    "s4_upstream_2d": {
        "build": lambda lat: td.S4(32, 4, lat, d_input=1, d_state=16),
        "lat": sparse_2d,
    },
    "mamba_portable_2d": {
        "build": lambda lat: td.Mamba(32, 4, lat, d_input=1, portable=True, d_state=8),
        "lat": sparse_2d,
    },
    "mamba_upstream_2d": {
        "build": lambda lat: td.Mamba(32, 4, lat, d_input=1, d_state=8),
        "lat": sparse_2d,
    },
    "mamba2_2d": {
        "build": lambda lat: td.Mamba2(
            64, 4, lat, d_input=1, mixer_kwargs={"d_state": 16, "headdim": 32}
        ),
        "lat": sparse_2d,
    },
    "mamba3_2d": {
        "build": lambda lat: td.Mamba3(
            64, 4, lat, d_input=1, mixer_kwargs={"d_state": 64, "headdim": 32}
        ),
        "lat": sparse_2d,
    },
    # --- attention, across all three compositions
    "transformer_scan_2d": {
        "build": lambda lat: td.Transformer(32, 4, lat, d_input=1),
        "lat": sparse_2d,
    },
    "transformer_cafa_2d": {
        "build": lambda lat: td.Transformer(32, 4, lat, d_input=1, method=td.cafa),
        "lat": sparse_2d,
    },
    "transformer_flatten_2d": {
        "build": lambda lat: td.Transformer(32, 4, lat, d_input=1, method=td.flatten),
        "lat": sparse_2d,
    },
    # --- convolutional
    "cnn_2d_sparse": {"build": lambda lat: td.CNN(32, 4, lat, d_input=1), "lat": sparse_2d},
    "tcn_2d_sparse": {"build": lambda lat: td.TCN(32, 4, lat, d_input=1), "lat": sparse_2d},
    # --- rank 3, where the composition has more to do
    "mamba_upstream_3d": {
        "build": lambda lat: td.Mamba(32, 6, lat, d_input=1, d_state=8),
        "lat": dense_3d,
    },
    "lstm_3d": {"build": lambda lat: td.LSTM(32, 6, lat, d_input=1), "lat": dense_3d},
}


def train_one(
    name: str,
    cfg: dict,
    device: str,
    steps: int,
    batch: int,
    t_len: int,
    init: Path | None = None,
) -> dict:
    """Build on CPU under a fixed seed, move, train, and record.

    Building on CPU first is most of what makes two machines comparable. It is
    not all of it: S4 and S4D diagonalise HiPPO with `torch.linalg.eigh`, whose
    eigenvectors are only defined up to a phase, so their `B` and `P` differ
    between macOS and Linux under the same seed. Pass `--init` to share one set
    of starting weights and remove that difference from the measurement — see
    benchmarks/init_weights.py.
    """
    lat = cfg["lat"]()
    torch.manual_seed(SEED)
    model = cfg["build"](lat)
    head = nn.Linear(model.config["d_model"], 1)
    weights_from = init_weights.sync(model, init, name)
    init_weights.sync(head, init, f"{name}.head")

    # On CPU in float64: MPS has no float64 at all, and a norm computed in the
    # device's own precision would differ between machines for that reason
    # rather than for the reason being measured.
    def weight_norm(m: nn.Module) -> float:
        total = sum((p.detach().cpu().double() ** 2).sum() for p in m.parameters())
        return float(torch.sqrt(total))

    init_norm = weight_norm(model)
    n_params = sum(p.numel() for p in model.parameters())

    model, head = model.to(device), head.to(device)
    lr = cfg.get("lr", LR)
    opt = torch.optim.AdamW(
        td.param_groups(model, lr=lr, weight_decay=WEIGHT_DECAY)
        + [{"params": list(head.parameters()), "lr": lr, "weight_decay": WEIGHT_DECAY}],
        lr=lr,
        betas=BETAS,
    )
    sched = td.warmup_cosine(opt, warmup=max(1, int(steps * WARMUP_FRACTION)), total=steps)

    mask = lat.mask(torch.float32).to(device)
    w_dim = lat.tensor_dim(lat.axis_names[-1])
    gen = torch.Generator().manual_seed(SEED + 1)

    def draw():
        # Drawn on CPU so both machines see the same numbers, then moved.
        x = torch.randn(batch, t_len, *lat.shape, 1, generator=gen)
        return x.to(device) * mask, x.to(device).cumsum(dim=w_dim) * mask

    losses: list[float] = []
    sync(device)
    started = time.perf_counter()
    for _ in range(steps):
        x, y = draw()
        loss = (head(model(x)) - y).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([*model.parameters(), *head.parameters()], GRAD_CLIP)
        opt.step()
        sched.step()
        losses.append(float(loss.detach()))
    sync(device)
    seconds = time.perf_counter() - started

    # A fixed probe batch, so the two machines can be compared on outputs and
    # not only on losses. Same input on both, by construction.
    probe_gen = torch.Generator().manual_seed(SEED + 2)
    probe = torch.randn(1, t_len, *lat.shape, 1, generator=probe_gen).to(device) * mask
    model.eval()
    with torch.no_grad():
        out = head(model(probe)).float().cpu()
    model.train()

    return (
        {
            "name": name,
            "n_params": n_params,
            "weights_from": weights_from,
            "lr": lr,
            "recipe": "adamw+param_groups+warmup_cosine+clip",
            "steps": steps,
            "init_weight_norm": init_norm,
            "final_weight_norm": weight_norm(model),
            "loss_first": losses[0],
            "loss_final": losses[-1],
            "loss_min": min(losses),
            "losses": [round(v, 8) for v in losses],
            "seconds": round(seconds, 3),
            "steps_per_second": round(steps / seconds, 3),
            "probe_mean": float(out.mean()),
            "probe_absmax": float(out.abs().max()),
            "probe_sum": float(out.double().sum()),
            "mixer": type(model.nd.mixers[0]).__name__
            if getattr(model.nd, "mixers", None)
            else None,
        },
        model,
        head,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help='output directory, e.g. "MPS bench"')
    ap.add_argument("--device", default=None, help="cuda | mps | cpu (default: best available)")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--t-len", type=int, default=6)
    ap.add_argument("--only", default=None, help="comma-separated subset of model names")
    init_weights.add_argument(ap)
    precision.add_arguments(ap, tf32_default="torch")
    args = ap.parse_args()

    device = pick_device(args.device)
    prec = precision.apply(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    init = Path(args.init) if args.init else None

    names = args.only.split(",") if args.only else list(ZOO)
    print(f"device: {device} ({device_name(device)})")
    print(f"models: {len(names)}   steps: {args.steps}")
    print(f"weights: {init if init else 'from the seed (see --init)'}")
    print(f"{precision.describe(prec)}\n")

    results = []
    for i, name in enumerate(names, 1):
        cfg = ZOO[name]
        print(f"[{i:2d}/{len(names)}] {name:26s} ", end="", flush=True)
        try:
            record, model, head = train_one(
                name, cfg, device, args.steps, args.batch, args.t_len, init
            )
        except Exception as exc:  # noqa: BLE001 - one model failing is a result
            print(f"FAILED — {type(exc).__name__}: {exc}")
            results.append({"name": name, "error": f"{type(exc).__name__}: {exc}"})
            continue

        model_dir = out / name
        model_dir.mkdir(exist_ok=True)
        # Weights on CPU so the file is identical in layout across devices and
        # loadable anywhere.
        torch.save(
            {
                "model": {k: v.cpu() for k, v in model.state_dict().items()},
                "head": {k: v.cpu() for k, v in head.state_dict().items()},
            },
            model_dir / "weights.pt",
        )
        (model_dir / "metrics.json").write_text(json.dumps(record, indent=2) + "\n")
        results.append(record)
        print(
            f"loss {record['loss_first']:.4f} -> {record['loss_final']:.5f}   "
            f"{record['steps_per_second']:6.1f} steps/s   {record['n_params']:,} params"
        )

    manifest = {
        "device": device,
        "device_name": device_name(device),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_dimensions": getattr(td, "__version__", "unknown"),
        "seed": SEED,
        "precision": prec,
        "steps": args.steps,
        "batch": args.batch,
        "t_len": args.t_len,
        "models": results,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    ok = sum(1 for r in results if "error" not in r)
    print(f"\nwrote {out}/manifest.json — {ok}/{len(results)} models trained")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
