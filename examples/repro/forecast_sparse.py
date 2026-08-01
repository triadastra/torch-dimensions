"""Forecasting on a sparse lattice — the experiment nobody has published.

    python -m examples.repro.forecast_sparse --arm masked --arm zeros
    python -m examples.repro.forecast_sparse --arm masked --arm cafa

Air quality over Beijing: 12 monitoring stations x 6 pollutants, hourly, a
genuinely 2-D lattice (station and pollutant are different kinds of axis, not
a reshaped sequence). The task is to forecast the next hour from the previous
24 at every cell.

**Two claims get numbers here.**

*Absent cells.* Every source implementation of an N-D sequence model assumes a
full grid. When some series do not exist — a station that never measured
ozone — the only options they leave are zero-filling or dropping the axis.
This library marks them absent instead, and the claim is that the difference
matters. So: same data, same model, same budget, and the `masked` arm gets a
validity mask while the `zeros` arm gets a dense lattice with zeros in the
same positions. Both are scored on present cells only, so the comparison is
about *representation*, not about which cells count.

The sparsity is **induced**, and that is deliberate rather than a compromise:
a naturally sparse dataset confounds "is missing" with "is different", while
removing cells at random from a complete dataset isolates the one variable.
The `--drop` fraction lands in every recorded row.

*Method of multidimensionality.* The other arms swap `nd_method` and change
nothing else — `axial_scan` sweeps a mixer along each axis, `cafa` builds
per-axis kernels across space while the mixer owns time. That comparison is
one flag here and a fork of somebody's repository anywhere else, which is the
entire argument for the library.

No published baseline exists for this task in this form. These rows are the
baseline; they are reported with hardware, seeds, and the exact split, so
somebody can beat them.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn

import torch_dimensions as td

from .data import beijing
from .harness import Config, pick_device, record, sync, train_regressor

INPUT_LEN = 24
HORIZON = 1


class Forecaster(nn.Module):
    """Body over the lattice, last timestep read out, one value per cell."""

    def __init__(self, body: nn.Module, d_model: int):
        super().__init__()
        self.body = body
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, *lat, 1)
        return self.head(self.body(x)[:, -1:])  # (B, 1, *lat, 1)


def prepare(drop: float, seed: int) -> dict:
    """Standardize, impute, window, split — and choose which cells exist.

    Standardization statistics come from the training window only. Gaps in the
    real data are filled with the training mean *after* standardizing, i.e.
    with zero, and the same imputation is used by every arm so it cannot
    explain a difference between them.
    """
    raw = beijing()
    series = raw["series"]  # (T, station, pollutant), NaN where unmeasured
    n_time = series.shape[0]
    cut = int(n_time * 0.8)

    train_part = series[:cut]
    mean = torch.nanmean(train_part, dim=0)
    centered = train_part - mean
    std = torch.sqrt(torch.nanmean(centered * centered, dim=0)).clamp_min(1e-6)
    values = (series - mean) / std
    values = torch.nan_to_num(values, nan=0.0)

    shape = tuple(values.shape[1:])
    g = torch.Generator().manual_seed(seed + 777)
    keep = torch.ones(shape, dtype=torch.bool)
    if drop > 0:
        n_drop = int(round(drop * keep.numel()))
        idx = torch.randperm(keep.numel(), generator=g)[:n_drop]
        keep.reshape(-1)[idx] = False
    return {
        "values": values.unsqueeze(-1),  # (T, station, pollutant, 1)
        "keep": keep,
        "shape": shape,
        "names": raw["names"],
        "cut": cut,
    }


def windows(values: torch.Tensor, lo: int, hi: int, batch: int, device: str) -> list:
    """Fixed, non-overlapping-by-stride windows as ready-to-use batches."""
    starts = list(range(lo, hi - INPUT_LEN - HORIZON, 4))
    out = []
    for i in range(0, len(starts), batch):
        chunk = starts[i : i + batch]
        if not chunk:
            continue
        x = torch.stack([values[s : s + INPUT_LEN] for s in chunk])
        y = torch.stack([values[s + INPUT_LEN : s + INPUT_LEN + HORIZON] for s in chunk])
        out.append((x.to(device), y.to(device)))
    return out


ARMS = {
    "masked": "sparse lattice, absent cells marked",
    "zeros": "dense lattice, absent cells zero-filled",
    "cafa": "sparse lattice + td.cafa across space",
    "attention": "sparse lattice + td.axial_attention across space",
}


def build_arm(arm: str, prep: dict, d_model: int, n_layers: int):
    """One arm's model. Every arm differs in exactly one thing, named above."""
    shape, names, keep = prep["shape"], prep["names"], prep["keep"]
    valid = None if arm == "zeros" else keep
    lattice = td.Lattice(shape=shape, names=names, valid=valid, time=True)
    axes = ("time", *names)
    kw: dict = {"d_model": d_model, "lattice": lattice, "d_input": 1}
    if arm in ("cafa", "attention"):
        # The kernel family owns the spatial axes; the mixer sweeps time. The
        # plan still names every axis, because the plan is what tells the block
        # which axes are spatial.
        plan = td.ScanPlan.cyclic(axes, n_layers=n_layers, warn=False)
        method = td.cafa if arm == "cafa" else td.axial_attention
        body = td.LSTM(plan=plan, method=method, **kw)
    else:
        plan = td.ScanPlan.cyclic(axes, n_layers=n_layers, bidirectional=names, warn=False)
        body = td.LSTM(plan=plan, **kw)
    return Forecaster(body, d_model)


def make_step(prep: dict, device: str):
    """MSE over *present* cells only, identical for every arm.

    The scored set never changes with the arm — otherwise "masking helps"
    could just mean "we stopped scoring the hard cells".
    """
    mask = prep["keep"].to(device).reshape(1, 1, *prep["shape"], 1).float()
    denom = mask.sum().clamp_min(1.0)

    def step(model: nn.Module, batch) -> torch.Tensor:
        x, y = batch
        pred = model(x * mask)
        return (((pred - y) * mask) ** 2).sum() / (denom * x.shape[0])

    return step


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", action="append", choices=sorted(ARMS), help="repeatable")
    ap.add_argument("--drop", type=float, default=0.3, help="fraction of cells made absent")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--d-model", type=int, default=48)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    arms = args.arm or ["masked", "zeros"]
    device = pick_device(args.device)
    prep = prepare(args.drop, args.seed)
    values = prep["values"]
    train = windows(values, 0, prep["cut"], args.batch, device)
    test = windows(values, prep["cut"], values.shape[0], args.batch, device)
    step = make_step(prep, device)
    print(
        f"lattice {prep['shape']} {prep['names']}, "
        f"{int(prep['keep'].sum())}/{prep['keep'].numel()} cells present "
        f"(drop={args.drop}), {len(train)} train / {len(test)} test batches"
    )

    for arm in arms:

        def build(arm=arm):
            return build_arm(arm, prep, args.d_model, args.n_layers)

        if args.dry_run:
            import time

            model = build().to(device)
            step(model, train[0]).backward()
            sync(device)
            t0 = time.time()
            step(model, train[0]).backward()
            sync(device)  # without this, the number is the dispatch queue's
            n = sum(p.numel() for p in model.parameters())
            print(
                f"{arm}: {n:,} params, {time.time() - t0:.2f}s/batch "
                f"-> {(time.time() - t0) * len(train) / 60:.1f} min/epoch"
            )
            continue

        cfg = Config(
            name=f"beijing forecast · {arm}",
            task=f"air quality 12×6 lattice, {int(args.drop * 100)}% cells absent",
            model=f"{arm} ({ARMS[arm]}), d_model={args.d_model} n_layers={args.n_layers}",
            epochs=args.epochs,
            batch=args.batch,
            lr=args.lr,
            seed=args.seed,
            device=args.device,
            extra={"arm": arm, "drop": args.drop, "input_len": INPUT_LEN, "horizon": HORIZON},
        )
        record(train_regressor(build, step, train, test, cfg, log_every=200))


if __name__ == "__main__":
    main()
