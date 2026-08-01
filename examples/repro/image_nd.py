"""Image classification over a 2-D lattice — the N-D machinery under load.

    python -m examples.repro.image_nd --model mamba_nd --epochs 5
    python -m examples.repro.image_nd --dataset cifar10 --model s4d_nd --epochs 10

Where `smnist.py` deliberately removes the lattice (a sequence is a lattice
with no spatial axes), this puts it back: an image is a 2-D lattice, the model
sweeps rows and columns with the schedule Mamba-ND describes, and no pixel is
ever flattened into a sequence. Same mixer, same library, one argument
different — which is the claim the whole project rests on.

**On scale.** The papers this echoes train for hundreds of epochs on datacenter
GPUs. These runs are small and are reported as small: RESULTS.md carries the
epoch count, the wall clock, and the hardware, and the number is this
configuration's number. What is being reproduced here is the *construction* —
that the paired schedule over a 2-D lattice learns images — not a leaderboard
entry.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn

import torch_dimensions as td

from .data import cifar10, mnist
from .harness import Config, record, train_classifier

DATASETS = {
    "mnist": (mnist, (28, 28), 1),
    "cifar10": (cifar10, (32, 32), 3),
}


class LatticeClassifier(nn.Module):
    """An N-D body, mean-pooled over the lattice, classified.

    Pooling over *present* cells only — on a dense image that is every cell,
    but writing it the other way would silently break the moment someone hands
    this a sparse lattice, and that is the case the library exists for.
    """

    def __init__(self, body: nn.Module, lattice: td.Lattice, d_model: int, n_classes: int):
        super().__init__()
        self.body = body
        self.lattice = lattice
        self.register_buffer("cell_mask", lattice.mask(torch.float32), persistent=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, H, W, C) -> (B, classes)
        h = self.body(x) * self.cell_mask
        dims = tuple(range(1, h.ndim - 1))
        pooled = h.sum(dims) / self.cell_mask.sum().clamp_min(1.0)
        return self.head(self.norm(pooled))


def build_model(name: str, shape, d_input: int, d_model: int, n_layers: int, n_classes: int):
    lattice = td.Lattice(shape=shape, names=("row", "col"))
    kw = dict(d_model=d_model, lattice=lattice, d_input=d_input)
    if name == "mamba_nd":
        # The official Mamba-ND schedule: each axis forward then immediately
        # backward, ordering advancing every two layers.
        plan = td.ScanPlan.paired(("row", "col"), n_layers=n_layers, bidirectional=True)
        body = td.Mamba(plan=plan, d_state=16, **kw)
    elif name == "s4d_nd":
        plan = td.ScanPlan.cyclic(("row", "col"), n_layers=n_layers, bidirectional=True)
        body = td.S4D(plan=plan, **kw)
    elif name == "lstm":
        plan = td.ScanPlan.cyclic(("row", "col"), n_layers=n_layers, bidirectional=True)
        body = td.LSTM(plan=plan, **kw)
    else:
        raise ValueError(f"unknown model {name!r}")
    return LatticeClassifier(body, lattice, d_model, n_classes)


def build_data(dataset: str) -> dict[str, torch.Tensor]:
    load, shape, channels = DATASETS[dataset]
    raw = load()
    out = {}
    for split in ("train", "test"):
        x = raw[f"{split}_x"].float() / 255.0
        x = x.reshape(-1, *shape, channels)
        # Per-channel standardization from the training split only.
        if split == "train":
            mean, std = x.mean((0, 1, 2)), x.std((0, 1, 2)).clamp_min(1e-6)
        out[f"{split}_x"] = (x - mean) / std
        out[f"{split}_y"] = raw[f"{split}_y"].long()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="mnist")
    ap.add_argument("--model", choices=["mamba_nd", "s4d_nd", "lstm"], default="mamba_nd")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    _, shape, channels = DATASETS[args.dataset]
    data = build_data(args.dataset)

    def build() -> nn.Module:
        return build_model(args.model, shape, channels, args.d_model, args.n_layers, 10)

    cfg = Config(
        name=f"{args.dataset} 2-D · {args.model}",
        task=f"{args.dataset} (2-D lattice {shape[0]}×{shape[1]})",
        model=f"td.{args.model} d_model={args.d_model} n_layers={args.n_layers}",
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        limit_train=args.limit_train,
        extra={"dataset": args.dataset, "arch": args.model, "n_layers": args.n_layers},
    )

    if args.dry_run:
        import time

        from .harness import pick_device, sync

        device = pick_device(args.device)
        model = build().to(device)
        xb = data["train_x"][: args.batch].to(device)
        yb = data["train_y"][: args.batch].to(device)
        for _ in range(2):  # warm the kernels before timing
            nn.functional.cross_entropy(model(xb), yb).backward()
        sync(device)
        t0 = time.time()
        nn.functional.cross_entropy(model(xb), yb).backward()
        sync(device)  # without this, the number is the dispatch queue's
        per_step = time.time() - t0
        n = sum(p.numel() for p in model.parameters())
        steps = (data["train_x"].shape[0] + args.batch - 1) // args.batch
        print(
            f"{n:,} params on {device}; {per_step:.2f}s/step "
            f"-> {per_step * steps / 60:.1f} min/epoch"
        )
        return

    record(train_classifier(build, data, cfg))


if __name__ == "__main__":
    main()
