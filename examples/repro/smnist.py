"""Sequential and permuted-sequential MNIST with ``td.S4D``.

    python -m examples.repro.smnist --epochs 10
    python -m examples.repro.smnist --permuted --epochs 10

The first reproduction, chosen first because it isolates the mixer from the
N-D machinery: the lattice here is the degenerate one — no spatial axes, just
time — so what is under test is whether the portable S4D kernel learns the
long-range task the S4D paper reports, at a scale one laptop can run.

The task: an MNIST digit is delivered one pixel at a time, 784 steps, and the
classification happens at the end. Permuted mode applies one fixed permutation
to the pixel order (seeded, so every run and every model sees the same one),
which destroys locality and leaves only genuinely long-range structure — the
version that separates state-space models from convolutions.

Published reference points: the S4/S4D papers report ~99.6% on sMNIST and
~98.5% on psMNIST with ~100k-parameter models trained far longer than this.
Rows land in RESULTS.md with epochs and hardware attached; a number here is
this configuration's number, not a claim to have matched a paper's budget.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn

import torch_dimensions as td

from .data import mnist
from .harness import Config, record, train_classifier


class SequenceClassifier(nn.Module):
    """Model, pooled over time, classified. The pooling is the whole head.

    Mean pooling rather than last-step readout on purpose: the last step of a
    causal model has seen everything, but its representation is dominated by
    the most recent inputs, and on a 784-step sequence that is the bottom rows
    of the image. Pooling is what the S4 classification experiments use.
    """

    def __init__(self, d_model: int, n_layers: int, n_classes: int, d_input: int = 1, **kw):
        super().__init__()
        lattice = td.Lattice(shape=(), time=True)  # a sequence is a lattice with no space
        self.body = td.S4D(d_model, n_layers, lattice, d_input=d_input, **kw)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, d_input) -> (B, classes)
        return self.head(self.norm(self.body(x).mean(dim=1)))


def build_data(permuted: bool, seed: int = 1234) -> dict[str, torch.Tensor]:
    """MNIST as (N, 784, 1) float sequences in [0, 1], optionally permuted."""
    raw = mnist()
    perm = None
    if permuted:
        # One permutation for every split and every run: a per-run permutation
        # would make two rows in RESULTS.md two different tasks.
        perm = torch.randperm(784, generator=torch.Generator().manual_seed(seed))
    out = {}
    for split in ("train", "test"):
        x = raw[f"{split}_x"].reshape(-1, 784, 1).float() / 255.0
        if perm is not None:
            x = x[:, perm]
        out[f"{split}_x"] = x
        out[f"{split}_y"] = raw[f"{split}_y"].long()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--permuted", action="store_true", help="psMNIST: fixed pixel permutation")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--d-state", type=int, default=64)
    ap.add_argument("--lr", type=float, default=4e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit-train", type=int, default=None, help="smoke runs only")
    ap.add_argument("--dry-run", action="store_true", help="build, time one step, exit")
    args = ap.parse_args()

    task = "psMNIST" if args.permuted else "sMNIST"
    cfg = Config(
        name=f"{task} · td.S4D",
        task=task,
        model=f"td.S4D d_model={args.d_model} n_layers={args.n_layers} d_state={args.d_state}",
        epochs=args.epochs,
        batch=args.batch,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        limit_train=args.limit_train,
        extra={"d_model": args.d_model, "n_layers": args.n_layers, "d_state": args.d_state},
    )

    def build() -> nn.Module:
        return SequenceClassifier(args.d_model, args.n_layers, 10, d_state=args.d_state)

    data = build_data(args.permuted)
    if args.dry_run:
        import time

        device = cfg.device if cfg.device != "auto" else "cpu"
        model = build().to(device)
        xb = data["train_x"][: args.batch].to(device)
        yb = data["train_y"][: args.batch].to(device)
        t0 = time.time()
        nn.functional.cross_entropy(model(xb), yb).backward()
        n = sum(p.numel() for p in model.parameters())
        print(f"one step: {time.time() - t0:.2f}s on {device}, {n:,} params")
        return

    record(train_classifier(build, data, cfg))


if __name__ == "__main__":
    main()
