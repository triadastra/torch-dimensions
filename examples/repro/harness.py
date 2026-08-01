"""The shared training loop for the reproductions, and the results ledger.

One loop, used by every reproduction, so that a difference between two rows in
RESULTS.md is a difference in the *model* and not in somebody's training
script. Everything that could vary and would explain away a result is fixed
here and recorded with the row: seed, device, dtype, optimizer, schedule,
wall-clock, and the library version that produced it.

This lives in ``examples/`` rather than in the package because it is a
training loop, and the library's scope note says training loops are the
caller's, permanently. It is a good training loop; it is still not the
library's job.
"""

from __future__ import annotations

import json
import platform
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

import torch_dimensions as td

RESULTS = Path(__file__).resolve().parent / "results.json"

# SSM kernel parameters want a smaller learning rate and no weight decay — the
# S4 papers are explicit about it, and it is the single recipe detail most
# likely to explain a failed reproduction. Matched by name so that a new mixer
# gets the treatment by naming its parameters the way the family does.
SSM_PARAM_MARKERS = ("log_dt", "log_a_real", "a_imag", "a_log", "inv_w", "_p", "_b", "d_skip")


def pick_device(requested: str | None = None) -> str:
    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def param_groups(model: nn.Module, lr: float, ssm_lr: float, weight_decay: float) -> list[dict]:
    slow, fast = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (slow if any(m in name.lower() for m in SSM_PARAM_MARKERS) else fast).append(p)
    groups = [{"params": fast, "lr": lr, "weight_decay": weight_decay}]
    if slow:
        groups.append({"params": slow, "lr": ssm_lr, "weight_decay": 0.0})
    return groups


@dataclass
class Config:
    """Everything that would change the number. Serialized with the result."""

    name: str
    task: str
    model: str
    epochs: int = 10
    batch: int = 64
    lr: float = 4e-3
    ssm_lr: float = 1e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 0
    device: str = "auto"
    limit_train: int | None = None
    """Cap the training set — for smoke runs only, and it lands in the row so a
    capped run can never be mistaken for a full one."""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Result:
    config: dict
    metric: float
    """The headline number. Its meaning is in `metric_name`, never implied."""
    metric_name: str
    train_loss: float
    epochs_run: int
    seconds: float
    machine: dict
    history: list[dict]

    def formatted(self) -> str:
        if self.metric_name == "test_acc":
            return f"{self.metric * 100:.2f}%"
        return f"{self.metric:.4f}"

    def row(self) -> str:
        c = self.config
        return (
            f"| {c['task']} | {c['model']} | {c['epochs']} | {c['seed']} | "
            f"{self.metric_name} {self.formatted()} | {self.seconds / 60:.1f} min | "
            f"{self.machine['accelerator']} |"
        )


def machine(device: str) -> dict[str, str]:
    name = device
    if device == "cuda":
        name = torch.cuda.get_device_name(0)
    elif device == "mps":
        name = f"Apple Silicon (MPS) {platform.machine()}"
    return {
        "accelerator": name,
        "platform": f"{platform.system()} {platform.release()}",
        "torch": torch.__version__,
        "torch_dimensions": td.__version__,
    }


def batches(
    x: torch.Tensor, y: torch.Tensor, batch: int, *, shuffle: bool, generator: torch.Generator
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    n = x.shape[0]
    order = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    for i in range(0, n, batch):
        idx = order[i : i + batch]
        yield x[idx], y[idx]


@torch.no_grad()
def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch: int, device: str) -> float:
    model.eval()
    correct = 0
    for xb, yb in batches(x, y, batch, shuffle=False, generator=torch.Generator()):
        pred = model(xb.to(device)).argmax(-1).cpu()
        correct += int((pred == yb).sum())
    model.train()
    return correct / x.shape[0]


def train_classifier(
    build: Callable[[], nn.Module],
    data: dict[str, torch.Tensor],
    cfg: Config,
    *,
    log_every: int = 100,
) -> Result:
    """Fit a classifier and return the row that goes in RESULTS.md.

    ``build`` is called *after* the seed is set, so the model's initialization
    is part of what the seed controls — a reproduction whose init is outside
    the seed is not seeded.
    """
    device = pick_device(cfg.device)
    torch.manual_seed(cfg.seed)
    model = build().to(device)
    n_params = sum(p.numel() for p in model.parameters())

    train_x, train_y = data["train_x"], data["train_y"]
    if cfg.limit_train:
        train_x, train_y = train_x[: cfg.limit_train], train_y[: cfg.limit_train]

    opt = torch.optim.AdamW(param_groups(model, cfg.lr, cfg.ssm_lr, cfg.weight_decay))
    steps_per_epoch = (train_x.shape[0] + cfg.batch - 1) // cfg.batch
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs * steps_per_epoch)
    loss_fn = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(cfg.seed + 1)

    print(
        f"{cfg.name}: {n_params:,} parameters on {device}, "
        f"{train_x.shape[0]:,} train / {data['test_x'].shape[0]:,} test, "
        f"{cfg.epochs} epochs x {steps_per_epoch} steps"
    )
    history: list[dict] = []
    t0 = time.time()
    last_loss = float("nan")
    for epoch in range(cfg.epochs):
        running = 0.0
        for step, (xb, yb) in enumerate(
            batches(train_x, train_y, cfg.batch, shuffle=True, generator=g)
        ):
            xb, yb = xb.to(device), yb.to(device)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            running += float(loss.detach())
            if log_every and step % log_every == 0:
                print(
                    f"  epoch {epoch} step {step}/{steps_per_epoch} "
                    f"loss {float(loss.detach()):.4f} lr {sched.get_last_lr()[0]:.2e}",
                    flush=True,
                )
        last_loss = running / max(steps_per_epoch, 1)
        acc = accuracy(model, data["test_x"], data["test_y"], cfg.batch, device)
        history.append({"epoch": epoch, "train_loss": last_loss, "test_acc": acc})
        print(
            f"  epoch {epoch}: train loss {last_loss:.4f}  test acc {acc * 100:.2f}%  "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )

    return Result(
        config={**asdict(cfg), "n_params": n_params, "device": device},
        metric=history[-1]["test_acc"] if history else float("nan"),
        metric_name="test_acc",
        train_loss=last_loss,
        epochs_run=len(history),
        seconds=time.time() - t0,
        machine=machine(device),
        history=history,
    )


def record(result: Result) -> None:
    """Append to the results ledger. Nothing is ever overwritten.

    A result file that gets rewritten is a result file that can quietly lose
    the run that disagreed.
    """
    ledger = json.loads(RESULTS.read_text()) if RESULTS.exists() else []
    ledger.append({"recorded": time.strftime("%Y-%m-%d %H:%M:%S"), **asdict(result)})
    RESULTS.write_text(json.dumps(ledger, indent=2))
    print(f"recorded -> {RESULTS}")
    print(result.row())


def train_regressor(
    build: Callable[[], nn.Module],
    step_fn: Callable[[nn.Module, tuple], torch.Tensor],
    train: list,
    test: list,
    cfg: Config,
    *,
    metric_name: str = "test_mse",
    log_every: int = 100,
) -> Result:
    """The same loop for a regression task, with the loss supplied by the caller.

    Forecasting on a lattice has no single right loss — which cells count,
    whether absent ones are scored, how the target is standardized — so
    `step_fn(model, batch) -> scalar loss` stays the experiment's own. What is
    shared is everything that would otherwise silently differ between two rows:
    the optimizer, the schedule, the seeding, and the clock.
    """
    device = pick_device(cfg.device)
    torch.manual_seed(cfg.seed)
    model = build().to(device)
    n_params = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW(param_groups(model, cfg.lr, cfg.ssm_lr, cfg.weight_decay))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs * max(len(train), 1))
    g = torch.Generator().manual_seed(cfg.seed + 1)

    print(f"{cfg.name}: {n_params:,} parameters on {device}, {len(train)} train batches")
    history: list[dict] = []
    t0 = time.time()
    last_loss = float("nan")
    for epoch in range(cfg.epochs):
        model.train()
        order = torch.randperm(len(train), generator=g).tolist()
        running = 0.0
        for i, idx in enumerate(order):
            loss = step_fn(model, train[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            running += float(loss.detach())
            if log_every and i % log_every == 0:
                print(
                    f"  epoch {epoch} batch {i}/{len(train)} loss {float(loss.detach()):.5f}",
                    flush=True,
                )
        last_loss = running / max(len(order), 1)

        model.eval()
        with torch.no_grad():
            held = sum(float(step_fn(model, b)) for b in test) / max(len(test), 1)
        history.append({"epoch": epoch, "train_loss": last_loss, metric_name: held})
        print(
            f"  epoch {epoch}: train {last_loss:.5f}  {metric_name} {held:.5f} "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )

    return Result(
        config={**asdict(cfg), "n_params": n_params, "device": device},
        metric=history[-1][metric_name] if history else float("nan"),
        metric_name=metric_name,
        train_loss=last_loss,
        epochs_run=len(history),
        seconds=time.time() - t0,
        machine=machine(device),
        history=history,
    )
