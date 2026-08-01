"""Train a 2-D LSTM and watch it live in the viewer.

Run the viewer dev server, then:

    python examples/viewer_live.py

The script trains ``td.LSTM`` over a sparse 2-D lattice on the best available
device (MPS on Apple Silicon, else CUDA, else CPU) and writes
``viewer/public/run.json`` after every optimizer step — the model's
architecture spec plus the loss history. The viewer polls that file and
switches itself to the live run: architecture on the right, loss curve on the
left, exactly the "watch it train" half of the GUI-mode idea (VIEWER.md V4).

The task is the library's own trainability task: a cumulative sum along the
``w`` axis, which a model that never mixes along ``w`` provably cannot learn.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn

import torch_dimensions as td

OUT = Path(__file__).resolve().parent.parent / "viewer" / "public" / "run.json"
STEPS = 300
EVAL_EVERY = 10


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    device = pick_device()
    torch.manual_seed(0)

    valid = torch.rand(6, 8) > 0.25
    valid[0, 0] = True
    lat = td.Lattice(shape=(6, 8), names=("h", "w"), valid=valid, time=True)
    model = td.LSTM(d_model=32, n_layers=6, lattice=lat, d_input=1, bidirectional=("h", "w"))
    model = model.to(device)
    head = nn.Linear(32, 1).to(device)
    opt = torch.optim.Adam([*model.parameters(), *head.parameters()], lr=1e-2)

    mask = lat.mask(torch.float32).to(device)
    w_dim = lat.tensor_dim("w")

    def draw(g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(8, 5, 6, 8, 1, generator=g).to(device) * mask
        return x, x.cumsum(dim=w_dim) * mask

    run: dict = {
        "started": time.time(),
        "device": device,
        "task": "cumsum along w (sparse 6×8 lattice)",
        "status": "training",
        "spec": model.to_spec(),
        "metrics": [],
    }

    def flush() -> None:
        tmp = OUT.with_suffix(".tmp")
        OUT.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(run))
        os.replace(tmp, OUT)

    g = torch.Generator().manual_seed(1)
    g_eval = torch.Generator().manual_seed(9973)
    x_eval, y_eval = draw(g_eval)

    print(f"training on {device}; writing {OUT}")
    for step in range(STEPS):
        x, y = draw(g)
        loss = (head(model(x)) - y).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        entry: dict = {"step": step, "loss": float(loss.detach())}
        if step % EVAL_EVERY == 0 or step == STEPS - 1:
            model.eval()
            with torch.no_grad():
                entry["held_out"] = float((head(model(x_eval)) - y_eval).pow(2).mean())
            model.train()
            print(f"step {step:4d}  loss {entry['loss']:.5f}  held-out {entry['held_out']:.5f}")
        run["metrics"].append(entry)
        flush()

    run["status"] = "done"
    flush()
    print("done")


if __name__ == "__main__":
    main()
