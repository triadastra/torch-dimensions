"""Train a 2-D LSTM under the viewer's control panel.

Run the viewer dev server, then:

    python examples/viewer_live.py

The script builds ``td.LSTM`` over a sparse 2-D lattice on the best available
device (MPS on Apple Silicon, else CUDA, else CPU) and then **waits**: nothing
trains until the Start button in the viewer is pressed. State flows one way,
control the other:

- state:   ``viewer/public/run.json`` is rewritten after every step — the
  architecture spec, loss history, status, and progress. The viewer polls it.
- control: a small HTTP server (default port 8765) accepts
  ``POST /control {"action": "start" | "pause" | "resume" | "stop"}`` from the
  panel's buttons. The run document carries the control URL, so the viewer
  knows where to send them.

The task is the library's own trainability task: a cumulative sum along the
``w`` axis, which a model that never mixes along ``w`` provably cannot learn.
"""

from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
import torch.nn as nn

import torch_dimensions as td

OUT = Path(__file__).resolve().parent.parent / "viewer" / "public" / "run.json"
PRESET = os.environ.get("TD_VIEWER_PRESET", "lstm2d")
STEPS = int(os.environ.get("TD_VIEWER_STEPS", "2000" if PRESET == "mamba4d" else "300"))
EVAL_EVERY = 10
CONTROL_PORT = 8765


class Control:
    """The run's state machine, shared between the HTTP thread and training.

    waiting → (start) → training ⇄ (pause/resume) paused
    any state → (stop) → stopped; training ends naturally → done.
    """

    ACTIONS = {
        "start": ("waiting", "training"),
        "pause": ("training", "paused"),
        "resume": ("paused", "training"),
    }

    def __init__(self) -> None:
        self.state = "waiting"
        self.cond = threading.Condition()

    def apply(self, action: str) -> str:
        with self.cond:
            if action == "stop" and self.state in ("waiting", "training", "paused"):
                self.state = "stopped"
            else:
                expected = self.ACTIONS.get(action)
                if expected and self.state == expected[0]:
                    self.state = expected[1]
            self.cond.notify_all()
            return self.state

    def wait_while(self, *states: str) -> str:
        with self.cond:
            while self.state in states:
                self.cond.wait(timeout=1.0)
            return self.state

    def finish(self) -> None:
        with self.cond:
            if self.state != "stopped":
                self.state = "done"


def serve_control(ctrl: Control, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def _respond(self, code: int, body: dict | None = None) -> None:
            payload = json.dumps(body or {}).encode()
            self.send_response(code)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_OPTIONS(self) -> None:  # CORS preflight
            self._respond(204)

        def do_GET(self) -> None:
            self._respond(200, {"state": ctrl.state})

        def do_POST(self) -> None:
            if self.path != "/control":
                self._respond(404, {"error": "POST /control"})
                return
            try:
                raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                action = json.loads(raw or b"{}").get("action", "")
            except (ValueError, TypeError):
                self._respond(400, {"error": "body must be JSON with an 'action'"})
                return
            self._respond(200, {"state": ctrl.apply(action)})

        def log_message(self, *args) -> None:  # keep stdout for training progress
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build(preset: str):
    """Presets: lstm2d (small, fast) and mamba4d (the deep end — a sparse 4-D
    lattice under Mamba-ND with the official paired schedule)."""
    torch.manual_seed(0)
    if preset == "mamba4d":
        valid = torch.rand(4, 5, 6, 4) > 0.3
        valid.reshape(-1)[0] = True
        lat = td.Lattice(
            shape=(4, 5, 6, 4), names=("depth", "row", "col", "group"), valid=valid, time=True
        )
        plan = td.ScanPlan.paired(
            lat.axis_names, n_layers=18, bidirectional=("depth", "row", "col", "group")
        )
        model = td.Mamba(d_model=48, lattice=lat, plan=plan, d_input=1, d_state=16)
        task = "cumsum along col — sparse 4×5×6×4, Mamba-ND, paired schedule"
        return lat, model, task, "col", 4, 6
    valid = torch.rand(6, 8) > 0.25
    valid[0, 0] = True
    lat = td.Lattice(shape=(6, 8), names=("h", "w"), valid=valid, time=True)
    model = td.LSTM(d_model=32, n_layers=6, lattice=lat, d_input=1, bidirectional=("h", "w"))
    return lat, model, "cumsum along w (sparse 6×8 lattice)", "w", 8, 5


def main() -> None:
    device = pick_device()
    lat, model, task, target_axis, batch, t_len = build(PRESET)
    model = model.to(device)
    head = nn.Linear(model.config["d_model"], 1).to(device)
    opt = torch.optim.Adam([*model.parameters(), *head.parameters()], lr=1e-2)

    mask = lat.mask(torch.float32).to(device)
    w_dim = lat.tensor_dim(target_axis)

    def draw(g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.randn(batch, t_len, *lat.shape, 1, generator=g).to(device) * mask
        return x, x.cumsum(dim=w_dim) * mask

    ctrl = Control()
    serve_control(ctrl, CONTROL_PORT)

    run: dict = {
        "started": time.time(),
        "device": device,
        "task": task,
        "status": "waiting",
        "control": f"http://127.0.0.1:{CONTROL_PORT}",
        "total_steps": STEPS,
        "spec": model.to_spec(),
        "metrics": [],
    }

    def flush() -> None:
        run["status"] = ctrl.state
        tmp = OUT.with_suffix(".tmp")
        OUT.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(run))
        os.replace(tmp, OUT)

    flush()
    print(f"model built on {device}; waiting for Start in the viewer (control :{CONTROL_PORT})")
    if ctrl.wait_while("waiting") == "stopped":
        flush()
        print("stopped before training began")
        return

    g = torch.Generator().manual_seed(1)
    x_eval, y_eval = draw(torch.Generator().manual_seed(9973))

    for step in range(STEPS):
        if ctrl.state == "paused":
            flush()
            print(f"paused at step {step}")
            if ctrl.wait_while("paused") == "stopped":
                break
            print("resumed")
        if ctrl.state == "stopped":
            break

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

    ctrl.finish()
    flush()
    print(ctrl.state)


if __name__ == "__main__":
    main()
