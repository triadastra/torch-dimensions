"""``td.viz.show(model)`` — look at the architecture in a browser.

    import torch_dimensions as td

    model = td.MambaND(64, 12, dim=3, shape=(8, 8, 8))
    td.viz.show(model)

The viewer is a static bundle built from ``viewer/`` and shipped inside the
wheel, so this needs no node, no network, and no build step at install time.
What it serves is the same versioned spec document the library has always
emitted — this module opens a socket, the library still does not.

Accepts a model, a spec dict, a path to a spec JSON, or a path to a
checkpoint saved with :meth:`~torch_dimensions.models.base.LatticeModel.save`.
"""

from __future__ import annotations

import json
import os
import threading
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import torch.nn as nn

from torch_dimensions.spec import spec as model_spec

__all__ = ["BUNDLE", "bundle_exists", "resolve_spec", "serve", "show"]

BUNDLE = Path(__file__).resolve().parent / "static"

_MISSING = (
    "the viewer bundle is not in this install ({path} does not exist).\n"
    "A wheel from PyPI ships it; a git checkout does not, because the bundle is a build "
    "artifact. Build it once with:\n"
    "    cd viewer && npm install && npm run build && python viewer/install_bundle.py"
)


def bundle_exists() -> bool:
    return (BUNDLE / "index.html").exists()


def resolve_spec(target: Any) -> dict:
    """Turn whatever the caller has into a spec document.

    A model, a spec dict, a ``.json`` path, or a checkpoint path — the four
    things someone plausibly has in hand when they want to look at an
    architecture. Anything else raises here rather than serving a blank page,
    because an empty viewer is a much worse error message than an exception.
    """
    if isinstance(target, nn.Module):
        return model_spec(target)
    if isinstance(target, dict):
        if "layers" not in target or "lattice" not in target:
            raise ValueError(
                "that dict is not a spec document (no 'layers'/'lattice' keys); "
                "pass a model, or td.spec(model)"
            )
        return target
    if isinstance(target, (str, os.PathLike)):
        path = Path(target)
        if not path.exists():
            raise FileNotFoundError(f"no such file: {path}")
        if path.suffix == ".json":
            return json.loads(path.read_text())
        from torch_dimensions.config import load  # local: avoids an import cycle

        return model_spec(load(path))
    raise TypeError(
        f"cannot show a {type(target).__name__}; pass a model, a spec dict, a .json spec, "
        "or a checkpoint path"
    )


class _Handler(SimpleHTTPRequestHandler):
    """Static files from the bundle, plus the one document that is not in it."""

    def __init__(self, *args, payload: bytes, **kwargs) -> None:
        self._payload = payload
        super().__init__(*args, directory=str(BUNDLE), **kwargs)

    def do_GET(self) -> None:
        if self.path.split("?")[0] == "/spec.json":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(self._payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(self._payload)
            return
        super().do_GET()

    def log_message(self, *args: Any) -> None:  # a viewer is not a web server log
        pass


def serve(target: Any, port: int = 0) -> ThreadingHTTPServer:
    """Start the viewer server and return it without blocking.

    ``port=0`` asks the OS for a free one — the returned server's
    ``server_port`` says which. Call ``shutdown()`` when done. This is the
    entry point tests use; :func:`show` is the one humans use.
    """
    if not bundle_exists():
        raise FileNotFoundError(_MISSING.format(path=BUNDLE))
    payload = json.dumps(resolve_spec(target)).encode()
    server = ThreadingHTTPServer(("127.0.0.1", port), partial(_Handler, payload=payload))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def show(
    target: Any,
    *,
    port: int = 0,
    open_browser: bool = True,
    block: bool = True,
) -> ThreadingHTTPServer:
    """Serve the viewer for ``target`` and open a browser at it.

    Args:
        target: a model, a spec dict, a ``.json`` spec, or a checkpoint path.
        port: ``0`` picks a free port.
        open_browser: turn off over SSH, where opening a browser opens the
            wrong machine's browser.
        block: hold the process until interrupted. Off in notebooks and tests,
            where the server should live alongside the caller.

    Returns the server, so a non-blocking caller can shut it down.
    """
    server = serve(target, port)
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"torch-dimensions viewer: {url}")
    if open_browser:
        webbrowser.open(url)
    if block:
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            print("\nstopping viewer")
        finally:
            server.shutdown()
    return server
