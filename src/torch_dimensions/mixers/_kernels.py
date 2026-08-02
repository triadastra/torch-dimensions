"""Which implementation runs: the authors' fused kernels, or our torch path.

The rule, decided per call rather than per install:

1. **CUDA tensors and an importable Triton kernel** — the upstream kernel
   runs, unchanged. A machine that can run the authors' code does run it, and
   its numbers are theirs.
2. **Anything else** — no Triton, a build that fails to import, or tensors on
   CPU/MPS — the torch path runs.

Deciding at call time rather than at import time matters for the case that
falls between: Triton installs happily on a CUDA-less Linux box, so an
import-time choice would pick a kernel that then dies on the first CPU
tensor. Asking "where is this tensor" answers the question that actually
determines whether the kernel can run.

``TD_FORCE_TORCH_KERNELS=1`` forces the torch path everywhere, which is how
the fallback gets exercised on a CUDA machine — an untested fallback is a
fallback that does not work.
"""

from __future__ import annotations

import importlib
import os
from typing import Any

import torch

__all__ = ["forced_torch", "load_upstream", "prefer_upstream"]

_MISSING = object()
_cache: dict[tuple[str, str], Any] = {}


def forced_torch() -> bool:
    """True when the environment demands the torch path regardless of device."""
    return bool(os.environ.get("TD_FORCE_TORCH_KERNELS"))


def load_upstream(module: str, name: str):
    """Import an upstream kernel entry point, or return ``None``.

    Cached, including the failure: a missing Triton should cost one failed
    import for the whole process, not one per forward pass.
    """
    key = (module, name)
    if key not in _cache:
        try:
            _cache[key] = getattr(importlib.import_module(module), name)
        except Exception:  # noqa: BLE001 - any import failure means "not available"
            _cache[key] = _MISSING
    found = _cache[key]
    return None if found is _MISSING else found


def prefer_upstream(*tensors: Any) -> bool:
    """Whether the fused upstream kernels can and should run for these tensors."""
    if forced_torch():
        return False
    return any(isinstance(t, torch.Tensor) and t.is_cuda for t in tensors)
