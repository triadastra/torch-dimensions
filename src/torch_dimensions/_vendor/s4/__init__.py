"""The real state-spaces/s4 training pipeline, redistributed verbatim (Apache-2.0).

This is not the authors' standalone file — it is the subtree their
``train.py`` actually runs: ``src/models/sequence/modules/s4block.py`` (the
layer registered as ``"s4"``), ``s4nd.py`` (the S4ND layer), the kernel
modules (``fftconv``, ``ssm``, ``dplr``, ``kernel``), the HiPPO construction,
the functional kernels (``cauchy``, ``vandermonde``, ``krylov``,
``toeplitz``), the ``nn`` components, and the ``utils`` config/registry/train
modules they import. Every file is byte-identical to the pinned upstream
commit (see ``../MANIFEST.json``) except ``utils/train.py``, whose
training-only imports (pytorch_lightning, rich, omegaconf) are guarded —
each changed line tagged ``torch-dimensions patch``, the pristine copy in
``train.py.orig``.

**Why :func:`mount` exists.** The pipeline imports itself absolutely —
``from src.models.nn import ...`` — and resolves registry entries through
hydra by strings like ``"src.models.sequence.kernels.fftconv.FFTConv"``.
That is upstream's own convention: their repo runs with the repo root on
``sys.path`` so that ``src`` is importable. :func:`mount` reproduces exactly
that, registering this directory's ``src/`` as the ``src`` package. Nothing
is patched to rename imports, which is what keeps the files byte-identical.

The cost is honesty about a namespace: a process can hold only one package
named ``src``. If your own project also imports a top-level ``src``,
:func:`mount` refuses with an explanation rather than silently shadowing
either side.

See ``torch_dimensions._vendor`` for the verification scheme.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"


def mount() -> None:
    """Make the vendored pipeline importable under its canonical name ``src``.

    Idempotent. Raises ``ImportError`` if a foreign ``src`` module is already
    imported, because the upstream code cannot run under any other name and
    two packages cannot share one.
    """
    existing = sys.modules.get("src")
    if existing is not None:
        if getattr(existing, "__td_vendor__", False):
            return
        where = getattr(existing, "__file__", None) or list(getattr(existing, "__path__", []))
        raise ImportError(
            "cannot mount the vendored s4 pipeline: a module named 'src' is already "
            f"imported (from {where!r}). The upstream code imports itself as 'src.*' — "
            "its own repo convention — so the two cannot coexist in one process. "
            "Construct the upstream mixers before importing your own 'src' package, "
            "or rename yours."
        )
    pkg = types.ModuleType("src")
    pkg.__path__ = [str(_SRC)]  # type: ignore[attr-defined]
    pkg.__td_vendor__ = True  # type: ignore[attr-defined]
    sys.modules["src"] = pkg
