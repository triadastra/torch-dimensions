"""Import shims that let the upstream research repos load off-GPU.

These repos are *not* vendored into this one — they stay wherever you cloned
them and this file makes them importable, so nothing here is redistribution
and no license question arises. Point `TD_EXTERNAL` at the clone directory.

Every stub below replaces something a research repo imports at module scope
for reasons unrelated to the mathematics: a trainer, a config framework, a hub
mixin, a CUDA kernel. Stubbing them is what separates "this model needs a GPU"
from "this repo's *packaging* needs a GPU", and the whole point of the
exercise is that the second is far more common than the first.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path


def external(name: str) -> Path:
    root = Path(os.environ.get("TD_EXTERNAL", Path.home() / "Desktop/Safe/code/github/external"))
    path = root / name
    if not path.exists():
        raise FileNotFoundError(
            f"upstream repo {name!r} not found under {root}. Clone it there, or set "
            "TD_EXTERNAL to the directory holding the clones."
        )
    return path


def _module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


def stub_lightning() -> None:
    """`src/utils/train.py` imports it for a logger and a LightningModule base.

    Neither participates in a forward pass, so an empty class is a faithful
    stand-in for the purpose of checking arithmetic.
    """
    if "pytorch_lightning" in sys.modules:
        return

    class _LightningModule:
        pass

    class _Callback:
        pass

    pl = _module(
        "pytorch_lightning",
        LightningModule=_LightningModule,
        Callback=_Callback,
        Trainer=object,
        seed_everything=lambda *a, **k: None,
        __version__="0.0.0-stub",
    )
    _module("pytorch_lightning.utilities", rank_zero_only=lambda f: f)
    _module("pytorch_lightning.utilities.rank_zero", rank_zero_only=lambda f: f)
    _module("pytorch_lightning.loggers", Logger=object, WandbLogger=object)
    _module("pytorch_lightning.callbacks", Callback=_Callback)
    pl.utilities = sys.modules["pytorch_lightning.utilities"]


def stub_mm_and_mamba_cuda() -> None:
    """Stubs for Mamba-ND: the mm* ecosystem, and the fused CUDA kernels.

    `video_pretraining/models/mamband.py` imports `mmcv`, `mmengine`,
    `mmaction`, `prettytable` and `mamba_ssm` at module scope. Only the last
    is about the model's arithmetic, and even that is injected rather than
    hardcoded — `Block.__init__` takes `mixer_cls` as an argument, so the
    scan's *composition* can be studied with any 1-D mixer, including a
    portable one.

    That is the whole finding for this repo: Mamba-ND's schedule needs no
    CUDA. Its packaging does.
    """
    import torch.nn as nn

    if "mmcv" in sys.modules:
        return

    class _Base(nn.Module):
        def __init__(self, *a, **k):
            super().__init__()

    def _build_dropout(cfg, *a, **k):
        if isinstance(cfg, dict):
            p = cfg.get("drop_prob", cfg.get("p", 0.0))
            return nn.Dropout(p) if p else nn.Identity()
        return nn.Identity()

    def _build_norm_layer(cfg, num_features, **k):
        return "norm", nn.LayerNorm(num_features)

    _module("mmcv")
    _module("mmcv.cnn", build_norm_layer=_build_norm_layer)
    _module("mmcv.cnn.bricks")
    _module("mmcv.cnn.bricks.transformer", FFN=_Base, PatchEmbed=_Base)
    _module("mmcv.cnn.bricks.drop", build_dropout=_build_dropout)
    sys.modules["mmcv"].cnn = sys.modules["mmcv.cnn"]

    _module("mmengine")
    _module("mmengine.model", BaseModule=_Base, ModuleList=nn.ModuleList)
    _module("mmengine.model.weight_init", trunc_normal_=nn.init.trunc_normal_)
    _module("mmengine.utils", to_2tuple=lambda x: x if isinstance(x, tuple) else (x, x))
    _logger = type("MMLogger", (), {"get_current_instance": staticmethod(lambda: None)})
    _module("mmengine.logging", MMLogger=_logger)
    _module("mmengine.runner")
    _module("mmengine.runner.checkpoint", _load_checkpoint=lambda *a, **k: {})

    _module("mmaction")
    _registry = type("R", (), {"register_module": staticmethod(lambda *a, **k: lambda c: c)})
    _module("mmaction.registry", MODELS=_registry())
    _module("mmaction.utils", ConfigType=dict, OptConfigType=dict)
    _module("prettytable", PrettyTable=type("PrettyTable", (), {}))

    # The fused kernels. `Mamba` is only used as a default mixer_cls, and the
    # comparison injects its own; the RMSNorm helpers are imported at module
    # scope and never reached on the path we exercise.
    _module("mamba_ssm", Mamba=_Base)
    _module("mamba_ssm.ops")
    _module("mamba_ssm.ops.triton")
    _module(
        "mamba_ssm.ops.triton.layernorm",
        RMSNorm=nn.LayerNorm,
        layer_norm_fn=None,
        rms_norm_fn=None,
    )
