"""Import shims that let the upstream research repos load off-GPU.

The clones this file manages live **outside** this repository and are used for
comparison only — nothing *this file fetches* is redistributed. (Separately,
the specific Apache-2.0/MIT-licensed reference files that the library does
redistribute live in `src/torch_dimensions/_vendor`, with their own manifest
and byte-diff verification; `dossier/verify_vendored.py` checks them against
the clones made here. Mamba-ND grants no license and is never vendored — it is
only ever imported from a local clone.)

`external(name)` fetches a shallow clone the first time it is asked for; set
`TD_EXTERNAL` to choose where they live (default `~/.cache/torch-dimensions/
upstream`). It refuses outright to place a clone inside this repository, which
is the one mistake that would turn "we read their code" into "we shipped their
code".

Every stub below replaces something a research repo imports at module scope
for reasons unrelated to the mathematics: a trainer, a config framework, a hub
mixin, a CUDA kernel. Stubbing them is what separates "this model needs a GPU"
from "this repo's *packaging* needs a GPU", and the whole point of the
exercise is that the second is far more common than the first.
"""

from __future__ import annotations

import os
import subprocess
import sys
import types
from pathlib import Path

# The repositories these comparisons read, and what each one grants. The
# license column is not decoration: it decides what may be done with what is
# fetched, and Mamba-ND grants nothing at all.
# `paths` are the sparse-checkout directories: only these are downloaded.
# A full clone of s4 is 39 MB and the comparison reads 420 KB of it, so this
# is 26x smaller and produces identical numbers (verified). Sparse checkout is
# preferred over vendoring a trimmed copy for one reason above the others: the
# comparison is only worth something because it runs *their* code, and hand-
# picking "the modules that matter" makes us the ones deciding what matters.
# A wrong call there makes the check agree for the wrong reason.
UPSTREAM = {
    "s4": {
        "url": "https://github.com/state-spaces/s4",
        "license": "Apache-2.0",
        "paths": ["src/models", "src/utils"],
    },
    "mamba": {
        "url": "https://github.com/state-spaces/mamba",
        "license": "Apache-2.0",
        "paths": ["mamba_ssm"],
    },
    "CaFA": {
        "url": "https://github.com/BaratiLab/CaFA",
        "license": "MIT",
        "paths": ["libs"],
    },
    "Mamba-ND": {
        "url": "https://github.com/jacklishufan/Mamba-ND",
        "license": "NO LICENSE",
        "paths": ["video_pretraining"],
    },
}

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _clone_root() -> Path:
    root = Path(
        os.environ.get("TD_EXTERNAL", Path.home() / ".cache" / "torch-dimensions" / "upstream")
    ).expanduser()
    # A clone inside this repository would put unlicensed third-party code one
    # `git add -A` away from being committed and published. Refused, not warned
    # about: a warning is something you scroll past.
    resolved = root.resolve()
    if resolved == _REPO_ROOT or _REPO_ROOT in resolved.parents:
        raise ValueError(
            f"TD_EXTERNAL points inside this repository ({resolved}). Upstream clones must "
            "live outside it — one `git add -A` away from committing someone else's code is "
            "not a place to keep it. Choose a path outside the repo."
        )
    return root


def external(name: str, *, clone: bool = True) -> Path:
    """Path to an upstream clone, fetching it on first use.

    Shallow-clones into `TD_EXTERNAL` (outside this repo, enforced). Pass
    `clone=False` to require that it already exists.
    """
    if name not in UPSTREAM:
        raise KeyError(f"unknown upstream {name!r}; known: {sorted(UPSTREAM)}")
    spec = UPSTREAM[name]
    root = _clone_root()
    path = root / name
    if path.exists():
        return path
    if not clone:
        raise FileNotFoundError(f"upstream repo {name!r} not found under {root}")

    print(f"cloning {name} from {spec['url']} into {path}")
    print(f"  license: {spec['license']}")
    if spec["license"] == "NO LICENSE":
        print(
            "  ^ this repository states no license. Cloning it to read and compare against\n"
            "    is your own business; redistributing it, or copying its code into a project,\n"
            "    is not granted. Nothing from it enters torch-dimensions."
        )
    print(f"  fetching only: {', '.join(spec['paths'])}")
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--filter=blob:none", "--sparse", "--depth", "1", spec["url"], str(path)],
        check=True,
    )
    subprocess.run(["git", "sparse-checkout", "set", *spec["paths"]], cwd=path, check=True)
    print(f"  at commit {head(name)}")
    return path


def head(name: str) -> str:
    """The upstream commit a clone is sitting on, or "unknown".

    Worth having because a number is only reproducible if you can say what it
    was measured against. Clones made by hand often cannot answer this: three
    of the four repositories originally used here had no `.git` of their own,
    so `git -C` silently reported the *enclosing* repository's commit — the
    same SHA for three different projects, which is how the problem announced
    itself.
    """
    path = _clone_root() / name
    if not (path / ".git").exists():
        return "unknown (no .git — not a clone)"
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True, check=False
    )
    return out.stdout.strip() or "unknown"


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
