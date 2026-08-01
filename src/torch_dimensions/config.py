"""Registry, config, and save/load — the constructive counterpart of ``spec``.

``spec(model)`` *describes* a model for a reader; ``build(cfg)`` *constructs*
one from a plain dict (or YAML), and ``save``/``load`` round-trip architecture
and weights in one file, so a checkpoint reconstructs its own model without
the user re-specifying anything:

    model = td.build({"kind": "s4nd", "d_model": 64, "n_layers": 8,
                      "dim": 2, "shape": [32, 32]})
    td.save(model, "run.td")
    same = td.load("run.td")     # identical outputs, bitwise

Two properties are non-negotiable here. A checkpoint records its format
version and refuses to load under an incompatible one, rather than loading
wrong silently. And a lattice's validity mask travels *with* the checkpoint —
a model restored against a different sparsity pattern is a wrong model, not a
warning.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from torch_dimensions.compose import ND_METHODS
from torch_dimensions.lattice import Lattice
from torch_dimensions.plan import ScanPlan

__all__ = [
    "MODELS",
    "build",
    "list_models",
    "load",
    "register_model",
    "save",
]

CHECKPOINT_FORMAT = "torch-dimensions/checkpoint"
CHECKPOINT_VERSION = 1


def _models() -> dict[str, type[nn.Module]]:
    # Imported lazily: models import config (for recording), so importing
    # models here at module import would be circular.
    from torch_dimensions.models.attention import Transformer
    from torch_dimensions.models.rnn import GRU, LSTM
    from torch_dimensions.models.ssm import S4, S4D, S4DND, S4ND, Mamba, MambaND

    return {
        "lstm": LSTM,
        "gru": GRU,
        "transformer": Transformer,
        "s4": S4,
        "s4d": S4D,
        "mamba": Mamba,
        "s4nd": S4ND,
        "s4dnd": S4DND,
        "mamband": MambaND,
    }


MODELS: dict[str, type[nn.Module]] = {}


def _ensure_registry() -> None:
    if not MODELS:
        MODELS.update(_models())


def register_model(name: str, cls: type[nn.Module]) -> None:
    """Make a model class addressable by ``kind`` in configs and checkpoints."""
    _ensure_registry()
    name = name.lower()
    if name in MODELS:
        raise ValueError(f"model kind {name!r} is already registered")
    MODELS[name] = cls


def list_models() -> list[str]:
    """The registered ``kind`` names, sorted."""
    _ensure_registry()
    return sorted(MODELS)


def nd_method_name(method: object) -> str | None:
    """The registry name of an nd_method, or None for an unregistered callable.

    None is legal to *run* with but refuses to :func:`save` — a checkpoint
    holding a function it cannot name could never rebuild itself.
    """
    if isinstance(method, str):
        return method
    for name, fn in ND_METHODS.items():
        if fn is method:
            return name
    return None


def lattice_to_dict(lat: Lattice) -> dict[str, Any]:
    """A lattice as plain JSON-able types; the validity mask as nested lists."""
    return {
        "shape": list(lat.shape),
        "names": list(lat.names or ()),
        "time": lat.time,
        "valid": None if lat.valid is None else lat.valid.tolist(),
    }


def lattice_from_dict(d: dict[str, Any] | Lattice) -> Lattice:
    if isinstance(d, Lattice):
        return d
    valid = d.get("valid")
    return Lattice(
        shape=tuple(d["shape"]),
        names=tuple(d["names"]) if d.get("names") else None,
        valid=None if valid is None else torch.as_tensor(valid, dtype=torch.bool),
        time=bool(d.get("time", False)),
    )


def _plan_from(cfg_plan: Any, lattice: Lattice | None, n_layers: int | None) -> ScanPlan:
    """Accept a plan as explicit steps, a constructor-style dict, or a ScanPlan."""
    if isinstance(cfg_plan, ScanPlan):
        return cfg_plan
    if "steps" in cfg_plan:
        return ScanPlan.from_dict(cfg_plan)
    kind = cfg_plan.get("type")
    if kind not in ("cyclic", "paired"):
        raise ValueError(
            f"a plan dict needs either 'steps' or type: cyclic|paired; got {sorted(cfg_plan)}"
        )
    axes = cfg_plan.get("axes")
    if axes is None:
        if lattice is None:
            raise ValueError(f"a {kind} plan needs 'axes', or a lattice to take them from")
        axes = lattice.axis_names
    if n_layers is None:
        raise ValueError(f"a {kind} plan needs 'n_layers' in the config")
    ctor = ScanPlan.cyclic if kind == "cyclic" else ScanPlan.paired
    return ctor(tuple(axes), n_layers, bidirectional=cfg_plan.get("bidirectional", False))


def _accepted_keys(cls: type[nn.Module], cfg: dict[str, Any]) -> set[str]:
    """Every key this kind can accept: the model's own signature plus the
    resolved nd_method's target signature (strategies forward **kwargs)."""
    from torch_dimensions.compose import resolve_nd_method
    from torch_dimensions.compose.attention import AxialKernel
    from torch_dimensions.compose.scan import AxialScan

    keys: set[str] = {"kind"}
    seen: list[type] = [cls]
    for c in seen:
        for name, p in inspect.signature(c.__init__).parameters.items():  # type: ignore[misc]
            if name != "self" and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL):
                keys.add(name)
        base = [b for b in c.__mro__[1:] if b not in (nn.Module, object)]
        if base and base[0] not in seen:
            seen.append(base[0])

    method = cfg.get("method") or cfg.get("nd_method")
    target: Any = AxialScan
    if method is not None:
        resolved = resolve_nd_method(method)
        if resolved is ND_METHODS.get("axial_scan"):
            target = AxialScan
        elif resolved in (ND_METHODS.get("axial_attention"), ND_METHODS.get("cafa")):
            target = AxialKernel
        else:
            target = resolved
    for name, p in inspect.signature(
        target.__init__ if inspect.isclass(target) else target
    ).parameters.items():
        if name not in ("self", "mixer") and p.kind is not p.VAR_KEYWORD:
            keys.add(name)
    return keys


def build(cfg: dict[str, Any] | str | Path) -> nn.Module:
    """Construct a model from a config dict, or from a YAML file path.

    ``cfg["kind"]`` names a registered model; everything else is constructor
    keywords, with ``lattice`` and ``plan`` accepted as plain dicts. Unknown
    keys are a hard error naming the key and listing the accepted ones — a
    silently ignored typo is a silently different model.
    """
    if isinstance(cfg, (str, Path)):
        try:
            import yaml
        except ImportError as e:  # pragma: no cover - environment-dependent
            raise ImportError("building from a YAML path needs pyyaml installed") from e
        with open(cfg) as f:
            cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise TypeError(f"config must be a dict or a YAML path; got {type(cfg).__name__}")

    cfg = dict(cfg)
    _ensure_registry()
    kind = cfg.pop("kind", None)
    if kind is None or str(kind).lower() not in MODELS:
        raise ValueError(f"config needs a registered 'kind'; got {kind!r}. Known: {list_models()}")
    cls = MODELS[str(kind).lower()]

    unknown = sorted(set(cfg) - _accepted_keys(cls, cfg))
    if unknown:
        raise ValueError(
            f"unknown config key(s) {unknown} for kind {str(kind).lower()!r}; "
            f"accepted: {sorted(_accepted_keys(cls, cfg) - {'kind'})}"
        )

    if cfg.get("lattice") is not None:
        cfg["lattice"] = lattice_from_dict(cfg["lattice"])
    if cfg.get("valid") is not None and not isinstance(cfg["valid"], torch.Tensor):
        cfg["valid"] = torch.as_tensor(cfg["valid"], dtype=torch.bool)  # the ND-name sugar path
    if cfg.get("plan") is not None:
        cfg["plan"] = _plan_from(cfg["plan"], cfg.get("lattice"), cfg.get("n_layers"))
    return cls(**cfg)


def _kind_of(model: nn.Module) -> str:
    _ensure_registry()
    for name, cls in MODELS.items():
        if type(model) is cls:
            return name
    raise ValueError(
        f"{type(model).__name__} is not a registered model kind, so a checkpoint "
        "could not rebuild it; register_model() it first"
    )


def _is_safetensors(path: str | Path) -> bool:
    return str(path).endswith(".safetensors")


def _checkpoint_header(model: nn.Module) -> tuple[str, dict]:
    kind = _kind_of(model)
    config = getattr(model, "config", None)
    if config is None:
        raise ValueError(f"{type(model).__name__} records no construction config; cannot save")
    if config.get("nd_method") is None:
        raise ValueError(
            "this model was built with an unregistered nd_method callable; a checkpoint "
            "cannot name it to rebuild itself. register_nd_method() it, rebuild, then save"
        )
    substituted = getattr(model, "_substituted_mixer", None)
    if substituted:
        raise ValueError(
            f"this model was built with mixer={substituted}, which the recipe cannot record; "
            f"loading the checkpoint would rebuild it with {type(model).__name__}'s own mixer "
            "and return a different model that looks fine. Substituted mixers are for "
            "debugging — save the stock model, or register a model kind for this one"
        )
    return kind, config


def save(model: nn.Module, path: str | Path) -> None:
    """Write architecture and weights to one file. See :func:`load`.

    The container follows the extension. ``.safetensors`` writes the weights
    in that format with the config carried in its metadata — still one file,
    and one that cannot execute code when it is opened. Any other extension
    writes a torch pickle, which stays the default only because it is the
    format everything already reads.

    A torch pickle is arbitrary code at load time. That is a real
    supply-chain liability for a file people download from strangers, and it
    is why `.safetensors` exists as an option here even though the library's
    own tests exercise both.
    """
    kind, config = _checkpoint_header(model)
    from torch_dimensions import __version__

    if _is_safetensors(path):
        try:
            from safetensors.torch import save_file
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "saving .safetensors needs the safetensors package: "
                'pip install "torch-dimensions[safetensors]"'
            ) from e
        state = model.state_dict()
        complex_keys = sorted(k for k, v in state.items() if v.is_complex())
        if complex_keys:
            raise ValueError(
                f"safetensors cannot store complex tensors, and this model has "
                f"{len(complex_keys)}: {complex_keys[:3]}. Save to a .td file instead."
            )
        save_file(
            {k: v.detach().contiguous() for k, v in state.items()},
            str(path),
            metadata={
                "format": CHECKPOINT_FORMAT,
                "version": str(CHECKPOINT_VERSION),
                "library": __version__,
                "kind": kind,
                # safetensors metadata is string-to-string, so the config
                # travels as JSON. It round-trips through `lattice_from_dict`
                # on the way back, exactly as the pickle path does.
                "config": json.dumps(config),
            },
        )
        return

    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "version": CHECKPOINT_VERSION,
            "library": __version__,
            "kind": kind,
            "config": config,
            "state_dict": model.state_dict(),
        },
        path,
    )


def _check_version(version: Any, library: Any) -> None:
    if int(version) != CHECKPOINT_VERSION:
        raise ValueError(
            f"checkpoint format v{version} is not v{CHECKPOINT_VERSION}; "
            "refusing to load it silently — convert it or pin the library version "
            f"that wrote it (recorded: {library or 'unknown'})"
        )


def load(path: str | Path, map_location: Any = None) -> nn.Module:
    """Rebuild a model from a :func:`save` checkpoint, weights included.

    Reads either container — the extension says which — and refuses, rather
    than guesses at, a checkpoint from an incompatible format version.
    """
    if _is_safetensors(path):
        try:
            from safetensors import safe_open
            from safetensors.torch import load_file
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "reading .safetensors needs the safetensors package: "
                'pip install "torch-dimensions[safetensors]"'
            ) from e

        with safe_open(str(path), framework="pt") as fh:
            meta = fh.metadata() or {}
        if meta.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"{path} is not a torch-dimensions checkpoint")
        _check_version(meta.get("version", -1), meta.get("library"))
        model = build({"kind": meta["kind"], **json.loads(meta["config"])})
        model.load_state_dict(load_file(str(path), device=str(map_location or "cpu")))
        return model

    ckpt = torch.load(path, map_location=map_location)
    if not isinstance(ckpt, dict) or ckpt.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a torch-dimensions checkpoint")
    _check_version(ckpt.get("version", -1), ckpt.get("library"))
    model = build({"kind": ckpt["kind"], **ckpt["config"]})
    model.load_state_dict(ckpt["state_dict"])
    return model
