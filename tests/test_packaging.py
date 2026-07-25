"""Phase 0 acceptance: the package installs and imports with no CUDA and no
optional dependencies present. See PLAN.md, Phase 0."""

import importlib


def test_imports_without_optional_deps():
    td = importlib.import_module("torch_dimensions")
    assert td.__version__


def test_import_does_not_require_cuda():
    import torch

    td = importlib.import_module("torch_dimensions")
    # Importing must never touch a device. If this ever fails it means an
    # optional kernel adapter is being imported eagerly instead of defensively.
    assert torch.tensor([1.0]).device.type == "cpu"
    assert isinstance(td.__all__, list)
