"""torch-dimensions — N-dimensional sequence models for PyTorch.

Every model in scope is a 1-D mixer plus a plan for sweeping it over an N-D
lattice. See DESIGN.md for the architecture and PLAN.md for build order.

Conventionally imported as::

    import torch_dimensions as td

Public names land here as each phase completes, and optional-dependency blocks
are imported defensively so a CPU-only install stays importable.
"""

from torch_dimensions import data, mixers, testing, viz
from torch_dimensions.compose import (
    ND_METHODS,
    AxialKernel,
    AxialScan,
    axial_apply,
    axial_attention,
    axial_contract,
    axial_scan,
    cafa,
    kron_operator,
    register_nd_method,
    resolve_nd_method,
)
from torch_dimensions.config import MODELS, build, list_models, load, register_model, save
from torch_dimensions.lattice import Lattice, Restore, Sub
from torch_dimensions.models.rnn import GRU, LSTM
from torch_dimensions.models.ssm import S4, S4D, S4DND, S4ND, Mamba, MambaND
from torch_dimensions.plan import AxisCoverage, Coverage, ScanPlan, Step
from torch_dimensions.spec import spec

__version__ = "0.1.0"

__all__ = [
    "GRU",
    "LSTM",
    "Mamba",
    "MambaND",
    "S4",
    "S4D",
    "S4DND",
    "S4ND",
    "MODELS",
    "ND_METHODS",
    "AxialKernel",
    "AxialScan",
    "AxisCoverage",
    "Coverage",
    "Lattice",
    "Restore",
    "ScanPlan",
    "Step",
    "Sub",
    "axial_apply",
    "build",
    "axial_attention",
    "axial_contract",
    "axial_scan",
    "cafa",
    "kron_operator",
    "data",
    "list_models",
    "load",
    "mixers",
    "testing",
    "viz",
    "register_model",
    "register_nd_method",
    "save",
    "resolve_nd_method",
    "spec",
]
