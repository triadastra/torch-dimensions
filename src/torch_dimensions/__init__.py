"""torch-dimensions — N-dimensional sequence models for PyTorch.

Every model in scope is a 1-D mixer plus a plan for sweeping it over an N-D
lattice. See DESIGN.md for the architecture and PLAN.md for build order.

Conventionally imported as::

    import torch_dimensions as td

Public names land here as each phase completes, and optional-dependency blocks
are imported defensively so a CPU-only install stays importable.
"""

from torch_dimensions import data, mixers, testing
from torch_dimensions.compose import (
    ND_METHODS,
    AxialScan,
    axial_apply,
    axial_contract,
    axial_scan,
    kron_operator,
    register_nd_method,
    resolve_nd_method,
)
from torch_dimensions.lattice import Lattice, Restore
from torch_dimensions.models.rnn import GRU, LSTM
from torch_dimensions.plan import ScanPlan, Step
from torch_dimensions.spec import spec

__version__ = "0.0.0"

__all__ = [
    "GRU",
    "LSTM",
    "ND_METHODS",
    "AxialScan",
    "Lattice",
    "Restore",
    "ScanPlan",
    "Step",
    "axial_apply",
    "axial_contract",
    "axial_scan",
    "kron_operator",
    "data",
    "mixers",
    "testing",
    "register_nd_method",
    "resolve_nd_method",
    "spec",
]
