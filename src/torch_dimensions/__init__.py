"""torch-dimensions — N-dimensional sequence models for PyTorch.

Every model in scope is a 1-D mixer plus a plan for sweeping it over an N-D
lattice. See DESIGN.md for the architecture and PLAN.md for build order.

Conventionally imported as::

    import torch_dimensions as td

Public names land here as each phase completes, and optional-dependency blocks
are imported defensively so a CPU-only install stays importable.
"""

from torch_dimensions import mixers
from torch_dimensions.compose.scan import AxialScan, axial_apply
from torch_dimensions.lattice import Lattice, Restore
from torch_dimensions.models.rnn_nd import GRUND, LSTMND
from torch_dimensions.plan import ScanPlan, Step

__version__ = "0.0.0"

__all__ = [
    "GRUND",
    "LSTMND",
    "AxialScan",
    "Lattice",
    "Restore",
    "ScanPlan",
    "Step",
    "axial_apply",
    "mixers",
]
