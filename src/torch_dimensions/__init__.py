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
    Flatten,
    axial_apply,
    axial_attention,
    axial_contract,
    axial_scan,
    cafa,
    flatten,
    kron_operator,
    register_nd_method,
    resolve_nd_method,
)
from torch_dimensions.config import (
    MODELS,
    build,
    list_models,
    load,
    read_config,
    register_model,
    save,
)
from torch_dimensions.lattice import Lattice, Restore, Sub
from torch_dimensions.mixers.conv import axis_receptive_field as receptive_field
from torch_dimensions.models.attention import Transformer, TransformerND
from torch_dimensions.models.conv import CNN, CNNND, TCN, TCNND
from torch_dimensions.models.rnn import GRU, LSTM
from torch_dimensions.models.ssm import S4, S4D, S4DND, S4ND, Mamba, MambaND
from torch_dimensions.models.vit import PatchEmbed, ViT
from torch_dimensions.plan import AxisCoverage, Coverage, ScanPlan, Step
from torch_dimensions.spec import SPEC_VERSION as spec_version
from torch_dimensions.spec import spec

__version__ = "0.1.0"

__all__ = [
    "CNN",
    "CNNND",
    "GRU",
    "LSTM",
    "TCN",
    "TCNND",
    "Mamba",
    "MambaND",
    "S4",
    "S4D",
    "S4DND",
    "S4ND",
    "PatchEmbed",
    "Transformer",
    "TransformerND",
    "ViT",
    "MODELS",
    "ND_METHODS",
    "AxialKernel",
    "AxialScan",
    "Flatten",
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
    "flatten",
    "kron_operator",
    "data",
    "list_models",
    "load",
    "read_config",
    "mixers",
    "receptive_field",
    "testing",
    "viz",
    "register_model",
    "register_nd_method",
    "save",
    "resolve_nd_method",
    "spec",
    "spec_version",
]
