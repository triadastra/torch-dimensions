"""torch-dimensions — N-dimensional sequence models for PyTorch.

Every model in scope is a 1-D mixer plus a plan for sweeping it over an N-D
lattice. See DESIGN.md for the architecture and PLAN.md for build order.

Conventionally imported as::

    import torch_dimensions as td

Nothing is implemented yet: this is the Phase 0 packaging skeleton. Public
names land here as each phase completes, and optional-dependency blocks are
imported defensively so a CPU-only install stays importable.
"""

__version__ = "0.0.0"

__all__: list[str] = []
