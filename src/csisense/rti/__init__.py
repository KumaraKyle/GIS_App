"""Radio tomographic imaging.

Produces a blob heatmap of where the field changed -- the honest version of
"seeing someone through a wall". It is not a body outline and cannot become one:
that needs antenna arrays and camera-supervised training, and does not generalise
across rooms even then.
"""

from csisense.rti.engine import RtiEngine, RtiFrame, RtiStatus
from csisense.rti.geometry import Link, Node, RoomGeometry, VoxelGrid, from_config
from csisense.rti.solver import Resolution, TikhonovSolver, build_weight_matrix

__all__ = [
    "Link",
    "Node",
    "Resolution",
    "RoomGeometry",
    "RtiEngine",
    "RtiFrame",
    "RtiStatus",
    "TikhonovSolver",
    "VoxelGrid",
    "build_weight_matrix",
    "from_config",
]
