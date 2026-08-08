"""Room geometry and the voxel grid.

Node positions are configuration, not measurement. Nothing detects that a node
was moved 40cm from where the config says it is -- the image simply smears, and
it smears in a way that looks like a plausible detection rather than like an
error. Measure them properly, and re-measure after moving furniture.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from csisense.config import GeometryConfig


@dataclass(frozen=True, slots=True)
class Node:
    id: str
    position: np.ndarray  # (2,) metres


@dataclass(frozen=True, slots=True)
class Link:
    """One transmitter/receiver pair with known endpoints."""

    link_id: str
    tx: np.ndarray
    rx: np.ndarray

    @property
    def distance(self) -> float:
        return float(np.linalg.norm(self.rx - self.tx))


class VoxelGrid:
    """Regular 2D grid over the room.

    2D rather than 3D deliberately: a third dimension multiplies the unknowns
    without adding measurements, and with the link counts realistically
    available the inversion is already massively underdetermined.
    """

    def __init__(self, width_m: float, height_m: float, voxel_m: float) -> None:
        if voxel_m <= 0:
            raise ValueError(f"voxel_m must be positive, got {voxel_m}")
        self.width_m = width_m
        self.height_m = height_m
        self.voxel_m = voxel_m
        self.nx = max(1, int(round(width_m / voxel_m)))
        self.ny = max(1, int(round(height_m / voxel_m)))

        # Voxel centres, row-major (y outer, x inner).
        xs = (np.arange(self.nx) + 0.5) * voxel_m
        ys = (np.arange(self.ny) + 0.5) * voxel_m
        gx, gy = np.meshgrid(xs, ys)
        self.centres = np.column_stack([gx.ravel(), gy.ravel()])

    @property
    def n_voxels(self) -> int:
        return int(self.centres.shape[0])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.ny, self.nx)

    def reshape(self, flat: np.ndarray) -> np.ndarray:
        """Flat voxel vector to a (ny, nx) image."""
        return np.asarray(flat).reshape(self.shape)


class RoomGeometry:
    """Nodes, links and the grid they illuminate."""

    def __init__(self, config: GeometryConfig) -> None:
        self.config = config
        self.nodes = {
            n.id: Node(id=n.id, position=np.asarray(n.position, dtype=np.float64))
            for n in config.nodes
        }
        self.links = [
            Link(
                link_id=link.link_id,
                tx=self.nodes[link.tx].position,
                rx=self.nodes[link.rx].position,
            )
            for link in config.links
            if link.enabled
        ]
        self.grid = VoxelGrid(config.room_size_m[0], config.room_size_m[1], config.voxel_m)

    @property
    def n_links(self) -> int:
        return len(self.links)

    @property
    def link_ids(self) -> tuple[str, ...]:
        return tuple(link.link_id for link in self.links)


def from_config(config: GeometryConfig) -> RoomGeometry:
    return RoomGeometry(config)


__all__ = ["Link", "Node", "RoomGeometry", "VoxelGrid", "from_config"]
