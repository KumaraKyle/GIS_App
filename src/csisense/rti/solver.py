"""Radio tomographic imaging: weights, inversion, and honesty about it.

The model (Wilson & Patwari) is a linear one::

    y = W x + n

``y`` is per-link attenuation change, ``x`` is a per-voxel image of how much the
field changed there, and ``W`` says which voxels a given link is sensitive to.

A link is sensitive to voxels near the ellipse whose foci are its two radios,
because a scatterer there adds little excess path length. Rather than the hard
in/out threshold of the original formulation, this uses a soft Gaussian falloff::

    delta = |p - tx| + |p - rx| - d_link
    w     = (1 / sqrt(d_link)) * exp(-delta^2 / (2 sigma_e^2))

Soft weights are much better conditioned when the number of links is small,
which is the entire regime this system operates in.

Inversion is Tikhonov-regularised with an exponential spatial prior::

    C[j,k] = sigma_x^2 exp(-|p_j - p_k| / delta_c)
    Pi     = (W^T W + alpha C^-1)^-1 W^T

``Pi`` depends only on geometry, so it is computed once and reused; per frame the
cost is a single matrix-vector product.

**The part that matters most is how underdetermined this is.** A 5x4m room at
0.25m is 320 voxels. With 2 links, ``W`` has rank 2, so 318 directions in voxel
space are unconstrained by any measurement and are determined entirely by the
prior. An image produced from that is not a measurement of the room; it is a
picture of the regulariser. That is why ``RtiEngine`` refuses to produce one
below ``min_links`` and draws the link ellipses instead, which is truthful.

The model resolution matrix ``R = Pi W`` is what quantifies this. Its diagonal
says how much of what is reported at a voxel actually came from that voxel, and
the spread of each row says how far a point object there would be smeared.

What the geometry buys you
--------------------------
Measured on a 5x4m room at 0.25m (320 voxels), nodes evenly spaced around the
perimeter and fully meshed, with 5% measurement noise::

    nodes  links  rank  eff.res  room below 3-link coverage  median error
      2      1      -     -        -                          gated, no image
      3      3      -     -        -                          gated, no image
      4      6      6    1.59m    62%                         0.30m
      6     15     15    1.33m    20%                         0.15m
      8     28     28    1.18m     9%                         0.15m

**Treat the error column as a floor, not a prediction.** Those figures come from
simulating measurements with the same weight matrix that is then inverted --
"inverse crime" -- so they measure the solver, not the physics. A real body is
not a point voxel, does not attenuate uniformly, and the elliptical model is an
approximation, which is why published RTI results with comparable node counts
report 0.5-2.85m rather than 15cm.

The honest column is coverage. At four nodes, 62% of the room is illuminated by
fewer than three links, meaning most of it cannot be localised at all regardless
of how good the solver is. That fraction, not the error estimate, is what the UI
shows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from csisense.config import RtiConfig
from csisense.rti.geometry import RoomGeometry


def build_weight_matrix(geometry: RoomGeometry, sigma_e_m: float = 0.25) -> np.ndarray:
    """``(n_links, n_voxels)`` sensitivity matrix."""
    centres = geometry.grid.centres
    W = np.zeros((geometry.n_links, geometry.grid.n_voxels), dtype=np.float64)

    for i, link in enumerate(geometry.links):
        d_tx = np.linalg.norm(centres - link.tx, axis=1)
        d_rx = np.linalg.norm(centres - link.rx, axis=1)
        excess = d_tx + d_rx - link.distance
        # Longer links spread their sensitivity over more area, so each voxel
        # contributes proportionally less.
        norm = 1.0 / np.sqrt(max(link.distance, 1e-6))
        W[i] = norm * np.exp(-(excess**2) / (2.0 * sigma_e_m**2))
    return W


def build_prior_inverse(geometry: RoomGeometry, *, sigma_x: float, delta_c_m: float) -> np.ndarray:
    """Inverse of the exponential spatial covariance prior.

    Encodes "nearby voxels are similar", which is the only thing keeping the
    inversion from producing salt-and-pepper noise when the measurements do not
    constrain it.
    """
    centres = geometry.grid.centres
    dists = np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=2)
    C = (sigma_x**2) * np.exp(-dists / delta_c_m)
    # Jitter for numerical positive-definiteness.
    C += np.eye(C.shape[0]) * 1e-9
    return np.linalg.inv(C)


@dataclass(slots=True)
class Resolution:
    """What the geometry can actually resolve, computed once from R = Pi W."""

    coverage: np.ndarray  # links illuminating each voxel
    spread_m: np.ndarray  # smearing radius per voxel, metres
    confidence: np.ndarray  # 0..1, how much of a voxel's value is its own
    effective_resolution_m: float
    frac_below_min_coverage: float


class TikhonovSolver:
    """Precomputes the inversion for a fixed geometry."""

    def __init__(self, geometry: RoomGeometry, config: RtiConfig) -> None:
        self.geometry = geometry
        self.config = config
        self.W = build_weight_matrix(geometry, config.sigma_e_m)

        if geometry.n_links == 0:
            self.Pi = np.zeros((geometry.grid.n_voxels, 0))
            self.R = np.zeros((geometry.grid.n_voxels, geometry.grid.n_voxels))
        else:
            C_inv = build_prior_inverse(
                geometry, sigma_x=config.sigma_x, delta_c_m=config.delta_c_m
            )
            A = self.W.T @ self.W + config.alpha * C_inv
            self.Pi = np.linalg.solve(A, self.W.T)
            self.R = self.Pi @ self.W

        self.resolution = self._resolution()

    def solve(self, y: np.ndarray) -> np.ndarray:
        """One frame: link measurements to a voxel image."""
        if self.geometry.n_links == 0:
            return np.zeros(self.geometry.grid.n_voxels)
        return self.Pi @ np.asarray(y, dtype=np.float64)

    def _resolution(self) -> Resolution:
        grid = self.geometry.grid
        centres = grid.centres

        coverage = (0.05 * (self.W.max() or 1.0) < self.W).sum(axis=0).astype(np.float64)

        n = grid.n_voxels
        spread = np.zeros(n)
        confidence = np.zeros(n)

        if self.geometry.n_links > 0:
            absR = np.abs(self.R)
            row_sums = absR.sum(axis=1)
            for j in range(n):
                w = absR[j]
                total = row_sums[j]
                if total <= 1e-12:
                    spread[j] = float("inf")
                    continue
                # Second-moment radius of the row: literally "a point object
                # here would be reported this wide".
                delta = centres - centres[j]
                spread[j] = float(np.sqrt((w * (delta**2).sum(axis=1)).sum() / total))
                peak = float(w.max())
                confidence[j] = float(np.clip(absR[j, j] / peak, 0.0, 1.0)) if peak > 0 else 0.0

        covered = coverage >= self.config.min_coverage
        finite = np.isfinite(spread) & covered
        effective = float(spread[finite].mean()) if finite.any() else float("inf")

        return Resolution(
            coverage=coverage,
            spread_m=spread,
            confidence=confidence,
            effective_resolution_m=effective,
            frac_below_min_coverage=float((~covered).mean()),
        )


__all__ = [
    "Resolution",
    "TikhonovSolver",
    "build_prior_inverse",
    "build_weight_matrix",
]
