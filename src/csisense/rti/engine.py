"""The RTI engine, and its refusal to over-claim.

Two modes:

``imaging``
    Enough links to invert. Produces a voxel image plus per-voxel confidence,
    and the caller is expected to render confidence as alpha so that
    poorly-constrained regions fade out rather than appearing as blue.

``ellipses``
    Fewer than ``min_links``. Produces no image at all -- only each link's
    sensitivity ellipse coloured by that link's z-score. With one or two links
    this is the *complete* truth available: you know something disturbed a
    particular link, and therefore that it is somewhere on that ellipse. Drawing
    a blob would be drawing the prior.

Why the gate exists, concretely: a 5x4m room at 0.25m resolution is 320 voxels.
Two links give a rank-2 weight matrix, so 318 dimensions of the answer come from
the regulariser and none from the radio. The resulting picture looks like a
detection, moves when someone moves, and is mostly fabricated. Since the
uncertainty is not visible in the output, the only honest option is not to
produce it.

Adding nodes lifts the gate automatically -- there is no code change, only more
entries under ``geometry.nodes``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from csisense.config import AppConfig
from csisense.rti.geometry import RoomGeometry, from_config
from csisense.rti.solver import TikhonovSolver

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RtiStatus:
    """Whether imaging is possible, and what it would be worth."""

    enabled: bool
    mode: str
    n_links: int
    min_links: int
    reason: str = ""
    effective_resolution_m: float = float("inf")
    frac_below_min_coverage: float = 1.0

    def caption(self) -> str:
        """One line the UI shows permanently, never hidden behind a tooltip."""
        if not self.enabled:
            return self.reason
        return (
            f"{self.n_links} links · effective resolution ≈ "
            f"{self.effective_resolution_m:.1f} m · "
            f"{self.frac_below_min_coverage * 100:.0f}% of room below "
            f"{self.min_links}-link coverage"
        )


@dataclass(slots=True)
class RtiFrame:
    """One reconstruction, or the ellipse fallback."""

    ts_ns: int
    mode: str
    # (ny, nx) image, or None in ellipse mode.
    image: np.ndarray | None = None
    confidence: np.ndarray | None = None
    coverage: np.ndarray | None = None
    # Per-link z-scores, which is all there is to show in ellipse mode.
    per_link: dict[str, float] = field(default_factory=dict)
    peak_xy: tuple[float, float] | None = None
    peak_confidence: float = 0.0


class RtiEngine:
    """Turns per-link motion z-scores into an image, when that is justified."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.geometry: RoomGeometry = from_config(config.geometry)
        self.rti_config = config.rti
        self._solver: TikhonovSolver | None = None

        if self.geometry.n_links >= self.rti_config.min_links:
            self._solver = TikhonovSolver(self.geometry, self.rti_config)
            log.info(
                "RTI enabled: %d links, effective resolution %.2f m",
                self.geometry.n_links,
                self._solver.resolution.effective_resolution_m,
            )
        elif self.geometry.n_links:
            log.info(
                "RTI in ellipse mode: %d link(s), needs %d to invert",
                self.geometry.n_links,
                self.rti_config.min_links,
            )

    @property
    def solver(self) -> TikhonovSolver | None:
        return self._solver

    def status(self) -> RtiStatus:
        n = self.geometry.n_links
        if not self.rti_config.enabled:
            return RtiStatus(False, "off", n, self.rti_config.min_links, "RTI disabled in config")
        if self._solver is None:
            return RtiStatus(
                enabled=False,
                mode="ellipses",
                n_links=n,
                min_links=self.rti_config.min_links,
                reason=(
                    f"RTI needs ≥ {self.rti_config.min_links} links to invert; {n} configured. "
                    "Showing which links are disturbed instead — with this many links a "
                    "position estimate would come from the smoothness prior, not the radios."
                ),
            )
        res = self._solver.resolution
        return RtiStatus(
            enabled=True,
            mode="imaging",
            n_links=n,
            min_links=self.rti_config.min_links,
            effective_resolution_m=res.effective_resolution_m,
            frac_below_min_coverage=res.frac_below_min_coverage,
        )

    def update(self, ts_ns: int, per_link_z: dict[str, float]) -> RtiFrame:
        """Reconstruct from the presence detector's per-link z-scores.

        z-scores rather than raw dB, because each link is then normalised by its
        own measured noise -- which is what makes it legitimate to combine the
        Pi's 5GHz link with the ESP32s' 2.4GHz links in one inversion at all.
        """
        if self._solver is None:
            return RtiFrame(ts_ns=ts_ns, mode="ellipses", per_link=dict(per_link_z))

        y = np.array(
            [max(per_link_z.get(lid, 0.0), 0.0) for lid in self.geometry.link_ids],
            dtype=np.float64,
        )
        flat = self._solver.solve(y)
        res = self._solver.resolution

        # Mask voxels too poorly covered to mean anything before picking a peak,
        # otherwise the argmax lands in a corner no link illuminates.
        covered = res.coverage >= self.rti_config.min_coverage
        masked = np.where(covered, flat, -np.inf)

        peak_xy = None
        peak_conf = 0.0
        if covered.any() and np.isfinite(masked).any():
            idx = int(np.argmax(masked))
            if flat[idx] > 0:
                peak_xy = (
                    float(self.geometry.grid.centres[idx, 0]),
                    float(self.geometry.grid.centres[idx, 1]),
                )
                peak_conf = float(res.confidence[idx])

        return RtiFrame(
            ts_ns=ts_ns,
            mode="imaging",
            image=self.geometry.grid.reshape(flat),
            confidence=self.geometry.grid.reshape(res.confidence),
            coverage=self.geometry.grid.reshape(res.coverage),
            per_link=dict(per_link_z),
            peak_xy=peak_xy,
            peak_confidence=peak_conf,
        )

    def link_ellipse(self, link_id: str, n_points: int = 64) -> np.ndarray | None:
        """Points tracing a link's sensitivity ellipse, for the fallback view.

        This is the honest one-and-two-link visualisation: a disturbance on this
        link means something is somewhere on this curve.
        """
        link = next((x for x in self.geometry.links if x.link_id == link_id), None)
        if link is None:
            return None

        centre = (link.tx + link.rx) / 2.0
        axis = link.rx - link.tx
        d = float(np.linalg.norm(axis))
        if d < 1e-9:
            return None
        angle = float(np.arctan2(axis[1], axis[0]))

        # Semi-axes for the locus where excess path length equals sigma_e.
        a = (d + self.rti_config.sigma_e_m) / 2.0
        b = float(np.sqrt(max(a**2 - (d / 2.0) ** 2, 1e-9)))

        t = np.linspace(0, 2 * np.pi, n_points)
        pts = np.column_stack([a * np.cos(t), b * np.sin(t)])
        rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        return pts @ rot.T + centre


__all__ = ["RtiEngine", "RtiFrame", "RtiStatus"]
