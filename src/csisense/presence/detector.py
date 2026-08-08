"""Presence detection: the pipeline-facing object.

Owns one :class:`NacdStatistic` per link, the calibrated baselines, and the
state machine, and turns DSP features into :class:`PresenceEvent`s.

**Fusion across links is max, not mean.** A person standing near one link
perturbs that link and barely touches the others, so averaging divides the
signal by the number of links -- meaning a mean-based detector gets *worse*
every time you add a sensor. Max keeps sensitivity constant as the deployment
grows, which is what makes the whole design link-count-agnostic.

Per-link z-scores are published alongside the fused value. With enough links
that is a position estimate (see ``rti/``); with two it is still the honest
answer to "which direction did that come from", because the link that fired
tells you which part of the room was disturbed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from csisense.config import PresenceConfig
from csisense.dsp.chain import DspFeatures
from csisense.events.types import PresenceEvent, PresenceState
from csisense.presence.baseline import CalibrationRun, LinkBaseline
from csisense.presence.state import PresenceStateMachine
from csisense.presence.statistic import NacdStatistic

log = logging.getLogger(__name__)


@dataclass(slots=True)
class LinkReading:
    """Latest per-link values, for diagnostics and the dashboard."""

    raw: float
    z: float
    energy: float
    coherence: float
    healthy: bool = True


class PresenceDetector:
    """Fuses per-link motion statistics into presence state."""

    def __init__(
        self,
        config: PresenceConfig,
        *,
        fs: float = 100.0,
        baselines: dict[str, LinkBaseline] | None = None,
    ) -> None:
        self.config = config
        self.fs = fs
        self.baselines: dict[str, LinkBaseline] = dict(baselines or {})
        self.readings: dict[str, LinkReading] = {}

        self._stats: dict[str, NacdStatistic] = {}
        self._healthy: dict[str, bool] = {}
        self._calibration: CalibrationRun | None = None

        self.machine = PresenceStateMachine(
            fs=fs,
            k_on=config.k_on,
            k_off=config.k_off,
            min_frames_on=config.min_frames_on,
            still_debounce_s=config.still_debounce_s,
            occupancy_timeout_s=config.occupancy_timeout_s,
            link_loss_timeout_s=config.link_loss_timeout_s,
        )

    # -- calibration -------------------------------------------------------

    def start_calibration(self, duration_s: float | None = None) -> CalibrationRun:
        """Begin measuring the quiet room. The room must actually be empty."""
        self._calibration = CalibrationRun(
            duration_s=duration_s if duration_s is not None else self.config.calibration_s,
            fs=self.fs,
        )
        log.info("calibration started: %.0fs, room should be empty", self._calibration.duration_s)
        return self._calibration

    @property
    def calibrating(self) -> bool:
        return self._calibration is not None

    @property
    def calibrated(self) -> bool:
        return bool(self.baselines)

    def calibration_progress(self) -> float:
        return self._calibration.progress() if self._calibration else 0.0

    def _finish_calibration(self) -> None:
        assert self._calibration is not None
        self.baselines = self._calibration.finish()
        self._calibration = None
        self.machine.reset()
        log.info(
            "calibration complete for %d link(s): %s",
            len(self.baselines),
            ", ".join(f"{k} mu={v.mu:.4f} sigma={v.sigma:.4f}" for k, v in self.baselines.items()),
        )

    # -- runtime -----------------------------------------------------------

    def set_link_health(self, link_id: str, healthy: bool) -> None:
        self._healthy[link_id] = healthy

    def _statistic(self, link_id: str) -> NacdStatistic:
        stat = self._stats.get(link_id)
        if stat is None:
            stat = NacdStatistic(
                fs=self.fs,
                tau_long_s=self.config.tau_long_s,
                tau_short_s=self.config.tau_short_s,
                lag_frames=self.config.deviation_lag_frames,
                coherence_gate=self.config.coherence_gate,
            )
            self._stats[link_id] = stat
        return stat

    def update(self, features: DspFeatures) -> PresenceEvent | None:
        """Process one link's features. Returns an event only on a state change."""
        link_id = features.link_id
        stat = self._statistic(link_id)
        sample = stat.update(features.amplitude)

        if self._calibration is not None:
            if stat.warm:
                self._calibration.add(link_id, sample.value)
            if self._calibration.complete():
                self._finish_calibration()
            return None

        baseline = self.baselines.get(link_id)
        healthy = self._healthy.get(link_id, True)
        z = baseline.z(sample.value) if baseline else 0.0

        self.readings[link_id] = LinkReading(
            raw=sample.value,
            z=z,
            energy=sample.energy,
            coherence=sample.coherence,
            healthy=healthy,
        )

        if not self.calibrated:
            # Without a baseline there is no scale, so any threshold would be
            # arbitrary. Stay silent rather than guessing.
            return None

        usable = [r.z for r in self.readings.values() if r.healthy]
        any_healthy = bool(usable)
        # Max, not mean: see the module docstring.
        fused = max(usable) if usable else 0.0

        transition = self.machine.update(fused, any_link_healthy=any_healthy)

        # Freeze every link's long-term reference while anyone is present,
        # otherwise a motionless person is absorbed into it and disappears.
        if self.config.freeze_baseline_when_active:
            frozen = self.machine.state is not PresenceState.IDLE
            for s in self._stats.values():
                s.freeze(frozen)

        if transition is None:
            return None

        return PresenceEvent(
            ts_ns=features.ts_ns,
            state=transition.state,
            previous=transition.previous,
            motion_stat=fused,
            per_link={k: v.z for k, v in self.readings.items()},
            reason=transition.reason,
        )

    @property
    def state(self) -> PresenceState:
        return self.machine.state

    @property
    def motion_stat(self) -> float:
        usable = [r.z for r in self.readings.values() if r.healthy]
        return max(usable) if usable else 0.0

    @property
    def blind(self) -> bool:
        """True when no link is healthy enough to be believed.

        A blind detector reports "no motion", which is indistinguishable
        downstream from "empty room" -- so this must be surfaced explicitly.
        Consumers should mark the entity *unavailable* rather than publishing a
        confident negative: silently reporting an empty house because the sensor
        died is the worst failure mode this system has.
        """
        if not self.calibrated or not self.readings:
            return True
        return not any(r.healthy for r in self.readings.values())


__all__ = ["LinkReading", "PresenceDetector"]
