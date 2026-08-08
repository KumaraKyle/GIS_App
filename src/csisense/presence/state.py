"""The presence state machine.

Three states rather than two, because "motion" and "someone is here" are
genuinely different questions and conflating them is what makes cheap PIR
sensors turn the lights off while you are sitting reading.

    IDLE ---- z > k_on for min_frames ----> MOTION
    MOTION -- z < k_off for still_debounce -> OCCUPIED_STILL
    OCCUPIED_STILL -- z > k_on ------------> MOTION
    OCCUPIED_STILL -- timeout -------------> IDLE
    any ------------ all links unhealthy --> IDLE  (reason: link_lost)

Hysteresis (``k_off`` strictly below ``k_on``) exists so that a statistic
hovering near the threshold latches instead of chattering. Config validation
enforces the ordering.

That last transition matters more than it looks. Without it a dead sensor leaves
Home Assistant holding ``occupancy: on`` forever, the automations never fire,
and the failure is invisible -- the system looks like a house with someone
permanently in it.
"""

from __future__ import annotations

from dataclasses import dataclass

from csisense.events.types import PresenceState


@dataclass(slots=True)
class Transition:
    """A state change, or None-equivalent when nothing happened."""

    previous: PresenceState
    state: PresenceState
    reason: str


class PresenceStateMachine:
    """Turns a stream of z-scores into presence transitions."""

    def __init__(
        self,
        *,
        fs: float = 100.0,
        k_on: float = 6.0,
        k_off: float = 3.0,
        min_frames_on: int = 3,
        still_debounce_s: float = 2.0,
        occupancy_timeout_s: float = 120.0,
        link_loss_timeout_s: float = 10.0,
    ) -> None:
        if k_off >= k_on:
            raise ValueError(f"k_off ({k_off}) must be below k_on ({k_on})")
        self.fs = fs
        self.k_on = k_on
        self.k_off = k_off
        self.min_frames_on = min_frames_on
        self.still_frames = max(1, int(still_debounce_s * fs))
        self.occupancy_frames = max(1, int(occupancy_timeout_s * fs))
        self.link_loss_frames = max(1, int(link_loss_timeout_s * fs))

        self.state = PresenceState.IDLE
        self._above = 0
        self._below = 0
        self._since_motion = 0
        self._unhealthy = 0

    def reset(self) -> None:
        self.state = PresenceState.IDLE
        self._above = 0
        self._below = 0
        self._since_motion = 0
        self._unhealthy = 0

    def update(self, z: float, *, any_link_healthy: bool = True) -> Transition | None:
        """Feed one fused z-score. Returns a Transition only when state changes."""
        if not any_link_healthy:
            self._unhealthy += 1
            if self._unhealthy >= self.link_loss_frames and self.state is not PresenceState.IDLE:
                return self._go(PresenceState.IDLE, "link_lost")
            return None
        self._unhealthy = 0

        self._above = self._above + 1 if z > self.k_on else 0
        self._below = self._below + 1 if z < self.k_off else 0

        if self.state is PresenceState.MOTION:
            self._since_motion = 0
        else:
            self._since_motion += 1

        if self.state is PresenceState.IDLE:
            if self._above >= self.min_frames_on:
                return self._go(PresenceState.MOTION, "motion_started")
            return None

        if self.state is PresenceState.MOTION:
            if self._below >= self.still_frames:
                return self._go(PresenceState.OCCUPIED_STILL, "motion_stopped")
            return None

        # OCCUPIED_STILL
        if self._above >= self.min_frames_on:
            return self._go(PresenceState.MOTION, "motion_resumed")
        if self._since_motion >= self.occupancy_frames:
            return self._go(PresenceState.IDLE, "occupancy_timeout")
        return None

    def _go(self, state: PresenceState, reason: str) -> Transition:
        previous = self.state
        self.state = state
        self._above = 0
        self._below = 0
        if state is PresenceState.MOTION:
            self._since_motion = 0
        if state is PresenceState.IDLE:
            self._unhealthy = 0
        return Transition(previous=previous, state=state, reason=reason)

    @property
    def motion(self) -> bool:
        return self.state is PresenceState.MOTION

    @property
    def occupancy(self) -> bool:
        return self.state is not PresenceState.IDLE


__all__ = ["PresenceStateMachine", "Transition"]
