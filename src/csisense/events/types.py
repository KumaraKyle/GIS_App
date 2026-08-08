"""Events emitted by the pipeline.

These are the pipeline's only output. MQTT publishing, the WebSocket feed and
the golden regression tests all consume this same list, so an event must carry
everything a consumer needs without reaching back into pipeline state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class PresenceState(StrEnum):
    """Presence is deliberately three-valued.

    ``OCCUPIED_STILL`` is the state that separates a useful sensor from an
    annoying one: a person who stops moving is still in the room, and collapsing
    this into IDLE is what makes cheap motion sensors turn the lights off on you.
    """

    IDLE = "idle"
    MOTION = "motion"
    OCCUPIED_STILL = "occupied_still"


@dataclass(frozen=True, slots=True)
class PresenceEvent:
    """A presence state transition."""

    ts_ns: int
    state: PresenceState
    previous: PresenceState
    motion_stat: float
    # Per-link statistics. Which link fired is the crude directionality signal
    # that answers "are they approaching from the hallway?".
    per_link: dict[str, float] = field(default_factory=dict)

    @property
    def motion(self) -> bool:
        return self.state is PresenceState.MOTION

    @property
    def occupancy(self) -> bool:
        return self.state is not PresenceState.IDLE


@dataclass(frozen=True, slots=True)
class GestureEvent:
    """A recognised gesture.

    ``confidence`` is calibrated (temperature-scaled), not a raw softmax, so it
    is meaningful to threshold on. ``energy`` is the out-of-distribution score;
    a low-energy sample is one the model has genuinely seen something like
    before, as opposed to being confidently wrong about noise.
    """

    ts_ns: int
    tier: str
    gesture: str
    confidence: float
    energy: float
    model_version: str = ""


@dataclass(frozen=True, slots=True)
class LinkQualityEvent:
    """Transport health for one link. Diagnostic, not signal."""

    ts_ns: int
    link_id: str
    rate_hz: float
    loss_pct: float
    rssi: float
    # False when the radio's gain was free-running, which makes amplitude
    # features untrustworthy and disqualifies the stretch from training use.
    gain_locked: bool = False


Event = PresenceEvent | GestureEvent | LinkQualityEvent

__all__ = [
    "Event",
    "GestureEvent",
    "LinkQualityEvent",
    "PresenceEvent",
    "PresenceState",
]
