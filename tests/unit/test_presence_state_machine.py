"""State machine behaviour, driven from a literal table.

The machine is the part users actually experience: it decides when the lights
come on and, more importantly, when they are allowed to go off. Every transition
here corresponds to a real complaint someone would make about a motion sensor.
"""

from __future__ import annotations

import pytest

from csisense.events.types import PresenceState
from csisense.presence.state import PresenceStateMachine

FS = 100.0


def machine(**kw) -> PresenceStateMachine:
    params = {
        "fs": FS,
        "k_on": 6.0,
        "k_off": 3.0,
        "min_frames_on": 3,
        "still_debounce_s": 2.0,
        "occupancy_timeout_s": 120.0,
        "link_loss_timeout_s": 10.0,
    }
    params.update(kw)
    return PresenceStateMachine(**params)


def drive(m: PresenceStateMachine, z: float, n: int, *, healthy: bool = True) -> list:
    return [t for _ in range(n) if (t := m.update(z, any_link_healthy=healthy)) is not None]


def test_starts_idle():
    assert machine().state is PresenceState.IDLE


def test_requires_consecutive_frames_to_fire():
    """One noisy frame above threshold is not a person."""
    m = machine(min_frames_on=3)
    assert m.update(10.0) is None
    assert m.update(10.0) is None
    t = m.update(10.0)
    assert t is not None and t.state is PresenceState.MOTION


def test_a_single_spike_does_not_fire():
    m = machine(min_frames_on=3)
    for z in (10.0, 0.0, 10.0, 0.0, 10.0, 0.0):
        assert m.update(z) is None
    assert m.state is PresenceState.IDLE


def test_motion_relaxes_to_occupied_still_not_idle():
    """The central design decision: someone who stops moving is still there.
    Collapsing this into IDLE is what makes cheap sensors switch the lights off
    while you are sitting reading."""
    m = machine()
    drive(m, 10.0, 5)
    assert m.state is PresenceState.MOTION

    transitions = drive(m, 0.0, int(2.0 * FS) + 5)
    assert transitions[-1].state is PresenceState.OCCUPIED_STILL
    assert m.occupancy is True
    assert m.motion is False


def test_occupancy_survives_a_long_stillness_then_times_out():
    m = machine(occupancy_timeout_s=5.0)
    drive(m, 10.0, 5)
    drive(m, 0.0, int(2.0 * FS) + 5)
    assert m.state is PresenceState.OCCUPIED_STILL

    # Still occupied a second later.
    drive(m, 0.0, int(1.0 * FS))
    assert m.occupancy is True

    transitions = drive(m, 0.0, int(5.0 * FS))
    assert transitions[-1].state is PresenceState.IDLE
    assert transitions[-1].reason == "occupancy_timeout"


def test_motion_resumes_from_occupied_still():
    m = machine()
    drive(m, 10.0, 5)
    drive(m, 0.0, int(2.0 * FS) + 5)
    assert m.state is PresenceState.OCCUPIED_STILL

    t = drive(m, 12.0, 5)
    assert t and t[0].state is PresenceState.MOTION
    assert t[0].reason == "motion_resumed"


def test_hysteresis_prevents_chatter():
    """A statistic parked between k_off and k_on must latch, not oscillate."""
    m = machine(k_on=6.0, k_off=3.0)
    drive(m, 10.0, 5)
    transitions = drive(m, 4.5, 10 * int(FS))  # between the thresholds
    assert transitions == []
    assert m.state is PresenceState.MOTION


def test_k_off_must_be_below_k_on():
    with pytest.raises(ValueError, match="below"):
        machine(k_on=3.0, k_off=5.0)


def test_link_loss_forces_idle():
    """Without this a dead sensor leaves Home Assistant holding occupancy
    forever, the automations never fire, and nothing looks broken."""
    m = machine(link_loss_timeout_s=2.0)
    drive(m, 10.0, 5)
    assert m.state is PresenceState.MOTION

    transitions = drive(m, 0.0, int(2.0 * FS) + 5, healthy=False)
    assert transitions[-1].state is PresenceState.IDLE
    assert transitions[-1].reason == "link_lost"


def test_brief_link_loss_does_not_force_idle():
    """Momentary loss is normal on Wi-Fi; only sustained loss means failure."""
    m = machine(link_loss_timeout_s=10.0)
    drive(m, 10.0, 5)
    assert drive(m, 0.0, int(1.0 * FS), healthy=False) == []
    assert m.state is PresenceState.MOTION


def test_unhealthy_while_idle_produces_nothing():
    assert drive(machine(), 0.0, 5000, healthy=False) == []


def test_reset_returns_to_idle():
    m = machine()
    drive(m, 10.0, 5)
    m.reset()
    assert m.state is PresenceState.IDLE


@pytest.mark.parametrize(
    ("sequence", "expected"),
    [
        # (z, repeats) pairs -> final state
        ([(0.0, 500)], PresenceState.IDLE),
        ([(10.0, 5)], PresenceState.MOTION),
        ([(10.0, 5), (0.0, 300)], PresenceState.OCCUPIED_STILL),
        ([(10.0, 5), (0.0, 300), (10.0, 5)], PresenceState.MOTION),
        ([(4.0, 1000)], PresenceState.IDLE),  # below k_on, never fires
    ],
)
def test_transition_table(sequence, expected):
    m = machine()
    for z, n in sequence:
        drive(m, z, n)
    assert m.state is expected
