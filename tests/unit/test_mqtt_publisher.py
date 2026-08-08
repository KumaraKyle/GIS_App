"""MQTT publishing behaviour.

Uses a recording fake rather than a broker, because the behaviours worth testing
are about *what* gets published and with what retain/qos flags -- not about
whether paho can open a socket.
"""

from __future__ import annotations

import json

import pytest

from csisense.config import AppConfig
from csisense.events.types import (
    GestureEvent,
    LinkQualityEvent,
    PresenceEvent,
    PresenceState,
)
from csisense.mqtt.publisher import MqttPublisher
from csisense.mqtt.topics import Topics


class FakeTransport:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str, bool, int]] = []

    async def publish(self, topic, payload, *, qos=0, retain=False):
        text = payload.decode() if isinstance(payload, bytes) else payload
        self.messages.append((topic, text, retain, qos))

    def topics(self) -> list[str]:
        return [m[0] for m in self.messages]

    def last(self, topic: str) -> tuple[str, bool, int]:
        for t, payload, retain, qos in reversed(self.messages):
            if t == topic:
                return payload, retain, qos
        raise AssertionError(f"nothing published to {topic}; got {sorted(set(self.topics()))}")


@pytest.fixture
def setup():
    config = AppConfig()
    transport = FakeTransport()
    return config, transport, MqttPublisher(config, transport, sw_version="0.1.0")


def presence_event(state: PresenceState, previous=PresenceState.IDLE, stat=7.5):
    return PresenceEvent(
        ts_ns=1_000_000_000,
        state=state,
        previous=previous,
        motion_stat=stat,
        per_link={"esp:0": stat, "esp:1": 1.2},
        reason="motion_started",
    )


# --- lifecycle ------------------------------------------------------------


async def test_connect_publishes_discovery_and_goes_online(setup):
    config, transport, pub = setup
    await pub.on_connect(link_ids=("esp:0",))

    topics = Topics(config.mqtt.base_topic, config.deployment.id)
    payload, retain, _ = transport.last(topics.availability)
    assert payload == "online"
    assert retain, "availability must be retained or HA sees nothing until the next change"
    assert any("/config" in t for t in transport.topics())


async def test_discovery_is_retained(setup):
    _, transport, pub = setup
    await pub.on_connect(link_ids=("esp:0",))
    for topic, _payload, retain, _qos in transport.messages:
        if topic.endswith("/config"):
            assert retain, f"{topic} is not retained"


async def test_reconnect_republishes_discovery(setup):
    """Home Assistant forgets non-retained discovery on restart, and a broker
    without persistence drops retained messages. Republishing is the only thing
    that reliably brings the entities back."""
    _, transport, pub = setup
    await pub.on_connect(link_ids=("esp:0",))
    first = len([t for t in transport.topics() if t.endswith("/config")])

    transport.messages.clear()
    await pub.on_connect(link_ids=("esp:0",))
    second = len([t for t in transport.topics() if t.endswith("/config")])

    assert second == first


# --- presence -------------------------------------------------------------


async def test_presence_publishes_both_entities(setup):
    config, transport, pub = setup
    topics = Topics(config.mqtt.base_topic, config.deployment.id)

    await pub.publish_event(presence_event(PresenceState.MOTION))

    assert transport.last(topics.presence_motion)[0] == "ON"
    assert transport.last(topics.presence_occupancy)[0] == "ON"


async def test_occupancy_stays_on_when_motion_stops(setup):
    """The whole reason for three states: someone sitting still is present."""
    config, transport, pub = setup
    topics = Topics(config.mqtt.base_topic, config.deployment.id)

    await pub.publish_event(
        presence_event(PresenceState.OCCUPIED_STILL, previous=PresenceState.MOTION)
    )

    assert transport.last(topics.presence_motion)[0] == "OFF"
    assert transport.last(topics.presence_occupancy)[0] == "ON"


async def test_idle_clears_both(setup):
    config, transport, pub = setup
    topics = Topics(config.mqtt.base_topic, config.deployment.id)

    await pub.publish_event(
        presence_event(PresenceState.IDLE, previous=PresenceState.OCCUPIED_STILL)
    )

    assert transport.last(topics.presence_motion)[0] == "OFF"
    assert transport.last(topics.presence_occupancy)[0] == "OFF"


async def test_presence_transitions_use_qos1(setup):
    """These are automation triggers. Losing one loses an automation."""
    config, transport, pub = setup
    topics = Topics(config.mqtt.base_topic, config.deployment.id)

    await pub.publish_event(presence_event(PresenceState.MOTION))

    assert transport.last(topics.presence_motion)[2] == 1
    assert transport.last(topics.presence_occupancy)[2] == 1


async def test_presence_attributes_include_per_link(setup):
    """Which link fired is the crude directionality signal."""
    config, transport, pub = setup
    topics = Topics(config.mqtt.base_topic, config.deployment.id)

    await pub.publish_event(presence_event(PresenceState.MOTION))
    payload = json.loads(transport.last(topics.presence_state)[0])

    assert payload["per_link"] == {"esp:0": 7.5, "esp:1": 1.2}
    assert payload["reason"] == "motion_started"
    assert payload["state"] == "motion"


async def test_attributes_are_published_before_state(setup):
    """A template reading attributes alongside state must not see the previous
    attributes paired with the new state."""
    config, transport, pub = setup
    topics = Topics(config.mqtt.base_topic, config.deployment.id)

    await pub.publish_event(presence_event(PresenceState.MOTION))
    order = transport.topics()

    assert order.index(topics.presence_state) < order.index(topics.presence_motion)


# --- availability ---------------------------------------------------------


async def test_blindness_publishes_unavailable(setup):
    """A detector that cannot see must not report a confident 'clear'."""
    config, transport, pub = setup
    topics = Topics(config.mqtt.base_topic, config.deployment.id)

    await pub.set_available(True)
    await pub.set_available(False)

    assert transport.last(topics.availability)[0] == "offline"


async def test_availability_is_not_republished_unnecessarily(setup):
    config, transport, pub = setup
    topics = Topics(config.mqtt.base_topic, config.deployment.id)

    await pub.set_available(True)
    await pub.set_available(True)
    await pub.set_available(True)

    assert len([t for t in transport.topics() if t == topics.availability]) == 1


# --- gestures -------------------------------------------------------------


async def test_gesture_publishes_sensor_and_trigger(setup):
    config, transport, pub = setup
    topics = Topics(config.mqtt.base_topic, config.deployment.id)

    await pub.publish_event(
        GestureEvent(
            ts_ns=1,
            tier="coarse",
            gesture="sit",
            confidence=0.91,
            energy=-5.2,
            model_version="v3",
        )
    )

    sensor_payload, sensor_retain, _ = transport.last(topics.gesture("coarse"))
    trigger_payload, trigger_retain, trigger_qos = transport.last(topics.gesture_event("coarse"))

    assert json.loads(sensor_payload)["gesture"] == "sit"
    assert sensor_retain, "the history sensor should survive a restart"
    assert not trigger_retain, "a retained trigger re-fires automations on every HA restart"
    assert trigger_qos == 1


async def test_gesture_carries_confidence_and_energy(setup):
    """Both thresholds are what keep false positives to ~1/day, so both belong
    in the payload for tuning."""
    config, transport, pub = setup
    topics = Topics(config.mqtt.base_topic, config.deployment.id)

    await pub.publish_event(
        GestureEvent(ts_ns=1, tier="fine", gesture="swipe_left", confidence=0.83, energy=-4.1)
    )
    payload = json.loads(transport.last(topics.gesture("fine"))[0])

    assert payload["confidence"] == 0.83
    assert payload["energy"] == -4.1


# --- diagnostics ----------------------------------------------------------


async def test_link_quality_reports_windowed_and_lifetime_loss(setup):
    config, transport, pub = setup
    topics = Topics(config.mqtt.base_topic, config.deployment.id)

    await pub.publish_event(
        LinkQualityEvent(
            ts_ns=1,
            link_id="esp:0",
            rate_hz=99.6,
            loss_pct=4.2,
            loss_pct_1m=0.3,
            rssi=-52.4,
            gain_locked=True,
        )
    )
    payload = json.loads(transport.last(topics.link_quality("esp:0"))[0])

    assert payload["loss_pct_1m"] == 0.3
    assert payload["loss_pct_total"] == 4.2
    assert payload["gain_locked"] is True


async def test_motion_stat_is_rate_limited(setup):
    """2Hz telemetry, not 100Hz. Publishing every frame would flood the broker
    and gain nothing."""
    _, transport, pub = setup

    for i in range(100):
        await pub.publish_motion_stat(i * 10_000_000, float(i))  # 100Hz

    assert len(transport.messages) <= 4, f"published {len(transport.messages)} stat messages"


async def test_config_state_formats_booleans_for_home_assistant(setup):
    config, transport, pub = setup
    topics = Topics(config.mqtt.base_topic, config.deployment.id)

    await pub.publish_config_state("gestures_enabled", True)
    await pub.publish_config_state("k_on", 6.5)

    assert transport.last(topics.config_state("gestures_enabled"))[0] == "ON"
    assert transport.last(topics.config_state("k_on"))[0] == "6.5"
