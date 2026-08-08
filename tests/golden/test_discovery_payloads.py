"""Home Assistant discovery payloads, pinned to a golden file.

Discovery is the part that breaks silently. A schema change in Home Assistant,
or a careless edit here, produces entities that either never appear or appear
and never update -- and both look identical to a sensor that stopped working.
Comparing generated payloads against a checked-in file is the only way to notice
at test time rather than at 11pm.

Regenerate deliberately, never reflexively:

    python -m tests.golden.regen_discovery
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from csisense.config import AppConfig
from csisense.mqtt.discovery import discovery_payloads
from csisense.mqtt.topics import Topics, slug

GOLDEN = Path(__file__).parent / "data" / "discovery.json"

LINK_IDS = ("nexmon:0:c0s0", "esp:1")
GESTURE_CLASSES = {
    "coarse": ("walk_in", "walk_out", "sit", "stand", "fall", "arm_wave", "none"),
    "fine": ("swipe_left", "swipe_right", "push", "pull", "circle", "none"),
}


def generate() -> dict[str, dict]:
    config = AppConfig()
    payloads = discovery_payloads(
        config,
        link_ids=LINK_IDS,
        gesture_classes=GESTURE_CLASSES,
        sw_version="0.1.0",
    )
    return {topic: payload for topic, payload in payloads}


def test_matches_golden_file():
    if not GOLDEN.exists():  # pragma: no cover
        pytest.fail("golden file missing; regenerate with tests/golden/regen_discovery.py")
    expected = json.loads(GOLDEN.read_text())
    assert generate() == expected


def test_ordering_is_stable():
    """Republishing after a reconnect must produce identical bytes, or every
    reconnect churns retained messages in the broker."""
    assert list(generate()) == list(generate())


def test_every_payload_is_json_serialisable():
    for topic, payload in generate().items():
        json.dumps(payload), topic


def test_presence_is_two_entities():
    """motion is fast and transient, occupancy is sticky. Collapsing them loses
    the distinction that stops the lights going off while you sit still."""
    payloads = generate()
    motion = [k for k in payloads if k.endswith("/motion/config")]
    occupancy = [k for k in payloads if k.endswith("/occupancy/config")]

    assert len(motion) == 1
    assert len(occupancy) == 1
    assert payloads[motion[0]]["device_class"] == "motion"
    assert payloads[occupancy[0]]["device_class"] == "occupancy"


def test_gestures_are_device_triggers_not_sensors():
    """A gesture is a stateless event. As a sensor it would need a value between
    gestures, which re-fires automations on restart."""
    payloads = generate()
    triggers = [k for k in payloads if "/device_automation/" in k]

    assert triggers, "no device_automation entries"
    for key in triggers:
        p = payloads[key]
        assert p["automation_type"] == "trigger"
        assert p["type"] == "gesture"
        assert p["subtype"]


def test_background_class_is_not_a_trigger():
    """'none' is the model's background class. Firing automations on it would be
    firing on the absence of a gesture."""
    payloads = generate()
    assert not any("_none/config" in k for k in payloads)


def test_every_stateful_entity_has_availability():
    """When the detector goes blind, entities must become unavailable rather
    than reporting a confident 'clear'. A dead sensor reading as an empty house
    is the worst failure this system has."""
    payloads = generate()
    for topic, payload in payloads.items():
        if "/device_automation/" in topic:
            continue  # triggers have no state to be unavailable
        assert "availability_topic" in payload, f"{topic} has no availability_topic"


def test_all_entities_share_one_device():
    """Otherwise Home Assistant shows a dozen unrelated devices."""
    payloads = generate()
    identifiers = {tuple(p["device"]["identifiers"]) for p in payloads.values()}
    assert len(identifiers) == 1


def test_unique_ids_are_unique():
    payloads = generate()
    uids = [p["unique_id"] for p in payloads.values() if "unique_id" in p]
    assert len(uids) == len(set(uids))


def test_link_ids_are_slugged_for_home_assistant():
    """Link ids contain colons, which are legal in a topic but not in a
    unique_id."""
    payloads = generate()
    for payload in payloads.values():
        assert ":" not in payload.get("unique_id", "")


def test_topics_match_what_the_publisher_uses():
    """The failure this prevents: entities that exist and never update."""
    config = AppConfig()
    topics = Topics(config.mqtt.base_topic, config.deployment.id)
    payloads = generate()

    motion = next(p for k, p in payloads.items() if k.endswith("/motion/config"))
    assert motion["state_topic"] == topics.presence_motion

    stat = next(p for k, p in payloads.items() if k.endswith("/motion_stat/config"))
    assert stat["state_topic"] == topics.presence_stat

    link = next(p for k, p in payloads.items() if "link_nexmon_0_c0s0_rate_hz" in k)
    assert link["state_topic"] == topics.link_quality("nexmon:0:c0s0")


def test_controls_are_present():
    payloads = generate()
    assert any("/number/" in k and "k_on" in k for k in payloads)
    assert any("/button/" in k and "recalibrate" in k for k in payloads)
    assert any("/switch/" in k and "gestures_enabled" in k for k in payloads)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("nexmon:0:c0s0", "nexmon_0_c0s0"), ("esp:1", "esp_1"), ("A B", "a_b")],
)
def test_slug(value, expected):
    assert slug(value) == expected
