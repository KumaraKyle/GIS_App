"""Home Assistant MQTT discovery payloads.

A **pure function**: config in, list of (topic, payload) out. No client, no
socket, no clock. That is deliberate -- discovery is the part that silently
breaks on a Home Assistant upgrade, and the only way to notice is to compare
generated payloads against a checked-in golden file. A function that needed a
broker to test would not get tested.

Two shape decisions worth stating:

**Gestures are device triggers, not sensors.** A gesture is a stateless event.
Modelled as a sensor it would have to hold a value between gestures, which means
either a permanent "last gesture" state that fires automations on restart, or a
sentinel value everyone has to remember to filter. A ``device_automation``
trigger is what Home Assistant provides for exactly this, and it shows up in the
automation editor as *"When CSI Sense -- gesture 'sit'"*.

**Presence is two binary sensors, not one.** ``motion`` is fast and transient;
``occupancy`` is sticky and survives someone sitting still. Collapsing them
loses the distinction that makes the system worth building.

Everything carries ``availability_topic``. When the detector goes blind -- no
healthy link -- the entities must become *unavailable* rather than reporting a
confident "clear", because a dead sensor reading as an empty house is the worst
failure this system has.
"""

from __future__ import annotations

from typing import Any

from csisense.config import AppConfig
from csisense.mqtt.topics import Topics, slug

# Bumped when the payload shape changes in a way that needs re-discovery.
DISCOVERY_VERSION = "1"


def device_block(config: AppConfig, sw_version: str = "") -> dict[str, Any]:
    """Shared device identity, so every entity groups under one device in HA."""
    return {
        "identifiers": [f"csisense_{config.deployment.id}"],
        "name": config.deployment.name,
        "manufacturer": "csisense",
        "model": "Wi-Fi CSI sensing",
        "sw_version": sw_version,
    }


def discovery_payloads(
    config: AppConfig,
    *,
    link_ids: tuple[str, ...] = (),
    gesture_classes: dict[str, tuple[str, ...]] | None = None,
    sw_version: str = "",
) -> list[tuple[str, dict[str, Any]]]:
    """Every discovery message, in a stable order.

    Ordering is stable so the golden test compares cleanly and so republishing
    after a reconnect produces identical bytes.
    """
    topics = Topics(config.mqtt.base_topic, config.deployment.id)
    prefix = config.mqtt.discovery_prefix
    dev = device_block(config, sw_version)
    uid = f"csisense_{config.deployment.id}"

    def availability() -> dict[str, Any]:
        return {
            "availability_topic": topics.availability,
            "payload_available": "online",
            "payload_not_available": "offline",
        }

    out: list[tuple[str, dict[str, Any]]] = []

    # -- presence ----------------------------------------------------------

    out.append(
        (
            f"{prefix}/binary_sensor/{uid}/motion/config",
            {
                "name": "Motion",
                "unique_id": f"{uid}_motion",
                "object_id": f"{config.deployment.id}_csi_motion",
                "device_class": "motion",
                "state_topic": topics.presence_motion,
                "payload_on": "ON",
                "payload_off": "OFF",
                "json_attributes_topic": topics.presence_state,
                "device": dev,
                **availability(),
            },
        )
    )
    out.append(
        (
            f"{prefix}/binary_sensor/{uid}/occupancy/config",
            {
                "name": "Occupancy",
                "unique_id": f"{uid}_occupancy",
                "object_id": f"{config.deployment.id}_csi_occupancy",
                "device_class": "occupancy",
                "state_topic": topics.presence_occupancy,
                "payload_on": "ON",
                "payload_off": "OFF",
                "json_attributes_topic": topics.presence_state,
                "device": dev,
                **availability(),
            },
        )
    )
    out.append(
        (
            f"{prefix}/sensor/{uid}/motion_stat/config",
            {
                "name": "Motion strength",
                "unique_id": f"{uid}_motion_stat",
                "state_topic": topics.presence_stat,
                "state_class": "measurement",
                "suggested_display_precision": 2,
                # Deviations above the calibrated quiet baseline. Unitless by
                # nature; naming it makes the history graph legible.
                "unit_of_measurement": "sigma",
                "icon": "mdi:motion-sensor",
                "device": dev,
                **availability(),
            },
        )
    )

    # -- per-link diagnostics ---------------------------------------------

    for link_id in link_ids:
        s = slug(link_id)
        for key, name, unit, icon in (
            ("rate_hz", "rate", "Hz", "mdi:speedometer"),
            ("loss_pct_1m", "loss", "%", "mdi:package-variant-remove"),
            ("rssi", "RSSI", "dBm", "mdi:wifi"),
        ):
            out.append(
                (
                    f"{prefix}/sensor/{uid}/link_{s}_{key}/config",
                    {
                        "name": f"{link_id} {name}",
                        "unique_id": f"{uid}_link_{s}_{key}",
                        "state_topic": topics.link_quality(link_id),
                        "value_template": f"{{{{ value_json.{key} }}}}",
                        "unit_of_measurement": unit,
                        "state_class": "measurement",
                        "entity_category": "diagnostic",
                        "icon": icon,
                        "device": dev,
                        **availability(),
                    },
                )
            )

    # -- gestures ----------------------------------------------------------

    for tier, classes in sorted((gesture_classes or {}).items()):
        for cls in classes:
            # "none" is the model's background class, not something to fire on.
            if cls == "none":
                continue
            out.append(
                (
                    f"{prefix}/device_automation/{uid}/gesture_{tier}_{slug(cls)}/config",
                    {
                        "automation_type": "trigger",
                        "type": "gesture",
                        "subtype": cls,
                        "topic": topics.gesture_event(tier),
                        "payload": cls,
                        "value_template": "{{ value_json.gesture }}",
                        "device": dev,
                    },
                )
            )
        # A sensor alongside the triggers, so gesture history is visible in HA
        # for debugging. Triggers leave no trace once fired.
        out.append(
            (
                f"{prefix}/sensor/{uid}/gesture_{tier}/config",
                {
                    "name": f"Last {tier} gesture",
                    "unique_id": f"{uid}_gesture_{tier}",
                    "state_topic": topics.gesture(tier),
                    "value_template": "{{ value_json.gesture }}",
                    "json_attributes_topic": topics.gesture(tier),
                    "icon": "mdi:gesture-tap",
                    "entity_category": "diagnostic",
                    "device": dev,
                    **availability(),
                },
            )
        )

    # -- controls ----------------------------------------------------------

    out.append(
        (
            f"{prefix}/number/{uid}/k_on/config",
            {
                "name": "Motion sensitivity",
                "unique_id": f"{uid}_k_on",
                "command_topic": topics.config_set("k_on"),
                "state_topic": topics.config_state("k_on"),
                "min": 2.0,
                "max": 15.0,
                "step": 0.1,
                "mode": "slider",
                "entity_category": "config",
                "icon": "mdi:tune",
                "device": dev,
                **availability(),
            },
        )
    )
    out.append(
        (
            f"{prefix}/button/{uid}/recalibrate/config",
            {
                "name": "Recalibrate baseline",
                "unique_id": f"{uid}_recalibrate",
                "command_topic": topics.command_set("recalibrate"),
                "payload_press": "PRESS",
                "entity_category": "config",
                "icon": "mdi:tune-vertical",
                "device": dev,
                **availability(),
            },
        )
    )
    out.append(
        (
            f"{prefix}/switch/{uid}/gestures_enabled/config",
            {
                "name": "Gesture detection",
                "unique_id": f"{uid}_gestures_enabled",
                "command_topic": topics.config_set("gestures_enabled"),
                "state_topic": topics.config_state("gestures_enabled"),
                "payload_on": "ON",
                "payload_off": "OFF",
                "entity_category": "config",
                "icon": "mdi:gesture",
                "device": dev,
                **availability(),
            },
        )
    )

    return out


__all__ = ["DISCOVERY_VERSION", "device_block", "discovery_payloads"]
