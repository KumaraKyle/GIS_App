"""Publishing pipeline events to MQTT.

Three behaviours that matter more than the plumbing:

**Republish discovery on every reconnect.** Home Assistant forgets
non-retained discovery when it restarts, and a broker restart drops retained
messages if persistence is off. Republishing on connect is the only way the
entities reliably come back, and it costs nothing.

**Never drop a transition.** The outbound queue drops only ``presence/stat``,
which is a 2Hz telemetry stream where a missing sample is invisible. Presence
transitions and gesture events are the actual product; losing one loses an
automation trigger.

**Publish unavailability when blind.** If no link is healthy the detector cannot
distinguish "empty room" from "sensor died". Reporting a confident "clear" in
that state would silently turn off the lights on someone sitting in the room, so
availability goes offline instead.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Protocol

from csisense.config import AppConfig
from csisense.events.types import (
    Event,
    GestureEvent,
    LinkQualityEvent,
    PresenceEvent,
    PresenceState,
)
from csisense.mqtt.discovery import discovery_payloads
from csisense.mqtt.topics import Topics

log = logging.getLogger(__name__)


class MqttTransport(Protocol):
    """What the publisher needs from a client.

    Narrow on purpose: the tests use a recording fake, and nothing here should
    depend on aiomqtt's specifics.
    """

    async def publish(
        self, topic: str, payload: bytes | str, *, qos: int = 0, retain: bool = False
    ) -> None: ...


@dataclass(slots=True)
class PublishRecord:
    topic: str
    payload: str
    retain: bool = False
    qos: int = 0


class MqttPublisher:
    """Translates pipeline events into MQTT messages."""

    def __init__(
        self,
        config: AppConfig,
        transport: MqttTransport,
        *,
        sw_version: str = "",
    ) -> None:
        self.config = config
        self.transport = transport
        self.topics = Topics(config.mqtt.base_topic, config.deployment.id)
        self.sw_version = sw_version
        self._last_stat_ns = 0
        self._stat_interval_ns = int(1e9 / max(config.mqtt.publish_motion_stat_hz, 0.01))
        self._available: bool | None = None

    # -- lifecycle ---------------------------------------------------------

    async def on_connect(
        self,
        *,
        link_ids: tuple[str, ...] = (),
        gesture_classes: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        """Announce the device and go online.

        Called on every (re)connection, not just the first.
        """
        for topic, payload in discovery_payloads(
            self.config,
            link_ids=link_ids,
            gesture_classes=gesture_classes,
            sw_version=self.sw_version,
        ):
            await self.transport.publish(topic, json.dumps(payload), retain=True, qos=1)
        await self.set_available(True)
        log.info("published discovery for %d link(s)", len(link_ids))

    async def set_available(self, available: bool) -> None:
        """Mark the device online or offline.

        Offline is used both for shutdown and for blindness -- from Home
        Assistant's point of view a sensor that cannot see is unavailable, which
        is exactly right and stops automations acting on a stale "clear".
        """
        if self._available == available:
            return
        self._available = available
        await self.transport.publish(
            self.topics.availability,
            "online" if available else "offline",
            retain=True,
            qos=1,
        )

    # -- events ------------------------------------------------------------

    async def publish_event(self, event: Event) -> None:
        if isinstance(event, PresenceEvent):
            await self._publish_presence(event)
        elif isinstance(event, GestureEvent):
            await self._publish_gesture(event)
        elif isinstance(event, LinkQualityEvent):
            await self._publish_link(event)

    async def _publish_presence(self, event: PresenceEvent) -> None:
        motion = "ON" if event.state is PresenceState.MOTION else "OFF"
        occupancy = "OFF" if event.state is PresenceState.IDLE else "ON"

        # Attributes first, so a template reading them alongside the state sees
        # a consistent pair rather than the previous attributes.
        await self.transport.publish(
            self.topics.presence_state,
            json.dumps(
                {
                    "state": event.state.value,
                    "previous": event.previous.value,
                    "motion": motion,
                    "occupancy": occupancy,
                    "stat": round(event.motion_stat, 3),
                    "reason": event.reason,
                    "per_link": {k: round(v, 3) for k, v in event.per_link.items()},
                    "ts_ns": event.ts_ns,
                }
            ),
            retain=True,
            qos=1,
        )
        # qos=1 and never dropped: this is the automation trigger.
        await self.transport.publish(self.topics.presence_motion, motion, retain=True, qos=1)
        await self.transport.publish(self.topics.presence_occupancy, occupancy, retain=True, qos=1)

    async def publish_motion_stat(self, ts_ns: int, stat: float) -> None:
        """Rate-limited telemetry. Safe to drop; only used for tuning."""
        if ts_ns - self._last_stat_ns < self._stat_interval_ns:
            return
        self._last_stat_ns = ts_ns
        await self.transport.publish(self.topics.presence_stat, f"{stat:.3f}", retain=True)

    async def _publish_gesture(self, event: GestureEvent) -> None:
        payload = json.dumps(
            {
                "gesture": event.gesture,
                "confidence": round(event.confidence, 4),
                "energy": round(event.energy, 4),
                "model_version": event.model_version,
                "ts_ns": event.ts_ns,
            }
        )
        # The sensor keeps history; the event topic fires device triggers and is
        # deliberately not retained, or every HA restart would re-fire it.
        await self.transport.publish(self.topics.gesture(event.tier), payload, retain=True)
        await self.transport.publish(
            self.topics.gesture_event(event.tier), payload, retain=False, qos=1
        )

    async def _publish_link(self, event: LinkQualityEvent) -> None:
        await self.transport.publish(
            self.topics.link_quality(event.link_id),
            json.dumps(
                {
                    "rate_hz": round(event.rate_hz, 2),
                    "loss_pct_1m": round(event.loss_pct_1m, 2),
                    "loss_pct_total": round(event.loss_pct, 2),
                    "rssi": round(event.rssi, 1),
                    "gain_locked": event.gain_locked,
                    "tx_dropped": event.tx_dropped,
                }
            ),
            retain=True,
        )

    async def publish_config_state(self, name: str, value: Any) -> None:
        await self.transport.publish(
            self.topics.config_state(name), _format_value(value), retain=True, qos=1
        )


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "ON" if value else "OFF"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


__all__ = ["MqttPublisher", "MqttTransport", "PublishRecord"]
