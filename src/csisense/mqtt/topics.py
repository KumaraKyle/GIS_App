"""Topic layout.

One place that knows the topic strings, so a rename cannot leave the publisher
and the discovery payloads disagreeing -- a failure that shows up as entities
that exist in Home Assistant and never update, which looks exactly like a sensor
that stopped working.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Topics:
    base_topic: str = "csisense"
    deployment_id: str = "home"

    @property
    def base(self) -> str:
        return f"{self.base_topic}/{self.deployment_id}"

    @property
    def availability(self) -> str:
        """Last-will topic. Everything else hangs off this being 'online'."""
        return f"{self.base}/availability"

    @property
    def status(self) -> str:
        return f"{self.base}/status"

    @property
    def presence_state(self) -> str:
        return f"{self.base}/presence/state"

    @property
    def presence_motion(self) -> str:
        return f"{self.base}/presence/motion"

    @property
    def presence_occupancy(self) -> str:
        return f"{self.base}/presence/occupancy"

    @property
    def presence_stat(self) -> str:
        return f"{self.base}/presence/stat"

    def link_quality(self, link_id: str) -> str:
        return f"{self.base}/link/{slug(link_id)}/quality"

    def gesture(self, tier: str) -> str:
        return f"{self.base}/gesture/{tier}"

    def gesture_event(self, tier: str) -> str:
        """Stateless trigger topic that device_automation entries subscribe to."""
        return f"{self.base}/gesture/{tier}/event"

    @property
    def rti_status(self) -> str:
        return f"{self.base}/rti/status"

    def config_set(self, name: str) -> str:
        return f"{self.base}/config/{name}/set"

    def config_state(self, name: str) -> str:
        return f"{self.base}/config/{name}/state"

    def command_set(self, name: str) -> str:
        return f"{self.base}/cmd/{name}/set"

    @property
    def config_wildcard(self) -> str:
        return f"{self.base}/config/+/set"

    @property
    def command_wildcard(self) -> str:
        return f"{self.base}/cmd/+/set"


def slug(value: str) -> str:
    """MQTT- and Home-Assistant-safe identifier.

    Link ids contain colons (``nexmon:0:c0s0``), which are legal in a topic but
    illegal in a Home Assistant ``unique_id``, so both go through this.
    """
    out = []
    for ch in value.lower():
        out.append(ch if ch.isalnum() else "_")
    return "".join(out).strip("_")


__all__ = ["Topics", "slug"]
