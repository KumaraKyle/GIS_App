"""The tap between the pipeline and the dashboard.

**The pipeline never pushes to a WebSocket.** ``Pipeline.step`` stays pure and
synchronous; it calls ``TelemetryTap.record``, which writes into preallocated
numpy ring buffers and returns. No awaits, no allocation, no I/O.

A separate task reads the buffers at the display rate and broadcasts. Three
things fall out of that split:

* ``step`` remains a pure function of its input, so the golden replay tests keep
  working.
* Capture at 100Hz is decoupled from rendering at 20Hz, so a browser on a slow
  connection cannot backpressure the radio.
* Nothing is serialised for a dashboard nobody has open.

Downsampling happens here rather than in the browser because the Pi's uplink is
the scarce resource: raw CSI is ~102 kB/s per link, while the reduced,
incremental stream measures ~39 kB/s for two links at 20Hz. Columns are
aggregated with **max-abs, not mean** -- averaging five frames of a waterfall
smears out exactly the transients you are looking at it to see.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from csisense.dsp.chain import DspFeatures
from csisense.events.types import Event, PresenceState

# Subcarriers are binned to this many columns before transmission. 64 is enough
# to see banding and fades at any realistic display width.
WATERFALL_BINS = 64
# Rows of history kept per link, ~25s at 20Hz.
WATERFALL_ROWS = 512
TRACE_LEN = 1200


@dataclass(slots=True)
class LinkTelemetry:
    """Ring buffers for one link."""

    waterfall: np.ndarray  # (rows, WATERFALL_BINS) float32, newest last
    trace: np.ndarray  # (TRACE_LEN,) float32 motion z-score
    write_row: int = 0
    write_trace: int = 0
    n_rows: int = 0
    n_trace: int = 0
    # Accumulates frames between display ticks.
    pending: np.ndarray | None = None
    pending_count: int = 0
    last_z: float = 0.0
    rate_hz: float = 0.0
    loss_pct_1m: float = 0.0
    rssi: float = 0.0
    healthy: bool = True
    gain_locked: bool = False

    @classmethod
    def empty(cls) -> LinkTelemetry:
        return cls(
            waterfall=np.zeros((WATERFALL_ROWS, WATERFALL_BINS), dtype=np.float32),
            trace=np.zeros(TRACE_LEN, dtype=np.float32),
        )

    def recent_waterfall(self, n: int) -> np.ndarray:
        """Last ``n`` columns oldest-first."""
        n = min(n, self.n_rows)
        if n == 0:
            return np.zeros((0, WATERFALL_BINS), dtype=np.float32)
        idx = (self.write_row - n + np.arange(n)) % WATERFALL_ROWS
        return self.waterfall[idx]

    def recent_trace(self, n: int) -> np.ndarray:
        n = min(n, self.n_trace)
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        idx = (self.write_trace - n + np.arange(n)) % TRACE_LEN
        return self.trace[idx]


@dataclass(slots=True)
class TelemetrySnapshot:
    """What the broadcaster serialises each tick."""

    ts_ns: int
    presence: str
    motion_stat: float
    calibrated: bool
    calibration_progress: float
    blind: bool
    links: dict[str, dict[str, float | bool]] = field(default_factory=dict)
    waterfall: dict[str, np.ndarray] = field(default_factory=dict)
    traces: dict[str, np.ndarray] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)


class TelemetryTap:
    """Collects pipeline state for display without touching I/O."""

    def __init__(self, *, max_events: int = 200) -> None:
        self.links: dict[str, LinkTelemetry] = {}
        self.events: list[dict] = []
        self.max_events = max_events
        self.presence: PresenceState = PresenceState.IDLE
        self.motion_stat = 0.0
        self.calibrated = False
        self.calibration_progress = 0.0
        self.blind = True
        self.last_ts_ns = 0
        self.frames = 0
        self.event_seq = 0

    def _link(self, link_id: str) -> LinkTelemetry:
        tel = self.links.get(link_id)
        if tel is None:
            tel = LinkTelemetry.empty()
            self.links[link_id] = tel
        return tel

    def record_features(self, features: DspFeatures) -> None:
        """Accumulate one processed frame. O(1), no allocation after warmup."""
        self.frames += 1
        self.last_ts_ns = features.ts_ns
        tel = self._link(features.link_id)

        binned = _bin_to(features.amplitude, WATERFALL_BINS)
        if tel.pending is None:
            tel.pending = binned.copy()
            tel.pending_count = 1
        else:
            # Max-abs rather than mean: averaging smears out the transients.
            np.maximum(np.abs(tel.pending), np.abs(binned), out=tel.pending)
            tel.pending_count += 1

    def record_link_state(
        self,
        link_id: str,
        *,
        z: float | None = None,
        rate_hz: float | None = None,
        loss_pct_1m: float | None = None,
        rssi: float | None = None,
        healthy: bool | None = None,
        gain_locked: bool | None = None,
    ) -> None:
        tel = self._link(link_id)
        if z is not None:
            tel.last_z = z
        if rate_hz is not None:
            tel.rate_hz = rate_hz
        if loss_pct_1m is not None:
            tel.loss_pct_1m = loss_pct_1m
        if rssi is not None:
            tel.rssi = rssi
        if healthy is not None:
            tel.healthy = healthy
        if gain_locked is not None:
            tel.gain_locked = gain_locked

    def record_event(self, event: Event) -> None:
        record = _event_dict(event)
        # A monotonic sequence so the broadcaster can send only what a client
        # has not seen, rather than the whole tail every tick.
        record["seq"] = self.event_seq
        self.event_seq += 1
        self.events.append(record)
        if len(self.events) > self.max_events:
            del self.events[: len(self.events) - self.max_events]

    def events_since(self, seq: int) -> list[dict]:
        return [e for e in self.events if e["seq"] >= seq]

    def tick(self) -> dict[str, int]:
        """Commit accumulated frames into the ring buffers. Called at display rate.

        Returns how many waterfall columns each link gained, so the broadcaster
        can send only the new ones. Resending the whole history every tick would
        be ~85kB per frame at 20Hz -- 1.7MB/s, which a Pi's uplink cannot carry
        and which is entirely redundant since only one column is ever new.
        """
        added: dict[str, int] = {}
        for link_id, tel in self.links.items():
            n = 0
            if tel.pending is not None and tel.pending_count:
                tel.waterfall[tel.write_row] = tel.pending
                tel.write_row = (tel.write_row + 1) % WATERFALL_ROWS
                tel.n_rows = min(tel.n_rows + 1, WATERFALL_ROWS)
                tel.pending = None
                tel.pending_count = 0
                n = 1
            added[link_id] = n

            tel.trace[tel.write_trace] = tel.last_z
            tel.write_trace = (tel.write_trace + 1) % TRACE_LEN
            tel.n_trace = min(tel.n_trace + 1, TRACE_LEN)
        return added

    def snapshot(self, *, waterfall_rows: int = 128, trace_len: int = 300) -> TelemetrySnapshot:
        return TelemetrySnapshot(
            ts_ns=self.last_ts_ns,
            presence=self.presence.value,
            motion_stat=self.motion_stat,
            calibrated=self.calibrated,
            calibration_progress=self.calibration_progress,
            blind=self.blind,
            links={
                lid: {
                    "z": tel.last_z,
                    "rate_hz": tel.rate_hz,
                    "loss_pct_1m": tel.loss_pct_1m,
                    "rssi": tel.rssi,
                    "healthy": tel.healthy,
                    "gain_locked": tel.gain_locked,
                }
                for lid, tel in self.links.items()
            },
            waterfall={
                lid: tel.recent_waterfall(waterfall_rows) for lid, tel in self.links.items()
            },
            traces={lid: tel.recent_trace(trace_len) for lid, tel in self.links.items()},
            events=list(self.events[-40:]),
        )

    def drain_events(self) -> list[dict]:
        out = self.events
        self.events = []
        return out


def _bin_to(vector: np.ndarray, n_bins: int) -> np.ndarray:
    """Reduce a feature vector to a fixed number of bins by grouped mean."""
    v = np.asarray(vector, dtype=np.float32)
    if v.shape[0] == n_bins:
        return v.copy()
    if v.shape[0] < n_bins:
        # Interpolate up so the display width is constant regardless of radio.
        idx = np.linspace(0, v.shape[0] - 1, n_bins)
        return np.interp(idx, np.arange(v.shape[0]), v).astype(np.float32)
    edges = np.linspace(0, v.shape[0], n_bins + 1).astype(int)
    return np.array(
        [v[edges[i] : max(edges[i + 1], edges[i] + 1)].mean() for i in range(n_bins)],
        dtype=np.float32,
    )


def _event_dict(event: Event) -> dict:
    from csisense.events.types import GestureEvent, LinkQualityEvent, PresenceEvent

    if isinstance(event, PresenceEvent):
        return {
            "kind": "presence",
            "ts_ns": event.ts_ns,
            "state": event.state.value,
            "previous": event.previous.value,
            "stat": round(event.motion_stat, 3),
            "reason": event.reason,
            "per_link": {k: round(v, 3) for k, v in event.per_link.items()},
        }
    if isinstance(event, GestureEvent):
        return {
            "kind": "gesture",
            "ts_ns": event.ts_ns,
            "tier": event.tier,
            "gesture": event.gesture,
            "confidence": round(event.confidence, 4),
        }
    if isinstance(event, LinkQualityEvent):
        return {
            "kind": "link",
            "ts_ns": event.ts_ns,
            "link_id": event.link_id,
            "rate_hz": round(event.rate_hz, 1),
            "loss_pct_1m": round(event.loss_pct_1m, 2),
        }
    return {"kind": "unknown", "repr": repr(event)}


__all__ = [
    "TRACE_LEN",
    "WATERFALL_BINS",
    "WATERFALL_ROWS",
    "LinkTelemetry",
    "TelemetrySnapshot",
    "TelemetryTap",
]
