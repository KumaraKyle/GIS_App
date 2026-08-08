"""Link quality tracking.

Packet rate and loss are not cosmetic diagnostics here. CSI features are only
comparable when they were sampled at a consistent rate, so a stretch of data
recorded during Wi-Fi congestion is actively harmful as training material --
worse than having no data, because the model will happily learn from it.

``LinkQualityMonitor`` is therefore consumed in two places: it publishes
diagnostics to Home Assistant, and it marks stretches that the dataset builder
must exclude.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from csisense.events.types import LinkQualityEvent
from csisense.wire import CsiFrame


@dataclass(slots=True)
class _LinkState:
    last_seq: int | None = None
    received: int = 0
    # Gaps inferred from sequence numbers, which counts frames lost in transit
    # rather than merely late.
    lost: int = 0
    reordered: int = 0
    boot_id: int = 0
    ts_window: deque[int] = field(default_factory=lambda: deque(maxlen=256))
    rssi_window: deque[int] = field(default_factory=lambda: deque(maxlen=256))
    gain_locked: bool = False


class LinkQualityMonitor:
    """Tracks per-link rate, loss and RSSI from sequence numbers and timestamps."""

    def __init__(self, *, report_interval_ns: int = 1_000_000_000) -> None:
        self._links: dict[str, _LinkState] = {}
        self._report_interval_ns = report_interval_ns
        self._last_report_ns: dict[str, int] = {}

    def update(self, frame: CsiFrame) -> LinkQualityEvent | None:
        """Record a frame; return an event when the report interval has elapsed."""
        link_id = frame.link_id()
        st = self._links.get(link_id)
        if st is None:
            st = _LinkState(boot_id=frame.boot_id)
            self._links[link_id] = st
            self._last_report_ns[link_id] = frame.ts_host_ns

        # A sensor reboot restarts sequence numbering, so counting the jump as
        # loss would report a nonsense spike. Detect it and resynchronise.
        if frame.boot_id != st.boot_id:
            st.boot_id = frame.boot_id
            st.last_seq = None

        if st.last_seq is not None:
            delta = frame.seq - st.last_seq
            if delta <= 0:
                st.reordered += 1
            else:
                st.lost += delta - 1
        st.last_seq = frame.seq
        st.received += 1
        st.ts_window.append(frame.ts_host_ns)
        st.rssi_window.append(frame.rssi)
        st.gain_locked = frame.gain_locked

        if frame.ts_host_ns - self._last_report_ns[link_id] < self._report_interval_ns:
            return None
        self._last_report_ns[link_id] = frame.ts_host_ns
        return self.snapshot(link_id, frame.ts_host_ns)

    def snapshot(self, link_id: str, ts_ns: int) -> LinkQualityEvent:
        st = self._links[link_id]
        return LinkQualityEvent(
            ts_ns=ts_ns,
            link_id=link_id,
            rate_hz=self._rate_hz(st),
            loss_pct=self._loss_pct(st),
            rssi=sum(st.rssi_window) / len(st.rssi_window) if st.rssi_window else 0.0,
            gain_locked=st.gain_locked,
        )

    @property
    def link_ids(self) -> tuple[str, ...]:
        return tuple(self._links)

    @staticmethod
    def _rate_hz(st: _LinkState) -> float:
        if len(st.ts_window) < 2:
            return 0.0
        span_ns = st.ts_window[-1] - st.ts_window[0]
        if span_ns <= 0:
            return 0.0
        return (len(st.ts_window) - 1) * 1e9 / span_ns

    @staticmethod
    def _loss_pct(st: _LinkState) -> float:
        total = st.received + st.lost
        return 100.0 * st.lost / total if total else 0.0


__all__ = ["LinkQualityMonitor"]
