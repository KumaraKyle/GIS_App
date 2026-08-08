"""Link quality tracking.

Packet rate and loss are not cosmetic diagnostics here. CSI features are only
comparable when they were sampled at a consistent rate, so a stretch of data
recorded during Wi-Fi congestion is actively harmful as training material --
worse than having no data, because the model will happily learn from it.

``LinkQualityMonitor`` is therefore consumed in three places: it publishes
diagnostics to Home Assistant, it marks stretches the dataset builder must
exclude, and it decides which links are healthy enough to contribute to presence
fusion and RTI.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from csisense.events.types import LinkQualityEvent
from csisense.wire import SEQ_SPACE, CsiFrame, seq_delta

# Loss is reported over this trailing window as well as for the whole run. On an
# always-on Pi a lifetime figure never recovers: one bad ten minutes at 3am would
# pin the diagnostic at 4% forever and permanently mark the link unhealthy.
LOSS_WINDOW_S = 60


@dataclass(slots=True)
class _Bucket:
    """One second of counters, so the trailing window can expire whole seconds."""

    second: int
    received: int = 0
    lost: int = 0


@dataclass(slots=True)
class _LinkState:
    seq_space: int
    last_seq: int | None = None
    received: int = 0
    # Gaps inferred from sequence numbers, which counts frames lost in transit
    # rather than merely late.
    lost: int = 0
    reordered: int = 0
    # Frames the sender dropped before transmission. Invisible to sequence-gap
    # counting, so without this they would simply go unaccounted for.
    tx_dropped: int = 0
    boot_id: int = 0
    ts_window: deque[int] = field(default_factory=lambda: deque(maxlen=256))
    rssi_window: deque[int] = field(default_factory=lambda: deque(maxlen=256))
    buckets: deque[_Bucket] = field(default_factory=lambda: deque(maxlen=LOSS_WINDOW_S + 1))
    gain_locked: bool = False


class LinkQualityMonitor:
    """Tracks per-link rate, loss and RSSI from sequence numbers and timestamps."""

    def __init__(self, *, report_interval_ns: int = 1_000_000_000) -> None:
        self._links: dict[str, _LinkState] = {}
        self._report_interval_ns = report_interval_ns
        self._last_report_ns: dict[str, int] = {}

    def update(self, frame: CsiFrame, *, tx_dropped: int | None = None) -> LinkQualityEvent | None:
        """Record a frame; return an event when the report interval has elapsed."""
        link_id = frame.link_id()
        st = self._links.get(link_id)
        if st is None:
            st = _LinkState(
                seq_space=SEQ_SPACE.get(frame.source_kind, 1 << 32),
                boot_id=frame.boot_id,
            )
            self._links[link_id] = st
            self._last_report_ns[link_id] = frame.ts_host_ns

        # A sensor reboot restarts sequence numbering, so counting the jump as
        # loss would report a nonsense spike. Detect it and resynchronise.
        if frame.boot_id != st.boot_id:
            st.boot_id = frame.boot_id
            st.last_seq = None

        bucket = self._bucket(st, frame.ts_host_ns)

        lost_now = 0
        if st.last_seq is not None:
            # Wrap-aware: the 802.11 sequence number is only 12 bits, so a plain
            # subtraction books every wrap as a huge negative jump and hides the
            # real loss around it.
            delta = seq_delta(frame.seq, st.last_seq, st.seq_space)
            if delta <= 0:
                st.reordered += 1
            else:
                lost_now = delta - 1
                st.lost += lost_now
        st.last_seq = frame.seq
        st.received += 1
        bucket.received += 1
        bucket.lost += lost_now

        if tx_dropped is not None and tx_dropped > st.tx_dropped:
            st.tx_dropped = tx_dropped

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
            loss_pct=self._loss_pct(st.received, st.lost),
            loss_pct_1m=self._windowed_loss_pct(st, ts_ns),
            rssi=sum(st.rssi_window) / len(st.rssi_window) if st.rssi_window else 0.0,
            gain_locked=st.gain_locked,
            tx_dropped=st.tx_dropped,
        )

    def is_healthy(
        self,
        link_id: str,
        ts_ns: int,
        *,
        nominal_rate_hz: float,
        max_loss_pct: float = 20.0,
    ) -> bool:
        """Whether this link should contribute to presence fusion and RTI.

        Uses the trailing window rather than lifetime loss so a link recovers
        once conditions improve.
        """
        st = self._links.get(link_id)
        if st is None:
            return False
        return (
            self._windowed_loss_pct(st, ts_ns) < max_loss_pct
            and self._rate_hz(st) > 0.5 * nominal_rate_hz
        )

    @property
    def link_ids(self) -> tuple[str, ...]:
        return tuple(self._links)

    @staticmethod
    def _bucket(st: _LinkState, ts_ns: int) -> _Bucket:
        second = ts_ns // 1_000_000_000
        if not st.buckets or st.buckets[-1].second != second:
            st.buckets.append(_Bucket(second=second))
        return st.buckets[-1]

    @staticmethod
    def _rate_hz(st: _LinkState) -> float:
        if len(st.ts_window) < 2:
            return 0.0
        span_ns = st.ts_window[-1] - st.ts_window[0]
        if span_ns <= 0:
            return 0.0
        return (len(st.ts_window) - 1) * 1e9 / span_ns

    @staticmethod
    def _loss_pct(received: int, lost: int) -> float:
        total = received + lost
        return 100.0 * lost / total if total else 0.0

    @classmethod
    def _windowed_loss_pct(cls, st: _LinkState, ts_ns: int) -> float:
        cutoff = ts_ns // 1_000_000_000 - LOSS_WINDOW_S
        received = sum(b.received for b in st.buckets if b.second > cutoff)
        lost = sum(b.lost for b in st.buckets if b.second > cutoff)
        return cls._loss_pct(received, lost)


__all__ = ["LOSS_WINDOW_S", "LinkQualityMonitor"]
