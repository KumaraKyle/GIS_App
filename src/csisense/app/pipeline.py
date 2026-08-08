"""The processing pipeline.

Structured so that the interesting part is a *pure function*:
:meth:`Pipeline.step` takes one frame and returns the events it caused. No
sockets, no clocks, no MQTT. :meth:`Pipeline.run` is the thin async shell that
pumps a source through it.

That split is what makes the golden regression tests possible -- a recorded
session replayed through ``step`` must produce byte-identical events, forever.

Stages are added in later milestones (DSP, presence, gestures); each one plugs
in here and must preserve the purity of ``step``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from csisense.config import AppConfig
from csisense.dsp.chain import DspChain
from csisense.events.types import Event
from csisense.ingest.quality import LinkQualityMonitor
from csisense.presence.baseline import LinkBaseline
from csisense.presence.detector import PresenceDetector
from csisense.sources.base import CsiSource, SourceInfo
from csisense.wire import CsiFrame, SourceKind

log = logging.getLogger(__name__)

EventSink = Callable[[Event], None]


@dataclass(slots=True)
class PipelineStats:
    """Counters for the run as a whole, for the CLI and the dashboard."""

    frames: int = 0
    events: int = 0
    first_ts_ns: int = 0
    last_ts_ns: int = 0
    per_link_frames: dict[str, int] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        if not self.first_ts_ns or self.last_ts_ns <= self.first_ts_ns:
            return 0.0
        return (self.last_ts_ns - self.first_ts_ns) / 1e9

    @property
    def mean_rate_hz(self) -> float:
        d = self.duration_s
        if d <= 0.0 or not self.per_link_frames:
            return 0.0
        return self.frames / d / len(self.per_link_frames)


class Pipeline:
    """Wires a source through the processing stages to a set of sinks."""

    def __init__(
        self,
        config: AppConfig,
        *,
        sinks: Sequence[EventSink] = (),
        source_info: SourceInfo | None = None,
        baselines: dict[str, LinkBaseline] | None = None,
        enable_presence: bool = True,
    ) -> None:
        self.config = config
        self._sinks = list(sinks)
        self._quality = LinkQualityMonitor()
        self.stats = PipelineStats()
        self._source_info = source_info
        self._nominal_rate_hz = source_info.rate_hz if source_info else 100.0

        # One DSP chain per link: filter state, PCA basis and subcarrier
        # ranking are all link-specific, because two antennas in the same room
        # see genuinely different multipath.
        self._chains: dict[str, DspChain] = {}
        self.presence: PresenceDetector | None = (
            PresenceDetector(config.presence, fs=self._nominal_rate_hz, baselines=baselines)
            if enable_presence
            else None
        )

    def add_sink(self, sink: EventSink) -> None:
        self._sinks.append(sink)

    def _chain_for(self, frame: CsiFrame) -> DspChain:
        link_id = frame.link_id()
        chain = self._chains.get(link_id)
        if chain is None:
            chain = DspChain(
                self.config.dsp,
                source_kind=frame.source_kind,
                n_sub=frame.n_sub,
                rate_hz=self._nominal_rate_hz,
                sig_mode=frame.sig_mode,
            )
            self._chains[link_id] = chain
        return chain

    def step(self, frame: CsiFrame) -> list[Event]:
        """Process one frame and return the events it produced.

        Pure with respect to I/O: the only thing it touches is pipeline state.
        Keep it that way when adding stages -- this purity is what makes the
        golden replay tests possible.
        """
        self.stats.frames += 1
        if not self.stats.first_ts_ns:
            self.stats.first_ts_ns = frame.ts_host_ns
        self.stats.last_ts_ns = frame.ts_host_ns

        link_id = frame.link_id()
        self.stats.per_link_frames[link_id] = self.stats.per_link_frames.get(link_id, 0) + 1

        events: list[Event] = []
        quality = self._quality.update(frame)
        if quality is not None:
            events.append(quality)

        features = self._chain_for(frame).process(frame)
        if features is not None and self.presence is not None:
            self.presence.set_link_health(
                link_id,
                self._quality.is_healthy(
                    link_id,
                    frame.ts_host_ns,
                    nominal_rate_hz=self._nominal_rate_hz,
                    max_loss_pct=self.config.presence.max_link_loss_pct,
                ),
            )
            presence_event = self.presence.update(features)
            if presence_event is not None:
                events.append(presence_event)

        self.stats.events += len(events)
        return events

    def start_calibration(self, duration_s: float | None = None) -> None:
        if self.presence is not None:
            self.presence.start_calibration(duration_s)

    @property
    def source_kind(self) -> SourceKind | None:
        return self._source_info.kind if self._source_info else None

    def _emit(self, events: Sequence[Event]) -> None:
        for event in events:
            for sink in self._sinks:
                try:
                    sink(event)
                except Exception:
                    # A failing sink (a dropped MQTT connection, a closed
                    # WebSocket) must never take down capture. Presence
                    # detection is the thing that has to keep running.
                    log.exception("event sink raised, continuing")

    async def run(self, source: CsiSource) -> PipelineStats:
        """Pump a source to exhaustion, emitting events to the sinks."""
        info = source.info()
        if self._source_info is None:
            self._source_info = info
            self._nominal_rate_hz = info.rate_hz
            if self.presence is not None:
                self.presence.fs = info.rate_hz
        log.info(
            "pipeline start: source=%s n_sub=%d rate=%.1fHz links=%d",
            info.kind.value,
            info.n_sub,
            info.rate_hz,
            info.n_links,
        )
        try:
            async for frame in source.frames():
                self._emit(self.step(frame))
        finally:
            await source.aclose()
            log.info(
                "pipeline stop: %d frames over %.1fs (%.1f Hz/link), %d events",
                self.stats.frames,
                self.stats.duration_s,
                self.stats.mean_rate_hz,
                self.stats.events,
            )
        return self.stats


__all__ = ["EventSink", "Pipeline", "PipelineStats"]
