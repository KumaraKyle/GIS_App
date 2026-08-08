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
from csisense.events.types import Event
from csisense.ingest.quality import LinkQualityMonitor
from csisense.sources.base import CsiSource
from csisense.wire import CsiFrame

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
    ) -> None:
        self.config = config
        self._sinks = list(sinks)
        self._quality = LinkQualityMonitor()
        self.stats = PipelineStats()

    def add_sink(self, sink: EventSink) -> None:
        self._sinks.append(sink)

    def step(self, frame: CsiFrame) -> list[Event]:
        """Process one frame and return the events it produced.

        Pure with respect to I/O: the only thing it touches is pipeline state.
        Keep it that way when adding stages.
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

        # Later milestones insert DSP -> presence -> gesture stages here. Each
        # appends to `events`; nothing above this line needs to change.

        self.stats.events += len(events)
        return events

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
