"""Replaying recorded sessions.

The point of replay is that a recorded session must be **indistinguishable from
a live one** to everything downstream. Source kind, flags, boot ids and
timestamps are all restored exactly as captured, because the moment replay
differs from live capture the golden regression tests stop testing the real
pipeline and start testing a simulation of it.

``speed`` controls pacing:

* ``0.0`` -- as fast as the CPU allows. What tests use.
* ``1.0`` -- wall-clock real time. What demos and the dashboard use.
* ``2.0`` -- twice real time, for reviewing a long session quickly.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from pathlib import Path

from csisense.recorder.parquet import ParquetSessionReader
from csisense.sources.base import CsiSource, SourceInfo
from csisense.wire import CsiFrame, SourceKind


class ReplayCsiSource(CsiSource):
    """Streams a recorded Parquet session back through the pipeline."""

    def __init__(
        self,
        path: Path | str,
        *,
        speed: float = 1.0,
        loop: bool = False,
    ) -> None:
        if speed < 0.0:
            raise ValueError(f"speed must be >= 0, got {speed}")
        self.path = Path(path)
        self.speed = speed
        self.loop = loop
        self._reader = ParquetSessionReader(self.path)

    @property
    def meta(self):
        return self._reader.meta

    def info(self) -> SourceInfo:
        meta = self._reader.meta
        # Link ids are discovered from the data rather than assumed, so a
        # session recorded with three links replays as three links.
        link_ids: list[str] = []
        for frame in self._reader.iter_frames():
            lid = frame.link_id()
            if lid not in link_ids:
                link_ids.append(lid)
            if len(link_ids) >= 16:
                break
        return SourceInfo(
            kind=SourceKind(meta.source_kind),
            n_sub=meta.n_sub,
            rate_hz=meta.rate_hz,
            link_ids=tuple(link_ids),
        )

    async def frames(self) -> AsyncIterator[CsiFrame]:
        while True:
            wall_start = time.perf_counter()
            first_ts_ns: int | None = None

            for n, frame in enumerate(self._reader.iter_frames()):
                if self.speed > 0.0:
                    if first_ts_ns is None:
                        first_ts_ns = frame.ts_host_ns
                    target = wall_start + (frame.ts_host_ns - first_ts_ns) / 1e9 / self.speed
                    delay = target - time.perf_counter()
                    if delay > 0:
                        await asyncio.sleep(delay)
                elif n % 500 == 0:
                    # Yield periodically so a flat-out replay cannot starve the
                    # web server or MQTT client sharing this event loop.
                    await asyncio.sleep(0)
                yield frame

            if not self.loop:
                return


__all__ = ["ReplayCsiSource"]
