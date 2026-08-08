"""UDP capture from the radios.

Both real sources share the same shape: an asyncio datagram endpoint feeding a
bounded queue.

**Never block the socket.** If the queue fills, the oldest frame is dropped and
the loss is counted -- it is not backpressured. Blocking the event loop on a
slow consumer would stall the receive path, and on a 100Hz stream that loses an
entire second of capture rather than a single frame. A dropped CSI frame is
cheap; a stalled capture is not, which is also why the transport is UDP.

Neither class has ever seen a real packet. Everything here is written against
the documented formats and tested over loopback, and the parts most likely to be
wrong (the nexmon core/stream bit packing, the chanspec layout) are flagged in
``wire.py``. The first thing to do with real hardware is record raw datagrams
verbatim *before* decoding them, so a layout fix can be applied retroactively
rather than requiring a recapture.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator

from csisense.config import NexmonSourceConfig, UdpSourceConfig
from csisense.sources.base import CsiSource, SourceInfo
from csisense.wire import (
    FLAG_OVERFLOW_SINCE_LAST,
    CsiFrame,
    SourceKind,
    WireFormatError,
    decode_esp_datagram,
    decode_nexmon_udp,
)

log = logging.getLogger(__name__)

# ~20 seconds of one link at 100Hz. Large enough to ride out a GC pause or a
# slow consumer tick, small enough that a genuinely stuck consumer is noticed
# rather than silently accumulating gigabytes.
DEFAULT_QUEUE_SIZE = 2000


class _DatagramProtocol(asyncio.DatagramProtocol):
    """Pushes datagrams into a queue, dropping oldest on overflow."""

    def __init__(self, queue: asyncio.Queue[tuple[bytes, int]]) -> None:
        self._queue = queue
        self.dropped = 0
        self.received = 0

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        self.received += 1
        ts = time.time_ns()
        try:
            self._queue.put_nowait((data, ts))
        except asyncio.QueueFull:
            # Drop the oldest, keep the newest: stale CSI is worthless, and the
            # alternative is falling permanently behind real time.
            try:
                self._queue.get_nowait()
                self._queue.put_nowait((data, ts))
            except (asyncio.QueueEmpty, asyncio.QueueFull):  # pragma: no cover
                pass
            self.dropped += 1
            if self.dropped % 100 == 1:
                log.warning("capture queue full, dropped %d datagram(s)", self.dropped)

    def error_received(self, exc: Exception) -> None:  # pragma: no cover
        log.warning("UDP error: %s", exc)


class _UdpSourceBase(CsiSource):
    """Shared socket plumbing."""

    def __init__(self, bind: str, port: int, *, queue_size: int = DEFAULT_QUEUE_SIZE) -> None:
        self.bind = bind
        self.port = port
        self._queue: asyncio.Queue[tuple[bytes, int]] = asyncio.Queue(maxsize=queue_size)
        self._transport: asyncio.DatagramTransport | None = None
        self._protocol: _DatagramProtocol | None = None
        self._decode_errors = 0

    async def _open(self) -> None:
        loop = asyncio.get_running_loop()
        transport, protocol = await loop.create_datagram_endpoint(
            lambda: _DatagramProtocol(self._queue),
            local_addr=(self.bind, self.port),
            allow_broadcast=True,
        )
        self._transport = transport
        self._protocol = protocol
        log.info("%s listening on %s:%d", type(self).__name__, self.bind, self.port)

    async def aclose(self) -> None:
        if self._transport is not None:
            self._transport.close()
            self._transport = None

    @property
    def dropped(self) -> int:
        return self._protocol.dropped if self._protocol else 0

    @property
    def decode_errors(self) -> int:
        return self._decode_errors


class NexmonCsiSource(_UdpSourceBase):
    """CSI from nexmon_csi on a Raspberry Pi.

    nexmon broadcasts one UDP datagram per measurement to port 5500. Filtering
    happens here rather than in the firmware because ``makecsiparams -m`` takes a
    single MAC while a deployment may want several, and because rejecting frames
    whose chanspec disagrees with config catches the specific failure where
    ``nexutil`` was configured for a different channel than the software thinks:
    everything still flows, and every number downstream is quietly wrong.
    """

    def __init__(self, config: NexmonSourceConfig, *, sensor_id: int = 0) -> None:
        super().__init__(config.bind, config.port)
        self.config = config
        self.sensor_id = sensor_id
        self._tx_mac = config.tx_mac.lower().strip()
        self._cores = set(config.cores)
        self.filtered_mac = 0
        self.filtered_core = 0
        self.filtered_channel = 0

    def info(self) -> SourceInfo:
        return SourceInfo(
            kind=SourceKind.NEXMON,
            n_sub=self.config.n_sub,
            rate_hz=100.0,
            link_ids=tuple(f"nexmon:{self.sensor_id}:c{core}s0" for core in sorted(self._cores)),
        )

    async def frames(self) -> AsyncIterator[CsiFrame]:
        await self._open()
        try:
            while True:
                payload, ts_ns = await self._queue.get()
                try:
                    frame = decode_nexmon_udp(
                        payload,
                        ts_ns,
                        sensor_id=self.sensor_id,
                        expect_n_sub=self.config.n_sub,
                    )
                except WireFormatError as exc:
                    self._decode_errors += 1
                    if self._decode_errors % 100 == 1:
                        log.warning("nexmon decode error (%d so far): %s", self._decode_errors, exc)
                    continue

                if self._tx_mac and frame.tx_mac != self._tx_mac:
                    self.filtered_mac += 1
                    continue
                if self._cores and frame.rf_core not in self._cores:
                    self.filtered_core += 1
                    continue
                if self.config.channel and frame.channel != self.config.channel:
                    # nexutil is on a different channel than config claims.
                    # Everything would still flow, and be wrong.
                    self.filtered_channel += 1
                    if self.filtered_channel % 500 == 1:
                        log.error(
                            "nexmon frames on channel %d but config expects %d -- check "
                            "the makecsiparams invocation",
                            frame.channel,
                            self.config.channel,
                        )
                    continue

                yield frame
        finally:
            await self.aclose()


class EspUdpCsiSource(_UdpSourceBase):
    """CSI from the ESP32 receivers, batched several records per datagram.

    ``tx_dropped`` from the datagram header is surfaced separately from
    sequence-gap loss: frames the ESP discarded before transmitting never got a
    sequence number, so gap counting cannot see them and they would otherwise go
    entirely unaccounted for.
    """

    def __init__(self, config: UdpSourceConfig) -> None:
        super().__init__(config.bind, config.port)
        self.config = config
        self.tx_dropped: dict[int, int] = {}
        self._n_sub = 64
        self._link_ids: list[str] = []

    def info(self) -> SourceInfo:
        return SourceInfo(
            kind=SourceKind.ESP,
            n_sub=self._n_sub,
            rate_hz=100.0,
            link_ids=tuple(self._link_ids),
        )

    async def frames(self) -> AsyncIterator[CsiFrame]:
        await self._open()
        try:
            while True:
                payload, ts_ns = await self._queue.get()
                try:
                    info, frames = decode_esp_datagram(payload, ts_ns)
                except WireFormatError as exc:
                    self._decode_errors += 1
                    if self._decode_errors % 100 == 1:
                        log.warning("esp decode error (%d so far): %s", self._decode_errors, exc)
                    continue

                self.tx_dropped[info.sensor_id] = info.tx_dropped
                if info.flags & FLAG_OVERFLOW_SINCE_LAST:
                    log.debug("sensor %d reported a capture overflow", info.sensor_id)

                for frame in frames:
                    self._n_sub = frame.n_sub
                    link_id = frame.link_id()
                    if link_id not in self._link_ids:
                        self._link_ids.append(link_id)
                    yield frame
        finally:
            await self.aclose()


__all__ = ["DEFAULT_QUEUE_SIZE", "EspUdpCsiSource", "NexmonCsiSource"]
