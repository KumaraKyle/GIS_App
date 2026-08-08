"""UDP capture sources, exercised over loopback.

These cannot validate the wire layouts -- only real hardware can do that -- but
they do cover everything around the decode: filtering, overflow behaviour, error
tolerance, and the specific misconfiguration where nexutil is on a different
channel than the software believes.

The overflow test matters most. A capture path that backpressures instead of
dropping loses a whole second of a 100Hz stream every time the consumer
hiccups, and that failure only appears under load.
"""

from __future__ import annotations

import asyncio
import socket
import struct

import numpy as np
import pytest

from csisense.config import NexmonSourceConfig, UdpSourceConfig
from csisense.sources.udp import EspUdpCsiSource, NexmonCsiSource
from csisense.wire import (
    NEXMON_MAGIC,
    CsiFrame,
    SourceKind,
    encode_esp_datagram,
)

BCM43455C0 = 0x6A
# 5GHz ch157 @ 80MHz -- what the Pi is meant to be configured for.
CHANSPEC_157_80 = 0xE29D


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def nexmon_payload(
    *,
    n_sub: int = 256,
    mac: bytes = b"\xaa\xbb\xcc\xdd\xee\xff",
    seq: int = 1,
    core_stream: int = 0,
    chanspec: int = CHANSPEC_157_80,
    chip: int = BCM43455C0,
) -> bytes:
    header = struct.pack("<H6sHHHH", NEXMON_MAGIC, mac, seq, core_stream, chanspec, chip)
    return header + np.zeros(2 * n_sub, dtype="<i2").tobytes()


def esp_frame(n_sub: int = 64, seq: int = 0) -> CsiFrame:
    return CsiFrame(
        source_kind=SourceKind.ESP,
        sensor_id=2,
        csi=np.ones(n_sub, dtype=np.complex64),
        ts_host_ns=0,
        seq=seq,
        beacon_seq=seq,
        rssi=-50,
    )


async def send(port: int, payloads: list[bytes]) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        for p in payloads:
            s.sendto(p, ("127.0.0.1", port))
    await asyncio.sleep(0.15)


async def collect(source, payloads: list[bytes], *, expect: int, timeout: float = 3.0) -> list:
    """Start the source, send payloads, gather up to `expect` frames."""
    out: list = []

    async def pump() -> None:
        async for frame in source.frames():
            out.append(frame)
            if len(out) >= expect:
                return

    task = asyncio.create_task(pump())
    await asyncio.sleep(0.15)  # let the socket bind
    await send(source.port, payloads)
    try:
        await asyncio.wait_for(task, timeout=timeout)
    except TimeoutError:
        task.cancel()
    await source.aclose()
    return out


# --- nexmon ---------------------------------------------------------------


async def test_nexmon_receives_and_decodes():
    config = NexmonSourceConfig(port=free_port(), bandwidth_mhz=80)
    source = NexmonCsiSource(config)
    frames = await collect(source, [nexmon_payload() for _ in range(3)], expect=3)

    assert len(frames) == 3
    assert frames[0].n_sub == 256
    assert frames[0].source_kind is SourceKind.NEXMON


async def test_nexmon_filters_by_transmitter_mac():
    """Sensing works against one transmitter; everything else is interference."""
    config = NexmonSourceConfig(port=free_port(), tx_mac="aa:bb:cc:dd:ee:ff")
    source = NexmonCsiSource(config)
    payloads = [
        nexmon_payload(mac=b"\xaa\xbb\xcc\xdd\xee\xff"),
        nexmon_payload(mac=b"\x11\x22\x33\x44\x55\x66"),
        nexmon_payload(mac=b"\xaa\xbb\xcc\xdd\xee\xff"),
    ]
    frames = await collect(source, payloads, expect=2)

    assert len(frames) == 2
    assert source.filtered_mac == 1


async def test_nexmon_filters_by_rf_core():
    """Different antennas see different multipath, so cores are separate links
    rather than interchangeable samples."""
    config = NexmonSourceConfig(port=free_port(), cores=[0])
    source = NexmonCsiSource(config)
    # Rejected frame first: the collector stops at the first accepted one, so
    # anything after it would never be processed.
    payloads = [nexmon_payload(core_stream=1), nexmon_payload(core_stream=0)]
    frames = await collect(source, payloads, expect=1)

    assert len(frames) == 1
    assert frames[0].rf_core == 0
    assert source.filtered_core == 1


async def test_nexmon_rejects_a_channel_mismatch():
    """The nasty misconfiguration: nexutil on a different channel than config
    claims. Frames still flow, and every downstream number is wrong."""
    config = NexmonSourceConfig(port=free_port(), channel=157)
    source = NexmonCsiSource(config)
    # 2.4GHz ch6 @ 20MHz, not what was asked for.
    payloads = [nexmon_payload(chanspec=0x1006, n_sub=256) for _ in range(2)]
    frames = await collect(source, payloads, expect=1, timeout=1.0)

    assert frames == []
    assert source.filtered_channel == 2


async def test_nexmon_survives_garbage():
    """A malformed datagram must be counted and skipped, not kill capture."""
    config = NexmonSourceConfig(port=free_port())
    source = NexmonCsiSource(config)
    payloads = [b"\x00\x01\x02", nexmon_payload(), b"nonsense", nexmon_payload()]
    frames = await collect(source, payloads, expect=2)

    assert len(frames) == 2
    assert source.decode_errors == 2


async def test_nexmon_rejects_unexpected_width():
    """Catches a bandwidth mismatch between nexutil and config."""
    config = NexmonSourceConfig(port=free_port(), bandwidth_mhz=80)
    source = NexmonCsiSource(config)
    frames = await collect(source, [nexmon_payload(n_sub=64)], expect=1, timeout=1.0)

    assert frames == []
    assert source.decode_errors == 1


# --- esp ------------------------------------------------------------------


async def test_esp_receives_batched_records():
    config = UdpSourceConfig(port=free_port())
    source = EspUdpCsiSource(config)
    payload = encode_esp_datagram([esp_frame(seq=i) for i in range(7)], sensor_id=2)
    frames = await collect(source, [payload], expect=7)

    assert len(frames) == 7
    assert [f.seq for f in frames] == list(range(7))


async def test_esp_tracks_sender_side_drops():
    """Frames the ESP dropped before transmitting never got a sequence number,
    so gap counting cannot see them."""
    config = UdpSourceConfig(port=free_port())
    source = EspUdpCsiSource(config)
    payload = encode_esp_datagram([esp_frame()], sensor_id=2, tx_dropped=42)
    await collect(source, [payload], expect=1)

    assert source.tx_dropped[2] == 42


async def test_esp_survives_garbage():
    config = UdpSourceConfig(port=free_port())
    source = EspUdpCsiSource(config)
    good = encode_esp_datagram([esp_frame()], sensor_id=2)
    frames = await collect(source, [b"\xff" * 40, good], expect=1)

    assert len(frames) == 1
    assert source.decode_errors == 1


# --- overflow -------------------------------------------------------------


async def test_queue_overflow_drops_rather_than_blocking():
    """Backpressuring the socket would stall the receive path, losing a whole
    second of a 100Hz stream instead of one frame."""
    config = NexmonSourceConfig(port=free_port())
    source = NexmonCsiSource(config)
    source._queue = asyncio.Queue(maxsize=4)

    task = asyncio.create_task(anext(aiter(source.frames())))
    await asyncio.sleep(0.1)
    await send(config.port, [nexmon_payload(seq=i) for i in range(60)])

    assert source.dropped > 0, "queue never overflowed, so this proves nothing"
    task.cancel()
    await source.aclose()


@pytest.mark.parametrize("cls", [NexmonCsiSource, EspUdpCsiSource])
async def test_close_is_idempotent(cls):
    config = (
        NexmonSourceConfig(port=free_port())
        if cls is NexmonCsiSource
        else UdpSourceConfig(port=free_port())
    )
    source = cls(config)
    await source.aclose()
    await source.aclose()
