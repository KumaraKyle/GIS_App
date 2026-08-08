"""CSI wire formats and the common frame abstraction.

Two very different sources produce CSI in this system, and this module is the
only place that knows the difference:

* **nexmon_csi** on the Raspberry Pi 3B+ (bcm43455c0) emits one UDP datagram per
  CSI measurement, broadcast to port 5500. 80MHz => 256 subcarriers.
* **esp-csi** on the ESP32 emits our own batched binary datagram (several
  measurements per packet to keep the packet rate down). HT20 => 64 subcarriers,
  HT40 => 128.

Everything downstream consumes :class:`CsiFrame` and never inspects the encoding
again. Keep it that way: the DSP, presence and ML stages must stay source-
agnostic so the same pipeline runs against either radio, replayed Parquet, or
the synthetic generator.

The ESP struct layout here is mirrored byte-for-byte in
``firmware/common/csi_wire.h``. If you change one, change both, and the golden
fixture test will tell you if they drift.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import numpy as np


class WireFormatError(ValueError):
    """A datagram could not be decoded.

    Raised rather than returning ``None`` because a malformed datagram means
    either a firmware/host version mismatch or memory corruption, and silently
    dropping frames would hide both.
    """


class SourceKind(StrEnum):
    """Which radio produced a frame.

    Carried on every frame because models are trained per source: a
    256-subcarrier Nexmon model is meaningless applied to 64-subcarrier ESP data,
    and this is what lets inference refuse the mismatch instead of producing
    confident nonsense.
    """

    NEXMON = "nexmon"
    ESP = "esp"
    SYNTHETIC = "synthetic"


# --- Flag bits, shared with firmware/common/csi_wire.h ----------------------

FLAG_GAIN_LOCKED: Final = 1 << 0
FLAG_CLOCK_SYNCED: Final = 1 << 1
FLAG_OVERFLOW_SINCE_LAST: Final = 1 << 2


@dataclass(slots=True)
class CsiFrame:
    """One CSI measurement from one receiver.

    ``csi`` is complex64 of shape ``(n_sub,)``. It is raw: no gain compensation,
    no subcarrier selection, no normalisation. All of that belongs to the DSP
    chain, so that a recording can be re-processed with different DSP settings
    without recapturing.
    """

    source_kind: SourceKind
    sensor_id: int
    csi: np.ndarray
    ts_host_ns: int
    seq: int = 0
    rssi: int = 0
    noise_floor: int = 0
    ts_device_us: int = 0
    channel: int = 0
    flags: int = 0
    boot_id: int = 0
    # Which link-layer mode produced this measurement. The set of physically
    # meaningful subcarriers differs between legacy LLTF and HT-LTF, so the DSP
    # subcarrier selector genuinely needs this.
    sig_mode: int = 0
    # Transmitter MAC, "aa:bb:cc:dd:ee:ff". Nexmon sources use it to implement
    # the configured tx_mac filter; empty when the source does not report one.
    tx_mac: str = ""
    # Present only for ESP frames: the transmitter's beacon counter, which lets
    # multiple receivers be aligned exactly without any clock arithmetic.
    beacon_seq: int | None = None
    # Which node transmitted the beacon this record answers. Meaningful once
    # more than one ESP32 transmits in a round-robin mesh; 0 for a single TX.
    tx_node_id: int = 0
    # Gain the radio actually used, where it reports it. None means "unknown",
    # which is the common case on the original ESP32 and is why gain locking in
    # firmware matters so much.
    agc_gain: int | None = None
    fft_gain: int | None = None
    # Nexmon only: which RF core and spatial stream this measurement came from.
    # Different cores are physically different antennas, so they are separate
    # observation channels and must not be averaged together by accident.
    rf_core: int | None = None
    spatial_stream: int | None = None

    @property
    def n_sub(self) -> int:
        return int(self.csi.shape[0])

    @property
    def gain_locked(self) -> bool:
        return bool(self.flags & FLAG_GAIN_LOCKED)

    def link_id(self) -> str:
        """Stable identifier for the observation channel this frame belongs to.

        Used as the dict key for per-link DSP state and per-link motion
        statistics, so it must not change while a sensor is running. The RF core
        is part of the identity: two antennas on the same Pi see different
        multipath and each needs its own baseline.
        """
        if self.rf_core is None:
            return f"{self.source_kind.value}:{self.sensor_id}"
        return f"{self.source_kind.value}:{self.sensor_id}:c{self.rf_core}s{self.spatial_stream}"


# ---------------------------------------------------------------------------
# nexmon_csi (Raspberry Pi, bcm43455c0)
# ---------------------------------------------------------------------------

NEXMON_MAGIC: Final = 0x1111
NEXMON_UDP_PORT: Final = 5500
# Little-endian: the Broadcom chip and the Pi are both LE. The magic itself is
# palindromic so it cannot tell you if you got this wrong -- the seq and chip
# fields can, which is why decoding validates the chip id.
_NEXMON_HEADER: Final = struct.Struct("<H6sHHHH")
NEXMON_HEADER_LEN: Final = _NEXMON_HEADER.size  # 16

# Chip ids that encode CSI as pairs of little-endian int16 (imag, real).
# The bcm4366c0 (Asus RT-AC86U) instead uses a packed custom float and is not
# supported here; it would need a separate unpack path.
_INT16_CHIPS: Final = frozenset({0x4C, 0x6A})

# Broadcom "d11" chanspec bit layout. NOT YET VALIDATED against a real capture --
# the constants come from the Broadcom headers, but which chanspec revision a
# given nexmon build emits is build-specific. decode_chanspec is therefore
# advisory: a mismatch warns, it does not discard data.
_CHANSPEC_CHAN_MASK: Final = 0x00FF
_CHANSPEC_BW_MASK: Final = 0x3800
_CHANSPEC_BAND_MASK: Final = 0xC000
_CHANSPEC_BW: Final = {0x1000: 20, 0x1800: 40, 0x2000: 80, 0x2800: 160}


def decode_chanspec(chanspec: int) -> tuple[int, int, int]:
    """Split a Broadcom chanspec into ``(channel, bandwidth_mhz, band_ghz)``.

    Returns ``bandwidth_mhz = 0`` when the width bits are unrecognised rather
    than guessing, so callers can tell "unknown" from "20MHz".
    """
    channel = chanspec & _CHANSPEC_CHAN_MASK
    bandwidth = _CHANSPEC_BW.get(chanspec & _CHANSPEC_BW_MASK, 0)
    band = 5 if (chanspec & _CHANSPEC_BAND_MASK) == _CHANSPEC_BAND_MASK else 2
    return channel, bandwidth, band


def format_mac(raw: bytes) -> str:
    return ":".join(f"{b:02x}" for b in raw)


def decode_nexmon_udp(
    payload: bytes,
    ts_host_ns: int,
    *,
    sensor_id: int = 0,
    expect_n_sub: int | None = None,
) -> CsiFrame:
    """Decode one nexmon_csi UDP payload.

    ``payload`` is the UDP body only (the socket has already stripped IP/UDP
    headers).

    Layout, per the nexmon_csi README:

    ==========  ======  ===================================================
    offset      bytes   meaning
    ==========  ======  ===================================================
    0           2       magic ``0x1111``
    2           6       source MAC
    8           2       Wi-Fi frame sequence number
    10          2       core (low 3 bits) and spatial stream (next 3 bits)
    12          2       chanspec
    14          2       chip version
    16          4*N     CSI, N = 64/128/256 for 20/40/80MHz
    ==========  ======  ===================================================
    """
    if len(payload) < NEXMON_HEADER_LEN:
        raise WireFormatError(
            f"nexmon payload too short: {len(payload)} bytes, need at least {NEXMON_HEADER_LEN}"
        )

    magic, mac, seq, core_stream, chanspec, chip = _NEXMON_HEADER.unpack_from(payload)
    if magic != NEXMON_MAGIC:
        raise WireFormatError(f"bad nexmon magic 0x{magic:04x}, expected 0x{NEXMON_MAGIC:04x}")

    body = payload[NEXMON_HEADER_LEN:]
    if len(body) % 4 != 0:
        raise WireFormatError(f"nexmon CSI body is {len(body)} bytes, not a multiple of 4")

    n_sub = len(body) // 4
    if n_sub not in (64, 128, 256):
        raise WireFormatError(f"unexpected nexmon subcarrier count {n_sub}, expected 64/128/256")
    if expect_n_sub is not None and n_sub != expect_n_sub:
        raise WireFormatError(
            f"nexmon frame has {n_sub} subcarriers, configured for {expect_n_sub}"
        )

    if chip not in _INT16_CHIPS:
        raise WireFormatError(
            f"unsupported nexmon chip id 0x{chip:x}; only int16-format chips "
            "(bcm43455c0) are implemented"
        )

    # int16 pairs, imaginary part first. Cast through int16 so the sign is
    # interpreted correctly before widening to float.
    raw = np.frombuffer(body, dtype="<i2").astype(np.float32)
    csi = np.empty(n_sub, dtype=np.complex64)
    csi.real = raw[1::2]
    csi.imag = raw[0::2]

    channel, _bw, _band = decode_chanspec(chanspec)

    return CsiFrame(
        source_kind=SourceKind.NEXMON,
        sensor_id=sensor_id,
        csi=csi,
        ts_host_ns=ts_host_ns,
        seq=seq,
        channel=channel,
        tx_mac=format_mac(mac),
        # The Pi timestamps on arrival; the chip does not give us a usable one.
        ts_device_us=ts_host_ns // 1000,
        flags=FLAG_CLOCK_SYNCED,
        # Nexmon reports gain per-frame in some builds but not reliably on the
        # 43455c0, so leave it unknown and let host-side compensation decide.
        agc_gain=None,
        fft_gain=None,
        rf_core=core_stream & 0x7,
        spatial_stream=(core_stream >> 3) & 0x7,
    )


# ---------------------------------------------------------------------------
# esp-csi (ESP32), our own batched format
# ---------------------------------------------------------------------------

ESP_MAGIC: Final = 0x31495343  # b"CSI1" little-endian
# v2 spends one of v1's two padding bytes on tx_node_id. Done before any
# training data exists, because changing the record layout afterwards would
# invalidate every recorded session.
ESP_WIRE_VERSION: Final = 2
ESP_UDP_PORT: Final = 5566

_ESP_HEADER: Final = struct.Struct("<IBBBBII")
ESP_HEADER_LEN: Final = _ESP_HEADER.size  # 16

# Trailing 1x is explicit padding so the C struct's natural alignment and this
# format agree. Do not remove it without changing csi_wire.h.
_ESP_RECORD: Final = struct.Struct("<IIIbbBBBBBBHB1x")
ESP_RECORD_HEADER_LEN: Final = _ESP_RECORD.size  # 24

# Sequence counters wrap, and the width differs per source. Treating them as
# unbounded silently mis-counts loss at every wrap.
SEQ_SPACE: Final = {
    # 802.11 sequence number: 12 bits. At 100Hz this wraps every ~41 seconds.
    SourceKind.NEXMON: 1 << 12,
    SourceKind.ESP: 1 << 32,
    SourceKind.SYNTHETIC: 1 << 32,
}


def seq_delta(current: int, previous: int, seq_space: int) -> int:
    """Signed distance between two sequence numbers, wrap-aware.

    Returns a small negative number for genuine reordering and a small positive
    number across a wrap boundary, instead of the enormous jump a plain
    subtraction produces.
    """
    half = seq_space // 2
    return ((current - previous + half) % seq_space) - half


@dataclass(slots=True)
class EspDatagramInfo:
    """Per-datagram counters that describe link health, not signal.

    Kept separate from :class:`CsiFrame` because they describe the transport
    rather than a measurement; ``LinkQualityMonitor`` consumes these to decide
    which stretches of a recording are too lossy to train on.
    """

    sensor_id: int
    n_records: int
    flags: int
    tx_dropped: int
    boot_id: int


def decode_esp_datagram(
    payload: bytes,
    ts_host_ns: int,
) -> tuple[EspDatagramInfo, list[CsiFrame]]:
    """Decode one batched ESP32 datagram into its constituent frames.

    Raises :class:`WireFormatError` on any inconsistency rather than returning a
    partial batch — a truncated datagram means the sender and receiver disagree
    about the format, and half-decoding it would poison a recording silently.
    """
    if len(payload) < ESP_HEADER_LEN:
        raise WireFormatError(
            f"esp datagram too short: {len(payload)} bytes, need at least {ESP_HEADER_LEN}"
        )

    magic, version, sensor_id, n_records, flags, tx_dropped, boot_id = _ESP_HEADER.unpack_from(
        payload
    )
    if magic != ESP_MAGIC:
        raise WireFormatError(f"bad esp magic 0x{magic:08x}, expected 0x{ESP_MAGIC:08x}")
    if version != ESP_WIRE_VERSION:
        raise WireFormatError(
            f"esp wire version {version} != {ESP_WIRE_VERSION}; reflash the firmware "
            "or check out a matching host revision"
        )

    info = EspDatagramInfo(
        sensor_id=sensor_id,
        n_records=n_records,
        flags=flags,
        tx_dropped=tx_dropped,
        boot_id=boot_id,
    )

    frames: list[CsiFrame] = []
    offset = ESP_HEADER_LEN

    for i in range(n_records):
        if offset + ESP_RECORD_HEADER_LEN > len(payload):
            raise WireFormatError(
                f"esp datagram truncated: record {i} of {n_records} header runs past "
                f"end ({len(payload)} bytes)"
            )
        (
            rx_seq,
            beacon_seq,
            esp_ts_us,
            rssi,
            noise_floor,
            agc_gain,
            fft_gain,
            channel,
            _rate,
            sig_mode,
            n_sub,
            payload_len,
            tx_node_id,
        ) = _ESP_RECORD.unpack_from(payload, offset)
        offset += ESP_RECORD_HEADER_LEN

        if payload_len != 2 * n_sub:
            raise WireFormatError(
                f"esp record {i}: payload_len {payload_len} != 2 * n_sub ({2 * n_sub})"
            )
        if offset + payload_len > len(payload):
            raise WireFormatError(
                f"esp datagram truncated: record {i} needs {payload_len} CSI bytes, "
                f"only {len(payload) - offset} remain"
            )

        # ESP-IDF delivers interleaved int8 as (imag, real).
        raw = np.frombuffer(payload, dtype=np.int8, count=payload_len, offset=offset).astype(
            np.float32
        )
        offset += payload_len

        csi = np.empty(n_sub, dtype=np.complex64)
        csi.real = raw[1::2]
        csi.imag = raw[0::2]

        frames.append(
            CsiFrame(
                source_kind=SourceKind.ESP,
                sensor_id=sensor_id,
                csi=csi,
                ts_host_ns=ts_host_ns,
                seq=rx_seq,
                beacon_seq=beacon_seq,
                rssi=rssi,
                noise_floor=noise_floor,
                ts_device_us=esp_ts_us,
                channel=channel,
                sig_mode=sig_mode,
                tx_node_id=tx_node_id,
                flags=flags,
                boot_id=boot_id,
                # Only meaningful once firmware has locked the gain; the flag
                # says whether to believe them.
                agc_gain=agc_gain if flags & FLAG_GAIN_LOCKED else None,
                fft_gain=fft_gain if flags & FLAG_GAIN_LOCKED else None,
            )
        )

    return info, frames


def encode_esp_datagram(
    frames: list[CsiFrame],
    *,
    sensor_id: int,
    flags: int = 0,
    tx_dropped: int = 0,
    boot_id: int = 0,
) -> bytes:
    """Encode frames in the ESP wire format.

    The firmware is the only real producer of this format; this exists so tests
    can round-trip without hardware, and so ``scripts/make_fixtures.py`` can
    build golden capture files.
    """
    if len(frames) > 255:
        raise WireFormatError(f"cannot encode {len(frames)} records in one datagram (max 255)")

    out = bytearray(
        _ESP_HEADER.pack(
            ESP_MAGIC, ESP_WIRE_VERSION, sensor_id, len(frames), flags, tx_dropped, boot_id
        )
    )

    for f in frames:
        n_sub = f.n_sub
        interleaved = np.empty(2 * n_sub, dtype=np.int8)
        interleaved[0::2] = np.clip(np.round(f.csi.imag), -128, 127).astype(np.int8)
        interleaved[1::2] = np.clip(np.round(f.csi.real), -128, 127).astype(np.int8)

        out += _ESP_RECORD.pack(
            f.seq,
            f.beacon_seq if f.beacon_seq is not None else 0,
            f.ts_device_us & 0xFFFFFFFF,
            f.rssi,
            f.noise_floor,
            f.agc_gain or 0,
            f.fft_gain or 0,
            f.channel,
            0,  # rate
            f.sig_mode,
            n_sub,
            2 * n_sub,
            f.tx_node_id,
        )
        out += interleaved.tobytes()

    return bytes(out)


__all__ = [
    "ESP_HEADER_LEN",
    "ESP_MAGIC",
    "ESP_RECORD_HEADER_LEN",
    "ESP_UDP_PORT",
    "ESP_WIRE_VERSION",
    "FLAG_CLOCK_SYNCED",
    "FLAG_GAIN_LOCKED",
    "FLAG_OVERFLOW_SINCE_LAST",
    "NEXMON_HEADER_LEN",
    "NEXMON_MAGIC",
    "NEXMON_UDP_PORT",
    "SEQ_SPACE",
    "CsiFrame",
    "EspDatagramInfo",
    "SourceKind",
    "WireFormatError",
    "decode_chanspec",
    "decode_esp_datagram",
    "decode_nexmon_udp",
    "encode_esp_datagram",
    "format_mac",
    "seq_delta",
]
