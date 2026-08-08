"""Wire format tests.

A decoder bug here does not crash -- it silently produces plausible-looking
garbage that flows into recordings and training data. So these tests lean hard
on round-tripping and on rejecting malformed input loudly.
"""

from __future__ import annotations

import contextlib
import struct

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from csisense.wire import (
    ESP_HEADER_LEN,
    ESP_MAGIC,
    ESP_WIRE_VERSION,
    FLAG_GAIN_LOCKED,
    NEXMON_HEADER_LEN,
    NEXMON_MAGIC,
    CsiFrame,
    SourceKind,
    WireFormatError,
    decode_esp_datagram,
    decode_nexmon_udp,
    encode_esp_datagram,
)

BCM43455C0_CHIP_ID = 0x6A


def make_esp_frame(n_sub: int = 64, seq: int = 0, beacon: int = 0) -> CsiFrame:
    rng = np.random.default_rng(seq)
    csi = (rng.integers(-100, 100, n_sub) + 1j * rng.integers(-100, 100, n_sub)).astype(
        np.complex64
    )
    return CsiFrame(
        source_kind=SourceKind.ESP,
        sensor_id=3,
        csi=csi,
        ts_host_ns=1_700_000_000_000_000_000,
        seq=seq,
        beacon_seq=beacon,
        rssi=-57,
        noise_floor=-92,
        ts_device_us=123456,
        channel=6,
    )


def make_nexmon_payload(
    n_sub: int = 256,
    *,
    magic: int = NEXMON_MAGIC,
    chip: int = BCM43455C0_CHIP_ID,
    seq: int = 7,
    core_stream: int = 0,
    values: np.ndarray | None = None,
) -> bytes:
    header = struct.pack(
        "<H6sHHHH", magic, b"\xaa\xbb\xcc\xdd\xee\xff", seq, core_stream, 0x1234, chip
    )
    if values is None:
        rng = np.random.default_rng(0)
        values = rng.integers(-2000, 2000, 2 * n_sub).astype("<i2")
    return header + values.astype("<i2").tobytes()


# --- ESP ------------------------------------------------------------------


def test_esp_roundtrip_single_record():
    frame = make_esp_frame(seq=11, beacon=99)
    payload = encode_esp_datagram([frame], sensor_id=3, flags=FLAG_GAIN_LOCKED, boot_id=0xABCD)

    info, frames = decode_esp_datagram(payload, ts_host_ns=42)

    assert info.sensor_id == 3
    assert info.n_records == 1
    assert info.boot_id == 0xABCD
    assert len(frames) == 1

    out = frames[0]
    assert out.source_kind is SourceKind.ESP
    assert out.seq == 11
    assert out.beacon_seq == 99
    assert out.rssi == -57
    assert out.noise_floor == -92
    assert out.n_sub == 64
    assert out.gain_locked
    # int8 quantisation is the only permitted loss.
    np.testing.assert_allclose(out.csi.real, frame.csi.real, atol=1)
    np.testing.assert_allclose(out.csi.imag, frame.csi.imag, atol=1)


def test_esp_roundtrip_batched():
    frames_in = [make_esp_frame(seq=i, beacon=1000 + i) for i in range(7)]
    payload = encode_esp_datagram(frames_in, sensor_id=1)

    info, frames_out = decode_esp_datagram(payload, ts_host_ns=0)

    assert info.n_records == 7
    assert [f.seq for f in frames_out] == list(range(7))
    assert [f.beacon_seq for f in frames_out] == [1000 + i for i in range(7)]


def test_esp_gain_reported_only_when_locked():
    """An unlocked AGC value is worse than no value: it invites false trust."""
    frame = make_esp_frame()
    frame.agc_gain = 42
    frame.fft_gain = 7

    _, locked = decode_esp_datagram(
        encode_esp_datagram([frame], sensor_id=0, flags=FLAG_GAIN_LOCKED), ts_host_ns=0
    )
    _, unlocked = decode_esp_datagram(
        encode_esp_datagram([frame], sensor_id=0, flags=0), ts_host_ns=0
    )

    assert locked[0].agc_gain == 42
    assert locked[0].fft_gain == 7
    assert unlocked[0].agc_gain is None
    assert unlocked[0].fft_gain is None


def test_esp_rejects_bad_magic():
    payload = bytearray(encode_esp_datagram([make_esp_frame()], sensor_id=0))
    payload[0:4] = struct.pack("<I", 0xDEADBEEF)
    with pytest.raises(WireFormatError, match="magic"):
        decode_esp_datagram(bytes(payload), ts_host_ns=0)


def test_esp_rejects_version_mismatch():
    payload = bytearray(encode_esp_datagram([make_esp_frame()], sensor_id=0))
    payload[4] = ESP_WIRE_VERSION + 1
    with pytest.raises(WireFormatError, match="wire version"):
        decode_esp_datagram(bytes(payload), ts_host_ns=0)


@pytest.mark.parametrize("cut", [1, 8, 20, 60])
def test_esp_rejects_truncation(cut):
    """Truncated datagrams must raise, never yield a partial batch."""
    payload = encode_esp_datagram([make_esp_frame() for _ in range(3)], sensor_id=0)
    with pytest.raises(WireFormatError):
        decode_esp_datagram(payload[:-cut], ts_host_ns=0)


def test_esp_rejects_inconsistent_payload_len():
    payload = bytearray(encode_esp_datagram([make_esp_frame()], sensor_id=0))
    # payload_len lives in the last 4 bytes of the 24-byte record header.
    struct.pack_into("<H", payload, ESP_HEADER_LEN + 20, 999)
    with pytest.raises(WireFormatError, match="payload_len"):
        decode_esp_datagram(bytes(payload), ts_host_ns=0)


def test_esp_empty_batch_is_valid():
    info, frames = decode_esp_datagram(encode_esp_datagram([], sensor_id=2), ts_host_ns=0)
    assert info.n_records == 0
    assert frames == []


def test_esp_ht40_subcarrier_count():
    """HT40 doubles frequency resolution; the format must carry it unchanged."""
    _, frames = decode_esp_datagram(
        encode_esp_datagram([make_esp_frame(n_sub=128)], sensor_id=0), ts_host_ns=0
    )
    assert frames[0].n_sub == 128


@given(
    n_records=st.integers(min_value=0, max_value=8),
    sensor_id=st.integers(min_value=0, max_value=15),
    boot_id=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_esp_roundtrip_property(n_records, sensor_id, boot_id):
    frames_in = [make_esp_frame(seq=i) for i in range(n_records)]
    payload = encode_esp_datagram(frames_in, sensor_id=sensor_id, boot_id=boot_id)
    info, frames_out = decode_esp_datagram(payload, ts_host_ns=0)
    assert info.sensor_id == sensor_id
    assert info.boot_id == boot_id
    assert len(frames_out) == n_records


@given(data=st.binary(min_size=0, max_size=200))
def test_esp_never_crashes_on_arbitrary_bytes(data):
    """Fuzzing must produce WireFormatError, never an unhandled exception."""
    with contextlib.suppress(WireFormatError):
        decode_esp_datagram(data, ts_host_ns=0)


# --- Nexmon ---------------------------------------------------------------


@pytest.mark.parametrize(("n_sub", "bandwidth"), [(64, 20), (128, 40), (256, 80)])
def test_nexmon_decodes_each_bandwidth(n_sub, bandwidth):
    frame = decode_nexmon_udp(make_nexmon_payload(n_sub), ts_host_ns=99, sensor_id=1)
    assert frame.n_sub == n_sub, f"{bandwidth}MHz should give {n_sub} subcarriers"
    assert frame.source_kind is SourceKind.NEXMON
    assert frame.seq == 7


def test_nexmon_iq_order_is_imag_first():
    """The 43455c0 emits (imag, real) int16 pairs. Getting this backwards
    conjugates the channel, which survives amplitude processing unnoticed and
    then quietly breaks anything phase-based."""
    values = np.array([10, 20, -30, 40] + [0] * (2 * 64 - 4), dtype="<i2")
    frame = decode_nexmon_udp(make_nexmon_payload(64, values=values), ts_host_ns=0)
    assert frame.csi[0].imag == 10
    assert frame.csi[0].real == 20
    assert frame.csi[1].imag == -30
    assert frame.csi[1].real == 40


def test_nexmon_core_and_stream_are_decoded():
    """RF core is part of link identity: two antennas see different multipath
    and must not share a baseline."""
    # core = 2 (low 3 bits), stream = 1 (next 3 bits)
    frame = decode_nexmon_udp(make_nexmon_payload(64, core_stream=0b001_010), ts_host_ns=0)
    assert frame.rf_core == 2
    assert frame.spatial_stream == 1
    assert frame.link_id() == "nexmon:0:c2s1"


def test_nexmon_rejects_bad_magic():
    with pytest.raises(WireFormatError, match="magic"):
        decode_nexmon_udp(make_nexmon_payload(64, magic=0x2222), ts_host_ns=0)


def test_nexmon_rejects_unsupported_chip():
    """bcm4366c0 uses a packed float format we do not implement; failing loudly
    beats decoding it as int16 and producing noise."""
    with pytest.raises(WireFormatError, match="chip"):
        decode_nexmon_udp(make_nexmon_payload(64, chip=0xAA), ts_host_ns=0)


def test_nexmon_rejects_short_payload():
    with pytest.raises(WireFormatError, match="too short"):
        decode_nexmon_udp(b"\x11\x11\x00", ts_host_ns=0)


def test_nexmon_rejects_misaligned_body():
    with pytest.raises(WireFormatError, match="multiple of 4"):
        decode_nexmon_udp(make_nexmon_payload(64) + b"\x00", ts_host_ns=0)


def test_nexmon_rejects_odd_subcarrier_count():
    values = np.zeros(2 * 100, dtype="<i2")
    with pytest.raises(WireFormatError, match="subcarrier count"):
        decode_nexmon_udp(make_nexmon_payload(100, values=values), ts_host_ns=0)


def test_nexmon_rejects_unexpected_width_when_configured():
    """Catches a bandwidth mismatch between nexutil config and the host."""
    with pytest.raises(WireFormatError, match="configured for"):
        decode_nexmon_udp(make_nexmon_payload(64), ts_host_ns=0, expect_n_sub=256)


@given(data=st.binary(min_size=0, max_size=300))
def test_nexmon_never_crashes_on_arbitrary_bytes(data):
    with contextlib.suppress(WireFormatError):
        decode_nexmon_udp(data, ts_host_ns=0)


# --- Header size contract with firmware -----------------------------------


def test_header_sizes_match_firmware_contract():
    """These constants are duplicated in firmware/common/csi_wire.h. If this
    test fails, the C struct and the Python decoder have drifted apart."""
    assert ESP_HEADER_LEN == 16
    assert NEXMON_HEADER_LEN == 16
    assert int.from_bytes(b"CSI1", "little") == ESP_MAGIC
