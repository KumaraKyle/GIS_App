"""Sequence-wrap and windowed-loss regressions.

Both bugs these cover were silent: they produced plausible-looking numbers that
were simply wrong, and the wrong numbers gate which links presence fusion and
RTI are allowed to use.
"""

from __future__ import annotations

import numpy as np
import pytest

from csisense.ingest.quality import LOSS_WINDOW_S, LinkQualityMonitor
from csisense.wire import SEQ_SPACE, CsiFrame, SourceKind, seq_delta

NEXMON_SEQ_SPACE = 1 << 12


def frame(seq: int, *, ts_ns: int, kind: SourceKind = SourceKind.NEXMON, boot_id: int = 1):
    return CsiFrame(
        source_kind=kind,
        sensor_id=0,
        csi=np.ones(64, dtype=np.complex64),
        ts_host_ns=ts_ns,
        seq=seq,
        rssi=-50,
        boot_id=boot_id,
    )


# --- seq_delta ------------------------------------------------------------


@pytest.mark.parametrize(
    ("current", "previous", "expected"),
    [
        (5, 4, 1),  # ordinary step
        (4, 4, 0),  # duplicate
        (3, 4, -1),  # reordered
        (0, 4095, 1),  # wrap, adjacent
        (2, 4094, 4),  # wrap, with a gap
        (4094, 2, -4),  # reordered across the wrap
    ],
)
def test_seq_delta_is_wrap_aware(current, previous, expected):
    assert seq_delta(current, previous, NEXMON_SEQ_SPACE) == expected


def test_seq_space_is_defined_per_source():
    """The 802.11 sequence number is 12-bit; ESP's rx_seq is a uint32. Applying
    one source's width to the other mis-counts loss at every wrap."""
    assert SEQ_SPACE[SourceKind.NEXMON] == 1 << 12
    assert SEQ_SPACE[SourceKind.ESP] == 1 << 32


# --- the wrap bug itself --------------------------------------------------


def test_nexmon_wrap_is_not_counted_as_loss_or_reordering():
    """Regression: at 100Hz the 12-bit sequence wraps every ~41s. A plain
    subtraction booked every wrap as reordering and hid the real loss."""
    mon = LinkQualityMonitor()
    for i, seq in enumerate([4093, 4094, 4095, 0, 1, 2]):
        mon.update(frame(seq, ts_ns=i * 10_000_000))

    snap = mon.snapshot("nexmon:0", ts_ns=6 * 10_000_000)
    assert snap.loss_pct == 0.0
    assert snap.loss_pct_1m == 0.0


def test_loss_spanning_a_wrap_is_still_detected():
    """The wrap fix must not blind the counter to genuine loss at the boundary."""
    mon = LinkQualityMonitor()
    # 4094 -> 1 means 4095 and 0 were lost.
    for i, seq in enumerate([4093, 4094, 1, 2]):
        mon.update(frame(seq, ts_ns=i * 10_000_000))

    snap = mon.snapshot("nexmon:0", ts_ns=4 * 10_000_000)
    assert snap.loss_pct == pytest.approx(100 * 2 / 6)


def test_many_wraps_do_not_accumulate_phantom_loss():
    """Six full wraps is ~4 minutes of real capture."""
    mon = LinkQualityMonitor()
    for i in range(6 * NEXMON_SEQ_SPACE):
        mon.update(frame(i % NEXMON_SEQ_SPACE, ts_ns=i * 10_000_000))

    assert mon.snapshot("nexmon:0", ts_ns=0).loss_pct == 0.0


def test_esp_uses_the_wider_sequence_space():
    """A uint32 counter near 2^12 must not be treated as a nexmon wrap."""
    mon = LinkQualityMonitor()
    for i, seq in enumerate([4094, 4095, 4096, 4097]):
        mon.update(frame(seq, ts_ns=i * 10_000_000, kind=SourceKind.ESP))

    assert mon.snapshot("esp:0", ts_ns=0).loss_pct == 0.0


# --- windowed loss --------------------------------------------------------


def test_windowed_loss_recovers_after_a_bad_patch():
    """Regression: lifetime loss never recovers, so one bad ten minutes at 3am
    would mark an always-on link permanently unhealthy."""
    mon = LinkQualityMonitor()
    seq = 0
    ts = 0

    # 10 seconds where every other frame is lost.
    for _ in range(1000):
        mon.update(frame(seq % NEXMON_SEQ_SPACE, ts_ns=ts))
        seq += 2
        ts += 10_000_000

    during = mon.snapshot("nexmon:0", ts_ns=ts)
    assert during.loss_pct_1m > 40.0

    # Then two clean minutes, which should push the bad patch out of the window.
    for _ in range(2 * 60 * 100):
        mon.update(frame(seq % NEXMON_SEQ_SPACE, ts_ns=ts))
        seq += 1
        ts += 10_000_000

    after = mon.snapshot("nexmon:0", ts_ns=ts)
    assert after.loss_pct_1m == pytest.approx(0.0, abs=0.5)
    # Lifetime loss remains as history, which is why it must not gate health.
    assert after.loss_pct > 0.0


def test_health_uses_the_window_not_lifetime():
    mon = LinkQualityMonitor()
    seq = 0
    ts = 0
    for _ in range(500):
        mon.update(frame(seq % NEXMON_SEQ_SPACE, ts_ns=ts))
        seq += 3  # 2 of every 3 lost
        ts += 10_000_000
    assert not mon.is_healthy("nexmon:0", ts, nominal_rate_hz=100.0)

    for _ in range((LOSS_WINDOW_S + 5) * 100):
        mon.update(frame(seq % NEXMON_SEQ_SPACE, ts_ns=ts))
        seq += 1
        ts += 10_000_000
    assert mon.is_healthy("nexmon:0", ts, nominal_rate_hz=100.0)


def test_unknown_link_is_not_healthy():
    assert not LinkQualityMonitor().is_healthy("nope:0", 0, nominal_rate_hz=100.0)


def test_low_rate_link_is_unhealthy():
    """A link limping at 30Hz cannot support windowed DSP tuned for 100Hz."""
    mon = LinkQualityMonitor()
    for i in range(300):
        mon.update(frame(i % NEXMON_SEQ_SPACE, ts_ns=i * 33_000_000))  # ~30Hz
    assert not mon.is_healthy("nexmon:0", 300 * 33_000_000, nominal_rate_hz=100.0)


def test_tx_dropped_is_tracked_separately():
    """Frames the ESP dropped before transmitting are invisible to sequence-gap
    counting, so without this they go unaccounted for entirely."""
    mon = LinkQualityMonitor()
    for i in range(10):
        mon.update(frame(i, ts_ns=i * 10_000_000, kind=SourceKind.ESP), tx_dropped=7)
    assert mon.snapshot("esp:0", ts_ns=0).tx_dropped == 7
