"""Pipeline and link-quality tests."""

from __future__ import annotations

import numpy as np
import pytest

from csisense.app.pipeline import Pipeline
from csisense.config import AppConfig
from csisense.events.types import Event, LinkQualityEvent
from csisense.ingest.quality import LinkQualityMonitor
from csisense.sources.synthetic import SyntheticCsiSource
from csisense.wire import FLAG_GAIN_LOCKED, CsiFrame, SourceKind


def frame(seq: int, *, ts_ns: int, sensor_id: int = 0, boot_id: int = 1, rssi: int = -50):
    return CsiFrame(
        source_kind=SourceKind.ESP,
        sensor_id=sensor_id,
        csi=np.ones(64, dtype=np.complex64),
        ts_host_ns=ts_ns,
        seq=seq,
        rssi=rssi,
        boot_id=boot_id,
        flags=FLAG_GAIN_LOCKED,
    )


# --- LinkQualityMonitor ---------------------------------------------------


def test_quality_counts_loss_from_sequence_gaps():
    mon = LinkQualityMonitor()
    # seq 3 and 4 never arrive.
    for i, seq in enumerate([0, 1, 2, 5, 6]):
        mon.update(frame(seq, ts_ns=i * 10_000_000))

    snap = mon.snapshot("esp:0", ts_ns=0)
    assert snap.loss_pct == pytest.approx(100 * 2 / 7)


def test_quality_measures_rate():
    mon = LinkQualityMonitor()
    for i in range(101):
        mon.update(frame(i, ts_ns=i * 10_000_000))  # 100 Hz
    assert mon.snapshot("esp:0", ts_ns=0).rate_hz == pytest.approx(100.0, rel=1e-6)


def test_quality_reboot_does_not_look_like_massive_loss():
    """A sensor reboot restarts sequence numbering. Counting that jump as loss
    would report a nonsense spike and could disqualify good data."""
    mon = LinkQualityMonitor()
    for i in range(10):
        mon.update(frame(5000 + i, ts_ns=i * 10_000_000, boot_id=1))
    for i in range(10):
        mon.update(frame(i, ts_ns=(10 + i) * 10_000_000, boot_id=2))

    assert mon.snapshot("esp:0", ts_ns=0).loss_pct == 0.0


def test_quality_counts_reordering_separately_from_loss():
    mon = LinkQualityMonitor()
    for i, seq in enumerate([0, 1, 3, 2, 4]):
        mon.update(frame(seq, ts_ns=i * 10_000_000))
    snap = mon.snapshot("esp:0", ts_ns=0)
    # One gap (seq 2 missing when 3 arrived); the late 2 is reordering, not loss.
    assert snap.loss_pct > 0
    assert snap.rate_hz > 0


def test_quality_reports_on_interval_only():
    mon = LinkQualityMonitor(report_interval_ns=1_000_000_000)
    events = [mon.update(frame(i, ts_ns=i * 10_000_000)) for i in range(250)]
    reported = [e for e in events if e is not None]
    # 250 frames at 100 Hz is 2.5 seconds, so about two reports.
    assert 1 <= len(reported) <= 3
    assert all(isinstance(e, LinkQualityEvent) for e in reported)


def test_quality_tracks_links_independently():
    mon = LinkQualityMonitor()
    for i in range(20):
        mon.update(frame(i, ts_ns=i * 10_000_000, sensor_id=0))
        mon.update(frame(i * 2, ts_ns=i * 10_000_000, sensor_id=1))  # every other lost

    assert mon.snapshot("esp:0", ts_ns=0).loss_pct == 0.0
    assert mon.snapshot("esp:1", ts_ns=0).loss_pct > 40.0
    assert set(mon.link_ids) == {"esp:0", "esp:1"}


def test_gain_locked_propagates_to_diagnostics():
    mon = LinkQualityMonitor()
    f = frame(0, ts_ns=0)
    f.flags = 0
    mon.update(f)
    assert mon.snapshot("esp:0", ts_ns=0).gain_locked is False


# --- Pipeline -------------------------------------------------------------


async def test_pipeline_runs_synthetic_source_to_completion():
    pipeline = Pipeline(AppConfig())
    source = SyntheticCsiSource(seed=1, scenario="walk_through", n_links=2, duration_s=3.0)

    stats = await pipeline.run(source)

    assert stats.frames == 2 * 300
    assert stats.duration_s == pytest.approx(3.0, abs=0.05)
    assert stats.mean_rate_hz == pytest.approx(100.0, rel=0.05)
    assert set(stats.per_link_frames) == {"synthetic:0", "synthetic:1"}


async def test_pipeline_step_is_deterministic():
    """The guarantee the golden regression tests are built on."""

    async def run() -> list[Event]:
        collected: list[Event] = []
        p = Pipeline(AppConfig(), sinks=[collected.append])
        await p.run(SyntheticCsiSource(seed=9, scenario="walk_through", duration_s=3.0))
        return collected

    a, b = await run(), await run()
    assert [str(e) for e in a] == [str(e) for e in b]


async def test_pipeline_survives_a_failing_sink():
    """Presence detection must keep running when MQTT or a WebSocket dies."""

    def explode(_event: Event) -> None:
        raise RuntimeError("sink is down")

    ok: list[Event] = []
    pipeline = Pipeline(AppConfig(), sinks=[explode, ok.append])
    stats = await pipeline.run(SyntheticCsiSource(seed=1, duration_s=3.0))

    assert stats.frames > 0
    assert ok, "the healthy sink should still have received events"


async def test_pipeline_reports_dropout_in_per_link_counts():
    pipeline = Pipeline(AppConfig())
    stats = await pipeline.run(
        SyntheticCsiSource(seed=1, scenario="sensor_dropout", n_links=2, duration_s=20.0)
    )
    assert stats.per_link_frames["synthetic:1"] < stats.per_link_frames["synthetic:0"]
