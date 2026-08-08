"""Recording must be lossless and the pipeline deterministic.

One test proves both: run the synthetic source straight through the pipeline,
then record it, replay it, and run *that* through an identical pipeline. If the
event lists differ, either the recorder lost something or the pipeline has
hidden non-determinism -- and both are fatal to everything downstream. Training
data is only worth collecting if replaying it reproduces what was captured, and
golden regression tests only mean something if the pipeline is reproducible.

The one thing deliberately *not* asserted is that CSI survives bit-for-bit as
floats: the wire format is int16 pairs, so recording rounds to integers. Frames
are compared after that rounding, which is what a real capture would have stored
anyway.
"""

from __future__ import annotations

import numpy as np
import pytest
from helpers import calibrate, make_config

from csisense.app.pipeline import Pipeline
from csisense.recorder.parquet import ParquetSessionReader, ParquetSessionWriter, RecorderError
from csisense.sources.replay import ReplayCsiSource
from csisense.sources.synthetic import SyntheticCsiSource
from csisense.wire import SourceKind


async def record(path, *, scenario="walk_through", seed=1, n_links=2, n_sub=64, duration_s=20.0):
    source = SyntheticCsiSource(
        seed=seed, scenario=scenario, n_links=n_links, n_sub=n_sub, duration_s=duration_s
    )
    writer = ParquetSessionWriter(
        path, n_sub=n_sub, source_kind=SourceKind.SYNTHETIC, rate_hz=100.0
    )
    frames = []
    async for frame in source.frames():
        frames.append(frame)
        writer.add(frame)
    writer.close()
    return frames


async def test_recording_roundtrip_preserves_frames(tmp_path):
    path = tmp_path / "session.parquet"
    original = await record(path)

    replayed = ParquetSessionReader(path).frames()

    assert len(replayed) == len(original)
    for a, b in zip(original, replayed, strict=True):
        assert b.ts_host_ns == a.ts_host_ns
        assert b.seq == a.seq
        assert b.beacon_seq == a.beacon_seq
        assert b.rssi == a.rssi
        assert b.sensor_id == a.sensor_id
        assert b.flags == a.flags
        assert b.link_id() == a.link_id()
        # int16 rounding is the only permitted loss.
        np.testing.assert_allclose(b.csi.real, np.rint(a.csi.real), atol=0)
        np.testing.assert_allclose(b.csi.imag, np.rint(a.csi.imag), atol=0)


async def test_replay_reproduces_live_events(tmp_path):
    """The load-bearing assertion: replaying a recording through the pipeline
    produces exactly the events that live capture produced."""
    path = tmp_path / "session.parquet"
    baselines = await calibrate(seed=1, n_links=2)

    # Live.
    source = SyntheticCsiSource(seed=1, scenario="walk_through", n_links=2, duration_s=20.0)
    live_pipeline = Pipeline(make_config(), source_info=source.info(), baselines=baselines)
    live_events = []
    writer = ParquetSessionWriter(path, n_sub=64, source_kind=SourceKind.SYNTHETIC, rate_hz=100.0)
    async for frame in source.frames():
        live_events.extend(live_pipeline.step(frame))
        writer.add(frame)
    writer.close()

    # Replayed.
    replay = ReplayCsiSource(path, speed=0.0)
    replay_pipeline = Pipeline(make_config(), source_info=replay.info(), baselines=baselines)
    replay_events = []
    async for frame in replay.frames():
        replay_events.extend(replay_pipeline.step(frame))

    assert [str(e) for e in replay_events] == [str(e) for e in live_events]


async def test_replay_is_indistinguishable_from_live():
    """Replayed frames must not be recognisable as replayed. If they were, the
    golden tests would be testing a simulation rather than the real pipeline."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "s.parquet"
        original = await record(path, n_links=1)
        replayed = ParquetSessionReader(path).frames()

        assert replayed[0].source_kind is original[0].source_kind
        assert replayed[0].boot_id == original[0].boot_id
        assert replayed[0].flags == original[0].flags


async def test_replay_twice_gives_identical_results(tmp_path):
    path = tmp_path / "s.parquet"
    await record(path, n_links=1)

    def run():
        reader = ParquetSessionReader(path)
        return [(f.ts_host_ns, f.seq, float(np.abs(f.csi).sum())) for f in reader.iter_frames()]

    assert run() == run()


async def test_session_metadata_survives(tmp_path):
    """Provenance is what makes a two-month-old recording still meaningful."""
    from csisense.recorder.parquet import SessionMeta

    path = tmp_path / "s.parquet"
    meta = SessionMeta(
        session_id="abc123",
        source_kind="synthetic",
        n_sub=64,
        rate_hz=100.0,
        started_ns=0,
        dsp_fingerprint="deadbeef",
        git_sha="cafe",
        notes="living room, curtains open",
    )
    source = SyntheticCsiSource(seed=1, n_links=1, duration_s=2.0)
    writer = ParquetSessionWriter(path, n_sub=64, source_kind=SourceKind.SYNTHETIC, meta=meta)
    async for frame in source.frames():
        writer.add(frame)
    writer.close()

    loaded = ParquetSessionReader(path).meta
    assert loaded.session_id == "abc123"
    assert loaded.dsp_fingerprint == "deadbeef"
    assert loaded.notes == "living room, curtains open"
    assert loaded.n_frames == 200


async def test_session_stops_at_the_duration_limit(tmp_path):
    """Sessions, not streams. 3GB per link per day would destroy an SD card."""
    path = tmp_path / "s.parquet"
    source = SyntheticCsiSource(seed=1, n_links=1, duration_s=20.0)
    writer = ParquetSessionWriter(
        path, n_sub=64, source_kind=SourceKind.SYNTHETIC, max_session_s=5.0
    )
    accepted = 0
    async for frame in source.frames():
        if not writer.add(frame):
            break
        accepted += 1
    writer.close()

    assert writer.full
    assert "max_session_s" in writer.stopped_reason
    assert 450 < accepted < 550  # ~5s at 100Hz


async def test_mixed_subcarrier_widths_are_rejected(tmp_path):
    """n_sub is fixed per file because the CSI column is fixed-width."""
    path = tmp_path / "s.parquet"
    writer = ParquetSessionWriter(path, n_sub=64, source_kind=SourceKind.SYNTHETIC)
    source = SyntheticCsiSource(seed=1, n_links=1, n_sub=128, duration_s=0.1)

    with pytest.raises(RecorderError, match="subcarriers"):
        async for frame in source.frames():
            writer.add(frame)


def test_recording_refuses_to_fill_the_disk(tmp_path):
    with pytest.raises(RecorderError, match="free"):
        ParquetSessionWriter(
            tmp_path / "s.parquet",
            n_sub=64,
            source_kind=SourceKind.SYNTHETIC,
            min_free_bytes=10**18,
        )
