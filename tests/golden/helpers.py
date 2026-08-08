"""Shared helpers for the presence acceptance tests.

Calibration and detection are deliberately separate runs against separate
sources. An earlier version of these tests calibrated over the first 30 seconds
of the same stream that contained the motion -- so the detector learned the
*moving* room as its baseline and then, correctly, never fired again. Every
"zero false positives" assertion passed, while the detector was simply dead.

Calibration must see an empty room. That is true of the real system too, and it
is why the dashboard tells you to leave before pressing the button.
"""

from __future__ import annotations

import numpy as np

from csisense.app.pipeline import Pipeline
from csisense.config import AppConfig
from csisense.events.types import PresenceEvent
from csisense.presence.baseline import LinkBaseline
from csisense.sources.synthetic import SyntheticCsiSource
from csisense.wire import SourceKind

# The scenarios put motion at t=6-14s, so the subcarrier ranking must commit
# well before that. Two seconds is plenty for ranking; the shipped default of
# 1000 frames is sized for a real deployment that runs for months.
TEST_WARMUP_FRAMES = 200


def make_config(**overrides: object) -> AppConfig:
    config = AppConfig()
    config.dsp.subcarriers.warmup_frames = TEST_WARMUP_FRAMES
    for key, value in overrides.items():
        section, _, field = key.partition("__")
        setattr(getattr(config, section), field, value)
    return config


async def calibrate(
    *,
    seed: int = 1,
    n_links: int = 2,
    n_sub: int = 64,
    snr_db: float = 25.0,
    carrier_hz: float | None = None,
    source_kind: SourceKind = SourceKind.SYNTHETIC,
    config: AppConfig | None = None,
) -> dict[str, LinkBaseline]:
    """Measure the quiet-room baseline from an empty room."""
    kwargs: dict[str, object] = {}
    if carrier_hz is not None:
        kwargs["carrier_hz"] = carrier_hz
    source = SyntheticCsiSource(
        seed=seed,
        scenario="empty",
        n_links=n_links,
        n_sub=n_sub,
        snr_db=snr_db,
        duration_s=60.0,
        source_kind=source_kind,
        **kwargs,  # type: ignore[arg-type]
    )
    pipeline = Pipeline(config or make_config(), source_info=source.info())
    pipeline.start_calibration(duration_s=40.0)
    await pipeline.run(source)
    assert pipeline.presence is not None
    assert pipeline.presence.calibrated, "calibration did not complete"
    return pipeline.presence.baselines


async def detect(
    scenario: str,
    baselines: dict[str, LinkBaseline],
    *,
    seed: int = 1,
    n_links: int = 2,
    n_sub: int = 64,
    snr_db: float = 25.0,
    duration_s: float = 20.0,
    carrier_hz: float | None = None,
    source_kind: SourceKind = SourceKind.SYNTHETIC,
    config: AppConfig | None = None,
) -> tuple[list[PresenceEvent], np.ndarray]:
    """Run a scenario against a pre-measured baseline.

    Returns the presence events and the series of fused z-scores, so tests can
    assert on margin rather than only on whether something fired.
    """
    kwargs: dict[str, object] = {}
    if carrier_hz is not None:
        kwargs["carrier_hz"] = carrier_hz
    source = SyntheticCsiSource(
        seed=seed,
        scenario=scenario,
        n_links=n_links,
        n_sub=n_sub,
        snr_db=snr_db,
        duration_s=duration_s,
        source_kind=source_kind,
        **kwargs,  # type: ignore[arg-type]
    )
    pipeline = Pipeline(config or make_config(), source_info=source.info(), baselines=baselines)
    events: list[PresenceEvent] = []
    zs: list[float] = []
    async for frame in source.frames():
        for event in pipeline.step(frame):
            if isinstance(event, PresenceEvent):
                events.append(event)
        assert pipeline.presence is not None
        if pipeline.presence.readings:
            zs.append(pipeline.presence.motion_stat)
    return events, np.asarray(zs)


def event_time_s(event: PresenceEvent) -> float:
    """Seconds into the scenario, using the generator's fixed epoch."""
    from csisense.sources.synthetic import DEFAULT_EPOCH_NS

    return (event.ts_ns - DEFAULT_EPOCH_NS) / 1e9
