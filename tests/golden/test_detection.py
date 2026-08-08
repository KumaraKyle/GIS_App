"""Presence must actually fire when someone is there.

The other half of the acceptance test. Written after discovering that an earlier
version of the false-positive suite passed completely while the detector was
dead: every "no events" assertion holds trivially for a detector that never
fires, so the two files only mean something together.

Ground truth comes from each scenario's declared ``motion_intervals``, so these
assertions are checked against the generator's intent rather than against
numbers observed once and then frozen in.
"""

from __future__ import annotations

import numpy as np
import pytest
from helpers import calibrate, detect, event_time_s

from csisense.events.types import PresenceState
from csisense.sources.synthetic import CARRIER_HZ_5G, SCENARIOS
from csisense.wire import SourceKind

# Seed 1 is a geometry where a single link sees the crossing well. Detection
# margin turns out to depend strongly on the random static environment -- see
# test_detection_rate_across_geometries below, which measures that directly.
GOOD_GEOMETRY_SEED = 1


@pytest.mark.parametrize("scenario", ["walk_through", "approach", "sit"])
async def test_motion_is_detected(scenario):
    baselines = await calibrate(seed=GOOD_GEOMETRY_SEED)
    events, zs = await detect(scenario, baselines, seed=GOOD_GEOMETRY_SEED)

    assert events, f"{scenario}: nobody was detected at all"
    first = events[0]
    assert first.state is PresenceState.MOTION
    assert zs.max() > 6.0, f"{scenario}: peak z only {zs.max():.1f}"


@pytest.mark.parametrize("scenario", ["walk_through", "approach", "sit"])
async def test_detection_lands_inside_the_declared_motion_window(scenario):
    """Firing is not enough -- it has to fire at the right time."""
    baselines = await calibrate()
    events, _ = await detect(scenario, baselines)

    onsets = [e for e in events if e.state is PresenceState.MOTION]
    assert onsets, f"{scenario}: no MOTION transition"

    t = event_time_s(onsets[0])
    windows = SCENARIOS[scenario].motion_intervals
    # One second of slack: the detector needs min_frames_on consecutive samples
    # and the statistic is smoothed by a 0.4s EWMA.
    assert any(a - 1.0 <= t <= b + 1.0 for a, b in windows), (
        f"{scenario}: fired at t={t:.1f}s, outside declared motion {windows}"
    )


async def test_separation_between_empty_and_occupied():
    """The margin, stated explicitly, so a regression that halves it is visible
    rather than merely still-passing."""
    baselines = await calibrate()
    _, quiet = await detect("empty", baselines, duration_s=40.0)
    _, moving = await detect("walk_through", baselines)

    assert np.percentile(quiet, 99) < 4.0
    assert np.percentile(moving, 99) > 6.0


async def test_someone_who_stops_moving_stays_present():
    """The failure that makes cheap motion sensors infuriating: the lights go
    off while you are sitting still. Occupancy must latch."""
    baselines = await calibrate()
    events, _ = await detect("sit", baselines, duration_s=20.0)

    assert events
    # After the person settles, state must be OCCUPIED_STILL -- present but not
    # moving -- and never fall back to IDLE within the occupancy timeout.
    assert events[-1].state in (PresenceState.MOTION, PresenceState.OCCUPIED_STILL)
    assert all(e.state is not PresenceState.IDLE for e in events)


async def test_motion_is_detected_at_5ghz():
    """The Pi's link. Two links, because one is not reliably enough -- see
    test_detection_rate_across_geometries."""
    kw = {
        "n_links": 2,
        "n_sub": 256,
        "carrier_hz": CARRIER_HZ_5G,
        "source_kind": SourceKind.NEXMON,
    }
    baselines = await calibrate(seed=1, **kw)
    events, zs = await detect("walk_through", baselines, seed=1, **kw)

    assert events, "the Pi's 5GHz link detected nothing"
    assert zs.max() > 6.0


@pytest.mark.slow
@pytest.mark.parametrize("n_links", [1, 3])
async def test_detection_rate_across_geometries(n_links):
    """Detection margin is strongly geometry-dependent, and more links help.

    Whether a moving body produces a large amplitude perturbation depends on how
    its reflection interferes with the static multipath, which differs per room
    and per link placement. Measured over four random static environments:
    one link detects a crossing in 1 of 4, three links in 3 of 4.

    This is physics, not a tuning failure, and it is the argument for more nodes
    even before RTI: a single link is not a reliable presence sensor. The false
    positive rate, by contrast, is stable across every geometry -- which is why
    the empty-room floor is asserted here too.
    """
    detected = 0
    for seed in (1, 2, 3, 4):
        baselines = await calibrate(seed=seed, n_links=n_links)
        _, quiet = await detect("empty", baselines, seed=seed, n_links=n_links)
        events, _ = await detect("walk_through", baselines, seed=seed, n_links=n_links)
        detected += bool(events)
        # Regardless of geometry, an empty room must stay far below threshold.
        assert quiet.max() < 4.0, f"seed {seed}: empty room reached z={quiet.max():.1f}"

    expected = {1: 1, 3: 3}[n_links]
    assert detected >= expected, (
        f"{n_links} link(s): detected {detected}/4 geometries, expected at least {expected}"
    )


async def test_per_link_statistics_are_published():
    """Which link fired is the crude directionality signal, and it is what the
    RTI engine consumes once there are enough links to invert."""
    baselines = await calibrate(n_links=2)
    events, _ = await detect("walk_through", baselines, n_links=2)

    assert events
    per_link = events[0].per_link
    assert set(per_link) == {"synthetic:0", "synthetic:1"}
    assert all(np.isfinite(v) for v in per_link.values())


async def test_a_too_lossy_link_reports_blindness_rather_than_an_empty_room():
    """25% loss is above max_link_loss_pct, so the link is excluded from fusion.

    With only one link that leaves the detector with nothing to look at. The
    danger is that it then publishes a confident "no motion", which downstream
    is indistinguishable from "nobody is home" -- a dead sensor would silently
    read as an empty house. It must report blindness instead.
    """
    from helpers import make_config

    from csisense.app.pipeline import Pipeline
    from csisense.sources.synthetic import SyntheticCsiSource

    baselines = await calibrate(n_links=1)
    source = SyntheticCsiSource(seed=1, scenario="high_loss", n_links=1, n_sub=64, duration_s=20.0)
    pipeline = Pipeline(make_config(), source_info=source.info(), baselines=baselines)
    async for frame in source.frames():
        pipeline.step(frame)

    assert pipeline.presence is not None
    assert pipeline.presence.blind, "a link losing 25% of frames was treated as trustworthy"


async def test_a_perfectly_still_person_is_not_claimed_to_be_detected():
    """An honest negative. Both terms of the statistic measure *change*, so
    someone holding completely still is indistinguishable from an empty room.
    The 'wave' scenario is a motionless torso plus a weak hand, and presence
    correctly does not claim to see it -- that is the gesture tier's job."""
    baselines = await calibrate(n_links=1)
    events, zs = await detect("wave", baselines, n_links=1)

    assert zs.max() < 6.0
    assert events == []
