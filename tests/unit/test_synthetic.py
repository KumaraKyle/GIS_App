"""Tests for the synthetic source.

These assert the *physics* is right, not just that bytes flow. If the generator
does not actually encode motion as phase rotation, every downstream test built
on it is worthless.
"""

from __future__ import annotations

import numpy as np
import pytest

from csisense.sources.synthetic import SCENARIOS, SyntheticCsiSource, available_scenarios
from csisense.wire import SourceKind


async def collect(source: SyntheticCsiSource) -> list:
    return [f async for f in source.frames()]


async def test_deterministic_for_a_given_seed():
    """Golden regression tests depend on this absolutely."""
    a = await collect(SyntheticCsiSource(seed=7, scenario="walk_through", duration_s=1.0))
    b = await collect(SyntheticCsiSource(seed=7, scenario="walk_through", duration_s=1.0))

    assert len(a) == len(b)
    for fa, fb in zip(a, b, strict=True):
        np.testing.assert_array_equal(fa.csi, fb.csi)


async def test_timestamps_are_reproducible():
    """Regression: the generator once stamped frames from the wall clock, which
    made output differ between runs and would have defeated golden tests."""
    a = await collect(SyntheticCsiSource(seed=7, n_links=1, duration_s=0.5))
    b = await collect(SyntheticCsiSource(seed=7, n_links=1, duration_s=0.5))
    assert [f.ts_host_ns for f in a] == [f.ts_host_ns for f in b]


async def test_epoch_is_overridable():
    frames = await collect(SyntheticCsiSource(seed=1, n_links=1, duration_s=0.1, epoch_ns=5_000))
    assert frames[0].ts_host_ns == 5_000


async def test_different_seeds_differ():
    a = await collect(SyntheticCsiSource(seed=1, duration_s=1.0))
    b = await collect(SyntheticCsiSource(seed=2, duration_s=1.0))
    assert not np.array_equal(a[0].csi, b[0].csi)


async def test_frame_count_and_shape():
    source = SyntheticCsiSource(seed=0, n_links=3, n_sub=128, rate_hz=100.0, duration_s=2.0)
    frames = await collect(source)

    assert len(frames) == 3 * 200
    assert all(f.n_sub == 128 for f in frames)
    assert all(f.source_kind is SourceKind.SYNTHETIC for f in frames)
    assert source.info().n_links == 3
    assert {f.sensor_id for f in frames} == {0, 1, 2}


async def test_empty_scenario_is_static():
    """An empty room must produce only noise around a constant channel.

    This is the reference the presence detector's false-positive test relies on.
    """
    source = SyntheticCsiSource(seed=3, scenario="empty", n_links=1, snr_db=40.0, duration_s=5.0)
    frames = await collect(source)

    amps = np.array([np.abs(f.csi) for f in frames])
    # Per-subcarrier variation across time should be small relative to the mean.
    rel_std = float(np.mean(np.std(amps, axis=0) / (np.mean(amps, axis=0) + 1e-9)))
    assert rel_std < 0.05, f"empty room varies too much ({rel_std:.3f}); noise model is wrong"


async def test_motion_produces_more_variation_than_empty():
    """The core claim of the whole project, in one assertion."""
    common = {"seed": 5, "n_links": 1, "snr_db": 40.0, "duration_s": 20.0}
    empty = await collect(SyntheticCsiSource(scenario="empty", **common))
    moving = await collect(SyntheticCsiSource(scenario="walk_through", **common))

    def variability(frames) -> float:
        amps = np.array([np.abs(f.csi) for f in frames])
        amps = amps / (amps.mean(axis=1, keepdims=True) + 1e-9)
        return float(np.mean(np.std(amps, axis=0)))

    assert variability(moving) > 5 * variability(empty)


async def test_moving_reflector_rotates_phase():
    """A person moving half a metre should sweep the phase through several full
    turns at 2.4GHz -- that rotation is the signal being sensed."""
    source = SyntheticCsiSource(seed=1, scenario="approach", n_links=1, duration_s=10.0)
    mid = source._n_sub // 2

    phases = np.unwrap([np.angle(source.channel_at(0, t)[mid]) for t in np.arange(0, 10, 0.01)])
    total_rotation = abs(phases[-1] - phases[0])

    assert total_rotation > 2 * np.pi, (
        f"only {total_rotation / (2 * np.pi):.2f} turns of phase rotation; "
        "the reflector model is not producing usable motion"
    )


async def test_sit_scenario_goes_still_while_still_present():
    """The case that breaks naive detectors: motion stops, person remains."""
    source = SyntheticCsiSource(seed=1, scenario="sit", n_links=1, duration_s=20.0)

    def movement(t0: float, t1: float) -> float:
        h = np.array([source.channel_at(0, t) for t in np.arange(t0, t1, 0.01)])
        return float(np.mean(np.std(np.abs(h), axis=0)))

    assert movement(8.5, 10.5) > 10 * movement(12.0, 19.0)


async def test_packet_loss_drops_frames():
    lossy = await collect(
        SyntheticCsiSource(
            seed=2, scenario="walk_through", n_links=1, loss_pct=50.0, duration_s=10.0
        )
    )
    clean = await collect(
        SyntheticCsiSource(
            seed=2, scenario="walk_through", n_links=1, loss_pct=0.0, duration_s=10.0
        )
    )
    assert 0.4 < len(lossy) / len(clean) < 0.6


async def test_sensor_dropout_silences_one_link_only():
    source = SyntheticCsiSource(seed=1, scenario="sensor_dropout", n_links=2, duration_s=20.0)
    frames = await collect(source)

    def seconds_present(link: int) -> set[int]:
        return {int(f.ts_device_us / 1e6) for f in frames if f.sensor_id == link}

    # Link 1 drops out from 7s to 12s; link 0 must be unaffected.
    assert seconds_present(1).isdisjoint({8, 9, 10, 11})
    assert {8, 9, 10, 11} <= seconds_present(0)


async def test_agc_glitch_changes_amplitude_abruptly():
    """Uncontrolled AGC is the thing gain locking exists to prevent; the
    generator must reproduce it so compensation can be tested."""
    source = SyntheticCsiSource(seed=1, scenario="agc_glitch", n_links=1, duration_s=10.0)
    frames = await collect(source)

    def mean_amp(t0: float, t1: float) -> float:
        sel = [f for f in frames if t0 <= f.ts_device_us / 1e6 < t1]
        return float(np.mean([np.abs(f.csi).mean() for f in sel]))

    before, after = mean_amp(3.0, 4.5), mean_amp(5.5, 7.0)
    assert abs(np.log2(after / before)) > 0.5, "AGC glitch is not visible in amplitude"


async def test_gain_locked_flag_reflects_scenario():
    clean = await collect(SyntheticCsiSource(seed=1, scenario="walk_through", duration_s=0.5))
    glitchy = await collect(SyntheticCsiSource(seed=1, scenario="agc_glitch", duration_s=0.5))
    assert all(f.gain_locked for f in clean)
    assert not any(f.gain_locked for f in glitchy)


def test_unknown_scenario_is_rejected_with_a_useful_message():
    with pytest.raises(ValueError, match="walk_through"):
        SyntheticCsiSource(scenario="nope")


@pytest.mark.parametrize("name", list(available_scenarios()))
async def test_every_scenario_runs(name):
    """Guards against a scenario being added with a broken trajectory."""
    frames = await collect(SyntheticCsiSource(seed=0, scenario=name, n_links=2, duration_s=2.0))
    assert frames, f"scenario {name} produced no frames"
    assert all(np.all(np.isfinite(f.csi.view(np.float32))) for f in frames)


def test_scenarios_declare_ground_truth():
    """Scenarios with motion must say when, so detector tests can assert
    against something other than eyeballed numbers."""
    for name in ("walk_through", "approach", "sit", "wave"):
        assert SCENARIOS[name].motion_intervals, f"{name} declares no motion intervals"
    assert not SCENARIOS["empty"].motion_intervals
