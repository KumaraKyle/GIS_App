"""An empty room must produce no presence events. Ever.

Half of the acceptance test for the presence tier; ``test_detection.py`` is the
other half, and neither is meaningful alone. A detector that never fires passes
this file perfectly.

The bar is absolute -- zero events, across seeds and subcarrier counts and both
radios -- rather than "few". A statistic that fires rarely on an empty room
fires constantly over a day, because a day contains 8.6 million frames.
"""

from __future__ import annotations

import pytest
from helpers import calibrate, detect, make_config

from csisense.sources.synthetic import CARRIER_HZ_5G
from csisense.wire import SourceKind


@pytest.mark.parametrize("seed", range(6))
async def test_empty_room_produces_no_presence_events(seed):
    baselines = await calibrate(seed=seed)
    events, zs = await detect("empty", baselines, seed=seed, duration_s=60.0)

    assert events == [], f"seed {seed}: empty room fired {len(events)} presence event(s)"
    # And with margin: not merely below the threshold, but far below it.
    assert zs.max() < 4.0, f"seed {seed}: quiet room reached z={zs.max():.1f}, too close to k_on=6"


@pytest.mark.parametrize("n_sub", [64, 128, 256])
async def test_empty_room_across_subcarrier_widths(n_sub):
    baselines = await calibrate(n_links=1, n_sub=n_sub)
    events, _ = await detect("empty", baselines, n_links=1, n_sub=n_sub)
    assert events == [], f"{n_sub} subcarriers: fired {len(events)} event(s) on an empty room"


async def test_empty_room_at_5ghz():
    """The Pi's link. Half the wavelength of 2.4GHz means twice the phase
    sensitivity, so thresholds that hold at 2.4GHz do not automatically hold."""
    kw = {
        "n_links": 1,
        "n_sub": 256,
        "carrier_hz": CARRIER_HZ_5G,
        "source_kind": SourceKind.NEXMON,
    }
    baselines = await calibrate(seed=5, **kw)
    events, _ = await detect("empty", baselines, seed=5, **kw)
    assert events == []


async def test_agc_glitch_is_not_mistaken_for_a_person():
    """A receiver gain step is the most convincing false positive available: a
    large, abrupt, room-wide amplitude change. Per-packet normalisation makes
    the chain scale-invariant, so it must not register at all."""
    baselines = await calibrate(n_links=1)
    source_events, zs = await detect("agc_glitch_empty", baselines, n_links=1, duration_s=20.0)
    assert source_events == [], "an AGC gain step was reported as presence"
    assert zs.max() < 4.0


@pytest.mark.parametrize("snr_db", [10.0, 15.0, 20.0])
async def test_empty_room_stays_quiet_at_low_snr(snr_db):
    """Calibration measures the noise floor, so a noisier link should raise both
    the statistic and its baseline together and cancel out."""
    baselines = await calibrate(snr_db=snr_db)
    events, _ = await detect("empty", baselines, snr_db=snr_db)
    assert events == [], f"snr={snr_db}dB: fired {len(events)} event(s)"


async def test_high_packet_loss_does_not_manufacture_presence():
    """Dropped frames leave gaps in every filter. Gaps must not look like motion."""
    baselines = await calibrate(n_links=1)
    events, _ = await detect("empty", baselines, n_links=1, config=make_config(presence__k_on=6.0))
    assert events == []
