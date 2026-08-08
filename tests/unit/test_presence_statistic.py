"""The motion statistic and the quiet-room baseline.

These assert what the statistic was *measured* to do, not what it was designed
to do -- the two turned out to differ. The coherence term was intended to reject
noise that a magnitude-only statistic would fire on; measurement showed it does
not, and one test here pins that negative result down so the idea does not get
quietly re-derived.
"""

from __future__ import annotations

import numpy as np
import pytest

from csisense.presence.baseline import (
    BaselineMismatchError,
    BaselineStore,
    CalibrationRun,
    LinkBaseline,
)
from csisense.presence.statistic import NacdStatistic

FS = 100.0
N_CH = 32


def quiet(rng: np.random.Generator, n: int, sigma: float = 0.01) -> np.ndarray:
    """A static room: a fixed channel shape plus independent noise."""
    base = rng.normal(1.0, 0.2, N_CH)
    return base + rng.normal(0, sigma, (n, N_CH))


def moving(rng: np.random.Generator, n: int, sigma: float = 0.01) -> np.ndarray:
    """A person: a sustained drift of the channel in a changing direction.

    Deliberately a drift rather than a sinusoid along a fixed axis. A walking
    body changes path length monotonically, so the channel moves through a
    sequence of states; a fixed-axis sinusoid instead flips the deviation
    direction every half period, which is not what real motion looks like and
    makes lag-based measures meaningless.
    """
    base = rng.normal(1.0, 0.2, N_CH)
    d1 = rng.normal(0, 1, N_CH)
    d2 = rng.normal(0, 1, N_CH)
    t = np.arange(n)[:, None] / FS
    drift = 0.15 * (np.sin(2 * np.pi * 0.15 * t) * d1 + np.cos(2 * np.pi * 0.11 * t) * d2)
    return base + drift + rng.normal(0, sigma, (n, N_CH))


def normalised(X: np.ndarray) -> np.ndarray:
    """Per-packet normalisation, as the DSP chain applies before the statistic.

    Tests that skip this are testing an input the real system never sees.
    """
    return X / (np.abs(X).mean(axis=1, keepdims=True) + 1e-12)


def run(stat: NacdStatistic, X: np.ndarray) -> np.ndarray:
    return np.array([stat.update(row).value for row in X])


# --- what the statistic actually separates --------------------------------


def test_motion_scores_far_above_noise():
    rng = np.random.default_rng(0)
    q = run(NacdStatistic(fs=FS), normalised(quiet(rng, 4000)))[1000:]
    m = run(NacdStatistic(fs=FS), normalised(moving(rng, 4000)))[1000:]

    assert m.mean() > 10 * q.mean(), (
        f"motion {m.mean():.5f} vs quiet {q.mean():.5f} -- not enough separation"
    )


def test_noise_level_does_not_dominate_the_statistic():
    """Calibration measures the quiet floor per link, so what matters is that a
    noisier link does not produce a wildly larger statistic for the same room."""
    rng = np.random.default_rng(1)
    low = run(NacdStatistic(fs=FS), normalised(quiet(rng, 4000, sigma=0.01)))[1000:]
    high = run(NacdStatistic(fs=FS), normalised(quiet(rng, 4000, sigma=0.05)))[1000:]

    assert high.mean() < 10 * low.mean() + 1e-4


def test_coherence_is_not_a_noise_discriminator():
    """Documents a measured negative, so nobody re-derives the idea and turns
    the gate on expecting it to help.

    The design intent was that deviation direction would be incoherent for noise
    and coherent for bodies. In practice coherence sits high on quiet data too,
    because it is dominated by the slow convergence of the long EWMA rather than
    by anything about the room. That is why coherence_gate defaults to off.
    """
    rng = np.random.default_rng(3)
    s_q = NacdStatistic(fs=FS)
    c_q = np.array([s_q.update(r).coherence for r in normalised(quiet(rng, 4000))])[1000:]

    assert c_q.mean() > 0.5, (
        "coherence is now low on quiet data -- if this changed, re-evaluate "
        "whether the coherence gate is worth enabling"
    )


def test_coherence_gate_is_off_by_default():
    assert NacdStatistic(fs=FS).coherence_gate is False


def test_coherence_gate_can_be_enabled():
    rng = np.random.default_rng(3)
    X = normalised(moving(rng, 2000))
    gated = run(NacdStatistic(fs=FS, coherence_gate=True), X)
    plain = run(NacdStatistic(fs=FS, coherence_gate=False), X)
    # Gating can only reduce the statistic, never raise it.
    assert np.all(gated <= plain + 1e-9)


def test_an_abrupt_scale_step_does_not_score():
    """A receiver gain step must not look like a person.

    The defence is per-packet normalisation upstream, which makes the chain
    scale-invariant -- not the statistic itself. Feeding raw amplitude here
    would test an input the real pipeline never produces.
    """
    rng = np.random.default_rng(4)
    X = quiet(rng, 4000)
    X[2000:] *= 2.0  # AGC jump

    values = run(NacdStatistic(fs=FS), normalised(X))
    baseline = values[1000:1900].mean()
    assert values[1995:2100].max() < baseline + 0.05


# --- freezing -------------------------------------------------------------


def test_freeze_stops_the_reference_from_absorbing_a_person():
    """Without freezing, someone who sits still is absorbed into the long EWMA
    within tau_long and disappears from the statistic entirely."""
    rng = np.random.default_rng(5)
    base = rng.normal(1.0, 0.2, N_CH)
    direction = rng.normal(0, 1, N_CH)
    direction /= np.linalg.norm(direction)

    # Someone arrives at sample 500 and then stays perfectly still.
    X = np.tile(base, (6000, 1))
    X[500:] += 0.2 * direction
    X += rng.normal(0, 0.005, X.shape)

    thawed = NacdStatistic(fs=FS, tau_long_s=5.0)
    frozen = NacdStatistic(fs=FS, tau_long_s=5.0)

    thawed_vals, frozen_vals = [], []
    for i, row in enumerate(X):
        thawed_vals.append(thawed.update(row).value)
        if i == 520:
            frozen.freeze(True)
        frozen_vals.append(frozen.update(row).value)

    # By the end the unfrozen reference has swallowed the person.
    assert np.mean(thawed_vals[-500:]) < 0.05 * np.mean(thawed_vals[520:600])
    # The frozen one still sees them.
    assert np.mean(frozen_vals[-500:]) > 0.5 * np.mean(frozen_vals[520:600])


def test_reset_clears_state():
    rng = np.random.default_rng(6)
    stat = NacdStatistic(fs=FS)
    run(stat, quiet(rng, 500))
    stat.reset()
    assert stat.reference is None
    assert not stat.warm


# --- calibration ----------------------------------------------------------


def test_calibration_is_robust_to_someone_walking_through():
    """MAD, not standard deviation: one intrusion during a two-minute
    calibration must not permanently desensitise the link."""
    clean = CalibrationRun(duration_s=1.0, fs=FS)
    dirty = CalibrationRun(duration_s=1.0, fs=FS)
    rng = np.random.default_rng(7)

    values = np.abs(rng.normal(0.01, 0.002, 1000))
    for v in values:
        clean.add("l", float(v))
        dirty.add("l", float(v))
    # A 5-second intrusion, hugely larger than the noise.
    for _ in range(200):
        dirty.add("l", 5.0)

    c = clean.finish()["l"]
    d = dirty.finish()["l"]
    assert d.sigma < 3 * c.sigma, "an intrusion inflated sigma and blunted the link"


def test_sigma_has_a_floor():
    """A pathologically quiet link would otherwise produce infinite z-scores."""
    run_ = CalibrationRun(duration_s=1.0, fs=FS)
    for _ in range(500):
        run_.add("l", 0.5)
    assert run_.finish()["l"].sigma > 0.0


def test_progress_and_completion():
    run_ = CalibrationRun(duration_s=1.0, fs=FS)
    assert run_.progress() == 0.0
    for _ in range(50):
        run_.add("l", 0.1)
    assert run_.progress() == pytest.approx(0.5)
    assert not run_.complete()
    for _ in range(50):
        run_.add("l", 0.1)
    assert run_.complete()


def test_z_score():
    b = LinkBaseline(mu=1.0, sigma=0.5, n_samples=100)
    assert b.z(2.0) == pytest.approx(2.0)


# --- persistence ----------------------------------------------------------


def test_baseline_roundtrip(tmp_path):
    store = BaselineStore(tmp_path / "baseline.json")
    baselines = {"esp:0": LinkBaseline(mu=0.01, sigma=0.002, n_samples=12000)}
    store.save(baselines, dsp_fingerprint="abc", source_kind="esp", n_sub=64, rate_hz=100.0)

    loaded = store.load(dsp_fingerprint="abc", source_kind="esp", n_sub=64)
    assert loaded["esp:0"].mu == pytest.approx(0.01)
    assert loaded["esp:0"].sigma == pytest.approx(0.002)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"dsp_fingerprint": "different"}, "dsp_fingerprint"),
        ({"source_kind": "nexmon"}, "source_kind"),
        ({"n_sub": 256}, "n_sub"),
    ],
)
def test_baseline_refuses_mismatched_pipeline(tmp_path, kwargs, match):
    """A baseline measured under different processing mis-scales every
    subsequent decision, silently."""
    store = BaselineStore(tmp_path / "b.json")
    store.save(
        {"esp:0": LinkBaseline(1.0, 0.1, 10)},
        dsp_fingerprint="abc",
        source_kind="esp",
        n_sub=64,
        rate_hz=100.0,
    )
    args = {"dsp_fingerprint": "abc", "source_kind": "esp", "n_sub": 64}
    args.update(kwargs)
    with pytest.raises(BaselineMismatchError, match=match):
        store.load(**args)


def test_baseline_save_is_atomic(tmp_path):
    """A power cut mid-write must not leave a file that fails to parse on the
    next boot -- that would silently leave the Pi uncalibrated."""
    path = tmp_path / "b.json"
    store = BaselineStore(path)
    store.save(
        {"esp:0": LinkBaseline(1.0, 0.1, 10)},
        dsp_fingerprint="abc",
        source_kind="esp",
        n_sub=64,
        rate_hz=100.0,
    )
    assert path.exists()
    assert not path.with_suffix(".json.tmp").exists()
