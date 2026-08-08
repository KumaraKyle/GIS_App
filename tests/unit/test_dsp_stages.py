"""Behaviour of individual DSP stages.

Equivalence tests prove the offline and online paths agree. These prove they
agree on something *correct*.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from csisense.config import DspConfig
from csisense.dsp.chain import DspChain, chain_fingerprint
from csisense.dsp.doppler import StftDoppler, csi_ratio, pick_reference_subcarrier
from csisense.dsp.filters import CausalButter
from csisense.dsp.hampel import StreamingHampel
from csisense.dsp.normalise import PerPacketNormaliser
from csisense.dsp.pca import StreamingPCA
from csisense.dsp.subcarriers import SubcarrierSelector, valid_subcarrier_mask
from csisense.wire import FLAG_GAIN_LOCKED, CsiFrame, SourceKind

# --- subcarrier selection -------------------------------------------------


@pytest.mark.parametrize(
    ("kind", "n_sub", "sig_mode", "expected"),
    [
        # The 802.11 data-subcarrier counts. Drift here means the DSP is
        # silently treating guard bands or pilots as if they carried room
        # information.
        (SourceKind.NEXMON, 256, 1, 234),  # VHT80 -- the Pi's link
        (SourceKind.NEXMON, 128, 1, 108),  # HT40
        (SourceKind.ESP, 64, 1, 52),  # HT20 -- the ESP32's supported path
        (SourceKind.ESP, 128, 1, 108),  # HT40, experimental
        (SourceKind.ESP, 64, 0, 48),  # legacy 802.11a/g
    ],
)
def test_valid_subcarrier_counts(kind, n_sub, sig_mode, expected):
    assert int(valid_subcarrier_mask(kind, n_sub, sig_mode=sig_mode).sum()) == expected


def test_pilots_are_excluded():
    """A pilot carries a known reference symbol, so its variation is the
    transmitter's behaviour, not a person walking past."""
    mask = valid_subcarrier_mask(SourceKind.ESP, 64)
    dc = 32
    for pilot in (-21, -7, 7, 21):
        assert not mask[dc + pilot], f"pilot at {pilot} should be excluded"


def test_dc_is_excluded():
    for kind, n_sub in [(SourceKind.ESP, 64), (SourceKind.NEXMON, 256)]:
        assert not valid_subcarrier_mask(kind, n_sub)[n_sub // 2]


def test_legacy_sig_mode_uses_narrower_occupancy():
    """Legacy LLTF occupies fewer bins than HT-LTF; using the HT map on legacy
    frames would feed the chain bins that contain nothing."""
    ht = valid_subcarrier_mask(SourceKind.ESP, 64, sig_mode=1).sum()
    legacy = valid_subcarrier_mask(SourceKind.ESP, 64, sig_mode=0).sum()
    assert legacy < ht


def test_selector_freezes_to_requested_width():
    sel = SubcarrierSelector(SourceKind.ESP, 64, n_keep=16)
    rng = np.random.default_rng(0)
    for _ in range(200):
        sel.push(rng.normal(5.0, 1.0, 64).astype(np.float32))

    assert not sel.frozen
    idx = sel.freeze()
    assert sel.frozen
    assert len(idx) == 16
    assert sel.push(rng.normal(5.0, 1.0, 64).astype(np.float32)).shape == (16,)


def test_selector_prefers_varying_subcarriers():
    """Ranking must pick the informative bins, not merely the loud ones."""
    sel = SubcarrierSelector(SourceKind.ESP, 64, n_keep=4)
    rng = np.random.default_rng(1)
    lively = [10, 20, 40, 50]
    for _ in range(300):
        x = np.full(64, 5.0, dtype=np.float32)
        x += rng.normal(0, 0.001, 64)
        # A large but *constant* offset: loud, carries nothing.
        x[30] += 100.0
        for i in lively:
            x[i] += rng.normal(0, 1.0)
        sel.push(x)

    chosen = set(sel.freeze().tolist())
    assert 30 not in chosen, "selected a loud but constant subcarrier"
    assert len(chosen & set(lively)) >= 3


def test_selector_indices_are_in_the_fingerprint():
    """A model trained on one ranked subset must not be fed another."""
    a = SubcarrierSelector(SourceKind.ESP, 64, n_keep=8, mode="explicit", explicit=[1, 2, 3, 4])
    b = SubcarrierSelector(SourceKind.ESP, 64, n_keep=8, mode="explicit", explicit=[5, 6, 7, 8])
    assert a.fingerprint_fields() != b.fingerprint_fields()


# --- Hampel ---------------------------------------------------------------


def test_hampel_removes_an_impulse():
    h = StreamingHampel(window=11, n_sigma=3.0)
    x = np.full((50, 1), 1.0, dtype=np.float32)
    x += np.random.default_rng(0).normal(0, 0.01, (50, 1)).astype(np.float32)
    x[25, 0] = 99.0

    out = h.process_batch(x)
    assert out.max() < 2.0, "the impulse survived"


def test_hampel_preserves_a_real_step():
    """A genuine level change is not an outlier. Over-aggressive filtering here
    would erase the very transitions a gesture consists of."""
    h = StreamingHampel(window=11, n_sigma=3.0)
    x = np.concatenate([np.zeros((40, 1)), np.ones((40, 1))]).astype(np.float32)
    out = h.process_batch(x)
    assert out[-1, 0] == pytest.approx(1.0)
    assert out[0, 0] == pytest.approx(0.0)


@given(
    data=st.lists(
        st.floats(min_value=-100, max_value=100, allow_nan=False), min_size=30, max_size=200
    )
)
@settings(max_examples=50, deadline=None)
def test_hampel_never_invents_values(data):
    """Output must always be either the input or a local median -- never a value
    outside the range the data actually contained."""
    x = np.asarray(data, dtype=np.float32)[:, None]
    out = StreamingHampel(window=11).process_batch(x)
    assert out.min() >= x.min() - 1e-4
    assert out.max() <= x.max() + 1e-4


def test_hampel_rejects_even_window():
    with pytest.raises(ValueError, match="odd"):
        StreamingHampel(window=10)


# --- bandpass -------------------------------------------------------------


def test_bandpass_rejects_dc():
    """A constant input is the static room. It must not reach the detector."""
    bp = CausalButter(0.3, 3.0, fs=100.0, order=4)
    out = bp.process_batch(np.ones((3000, 1), dtype=np.float32))
    assert abs(out[-500:]).max() < 1e-3


def test_bandpass_passes_in_band_and_rejects_out_of_band():
    t = np.arange(4000) / 100.0
    in_band = np.sin(2 * np.pi * 1.0 * t)[:, None].astype(np.float32)
    out_band = np.sin(2 * np.pi * 30.0 * t)[:, None].astype(np.float32)

    passed = CausalButter(0.3, 3.0, fs=100.0).process_batch(in_band)[-1000:]
    blocked = CausalButter(0.3, 3.0, fs=100.0).process_batch(out_band)[-1000:]

    assert np.abs(passed).max() > 0.5
    assert np.abs(blocked).max() < 0.05


def test_bandpass_settles():
    """Order-4 at 0.006 normalised is numerically delicate; an unstable filter
    would ring forever and look like continuous motion."""
    bp = CausalButter(0.3, 3.0, fs=100.0, order=4)
    x = np.zeros((3000, 1), dtype=np.float32)
    x[0, 0] = 1.0
    out = bp.process_batch(x)
    assert np.abs(out[-100:]).max() < 1e-6


def test_bandpass_rejects_impossible_band():
    with pytest.raises(ValueError, match="invalid"):
        CausalButter(30.0, 60.0, fs=100.0)  # high above Nyquist


# --- normalisation --------------------------------------------------------


def test_per_packet_normalisation_is_scale_invariant():
    """This is what stops an AGC step masquerading as motion."""
    n = PerPacketNormaliser()
    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    np.testing.assert_allclose(n.push(x), n.push(x * 7.3), rtol=1e-5)


# --- PCA ------------------------------------------------------------------


def test_pca_reduces_dimensionality_and_drops_first():
    p = StreamingPCA(6, n_channels=16, fs=100.0, warmup=50, drop_first=True)
    rng = np.random.default_rng(0)
    out = None
    for _ in range(300):
        out = p.push(rng.normal(0, 1, 16))
    assert out is not None
    assert out.shape == (5,), "drop_first should remove one component"


def test_pca_basis_is_sign_stable():
    """Eigenvectors are defined up to sign. An unfixed sign flips arbitrarily
    between refits, which alone would destroy determinism."""
    rng = np.random.default_rng(3)
    data = rng.normal(0, 1, (2000, 12)).astype(np.float32)

    a = StreamingPCA(4, n_channels=12, fs=100.0, warmup=50, refit_every=100)
    b = StreamingPCA(4, n_channels=12, fs=100.0, warmup=50, refit_every=100)
    np.testing.assert_allclose(a.process_batch(data), b.process_batch(data), atol=1e-6)


def test_pca_drops_the_static_direction():
    """Component 0 is the room. After dropping it, a purely static offset should
    contribute nothing."""
    p = StreamingPCA(3, n_channels=8, fs=100.0, warmup=50, drop_first=True)
    rng = np.random.default_rng(5)
    static = np.array([10.0, -4.0, 7.0, 2.0, 0.5, -3.0, 8.0, 1.0])

    outs = []
    for _ in range(600):
        outs.append(p.push(static + rng.normal(0, 0.01, 8)))
    tail = np.stack([o for o in outs[-100:] if o is not None])
    assert np.abs(tail).mean() < 0.5


def test_pca_rejects_too_many_components():
    with pytest.raises(ValueError, match="cannot take"):
        StreamingPCA(20, n_channels=8)


# --- Doppler --------------------------------------------------------------


def test_csi_ratio_cancels_common_phase():
    """The whole point: a random per-packet phase must vanish."""
    rng = np.random.default_rng(0)
    H = rng.normal(0, 1, 32) + 1j * rng.normal(0, 1, 32)
    offset = np.exp(1j * 2.1)

    np.testing.assert_allclose(csi_ratio(H, 5), csi_ratio(H * offset, 5), atol=1e-9)


def test_reference_subcarrier_avoids_deep_fades():
    """Dividing by a bin sitting in a null amplifies noise enormously."""
    rng = np.random.default_rng(1)
    H = rng.normal(1.0, 0.05, (100, 8)) + 1j * rng.normal(0, 0.05, (100, 8))
    H[:, 3] *= 0.001  # a deep fade
    assert pick_reference_subcarrier(H) != 3


def test_doppler_distinguishes_direction():
    """Signed Doppler is the reason for the CSI ratio. Approach and retreat must
    not produce the same spectrum."""
    fs, n = 100.0, 400
    t = np.arange(n) / fs
    approaching = np.exp(2j * np.pi * 5.0 * t)[:, None]
    retreating = np.exp(-2j * np.pi * 5.0 * t)[:, None]

    def spectrum(sig):
        d = StftDoppler(fs=fs, nperseg=64, noverlap=48, max_hz=20.0)
        cols = [c for row in sig if (c := d.push(row)) is not None]
        return np.stack(cols).mean(axis=0), d.freqs

    a, freqs = spectrum(approaching)
    r, _ = spectrum(retreating)

    assert freqs[int(np.argmax(a))] > 0
    assert freqs[int(np.argmax(r))] < 0


def test_doppler_bin_count():
    """64-point FFT at 100Hz gives 1.5625Hz bins, so |f| <= 20Hz keeps k=-12..12."""
    d = StftDoppler(fs=100.0, nperseg=64, noverlap=48, max_hz=20.0)
    assert d.n_bins == 25
    assert d.freqs[0] == pytest.approx(-18.75)
    assert d.freqs[-1] == pytest.approx(18.75)


# --- chain ----------------------------------------------------------------


def make_frame(n_sub=64, seed=0, gain_locked=True):
    rng = np.random.default_rng(seed)
    csi = (rng.normal(0, 20, n_sub) + 1j * rng.normal(0, 20, n_sub)).astype(np.complex64)
    return CsiFrame(
        source_kind=SourceKind.ESP,
        sensor_id=0,
        csi=csi,
        ts_host_ns=seed * 10_000_000,
        seq=seed,
        rssi=-50,
        flags=FLAG_GAIN_LOCKED if gain_locked else 0,
    )


def test_chain_produces_features_after_warmup():
    chain = DspChain(DspConfig(), source_kind=SourceKind.ESP, n_sub=64)
    out = [chain.process(make_frame(seed=i)) for i in range(400)]
    produced = [f for f in out if f is not None]

    assert produced, "chain never produced a feature"
    assert produced[-1].amplitude.ndim == 1
    assert produced[-1].link_id == "esp:0"


def test_chain_fingerprint_changes_with_config():
    base = DspChain(DspConfig(), source_kind=SourceKind.ESP, n_sub=64)
    changed = DspChain(DspConfig(hampel_n_sigma=5.0), source_kind=SourceKind.ESP, n_sub=64)
    assert base.fingerprint() != changed.fingerprint()


def test_chain_fingerprint_differs_across_sources():
    """A 256-subcarrier nexmon model must not load against 64-subcarrier ESP
    data. The fingerprint is what makes that impossible."""
    esp = DspChain(DspConfig(), source_kind=SourceKind.ESP, n_sub=64)
    nexmon = DspChain(DspConfig(), source_kind=SourceKind.NEXMON, n_sub=256)
    assert esp.fingerprint() != nexmon.fingerprint()


def test_chain_fingerprint_is_stable_across_instances():
    a = DspChain(DspConfig(), source_kind=SourceKind.ESP, n_sub=64)
    b = DspChain(DspConfig(), source_kind=SourceKind.ESP, n_sub=64)
    assert a.fingerprint() == b.fingerprint()


def test_chain_fingerprint_ignores_runtime_state():
    """Otherwise every restart would invalidate every model bundle."""
    chain = DspChain(DspConfig(), source_kind=SourceKind.ESP, n_sub=64)
    before = chain.fingerprint()
    for i in range(300):
        chain.process(make_frame(seed=i))
    assert chain.fingerprint() == before


def test_chain_state_refuses_a_mismatched_fingerprint():
    a = DspChain(DspConfig(), source_kind=SourceKind.ESP, n_sub=64)
    b = DspChain(DspConfig(hampel_window=21), source_kind=SourceKind.ESP, n_sub=64)
    with pytest.raises(ValueError, match="fingerprint"):
        b.load_state_dict(a.state_dict())


def test_chain_marks_gain_estimated_when_unlocked():
    """Whether gain was locked propagates all the way to the dataset builder,
    which excludes unlocked stretches from amplitude-feature training."""
    chain = DspChain(DspConfig(), source_kind=SourceKind.ESP, n_sub=64)
    feats = [chain.process(make_frame(seed=i, gain_locked=False)) for i in range(200)]
    produced = [f for f in feats if f is not None]
    assert produced and all(f.gain_estimated for f in produced)


def test_fingerprint_helper_is_order_independent_of_dict_keys():
    chain = DspChain(DspConfig(), source_kind=SourceKind.ESP, n_sub=64)
    assert chain_fingerprint(chain.spec(), chain.stages) == chain.fingerprint()
