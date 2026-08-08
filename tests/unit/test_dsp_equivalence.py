"""Streaming/batch equivalence for every stateful DSP stage.

This is the most important test in the project.

Training runs offline over whole recordings; inference runs online sample by
sample. If those paths disagree even slightly, the model is evaluated on one
distribution and deployed on another, and **nothing errors** -- the system just
performs worse in the house than it did in the notebook, with no signal as to
why.

So: feed each stage 2000 samples as one batch, and again in randomly sized
chunks, and assert the outputs are identical. Chunk sizes are deliberately
irregular (1..97) because real capture delivers frames in ragged bursts, not
tidy blocks.
"""

from __future__ import annotations

import numpy as np
import pytest

from csisense.dsp.filters import CausalButter
from csisense.dsp.hampel import StreamingHampel
from csisense.dsp.normalise import PerPacketNormaliser
from csisense.dsp.pca import StreamingPCA
from csisense.dsp.stage import DspStage
from csisense.dsp.subcarriers import SubcarrierSelector
from csisense.dsp.window import SlidingWindower
from csisense.wire import SourceKind

N_SAMPLES = 2000
N_CHANNELS = 16


def signal_block(seed: int = 0, n: int = N_SAMPLES, c: int = N_CHANNELS) -> np.ndarray:
    """A signal with structure, drift, outliers and correlation between channels.

    Pure noise would let a broken stage pass by accident.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n)[:, None] / 100.0
    base = np.sin(2 * np.pi * 1.3 * t) + 0.4 * np.sin(2 * np.pi * 7.0 * t)
    drift = 0.002 * np.arange(n)[:, None]
    mix = rng.normal(0, 1, (1, c))
    x = 5.0 + base * mix + drift + rng.normal(0, 0.05, (n, c))
    # Impulsive outliers: the thing Hampel exists for.
    for idx in rng.choice(n, size=20, replace=False):
        x[idx, rng.integers(0, c)] += rng.choice([-1, 1]) * 40.0
    return x.astype(np.float32)


def make_stages() -> dict[str, tuple[DspStage, int]]:
    """Each stage with the input width it expects.

    The subcarrier selector is the odd one out: it consumes a full-width CSI
    vector and narrows it, so it must be fed 64 channels rather than 16.
    """
    return {
        "hampel": (StreamingHampel(window=11, n_sigma=3.0), N_CHANNELS),
        "normaliser": (PerPacketNormaliser(), N_CHANNELS),
        "bandpass_slow": (CausalButter(0.3, 3.0, fs=100.0, order=4), N_CHANNELS),
        "bandpass_fast": (CausalButter(2.0, 20.0, fs=100.0, order=4), N_CHANNELS),
        "pca": (
            StreamingPCA(6, n_channels=N_CHANNELS, fs=100.0, warmup=100, refit_every=250),
            N_CHANNELS,
        ),
        "selector": (SubcarrierSelector(SourceKind.ESP, 64, n_keep=16, mode="auto"), 64),
        "windower": (SlidingWindower(window=64, hop=7), N_CHANNELS),
    }


def run_streaming(stage: DspStage, X: np.ndarray, chunks: list[int]) -> np.ndarray:
    out = []
    i = 0
    for size in chunks:
        for row in X[i : i + size]:
            y = stage.push(row)
            if y is not None:
                out.append(y)
        i += size
    for row in X[i:]:
        y = stage.push(row)
        if y is not None:
            out.append(y)
    return np.stack(out) if out else np.empty((0,))


def random_chunks(rng: np.random.Generator, total: int) -> list[int]:
    chunks: list[int] = []
    remaining = total
    while remaining > 0:
        size = min(int(rng.integers(1, 98)), remaining)
        chunks.append(size)
        remaining -= size
    return chunks


@pytest.mark.parametrize("name", list(make_stages()))
def test_batch_matches_streaming(name):
    rng = np.random.default_rng(42)

    batch_stage, width = make_stages()[name]
    stream_stage, _ = make_stages()[name]
    X = signal_block(seed=1, c=width)

    batch_out = batch_stage.process_batch(X)
    stream_out = run_streaming(stream_stage, X, random_chunks(rng, len(X)))

    assert batch_out.shape == stream_out.shape, (
        f"{name}: batch produced {batch_out.shape}, streaming produced {stream_out.shape}"
    )
    np.testing.assert_allclose(
        batch_out,
        stream_out,
        atol=1e-6,
        rtol=1e-5,
        err_msg=f"{name}: offline and online paths disagree -- this is train/serve skew",
    )


@pytest.mark.parametrize("name", list(make_stages()))
def test_reset_returns_to_initial_behaviour(name):
    """A stage must be reusable; leftover state across sessions would make the
    first minute of every recording subtly different from the rest."""
    stage, width = make_stages()[name]
    X = signal_block(seed=2, n=500, c=width)

    first = stage.process_batch(X)
    stage.reset()
    second = stage.process_batch(X)

    np.testing.assert_allclose(first, second, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("name", ["hampel", "bandpass_slow", "pca", "windower"])
def test_state_roundtrip_preserves_output(name):
    """Checkpoint/restore must be exact, or a Pi restart changes the features
    mid-recording and the model silently sees a different distribution."""
    warm, width = make_stages()[name]
    X = signal_block(seed=3, n=800, c=width)
    warm.process_batch(X[:400])

    restored, _ = make_stages()[name]
    restored.load_state_dict(warm.state_dict())

    np.testing.assert_allclose(
        warm.process_batch(X[400:]),
        restored.process_batch(X[400:]),
        atol=1e-6,
        rtol=1e-5,
    )


def test_declared_latency_matches_reality():
    """The equivalence comparison aligns by declared latency, so a wrong
    declaration would mask a real mismatch."""
    for name, (stage, width) in make_stages().items():
        X = signal_block(seed=4, n=300, c=width)
        n_out = len(stage.process_batch(X))
        if name == "windower":
            expected = 1 + (300 - 64) // 7
        elif name == "pca":
            expected = 300 - 100 + 1  # basis unavailable until warmup completes
        elif name == "hampel":
            # A centred window must fill before the first output, so the stream
            # loses `latency` samples at each end.
            expected = 300 - 2 * stage.latency
        else:
            expected = 300
        assert n_out == expected, f"{name}: expected {expected} outputs, got {n_out}"
