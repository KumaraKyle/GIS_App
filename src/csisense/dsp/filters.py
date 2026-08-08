"""Causal bandpass filtering.

Deliberately **causal** (``sosfilt``, not ``sosfiltfilt``) in both training and
inference. Zero-phase filtering offline would be tempting -- it is cleaner and
removes group delay -- but it is non-causal, cannot run online, and using it for
training while inference uses a causal filter is a textbook train/serve skew.
A model trained on zero-phase data sees gesture onsets that leak backwards in
time and learns to depend on information the live system does not have.

Second-order sections in float64: at ``fs=100`` the slow band 0.3-3 Hz is
0.006-0.06 normalised, which is where a direct-form order-4 bandpass starts to
be numerically delicate.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import signal

from csisense.dsp.stage import DspStage


class CausalButter(DspStage):
    """Streaming Butterworth bandpass with persistent filter state."""

    latency = 0

    def __init__(
        self,
        low_hz: float,
        high_hz: float,
        *,
        fs: float,
        order: int = 4,
        n_channels: int | None = None,
    ) -> None:
        nyq = fs / 2.0
        if not 0.0 < low_hz < high_hz < nyq:
            raise ValueError(
                f"band {low_hz}-{high_hz}Hz is invalid for fs={fs}Hz "
                f"(must satisfy 0 < low < high < {nyq})"
            )
        self.low_hz = low_hz
        self.high_hz = high_hz
        self.fs = fs
        self.order = order
        self.sos = signal.butter(
            order, [low_hz, high_hz], btype="band", fs=fs, output="sos"
        ).astype(np.float64)
        self._zi: np.ndarray | None = None
        if n_channels is not None:
            self._init_state(n_channels)

    def _init_state(self, n_channels: int) -> None:
        # sosfilt state is (n_sections, 2, n_channels) when filtering along axis 0.
        self._zi = np.zeros((self.sos.shape[0], 2, n_channels), dtype=np.float64)

    def reset(self) -> None:
        if self._zi is not None:
            self._zi[:] = 0.0

    def resize(self, n_channels: int | None = None) -> None:
        """Discard filter state so it re-initialises at a new channel count.

        Needed when the subcarrier selector commits to its ranked subset and the
        feature vector narrows underneath the filter.
        """
        self._zi = None
        if n_channels is not None:
            self._init_state(n_channels)

    def push(self, x: np.ndarray) -> np.ndarray:
        v = np.asarray(x, dtype=np.float64)
        if self._zi is None:
            self._init_state(v.shape[0])
        y, self._zi = signal.sosfilt(self.sos, v[None, :], axis=0, zi=self._zi)
        return y[0].astype(np.float32)

    def process_batch(self, X: np.ndarray) -> np.ndarray:
        """Vectorised over time.

        ``sosfilt`` over the whole array is mathematically identical to feeding
        rows one at a time with carried state -- but "mathematically identical"
        is exactly the sort of claim that quietly stops being true, so
        test_dsp_equivalence asserts it.
        """
        V = np.asarray(X, dtype=np.float64)
        if self._zi is None:
            self._init_state(V.shape[1])
        Y, self._zi = signal.sosfilt(self.sos, V, axis=0, zi=self._zi)
        return Y.astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {"zi": None if self._zi is None else self._zi.tolist()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        zi = state.get("zi")
        self._zi = None if zi is None else np.asarray(zi, dtype=np.float64)

    def fingerprint_fields(self) -> dict[str, Any]:
        return {
            "low_hz": self.low_hz,
            "high_hz": self.high_hz,
            "order": self.order,
            "fs": self.fs,
        }


__all__ = ["CausalButter"]
