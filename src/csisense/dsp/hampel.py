"""Streaming Hampel filter.

CSI is full of impulsive outliers -- a single packet decoded during interference
produces a spike an order of magnitude off, and a bandpass filter will happily
ring on it for a second afterwards. The Hampel filter replaces a sample with the
local median when it sits more than ``n_sigma`` robust deviations away.

Median-and-MAD rather than mean-and-std, because the mean is dragged by the very
outlier being detected.

This is a *centred* filter: judging sample *i* uses ``k`` samples either side, so
it introduces exactly ``k`` samples of latency. Batch and streaming must agree on
the edge policy or the equivalence test fails, so both use the same truncated
window at the start of the stream.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from csisense.dsp.stage import DspStage

MAD_TO_SIGMA = 1.4826


class StreamingHampel(DspStage):
    """Per-channel Hampel outlier replacement over a centred window."""

    def __init__(self, window: int = 11, n_sigma: float = 3.0) -> None:
        if window < 3 or window % 2 == 0:
            raise ValueError(f"hampel window must be odd and >= 3, got {window}")
        self.window = window
        self.n_sigma = n_sigma
        self.k = window // 2
        self.latency = self.k
        self._buf: deque[np.ndarray] = deque(maxlen=window)

    def reset(self) -> None:
        self._buf.clear()

    def push(self, x: np.ndarray) -> np.ndarray | None:
        self._buf.append(np.asarray(x, dtype=np.float32))
        if len(self._buf) < self.window:
            return None
        return self._filter_centre()

    def _filter_centre(self) -> np.ndarray:
        block = np.stack(self._buf)  # (window, C)
        centre = block[self.k].copy()
        med = np.median(block, axis=0)
        mad = MAD_TO_SIGMA * np.median(np.abs(block - med), axis=0)
        # Where MAD is zero the channel is constant over the window, so nothing
        # can be an outlier; guard against dividing by it.
        threshold = self.n_sigma * mad
        outlier = (np.abs(centre - med) > threshold) & (mad > 0)
        centre[outlier] = med[outlier]
        return centre.astype(np.float32)

    def flush(self) -> list[np.ndarray]:
        """Emit the tail samples that never got a full centred window.

        Only for offline use at end of stream. Online, the last ``k`` samples are
        simply still in flight.
        """
        out: list[np.ndarray] = []
        while len(self._buf) > self.k:
            block = np.stack(self._buf)
            idx = min(self.k, len(self._buf) - 1)
            centre = block[idx].copy()
            med = np.median(block, axis=0)
            mad = MAD_TO_SIGMA * np.median(np.abs(block - med), axis=0)
            outlier = (np.abs(centre - med) > self.n_sigma * mad) & (mad > 0)
            centre[outlier] = med[outlier]
            out.append(centre.astype(np.float32))
            self._buf.popleft()
        return out

    def state_dict(self) -> dict[str, Any]:
        return {"buf": [b.tolist() for b in self._buf]}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._buf.clear()
        for row in state.get("buf", []):
            self._buf.append(np.asarray(row, dtype=np.float32))

    def fingerprint_fields(self) -> dict[str, Any]:
        return {"window": self.window, "n_sigma": self.n_sigma}


__all__ = ["MAD_TO_SIGMA", "StreamingHampel"]
