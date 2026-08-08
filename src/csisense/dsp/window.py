"""Sliding window framing.

Turns a continuous feature stream into the fixed-size tensors a classifier
consumes. Windows overlap: at ``window=256, hop=25`` each 2.56s window shares
90% of its samples with the next, which is what gives ~4 classification
opportunities per second without needing a 4Hz feature rate.

That overlap is also why random train/test splits on CSI are meaningless --
adjacent windows are near-duplicates, so a random split puts near-copies of the
same moment on both sides. See ``ml/split.py``.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from csisense.dsp.stage import DspStage


class SlidingWindower(DspStage):
    """Emits ``(window, C)`` blocks every ``hop`` samples."""

    def __init__(self, window: int = 256, hop: int = 25) -> None:
        if hop < 1 or hop > window:
            raise ValueError(f"hop must be in 1..window, got hop={hop} window={window}")
        self.window = window
        self.hop = hop
        self.latency = window - 1
        self._buf: deque[np.ndarray] = deque(maxlen=window)
        self._n_full = 0

    def reset(self) -> None:
        self._buf.clear()
        self._n_full = 0

    def push(self, x: np.ndarray) -> np.ndarray | None:
        self._buf.append(np.asarray(x, dtype=np.float32))
        if len(self._buf) < self.window:
            return None
        emit = self._n_full % self.hop == 0
        self._n_full += 1
        return np.stack(self._buf) if emit else None

    def process_batch(self, X: np.ndarray) -> np.ndarray:
        """Returns ``(n_windows, window, C)`` -- one rank higher than the input.

        The base-class contract assumes each stage maps samples to samples, so
        this stage is terminal: it is the boundary between the streaming chain
        and the model.
        """
        out = [w for row in X if (w := self.push(row)) is not None]
        if not out:
            return np.empty((0, self.window, X.shape[1]), dtype=np.float32)
        return np.stack(out).astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {"buf": [b.tolist() for b in self._buf], "n_full": self._n_full}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._buf.clear()
        for row in state.get("buf", []):
            self._buf.append(np.asarray(row, dtype=np.float32))
        self._n_full = int(state.get("n_full", 0))

    def fingerprint_fields(self) -> dict[str, Any]:
        return {"window": self.window, "hop": self.hop}


__all__ = ["SlidingWindower"]
