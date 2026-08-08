"""The DSP stage contract.

This abstraction exists for exactly one reason: **train/serve skew**. Training
runs offline over whole recordings; inference runs online one sample at a time.
If those two paths are separate implementations they will drift, and the drift
is invisible -- the model simply gets quietly worse in deployment than it looked
in evaluation, and nothing errors.

So every stateful stage implements ``push`` (one sample) and gets
``process_batch`` **as a loop over push**. A stage may override
``process_batch`` with a vectorised version for training throughput, but only
alongside an equivalence test asserting the two agree to 1e-6. That test
(``tests/unit/test_dsp_equivalence.py``) is the single highest-value test in the
project.

``latency`` is how many samples a stage delays its output by. Batch and
streaming results are compared after aligning by it.
"""

from __future__ import annotations

import abc
from typing import Any

import numpy as np


class DspStage(abc.ABC):
    """One step of the preprocessing chain.

    Stages operate on real-valued feature vectors of shape ``(n_channels,)``
    unless documented otherwise. The channel count may change across a stage
    (subcarrier selection narrows it, PCA reduces it).
    """

    #: Samples of delay this stage introduces before its first output.
    latency: int = 0

    @property
    def name(self) -> str:
        return type(self).__name__

    @abc.abstractmethod
    def reset(self) -> None:
        """Return to the state of a freshly constructed stage."""

    @abc.abstractmethod
    def push(self, x: np.ndarray) -> np.ndarray | None:
        """Consume one sample; return one output sample, or None while warming up."""

    def process_batch(self, X: np.ndarray) -> np.ndarray:
        """Process ``(T, C)`` offline.

        The default is a loop over :meth:`push`, which is what guarantees the
        offline and online paths cannot diverge. Override only with an
        equivalence test.
        """
        out: list[np.ndarray] = []
        for row in X:
            y = self.push(row)
            if y is not None:
                out.append(y)
        if not out:
            return np.empty((0, 0), dtype=np.float32)
        return np.stack(out).astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        """Serialisable runtime state, so a restart resumes instead of re-warming."""
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        return None

    @abc.abstractmethod
    def fingerprint_fields(self) -> dict[str, Any]:
        """Configuration that affects output. **Never** include runtime state.

        This feeds the chain fingerprint that model bundles are keyed on, so
        including state here would invalidate a bundle on every restart.
        """


class StatelessStage(DspStage):
    """Convenience base for stages with no memory.

    Narrows ``push`` to always return a sample: with no state there is nothing
    to warm up, so a stateless stage can never withhold output.
    """

    latency = 0

    def reset(self) -> None:
        return None

    @abc.abstractmethod
    def push(self, x: np.ndarray) -> np.ndarray: ...

    def process_batch(self, X: np.ndarray) -> np.ndarray:
        # Safe to vectorise trivially: with no state, per-row and whole-array
        # application are identical by construction.
        return np.stack([self.push(row) for row in X]).astype(np.float32)


__all__ = ["DspStage", "StatelessStage"]
