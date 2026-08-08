"""Streaming PCA over subcarrier amplitude.

Two jobs at once:

**Dimensionality.** 32 correlated subcarrier channels carry far fewer than 32
independent degrees of freedom; projecting onto 6 components keeps the structure
and shrinks everything downstream.

**Removing the room.** Component 0 of subcarrier amplitude is essentially always
the static multipath -- the walls, the furniture, the fixed reflections. It is
the largest-variance direction and it is the same direction whether or not
anyone is home. Dropping it is the cheapest and most effective defence against a
classifier learning the environment instead of the gesture, which is the failure
mode that kills projects like this.

Implemented as exponentially-weighted covariance plus periodic
eigendecomposition rather than incremental SVD: simpler, exactly reproducible,
and batch-equivalent by construction because both paths run the same update.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from csisense.dsp.stage import DspStage


class StreamingPCA(DspStage):
    """EW-covariance PCA with a periodically refitted basis."""

    latency = 0

    def __init__(
        self,
        n_components: int = 6,
        *,
        n_channels: int,
        fs: float = 100.0,
        tau_s: float = 60.0,
        refit_every: int = 500,
        drop_first: bool = True,
        warmup: int = 200,
    ) -> None:
        if n_components > n_channels:
            raise ValueError(f"cannot take {n_components} components from {n_channels} channels")
        self.n_components = n_components
        self.n_channels = n_channels
        self.fs = fs
        self.tau_s = tau_s
        self.refit_every = refit_every
        self.drop_first = drop_first
        self.warmup = warmup
        self.alpha = 1.0 - math.exp(-(1.0 / fs) / tau_s)

        self._mu = np.zeros(n_channels, dtype=np.float64)
        self._cov = np.eye(n_channels, dtype=np.float64) * 1e-6
        self._basis: np.ndarray | None = None
        self._n = 0

    @property
    def n_out(self) -> int:
        return self.n_components - (1 if self.drop_first else 0)

    def reset(self) -> None:
        self._mu[:] = 0.0
        self._cov = np.eye(self.n_channels, dtype=np.float64) * 1e-6
        self._basis = None
        self._n = 0

    def _refit(self) -> None:
        # eigh returns ascending eigenvalues; we want descending variance.
        _vals, vecs = np.linalg.eigh(self._cov)
        vecs = vecs[:, ::-1][:, : self.n_components]
        # Sign-fix: eigenvectors are only defined up to sign, and an unfixed sign
        # flips arbitrarily between refits. That alone would destroy determinism
        # and with it every golden test.
        for j in range(vecs.shape[1]):
            col = vecs[:, j]
            if col[np.argmax(np.abs(col))] < 0:
                vecs[:, j] = -col
        self._basis = vecs

    def push(self, x: np.ndarray) -> np.ndarray | None:
        v = np.asarray(x, dtype=np.float64)
        self._n += 1
        a = self.alpha
        self._mu = (1.0 - a) * self._mu + a * v
        d = v - self._mu
        self._cov = (1.0 - a) * self._cov + a * np.outer(d, d)

        if self._n == self.warmup or (self._n > self.warmup and self._n % self.refit_every == 0):
            self._refit()

        if self._basis is None:
            return None

        proj = self._basis.T @ d
        if self.drop_first:
            proj = proj[1:]
        return proj.astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {
            "mu": self._mu.tolist(),
            "cov": self._cov.tolist(),
            "basis": None if self._basis is None else self._basis.tolist(),
            "n": self._n,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._mu = np.asarray(state["mu"], dtype=np.float64)
        self._cov = np.asarray(state["cov"], dtype=np.float64)
        basis = state.get("basis")
        self._basis = None if basis is None else np.asarray(basis, dtype=np.float64)
        self._n = int(state.get("n", 0))

    def fingerprint_fields(self) -> dict[str, Any]:
        return {
            "n_components": self.n_components,
            "n_channels": self.n_channels,
            "tau_s": self.tau_s,
            "refit_every": self.refit_every,
            "drop_first": self.drop_first,
            "warmup": self.warmup,
            "fs": self.fs,
        }


__all__ = ["StreamingPCA"]
