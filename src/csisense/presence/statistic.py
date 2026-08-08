"""The motion statistic.

Two EWMAs of the normalised amplitude vector at different time constants: a slow
one that tracks "the room" and a fast one that tracks "now". Their difference is
what changed recently::

    E_t = ||s_t - m_t||^2 / (||m_t||^2 + eps)      # relative dynamic energy

That is the statistic, and it works. Measured across six random static
environments, an empty room peaks at z=4.5 while someone walking through reaches
a median of z=69, against a threshold of 6. The absolute scale is meaningless
(it depends on the room, the geometry, the transmit power), which is why
calibration measures it per link and thresholds are expressed in standard
deviations. See docs/measurements.md.

A coherence term is also computed and available::

    u_t = (s_t - m_t) / ||s_t - m_t||        # direction of deviation
    R_t = <u_t, u_{t-D}>                     # coherence at lag D
    N_t = sqrt(E_t) * max(R_t, 0)

The idea is that thermal noise should push the deviation in an uncorrelated
direction each sample while a body, being a physical object in a fixed place,
perturbs the same subcarriers the same way for as long as it is there.

**Measured, it does not help.** On the synthetic source it costs margin
(walk_through reaches z=8.2 with the gate versus z=11.4 without, measured before
the bias-correction fix below) and is identical on the other scenarios, because
``R_t`` sits near 0.9 on quiet data rather than near zero -- it is tracking the
convergence of the long EWMA, not distinguishing noise from bodies. So it is
**off by default**.

It is kept, and kept configurable, because the case it is meant to address
cannot be reproduced here: synthetic noise is Gaussian and stationary, whereas
the real threat is a neighbour's access point drifting the noise floor over
minutes. Settle it on a real 24-hour idle recording, not on this testbed.

Note what none of this can do: detect a completely motionless person. Both terms
measure *change*. Someone holding still is indistinguishable from an empty room
until they move, which is exactly why occupancy is sticky rather than being
recomputed from the statistic each frame.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

EPS = 1e-12


@dataclass(slots=True)
class StatisticSample:
    """One evaluation of the statistic, with its parts kept for diagnosis.

    When the detector misbehaves you need to know whether the magnitude or the
    coherence term was responsible, so both are exposed rather than only the
    product.
    """

    value: float
    energy: float
    coherence: float


class NacdStatistic:
    """Per-link motion statistic.

    One instance per link: the reference vector ``m_t`` is that link's own view
    of the static room, and two antennas in the same room see different rooms as
    far as multipath is concerned.
    """

    def __init__(
        self,
        *,
        fs: float = 100.0,
        tau_long_s: float = 30.0,
        tau_short_s: float = 0.4,
        lag_frames: int = 20,
        coherence_gate: bool = False,
        # Multiples of tau_long to wait before the reference is trusted.
        # Bias correction makes the average unbiased from the first sample, so
        # this only needs to cover the variance settling, not the mean.
        settle_tau: float = 0.5,
    ) -> None:
        dt = 1.0 / fs
        self.alpha_long = 1.0 - math.exp(-dt / tau_long_s)
        self.alpha_short = 1.0 - math.exp(-dt / tau_short_s)
        self.lag_frames = lag_frames
        self.coherence_gate = coherence_gate
        self.fs = fs
        self.tau_long_s = tau_long_s
        self.tau_short_s = tau_short_s

        self._m: np.ndarray | None = None
        self._s: np.ndarray | None = None
        self._dirs: deque[np.ndarray] = deque(maxlen=lag_frames + 1)
        self._frozen = False
        self._updates = 0
        # Calibration still waits a little: the running mean is unbiased
        # immediately but noisy until it has averaged enough samples, and a
        # baseline measured from a handful of frames is not a baseline.
        self._settle_frames = int(settle_tau * tau_long_s * fs)

    @property
    def reference(self) -> np.ndarray | None:
        """The current estimate of the static room."""
        return self._m

    @property
    def warm(self) -> bool:
        return self._dirs.maxlen is not None and len(self._dirs) > self.lag_frames

    @property
    def settled(self) -> bool:
        """Whether the long-term reference has converged.

        Calibration must wait for this. Measuring the quiet baseline while the
        reference is still converging records the transient as if it were the
        room, and every later z-score comes out far below it.
        """
        return self._updates >= self._settle_frames

    @property
    def settle_progress(self) -> float:
        return min(1.0, self._updates / max(self._settle_frames, 1))

    def freeze(self, frozen: bool = True) -> None:
        """Stop adapting the long-term reference.

        Called whenever presence is not IDLE. Without it, a person who sits
        still is absorbed into ``m_t`` within ``tau_long_s`` and simply vanishes
        from the statistic while still being in the room.
        """
        self._frozen = frozen

    def reset(self) -> None:
        self._m = None
        self._s = None
        self._dirs.clear()
        self._frozen = False
        self._updates = 0

    def update(self, a: np.ndarray) -> StatisticSample:
        v = np.asarray(a, dtype=np.float64)
        self._updates += 1
        if self._m is None or self._s is None:
            self._m = v.copy()
            self._s = v.copy()
            return StatisticSample(0.0, 0.0, 0.0)

        # Bias correction: for the first 1/alpha samples an EWMA seeded from a
        # single value is little more than that value, so it carries the noise
        # of one sample and the statistic decays for a full time constant
        # afterwards. Using max(alpha, 1/n) makes each average the exact running
        # mean until the EWMA horizon is reached, then hands over smoothly --
        # which removes the transient rather than waiting it out.
        a_long = max(self.alpha_long, 1.0 / self._updates)
        a_short = max(self.alpha_short, 1.0 / self._updates)
        if not self._frozen:
            self._m = (1.0 - a_long) * self._m + a_long * v
        self._s = (1.0 - a_short) * self._s + a_short * v

        d = self._s - self._m
        norm_d = float(np.linalg.norm(d))
        norm_m = float(np.linalg.norm(self._m))
        energy = (norm_d * norm_d) / (norm_m * norm_m + EPS)

        direction = d / (norm_d + EPS)
        coherence = 0.0
        if len(self._dirs) > self.lag_frames:
            coherence = float(np.dot(direction, self._dirs[0]))
        self._dirs.append(direction)

        magnitude = math.sqrt(max(energy, 0.0))
        value = magnitude * max(coherence, 0.0) if self.coherence_gate else magnitude
        return StatisticSample(value=value, energy=energy, coherence=coherence)


__all__ = ["NacdStatistic", "StatisticSample"]
