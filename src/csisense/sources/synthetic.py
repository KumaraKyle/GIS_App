"""Synthetic CSI from a multipath model.

This is what makes CI meaningful. Rather than replaying canned noise, it builds
CSI from the physics that actually produces it, so tests can assert things like
"presence fires when a reflector crosses the link" from first principles, and so
the DSP chain can be exercised against known ground truth.

The model
--------
The channel response at subcarrier *k* is the sum of contributions from every
propagation path::

    H_k(t) = sum_p  a_p(t) * exp(-j * 2*pi * f_k * d_p(t) / c)

A static environment (walls, furniture) contributes paths whose length never
changes. A person contributes paths whose length *does* change as they move, and
because a 2.4GHz wavelength is only ~12.5cm, walking even half a metre rotates
the phase through several full turns. That rotation, seen differently by each
subcarrier, is the entire signal this project is built on.

What this does *not* model: antenna patterns, polarisation, body shadowing
geometry, or realistic reflection coefficients. It is a testbed for the
processing chain, not a propagation simulator. Numbers coming out of it say
nothing about real-world accuracy.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from csisense.sources.base import CsiSource, SourceInfo
from csisense.wire import FLAG_GAIN_LOCKED, CsiFrame, SourceKind

SPEED_OF_LIGHT = 299_792_458.0
CARRIER_HZ_2G4 = 2.437e9  # channel 6
SUBCARRIER_SPACING_HZ = 312_500.0

# Fixed epoch (2024-01-01T00:00:00Z) so generated frames carry reproducible
# timestamps. Using the wall clock here would make output differ between runs
# and quietly defeat the golden regression tests.
DEFAULT_EPOCH_NS = 1_704_067_200_000_000_000


@dataclass(slots=True)
class Reflector:
    """A moving scatterer -- in practice, a person or a limb.

    ``path_length`` returns the total propagation distance (transmitter ->
    reflector -> receiver) in metres at time *t*. Amplitude falls off with that
    distance, which is what makes a reflector fade in and out as it moves rather
    than switching on abruptly.
    """

    path_length: Callable[[float], float]
    strength: float = 0.35

    def contribution(self, t: float, freqs: np.ndarray) -> np.ndarray:
        d = self.path_length(t)
        # Guard against a trajectory passing through zero distance.
        amp = self.strength / max(d, 0.5) ** 1.2
        phase = -2.0 * math.pi * freqs * d / SPEED_OF_LIGHT
        return (amp * np.exp(1j * phase)).astype(np.complex64)


@dataclass(slots=True)
class Scenario:
    """A named situation to generate.

    ``reflectors`` is built per link so that different receivers see genuinely
    different geometry, which is what makes multi-link tests meaningful.
    """

    name: str
    build_reflectors: Callable[[int, np.random.Generator], list[Reflector]]
    duration_s: float = 20.0
    # Injects an abrupt receiver gain change, as uncontrolled AGC would.
    agc_glitch_at_s: tuple[float, ...] = field(default=())
    # Fraction of frames dropped, simulating UDP loss.
    loss_pct: float = 0.0
    # Link index -> (start_s, end_s) during which that link produces nothing.
    dropout: dict[int, tuple[float, float]] = field(default_factory=dict)
    # Ground truth for tests: intervals where a person is actually moving.
    motion_intervals: tuple[tuple[float, float], ...] = field(default=())


def _static_only(_link: int, _rng: np.random.Generator) -> list[Reflector]:
    return []


def _crossing(link: int, _rng: np.random.Generator) -> list[Reflector]:
    """A person walking across the link, closest at t = 10s."""
    offset = 0.6 * link

    def d(t: float) -> float:
        # Hyperbolic: path length is minimised at closest approach, and the
        # rate of change is greatest either side of it.
        return 4.0 + math.hypot(1.2 * (t - 10.0 - offset), 1.0)

    return [Reflector(path_length=d, strength=0.4)]


def _approaching(link: int, _rng: np.random.Generator) -> list[Reflector]:
    """A person walking steadily towards the receiver, arriving at t = 12s."""
    offset = 0.8 * link

    def d(t: float) -> float:
        return max(8.0 - 0.55 * (t - offset), 1.0)

    return [Reflector(path_length=d, strength=0.45)]


def _sitting(link: int, _rng: np.random.Generator) -> list[Reflector]:
    """Someone moves, then sits and stays put from t = 11s.

    The important property for testing is that motion *stops* while the person
    is still present -- this is exactly the case that makes a naive detector
    forget them, and what the occupancy hold and frozen baseline exist for.
    """
    offset = 0.3 * link

    def d(t: float) -> float:
        if t < 8.0:
            return 5.0
        if t < 11.0:
            return 5.0 - 0.5 * (t - 8.0) - offset * 0.1
        return 3.5 - offset * 0.1

    return [Reflector(path_length=d, strength=0.4)]


def _waving(link: int, _rng: np.random.Generator) -> list[Reflector]:
    """A stationary torso plus a hand oscillating at ~2Hz.

    The hand is deliberately much weaker than the torso: that ratio is why fine
    gesture recognition is hard, and any detector that only works when the hand
    reflector is strong is cheating.
    """
    torso_d = 3.0 + 0.2 * link

    def torso(_t: float) -> float:
        return torso_d

    def hand(t: float) -> float:
        if t < 6.0 or t > 14.0:
            return torso_d + 0.5
        return torso_d + 0.5 + 0.18 * math.sin(2.0 * math.pi * 2.0 * t)

    return [
        Reflector(path_length=torso, strength=0.35),
        Reflector(path_length=hand, strength=0.06),
    ]


SCENARIOS: dict[str, Scenario] = {
    "empty": Scenario("empty", _static_only, duration_s=20.0),
    "walk_through": Scenario(
        "walk_through", _crossing, duration_s=20.0, motion_intervals=((6.0, 14.0),)
    ),
    "approach": Scenario(
        "approach", _approaching, duration_s=20.0, motion_intervals=((0.0, 20.0),)
    ),
    "sit": Scenario("sit", _sitting, duration_s=20.0, motion_intervals=((8.0, 11.0),)),
    "wave": Scenario("wave", _waving, duration_s=20.0, motion_intervals=((6.0, 14.0),)),
    "agc_glitch": Scenario(
        "agc_glitch",
        _crossing,
        duration_s=20.0,
        agc_glitch_at_s=(5.0, 12.0),
        motion_intervals=((6.0, 14.0),),
    ),
    "sensor_dropout": Scenario(
        "sensor_dropout",
        _crossing,
        duration_s=20.0,
        dropout={1: (7.0, 12.0)},
        motion_intervals=((6.0, 14.0),),
    ),
    "high_loss": Scenario(
        "high_loss", _crossing, duration_s=20.0, loss_pct=25.0, motion_intervals=((6.0, 14.0),)
    ),
}


class SyntheticCsiSource(CsiSource):
    """Deterministic CSI generator.

    Fully determined by ``seed``: the same seed and scenario always produce
    byte-identical frames, on any machine. Tests depend on this.
    """

    def __init__(
        self,
        *,
        seed: int = 42,
        scenario: str = "walk_through",
        n_links: int = 2,
        n_sub: int = 64,
        rate_hz: float = 100.0,
        snr_db: float = 25.0,
        duration_s: float | None = None,
        loss_pct: float | None = None,
        realtime: bool = False,
        scale: float = 40.0,
        epoch_ns: int = DEFAULT_EPOCH_NS,
    ) -> None:
        if scenario not in SCENARIOS:
            raise ValueError(
                f"unknown scenario {scenario!r}; available: {', '.join(sorted(SCENARIOS))}"
            )
        self._scenario = SCENARIOS[scenario]
        self._rng = np.random.default_rng(seed)
        self._n_links = n_links
        self._n_sub = n_sub
        self._rate_hz = rate_hz
        self._snr_db = snr_db
        self._duration_s = duration_s if duration_s is not None else self._scenario.duration_s
        self._loss_pct = loss_pct if loss_pct is not None else self._scenario.loss_pct
        self._realtime = realtime
        self._scale = scale
        self._epoch_ns = epoch_ns

        # Subcarrier frequencies, centred on the carrier.
        k = np.arange(n_sub) - n_sub // 2
        self._freqs = (CARRIER_HZ_2G4 + k * SUBCARRIER_SPACING_HZ).astype(np.float64)

        # The static environment: a handful of fixed paths per link. This is
        # "the room", and it dominates the received signal -- which is precisely
        # why the DSP chain spends so much effort removing it.
        self._static: list[np.ndarray] = []
        self._reflectors: list[list[Reflector]] = []
        for link in range(n_links):
            h = np.zeros(n_sub, dtype=np.complex64)
            n_paths = int(self._rng.integers(4, 9))
            for p in range(n_paths):
                d = float(self._rng.uniform(2.0, 14.0))
                a = (1.0 if p == 0 else float(self._rng.uniform(0.15, 0.5))) / max(d, 1.0) ** 0.9
                h += (a * np.exp(-2j * math.pi * self._freqs * d / SPEED_OF_LIGHT)).astype(
                    np.complex64
                )
            self._static.append(h)
            self._reflectors.append(self._scenario.build_reflectors(link, self._rng))

    def info(self) -> SourceInfo:
        return SourceInfo(
            kind=SourceKind.SYNTHETIC,
            n_sub=self._n_sub,
            rate_hz=self._rate_hz,
            link_ids=tuple(f"synthetic:{i}" for i in range(self._n_links)),
        )

    @property
    def scenario(self) -> Scenario:
        """Exposed so tests can assert against the scenario's ground truth."""
        return self._scenario

    def channel_at(self, link: int, t: float) -> np.ndarray:
        """Noise-free channel response, for tests that need ground truth."""
        h = self._static[link].copy()
        for r in self._reflectors[link]:
            h += r.contribution(t, self._freqs)
        return h

    def _gain_at(self, t: float) -> float:
        """Receiver gain multiplier, stepping abruptly at each glitch time.

        Real AGC does exactly this, and it is why amplitude features are
        untrustworthy until the gain is either locked or compensated.
        """
        g = 1.0
        for i, glitch_t in enumerate(self._scenario.agc_glitch_at_s):
            if t >= glitch_t:
                g = 2.0 if i % 2 == 0 else 0.5
        return g

    def _is_dropped_out(self, link: int, t: float) -> bool:
        window = self._scenario.dropout.get(link)
        return window is not None and window[0] <= t < window[1]

    async def frames(self) -> AsyncIterator[CsiFrame]:
        dt = 1.0 / self._rate_hz
        n_steps = int(self._duration_s * self._rate_hz)
        noise_sigma = 10.0 ** (-self._snr_db / 20.0)
        t0_ns = self._epoch_ns
        wall_start = time.perf_counter()

        for step in range(n_steps):
            t = step * dt
            gain = self._gain_at(t)

            for link in range(self._n_links):
                if self._is_dropped_out(link, t):
                    continue
                if self._loss_pct > 0.0 and self._rng.random() * 100.0 < self._loss_pct:
                    continue

                h = self.channel_at(link, t)
                power = float(np.mean(np.abs(h) ** 2)) or 1.0
                sigma = noise_sigma * math.sqrt(power)
                noise = (
                    self._rng.normal(0.0, sigma, self._n_sub)
                    + 1j * self._rng.normal(0.0, sigma, self._n_sub)
                ).astype(np.complex64)

                # Scale into the int8-ish range a real radio would report, so
                # quantisation behaviour resembles hardware.
                csi = ((h + noise) * gain * self._scale / max(abs(h).max(), 1e-9)).astype(
                    np.complex64
                )

                yield CsiFrame(
                    source_kind=SourceKind.SYNTHETIC,
                    sensor_id=link,
                    csi=csi,
                    ts_host_ns=t0_ns + int(t * 1e9),
                    seq=step,
                    beacon_seq=step,
                    rssi=-50 - link,
                    noise_floor=-92,
                    ts_device_us=int(t * 1e6),
                    channel=6,
                    flags=FLAG_GAIN_LOCKED if not self._scenario.agc_glitch_at_s else 0,
                )

            if self._realtime:
                target = wall_start + (step + 1) * dt
                delay = target - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
            elif step % 100 == 0:
                # Yield to the loop periodically so a fast replay cannot starve
                # the web server or MQTT client sharing this event loop.
                await asyncio.sleep(0)


def available_scenarios() -> Sequence[str]:
    return sorted(SCENARIOS)


__all__ = ["SCENARIOS", "Reflector", "Scenario", "SyntheticCsiSource", "available_scenarios"]
