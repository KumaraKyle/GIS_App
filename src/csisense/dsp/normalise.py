"""Amplitude normalisation and gain compensation.

Three defences against the same problem, applied in order of how much you can
trust them.

The problem: the absolute amplitude a radio reports is not a property of the
room. It depends on the receiver's automatic gain control, the transmitter's
power, and the rate the packet was sent at. Free-running AGC swings amplitude by
6-12 dB between consecutive packets, which is far larger than the variation a
person walking through causes. Train a classifier on uncompensated amplitude and
it learns the AGC's habits.

1. :class:`GainCompensator` -- undo the gain the radio *told* us it used. Only
   trustworthy when the firmware locked the gain, which is why the frame carries
   a flag rather than just a number.
2. :class:`RssiGainCompensator` -- when gain is unknown, rescale so the frame's
   total power matches the reported RSSI. RSSI is coarse (1 dB steps, its own
   averaging) but it is an absolute reference.
3. :class:`PerPacketNormaliser` -- divide each frame by its own mean. Always
   applicable, and destroys any absolute-power information, keeping only the
   *shape* across subcarriers. Shape is what carries multipath.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from csisense.dsp.stage import StatelessStage

# The ESP32's AGC gain is reported in undocumented hardware units. 1 dB per LSB
# is a placeholder: it CANNOT be verified without a bench sweep against a known
# attenuator, and getting it wrong scales amplitude by a constant factor per
# gain step. Per-packet normalisation downstream limits the damage.
AGC_STEP_DB = 1.0


class GainCompensator(StatelessStage):
    """Undoes reported AGC and FFT gain. No-op when the radio did not report them."""

    def __init__(self, *, agc_step_db: float = AGC_STEP_DB, enabled: bool = True) -> None:
        self.agc_step_db = agc_step_db
        self.enabled = enabled
        self._agc: int | None = None
        self._fft: int | None = None

    def set_gain(self, agc_gain: int | None, fft_gain: int | None) -> None:
        """Supply the gain reported alongside the frame about to be pushed."""
        self._agc = agc_gain
        self._fft = fft_gain

    def push(self, x: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return np.asarray(x, dtype=np.float32)
        scale = 1.0
        if self._agc is not None:
            scale *= 10.0 ** (-(self._agc * self.agc_step_db) / 20.0)
        if self._fft is not None:
            scale *= 2.0 ** (-self._fft)
        return (np.asarray(x, dtype=np.float32) * scale).astype(np.float32)

    def fingerprint_fields(self) -> dict[str, Any]:
        return {"agc_step_db": self.agc_step_db, "enabled": self.enabled}


class RssiGainCompensator(StatelessStage):
    """Rescales a frame so its mean power matches the reported RSSI.

    The fallback when the firmware could not lock the gain. Coarser than true
    gain compensation, but it re-anchors amplitude to something absolute instead
    of letting the AGC roam.
    """

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self._rssi: int | None = None

    def set_rssi(self, rssi: int | None) -> None:
        self._rssi = rssi

    def push(self, x: np.ndarray) -> np.ndarray:
        v = np.asarray(x, dtype=np.float32)
        if not self.enabled or self._rssi is None:
            return v
        current = float(np.mean(v**2))
        if current <= 0.0:
            return v
        target = 10.0 ** (self._rssi / 10.0)
        return (v * np.sqrt(target / current)).astype(np.float32)

    def fingerprint_fields(self) -> dict[str, Any]:
        return {"enabled": self.enabled}


class PerPacketNormaliser(StatelessStage):
    """Divides each frame by its own mean amplitude.

    Removes packet-level power fluctuation -- residual AGC, TX power changes,
    rate adaptation -- while preserving the across-subcarrier shape that
    actually carries multipath information.

    Note this makes the output scale-invariant, which is also why an AGC step
    cannot masquerade as motion downstream.
    """

    def push(self, x: np.ndarray) -> np.ndarray:
        v = np.asarray(x, dtype=np.float32)
        mean = float(np.mean(np.abs(v)))
        if mean <= 1e-12:
            return v
        return (v / mean).astype(np.float32)

    def fingerprint_fields(self) -> dict[str, Any]:
        return {}


__all__ = [
    "AGC_STEP_DB",
    "GainCompensator",
    "PerPacketNormaliser",
    "RssiGainCompensator",
]
