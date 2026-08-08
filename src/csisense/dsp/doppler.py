"""Doppler features via the CSI ratio.

Why not just take the STFT of amplitude: real-valued amplitude gives you
**unsigned** Doppler. Someone walking towards a sensor and someone walking away
produce the same amplitude spectrum, and "is that person approaching my room?"
is exactly the question we care about.

Signed Doppler needs complex CSI, and raw CSI phase is unusable on commodity
hardware -- carrier frequency offset, sampling frequency offset and packet
detection delay add a large random phase to every packet, swamping the tiny
phase change a moving body causes.

The CSI ratio (Zeng et al., FarSense) fixes this: divide the complex CSI of one
subcarrier by another subcarrier measured *in the same packet*. Both share the
same packet-level phase error, so the division cancels it exactly, leaving a
complex quantity whose rotation direction is physically meaningful.

The cost is that the result is a ratio of two channel responses rather than one
channel response, so the "velocity" it reports is relative between two
frequencies. That is fine for classification -- it is a consistent, sign-bearing
motion feature -- but it is not a calibrated velocity.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from scipy import signal

from csisense.dsp.stage import DspStage


def csi_ratio(H: np.ndarray, ref_idx: int) -> np.ndarray:
    """Complex CSI divided by a reference subcarrier from the same packet.

    Cancels the per-packet random phase, which is what makes the phase of the
    result usable at all.
    """
    ref = H[ref_idx]
    return H / (ref + 1e-12)


def pick_reference_subcarrier(H_window: np.ndarray) -> int:
    """Choose the most stable subcarrier to divide by.

    A reference sitting in a deep fade has a near-zero magnitude, so dividing by
    it amplifies noise enormously. Pick the bin with the highest mean magnitude
    relative to its own variability.
    """
    mag = np.abs(H_window)
    score = mag.mean(axis=0) / (mag.std(axis=0) + 1e-9)
    return int(np.argmax(score))


class StftDoppler(DspStage):
    """Sliding STFT producing a signed Doppler spectrogram column at a time."""

    def __init__(
        self,
        *,
        fs: float = 100.0,
        nperseg: int = 64,
        noverlap: int = 48,
        max_hz: float = 20.0,
    ) -> None:
        if noverlap >= nperseg:
            raise ValueError(f"noverlap ({noverlap}) must be below nperseg ({nperseg})")
        self.fs = fs
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.hop = nperseg - noverlap
        self.max_hz = max_hz
        self.latency = nperseg - 1

        self._win = signal.get_window("hann", nperseg).astype(np.float64)
        freqs = np.fft.fftshift(np.fft.fftfreq(nperseg, d=1.0 / fs))
        self._keep = np.abs(freqs) <= max_hz
        self.freqs = freqs[self._keep]

        self._buf: deque[np.ndarray] = deque(maxlen=nperseg)
        # Samples seen since the buffer first filled. A column is emitted on the
        # first full window and every `hop` samples after that.
        self._n_full = 0

    @property
    def n_bins(self) -> int:
        return int(self._keep.sum())

    def reset(self) -> None:
        self._buf.clear()
        self._n_full = 0

    def push(self, x: np.ndarray) -> np.ndarray | None:
        """Consume one complex sample vector; emit a column every ``hop`` samples."""
        self._buf.append(np.asarray(x, dtype=np.complex128))
        if len(self._buf) < self.nperseg:
            return None
        emit = self._n_full % self.hop == 0
        self._n_full += 1
        return self._column() if emit else None

    def _column(self) -> np.ndarray:
        block = np.stack(self._buf)  # (nperseg, C) complex
        # Remove the DC/static component within the window: what remains is the
        # part that changed, which is the moving scatterer.
        block = block - block.mean(axis=0, keepdims=True)
        spec = np.fft.fftshift(np.fft.fft(block * self._win[:, None], axis=0), axes=0)
        mag = np.abs(spec[self._keep])
        # Average across channels, then log-compress: Doppler energy spans orders
        # of magnitude and a linear scale is dominated by the strongest bin.
        return np.log1p(mag.mean(axis=1)).astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {
            "buf_real": [b.real.tolist() for b in self._buf],
            "buf_imag": [b.imag.tolist() for b in self._buf],
            "n_full": self._n_full,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._buf.clear()
        for re, im in zip(state.get("buf_real", []), state.get("buf_imag", []), strict=True):
            self._buf.append(np.asarray(re, dtype=np.float64) + 1j * np.asarray(im))
        self._n_full = int(state.get("n_full", 0))

    def fingerprint_fields(self) -> dict[str, Any]:
        return {
            "fs": self.fs,
            "nperseg": self.nperseg,
            "noverlap": self.noverlap,
            "max_hz": self.max_hz,
        }


__all__ = ["StftDoppler", "csi_ratio", "pick_reference_subcarrier"]
