"""Subcarrier selection.

Two separate jobs, deliberately kept apart:

1. :func:`valid_subcarrier_mask` -- which bins are *physically meaningful*. Null
   bins, guard bands and pilot tones carry no channel information about the
   room; a pilot is a known reference the transmitter inserts, so its "variation"
   is transmitter behaviour, not a person walking past. This is a pure lookup
   determined by the radio and bandwidth.

2. :class:`SubcarrierSelector` -- which of the valid bins to actually keep,
   ranked by how much they vary during calibration. This is learned per
   deployment and then **frozen**, because a model trained on one ranked subset
   must never be silently fed a different one. The frozen indices go into the
   chain fingerprint for exactly that reason.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from csisense.dsp.stage import DspStage
from csisense.wire import SourceKind

# Pilot subcarrier offsets from DC, per bandwidth. Pilots carry a known
# reference symbol, so they tell you about the transmitter, not the room.
_PILOTS_20 = (-21, -7, 7, 21)
_PILOTS_40 = (-53, -25, -11, 11, 25, 53)
_PILOTS_80 = (-103, -75, -39, -11, 11, 39, 75, 103)


def valid_subcarrier_mask(
    source_kind: SourceKind,
    n_sub: int,
    *,
    sig_mode: int = 1,
    exclude_pilots: bool = True,
) -> np.ndarray:
    """Boolean mask over ``n_sub`` bins marking the physically usable ones.

    Bins are indexed 0..n_sub-1 as the radio delivers them, with DC at
    ``n_sub // 2``.

    Counts this produces: nexmon 80MHz 256 -> 234, nexmon 40MHz 128 -> 108,
    ESP HT20 64 -> 48, ESP HT40 128 -> 108.
    """
    mask = np.zeros(n_sub, dtype=bool)
    dc = n_sub // 2
    offsets = np.arange(n_sub) - dc

    occupied: int
    dc_null: int
    pilots: tuple[int, ...]
    # (outermost occupied offset, half-width of the DC null, pilot offsets).
    # These follow the 802.11 subcarrier allocations: 20MHz nulls DC alone,
    # while 40 and 80MHz null DC and its two neighbours.
    if source_kind is SourceKind.ESP and sig_mode == 0:
        # Legacy (802.11a/g) LLTF: 52 used, 48 data.
        occupied, dc_null, pilots = 26, 0, _PILOTS_20
    elif n_sub == 64:
        # HT20: 56 used, 52 data.
        occupied, dc_null, pilots = 28, 0, _PILOTS_20
    elif n_sub == 128:
        # HT40: 114 used, 108 data.
        occupied, dc_null, pilots = 58, 1, _PILOTS_40
    elif n_sub == 256:
        # VHT80: 242 used, 234 data.
        occupied, dc_null, pilots = 122, 1, _PILOTS_80
    else:
        # Unknown width: keep everything but DC rather than silently discarding
        # data we have no rule for.
        mask[:] = True
        mask[dc] = False
        return mask

    mask[np.abs(offsets) <= occupied] = True
    # DC carries the carrier leak, not channel information.
    mask[np.abs(offsets) <= dc_null] = False
    if exclude_pilots:
        for p in pilots:
            idx = dc + p
            if 0 <= idx < n_sub:
                mask[idx] = False
    return mask


class SubcarrierSelector(DspStage):
    """Narrows a full CSI vector to the ranked subset actually used downstream.

    Before ``freeze()`` it passes the full valid set through and accumulates
    variance statistics. After freezing it emits exactly ``n_keep`` channels in
    a fixed order, and that order is part of the fingerprint.
    """

    latency = 0

    def __init__(
        self,
        source_kind: SourceKind,
        n_sub: int,
        *,
        n_keep: int = 32,
        sig_mode: int = 1,
        exclude_pilots: bool = True,
        mode: str = "auto",
        explicit: list[int] | None = None,
    ) -> None:
        self.source_kind = source_kind
        self.n_sub = n_sub
        self.n_keep = n_keep
        self.mode = mode
        self._valid = np.flatnonzero(
            valid_subcarrier_mask(
                source_kind, n_sub, sig_mode=sig_mode, exclude_pilots=exclude_pilots
            )
        )
        if self._valid.size == 0:
            raise ValueError(f"no valid subcarriers for {source_kind} with n_sub={n_sub}")

        self._indices: np.ndarray | None = None
        if mode == "explicit":
            if not explicit:
                raise ValueError("subcarrier mode 'explicit' requires a non-empty index list")
            self._indices = np.asarray(sorted(explicit), dtype=np.int64)
        elif mode == "all":
            self._indices = self._valid.copy()

        self._n = 0
        self._mean = np.zeros(self._valid.size, dtype=np.float64)
        self._m2 = np.zeros(self._valid.size, dtype=np.float64)

    @property
    def valid_indices(self) -> np.ndarray:
        return self._valid

    @property
    def indices(self) -> np.ndarray:
        """The frozen selection, or the full valid set while still learning."""
        return self._indices if self._indices is not None else self._valid

    @property
    def frozen(self) -> bool:
        return self._indices is not None

    def observe(self, x: np.ndarray) -> None:
        """Accumulate Welford statistics used to rank bins at freeze time."""
        v = x[self._valid].astype(np.float64)
        self._n += 1
        delta = v - self._mean
        self._mean += delta / self._n
        self._m2 += delta * (v - self._mean)

    def freeze(self) -> np.ndarray:
        """Pick and lock the ranked subset.

        Ranks by coefficient of variation rather than raw variance: a bin with a
        large mean will have large absolute variance without carrying more
        information, so raw variance just selects the loudest bins.
        """
        if self.mode in ("explicit", "all"):
            return self.indices
        if self._n < 2:
            self._indices = self._valid[: self.n_keep].copy()
            return self._indices

        var = self._m2 / (self._n - 1)
        cv = np.sqrt(var) / (np.abs(self._mean) + 1e-12)
        order = np.argsort(cv)[::-1][: self.n_keep]
        self._indices = np.sort(self._valid[np.sort(order)])
        return self._indices

    def reset(self) -> None:
        self._n = 0
        self._mean[:] = 0.0
        self._m2[:] = 0.0

    def push(self, x: np.ndarray) -> np.ndarray:
        if self._indices is None:
            self.observe(x)
            return x[self._valid].astype(np.float32)
        return x[self._indices].astype(np.float32)

    def state_dict(self) -> dict[str, Any]:
        return {
            "indices": None if self._indices is None else self._indices.tolist(),
            "n": self._n,
            "mean": self._mean.tolist(),
            "m2": self._m2.tolist(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        idx = state.get("indices")
        self._indices = None if idx is None else np.asarray(idx, dtype=np.int64)
        self._n = int(state.get("n", 0))
        if state.get("mean"):
            self._mean = np.asarray(state["mean"], dtype=np.float64)
            self._m2 = np.asarray(state["m2"], dtype=np.float64)

    def fingerprint_fields(self) -> dict[str, Any]:
        return {
            "source_kind": self.source_kind.value,
            "n_sub": self.n_sub,
            "mode": self.mode,
            "n_keep": self.n_keep,
            # The frozen indices are configuration once chosen: feeding a model
            # a different ranked subset is precisely the skew to catch.
            "indices": self.indices.tolist(),
        }


__all__ = ["SubcarrierSelector", "valid_subcarrier_mask"]
