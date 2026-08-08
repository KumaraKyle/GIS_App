"""The DSP chain: raw CSI in, features out.

Order matters, and each step earns its place:

1. **amplitude** -- ``|H|``. Phase is discarded here because raw phase on
   commodity radios is dominated by CFO/SFO/PDD noise. (The Doppler path gets
   usable phase a different way, via the CSI ratio.)
2. **gain compensation** -- undo what the receiver's AGC did, so amplitude means
   something across packets.
3. **subcarrier selection** -- drop nulls, guards and pilots; keep the ranked
   informative subset.
4. **Hampel** -- kill impulsive outliers before they ring through the filter.
5. **per-packet normalisation** -- remove residual power fluctuation, keep shape.
6. **bandpass** -- keep the band human motion actually occupies.
7. **PCA** -- decorrelate, reduce, and drop the static component (the room).

The chain is deliberately a plain list of :class:`DspStage` objects rather than a
fused function, so every stage can be individually equivalence-tested and the
fingerprint can be assembled from their declared config.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from csisense.config import DspConfig
from csisense.dsp.filters import CausalButter
from csisense.dsp.hampel import StreamingHampel
from csisense.dsp.normalise import GainCompensator, PerPacketNormaliser, RssiGainCompensator
from csisense.dsp.pca import StreamingPCA
from csisense.dsp.stage import DspStage
from csisense.dsp.subcarriers import SubcarrierSelector
from csisense.wire import CsiFrame, SourceKind

FINGERPRINT_SCHEMA = 1


@dataclass(slots=True)
class DspFeatures:
    """One processed frame.

    ``amplitude`` is kept alongside the reduced ``components`` because presence
    detection works on the normalised amplitude vector directly -- it needs the
    full shape, not a 5-dimensional projection -- while gesture models consume
    the components.
    """

    ts_ns: int
    link_id: str
    amplitude: np.ndarray
    components: np.ndarray | None = None
    gain_locked: bool = False
    gain_estimated: bool = False


@dataclass(slots=True)
class ChainSpec:
    """Everything about a chain that affects its output.

    Separated from the chain itself so a model bundle can record what it was
    trained against without pickling live filter state.
    """

    source_kind: SourceKind
    n_sub: int
    rate_hz: float
    band: str = "slow"
    fields: dict[str, Any] = field(default_factory=dict)


def chain_fingerprint(spec: ChainSpec, stages: list[DspStage]) -> str:
    """Stable hash of the processing configuration.

    Embedded in every model bundle and baseline file. Loading either against a
    chain with a different fingerprint raises, because a model fed features
    produced differently from its training data degrades silently -- which is
    far worse than failing.
    """
    payload = json.dumps(
        {
            "schema": FINGERPRINT_SCHEMA,
            "source_kind": spec.source_kind.value,
            "n_sub": spec.n_sub,
            "rate_hz": round(spec.rate_hz, 3),
            "band": spec.band,
            "stages": [{"name": s.name, "config": s.fingerprint_fields()} for s in stages],
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.blake2b(payload.encode(), digest_size=16).hexdigest()


class DspChain:
    """Per-link preprocessing.

    One instance per link: filter state, PCA basis and subcarrier ranking are all
    link-specific, because two antennas in the same room see genuinely different
    multipath.
    """

    def __init__(
        self,
        config: DspConfig,
        *,
        source_kind: SourceKind,
        n_sub: int,
        rate_hz: float = 100.0,
        band: str = "slow",
        sig_mode: int = 1,
    ) -> None:
        self.config = config
        self.source_kind = source_kind
        self.n_sub = n_sub
        self.rate_hz = rate_hz
        self.band = band

        self.gain = GainCompensator(enabled=config.gain_compensation)
        self.rssi_gain = RssiGainCompensator(enabled=config.gain_compensation)
        self.selector = SubcarrierSelector(
            source_kind,
            n_sub,
            n_keep=config.subcarriers.n_keep,
            sig_mode=sig_mode,
            exclude_pilots=config.subcarriers.exclude_pilots,
            mode=config.subcarriers.mode,
            explicit=config.subcarriers.explicit,
        )
        self.hampel = StreamingHampel(config.hampel_window, config.hampel_n_sigma)
        self.normaliser = PerPacketNormaliser()

        bp = config.bandpass_slow if band == "slow" else config.bandpass_fast
        self.bandpass = CausalButter(bp.low, bp.high, fs=rate_hz, order=bp.order)
        # Sized for the frozen selection, not the full valid set. PCA cost is
        # quadratic in channel count, so this is the difference between ~7% and
        # ~1% of a core per link.
        self.pca = StreamingPCA(
            config.pca_components,
            n_channels=self._target_channels(),
            fs=rate_hz,
            tau_s=config.tau_pca_s,
            refit_every=config.pca_refit_every,
            drop_first=config.pca_drop_first,
        )

        self._seen = 0

    def _target_channels(self) -> int:
        """Channel count after the selector freezes."""
        if self.selector.frozen:
            return len(self.selector.indices)
        return min(self.config.subcarriers.n_keep, len(self.selector.valid_indices))

    @property
    def ready(self) -> bool:
        """Whether the selector has committed to a subset and features can flow."""
        return self.selector.frozen

    @property
    def stages(self) -> list[DspStage]:
        return [
            self.gain,
            self.rssi_gain,
            self.selector,
            self.hampel,
            self.normaliser,
            self.bandpass,
            self.pca,
        ]

    def spec(self) -> ChainSpec:
        return ChainSpec(
            source_kind=self.source_kind,
            n_sub=self.n_sub,
            rate_hz=self.rate_hz,
            band=self.band,
        )

    def fingerprint(self) -> str:
        return chain_fingerprint(self.spec(), self.stages)

    def reset(self) -> None:
        for stage in self.stages:
            stage.reset()

    def process(self, frame: CsiFrame) -> DspFeatures | None:
        """Run one frame through the chain.

        Returns ``None`` while stages are still warming up (the Hampel window
        and the PCA basis both need history before they can emit anything).
        """
        amp = np.abs(frame.csi).astype(np.float32)

        # Trust reported gain only when the firmware says it locked it;
        # otherwise fall back to anchoring on RSSI.
        if frame.gain_locked and frame.agc_gain is not None:
            self.gain.set_gain(frame.agc_gain, frame.fft_gain)
            self.rssi_gain.set_rssi(None)
            estimated = False
        else:
            self.gain.set_gain(None, None)
            self.rssi_gain.set_rssi(frame.rssi)
            estimated = True

        x: np.ndarray = self.gain.push(amp)
        x = self.rssi_gain.push(x)
        x = self.selector.push(x)

        # The selector spends its first `warmup_frames` ranking subcarriers, and
        # emits the full valid set meanwhile. Nothing downstream may run until
        # it commits, because the feature width changes at that moment and a
        # baseline or PCA basis fitted at the wrong width is meaningless.
        self._seen += 1
        if not self.selector.frozen:
            if self._seen >= self.config.subcarriers.warmup_frames:
                self.selector.freeze()
                # Filter and window state were sized for the wide vector; drop
                # them so they re-initialise at the committed width.
                self.bandpass.resize()
                self.hampel.reset()
            return None

        out = self.hampel.push(x)
        # Hampel withholds output until its centred window fills.
        if out is None:
            return None
        normalised = self.normaliser.push(out)

        filtered = self.bandpass.push(normalised)
        components = self.pca.push(filtered)

        return DspFeatures(
            ts_ns=frame.ts_host_ns,
            link_id=frame.link_id(),
            amplitude=normalised,
            components=components,
            gain_locked=frame.gain_locked,
            gain_estimated=estimated,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint(),
            "stages": {s.name: s.state_dict() for s in self.stages},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        expected = self.fingerprint()
        got = state.get("fingerprint")
        if got != expected:
            raise ValueError(
                f"DSP state fingerprint {got} does not match this chain ({expected}); "
                "the processing configuration changed, so the saved state is meaningless"
            )
        for stage in self.stages:
            if stage.name in state.get("stages", {}):
                stage.load_state_dict(state["stages"][stage.name])


__all__ = ["ChainSpec", "DspChain", "DspFeatures", "chain_fingerprint"]
