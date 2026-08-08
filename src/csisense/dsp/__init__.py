"""Signal processing: raw CSI to features that mean something."""

from csisense.dsp.chain import ChainSpec, DspChain, DspFeatures, chain_fingerprint
from csisense.dsp.doppler import StftDoppler, csi_ratio, pick_reference_subcarrier
from csisense.dsp.filters import CausalButter
from csisense.dsp.hampel import StreamingHampel
from csisense.dsp.normalise import (
    GainCompensator,
    PerPacketNormaliser,
    RssiGainCompensator,
)
from csisense.dsp.pca import StreamingPCA
from csisense.dsp.stage import DspStage, StatelessStage
from csisense.dsp.subcarriers import SubcarrierSelector, valid_subcarrier_mask
from csisense.dsp.window import SlidingWindower

__all__ = [
    "CausalButter",
    "ChainSpec",
    "DspChain",
    "DspFeatures",
    "DspStage",
    "GainCompensator",
    "PerPacketNormaliser",
    "RssiGainCompensator",
    "SlidingWindower",
    "StatelessStage",
    "StftDoppler",
    "StreamingHampel",
    "StreamingPCA",
    "SubcarrierSelector",
    "chain_fingerprint",
    "csi_ratio",
    "pick_reference_subcarrier",
    "valid_subcarrier_mask",
]
