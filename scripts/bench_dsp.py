#!/usr/bin/env python3
"""Per-stage DSP timing.

The Pi 3B+ must sustain 100Hz on 256 subcarriers while also running capture,
presence, MQTT and a web server in 1GB of RAM. This measures the host-side cost
so the budget is a number rather than a hope.

Read the result as an upper bound on throughput, not a Pi prediction: a Cortex-A53
at 1.4GHz is roughly 5-10x slower per core than a modern x86 core, so divide
accordingly and treat anything above ~10% of a core here as a problem there.

    python scripts/bench_dsp.py --n-sub 256 --frames 3000
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from csisense.config import DspConfig, PresenceConfig
from csisense.dsp.chain import DspChain
from csisense.presence.detector import PresenceDetector
from csisense.wire import CsiFrame, SourceKind


def make_frames(n: int, n_sub: int, kind: SourceKind) -> list[CsiFrame]:
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        csi = (rng.normal(0, 20, n_sub) + 1j * rng.normal(0, 20, n_sub)).astype(np.complex64)
        out.append(
            CsiFrame(
                source_kind=kind,
                sensor_id=0,
                csi=csi,
                ts_host_ns=i * 10_000_000,
                seq=i,
                rssi=-50,
            )
        )
    return out


def bench(n_sub: int, frames: int, rate_hz: float) -> None:
    kind = SourceKind.NEXMON if n_sub >= 256 else SourceKind.ESP
    data = make_frames(frames, n_sub, kind)
    chain = DspChain(DspConfig(), source_kind=kind, n_sub=n_sub, rate_hz=rate_hz)
    detector = PresenceDetector(PresenceConfig(), fs=rate_hz)

    # Warm up so the PCA basis exists and the Hampel window is full.
    for f in data[:300]:
        feats = chain.process(f)
        if feats is not None:
            detector.update(feats)

    t0 = time.perf_counter()
    n_feats = 0
    for f in data[300:]:
        feats = chain.process(f)
        if feats is not None:
            detector.update(feats)
            n_feats += 1
    elapsed = time.perf_counter() - t0

    n = len(data) - 300
    per_frame_us = elapsed / n * 1e6
    budget_us = 1e6 / rate_hz
    print(f"  n_sub={n_sub:<4} channels={len(chain.selector.indices):<4}", end="")
    print(f" {per_frame_us:8.1f} us/frame", end="")
    print(f"  {100 * per_frame_us / budget_us:6.1f}% of one core at {rate_hz:g}Hz", end="")
    print(f"  ({n / elapsed:8.0f} frames/s)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=3000)
    ap.add_argument("--rate", type=float, default=100.0)
    ap.add_argument("--n-sub", type=int, nargs="*", default=[64, 128, 256])
    args = ap.parse_args()

    print(f"DSP + presence, {args.frames} frames per configuration\n")
    for n_sub in args.n_sub:
        bench(n_sub, args.frames, args.rate)
    print("\nDivide throughput by ~5-10x to estimate a Pi 3B+ Cortex-A53 core.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
