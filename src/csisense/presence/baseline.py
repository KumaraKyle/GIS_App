"""Calibration and the quiet-room baseline.

The motion statistic has no absolute scale -- it depends on the room, the link
geometry, the transmit power and the noise floor. So "is this a person" can only
be answered relative to what this link looks like when nobody is home. That is
what calibration measures, and it is why the thresholds in config are expressed
in standard deviations rather than raw units.

Two decisions here matter more than they look:

**Median and MAD, not mean and standard deviation.** Somebody *will* walk
through during a two-minute calibration, or a door will be open that is usually
shut. A single large sample drags the mean up and inflates the standard
deviation, and the result is a permanently desensitised link that never fires
again. The median and MAD shrug that off.

**The baseline is keyed on the DSP fingerprint.** A baseline measured with one
set of subcarriers, filter settings and normalisation is meaningless against
another. Loading a stale one silently mis-scales every subsequent decision, so
it raises instead.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

MAD_TO_SIGMA = 1.4826
SCHEMA = 1


class BaselineMismatchError(RuntimeError):
    """A stored baseline does not describe the pipeline trying to load it."""


@dataclass(slots=True)
class LinkBaseline:
    """Quiet-room statistics for one link."""

    mu: float
    sigma: float
    n_samples: int
    ts_ns: int = 0

    def z(self, value: float) -> float:
        return (value - self.mu) / self.sigma


@dataclass(slots=True)
class CalibrationRun:
    """Accumulates statistic samples during a calibration window."""

    duration_s: float
    fs: float = 100.0
    samples: dict[str, list[float]] = field(default_factory=dict)
    started_ns: int = 0

    @property
    def target_samples(self) -> int:
        return int(self.duration_s * self.fs)

    def add(self, link_id: str, value: float) -> None:
        self.samples.setdefault(link_id, []).append(value)

    def progress(self) -> float:
        if not self.samples:
            return 0.0
        got = min(len(v) for v in self.samples.values())
        return min(1.0, got / max(self.target_samples, 1))

    def complete(self) -> bool:
        return bool(self.samples) and self.progress() >= 1.0

    def finish(self) -> dict[str, LinkBaseline]:
        out: dict[str, LinkBaseline] = {}
        now = time.time_ns()
        for link_id, values in self.samples.items():
            arr = np.asarray(values, dtype=np.float64)
            mu = float(np.median(arr))
            mad = float(np.median(np.abs(arr - mu)))
            sigma = MAD_TO_SIGMA * mad
            # Two floors. The absolute one stops a pathologically quiet link
            # producing infinite z-scores; the relative one stops a link whose
            # noise happens to be briefly tiny from becoming hair-triggered.
            sigma = max(sigma, 1e-4, 0.05 * abs(mu))
            out[link_id] = LinkBaseline(mu=mu, sigma=sigma, n_samples=len(values), ts_ns=now)
        return out


class BaselineStore:
    """Persists baselines so a restart does not mean recalibrating.

    Two minutes of standing still is a genuinely annoying thing to be asked to
    do twice, and on an always-on Pi a reboot must not silently leave the system
    uncalibrated and unable to detect anything.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def save(
        self,
        baselines: dict[str, LinkBaseline],
        *,
        dsp_fingerprint: str,
        source_kind: str,
        n_sub: int,
        rate_hz: float,
    ) -> None:
        payload = {
            "schema": SCHEMA,
            "dsp_fingerprint": dsp_fingerprint,
            "source_kind": source_kind,
            "n_sub": n_sub,
            "rate_hz": rate_hz,
            "links": {k: asdict(v) for k, v in baselines.items()},
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename: a power cut mid-write must not leave a truncated
        # baseline that fails to parse on the next boot.
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp.replace(self.path)

    def load(
        self,
        *,
        dsp_fingerprint: str,
        source_kind: str,
        n_sub: int,
    ) -> dict[str, LinkBaseline]:
        data: dict[str, Any] = json.loads(self.path.read_text())
        if data.get("schema") != SCHEMA:
            raise BaselineMismatchError(f"baseline schema {data.get('schema')} != {SCHEMA}")
        for field_name, expected in (
            ("dsp_fingerprint", dsp_fingerprint),
            ("source_kind", source_kind),
            ("n_sub", n_sub),
        ):
            if data.get(field_name) != expected:
                raise BaselineMismatchError(
                    f"baseline {field_name} is {data.get(field_name)!r}, pipeline expects "
                    f"{expected!r}; recalibrate rather than trusting a baseline measured "
                    "under different processing"
                )
        return {k: LinkBaseline(**v) for k, v in data.get("links", {}).items()}

    def exists(self) -> bool:
        return self.path.is_file()


__all__ = [
    "MAD_TO_SIGMA",
    "BaselineMismatchError",
    "BaselineStore",
    "CalibrationRun",
    "LinkBaseline",
]
