"""Recording sessions to Parquet, and reading them back.

Sessions, not streams. 100Hz x 256 subcarriers is roughly 3 GB per link per day,
and a Raspberry Pi's SD card will not survive continuous recording -- nor should
it, since the useful unit of training data is a bounded labelled session rather
than an undifferentiated firehose. The writer enforces that with duration, size
and free-space limits rather than trusting anyone to remember.

Every file carries a metadata snapshot of the configuration, geometry, DSP
fingerprint and git revision that produced it. That is what makes a recording
still meaningful in two months: without it a Parquet file is a pile of integers
whose provenance you have to reconstruct from memory.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from csisense.recorder.schema import SCHEMA_VERSION, arrow_schema
from csisense.wire import CsiFrame, SourceKind

log = logging.getLogger(__name__)


class RecorderError(RuntimeError):
    """Recording could not start or continue."""


@dataclass(slots=True)
class SessionMeta:
    """Provenance for a recording. Written into the Parquet file footer."""

    session_id: str
    source_kind: str
    n_sub: int
    rate_hz: float
    started_ns: int
    ended_ns: int = 0
    n_frames: int = 0
    csisense_version: str = ""
    git_sha: str = ""
    dsp_fingerprint: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    geometry: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    @property
    def duration_s(self) -> float:
        if not self.ended_ns or self.ended_ns <= self.started_ns:
            return 0.0
        return (self.ended_ns - self.started_ns) / 1e9


class ParquetSessionWriter:
    """Buffers frames and flushes them as Parquet row groups.

    Row groups are the unit of partial reading: the Label view seeks to a time
    range by skipping groups whose statistics put them outside it, so the group
    size trades write-buffer memory against seek granularity. 6000 frames is 60
    seconds at 100Hz.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        n_sub: int,
        source_kind: SourceKind,
        rate_hz: float = 100.0,
        compression: str = "zstd",
        row_group_frames: int = 6000,
        max_session_s: float = 900.0,
        max_session_bytes: int = 512 * 1024 * 1024,
        min_free_bytes: int = 2 * 1024 * 1024 * 1024,
        meta: SessionMeta | None = None,
    ) -> None:
        self.path = Path(path)
        self.n_sub = n_sub
        self.source_kind = source_kind
        self.rate_hz = rate_hz
        self.compression = compression
        self.row_group_frames = row_group_frames
        self.max_session_s = max_session_s
        self.max_session_bytes = max_session_bytes

        self.path.parent.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(self.path.parent).free
        if free < min_free_bytes:
            raise RecorderError(
                f"only {free / 1e9:.1f} GB free at {self.path.parent}, need "
                f"{min_free_bytes / 1e9:.1f} GB; recording would fill the disk"
            )

        self.schema = arrow_schema(n_sub)
        self.meta = meta or SessionMeta(
            session_id=uuid.uuid4().hex[:12],
            source_kind=source_kind.value,
            n_sub=n_sub,
            rate_hz=rate_hz,
            started_ns=time.time_ns(),
        )
        self._writer: pq.ParquetWriter | None = None
        self._buf: list[dict[str, Any]] = []
        self._closed = False
        self.n_frames = 0
        self.first_ts_ns = 0
        self.last_ts_ns = 0
        self.stopped_reason = ""

    @property
    def full(self) -> bool:
        """Whether a limit has been reached and the session should end."""
        return bool(self.stopped_reason)

    def add(self, frame: CsiFrame) -> bool:
        """Record one frame. Returns False once a limit stops the session."""
        if self._closed:
            raise RecorderError("writer is closed")
        if self.stopped_reason:
            return False
        if frame.n_sub != self.n_sub:
            raise RecorderError(
                f"frame has {frame.n_sub} subcarriers, this session records {self.n_sub}; "
                "open a separate session per (source_kind, n_sub)"
            )

        if not self.first_ts_ns:
            self.first_ts_ns = frame.ts_host_ns
        self.last_ts_ns = frame.ts_host_ns

        self._buf.append(
            {
                "ts_host_ns": frame.ts_host_ns,
                "ts_device_us": frame.ts_device_us & 0xFFFFFFFF,
                "link_id": frame.link_id(),
                "sensor_id": frame.sensor_id,
                "seq": frame.seq & 0xFFFFFFFF,
                "beacon_seq": frame.beacon_seq,
                "tx_node_id": frame.tx_node_id,
                "rssi": frame.rssi,
                "noise_floor": frame.noise_floor,
                "channel": frame.channel,
                "sig_mode": frame.sig_mode,
                "flags": frame.flags,
                "boot_id": frame.boot_id,
                "rf_core": frame.rf_core,
                "spatial_stream": frame.spatial_stream,
                "agc_gain": frame.agc_gain,
                "fft_gain": frame.fft_gain,
                "csi": _pack_csi(frame.csi),
            }
        )
        self.n_frames += 1

        if len(self._buf) >= self.row_group_frames:
            self._flush()
        self._check_limits()
        return not self.stopped_reason

    def _check_limits(self) -> None:
        elapsed = (self.last_ts_ns - self.first_ts_ns) / 1e9
        if elapsed >= self.max_session_s:
            self.stopped_reason = f"reached max_session_s ({self.max_session_s:g}s)"
        elif self.path.exists() and self.path.stat().st_size >= self.max_session_bytes:
            self.stopped_reason = f"reached max_session_bytes ({self.max_session_bytes})"
        if self.stopped_reason:
            log.info("session %s stopping: %s", self.meta.session_id, self.stopped_reason)

    def _flush(self) -> None:
        if not self._buf:
            return
        table = pa.Table.from_pylist(self._buf, schema=self.schema)
        if self._writer is None:
            self._writer = pq.ParquetWriter(
                self.path,
                self.schema,
                compression=self.compression,
                # Needed so a reader can skip row groups by timestamp when
                # serving a time range to the Label view.
                write_statistics=True,
            )
        self._writer.write_table(table)
        self._buf.clear()

    def close(self) -> SessionMeta:
        """Flush, write the provenance footer, and return the session metadata."""
        if self._closed:
            return self.meta
        self._flush()
        self.meta.ended_ns = self.last_ts_ns or time.time_ns()
        self.meta.n_frames = self.n_frames

        if self._writer is not None:
            # Re-open the writer's schema with metadata attached: the footer is
            # the only place provenance survives a file being copied elsewhere.
            self._writer.close()
            self._writer = None
            _attach_metadata(self.path, self.meta)
        self._closed = True
        return self.meta

    def __enter__(self) -> ParquetSessionWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class ParquetSessionReader:
    """Reads a recorded session back as frames.

    Streams row group by row group so a multi-gigabyte session does not need to
    be resident. That matters on the Pi, where the whole point of replay is to
    re-process recordings the machine could not have held in memory.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise RecorderError(f"no recording at {self.path}")
        self._file = pq.ParquetFile(self.path)
        self.meta = _read_metadata(self._file)

    @property
    def n_frames(self) -> int:
        return int(self._file.metadata.num_rows)

    @property
    def n_row_groups(self) -> int:
        return int(self._file.metadata.num_row_groups)

    def frames(self) -> list[CsiFrame]:
        return list(self.iter_frames())

    def iter_frames(self):
        """Yield frames in recorded order."""
        source_kind = SourceKind(self.meta.source_kind)
        for i in range(self._file.metadata.num_row_groups):
            table = self._file.read_row_group(i)
            yield from _rows_to_frames(table, source_kind)


def _pack_csi(csi: np.ndarray) -> bytes:
    """Complex CSI to interleaved (imag, real) int16 bytes."""
    out = np.empty(2 * csi.shape[0], dtype="<i2")
    out[0::2] = np.rint(csi.imag).astype("<i2")
    out[1::2] = np.rint(csi.real).astype("<i2")
    return out.tobytes()


def _unpack_csi(raw: bytes) -> np.ndarray:
    vals = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    csi = np.empty(vals.shape[0] // 2, dtype=np.complex64)
    csi.imag = vals[0::2]
    csi.real = vals[1::2]
    return csi


def _rows_to_frames(table: pa.Table, source_kind: SourceKind):
    cols = {name: table.column(name).to_pylist() for name in table.schema.names}
    for i in range(table.num_rows):
        yield CsiFrame(
            source_kind=source_kind,
            sensor_id=cols["sensor_id"][i],
            csi=_unpack_csi(cols["csi"][i]),
            ts_host_ns=cols["ts_host_ns"][i],
            seq=cols["seq"][i],
            beacon_seq=cols["beacon_seq"][i],
            tx_node_id=cols["tx_node_id"][i],
            rssi=cols["rssi"][i],
            noise_floor=cols["noise_floor"][i],
            ts_device_us=cols["ts_device_us"][i],
            channel=cols["channel"][i],
            sig_mode=cols["sig_mode"][i],
            flags=cols["flags"][i],
            boot_id=cols["boot_id"][i],
            rf_core=cols["rf_core"][i],
            spatial_stream=cols["spatial_stream"][i],
            agc_gain=cols["agc_gain"][i],
            fft_gain=cols["fft_gain"][i],
        )


def _attach_metadata(path: Path, meta: SessionMeta) -> None:
    """Rewrite the file footer with the provenance snapshot.

    Parquet metadata is fixed at writer construction, and the session's frame
    count and end time are only known at close. Rewriting the footer is cheap --
    the row groups are copied verbatim, not re-encoded.
    """
    table = pq.read_table(path)
    payload = {
        "schema_version": str(SCHEMA_VERSION),
        "session": json.dumps(
            {
                "session_id": meta.session_id,
                "source_kind": meta.source_kind,
                "n_sub": meta.n_sub,
                "rate_hz": meta.rate_hz,
                "started_ns": meta.started_ns,
                "ended_ns": meta.ended_ns,
                "n_frames": meta.n_frames,
                "csisense_version": meta.csisense_version,
                "git_sha": meta.git_sha,
                "dsp_fingerprint": meta.dsp_fingerprint,
                "config": meta.config,
                "geometry": meta.geometry,
                "notes": meta.notes,
            },
            sort_keys=True,
        ),
    }
    existing = table.schema.metadata or {}
    merged = {**existing, **{k.encode(): v.encode() for k, v in payload.items()}}
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table.replace_schema_metadata(merged), tmp, compression="zstd")
    tmp.replace(path)


def _read_metadata(file: pq.ParquetFile) -> SessionMeta:
    meta = file.schema_arrow.metadata or {}
    raw = meta.get(b"session")
    if raw is None:
        raise RecorderError(
            "recording has no session metadata; it was written by an older "
            "version or truncated before close()"
        )
    data = json.loads(raw.decode())
    return SessionMeta(**data)


__all__ = [
    "ParquetSessionReader",
    "ParquetSessionWriter",
    "RecorderError",
    "SessionMeta",
]
