"""Parquet schema for recorded CSI.

CSI is stored as ``fixed_size_binary(4 * n_sub)`` holding the raw interleaved
int16 pairs, not as a list of floats. Three reasons, in order of how much they
matter:

1. **Size.** 100Hz x 256 subcarriers is ~102 kB/s raw and ~35 kB/s after zstd --
   about 3 GB per link per day. A float64 list column would be four times that
   and compress worse, because a fixed-width binary column gives the compressor
   long runs of similarly-scaled bytes.
2. **Read cost.** A fixed-width column is a zero-copy ``np.frombuffer`` on the
   way out. A list column requires per-row offset chasing.
3. **Fidelity.** Storing what the radio actually sent means a decoder bug found
   later can be fixed and the recording reprocessed. Storing derived floats
   would bake the bug in permanently.

The consequence is that ``n_sub`` must be constant within a file, so the writer
opens one file per ``(source_kind, n_sub)``.
"""

from __future__ import annotations

import pyarrow as pa

SCHEMA_VERSION = 1


def arrow_schema(n_sub: int) -> pa.Schema:
    """Column layout for a recording of a given subcarrier width."""
    return pa.schema(
        [
            pa.field("ts_host_ns", pa.int64(), nullable=False),
            pa.field("ts_device_us", pa.uint32(), nullable=False),
            # Dictionary-encoded: a handful of distinct link ids repeated
            # millions of times.
            pa.field("link_id", pa.dictionary(pa.int16(), pa.string()), nullable=False),
            pa.field("sensor_id", pa.uint8(), nullable=False),
            pa.field("seq", pa.uint32(), nullable=False),
            pa.field("beacon_seq", pa.uint32(), nullable=True),
            pa.field("tx_node_id", pa.uint8(), nullable=False),
            pa.field("rssi", pa.int8(), nullable=False),
            pa.field("noise_floor", pa.int8(), nullable=False),
            pa.field("channel", pa.uint8(), nullable=False),
            pa.field("sig_mode", pa.uint8(), nullable=False),
            pa.field("flags", pa.uint8(), nullable=False),
            pa.field("boot_id", pa.uint32(), nullable=False),
            pa.field("rf_core", pa.uint8(), nullable=True),
            pa.field("spatial_stream", pa.uint8(), nullable=True),
            pa.field("agc_gain", pa.uint8(), nullable=True),
            pa.field("fft_gain", pa.uint8(), nullable=True),
            # Interleaved (imag, real) int16 pairs, exactly as decoded.
            pa.field("csi", pa.binary(4 * n_sub), nullable=False),
        ],
        metadata={b"schema_version": str(SCHEMA_VERSION).encode()},
    )


__all__ = ["SCHEMA_VERSION", "arrow_schema"]
