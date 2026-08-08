/*
 * csisense ESP32 -> host wire format.
 *
 * This header and src/csisense/wire.py describe the same bytes. Change one,
 * change the other. tests/unit/test_wire.py asserts the sizes below, so drift
 * shows up as a test failure rather than as corrupted recordings.
 *
 * Transport is UDP, deliberately:
 *   - A dropped CSI frame should be *dropped*, not retransmitted late. TCP's
 *     head-of-line blocking would stall capture to redeliver data that is
 *     already useless by the time it arrives.
 *   - Loss is measured (via rx_seq gaps) and reported, never hidden.
 *
 * Records are batched so the packet rate stays reasonable: at 100 Hz with 7
 * records per datagram the ESP32 sends ~14 packets/second instead of 100.
 */

#ifndef CSISENSE_CSI_WIRE_H
#define CSISENSE_CSI_WIRE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* "CSI1" read as a little-endian uint32. */
#define CSI_WIRE_MAGIC 0x31495343u
#define CSI_WIRE_VERSION 1u
#define CSI_WIRE_UDP_PORT 5566u

/* Keep datagrams under the typical 1500-byte MTU with room for IP/UDP headers,
 * so nothing fragments. 1200 is conservative and survives most tunnels. */
#define CSI_WIRE_MAX_DATAGRAM 1200u

/* Flag bits in csi_datagram_header_t.flags. */
#define CSI_FLAG_GAIN_LOCKED (1u << 0)       /* AGC/FFT gain forced; amplitude is comparable */
#define CSI_FLAG_CLOCK_SYNCED (1u << 1)      /* esp_ts_us is believed monotonic */
#define CSI_FLAG_OVERFLOW_SINCE_LAST (1u << 2) /* capture queue overflowed; frames were lost */

/*
 * Datagram header, 16 bytes.
 *
 * boot_id is randomised at every boot so the host can tell a reboot (which
 * restarts rx_seq at zero) from catastrophic packet loss. Without it a reboot
 * looks like a multi-thousand-frame gap and would poison the loss statistics.
 */
typedef struct __attribute__((packed)) {
    uint32_t magic;       /* CSI_WIRE_MAGIC */
    uint8_t version;      /* CSI_WIRE_VERSION */
    uint8_t sensor_id;    /* 1..15, provisioned in NVS */
    uint8_t n_records;    /* records following this header */
    uint8_t flags;        /* CSI_FLAG_* */
    uint32_t tx_dropped;  /* cumulative records dropped on the ESP due to queue overflow */
    uint32_t boot_id;     /* random per boot */
} csi_datagram_header_t;

/*
 * Per-measurement record header, 24 bytes, followed by payload_len bytes of
 * interleaved int8 CSI.
 *
 * beacon_seq is the important field. The transmitter embeds a monotonically
 * increasing counter in every beacon; each receiver copies it into the record
 * for the beacon it heard. That lets the host align several receivers exactly,
 * sample for sample, with no clock synchronisation at all -- which removes an
 * entire category of subtle timestamp bugs.
 *
 * agc_gain and fft_gain are only meaningful when CSI_FLAG_GAIN_LOCKED is set.
 * Free-running AGC swings amplitude by 6-12 dB between packets and will destroy
 * any classifier trained on it, so the host treats unlocked gain as unknown.
 */
typedef struct __attribute__((packed)) {
    uint32_t rx_seq;      /* per-sensor monotonic receive counter */
    uint32_t beacon_seq;  /* transmitter's beacon counter, for cross-receiver alignment */
    uint32_t esp_ts_us;   /* rx_ctrl.timestamp; wraps every ~71 minutes */
    int8_t rssi;
    int8_t noise_floor;
    uint8_t agc_gain;     /* valid only when CSI_FLAG_GAIN_LOCKED */
    uint8_t fft_gain;     /* valid only when CSI_FLAG_GAIN_LOCKED */
    uint8_t channel;
    uint8_t rate;
    uint8_t sig_mode;     /* 0 = legacy, 1 = HT */
    uint8_t n_sub;        /* 64 for HT20, 128 for HT40 */
    uint16_t payload_len; /* always 2 * n_sub */
    uint8_t _pad[2];      /* explicit, so C alignment matches the Python struct format */
} csi_record_header_t;

_Static_assert(sizeof(csi_datagram_header_t) == 16, "datagram header must be 16 bytes");
_Static_assert(sizeof(csi_record_header_t) == 24, "record header must be 24 bytes");

/* CSI payload is interleaved (imag, real) int8 pairs, as ESP-IDF delivers it.
 * Do not reorder in firmware: the host decoder expects the driver's native
 * ordering, and silently swapping the pair conjugates the channel. */

#ifdef __cplusplus
}
#endif

#endif /* CSISENSE_CSI_WIRE_H */
