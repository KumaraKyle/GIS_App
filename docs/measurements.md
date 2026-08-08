# Measurements

Every empirical number this project relies on, where it came from, and how much
to trust it. Reproduce with the commands given; regenerate this file if any of
them stops matching.

Everything here was measured against the **synthetic source**, which models
multipath physics (moving reflectors whose changing path length rotates
subcarrier phase) but not bodies, antenna patterns, polarisation, or real
interference. Treat these as testing the *processing*, not predicting real-world
accuracy. The [unverified](#what-hardware-will-decide) section lists what only
hardware can settle.

---

## Presence detection

### Separation between quiet and occupied

Six random static environments, two links, 64 subcarriers, calibrated on an
empty room then run against each scenario.

| Scenario | quiet-room max z | occupied peak z |
|---|---|---|
| empty | **4.49** (worst of 6 seeds) | — |
| walk_through | — | median **69**, min 7.1 |
| approach | — | > 6 |
| sit | — | > 6 |
| wave (2Hz hand, torso static) | — | **38** |
| still_person (motionless) | — | **< 6, not detected** |

Threshold `k_on` is 6.0, so the margin is roughly 1.3× on the quiet side and
10× on the moving side.

Two of these matter more than the rest:

- **The hand wave at z=38** is the first evidence the gesture tier is plausible.
  The hand reflector is six times weaker than the torso and still shows clearly.
- **The motionless person is not detected, and that is correct.** A static body
  is a static reflector, physically indistinguishable from a wall. Nothing
  separates them, because CSI only measures change. This is exactly why
  occupancy is a sticky state with a timeout rather than a per-frame
  measurement.

```bash
uv run pytest tests/golden/test_detection.py tests/golden/test_no_false_positives.py
```

### Detection depends heavily on geometry

The single most actionable result. Four random static environments,
walk_through, counting how many produced a detection:

| Links | Detected |
|---|---|
| 1 | 1 of 4 |
| 2 | 2–3 of 4 |
| 3 | 3 of 4 |

Whether a body produces a large perturbation depends on how its reflection
interferes with the static multipath, which differs per room and per placement.
This is physics, not tuning.

**One link is not a reliable presence sensor.** More nodes help well before RTI
becomes relevant. The false-positive floor, by contrast, is stable across every
geometry — that half of the problem is solved.

```bash
uv run pytest "tests/golden/test_detection.py::test_detection_rate_across_geometries"
```

### The coherence gate does not work

The statistic optionally multiplies energy by the coherence of the deviation
direction, on the theory that noise decorrelates and bodies do not.

| | walk_through peak z |
|---|---|
| energy only (default) | **11.4** |
| × coherence | 8.2 |

It costs margin and gains nothing, because coherence sits near **0.9 on quiet
data** — it tracks the slow convergence of the long EWMA rather than
distinguishing noise from bodies. `presence.coherence_gate` therefore defaults
to `false`.

It is kept and kept configurable because the case it targets cannot be
reproduced here: synthetic noise is Gaussian and stationary, whereas the real
threat is a neighbour's AP drifting the noise floor over minutes. **Settle it on
a real 24-hour idle recording.** A unit test pins the negative result so the
idea is not silently re-derived.

### Calibration must wait for the reference to settle

The long EWMA is seeded from a single sample, so for roughly a time constant it
carries that one sample's noise and the statistic decays steadily. Calibrating
during that window records the transient as if it were the quiet room.

Observed live before the fix: the detector settled at **z = −14** and could
never fire. The golden tests missed it entirely because they calibrate for 40s,
long enough to average over the decay — it only appeared when the dashboard ran
with a short calibration.

Fixed with EWMA bias correction (`max(alpha, 1/n)`), which makes each average
the exact running mean until the EWMA horizon and then hands over smoothly. This
removes the transient rather than waiting it out, and is where the large
sensitivity improvement above came from.

Calibration additionally refuses to collect until the reference has settled, and
reports that as the first half of its progress rather than sitting at 0%.

---

## RTI: what node count buys

5×4m room at 0.25m resolution (320 voxels), nodes evenly spaced around the
perimeter and fully meshed, 5% measurement noise.

| Nodes | Links | Rank | Effective resolution | Room below 3-link coverage | Median error |
|---|---|---|---|---|---|
| 2 | 1 | — | — | — | **gated, no image** |
| 3 | 3 | — | — | — | **gated, no image** |
| 4 | 6 | 6 | 1.59 m | 62% | 0.30 m |
| 5 | 10 | 10 | 1.44 m | 31% | 0.26 m |
| 6 | 15 | 15 | 1.33 m | 20% | 0.15 m |
| 8 | 28 | 28 | 1.18 m | 9% | 0.15 m |
| 10 | 45 | 45 | 1.10 m | 8% | 0.15 m |

**The error column is a floor, not a prediction.** These simulate measurements
with the same weight matrix that is then inverted — "inverse crime" — so they
measure the solver, not the physics. A real body is not a point voxel, does not
attenuate uniformly, and the elliptical model is an approximation. Published RTI
results at comparable node counts report **0.5–2.85 m**, and the classic
Wilson & Patwari figures used **28 nodes**.

**The column worth trusting is coverage.** At four nodes, 62% of the room is lit
by fewer than three links and cannot be localised at all regardless of solver
quality. That fraction, not the error estimate, is what the dashboard displays.

### Why two links produce nothing

320 voxels from 2 measurements leaves 318 dimensions determined entirely by the
smoothness prior. An image from that would look like a detection, move when
someone moved, and be almost entirely fabricated — and the uncertainty would not
be visible in the output. Below `rti.min_links` the engine draws each link's
sensitivity ellipse instead, which with one or two links is the complete truth
available: something disturbed this link, so it is somewhere on this curve.

Node count to link count is `N(N-1)/2`: **4 nodes = 6 links, 6 = 15, 8 = 28.**

One genuine advantage over classical RTI: it used scalar RSS per link, badly
corrupted by fading nulls. Subcarrier-diverse CSI gives a far better-conditioned
per-link estimate — worth roughly a factor of two in node count for equal image
quality.

```bash
uv run pytest tests/golden/test_rti_synthetic_geometry.py
```

---

## Compute budget

Full DSP chain plus presence detection, per frame per link:

| Subcarriers | Channels after selection | Per frame | One core at 100Hz |
|---|---|---|---|
| 64 | 32 | 310 µs | 3.1% |
| 128 | 32 | 231 µs | 2.3% |
| 256 | 32 | 239 µs | 2.4% |

Cost is **independent of subcarrier count**, because everything downstream of
the selector sees `n_keep=32` channels regardless of the radio.

Measured on x86. A Cortex-A53 at 1.4GHz is roughly 5–10× slower per core, so
expect **13–27% of one Pi 3B+ core per link** — workable, with headroom for
capture, MQTT and the web server, but not measured on the Pi itself.

This got there via a bug: the subcarrier selector never froze, so PCA ran a
234×234 covariance per frame instead of 32×32 — 53× the intended work, at 750 µs
and 7.5% of a core.

```bash
uv run python scripts/bench_dsp.py
```

---

## Dashboard bandwidth

Two links at 20Hz:

| | Per frame | Sustained |
|---|---|---|
| Initial (full state) | 84.5 kB | once per connection |
| Incremental | **1.94 kB** | **39 kB/s** |

The first version resent the entire waterfall, all 300 trace samples and the
full event tail every tick — 1.7 MB/s for one new column of data. Waterfall
columns, trace samples, events and the static ellipse geometry are all
incremental now, with the browser keeping the scrollback.

---

## Wire formats

Verified against upstream documentation, **not against a real packet**.

**nexmon_csi** — chip `bcm43455c0`, firmware `7_45_189`, Pi 3B+ supported. CSI
arrives as UDP broadcast on port **5500** from `10.10.10.10`. Frame: magic
`0x1111`, 6-byte source MAC, 2-byte sequence, core/stream bits, chanspec, chip
version, then 256 four-byte values at 80MHz. Configured with
`makecsiparams -c 157/80 -C 1 -N 1 -m <mac>` → `nexutil -Iwlan0 -s500 -b -l34 -v<base64>`.

**esp-csi** — `wifi_csi_config_t { lltf_en, htltf_en, stbc_htltf2_en,
ltf_merge_en, channel_filter_en, manu_scale, shift }` on the original ESP32.
(The `acquire_csi_*` field names in some examples are C5/C6-only.) AGC gain is
not reliably exposed on this generation; esp-csi forces it via the undocumented
ROM symbols `phy_fft_scale_force()` / `phy_force_rx_gain()`.

### Subcarrier counts

Corrected against the actual 802.11 allocations while testing — 20MHz nulls DC
alone, while 40 and 80MHz null DC and both neighbours.

| Mode | Bins | Data subcarriers |
|---|---|---|
| Legacy 802.11a/g | 64 | 48 |
| HT20 | 64 | **52** |
| HT40 | 128 | 108 |
| VHT80 | 256 | 234 |

The plan originally quoted 48 for HT20; that is the legacy figure.

### Sequence numbers wrap

Nexmon's `seq` is the 12-bit 802.11 sequence number, so **at 100Hz it wraps
every ~41 seconds**. Treating it as unbounded books every wrap as reordering and
hides the real loss around it. Sequence space is now per source: 4096 for
nexmon, 2³² for the ESP32.

---

## What hardware will decide

Ranked by how much rests on it. Everything here is marked in-code as
not-yet-validated.

1. **The nexmon 80MHz byte layout.** The core/stream bit packing and the chip-id
   gate are build-specific and have never seen a real packet. *Mitigation: the
   first hardware session should record raw datagrams verbatim before decoding,
   so a layout fix can be applied retroactively rather than needing a recapture.*
2. **Whether the gain-forcing symbols link and work** on IDF 5.3 / ESP32 rev 3.
   Absent, present-but-no-op and present-and-crashing are all possible; the
   three-layer fallback plus an NVS crash latch covers all three.
3. **Every threshold.** `k_on=6`, `tau_long=30s`, the AGC dB-per-LSB step — all
   placeholders until calibrated in a real room.
4. **Achievable packet rate** on both radios.
5. **Pi 3B+ budget** — capture, DSP, presence, inference, FastAPI, in 1GB.
6. **5GHz traffic generation.** The Pi cannot transmit in monitor mode and AP
   beacons are ~10Hz. The intended fix — the Pi pinging a 5GHz client over
   Ethernet so the AP transmits downlink frames it can sniff — is untested.
7. **Clock drift magnitude**, so the 2ms RANSAC inlier threshold is a guess.
8. **All gesture accuracy.** The simulator models point reflectors, not bodies.
   Any LOSO number from synthetic data is meaningless, and the code refuses to
   write a model bundle trained on synthetic data without an explicit override.
9. **RTI physics.** The synthetic test validates the solver, not whether a body
   attenuates a link the way the ellipse model says.
10. **The ≤1 false positive per 24h gesture target**, which needs a real 24-hour
    idle recording. Note that this is reached mostly by *only classifying when
    presence reports MOTION* — no confidence threshold alone gets to the
    required 2.9e-6 per window.
11. **ESP32 HT40.** esp-csi's HT40 support is flaky and poorly documented; HT20
    is the supported path and HT40 experimental.

The one item that is verifiable now and worth doing early: **MQTT and Home
Assistant discovery against Mosquitto + HA in Docker.** No radios needed, and it
removes a whole class of late surprises.
