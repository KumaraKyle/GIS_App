# csisense

Wi-Fi CSI human sensing: through-wall presence detection, with gesture
recognition driving Home Assistant automations.

This reads **Channel State Information** — the per-subcarrier complex channel
response a Wi-Fi radio computes for every packet it receives. A person moving
through a space perturbs multipath propagation, and those perturbations are
measurable, including through interior walls. No camera, no wearable, and
everything stays on local hardware.

> **Not a safety device.** Fall detection, where enabled, is a convenience
> trigger. Do not rely on it for medical or emergency purposes.

## What it does not do

Worth stating first, because the expectation is common and the answer is no.

**There is no image of a person, and there cannot be one from this hardware.**
The research that produces body poses from Wi-Fi (CMU's *DensePose From WiFi*,
arXiv:2301.00250) used 3×3 antenna arrays with camera-supervised training, and
still fails on room layouts it was not trained in. Rendering a silhouette here
would mean inventing a shape the data does not contain.

What *is* honest is [radio tomographic imaging](#spatial-sensing): a blob
heatmap of where the field changed. That needs many links — published results
used 28 nodes — so with two links this project **refuses to draw one** and shows
which links are disturbed instead. See [`docs/measurements.md`](docs/measurements.md)
for the numbers behind that refusal.

## Status

Working end to end against synthetic and recorded data:

| | |
|---|---|
| ✅ | CSI wire formats, frame abstraction, deterministic physics-based simulator |
| ✅ | DSP chain with proven streaming/batch equivalence |
| ✅ | Presence detection — calibration, three-state machine, per-link fusion |
| ✅ | Session recording to Parquet, replay, SQLite label index |
| ✅ | MQTT + Home Assistant discovery |
| ✅ | RTI engine (gated — needs ≥4 links) |
| ✅ | Live dashboard |
| 🔜 | ESP-IDF firmware, nexmon Pi setup scripts |
| 🔜 | Recording/labeling UX, gesture training and inference |

**No part of it has run against a real radio yet.** The hardware-facing code is
written against documented formats and tested over loopback; everything most
likely to be wrong is flagged in [`docs/measurements.md`](docs/measurements.md).

## Try it now — no hardware

```bash
uv venv && uv pip install -e ".[dev,edge,web]"

uv run csisense scenarios                     # what the simulator can produce
uv run csisense simulate --scenario walk_through -v
uv run csisense serve --calibrate 120         # dashboard on :8080
uv run pytest
```

`csisense serve` runs the real pipeline against a multipath model that generates
CSI from physics — moving reflectors whose changing path length rotates
subcarrier phase, plus noise, packet loss and AGC glitches. It is the same code
path the radios will feed.

Calibration needs a quiet room, which for the simulator means a scenario without
motion. The dashboard shows calibration progress, including the settling phase
before it starts collecting.

## Hardware

| Device | Role | Resolution |
|---|---|---|
| Raspberry Pi 3B+ (BCM43455c0) | Always-on capture, presence, inference, MQTT | 5GHz, 80MHz → **256 subcarriers** |
| ESP32 ×2 (WROOM-32) | TX beacon + RX satellite | 2.4GHz, HT20 → 64, HT40 → 128 |
| PC with NVIDIA GPU | Training and labeling only, never in the runtime path | — |

Three constraints worth knowing before positioning anything:

- **The Pi needs wired Ethernet.** Its Wi-Fi chip is entirely consumed by
  monitor-mode capture and cannot also be a client.
- **The Pi and the ESP32s cannot share a link.** The ESP32 is 2.4GHz-only; the
  Pi's 256-subcarrier mode needs a 5GHz 80MHz channel. They are two independent
  subsystems, fused at the event level.
- **The Pi cannot generate its own sensing traffic.** In monitor mode it only
  sniffs, and AP beacons are ~10Hz where 100Hz is wanted. The intended fix is
  for the Pi to ping a 5GHz-connected device over its *Ethernet* link, so the AP
  transmits downlink frames the Pi can hear. Untested.

**One link is not a reliable presence sensor.** Measured across random static
environments, one link detects a crossing in 1 of 4 geometries and three links
in 3 of 4 — whether a body produces a large perturbation depends on how its
reflection interferes with the static multipath. More nodes help well before RTI
enters the picture.

## How it works

```
CsiSource (nexmon | esp | replay | synthetic)
   → DspChain → PresenceDetector ─┐
                                  ├→ events → MQTT / dashboard / recorder
   → SlidingWindower → Classifier ┘
```

**Everything above the source abstraction runs on replayed bytes.** Sources
yield `CsiFrame` objects and every stage downstream is a pure function of that
stream, so the whole pipeline runs against synthetic or recorded data with no
radio attached and every test is deterministic.

### Signal processing

amplitude → gain compensation → subcarrier selection → streaming Hampel →
per-packet normalisation → causal bandpass → streaming PCA.

Two design rules do most of the work:

- **Offline and online must be the same code.** Training runs over whole
  recordings, inference runs sample by sample, and if those paths diverge the
  model quietly underperforms in deployment with nothing to show for it. Every
  stateful stage implements `push()` and gets `process_batch()` as a loop over
  it; the two stages with genuinely vectorised paths are equivalence-tested to
  1e-6. A chain fingerprint keys every model bundle and baseline so a mismatch
  fails loudly.
- **Drop PCA component 0.** It is the static multipath — the room. Removing it
  is the cheapest defence against a classifier learning your furniture instead
  of your gestures.

Doppler uses the **CSI ratio** (dividing by a reference subcarrier from the same
packet), which cancels the per-packet CFO/SFO phase and gives Doppler a *sign* —
approach versus retreat, which amplitude alone cannot distinguish.

### Presence

Two EWMAs of the normalised amplitude vector: a slow one tracking "the room" and
a fast one tracking "now". Their relative difference is the statistic. It has no
absolute scale, so calibration measures it per link in an empty room and
thresholds are expressed in standard deviations.

Three states, not two:

```
IDLE ──z > k_on──> MOTION ──quiet──> OCCUPIED_STILL ──timeout──> IDLE
                                              └──z > k_on──> MOTION
any state ── no healthy link for 10s ──> IDLE (reason: link_lost)
```

`OCCUPIED_STILL` exists because a motionless body is physically
indistinguishable from a wall — CSI only measures change. Occupancy therefore
has to be a *memory* with a timeout, not a measurement. That is also the
difference between this and a cheap PIR sensor turning the lights off while you
sit reading.

The last transition matters more than it looks: without it, a dead sensor leaves
Home Assistant holding `occupancy: on` forever and nothing appears broken.
Blindness is published as *unavailable*, never as a confident "clear".

Fusion across links is **max, not mean** — a person near one link barely
perturbs the others, so averaging divides the signal by the link count and a
mean-based detector gets *worse* every time you add a sensor.

### Spatial sensing

Wilson & Patwari radio tomography: `y = Wx + n`, soft elliptical weights,
Tikhonov inversion with an exponential spatial prior. `Π` depends only on
geometry so it is precomputed; per frame the cost is one matrix-vector product.

The engine refuses to produce an image below `rti.min_links` (default 4). A 5×4m
room at 0.25m is 320 voxels; two links give a rank-2 weight matrix, so 318
dimensions of any image would come from the prior and none from the radios. It
would look like a detection, move when you moved, and be almost entirely
fabricated. Below the gate it draws each link's sensitivity ellipse instead —
with one or two links that is the complete truth available.

Adding nodes lifts the gate with no code change. See
[`docs/measurements.md`](docs/measurements.md#rti-what-node-count-buys) for what
each node count actually buys.

### Home Assistant

Entities appear automatically via MQTT discovery, all under one device:

- `binary_sensor` **motion** (fast) and **occupancy** (sticky)
- `sensor` **motion strength**, plus per-link rate/loss/RSSI diagnostics
- one `device_automation` **trigger per gesture class** — gestures are stateless
  events, so modelling them as sensors would re-fire automations on every HA
  restart. Triggers show up as *"When CSI Sense — gesture 'sit'"*.
- `number` sensitivity slider, `button` recalibrate, `switch` gesture enable

Discovery payloads are generated by a pure function pinned to a golden file,
because discovery is what breaks silently on an upgrade — producing entities
that either never appear or never update, both of which look exactly like a
broken sensor.

## Development

```bash
uv run pytest                    # unit + golden
uv run pytest -m "not slow"      # skip the multi-minute acceptance sweeps
uv run pytest -m hardware        # requires attached radios; deselected by default
uv run ruff check . && uv run ruff format --check . && uv run mypy src
uv run python scripts/bench_dsp.py
```

| Path | Contents |
|---|---|
| `src/csisense/wire.py` | Both CSI wire formats and the `CsiFrame` abstraction |
| `src/csisense/sources/` | Radios, replay, and the physics-based simulator |
| `src/csisense/dsp/` | Preprocessing chain |
| `src/csisense/presence/` | Statistic, calibration, state machine |
| `src/csisense/rti/` | Radio tomography and its honesty metrics |
| `src/csisense/mqtt/` | Topics, discovery, publishing |
| `src/csisense/web/` | Dashboard backend and its single static page |
| `src/csisense/recorder/` | Parquet sessions and the label index |
| `firmware/common/csi_wire.h` | C side of the ESP32 format, static-asserted in CI |
| `docs/measurements.md` | Every empirical result, and what is still unverified |

`wire.py` and `firmware/common/csi_wire.h` describe the same bytes. Change one,
change the other; a test catches the drift.

### The tests worth knowing about

- **`test_dsp_equivalence.py`** — offline and online paths agree for every
  stage. The highest-value test here: without it, train/serve skew is invisible.
- **`test_no_false_positives.py`** and **`test_detection.py`** — an empty room
  must fire nothing, and a person must be detected. Neither means anything
  alone: an earlier version of the first passed completely against a detector
  that was simply dead.
- **`test_replay_determinism.py`** — a recording replayed through the pipeline
  produces byte-identical events. Proves the recorder is lossless *and* the
  pipeline deterministic in one assertion.
- **`test_rti_synthetic_geometry.py`** — includes asserting the refusal: two
  links must produce no image.

The dashboard is a single hand-written page, deliberately not a build artefact:
no Node toolchain on the Pi, one file to deploy, nothing to go stale between
source and bundle.

## Roadmap

1. **ESP-IDF firmware** — TX beacon and RX satellite, with the three-layer gain
   control fallback.
2. **Pi setup** — pinned nexmon build, `nexutil` configuration, systemd units.
3. **Recording and labeling UX** — the guided protocol is the scarce resource,
   not the model code.
4. **Gesture training and inference** — leave-one-session-out only, calibrated
   confidence, ONNX int8 on the Pi.

## Privacy

All processing is local; nothing leaves the device. Raw CSI cannot be
reconstructed into imagery, but it does reveal presence and movement patterns
inside a home and should be treated as sensitive.

## License

MIT
