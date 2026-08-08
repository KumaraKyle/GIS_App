# csisense

Wi-Fi CSI human sensing: through-wall presence detection and gesture recognition
driving IoT automations via Home Assistant.

This reads **Channel State Information** — the per-subcarrier complex channel
response a Wi-Fi radio computes for every packet it receives. A person moving
through a space perturbs multipath propagation, and those perturbations are
measurable, including through interior walls. No camera, no wearable, and all
processing stays on local hardware.

> **Not a safety device.** Fall detection, where enabled, is a convenience
> trigger and nothing more. Do not rely on it for medical or emergency purposes.

## Status

Early. See [milestones](#milestones) — M0 (skeleton and synthetic pipeline) is
in place; presence detection lands in M2.

## How it is built

Three tiers, staged so each is independently useful even if the later, harder
ones underperform:

| Tier | Method | Realistic accuracy |
|---|---|---|
| Presence / motion | Classical DSP, no ML | 95–99% |
| Coarse body gestures | Small 1D TCN | 85–95% (leave-one-session-out) |
| Fine hand gestures | Doppler + TCN | 55–80%, close range only |

Fine hand gestures are genuinely marginal. They are opt-in, they publish
calibrated confidence, and the system works fully with them disabled.

## Hardware

| Device | Role | Resolution |
|---|---|---|
| Raspberry Pi 3B+ (BCM43455c0) | Always-on capture, presence, inference, MQTT | 5GHz, 80MHz → **256 subcarriers** |
| ESP32 ×2 (WROOM-32) | TX beacon + RX satellite | 2.4GHz, HT20 → 64, HT40 → 128 |
| PC with NVIDIA GPU | Training and labeling only, not in the runtime path | — |

Two constraints worth knowing before you position anything:

- **The Pi needs wired Ethernet.** Its Wi-Fi chip is fully consumed by
  monitor-mode capture and cannot also be a client.
- **The Pi and the ESP32s cannot share a sensing link.** The ESP32 is
  2.4GHz-only; the Pi's 256-subcarrier mode needs a 5GHz 80MHz channel. They are
  two independent subsystems, fused at the event level.

## Design principle

**Everything above the source abstraction runs on replayed bytes.** Sources
yield `CsiFrame` objects; every stage downstream is a pure function of that
stream. The full pipeline therefore runs against synthetic or recorded data with
no hardware attached, and every test is deterministic.

```
CsiSource (nexmon | esp | replay | synthetic)
   → DspChain → PresenceDetector ─┐
                                  ├→ EventBus → MQTT / WebSocket / Recorder
   → SlidingWindower → Classifier ┘
```

## Quick start

No hardware required:

```bash
uv venv && uv pip install -e ".[dev]"
uv run csisense scenarios                          # list synthetic scenarios
uv run csisense simulate --scenario walk_through -v
uv run pytest
```

`csisense simulate` runs the real pipeline against a multipath model that
generates CSI from physics — moving reflectors whose changing path length
rotates subcarrier phase, plus noise, packet loss and AGC glitches. It is the
same code path that will later consume the radios.

## Development

```bash
uv run pytest                  # unit + golden tests
uv run pytest -m hardware      # requires attached boards, deselected by default
uv run ruff check . && uv run mypy src
```

Layout:

| Path | Contents |
|---|---|
| `src/csisense/wire.py` | Both CSI wire formats and the `CsiFrame` abstraction |
| `src/csisense/sources/` | Radios, replay, and the synthetic generator |
| `src/csisense/dsp/` | Preprocessing chain |
| `src/csisense/presence/` | Classical presence and motion detection |
| `src/csisense/ml/` | Training (PC) and inference (Pi, via ONNX) |
| `firmware/` | ESP-IDF projects for the TX beacon and RX satellite |
| `pi/` | nexmon setup and systemd units |

`src/csisense/wire.py` and `firmware/common/csi_wire.h` describe the same bytes.
Change one, change the other; a golden test will catch the drift.

## Milestones

- **M0** — Skeleton, wire formats, synthetic source, pipeline shell. *No hardware.*
- **M1** — Pi capture via nexmon_csi, live dashboard, recording to Parquet.
- **M2** — Presence detection, MQTT, Home Assistant discovery. *Useful on its own.*
- **M3** — ESP32 firmware and the second sensing link.
- **M4** — Recording, labeling and training UX.
- **M5** — Coarse body gestures.
- **M6** — Fine hand gestures (exploratory).

## Privacy

All processing is local; nothing is sent off the device. Raw CSI cannot be
reconstructed into imagery, but it does reveal presence and movement patterns
within a home and should be treated as sensitive.

## License

MIT
