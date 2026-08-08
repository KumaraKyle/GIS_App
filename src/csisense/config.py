"""Typed configuration.

Loaded from YAML, overridable by environment variables using a ``CSISENSE__``
prefix and ``__`` as the nesting separator, e.g.::

    CSISENSE__PRESENCE__K_ON=5.5

Every tunable that you might want to change at 11pm from the couch should live
here rather than being hardcoded, because the alternative is editing source on a
Raspberry Pi over SSH.
"""

from __future__ import annotations

from contextvars import ContextVar
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

# Threaded through to settings_customise_sources, which pydantic calls during
# __init__ and so cannot be passed an argument directly. A ContextVar rather
# than a module global so concurrent loads cannot race.
_yaml_path: ContextVar[Path | None] = ContextVar("_yaml_path", default=None)


class DeploymentConfig(BaseModel):
    id: str = "home"
    name: str = "CSI Sense"


class UdpSourceConfig(BaseModel):
    bind: str = "0.0.0.0"
    port: int = 5566


class NexmonSourceConfig(BaseModel):
    bind: str = "0.0.0.0"
    port: int = 5500
    # 80MHz gives 256 subcarriers, which is the whole reason the Pi is the
    # primary sensor. Dropping to 20/40 is only for debugging.
    bandwidth_mhz: Literal[20, 40, 80] = 80
    # Sense against this transmitter only. Empty means accept any, which is
    # noisier but useful during bring-up when you don't know the MAC yet.
    tx_mac: str = ""
    # Only these RF cores are used. The Pi 3B+ has one usable core for CSI.
    cores: list[int] = Field(default_factory=lambda: [0])
    # Reject frames whose decoded chanspec disagrees. 0 disables the check.
    # Worth setting: if nexutil was configured for a different channel than the
    # software believes, everything still flows and every number is wrong.
    channel: int = 0

    @property
    def n_sub(self) -> int:
        return {20: 64, 40: 128, 80: 256}[self.bandwidth_mhz]


class SerialSourceConfig(BaseModel):
    """esp-csi's stock CSV over USB. Bring-up only.

    Useful for the hour when an ESP32 is on your desk and the UDP path is not
    working yet. Not for recording: the rate is unreliable and the CSV loses
    fields the binary format carries.
    """

    port: str = "/dev/ttyUSB0"
    baud: int = 921600


class ReplaySourceConfig(BaseModel):
    path: Path = Path("data/recordings")
    # 1.0 replays in real time (for demos), 0.0 as fast as possible (for tests).
    speed: float = 1.0
    loop: bool = False


class SyntheticSourceConfig(BaseModel):
    seed: int = 42
    scenario: str = "walk_through"
    n_links: int = 2
    n_sub: int = 64
    rate_hz: float = 100.0
    snr_db: float = 25.0
    loss_pct: float = 0.0
    # Seconds of data to generate; None runs forever.
    duration_s: float | None = None
    # 2.437e9 simulates the ESP32 links, 5.785e9 the Pi's. At 5GHz the same
    # motion produces twice the phase rotation, so thresholds do not transfer.
    carrier_hz: float = 2.437e9


class SourceConfig(BaseModel):
    kind: Literal["udp", "nexmon", "replay", "synthetic", "serial"] = "synthetic"
    udp: UdpSourceConfig = Field(default_factory=UdpSourceConfig)
    nexmon: NexmonSourceConfig = Field(default_factory=NexmonSourceConfig)
    replay: ReplaySourceConfig = Field(default_factory=ReplaySourceConfig)
    synthetic: SyntheticSourceConfig = Field(default_factory=SyntheticSourceConfig)
    serial: SerialSourceConfig = Field(default_factory=SerialSourceConfig)


class NodeConfig(BaseModel):
    """A radio at a known position in the room.

    Positions are metres in a room-local frame, origin wherever you like as long
    as it stays put. RTI resolution is computed from these, so a node whose real
    position differs from what is configured will smear the image rather than
    fail loudly -- measure them properly.
    """

    id: str
    position: tuple[float, float]
    role: Literal["tx", "rx", "both"] = "both"


class LinkConfig(BaseModel):
    """One TX/RX pair, resolved from node ids.

    Link identity lives here and not on the wire, which is what makes "adding a
    node is a config change" literally true.
    """

    link_id: str
    tx: str
    rx: str
    enabled: bool = True


class GeometryConfig(BaseModel):
    """Room dimensions and node placement, for RTI."""

    room_size_m: tuple[float, float] = (5.0, 4.0)
    voxel_m: float = 0.25
    nodes: list[NodeConfig] = Field(default_factory=list)
    links: list[LinkConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_links_reference_known_nodes(self) -> GeometryConfig:
        known = {n.id for n in self.nodes}
        for link in self.links:
            missing = {link.tx, link.rx} - known
            if missing:
                raise ValueError(
                    f"link {link.link_id!r} references unknown node(s) {sorted(missing)}; "
                    f"known nodes: {sorted(known)}"
                )
        return self


class RtiConfig(BaseModel):
    """Radio tomographic imaging.

    Below ``min_links`` the inversion is dominated by the regularisation prior
    rather than by measurements, so any image would be mostly fabricated. The
    engine falls back to drawing per-link ellipses, which is truthful, and
    switches to imaging by itself once enough links exist.
    """

    enabled: bool = True
    # 2 links leaves ~318 of 320 voxel dimensions unconstrained. 4 is the point
    # at which a centroid becomes weakly meaningful.
    min_links: int = 4
    # Width of the elliptical sensitivity region around each link, in metres.
    sigma_e_m: float = 0.25
    # Spatial correlation distance of the smoothness prior.
    delta_c_m: float = 0.5
    sigma_x: float = 0.03
    alpha: float = 0.1
    # Voxels illuminated by fewer links than this are masked as unreliable.
    min_coverage: int = 3
    publish_hz: float = 5.0


class SubcarrierConfig(BaseModel):
    mode: Literal["auto", "all", "explicit"] = "auto"
    # Everything downstream is sized by this. It is the dominant cost in the
    # chain: PCA covariance is O(n_keep^2) per frame, so running at the full
    # 234-bin valid set instead of 32 is ~53x the work.
    n_keep: int = 32
    exclude_pilots: bool = True
    explicit: list[int] = Field(default_factory=list)
    # Frames of observation before the ranked subset is chosen and frozen. The
    # chain emits nothing until then, since the feature width changes at freeze.
    warmup_frames: int = 1000


class BandpassConfig(BaseModel):
    low: float
    high: float
    order: int = 4

    @model_validator(mode="after")
    def _check_band(self) -> BandpassConfig:
        if self.low >= self.high:
            raise ValueError(f"bandpass low ({self.low}) must be below high ({self.high})")
        return self


class DspConfig(BaseModel):
    subcarriers: SubcarrierConfig = Field(default_factory=SubcarrierConfig)
    hampel_window: int = 11
    hampel_n_sigma: float = 3.0
    # Torso motion. Deliberately excludes breathing for now.
    bandpass_slow: BandpassConfig = Field(default_factory=lambda: BandpassConfig(low=0.3, high=3.0))
    # Hand motion.
    bandpass_fast: BandpassConfig = Field(
        default_factory=lambda: BandpassConfig(low=2.0, high=20.0)
    )
    pca_components: int = 6
    # Component 0 is the static path: dropping it removes "the room" and keeps
    # "the motion". This is the cheapest and most effective defence against a
    # model learning the environment instead of the gesture.
    pca_drop_first: bool = True
    gain_compensation: bool = True
    # Time constant of the exponentially-weighted covariance the PCA basis is
    # fitted from, and how often the basis is re-derived.
    tau_pca_s: float = 60.0
    pca_refit_every: int = 500
    # Doppler STFT. noverlap must be below nperseg.
    doppler_nperseg: int = 64
    doppler_noverlap: int = 48
    doppler_max_hz: float = 20.0

    @model_validator(mode="after")
    def _check_doppler(self) -> DspConfig:
        if self.doppler_noverlap >= self.doppler_nperseg:
            raise ValueError(
                f"doppler_noverlap ({self.doppler_noverlap}) must be below "
                f"doppler_nperseg ({self.doppler_nperseg})"
            )
        return self


class PresenceConfig(BaseModel):
    tau_long_s: float = 30.0
    tau_short_s: float = 0.4
    k_on: float = 6.0
    k_off: float = 3.0
    min_frames_on: int = 3
    occupancy_timeout_s: float = 120.0
    calibration_s: float = 120.0
    # Without this, a person who sits still is absorbed into the baseline within
    # tau_long_s and vanishes. Effectively never turn it off.
    freeze_baseline_when_active: bool = True
    # How long the statistic must stay below k_off before MOTION relaxes into
    # OCCUPIED_STILL, so a pause mid-stride does not toggle the state.
    still_debounce_s: float = 2.0
    # If every link goes unhealthy for this long, drop to IDLE. Without it a
    # dead sensor leaves Home Assistant holding occupancy forever and the lights
    # never turn off.
    link_loss_timeout_s: float = 10.0
    # Lag used for the deviation-direction coherence check.
    deviation_lag_frames: int = 20
    # Multiply the statistic by the deviation coherence. Off by default: on the
    # synthetic testbed it costs detection margin rather than gaining noise
    # rejection, because coherence sits near 0.9 on quiet data. The interference
    # it is meant to reject (non-stationary drift from neighbouring APs) cannot
    # be reproduced synthetically, so evaluate it on a real idle recording
    # before turning it on.
    coherence_gate: bool = False
    # A link contributing above this trailing-window loss is excluded from
    # fusion entirely rather than dragging the statistic around.
    max_link_loss_pct: float = 20.0

    @model_validator(mode="after")
    def _check_hysteresis(self) -> PresenceConfig:
        if self.k_off >= self.k_on:
            raise ValueError(
                f"k_off ({self.k_off}) must be below k_on ({self.k_on}) or the state "
                "machine will chatter instead of latching"
            )
        return self


class GestureTierConfig(BaseModel):
    enabled: bool = False
    model: Path | None = None
    window: int = 256
    hop: int = 25
    min_confidence: float = 0.75
    max_energy: float = -4.0
    refractory_s: float = 2.0


class GesturesConfig(BaseModel):
    coarse: GestureTierConfig = Field(
        default_factory=lambda: GestureTierConfig(window=256, hop=25, min_confidence=0.75)
    )
    fine: GestureTierConfig = Field(
        default_factory=lambda: GestureTierConfig(window=128, hop=12, min_confidence=0.85)
    )


class MqttConfig(BaseModel):
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 1883
    username: str = ""
    password: str = ""
    base_topic: str = "csisense"
    discovery_prefix: str = "homeassistant"
    publish_motion_stat_hz: float = 2.0


class WebConfig(BaseModel):
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8080
    live_stream_hz: float = 20.0


class RecorderConfig(BaseModel):
    """Recording limits.

    100Hz x 256 subcarriers is ~102 kB/s raw and ~35 kB/s after zstd -- roughly
    3 GB per link per day. The Pi's SD card will not survive continuous
    recording, so sessions are bounded by construction rather than by hoping.
    """

    dir: Path = Path("data/recordings")
    compression: str = "zstd"
    max_session_s: float = 900.0
    max_session_bytes: int = 512 * 1024 * 1024
    # Refuse to start a session below this much free disk.
    min_free_bytes: int = 2 * 1024 * 1024 * 1024
    row_group_frames: int = 6000
    row_group_bytes: int = 32 * 1024 * 1024


class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: Path | None = None


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CSISENSE__",
        env_nested_delimiter="__",
        extra="forbid",
    )

    deployment: DeploymentConfig = Field(default_factory=DeploymentConfig)
    source: SourceConfig = Field(default_factory=SourceConfig)
    geometry: GeometryConfig = Field(default_factory=GeometryConfig)
    rti: RtiConfig = Field(default_factory=RtiConfig)
    dsp: DspConfig = Field(default_factory=DspConfig)
    presence: PresenceConfig = Field(default_factory=PresenceConfig)
    gestures: GesturesConfig = Field(default_factory=GesturesConfig)
    mqtt: MqttConfig = Field(default_factory=MqttConfig)
    web: WebConfig = Field(default_factory=WebConfig)
    recorder: RecorderConfig = Field(default_factory=RecorderConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Priority, highest first: explicit kwargs, then env vars, then YAML.

        Env above YAML is the point: it lets a deployment override a checked-in
        config without editing files on the Pi over SSH.
        """
        sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings]
        path = _yaml_path.get()
        if path is not None:
            sources.append(YamlConfigSettingsSource(settings_cls, yaml_file=path))
        return tuple(sources)

    @classmethod
    def load(cls, path: Path | str | None = None) -> AppConfig:
        """Load from YAML, falling back to defaults when no file is given.

        Env vars apply on top either way, so a container can be configured with
        no config file at all.
        """
        if path is None:
            return cls()
        token = _yaml_path.set(Path(path))
        try:
            return cls()
        finally:
            _yaml_path.reset(token)


__all__ = [
    "AppConfig",
    "BandpassConfig",
    "DspConfig",
    "GeometryConfig",
    "GestureTierConfig",
    "GesturesConfig",
    "LinkConfig",
    "MqttConfig",
    "NexmonSourceConfig",
    "NodeConfig",
    "PresenceConfig",
    "RecorderConfig",
    "RtiConfig",
    "SerialSourceConfig",
    "SourceConfig",
    "SyntheticSourceConfig",
    "WebConfig",
]
