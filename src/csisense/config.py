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

    @property
    def n_sub(self) -> int:
        return {20: 64, 40: 128, 80: 256}[self.bandwidth_mhz]


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


class SourceConfig(BaseModel):
    kind: Literal["udp", "nexmon", "replay", "synthetic", "serial"] = "synthetic"
    udp: UdpSourceConfig = Field(default_factory=UdpSourceConfig)
    nexmon: NexmonSourceConfig = Field(default_factory=NexmonSourceConfig)
    replay: ReplaySourceConfig = Field(default_factory=ReplaySourceConfig)
    synthetic: SyntheticSourceConfig = Field(default_factory=SyntheticSourceConfig)


class SubcarrierConfig(BaseModel):
    mode: Literal["auto", "all", "explicit"] = "auto"
    n_keep: int = 32
    exclude_pilots: bool = True
    explicit: list[int] = Field(default_factory=list)


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
    dir: Path = Path("data/recordings")
    compression: str = "zstd"


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
    "GestureTierConfig",
    "GesturesConfig",
    "MqttConfig",
    "NexmonSourceConfig",
    "PresenceConfig",
    "RecorderConfig",
    "SourceConfig",
    "SyntheticSourceConfig",
    "WebConfig",
]
