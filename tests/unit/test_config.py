"""Config tests.

The validators here exist to catch settings that would produce a silently
misbehaving detector rather than an error, so they are worth testing directly.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from csisense.config import AppConfig, BandpassConfig, NexmonSourceConfig, PresenceConfig


def test_defaults_load_without_a_file():
    config = AppConfig()
    assert config.source.kind == "synthetic"
    assert config.presence.k_on > config.presence.k_off
    assert config.dsp.pca_drop_first is True


def test_loads_from_yaml(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "deployment:\n  id: bedroom\npresence:\n  k_on: 8.0\n  k_off: 4.0\n"
        "source:\n  kind: nexmon\n"
    )
    config = AppConfig.load(path)
    assert config.deployment.id == "bedroom"
    assert config.presence.k_on == 8.0
    assert config.source.kind == "nexmon"


def test_env_overrides_yaml(tmp_path, monkeypatch):
    """Lets a container be configured without editing files on the Pi."""
    path = tmp_path / "config.yaml"
    path.write_text("presence:\n  k_on: 6.0\n  k_off: 3.0\n")
    monkeypatch.setenv("CSISENSE__PRESENCE__K_ON", "9.5")
    assert AppConfig.load(path).presence.k_on == 9.5


def test_hysteresis_must_latch():
    """k_off >= k_on gives a state machine that chatters on every noisy frame."""
    with pytest.raises(ValidationError, match="chatter"):
        PresenceConfig(k_on=3.0, k_off=5.0)


def test_bandpass_bounds_must_be_ordered():
    with pytest.raises(ValidationError, match="below high"):
        BandpassConfig(low=10.0, high=2.0)


@pytest.mark.parametrize(("bandwidth", "n_sub"), [(20, 64), (40, 128), (80, 256)])
def test_nexmon_subcarrier_count_follows_bandwidth(bandwidth, n_sub):
    assert NexmonSourceConfig(bandwidth_mhz=bandwidth).n_sub == n_sub


def test_unknown_keys_are_rejected():
    """A typo in config.yaml should fail loudly, not be silently ignored."""
    with pytest.raises(ValidationError):
        AppConfig(nonexistent_section={"a": 1})
