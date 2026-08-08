"""The shipped example config must stay in step with the code.

A config key that exists but is undocumented is a key nobody will ever set, and
an example containing a key the code has dropped fails at startup with
``extra="forbid"``. Both drift silently otherwise.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from csisense.config import AppConfig

EXAMPLE = Path(__file__).parents[2] / "config" / "config.example.yaml"


def _keys(d: dict, prefix: str = "") -> set[str]:
    out: set[str] = set()
    for key, value in d.items():
        out.add(prefix + key)
        if isinstance(value, dict):
            out |= _keys(value, prefix + key + ".")
    return out


def test_example_loads():
    config = AppConfig.load(EXAMPLE)
    assert config.deployment.id
    assert config.presence.k_on > config.presence.k_off


def test_example_documents_every_setting():
    example = yaml.safe_load(EXAMPLE.read_text())
    missing = sorted(_keys(AppConfig().model_dump()) - _keys(example))
    assert not missing, f"undocumented config keys: {missing}"


def test_example_has_no_settings_the_code_dropped():
    example = yaml.safe_load(EXAMPLE.read_text())
    unknown = sorted(_keys(example) - _keys(AppConfig().model_dump()))
    assert not unknown, f"example references removed keys: {unknown}"
