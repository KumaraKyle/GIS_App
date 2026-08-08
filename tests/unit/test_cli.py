"""CLI tests.

Cheap to run and they catch the argument-plumbing mistakes that are otherwise
only discovered by a human typing the command.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from csisense.app.cli import build_parser, main


def test_config_flag_works_before_the_subcommand():
    args = build_parser().parse_args(["--config", "a.yaml", "simulate"])
    assert args.config == Path("a.yaml")


def test_config_flag_works_after_the_subcommand():
    """Regression: --config was originally only accepted before the subcommand,
    which is the opposite of where anyone would type it."""
    args = build_parser().parse_args(["simulate", "--config", "b.yaml"])
    assert args.config == Path("b.yaml")


def test_subcommand_flag_does_not_clobber_the_global_one():
    args = build_parser().parse_args(["--config", "a.yaml", "simulate", "--scenario", "empty"])
    assert args.config == Path("a.yaml")
    assert args.scenario == "empty"


def test_unknown_scenario_is_rejected_by_the_parser():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["simulate", "--scenario", "does_not_exist"])


def test_scenarios_command_runs(capsys):
    assert main(["scenarios"]) == 0
    out = capsys.readouterr().out
    assert "walk_through" in out
    assert "motion during" in out


def test_simulate_runs_end_to_end(capsys):
    assert main(["simulate", "--scenario", "empty", "--duration", "1", "--links", "2"]) == 0
    out = capsys.readouterr().out
    assert "200 frames" in out
    assert "synthetic:0" in out
    assert "synthetic:1" in out


def test_simulate_accepts_the_example_config():
    """The shipped example config must actually load."""
    example = Path(__file__).parents[2] / "config" / "config.example.yaml"
    assert main(["simulate", "--config", str(example), "--duration", "1"]) == 0
