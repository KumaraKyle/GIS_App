"""Presence and motion detection. Classical signal processing, no ML."""

from csisense.presence.baseline import (
    BaselineMismatchError,
    BaselineStore,
    CalibrationRun,
    LinkBaseline,
)
from csisense.presence.detector import LinkReading, PresenceDetector
from csisense.presence.state import PresenceStateMachine, Transition
from csisense.presence.statistic import NacdStatistic, StatisticSample

__all__ = [
    "BaselineMismatchError",
    "BaselineStore",
    "CalibrationRun",
    "LinkBaseline",
    "LinkReading",
    "NacdStatistic",
    "PresenceDetector",
    "PresenceStateMachine",
    "StatisticSample",
    "Transition",
]
