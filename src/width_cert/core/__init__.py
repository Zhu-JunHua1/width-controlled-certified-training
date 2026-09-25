"""Stable public core API."""

from .calibration import (
    CalibrationResult,
    CalibrationStatus,
    IWAUResult,
    calibrate_iwau,
    continue_with_iwau,
)
from .state import IntervalState, clone_state
from .trainer import CertifiedTrainer
from .update_rules import (
    ExponentialDecayUpdate,
    FixedUpdate,
    IntervalWidthAdaptiveUpdate,
    UpdateDecision,
    UpdateRule,
)

__all__ = [
    "CalibrationResult",
    "CalibrationStatus",
    "CertifiedTrainer",
    "ExponentialDecayUpdate",
    "FixedUpdate",
    "IWAUResult",
    "IntervalState",
    "IntervalWidthAdaptiveUpdate",
    "UpdateDecision",
    "UpdateRule",
    "calibrate_iwau",
    "clone_state",
    "continue_with_iwau",
]

