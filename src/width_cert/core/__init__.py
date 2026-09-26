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
from .fairness import (
    BudgetRecord,
    budget_record,
    continue_from_fixed_prefix,
    final_continuation_step,
    matched_final_step_rules,
)
from .update_rules import (
    CosineDecayUpdate,
    ExponentialDecayUpdate,
    FixedUpdate,
    GlobalWidthAdaptiveUpdate,
    InverseTimeUpdate,
    IntervalWidthAdaptiveUpdate,
    StepDecayUpdate,
    UpdateDecision,
    UpdateRule,
)

__all__ = [
    "CalibrationResult",
    "CalibrationStatus",
    "BudgetRecord",
    "CertifiedTrainer",
    "CosineDecayUpdate",
    "ExponentialDecayUpdate",
    "FixedUpdate",
    "GlobalWidthAdaptiveUpdate",
    "IWAUResult",
    "InverseTimeUpdate",
    "IntervalState",
    "IntervalWidthAdaptiveUpdate",
    "StepDecayUpdate",
    "UpdateDecision",
    "UpdateRule",
    "calibrate_iwau",
    "budget_record",
    "clone_state",
    "continue_from_fixed_prefix",
    "continue_with_iwau",
    "final_continuation_step",
    "matched_final_step_rules",
]
