"""Typed run records with unambiguous state/update epoch semantics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


SCHEMA_VERSION = "1.2"


class RunStatus(str, Enum):
    COMPLETED = "completed"
    DIVERGED = "diverged"
    NUMERICAL_ERROR = "numerical_error"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True)
class StateRecord:
    """Metrics for one interval state ``B_t`` after exactly ``t`` updates."""

    epoch: int
    certified_accuracy: float
    weight_width: float
    clean_accuracy: float | None = None


@dataclass(frozen=True)
class UpdateRecord:
    """Metrics for the transition ``B_t -> B_{t+1}``."""

    epoch: int
    next_epoch: int
    gradient_width: float
    effective_step_mean: float
    iwau_active: bool


@dataclass
class TrainingRun:
    """In-memory result of one fixed, decayed, or IWAU training arm."""

    states: list[StateRecord]
    updates: list[UpdateRecord]
    peak_certified_accuracy: float
    cert_at_horizon: float | None
    status: RunStatus
    status_reason: str
    final_state: dict[str, Any]
    snapshots: dict[int, dict[str, Any]] = field(default_factory=dict, repr=False)

    @property
    def peak_epoch(self) -> int:
        peak = self.peak_certified_accuracy
        return next(s.epoch for s in self.states if s.certified_accuracy == peak)

    @property
    def reached_horizon(self) -> bool:
        return self.status is RunStatus.COMPLETED and self.cert_at_horizon is not None

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "status": self.status.value,
            "status_reason": self.status_reason,
            "peak_certified_accuracy": self.peak_certified_accuracy,
            "peak_epoch": self.peak_epoch,
            "cert_at_horizon": self.cert_at_horizon,
            "num_states": len(self.states),
            "num_updates": len(self.updates),
        }

    def state_rows(self) -> list[dict[str, Any]]:
        return [asdict(record) for record in self.states]

    def update_rows(self) -> list[dict[str, Any]]:
        return [asdict(record) for record in self.updates]
