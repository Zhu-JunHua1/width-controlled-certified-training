"""Update policies for interval certified training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class UpdateDecision:
    """Arguments passed to the architecture-specific interval update."""

    lower_gradient: Any
    upper_gradient: Any
    scalar_step: float
    effective_step_mean: float
    iwau_active: bool


class UpdateRule(Protocol):
    def decide(self, g_lower: Any, g_upper: Any, width: Any, epoch: int) -> UpdateDecision:
        """Return the sound update representation for ``B_epoch``."""


@dataclass(frozen=True)
class FixedUpdate:
    base_step: float

    def decide(self, g_lower: Any, g_upper: Any, width: Any, epoch: int) -> UpdateDecision:
        return UpdateDecision(g_lower, g_upper, self.base_step, self.base_step, False)


@dataclass(frozen=True)
class ExponentialDecayUpdate:
    base_step: float
    decay: float = 0.6
    minimum_step: float = 1e-3

    def decide(self, g_lower: Any, g_upper: Any, width: Any, epoch: int) -> UpdateDecision:
        step = max(self.base_step * (self.decay**epoch), self.minimum_step)
        return UpdateDecision(g_lower, g_upper, step, step, False)


@dataclass(frozen=True)
class IntervalWidthAdaptiveUpdate:
    """Coordinate-wise IWAU rule ``eta_i = eta_base/(1+lambda*Omega_i)``."""

    base_step: float
    coefficient: float

    def decide(self, g_lower: Any, g_upper: Any, width: Any, epoch: int) -> UpdateDecision:
        if self.coefficient < 0:
            raise ValueError("IWAU coefficient must be non-negative")
        effective = self.base_step / (1.0 + self.coefficient * width)
        return UpdateDecision(
            effective * g_lower,
            effective * g_upper,
            1.0,
            float(effective.mean()),
            True,
        )

