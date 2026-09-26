"""Update policies for interval certified training."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Protocol


@dataclass(frozen=True)
class UpdateDecision:
    """Arguments passed to the architecture-specific interval update."""

    lower_gradient: Any
    upper_gradient: Any
    scalar_step: float
    effective_step_mean: float
    iwau_active: bool
    rule_name: str = "fixed"


class UpdateRule(Protocol):
    def decide(self, g_lower: Any, g_upper: Any, width: Any, epoch: int) -> UpdateDecision:
        """Return the sound update representation for ``B_epoch``."""


@dataclass(frozen=True)
class FixedUpdate:
    base_step: float

    def decide(self, g_lower: Any, g_upper: Any, width: Any, epoch: int) -> UpdateDecision:
        return UpdateDecision(
            g_lower, g_upper, self.base_step, self.base_step, False, "fixed"
        )


@dataclass(frozen=True)
class ExponentialDecayUpdate:
    base_step: float
    decay: float = 0.6
    minimum_step: float = 1e-3

    def decide(self, g_lower: Any, g_upper: Any, width: Any, epoch: int) -> UpdateDecision:
        step = max(self.base_step * (self.decay**epoch), self.minimum_step)
        return UpdateDecision(g_lower, g_upper, step, step, False, "exponential")


def _continuation_index(epoch: int, start_epoch: int) -> int:
    """Return the one-based update index after a shared fixed prefix."""

    if epoch < start_epoch:
        raise ValueError("schedule epoch precedes its declared start_epoch")
    return epoch - start_epoch + 1


@dataclass(frozen=True)
class InverseTimeUpdate:
    """Scalar schedule ``eta_k = eta_0 / (1 + alpha*k)``."""

    base_step: float
    start_epoch: int
    alpha: float

    def decide(self, g_lower: Any, g_upper: Any, width: Any, epoch: int) -> UpdateDecision:
        if self.alpha < 0:
            raise ValueError("inverse-time alpha must be non-negative")
        index = _continuation_index(epoch, self.start_epoch)
        step = self.base_step / (1.0 + self.alpha * index)
        return UpdateDecision(g_lower, g_upper, step, step, False, "inverse_time")


@dataclass(frozen=True)
class CosineDecayUpdate:
    """Cosine schedule over a predeclared number of continuation updates."""

    base_step: float
    minimum_step: float
    start_epoch: int
    total_steps: int

    def decide(self, g_lower: Any, g_upper: Any, width: Any, epoch: int) -> UpdateDecision:
        if self.total_steps <= 0:
            raise ValueError("cosine total_steps must be positive")
        if not 0.0 <= self.minimum_step <= self.base_step:
            raise ValueError("cosine minimum_step must lie in [0, base_step]")
        index = min(_continuation_index(epoch, self.start_epoch), self.total_steps)
        phase = index / self.total_steps
        step = self.minimum_step + 0.5 * (self.base_step - self.minimum_step) * (
            1.0 + math.cos(math.pi * phase)
        )
        return UpdateDecision(g_lower, g_upper, step, step, False, "cosine")


@dataclass(frozen=True)
class StepDecayUpdate:
    """Piecewise-constant scalar decay after a shared fixed prefix."""

    base_step: float
    start_epoch: int
    milestone: int
    factor: float

    def decide(self, g_lower: Any, g_upper: Any, width: Any, epoch: int) -> UpdateDecision:
        if self.milestone <= 0:
            raise ValueError("step-decay milestone must be positive")
        if not 0.0 < self.factor <= 1.0:
            raise ValueError("step-decay factor must lie in (0, 1]")
        index = _continuation_index(epoch, self.start_epoch)
        step = self.base_step if index < self.milestone else self.base_step * self.factor
        return UpdateDecision(g_lower, g_upper, step, step, False, "step_decay")


@dataclass(frozen=True)
class GlobalWidthAdaptiveUpdate:
    """Scalar control using the mean gradient-bound width."""

    base_step: float
    coefficient: float

    def decide(self, g_lower: Any, g_upper: Any, width: Any, epoch: int) -> UpdateDecision:
        if self.coefficient < 0:
            raise ValueError("global-width coefficient must be non-negative")
        step = self.base_step / (1.0 + self.coefficient * float(width.mean()))
        return UpdateDecision(g_lower, g_upper, step, step, False, "global_width")


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
            "IWAU",
        )
