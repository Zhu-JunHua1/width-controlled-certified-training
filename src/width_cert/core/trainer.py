"""Architecture-independent certified-training loop."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from .state import IntervalState, clone_state
from .update_rules import UpdateRule
from ..logging.records import RunStatus, StateRecord, TrainingRun, UpdateRecord


class CertifiedTrainer:
    """Run interval training while keeping states and updates separate.

    ``states[t]`` describes ``B_t``.  ``updates[t]`` describes the transition
    ``B_t -> B_{t+1}``.  This separation prevents the off-by-one ambiguity of
    mixed trajectory rows.
    """

    def __init__(
        self,
        *,
        gradient_bounds: Callable[[IntervalState], tuple[Any, Any]],
        certified_accuracy: Callable[[IntervalState], float],
        apply_update: Callable[[dict[str, Any], Any, Any, float], None],
        horizon: int,
        weight_width: Callable[[IntervalState], float] | None = None,
        clean_accuracy: Callable[[IntervalState], float] | None = None,
        max_gradient_width: float = 50.0,
    ) -> None:
        if horizon < 0:
            raise ValueError("horizon must be non-negative")
        self.gradient_bounds = gradient_bounds
        self.certified_accuracy = certified_accuracy
        self.apply_update = apply_update
        self.horizon = int(horizon)
        self.weight_width = weight_width
        self.clean_accuracy = clean_accuracy
        self.max_gradient_width = float(max_gradient_width)

    def _state_record(self, epoch: int, state: IntervalState) -> StateRecord:
        return StateRecord(
            epoch=epoch,
            certified_accuracy=float(self.certified_accuracy(state)),
            weight_width=float(self.weight_width(state)) if self.weight_width else 0.0,
            clean_accuracy=(
                float(self.clean_accuracy(state)) if self.clean_accuracy else None
            ),
        )

    def run(
        self,
        initial_state: IntervalState,
        update_rule: UpdateRule,
        *,
        start_epoch: int = 0,
        peak_floor: float | None = None,
        capture_snapshots: bool = False,
        on_epoch: Callable[[StateRecord, UpdateRecord | None], None] | None = None,
    ) -> TrainingRun:
        state = clone_state(initial_state)
        first = self._state_record(start_epoch, state)
        states = [first]
        updates: list[UpdateRecord] = []
        snapshots = {start_epoch: clone_state(state)} if capture_snapshots else {}
        peak = max(first.certified_accuracy, peak_floor or float("-inf"))
        status = RunStatus.COMPLETED
        reason = "horizon_reached"
        if on_epoch:
            on_epoch(first, None)

        for epoch in range(start_epoch, self.horizon):
            try:
                g_lower, g_upper = self.gradient_bounds(state)
                gradient_width = (g_upper - g_lower).clamp(min=0.0)
                width_mean = float(gradient_width.mean())
            except (FloatingPointError, RuntimeError, ValueError) as exc:
                status = RunStatus.NUMERICAL_ERROR
                reason = f"gradient_bound_error:{type(exc).__name__}"
                break

            if not math.isfinite(width_mean):
                status = RunStatus.NUMERICAL_ERROR
                reason = "non_finite_gradient_width"
                break
            if width_mean > self.max_gradient_width:
                status = RunStatus.DIVERGED
                reason = "gradient_width_limit_exceeded"
                break

            decision = update_rule.decide(g_lower, g_upper, gradient_width, epoch)
            self.apply_update(
                state,
                decision.lower_gradient,
                decision.upper_gradient,
                decision.scalar_step,
            )
            update = UpdateRecord(
                epoch=epoch,
                next_epoch=epoch + 1,
                gradient_width=width_mean,
                effective_step_mean=decision.effective_step_mean,
                iwau_active=decision.iwau_active,
                update_rule=decision.rule_name,
            )
            current = self._state_record(epoch + 1, state)
            updates.append(update)
            states.append(current)
            peak = max(peak, current.certified_accuracy)
            if capture_snapshots:
                snapshots[epoch + 1] = clone_state(state)
            if on_epoch:
                on_epoch(current, update)

        reached = states[-1].epoch == self.horizon and status is RunStatus.COMPLETED
        return TrainingRun(
            states=states,
            updates=updates,
            peak_certified_accuracy=peak,
            cert_at_horizon=states[-1].certified_accuracy if reached else None,
            status=status,
            status_reason=reason,
            final_state=clone_state(state),
            snapshots=snapshots,
        )
