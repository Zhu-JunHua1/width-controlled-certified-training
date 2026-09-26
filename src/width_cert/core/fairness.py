"""Helpers for common-prefix update-rule comparisons."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .trainer import CertifiedTrainer
from .update_rules import (
    CosineDecayUpdate,
    InverseTimeUpdate,
    StepDecayUpdate,
    UpdateRule,
)
from ..logging.records import TrainingRun


@dataclass(frozen=True)
class BudgetRecord:
    """Update-count accounting for a two-stage reported arm."""

    fixed_pre_run_updates: int
    continuation_updates: int

    @property
    def total_bound_evaluations(self) -> int:
        return self.fixed_pre_run_updates + self.continuation_updates


def continue_from_fixed_prefix(
    trainer: CertifiedTrainer,
    fixed_run: TrainingRun,
    *,
    start_epoch: int,
    update_rule: UpdateRule,
) -> TrainingRun:
    """Continue from the exact frozen ``B_start_epoch`` and merge the prefix."""

    if start_epoch not in fixed_run.snapshots:
        raise ValueError("fixed run does not contain the requested prefix snapshot")
    prefix_states = [state for state in fixed_run.states if state.epoch <= start_epoch]
    prefix_updates = [update for update in fixed_run.updates if update.epoch < start_epoch]
    if not prefix_states or prefix_states[-1].epoch != start_epoch:
        raise ValueError("fixed run has an incomplete prefix")
    prefix_peak = max(state.certified_accuracy for state in prefix_states)
    continuation = trainer.run(
        fixed_run.snapshots[start_epoch],
        update_rule,
        start_epoch=start_epoch,
        peak_floor=prefix_peak,
        capture_snapshots=True,
    )
    return TrainingRun(
        states=prefix_states + continuation.states[1:],
        updates=prefix_updates + continuation.updates,
        peak_certified_accuracy=continuation.peak_certified_accuracy,
        cert_at_horizon=continuation.cert_at_horizon,
        status=continuation.status,
        status_reason=continuation.status_reason,
        final_state=continuation.final_state,
        snapshots={
            **{
                epoch: snapshot
                for epoch, snapshot in fixed_run.snapshots.items()
                if epoch <= start_epoch
            },
            **continuation.snapshots,
        },
    )


def budget_record(
    fixed_run: TrainingRun,
    continued_run: TrainingRun,
    *,
    start_epoch: int,
) -> BudgetRecord:
    """Count expensive gradient-bound evaluations in a two-stage arm."""

    continuation_updates = sum(
        update.epoch >= start_epoch for update in continued_run.updates
    )
    return BudgetRecord(len(fixed_run.updates), continuation_updates)


def matched_final_step_rules(
    *,
    base_step: float,
    target_final_step: float,
    start_epoch: int,
    total_steps: int,
    step_milestone_fraction: float = 0.5,
) -> dict[str, UpdateRule]:
    """Build scalar schedules matched to one target final mean step.

    Matching the final mean step is the predeclared update-budget criterion.
    The common prefix and number of continuation updates are handled separately.
    """

    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0.0 < target_final_step <= base_step:
        raise ValueError("target_final_step must lie in (0, base_step]")
    if not 0.0 < step_milestone_fraction <= 1.0:
        raise ValueError("step_milestone_fraction must lie in (0, 1]")
    inverse_alpha = (base_step / target_final_step - 1.0) / total_steps
    milestone = max(1, min(total_steps, math.ceil(total_steps * step_milestone_fraction)))
    return {
        "inverse_time": InverseTimeUpdate(
            base_step=base_step,
            start_epoch=start_epoch,
            alpha=inverse_alpha,
        ),
        "cosine": CosineDecayUpdate(
            base_step=base_step,
            minimum_step=target_final_step,
            start_epoch=start_epoch,
            total_steps=total_steps,
        ),
        "step_decay": StepDecayUpdate(
            base_step=base_step,
            start_epoch=start_epoch,
            milestone=milestone,
            factor=target_final_step / base_step,
        ),
    }


def final_continuation_step(run: TrainingRun, *, start_epoch: int) -> float:
    """Return the last logged mean step after the common prefix."""

    updates = [update for update in run.updates if update.epoch >= start_epoch]
    if not updates:
        raise ValueError("run contains no continuation updates")
    return updates[-1].effective_step_mean
