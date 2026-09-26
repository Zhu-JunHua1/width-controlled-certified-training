"""Same-method calibration and direct peak-state IWAU continuation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .trainer import CertifiedTrainer
from .update_rules import IntervalWidthAdaptiveUpdate
from ..logging.records import RunStatus, TrainingRun


class CalibrationStatus(str, Enum):
    ACTIVE = "active"
    NOT_REQUIRED = "not_required"
    INVALID = "invalid"


@dataclass(frozen=True)
class CalibrationResult:
    peak_epoch: int
    reference_epoch: int | None
    coefficient: float
    peak_certified_accuracy: float
    cumulative_gradient_width: float
    status: CalibrationStatus
    reason: str

    @property
    def iwau_effective(self) -> bool:
        return self.status is CalibrationStatus.ACTIVE and self.coefficient > 0.0


@dataclass(frozen=True)
class IWAUResult:
    run: TrainingRun
    calibration: CalibrationResult


def calibrate_iwau(
    fixed_run: TrainingRun,
    *,
    horizon: int,
    retention: float = 0.8,
    minimum_width_sum: float = 1e-6,
) -> CalibrationResult:
    """Calibrate IWAU from the same method's complete fixed-update run."""

    if not 0.0 < retention <= 1.0:
        raise ValueError("retention must lie in (0, 1]")
    states = fixed_run.states
    if not states or states[0].epoch != 0:
        return CalibrationResult(
            0, None, 0.0, float("nan"), 0.0,
            CalibrationStatus.INVALID, "fixed_run_missing_B0",
        )

    peak = max(s.certified_accuracy for s in states)
    peak_epoch = next(s.epoch for s in states if s.certified_accuracy == peak)
    reference = next(
        (
            s.epoch
            for s in states
            if s.epoch > peak_epoch and s.certified_accuracy < retention * peak
        ),
        None,
    )
    if reference is None:
        if fixed_run.status is not RunStatus.COMPLETED or states[-1].epoch < horizon:
            return CalibrationResult(
                peak_epoch, None, 0.0, peak, 0.0,
                CalibrationStatus.INVALID, "incomplete_fixed_run_without_reference_decline",
            )
        return CalibrationResult(
            peak_epoch, None, 0.0, peak, 0.0,
            CalibrationStatus.NOT_REQUIRED, "no_reference_decline_within_horizon",
        )

    width_sum = sum(
        max(update.gradient_width, 0.0)
        for update in fixed_run.updates
        if peak_epoch <= update.epoch < reference
    )
    if width_sum <= minimum_width_sum:
        return CalibrationResult(
            peak_epoch, reference, 0.0, peak, width_sum,
            CalibrationStatus.NOT_REQUIRED, "negligible_gradient_width_growth",
        )
    coefficient = max(horizon - peak_epoch, 1) / width_sum
    return CalibrationResult(
        peak_epoch, reference, coefficient, peak, width_sum,
        CalibrationStatus.ACTIVE, "reference_decline_found",
    )


def continue_with_iwau(
    trainer: CertifiedTrainer,
    fixed_run: TrainingRun,
    *,
    base_step: float,
    retention: float = 0.8,
    coefficient_scale: float = 1.0,
) -> IWAUResult:
    """Reuse the fixed prefix through ``B_tp`` and continue with IWAU."""

    calibration = calibrate_iwau(
        fixed_run, horizon=trainer.horizon, retention=retention
    )
    if calibration.status is CalibrationStatus.INVALID:
        raise RuntimeError(f"invalid IWAU calibration: {calibration.reason}")
    if not calibration.iwau_effective:
        return IWAUResult(fixed_run, calibration)
    if calibration.peak_epoch not in fixed_run.snapshots:
        raise ValueError(
            "fixed run must capture the peak snapshot before IWAU continuation"
        )

    continuation = trainer.run(
        fixed_run.snapshots[calibration.peak_epoch],
        IntervalWidthAdaptiveUpdate(
            base_step=base_step,
            coefficient=calibration.coefficient * coefficient_scale,
        ),
        start_epoch=calibration.peak_epoch,
        peak_floor=calibration.peak_certified_accuracy,
        capture_snapshots=True,
    )
    prefix_states = [s for s in fixed_run.states if s.epoch <= calibration.peak_epoch]
    prefix_updates = [u for u in fixed_run.updates if u.epoch < calibration.peak_epoch]
    merged = TrainingRun(
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
                if epoch <= calibration.peak_epoch
            },
            **continuation.snapshots,
        },
    )
    return IWAUResult(merged, calibration)
