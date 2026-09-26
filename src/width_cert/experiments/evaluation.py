"""Final test evaluation for checkpoints frozen on the calibration split.

This module is intentionally separate from training and IWAU calibration.  Its
API accepts an already completed ``TrainingRun`` and never returns information
that can alter checkpoint selection or update schedules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from torch.utils.data import DataLoader

from ..bounds import BoundMethod, IntersectionDiagnostics
from ..certification import evaluate_center_accuracy, evaluate_certification
from ..logging.records import TrainingRun


@dataclass(frozen=True)
class FrozenTestEvaluation:
    """Metrics from one test certifier on two preselected checkpoints."""

    evaluation_method: str
    test_cert_at_selected: float
    test_cert_at_T: float | None
    test_clean_at_selected: float
    test_clean_at_T: float | None

    def as_record(self) -> dict[str, object]:
        return {
            "evaluation_method": self.evaluation_method,
            "test_cert_at_selected": self.test_cert_at_selected,
            "test_cert_at_T": self.test_cert_at_T,
            "test_clean_at_selected": self.test_clean_at_selected,
            "test_clean_at_T": self.test_clean_at_T,
        }


def evaluate_frozen_checkpoints(
    *,
    run: TrainingRun,
    evaluation_graph,
    test_loader: DataLoader,
    input_radius: float,
    methods: Iterable[BoundMethod],
    dimensions: list[int],
    diagnostics: dict[BoundMethod, IntersectionDiagnostics] | None = None,
) -> list[FrozenTestEvaluation]:
    """Evaluate a calibration-selected checkpoint and the terminal state.

    The selected epoch is fixed by ``run`` before this function sees test data.
    Consequently, changing test samples cannot alter the selected checkpoint,
    the IWAU coefficient, or any training update.
    """

    if run.peak_epoch not in run.snapshots:
        raise RuntimeError("calibration-selected checkpoint was not captured")

    selected_state = run.snapshots[run.peak_epoch]
    selected_clean = evaluate_center_accuracy(
        test_loader, selected_state, dimensions=dimensions
    )
    endpoint_available = run.reached_horizon
    endpoint_clean = (
        evaluate_center_accuracy(test_loader, run.final_state, dimensions=dimensions)
        if endpoint_available
        else None
    )

    evaluations: list[FrozenTestEvaluation] = []
    for method in methods:
        selected_cert = evaluate_certification(
            evaluation_graph,
            test_loader,
            selected_state,
            input_radius=input_radius,
            method=method,
            dimensions=dimensions,
            diagnostics=None if diagnostics is None else diagnostics[method],
        ).certified_accuracy
        endpoint_cert = (
            evaluate_certification(
                evaluation_graph,
                test_loader,
                run.final_state,
                input_radius=input_radius,
                method=method,
                dimensions=dimensions,
                diagnostics=None if diagnostics is None else diagnostics[method],
            ).certified_accuracy
            if endpoint_available
            else None
        )
        evaluations.append(
            FrozenTestEvaluation(
                evaluation_method=method.name,
                test_cert_at_selected=selected_cert,
                test_cert_at_T=endpoint_cert,
                test_clean_at_selected=selected_clean,
                test_clean_at_T=endpoint_clean,
            )
        )
    return evaluations
