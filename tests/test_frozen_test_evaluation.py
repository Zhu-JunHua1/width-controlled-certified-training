"""Tests for the calibration/test protocol firewall."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader, TensorDataset

from width_cert.bounds import BoundMethod
from width_cert.experiments.evaluation import evaluate_frozen_checkpoints
from width_cert.logging import RunStatus
from width_cert.logging.records import StateRecord, TrainingRun


def _completed_run() -> TrainingRun:
    selected = {"tag": torch.tensor([1.0])}
    endpoint = {"tag": torch.tensor([2.0])}
    return TrainingRun(
        states=[
            StateRecord(0, 50.0, 0.0),
            StateRecord(1, 80.0, 0.1),
            StateRecord(2, 70.0, 0.2),
        ],
        updates=[],
        peak_certified_accuracy=80.0,
        cert_at_horizon=70.0,
        status=RunStatus.COMPLETED,
        status_reason="horizon_reached",
        final_state=endpoint,
        snapshots={0: {"tag": torch.tensor([0.0])}, 1: selected, 2: endpoint},
    )


def test_test_evaluation_cannot_change_calibration_selected_epoch(monkeypatch):
    """Test samples are consumed only after the selected epoch is immutable."""

    run = _completed_run()
    selected_epoch_before = run.peak_epoch
    seen_state_tags: list[float] = []

    def fake_clean(_loader, state, *, dimensions):
        del dimensions
        seen_state_tags.append(float(state["tag"]))
        return 90.0 + float(state["tag"])

    def fake_cert(
        _graph,
        _loader,
        state,
        *,
        input_radius,
        method,
        dimensions,
        diagnostics,
    ):
        del input_radius, dimensions, diagnostics
        seen_state_tags.append(float(state["tag"]))
        offset = 10.0 if method is BoundMethod.IC else 0.0
        return SimpleNamespace(certified_accuracy=50.0 + float(state["tag"]) + offset)

    monkeypatch.setattr(
        "width_cert.experiments.evaluation.evaluate_center_accuracy", fake_clean
    )
    monkeypatch.setattr(
        "width_cert.experiments.evaluation.evaluate_certification", fake_cert
    )
    loader = DataLoader(
        TensorDataset(torch.tensor([[999.0]]), torch.tensor([999.0])),
        batch_size=1,
    )
    records = evaluate_frozen_checkpoints(
        run=run,
        evaluation_graph=object(),
        test_loader=loader,
        input_radius=0.1,
        methods=(BoundMethod.IBP, BoundMethod.IC),
        dimensions=[1, 1],
    )

    assert run.peak_epoch == selected_epoch_before == 1
    assert seen_state_tags == [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]
    assert [record.evaluation_method for record in records] == ["IBP", "IC"]
    assert records[0].test_cert_at_selected == 51.0
    assert records[1].test_cert_at_selected == 61.0


def test_test_evaluation_rejects_a_run_without_the_frozen_checkpoint():
    run = _completed_run()
    del run.snapshots[run.peak_epoch]
    loader = DataLoader(TensorDataset(torch.zeros(1, 1), torch.zeros(1)), batch_size=1)
    try:
        evaluate_frozen_checkpoints(
            run=run,
            evaluation_graph=object(),
            test_loader=loader,
            input_radius=0.0,
            methods=(BoundMethod.IBP,),
            dimensions=[1, 1],
        )
    except RuntimeError as error:
        assert "calibration-selected checkpoint" in str(error)
    else:
        raise AssertionError("missing frozen checkpoint must stop final test evaluation")
