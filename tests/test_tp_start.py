"""Regression tests for direct peak-state activation of IWAU."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest
import torch

from width_cert.core import (
    CalibrationStatus,
    CertifiedTrainer,
    FixedUpdate,
    IntervalWidthAdaptiveUpdate,
    calibrate_iwau,
    continue_with_iwau,
)
from width_cert.logging import RunLogger, RunStatus
from width_cert.logging.checkpoints import load_checkpoint


def _make_trainer(epochs: int = 3, base_step: float = 0.1):
    widths = torch.tensor([4.0, 20.0])
    g_lower = -widths / 2.0
    g_upper = widths / 2.0

    def gradient_bounds(_state):
        return g_lower.clone(), g_upper.clone()

    def apply_update(state, lower, upper, step):
        state["w_L"] -= step * upper
        state["w_U"] -= step * lower

    def certified_accuracy(state):
        return 100.0 - float((state["w_U"] - state["w_L"]).mean())

    trainer = CertifiedTrainer(
        gradient_bounds=gradient_bounds,
        certified_accuracy=certified_accuracy,
        apply_update=apply_update,
        horizon=epochs,
        weight_width=lambda state: float((state["w_U"] - state["w_L"]).mean()),
    )
    initial = {"w_L": torch.zeros(2), "w_U": torch.zeros(2)}
    return trainer, initial, base_step


def _make_peak_then_decline_run():
    g_lower = torch.tensor([-2.0])
    g_upper = torch.tensor([0.0])

    def gradient_bounds(_state):
        return g_lower.clone(), g_upper.clone()

    def apply_update(state, lower, upper, step):
        state["w_L"] -= step * upper
        state["w_U"] -= step * lower

    def certified_accuracy(state):
        center = float((state["w_L"] + state["w_U"]) / 2.0)
        return 100.0 - 10.0 * abs(center - 2.0)

    trainer = CertifiedTrainer(
        gradient_bounds=gradient_bounds,
        certified_accuracy=certified_accuracy,
        apply_update=apply_update,
        horizon=4,
        weight_width=lambda state: float(state["w_U"] - state["w_L"]),
    )
    initial = {"w_L": torch.zeros(1), "w_U": torch.zeros(1)}
    fixed = trainer.run(initial, FixedUpdate(1.0), capture_snapshots=True)
    return trainer, initial, fixed


def test_states_and_updates_have_unambiguous_epoch_mapping():
    trainer, initial, base_step = _make_trainer(epochs=3)
    run = trainer.run(initial, FixedUpdate(base_step), capture_snapshots=True)
    assert [state.epoch for state in run.states] == [0, 1, 2, 3]
    assert [(u.epoch, u.next_epoch) for u in run.updates] == [(0, 1), (1, 2), (2, 3)]
    assert sorted(run.snapshots) == [0, 1, 2, 3]


def test_calibration_uses_exact_sum_window_and_peak_epoch():
    trainer, _, fixed = _make_peak_then_decline_run()
    result = calibrate_iwau(fixed, horizon=trainer.horizon, retention=0.95)
    assert result.status is CalibrationStatus.ACTIVE
    assert result.peak_epoch == 2
    assert result.reference_epoch == 3
    expected = sum(u.gradient_width for u in fixed.updates if 2 <= u.epoch < 3)
    assert result.cumulative_gradient_width == pytest.approx(expected)
    assert result.coefficient == pytest.approx((4 - 2) / expected)


def test_peak_search_includes_initial_state():
    trainer, initial, _ = _make_trainer(epochs=3)
    fixed = trainer.run(initial, FixedUpdate(1.0), capture_snapshots=True)
    result = calibrate_iwau(fixed, horizon=3, retention=0.8)
    assert result.peak_epoch == 0
    assert 0 in fixed.snapshots


def test_no_post_peak_drop_is_not_required():
    def zero_bounds(_state):
        zero = torch.zeros(1)
        return zero, zero

    trainer = CertifiedTrainer(
        gradient_bounds=zero_bounds,
        certified_accuracy=lambda _state: 50.0,
        apply_update=lambda *_args: None,
        horizon=3,
    )
    initial = {"w_L": torch.zeros(1), "w_U": torch.zeros(1)}
    fixed = trainer.run(initial, FixedUpdate(0.1), capture_snapshots=True)
    result = calibrate_iwau(fixed, horizon=3)
    assert result.status is CalibrationStatus.NOT_REQUIRED
    assert result.reason == "no_reference_decline_within_horizon"
    assert result.coefficient == 0.0


def test_truncated_prerun_without_reference_decline_is_invalid():
    trainer, initial, base_step = _make_trainer(epochs=2)
    fixed = trainer.run(initial, FixedUpdate(base_step), capture_snapshots=True)
    result = calibrate_iwau(fixed, horizon=5)
    assert result.status is CalibrationStatus.INVALID


def test_direct_continuation_starts_with_transition_tp_to_tp_plus_one():
    trainer, _, fixed = _make_peak_then_decline_run()
    result = continue_with_iwau(trainer, fixed, base_step=1.0, retention=0.95)
    assert result.calibration.peak_epoch == 2
    assert [state.epoch for state in result.run.states] == [0, 1, 2, 3, 4]
    first = next(update for update in result.run.updates if update.iwau_active)
    assert (first.epoch, first.next_epoch) == (2, 3)
    assert result.run.states[:3] == fixed.states[:3]
    assert sorted(result.run.snapshots) == [0, 1, 2, 3, 4]


def test_zero_coefficient_iwau_is_exactly_fixed_update():
    trainer, initial, base_step = _make_trainer(epochs=3)
    fixed = trainer.run(initial, FixedUpdate(base_step))
    iwau = trainer.run(initial, IntervalWidthAdaptiveUpdate(base_step, 0.0))
    assert iwau.peak_certified_accuracy == pytest.approx(fixed.peak_certified_accuracy)
    assert iwau.cert_at_horizon == pytest.approx(fixed.cert_at_horizon)
    assert [s.certified_accuracy for s in iwau.states] == pytest.approx(
        [s.certified_accuracy for s in fixed.states]
    )


def test_coordinatewise_width_increment_is_capped():
    base_step, coefficient = 0.1, 2.0
    omega = torch.tensor([4.0, 20.0])
    decision = IntervalWidthAdaptiveUpdate(base_step, coefficient).decide(
        -omega / 2.0, omega / 2.0, omega, epoch=0
    )
    increment = base_step * omega / (1.0 + coefficient * omega)
    assert torch.all(increment <= base_step / coefficient + 1e-12)
    assert math.isclose(
        decision.effective_step_mean,
        float((base_step / (1.0 + coefficient * omega)).mean()),
    )


def test_divergence_is_not_silently_reported_as_zero_cert_at_T():
    trainer, initial, _ = _make_trainer(epochs=3)
    trainer.max_gradient_width = 1.0
    run = trainer.run(initial, FixedUpdate(0.1))
    assert run.status is RunStatus.DIVERGED
    assert run.cert_at_horizon is None


def test_structured_logger_writes_states_updates_and_status():
    output = Path(__file__).parent / ".test-output" / "structured-logger"
    output.mkdir(parents=True, exist_ok=True)
    for name in (
        "config.json", "states.csv", "updates.csv", "summary.json", "status.json",
        "last_state.pt", "selected_state.pt",
    ):
        path = output / name
        if path.exists():
            path.unlink()
    trainer, initial, base_step = _make_trainer(epochs=2)
    run = trainer.run(initial, FixedUpdate(base_step), capture_snapshots=True)
    logger = RunLogger(output, echo=False)
    logger.write_config({"method": "IC+IWAU", "evaluation": {"input_domain": "clean"}})
    logger.write_run(run, extra_summary={"seconds_per_epoch": 0.01})
    assert {p.name for p in output.iterdir()} == {
        "config.json", "states.csv", "updates.csv", "summary.json", "status.json",
            "last_state.pt", "selected_state.pt",
    }
    with (output / "states.csv").open(encoding="utf-8") as handle:
        states = list(csv.DictReader(handle))
    with (output / "updates.csv").open(encoding="utf-8") as handle:
        updates = list(csv.DictReader(handle))
    status = json.loads((output / "status.json").read_text(encoding="utf-8"))
    assert [int(row["epoch"]) for row in states] == [0, 1, 2]
    assert "calibration_certified_accuracy" in states[0]
    assert "certified_accuracy" not in states[0]
    assert [(int(row["epoch"]), int(row["next_epoch"])) for row in updates] == [(0, 1), (1, 2)]
    assert status["status"] == "completed"
    assert status["calibration_cert_at_T"] is not None
    last_epoch, _ = load_checkpoint(output / "last_state.pt")
    peak_epoch, _ = load_checkpoint(output / "selected_state.pt")
    assert last_epoch == 2
    assert peak_epoch == run.peak_epoch


def test_terminal_logger_uses_aligned_unambiguous_metric_names(capsys):
    trainer, initial, base_step = _make_trainer(epochs=1)
    run = trainer.run(initial, FixedUpdate(base_step))
    output = Path(__file__).parent / ".test-output" / "terminal-format"
    logger = RunLogger(output, echo=True)
    logger.begin_run(
        index=1,
        total=4,
        method="IBP",
        seed=0,
        training_radius=0.001,
        evaluation_radius=0.001,
        horizon=1,
    )
    logger.echo_run(run)
    logger.finish_run(run, elapsed_seconds=0.25)
    terminal = capsys.readouterr().out
    assert "[RUN 001/004 START]" in terminal
    assert "cal_cert" in terminal
    assert "cal_clean" in terminal
    assert "weight_width" in terminal
    assert "grad_width" in terminal
    assert "mean_step" in terminal
    assert "[TRAINING COMPLETE]" in terminal
    assert "omega=" not in terminal
    assert " clean=" not in terminal
