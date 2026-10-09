"""Fail-closed tests for calibration-only scalar-schedule selection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.select_tuned_scalar_schedule import select


def _write_summary(
    root: Path,
    *,
    candidate: str,
    seed: int,
    family: str,
    target: float,
    terminal: float,
    peak: float,
    elapsed: float,
    test_evaluations: list[object] | None = None,
) -> None:
    path = root / candidate / "eps_0.01" / f"seed_{seed}" / "IC_TunedScalar"
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "analysis_tier": "development",
        "primary_endpoint": "not_evaluated_in_development",
        "test_evaluations": test_evaluations or [],
        "update_rule": "TunedScalar",
        "tuned_schedule_family": family,
        "tuned_schedule_target_final_step": target,
        "status": "completed",
        "calibration_cert_at_horizon": terminal,
        "peak_calibration_certified_accuracy": peak,
        "elapsed_seconds": elapsed,
    }
    (path / "summary.json").write_text(json.dumps(payload), encoding="utf-8")


def test_selector_uses_only_complete_calibration_candidates(tmp_path: Path):
    root = tmp_path / "results"
    dependency = root / "inverse" / "eps_0.01" / "seed_1" / "IC_TunedScalar" / "calibration_pre_run"
    dependency.mkdir(parents=True)
    (dependency / "summary.json").write_text(
        json.dumps({"reported_result": False, "status": "completed"}),
        encoding="utf-8",
    )
    for seed in (1, 2):
        _write_summary(
            root,
            candidate="inverse",
            seed=seed,
            family="inverse_time",
            target=0.04,
            terminal=60.0,
            peak=70.0,
            elapsed=10.0,
        )
        _write_summary(
            root,
            candidate="cosine",
            seed=seed,
            family="cosine",
            target=0.06,
            terminal=65.0,
            peak=68.0,
            elapsed=11.0,
        )
    _, manifest_path = select(
        root,
        tmp_path / "selected",
        expected_candidates=2,
        expected_starts=2,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["test_metrics_accessed"] is False
    assert manifest["winner"]["family"] == "cosine"


def test_selector_rejects_any_test_evaluation(tmp_path: Path):
    root = tmp_path / "results"
    _write_summary(
        root,
        candidate="inverse",
        seed=1,
        family="inverse_time",
        target=0.04,
        terminal=60.0,
        peak=70.0,
        elapsed=10.0,
        test_evaluations=[{"evaluation_method": "IC", "test_cert_at_T": 99.0}],
    )
    with pytest.raises(ValueError, match="test result"):
        select(
            root,
            tmp_path / "selected",
            expected_candidates=1,
            expected_starts=1,
        )


def test_selector_rejects_incomplete_grid(tmp_path: Path):
    root = tmp_path / "results"
    _write_summary(
        root,
        candidate="inverse",
        seed=1,
        family="inverse_time",
        target=0.04,
        terminal=60.0,
        peak=70.0,
        elapsed=10.0,
    )
    with pytest.raises(ValueError, match="incomplete candidate"):
        select(
            root,
            tmp_path / "selected",
            expected_candidates=1,
            expected_starts=2,
        )
