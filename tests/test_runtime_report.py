"""Runtime aggregation keeps pipeline and per-update costs distinct."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from width_cert.analysis import export_runtime_report


def _write_summary(
    path: Path,
    *,
    elapsed: float,
    executed_updates: int,
    horizon: int,
    status: str = "completed",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "status": status,
                "elapsed_seconds": elapsed,
                "complete_elapsed_seconds": elapsed + 2.0,
                "test_evaluation_seconds": 2.0,
                "pre_run_seconds": elapsed / 2.0,
                "continuation_seconds": elapsed / 2.0,
                "executed_updates": executed_updates,
                "seconds_per_executed_update": elapsed / executed_updates,
                "wall_seconds_per_reported_epoch": elapsed / horizon,
                "peak_device_memory_bytes": 64 * 1024 * 1024,
            }
        ),
        encoding="utf-8",
    )


def test_runtime_report_aggregates_two_denominators_and_is_traceable(tmp_path: Path):
    output = tmp_path / "runtime-report"
    results = output / "results"
    _write_summary(
        results / "P1" / "eps_0.01" / "seed_0" / "IC_IWAU" / "summary.json",
        elapsed=80.0,
        executed_updates=80,
        horizon=40,
    )
    _write_summary(
        results / "P1" / "eps_0.01" / "seed_1" / "IC_IWAU" / "summary.json",
        elapsed=120.0,
        executed_updates=60,
        horizon=40,
        status="diverged",
    )

    report, manifest = export_runtime_report(results, output / "export")
    with report.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    row = rows[0]
    assert row["experiment"] == "P1"
    assert float(row["training_seconds_mean"]) == 100.0
    assert float(row["wall_seconds_per_reported_epoch_mean"]) == 2.5
    assert float(row["seconds_per_executed_update_mean"]) == 1.5
    assert row["completed_seeds"] == "1"
    assert row["diverged_seeds"] == "1"

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0"
    assert len(payload["sources"]) == 2
    assert str(output.resolve()) not in manifest.read_text(encoding="utf-8")
