import csv
import json
from pathlib import Path

import pytest

from width_cert.analysis import ResultGridSpec, validate_result_grid


FIELDS = [
    "radius",
    "seed",
    "method",
    "evaluation_method",
    "test_cert_at_selected",
    "test_cert_at_T",
    "status",
    "resolved_config_sha256",
    "experiment_lock_hash",
    "code_commit",
    "code_dirty",
]


def _write_grid(root: Path, *, diverged: bool = False) -> ResultGridSpec:
    spec = ResultGridSpec(
        radii=(0.0001, 0.001, 0.01),
        seeds=(2, 3),
        methods=("IBP", "IBP+IWAU", "IC", "IC+IWAU"),
        evaluators=("IBP", "IC"),
        config_sha256="config",
        lock_sha256="lock",
        code_commit="commit",
    )
    rows = []
    for radius in spec.radii:
        for seed in spec.seeds:
            for method in spec.methods:
                cell = root / f"eps_{radius:g}" / f"seed_{seed}" / method.replace("+", "_")
                cell.mkdir(parents=True)
                status = "diverged" if diverged and radius == 0.01 and seed == 3 else "completed"
                (cell / "summary.json").write_text(
                    json.dumps({"status": status}), encoding="utf-8"
                )
                for evaluator in spec.evaluators:
                    rows.append(
                        {
                            "radius": f"{radius:g}",
                            "seed": str(seed),
                            "method": method,
                            "evaluation_method": evaluator,
                            "test_cert_at_selected": "70.0",
                            "test_cert_at_T": "" if status == "diverged" else "60.0",
                            "status": status,
                            "resolved_config_sha256": "config",
                            "experiment_lock_hash": "lock",
                            "code_commit": "commit",
                            "code_dirty": "False",
                        }
                    )
    with (root / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return spec


def test_result_grid_validator_accepts_exact_completed_and_diverged_cells(tmp_path):
    spec = _write_grid(tmp_path, diverged=True)
    report = validate_result_grid(tmp_path, spec)
    assert report["summaries"] == 24
    assert report["rows"] == 48


def test_result_grid_validator_rejects_duplicate_or_missing_cells(tmp_path):
    spec = _write_grid(tmp_path)
    table = tmp_path / "results.csv"
    rows = list(csv.DictReader(table.open(encoding="utf-8")))
    with table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows[:-1] + [rows[0]])
    with pytest.raises(ValueError, match="duplicate result cells"):
        validate_result_grid(tmp_path, spec)


def test_result_grid_validator_rejects_non_null_diverged_terminal(tmp_path):
    spec = _write_grid(tmp_path, diverged=True)
    table = tmp_path / "results.csv"
    rows = list(csv.DictReader(table.open(encoding="utf-8")))
    for row in rows:
        if row["status"] == "diverged":
            row["test_cert_at_T"] = "0.0"
    with table.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="null raw terminal"):
        validate_result_grid(tmp_path, spec)
