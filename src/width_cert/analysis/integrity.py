"""Fail-closed integrity checks for a frozen experiment result grid."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ResultGridSpec:
    radii: tuple[float, ...]
    seeds: tuple[int, ...]
    methods: tuple[str, ...]
    evaluators: tuple[str, ...]
    config_sha256: str | None = None
    lock_sha256: str | None = None
    code_commit: str | None = None

    @property
    def summary_count(self) -> int:
        return len(self.radii) * len(self.seeds) * len(self.methods)

    @property
    def row_count(self) -> int:
        return self.summary_count * len(self.evaluators)


def validate_result_grid(root: str | Path, spec: ResultGridSpec) -> dict[str, object]:
    """Validate exact cells, provenance, and terminal-field semantics.

    The function never substitutes analysis values for missing raw fields.  It
    verifies the public ``results.csv`` and the one reported ``summary.json``
    expected for each radius/start/method cell.
    """

    result_root = Path(root)
    table_path = result_root / "results.csv"
    if not table_path.is_file():
        raise ValueError(f"missing result table: {table_path}")
    with table_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != spec.row_count:
        raise ValueError(f"expected {spec.row_count} result rows, found {len(rows)}")

    expected_cells = {
        (f"{radius:g}", str(seed), method, evaluator)
        for radius in spec.radii
        for seed in spec.seeds
        for method in spec.methods
        for evaluator in spec.evaluators
    }
    actual_cells = [
        (row["radius"], row["seed"], row["method"], row["evaluation_method"])
        for row in rows
    ]
    if len(set(actual_cells)) != len(actual_cells):
        raise ValueError("duplicate result cells detected")
    missing = expected_cells - set(actual_cells)
    extra = set(actual_cells) - expected_cells
    if missing or extra:
        raise ValueError(f"result-grid mismatch: missing={sorted(missing)}, extra={sorted(extra)}")

    summary_paths = {
        result_root
        / f"eps_{radius:g}"
        / f"seed_{seed}"
        / method.replace("+", "_")
        / "summary.json"
        for radius in spec.radii
        for seed in spec.seeds
        for method in spec.methods
    }
    missing_summaries = sorted(path for path in summary_paths if not path.is_file())
    if missing_summaries:
        raise ValueError(f"missing {len(missing_summaries)} reported summaries")
    reported_summaries = {
        path for path in result_root.rglob("summary.json") if "calibration_pre_run" not in path.parts
    }
    if reported_summaries != summary_paths:
        raise ValueError(
            "reported-summary grid mismatch: "
            f"expected={len(summary_paths)}, found={len(reported_summaries)}"
        )

    for row in rows:
        if spec.config_sha256 and row["resolved_config_sha256"] != spec.config_sha256:
            raise ValueError("resolved configuration hash mismatch")
        if spec.lock_sha256 and row["experiment_lock_hash"] != spec.lock_sha256:
            raise ValueError("experiment lock hash mismatch")
        if spec.code_commit and row["code_commit"] != spec.code_commit:
            raise ValueError("code commit mismatch")
        if spec.code_commit and row["code_dirty"].lower() != "false":
            raise ValueError("frozen result reports a dirty code tree")
        selected = row.get("test_cert_at_selected", "")
        terminal = row.get("test_cert_at_T", "")
        status = row.get("status", "")
        if status == "completed" and (selected == "" or terminal == ""):
            raise ValueError("completed result has a missing selected or terminal value")
        if status == "diverged" and terminal != "":
            raise ValueError("diverged result must retain a null raw terminal value")

    return {
        "summaries": spec.summary_count,
        "rows": spec.row_count,
        "radii": list(spec.radii),
        "seeds": list(spec.seeds),
        "methods": list(spec.methods),
        "evaluators": list(spec.evaluators),
    }
