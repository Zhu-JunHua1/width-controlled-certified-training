"""Aggregate per-seed results without paper-specific dependencies."""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def summarize_results(
    results_csv: str | Path,
    output_csv: str | Path | None = None,
    *,
    diverged_as_zero: bool = False,
) -> list[dict[str, object]]:
    with Path(results_csv).open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "radius",
        "seed",
        "method",
        "evaluation_method",
        "test_cert_at_selected",
        "test_cert_at_T",
        "status",
    }
    missing = required - (set(rows[0]) if rows else set())
    if missing:
        raise ValueError(f"results file is missing columns: {sorted(missing)}")

    groups: dict[tuple[float, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["status"] in {"completed", "diverged"}:
            groups[
                (float(row["radius"]), row["method"], row["evaluation_method"])
            ].append(row)

    summary: list[dict[str, object]] = []
    for (radius, method, evaluation_method), group in sorted(groups.items()):
        selected = [float(row["test_cert_at_selected"]) for row in group]
        endpoints = []
        for row in group:
            if row["status"] == "completed" and row["test_cert_at_T"]:
                endpoints.append(float(row["test_cert_at_T"]))
            elif row["status"] == "diverged" and diverged_as_zero:
                endpoints.append(0.0)
        summary.append(
            {
                "radius": radius,
                "method": method,
                "evaluation_method": evaluation_method,
                "seeds": len({int(row["seed"]) for row in group}),
                "completed_seeds": sum(row["status"] == "completed" for row in group),
                "diverged_seeds": sum(row["status"] == "diverged" for row in group),
                "endpoint_seeds": len(endpoints),
                "test_selected_mean": statistics.mean(selected),
                "test_selected_std": statistics.pstdev(selected),
                "test_at_T_mean": statistics.mean(endpoints) if endpoints else None,
                "test_at_T_std": statistics.pstdev(endpoints) if endpoints else None,
                "iwau_active_seeds": sum(
                    _truthy(row.get("iwau_effective", "false")) for row in group
                ),
                "endpoint_policy": "diverged_as_zero" if diverged_as_zero else "completed_only",
            }
        )
    if output_csv is not None:
        target = Path(output_csv)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]) if summary else [])
            if summary:
                writer.writeheader()
                writer.writerows(summary)
    return summary
