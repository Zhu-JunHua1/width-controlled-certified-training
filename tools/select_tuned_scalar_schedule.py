"""Select a scalar schedule from calibration-only development runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


def _read_candidates(
    root: Path,
    *,
    expected_candidates: int,
    expected_starts: int,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float], list[dict[str, object]]] = {}
    for path in sorted(root.rglob("summary.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        # Internal fixed-prefix records are auditable dependencies, not
        # schedule candidates.
        if record.get("update_rule") != "TunedScalar":
            continue
        if record.get("test_evaluations"):
            raise ValueError(f"test result found in calibration-only search: {path}")
        if record.get("analysis_tier") != "development":
            raise ValueError(f"non-development result found in schedule search: {path}")
        if record.get("primary_endpoint") != "not_evaluated_in_development":
            raise ValueError(f"unexpected final-test endpoint found in schedule search: {path}")
        family = record.get("tuned_schedule_family")
        target = record.get("tuned_schedule_target_final_step")
        if family is None or target is None:
            continue
        tagged = {**record, "_seed_directory": path.parent.parent.name}
        grouped.setdefault((str(family), float(target)), []).append(tagged)
    if not grouped:
        raise ValueError(f"no tuned scalar summaries found under {root}")
    if len(grouped) != expected_candidates:
        raise ValueError(
            f"candidate count mismatch: {len(grouped)} != {expected_candidates}"
        )
    rows = []
    for (family, target), records in sorted(grouped.items()):
        if len(records) != expected_starts:
            raise ValueError(
                f"incomplete candidate {family}/{target}: "
                f"{len(records)} != {expected_starts}"
            )
        seed_directories = {str(record["_seed_directory"]) for record in records}
        if len(seed_directories) != expected_starts:
            raise ValueError(f"duplicate start found for candidate {family}/{target}")
        terminal = [
            float(record.get("calibration_cert_at_horizon") or 0.0)
            if record.get("status") == "completed"
            else 0.0
            for record in records
        ]
        selected = [
            float(record.get("peak_calibration_certified_accuracy") or 0.0)
            for record in records
        ]
        elapsed = [float(record.get("elapsed_seconds") or 0.0) for record in records]
        rows.append(
            {
                "family": family,
                "target_final_step": target,
                "starts": len(records),
                "completed": sum(record.get("status") == "completed" for record in records),
                "mean_calibration_cert_at_T": mean(terminal),
                "mean_peak_calibration_cert": mean(selected),
                "mean_elapsed_seconds": mean(elapsed),
            }
        )
    return rows


def select(
    root: Path,
    output_dir: Path,
    *,
    expected_candidates: int = 9,
    expected_starts: int = 3,
) -> tuple[Path, Path]:
    rows = _read_candidates(
        root,
        expected_candidates=expected_candidates,
        expected_starts=expected_starts,
    )
    ranked = sorted(
        rows,
        key=lambda row: (
            -row["mean_calibration_cert_at_T"],
            -row["mean_peak_calibration_cert"],
            row["mean_elapsed_seconds"],
            row["family"],
            row["target_final_step"],
        ),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "scalar_schedule_search.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ranked[0]))
        writer.writeheader()
        writer.writerows(ranked)
    winner = ranked[0]
    manifest = {
        "schema_version": "1.0",
        "selection_split": "calibration_only",
        "test_metrics_accessed": False,
        "failure_policy": "non-completed run contributes zero cert_at_T",
        "primary_selection_metric": "mean_calibration_cert_at_T",
        "tie_breaking": [
            "mean_peak_calibration_cert_descending",
            "mean_elapsed_seconds_ascending",
            "family_lexicographic",
            "target_final_step_ascending",
        ],
        "candidates": len(ranked),
        "winner": winner,
    }
    json_path = output_dir / "scalar_schedule_selection.json"
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return csv_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--expected-candidates", type=int, default=9)
    parser.add_argument("--expected-starts", type=int, default=3)
    args = parser.parse_args()
    csv_path, json_path = select(
        args.results_root,
        args.output_dir,
        expected_candidates=args.expected_candidates,
        expected_starts=args.expected_starts,
    )
    print(f"candidate table: {csv_path}")
    print(f"selection manifest: {json_path}")


if __name__ == "__main__":
    main()
