"""Aggregate end-to-end runtime from public per-run ``summary.json`` files.

The runtime report deliberately keeps two denominators separate:

``wall_seconds_per_reported_epoch``
    Complete training-pipeline time (including a required fixed pre-run) divided
    by the manuscript horizon ``T``.  This is the deployable wall-clock cost.

``seconds_per_executed_update``
    The same training time divided by the number of interval bound/update
    evaluations actually executed.  This distinguishes a more expensive
    update from a method that performs more updates because it calibrates on a
    fixed pre-run.

Frozen result files are never rewritten; this module creates a derived,
traceable report from their existing summaries.
"""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _pstdev(values: list[float]) -> float | None:
    return statistics.pstdev(values) if values else None


def _required_float(payload: dict[str, object], key: str, source: Path) -> float:
    value = payload.get(key)
    if value is None:
        raise ValueError(f"{source} is missing required runtime field {key!r}")
    return float(value)


def _run_identity(source: Path, root: Path) -> tuple[str, float, int, str]:
    """Return ``(experiment, radius, seed, method)`` from a public run path."""

    relative_parts = source.relative_to(root).parts
    eps_index = next(
        (index for index, part in enumerate(relative_parts) if part.startswith("eps_")),
        None,
    )
    if eps_index is None or len(relative_parts) < eps_index + 4:
        raise ValueError(
            "runtime summaries must follow "
            "EXPERIMENT/eps_RADIUS/seed_SEED/METHOD/summary.json: "
            f"{source}"
        )
    radius = float(relative_parts[eps_index][len("eps_") :])
    seed_part = relative_parts[eps_index + 1]
    if not seed_part.startswith("seed_"):
        raise ValueError(f"invalid seed directory in runtime path: {source}")
    seed = int(seed_part[len("seed_") :])
    method = relative_parts[eps_index + 2]
    experiment = "/".join(relative_parts[:eps_index]) or "."
    return experiment, radius, seed, method


def export_runtime_report(
    results_root: str | Path,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Write an aggregate runtime CSV and a content-hash manifest."""

    root = Path(results_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"results root does not exist: {root}")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    sources = sorted(root.rglob("summary.json"))
    if not sources:
        raise ValueError(f"no summary.json files found below {root}")

    grouped: dict[tuple[str, float, str], list[dict[str, object]]] = defaultdict(list)
    manifest_sources: list[dict[str, str]] = []
    for source in sources:
        experiment, radius, seed, method = _run_identity(source, root)
        payload = json.loads(source.read_text(encoding="utf-8"))
        grouped[(experiment, radius, method)].append(
            {
                "seed": seed,
                "status": str(payload.get("status", "unknown")),
                "training_seconds": _required_float(payload, "elapsed_seconds", source),
                "complete_seconds": _required_float(
                    payload, "complete_elapsed_seconds", source
                ),
                "test_seconds": _required_float(
                    payload, "test_evaluation_seconds", source
                ),
                "pre_run_seconds": _required_float(payload, "pre_run_seconds", source),
                "continuation_seconds": _required_float(
                    payload, "continuation_seconds", source
                ),
                "executed_updates": _required_float(payload, "executed_updates", source),
                "seconds_per_executed_update": _required_float(
                    payload, "seconds_per_executed_update", source
                ),
                "wall_seconds_per_reported_epoch": _required_float(
                    payload, "wall_seconds_per_reported_epoch", source
                ),
                "peak_memory_mib": _required_float(
                    payload, "peak_device_memory_bytes", source
                )
                / (1024.0**2),
            }
        )
        manifest_sources.append(
            {
                "source": source.relative_to(root).as_posix(),
                "sha256": _sha256(source),
            }
        )

    rows: list[dict[str, object]] = []
    for (experiment, radius, method), group in sorted(grouped.items()):
        def values(key: str) -> list[float]:
            return [float(item[key]) for item in group]

        rows.append(
            {
                "experiment": experiment,
                "radius": radius,
                "method": method,
                "seeds": len({int(item["seed"]) for item in group}),
                "completed_seeds": sum(
                    item["status"] == "completed" for item in group
                ),
                "diverged_seeds": sum(item["status"] == "diverged" for item in group),
                "training_seconds_mean": _mean(values("training_seconds")),
                "training_seconds_std": _pstdev(values("training_seconds")),
                "complete_seconds_mean": _mean(values("complete_seconds")),
                "test_seconds_mean": _mean(values("test_seconds")),
                "pre_run_seconds_mean": _mean(values("pre_run_seconds")),
                "continuation_seconds_mean": _mean(values("continuation_seconds")),
                "executed_updates_mean": _mean(values("executed_updates")),
                "seconds_per_executed_update_mean": _mean(
                    values("seconds_per_executed_update")
                ),
                "seconds_per_executed_update_std": _pstdev(
                    values("seconds_per_executed_update")
                ),
                "wall_seconds_per_reported_epoch_mean": _mean(
                    values("wall_seconds_per_reported_epoch")
                ),
                "wall_seconds_per_reported_epoch_std": _pstdev(
                    values("wall_seconds_per_reported_epoch")
                ),
                "peak_memory_mib_mean": _mean(values("peak_memory_mib")),
            }
        )

    report_path = destination / "runtime_summary.csv"
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    manifest_path = destination / "runtime_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "training_time_scope": (
                    "fixed pre-run plus continuation; excludes frozen-test evaluation"
                ),
                "reported_epoch_denominator": (
                    "manuscript horizon T; fixed pre-run cost remains in the numerator"
                ),
                "executed_update_denominator": (
                    "actual interval bound/update evaluations, including a required pre-run"
                ),
                "report": report_path.name,
                "report_sha256": _sha256(report_path),
                "sources": manifest_sources,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return report_path, manifest_path
