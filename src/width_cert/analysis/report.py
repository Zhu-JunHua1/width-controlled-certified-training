"""Portable report export from the public result schema.

The exporter deliberately operates only on ``results.csv`` files.  It does
not depend on private spreadsheets, workstation paths, or paper-specific
cell coordinates.  Every emitted row carries a repository-relative source
path so that a reported aggregate can be traced back to its input file.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from .summary import summarize_results


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_result_files(results_root: str | Path) -> list[Path]:
    """Return experiment-level result files in deterministic order."""

    root = Path(results_root)
    if not root.is_dir():
        raise FileNotFoundError(f"results root does not exist: {root}")
    return sorted(path for path in root.rglob("results.csv") if path.is_file())


def export_report(
    results_root: str | Path,
    output_dir: str | Path,
    *,
    diverged_as_zero: bool = False,
) -> tuple[Path, Path]:
    """Aggregate every experiment below ``results_root``.

    Returns ``(summary_csv, manifest_json)``.  The manifest contains only
    paths relative to ``results_root`` and content hashes; it is safe to
    publish and does not leak a local checkout path.
    """

    root = Path(results_root).resolve()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    sources = discover_result_files(root)
    if not sources:
        raise ValueError(f"no results.csv files found below {root}")

    rows: list[dict[str, object]] = []
    manifest_sources: list[dict[str, object]] = []
    for source in sources:
        relative = source.relative_to(root).as_posix()
        experiment = source.parent.relative_to(root).as_posix() or "."
        aggregates = summarize_results(source, diverged_as_zero=diverged_as_zero)
        for aggregate in aggregates:
            rows.append(
                {
                    "experiment": experiment,
                    "source": relative,
                    **aggregate,
                }
            )
        manifest_sources.append(
            {
                "source": relative,
                "sha256": _sha256(source),
                "aggregate_rows": len(aggregates),
            }
        )

    summary_path = destination / "protocol_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    manifest_path = destination / "report_manifest.json"
    manifest = {
        "schema_version": "1.0",
        "endpoint_policy": "diverged_as_zero" if diverged_as_zero else "completed_only",
        "summary": summary_path.name,
        "summary_sha256": _sha256(summary_path),
        "sources": manifest_sources,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary_path, manifest_path
