"""Tests for portable multi-experiment report export."""

import csv
import json
from pathlib import Path

from width_cert.analysis import export_report


def _write_results(path: Path, selected: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "radius": 0.01,
            "seed": 0,
            "method": "IC",
            "evaluation_method": "IC",
            "test_cert_at_selected": selected,
            "test_cert_at_T": selected - 1,
            "status": "completed",
            "iwau_effective": False,
        }
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_export_report_is_relative_and_traceable():
    output = Path(__file__).parent / ".test-output" / "report"
    results = output / "results"
    _write_results(results / "P1" / "results.csv", 70.0)
    _write_results(results / "P2" / "variant" / "results.csv", 80.0)

    summary, manifest = export_report(results, output / "export")
    with summary.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["experiment"] for row in rows] == ["P1", "P2/variant"]
    assert [row["source"] for row in rows] == [
        "P1/results.csv",
        "P2/variant/results.csv",
    ]

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "2.0"
    assert len(payload["sources"]) == 2
    assert str(output.resolve()) not in manifest.read_text(encoding="utf-8")
