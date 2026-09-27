"""Paired analysis is deterministic and follows the preregistered family."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml

from width_cert.analysis.statistics import (
    analyze_paired_hypotheses,
    exact_sign_test,
    holm_adjust,
)


OUTPUT = Path(__file__).parent / ".test-output" / "statistics"


def test_exact_one_sided_sign_test_and_holm_adjustment():
    p_value, positive, zero, negative = exact_sign_test([1.0] * 10)
    assert (positive, zero, negative) == (10, 0, 0)
    assert p_value == pytest.approx(1.0 / 1024.0)
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_paired_report_uses_common_seeds_and_writes_corrected_family():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT / "results.csv"
    fieldnames = [
        "radius",
        "seed",
        "method",
        "evaluation_method",
        "test_cert_at_selected",
        "test_cert_at_T",
        "status",
    ]
    rows = []
    for seed in range(4):
        rows.extend(
            [
                {
                    "radius": 0.01,
                    "seed": seed,
                    "method": "IBP",
                    "evaluation_method": "IBP",
                    "test_cert_at_selected": 50 + seed,
                    "test_cert_at_T": "",
                    "status": "diverged",
                },
                {
                    "radius": 0.01,
                    "seed": seed,
                    "method": "IC",
                    "evaluation_method": "IBP",
                    "test_cert_at_selected": 52 + seed,
                    "test_cert_at_T": 40 + seed,
                    "status": "completed",
                },
            ]
        )
    with results_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    specification_path = OUTPUT / "hypotheses.yaml"
    specification_path.write_text(
        yaml.safe_dump(
            {
                "bootstrap": {"samples": 1000, "seed": 7},
                "hypotheses": [
                    {
                        "id": "H1",
                        "radius": 0.01,
                        "method_a": "IBP",
                        "method_b": "IC",
                        "evaluation_method": "IBP",
                        "endpoint": "test_cert_at_selected",
                        "alternative": "greater",
                        "expected_pairs": 4,
                    },
                    {
                        "id": "H2",
                        "radius": 0.01,
                        "method_a": "IBP",
                        "method_b": "IC",
                        "evaluation_method": "IBP",
                        "endpoint": "test_cert_at_T",
                        "alternative": "greater",
                        "missing_as_zero": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    output_path = OUTPUT / "paired.csv"
    report = analyze_paired_hypotheses(results_path, specification_path, output_path)
    assert len(report) == 2
    assert report[0].mean_difference == pytest.approx(2.0)
    assert report[0].positive == 4
    assert report[0].p_value == pytest.approx(1.0 / 16.0)
    assert report[0].holm_adjusted_p == pytest.approx(1.0 / 8.0)
    assert report[1].mean_difference == pytest.approx(41.5)
    assert output_path.is_file()
