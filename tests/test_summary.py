"""Tests for generic result aggregation."""

import csv
from pathlib import Path

from width_cert.analysis import summarize_results
from width_cert.logging import ConsoleReporter


def test_summary_uses_population_standard_deviation():
    output = Path(__file__).parent / ".test-output" / "summary"
    output.mkdir(parents=True, exist_ok=True)
    source = output / "results.csv"
    rows = [
        {"radius": 0.1, "seed": 0, "method": "IC", "evaluation_method": "IC",
         "test_cert_at_selected": 60.0, "test_cert_at_T": 10.0,
         "status": "completed", "iwau_effective": False},
        {"radius": 0.1, "seed": 1, "method": "IC", "evaluation_method": "IC",
         "test_cert_at_selected": 80.0, "test_cert_at_T": 20.0,
         "status": "completed", "iwau_effective": False},
    ]
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    row = summarize_results(source)[0]
    assert row["test_selected_mean"] == 70.0
    assert row["test_selected_std"] == 10.0
    assert row["test_at_T_std"] == 5.0


def test_divergence_policy_preserves_peak_and_optionally_sets_endpoint_zero():
    output = Path(__file__).parent / ".test-output" / "summary-diverged"
    output.mkdir(parents=True, exist_ok=True)
    source = output / "results.csv"
    rows = [
        {"radius": 0.1, "seed": 0, "method": "IBP", "evaluation_method": "IBP",
         "test_cert_at_selected": 70.0, "test_cert_at_T": "",
         "status": "diverged", "iwau_effective": False},
        {"radius": 0.1, "seed": 1, "method": "IBP", "evaluation_method": "IBP",
         "test_cert_at_selected": 50.0, "test_cert_at_T": 20.0,
         "status": "completed", "iwau_effective": False},
    ]
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    completed_only = summarize_results(source)[0]
    paper_policy = summarize_results(source, diverged_as_zero=True)[0]
    assert completed_only["test_selected_mean"] == 60.0
    assert completed_only["test_at_T_mean"] == 20.0
    assert completed_only["endpoint_seeds"] == 1
    assert paper_policy["test_at_T_mean"] == 10.0
    assert paper_policy["endpoint_seeds"] == 2


def test_console_reports_requested_methods_per_seed_and_in_aggregate(capsys):
    rows = [
        {
            "radius": 0.01,
            "seed": 3,
            "method": "IBP",
            "evaluation_method": "IC",
            "test_cert_at_selected": 60.0,
            "test_cert_at_T": 0.0,
            "test_clean_at_selected": 73.0,
            "test_clean_at_T": 74.0,
            "status": "completed",
            "iwau_status": "not_applicable",
            "iwau_effective": False,
        },
        {
            "radius": 0.01,
            "seed": 3,
            "method": "IC+IWAU",
            "evaluation_method": "IC",
            "test_cert_at_selected": 70.0,
            "test_cert_at_T": 68.0,
            "test_clean_at_selected": 76.0,
            "test_clean_at_T": 77.0,
            "status": "completed",
            "iwau_status": "active",
            "iwau_effective": True,
        },
    ]
    reporter = ConsoleReporter()
    reporter.seed_summary(seed=3, rows=rows)
    reporter.experiment_summary(rows=rows)
    output = capsys.readouterr().out
    assert "[SEED SUMMARY] seed=3" in output
    assert "[EXPERIMENT SUMMARY]" in output
    assert "IBP" in output
    assert "IC+IWAU" in output
    assert "70.00+/- 0.00%" in output
