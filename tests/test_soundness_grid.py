import csv
import json

from width_cert.analysis.soundness_grid import GridCase, validate_small_network_grid


def test_dense_grid_checks_relu_kink_and_all_bound_methods(tmp_path):
    case = GridCase(
        "test_crossing_relu",
        (-0.2, 0.4),
        (0.8, 1.2),
        (-0.1, 0.1),
        (-0.8, 0.9),
        (-0.1, 0.2),
    )
    csv_path, json_path = validate_small_network_grid(tmp_path, cases=(case,))
    report = json.loads(json_path.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["total_violations"] == 0
    assert report["relu_kink_points"] > 0
    assert report["relu_kink_derivative_choices"] == [0.0, 1.0]
    assert report["is_real_arithmetic_soundness_proof"] is False
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["method"] for row in rows} == {
        "IBP",
        "AGT-inspired hybrid",
        "Full-CROWN-box",
        "Full-IC",
        "Hybrid-IC",
    }
    assert all(int(row["violations"]) == 0 for row in rows)
