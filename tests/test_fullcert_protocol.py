"""Fail-closed checks for the official FullCert comparison candidate."""

import hashlib
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_official_fullcert_candidate_is_sealed_from_test_access():
    path = (
        ROOT
        / "baselines"
        / "fullcert_protocols"
        / "P1_twomoons_official_fullcert_candidate.yaml"
    )
    candidate = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert candidate["status"] == "candidate"
    assert candidate["analysis_tier"] == "development"
    assert candidate["test_access"] == "disabled"
    assert candidate["official_source"]["commit"] == (
        "3387a8af3ab143fae238b5d3a3f5b24fa815cca6"
    )
    adapter = ROOT / candidate["official_source"]["adapter"]
    assert hashlib.sha256(adapter.read_bytes()).hexdigest() == (
        candidate["official_source"]["adapter_sha256"]
    )


def test_official_fullcert_candidate_matches_the_main_protocol_grid():
    path = (
        ROOT
        / "baselines"
        / "fullcert_protocols"
        / "P1_twomoons_official_fullcert_candidate.yaml"
    )
    candidate = yaml.safe_load(path.read_text(encoding="utf-8"))
    manifest_path = ROOT / candidate["starts"]["manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    starts = manifest["groups"][candidate["starts"]["group"]]["selected_seeds"]
    assert starts == candidate["starts"]["expected"]
    assert candidate["model"]["dimensions"] == [2, 20, 20, 1]
    assert candidate["training"]["radii"] == [0.0001, 0.001, 0.01]
    assert candidate["training"]["horizon"] == 40
    assert candidate["training"]["early_stopping"] is False
    assert candidate["training"]["official_api_learning_rate"] == (
        candidate["training"]["full_batch_size"]
        * candidate["training"]["paper_mean_gradient_step"]
    )
    assert candidate["evaluation"]["common_test_certifiers"] == ["IBP", "IC"]
    assert len(starts) * len(candidate["training"]["radii"]) == 30
    assert candidate["evaluation"]["expected_summaries"] == 30
    assert candidate["evaluation"]["expected_evaluator_rows"] == 60
