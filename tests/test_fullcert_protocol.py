"""Fail-closed checks for the official FullCert comparison candidate."""

import hashlib
import json
from pathlib import Path

import yaml

from width_cert.experiments.study_lock import (
    canonical_mapping_sha256,
    verify_frozen_study_lock,
)


ROOT = Path(__file__).resolve().parents[1]


def _frozen():
    path = (
        ROOT
        / "baselines"
        / "fullcert_protocols"
        / "P1_twomoons_official_fullcert_frozen.yaml"
    )
    return yaml.safe_load(path.read_text(encoding="utf-8"))


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
    assert candidate["pretraining"] == {
        "enabled": True,
        "target_accuracy": 0.50,
        "maximum_initial_accuracy": 0.505,
        "maximum_final_accuracy": 0.52,
        "step_size": 0.02,
        "maximum_epochs": 4000,
        "minimum_epochs": 0,
        "minimum_accuracy_gain": 0.0,
        "optimizer": "sgd",
    }
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


def test_frozen_fullcert_extension_is_explicitly_posthoc_and_fail_closed():
    config = _frozen()
    assert config["status"] == "frozen"
    assert config["test_access"] == "frozen_final"
    assert config["analysis_tier"] == "posthoc_secondary_matched_external_baseline"
    assert config["data"]["test_set_already_opened_by_protocol_v5"] is True
    assert config["data"]["test_outcomes_used_to_design_this_adapter"] is False
    assert config["output_dir"].startswith("results/protocol_v5_fullcert_final/")
    assert config["study_lock"] == "study_locks/protocol_v5_fullcert_frozen.yaml"


def test_frozen_fullcert_extension_binds_runner_and_adapter_hashes():
    config = _frozen()
    source = config["official_source"]
    for path_key, hash_key in (("adapter", "adapter_sha256"), ("runner", "runner_sha256")):
        path = ROOT / source[path_key]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source[hash_key]


def test_frozen_fullcert_extension_matches_its_metadata_lock():
    config = _frozen()
    config_hash = canonical_mapping_sha256(config)
    payload = verify_frozen_study_lock(
        ROOT / config["study_lock"],
        protocol_name=config["protocol_name"],
        config_sha256=config_hash,
        seed_selection_manifest=ROOT / config["starts"]["manifest"],
    )
    assert payload["code"]["git_commit"] == (
        "290371124e3314cdb42ba02a77e1e72d3370df3c"
    )
    assert payload["data"]["test_set_already_opened_by_protocol_v5"] is True
