"""Fail-closed checks for final-test access."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from width_cert.experiments.study_lock import verify_frozen_study_lock
from width_cert.experiments import load_config
from width_cert.experiments.study_lock import experiment_config_sha256


OUTPUT = Path(__file__).parent / ".test-output" / "study-lock"


def _write_lock(*, status="frozen", protocol="final_protocol", unresolved=False):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "lock.yaml"
    value = "TO_BE_FROZEN" if unresolved else "resolved"
    path.write_text(
        yaml.safe_dump(
            {
                "status": status,
                "code": {
                    "git_commit": value,
                    "dependency_lock_sha256": value,
                    "auto_lirpa_source_sha256": value,
                },
                "data": {
                    "split_policy": "explicit_train_calibration_test",
                    "split_hash_manifest": value,
                    "retired_test_seeds": [42],
                    "final_calibration_seeds": [271828],
                    "final_test_seeds": [161803],
                },
                "selection": {
                    "split": "calibration",
                    "checkpoint_rule": "first_argmax_calibration_certified_accuracy",
                    "retention": 0.8,
                    "tie_breaking": "earliest_epoch",
                },
                "endpoints": {
                    "primary": "test_cert_at_calibration_selected_checkpoint",
                    "secondary": "test_cert_at_T",
                },
                "evaluation": {
                    "test_certifiers": ["IBP", "IC"],
                    "require_common_test_samples": True,
                    "require_common_checkpoint_rule": True,
                },
                "update_budget": {
                    "comparison_start": "common_calibration_t_p",
                    "scalar_target_source": "frozen_development_median",
                    "complete_cost_includes_fixed_pre_run": True,
                },
                "statistics": {
                    "hypotheses": ["H1"],
                    "sidedness": "one-sided",
                    "multiple_comparison_family": "primary",
                    "correction": "Holm",
                },
                "protocols": {
                    protocol: {
                        "config_sha256": value,
                        "split_manifest_sha256": "not_applicable",
                        "seed_selection_manifest_sha256": "not_applicable",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_complete_frozen_lock_authorizes_only_enumerated_protocol():
    path = _write_lock()
    assert verify_frozen_study_lock(
        path,
        protocol_name="final_protocol",
        config_sha256="resolved",
    )["status"] == "frozen"
    with pytest.raises(ValueError, match="not authorized"):
        verify_frozen_study_lock(
            path,
            protocol_name="unlisted",
            config_sha256="resolved",
        )


@pytest.mark.parametrize(
    ("status", "unresolved", "message"),
    (("draft", False, "status=frozen"), ("frozen", True, "unresolved")),
)
def test_incomplete_or_draft_lock_cannot_open_final_test(status, unresolved, message):
    path = _write_lock(status=status, unresolved=unresolved)
    with pytest.raises(ValueError, match=message):
        verify_frozen_study_lock(
            path,
            protocol_name="final_protocol",
            config_sha256="resolved",
        )


def test_frozen_lock_rejects_a_changed_resolved_config():
    path = _write_lock()
    with pytest.raises(ValueError, match="config hash"):
        verify_frozen_study_lock(
            path,
            protocol_name="final_protocol",
            config_sha256="changed",
        )


def test_frozen_lock_rejects_a_changed_seed_selection_manifest():
    path = _write_lock()
    lock = yaml.safe_load(path.read_text(encoding="utf-8"))
    manifest = OUTPUT / "seeds.json"
    manifest.write_text('{"schema_version":"1.0","groups":{}}', encoding="utf-8")
    lock["protocols"]["final_protocol"]["seed_selection_manifest_sha256"] = "wrong"
    path.write_text(yaml.safe_dump(lock), encoding="utf-8")
    with pytest.raises(ValueError, match="seed-selection manifest"):
        verify_frozen_study_lock(
            path,
            protocol_name="final_protocol",
            config_sha256="resolved",
            seed_selection_manifest=manifest,
        )


def test_protocol_v5_main_lock_authorizes_only_the_committed_grid():
    root = Path(__file__).resolve().parents[1]
    config = load_config(
        root / "configs" / "protocol_v5_frozen" / "P1_twomoons_multiradius_main.yaml"
    )
    payload = verify_frozen_study_lock(
        root / "study_locks" / "protocol_v5_main_frozen.yaml",
        protocol_name=config.name,
        config_sha256=experiment_config_sha256(config),
        seed_selection_manifest=(
            root / "study_locks" / "candidate_audits" / "protocol_v5_P1_main_seeds.json"
        ),
    )
    assert payload["code"]["git_commit"] == (
        "3dceb438daf9ad04b8eb1b78596958beef428809"
    )
    assert payload["data"]["blind_test_opened"] is False
    assert payload["evaluation"]["expected_summary_rows"] == 120
    assert payload["evaluation"]["expected_evaluator_rows"] == 240
