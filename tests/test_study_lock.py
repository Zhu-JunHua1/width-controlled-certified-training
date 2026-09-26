"""Fail-closed checks for final-test access."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from width_cert.experiments.study_lock import verify_frozen_study_lock


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
                "statistics": {
                    "hypotheses": ["H1"],
                    "sidedness": "one-sided",
                    "multiple_comparison_family": "primary",
                    "correction": "Holm",
                },
                "protocols": [protocol],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_complete_frozen_lock_authorizes_only_enumerated_protocol():
    path = _write_lock()
    assert verify_frozen_study_lock(path, protocol_name="final_protocol")["status"] == "frozen"
    with pytest.raises(ValueError, match="not authorized"):
        verify_frozen_study_lock(path, protocol_name="unlisted")


@pytest.mark.parametrize(
    ("status", "unresolved", "message"),
    (("draft", False, "status=frozen"), ("frozen", True, "unresolved")),
)
def test_incomplete_or_draft_lock_cannot_open_final_test(status, unresolved, message):
    path = _write_lock(status=status, unresolved=unresolved)
    with pytest.raises(ValueError, match=message):
        verify_frozen_study_lock(path, protocol_name="final_protocol")
