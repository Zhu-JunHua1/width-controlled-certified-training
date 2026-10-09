"""Tests for path-free repository provenance metadata."""

from __future__ import annotations

import subprocess
from pathlib import Path

from width_cert.experiments.environment import repository_metadata


def test_repository_metadata_records_commit_and_dirty_state(tmp_path: Path):
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "user.name", "Test User"],
        check=True,
    )
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("frozen\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.txt"], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "commit", "-m", "freeze"],
        check=True,
        capture_output=True,
    )
    clean = repository_metadata(tmp_path)
    assert len(str(clean["git_commit"])) == 40
    assert clean["git_dirty"] is False
    tracked.write_text("changed\n", encoding="utf-8")
    assert repository_metadata(tmp_path)["git_dirty"] is True
