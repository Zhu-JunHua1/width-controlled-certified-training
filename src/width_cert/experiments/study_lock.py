"""Validation of frozen-study locks before final test access."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _is_unresolved(value: Any) -> bool:
    if value is None or value == "" or value == "TO_BE_FROZEN":
        return True
    if isinstance(value, dict):
        return not value or any(_is_unresolved(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return not value or any(_is_unresolved(item) for item in value)
    return False


def verify_frozen_study_lock(path: str | Path, *, protocol_name: str) -> dict[str, Any]:
    """Fail closed unless a complete frozen lock authorizes the protocol."""

    lock_path = Path(path)
    if not lock_path.is_file():
        raise FileNotFoundError(f"study lock not found: {lock_path}")
    payload = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("status") != "frozen":
        raise ValueError("final test access requires a study lock with status=frozen")
    code = payload.get("code")
    if not isinstance(code, dict):
        raise ValueError("frozen study lock is missing code provenance")
    for field in ("git_commit", "dependency_lock_sha256", "auto_lirpa_source_sha256"):
        if _is_unresolved(code.get(field)):
            raise ValueError(f"frozen study lock has unresolved code.{field}")
    protocols = payload.get("protocols")
    if isinstance(protocols, dict):
        allowed = set(protocols)
    elif isinstance(protocols, list):
        allowed = set(protocols)
    else:
        raise ValueError("frozen study lock must enumerate protocols")
    if protocol_name not in allowed:
        raise ValueError(
            f"protocol {protocol_name!r} is not authorized by frozen study lock"
        )
    statistics = payload.get("statistics")
    required_statistics = (
        "hypotheses",
        "sidedness",
        "multiple_comparison_family",
        "correction",
    )
    if not isinstance(statistics, dict) or any(
        _is_unresolved(statistics.get(field)) for field in required_statistics
    ):
        raise ValueError("frozen study lock has unresolved statistical declarations")
    return payload
