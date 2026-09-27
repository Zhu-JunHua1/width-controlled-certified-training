"""Validation of frozen-study locks before final test access."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from ..data.manifests import canonical_json_sha256


def _is_unresolved(value: Any) -> bool:
    if value is None or value == "" or value == "TO_BE_FROZEN":
        return True
    if isinstance(value, dict):
        return not value or any(_is_unresolved(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return not value or any(_is_unresolved(item) for item in value)
    return False


def experiment_config_sha256(config: Any) -> str:
    """Hash the complete resolved configuration used by one protocol."""

    encoded = json.dumps(
        asdict(config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_mapping_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_frozen_study_lock(
    path: str | Path,
    *,
    protocol_name: str,
    config_sha256: str | None = None,
    split_manifest: str | Path | None = None,
) -> dict[str, Any]:
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
    required_sections = ("data", "selection", "endpoints", "evaluation", "update_budget")
    for section in required_sections:
        if _is_unresolved(payload.get(section)):
            raise ValueError(f"frozen study lock has unresolved {section}")

    protocols = payload.get("protocols")
    if not isinstance(protocols, dict) or _is_unresolved(protocols):
        raise ValueError("frozen study lock must map protocols to frozen metadata")
    if protocol_name not in protocols:
        raise ValueError(
            f"protocol {protocol_name!r} is not authorized by frozen study lock"
        )
    protocol = protocols[protocol_name]
    if not isinstance(protocol, dict) or _is_unresolved(protocol.get("config_sha256")):
        raise ValueError(f"protocol {protocol_name!r} has no frozen config hash")
    if config_sha256 is None or protocol["config_sha256"] != config_sha256:
        raise ValueError(f"protocol {protocol_name!r} does not match its frozen config hash")
    expected_manifest_hash = protocol.get("split_manifest_sha256")
    if split_manifest is not None:
        if _is_unresolved(expected_manifest_hash):
            raise ValueError(
                f"protocol {protocol_name!r} has no frozen split-manifest hash"
            )
        if canonical_json_sha256(split_manifest) != expected_manifest_hash:
            raise ValueError(
                f"protocol {protocol_name!r} split manifest does not match its frozen hash"
            )
    elif expected_manifest_hash not in (None, "not_applicable"):
        raise ValueError(
            f"protocol {protocol_name!r} requires a split manifest that was not configured"
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
