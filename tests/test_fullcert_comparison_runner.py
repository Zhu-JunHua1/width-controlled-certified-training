"""Fail-closed tests for the official FullCert comparison entry point."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _runner():
    path = ROOT / "tools" / "run_fullcert_comparison.py"
    spec = importlib.util.spec_from_file_location("run_fullcert_comparison", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _candidate():
    path = (
        ROOT
        / "baselines"
        / "fullcert_protocols"
        / "P1_twomoons_official_fullcert_candidate.yaml"
    )
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _frozen():
    path = (
        ROOT
        / "baselines"
        / "fullcert_protocols"
        / "P1_twomoons_official_fullcert_frozen.yaml"
    )
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_candidate_passes_comparison_runner_static_validation():
    module = _runner()
    module._validate_config(_candidate(), ROOT)


def test_frozen_config_passes_comparison_runner_static_validation():
    module = _runner()
    module._validate_config(_frozen(), ROOT)


def test_comparison_runner_rejects_normalization_mismatch():
    module = _runner()
    candidate = _candidate()
    candidate["training"]["official_api_learning_rate"] = 0.1
    with pytest.raises(ValueError, match="double mean"):
        module._validate_config(candidate, ROOT)


def test_comparison_runner_rejects_changed_grid():
    module = _runner()
    candidate = _candidate()
    candidate["training"]["radii"] = [0.01]
    with pytest.raises(ValueError, match="radius grid"):
        module._validate_config(candidate, ROOT)


def test_comparison_runner_refuses_existing_output(tmp_path):
    module = _runner()
    config = _candidate()
    config["output_dir"] = str(tmp_path / "occupied")
    occupied = Path(config["output_dir"])
    occupied.mkdir(parents=True)
    (occupied / "sentinel.txt").write_text("do not overwrite", encoding="utf-8")
    config_path = tmp_path / "candidate.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        module.run(config_path, ROOT, ROOT)
