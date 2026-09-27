from dataclasses import replace
import json
from pathlib import Path

from width_cert.experiments import load_config, select_common_seeds, select_seeds
from width_cert.experiments.config import ModelConfig, SeedSelectionConfig


def test_common_seed_manifest_is_consumed_and_revalidated(tmp_path):
    root = Path(__file__).resolve().parents[1]
    base = load_config(root / "configs" / "examples" / "example_twomoons.yaml")
    selection = SeedSelectionConfig(start=0, stop=3, count=2)
    pretraining = replace(base.pretraining, enabled=False)
    first = replace(
        base,
        name="common_L1",
        model=ModelConfig(hidden_layers=1, hidden_width=4),
        pretraining=pretraining,
        seed_selection=selection,
        echo_epochs=False,
    )
    second = replace(
        base,
        name="common_L2",
        model=ModelConfig(hidden_layers=2, hidden_width=4),
        pretraining=pretraining,
        seed_selection=selection,
        echo_epochs=False,
    )
    manifest = select_common_seeds(
        [first, second],
        tmp_path / "common.json",
        group="depth",
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["groups"]["depth"]["selected_seeds"] == [0, 1]
    assert payload["groups"]["depth"]["test_metrics_read"] is False

    selected_config = replace(
        first,
        seed_selection=replace(
            selection,
            manifest=str(manifest),
            group="depth",
        ),
    )
    audit = select_seeds(selected_config, tmp_path / "revalidated")
    audit_payload = json.loads(audit.read_text(encoding="utf-8"))
    assert audit_payload["selected_seeds"] == [0, 1]
    assert audit_payload["selection_manifest"]["group"] == "depth"
    assert len(audit_payload["selection_manifest"]["sha256"]) == 64
