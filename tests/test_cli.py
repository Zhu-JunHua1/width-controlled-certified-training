"""Command-line entry-point tests."""

from pathlib import Path

from width_cert import cli
from width_cert.experiments import load_config


def test_run_without_config_uses_workspace_certified_training_example(monkeypatch, capsys):
    observed = []

    def fake_run(config):
        observed.append(config)
        return Path("results/example_twomoons")

    monkeypatch.setattr(cli, "run_experiment", fake_run)
    assert cli.main(["run"]) == 0
    assert len(observed) == 1
    config = observed[0]
    assert config.name == "example_twomoons_certified_training"
    assert config.seed_selection.count == 1
    assert config.pretraining.enabled
    assert config.pretraining.target_accuracy == 0.50
    assert config.training.horizon == 30
    assert config.radii == (0.01,)
    assert config.seed_selection.candidates() == tuple(range(10))
    assert config.methods == ("IBP", "IC+IWAU")
    terminal = capsys.readouterr().out
    assert "workspace TwoMoons example" in terminal
    assert "configuration source:" in terminal
    assert "example_twomoons.yaml" in terminal


def test_packaged_example_matches_editable_workspace_example():
    root = Path(__file__).resolve().parents[1]
    workspace = load_config(root / "configs" / "examples" / "example_twomoons.yaml")
    packaged = load_config(root / "src" / "width_cert" / "configs" / "example_twomoons.yaml")
    assert workspace == packaged
