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


def test_audit_bounds_cli_passes_explicit_protocol_arguments(monkeypatch):
    observed = {}
    output = Path(__file__).parent / ".test-output" / "audit-cli"

    def fake_audit(config, checkpoint, output_dir, **kwargs):
        observed.update(
            config=config,
            checkpoint=checkpoint,
            output_dir=output_dir,
            **kwargs,
        )
        return output / "audit.csv", output / "audit.json"

    monkeypatch.setattr(cli, "audit_same_box", fake_audit)
    root = Path(__file__).resolve().parents[1]
    config = root / "configs" / "examples" / "example_twomoons.yaml"
    assert (
        cli.main(
            [
                "audit-bounds",
                str(config),
                "checkpoint.pt",
                str(output),
                "--training-seed",
                "3",
                "--input-radius",
                "0.01",
                "--batch-size",
                "4",
                "--parameter-samples",
                "7",
            ]
        )
        == 0
    )
    assert observed["training_seed"] == 3
    assert observed["input_radius"] == 0.01
    assert observed["batch_size"] == 4
    assert observed["parameter_samples"] == 7


def test_validate_small_bound_cli_writes_reports(monkeypatch, capsys, tmp_path):
    observed = {}

    def fake_validate(output_dir, *, tolerance):
        observed["output_dir"] = output_dir
        observed["tolerance"] = tolerance
        return tmp_path / "grid.csv", tmp_path / "grid.json"

    monkeypatch.setattr(cli, "validate_small_network_grid", fake_validate)
    assert cli.main(["validate-small-bound", str(tmp_path), "--tolerance", "1e-6"]) == 0
    assert observed == {"output_dir": str(tmp_path), "tolerance": 1e-6}
    terminal = capsys.readouterr().out
    assert "implementation check, not a proof" in terminal
