"""Protocol configuration and dataset-construction tests."""

from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from width_cert.data import build_dataset
from width_cert.experiments import load_config, load_configs
from width_cert.experiments.environment import environment_metadata
from width_cert.logging.records import SCHEMA_VERSION


ROOT = Path(__file__).resolve().parents[1]


def test_p4_encodes_clean_evaluation_and_combined_split():
    config = load_config(ROOT / "configs" / "paper" / "P4_twomoons_baseline.yaml")
    assert config.training.base_step == 2.0
    assert config.evaluation.input_domain == "clean"
    assert config.evaluation.radius(0.01) == 0.0
    assert config.data.strategy == "combined_split"
    assert config.data.per_training_seed is True
    assert config.seed_selection.candidates() == tuple(range(10))
    assert config.seed_selection.count == 10


def test_p3_sweep_has_all_axis_variants():
    configs = load_configs(ROOT / "configs" / "paper" / "P3_twomoons_sensitivity.yaml")
    assert len(configs) == 13
    assert {config.name for config in configs} >= {"P3_depth_L4", "P3_width_80", "P3_step_10"}
    assert all(config.pretraining.target_accuracy == 0.50 for config in configs)
    assert all(config.pretraining.minimum_accuracy_gain == 0.002 for config in configs)
    assert all(config.pretraining.maximum_final_accuracy == 0.52 for config in configs)
    assert all(config.radii == (0.001,) for config in configs)
    assert all(config.seed_selection.candidates() == tuple(range(100)) for config in configs)


def test_p7_stability_uses_two_hidden_layers():
    configs = load_configs(ROOT / "configs" / "paper" / "P7_twomoons_stability.yaml")
    assert len(configs) == 4
    assert all(config.model.hidden_layers == 2 for config in configs)
    assert {config.training.horizon for config in configs} == {20, 40, 60, 80}
    assert all(config.radii == (0.001,) for config in configs)


def test_paper_protocols_encode_candidate_pools_instead_of_selected_seed_lists():
    paper = ROOT / "configs" / "paper"
    for path in paper.glob("*.yaml"):
        text = path.read_text(encoding="utf-8")
        assert "seed_selection:" in text
        assert "\nseeds:" not in text
        assert "\n  seeds:" not in text

    p1 = load_config(paper / "P1_twomoons_ablation.yaml")
    p2 = load_config(paper / "P2_mnist17_ablation.yaml")
    l2 = load_config(paper / "P6_mnist17_depth_L2.yaml")
    assert p1.seed_selection.candidates() == tuple(range(25))
    assert p2.seed_selection.candidates() == tuple(range(15))
    assert l2.seed_selection.candidates() == tuple(range(25))
    assert {p1.seed_selection.count, p2.seed_selection.count, l2.seed_selection.count} == {10}
    assert p1.pretraining.optimizer == "sgd"
    assert p2.pretraining.optimizer == "adam"
    assert l2.pretraining.optimizer == "adam"


def test_adaptive_method_can_hide_its_fixed_calibration_dependency():
    config = load_config(ROOT / "configs" / "examples" / "example_twomoons.yaml")
    adaptive_only = replace(config, methods=("IBP", "IC+IWAU"))
    adaptive_only.validate()


def test_controlled_gradient_bound_ablation_methods_are_valid():
    config = load_config(ROOT / "configs" / "examples" / "example_twomoons.yaml")
    replace(
        config,
        methods=("IBP", "AGT-Hybrid", "CROWN", "IC"),
    ).validate()


def test_matched_schedule_requires_a_predeclared_target_step():
    config = load_config(ROOT / "configs" / "examples" / "example_twomoons.yaml")
    candidate = replace(config, methods=("IC+InverseTime",))
    try:
        candidate.validate()
    except ValueError as error:
        assert "schedule_target_final_step" in str(error)
    else:
        raise AssertionError("matched scalar schedules require a frozen target step")


def test_all_fairness_update_arms_are_valid_with_a_frozen_target():
    config = load_config(ROOT / "configs" / "examples" / "example_twomoons.yaml")
    candidate = replace(
        config,
        methods=(
            "IC",
            "IC+InverseTime",
            "IC+Cosine",
            "IC+StepDecay",
            "IC+GlobalIWAU",
            "IC+IWAU",
        ),
        training=replace(config.training, schedule_target_final_step=0.02),
    )
    candidate.validate()


def test_default_protocol_uses_a_common_calibration_certifier_and_dual_test_evaluation():
    config = load_config(ROOT / "configs" / "examples" / "example_twomoons.yaml")
    assert config.evaluation.selection_bound_method == "IC"
    assert config.evaluation.bound_methods == ("IBP", "IC")
    assert config.data.calibration_size == 200


def test_coefficient_scale_cannot_weaken_the_calibrated_guarantee():
    config = load_config(ROOT / "configs" / "examples" / "example_twomoons.yaml")
    config = replace(
        config,
        training=replace(config.training, coefficient_scale=0.75),
    )
    try:
        config.validate()
    except ValueError as error:
        assert "coefficient_scale" in str(error)
    else:
        raise AssertionError("scales below one must not silently weaken lambda_min")


def test_coefficient_scale_at_least_one_is_valid():
    config = load_config(ROOT / "configs" / "examples" / "example_twomoons.yaml")
    replace(config, training=replace(config.training, coefficient_scale=1.25)).validate()


def test_twomoons_combined_split_matches_reference_protocol():
    bundle = build_dataset(
        "twomoons",
        train_size=100,
        calibration_size=100,
        test_size=200,
        seed=3,
        calibration_seed=314159,
        test_seed=42,
        strategy="combined_split",
    )
    inputs, targets = make_moons(n_samples=400, noise=0.1, random_state=3)
    expected_train_x, holdout_x, expected_train_y, holdout_y = train_test_split(
        inputs,
        targets,
        train_size=100,
        test_size=300,
        random_state=314159,
        stratify=targets,
    )
    expected_calibration_x, expected_test_x, expected_calibration_y, expected_test_y = train_test_split(
        holdout_x,
        holdout_y,
        train_size=100,
        test_size=200,
        random_state=42,
        stratify=holdout_y,
    )
    assert np.allclose(bundle.train_inputs.numpy(), expected_train_x)
    assert np.allclose(bundle.calibration_inputs.numpy(), expected_calibration_x)
    assert np.allclose(bundle.test_inputs.numpy(), expected_test_x)
    assert np.allclose(bundle.train_targets.numpy(), expected_train_y)
    assert np.allclose(bundle.calibration_targets.numpy(), expected_calibration_y)
    assert np.allclose(bundle.test_targets.numpy(), expected_test_y)


def test_every_public_config_has_an_explicit_calibration_split():
    for path in (ROOT / "configs").rglob("*.yaml"):
        for config in load_configs(path):
            assert config.data.calibration_size > 0
            assert config.data.calibration_seed != config.data.test_seed


def test_public_environment_metadata_omits_absolute_dependency_path():
    metadata = environment_metadata()
    assert "auto_lirpa_path" not in metadata
    assert metadata["auto_lirpa_module"] in {"auto_LiRPA", None}
    assert "auto_lirpa_python_tree_sha256" in metadata


def test_log_schema_version_reflects_center_accuracy_semantics():
    assert SCHEMA_VERSION == "2.1"
