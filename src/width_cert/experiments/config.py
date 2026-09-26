"""Validated configuration schema for the unified experiment entry point."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    dataset: str
    train_size: int
    calibration_size: int
    test_size: int
    seed: int = 10123310
    calibration_seed: int = 314159
    test_seed: int = 42
    per_training_seed: bool = False
    strategy: str = "separate"


@dataclass(frozen=True)
class ModelConfig:
    hidden_layers: int
    hidden_width: int


@dataclass(frozen=True)
class PretrainingConfig:
    enabled: bool = True
    target_accuracy: float = 0.5
    maximum_initial_accuracy: float = 1.0
    maximum_final_accuracy: float = 1.0
    step_size: float = 0.02
    maximum_epochs: int = 8000
    minimum_epochs: int = 0
    minimum_accuracy_gain: float = 0.0
    optimizer: str = "sgd"


@dataclass(frozen=True)
class TrainingConfig:
    horizon: int
    base_step: float
    retention: float = 0.8
    coefficient_scale: float = 1.0
    gradient_batch_size: int = 4
    max_gradient_width: float = 50.0


@dataclass(frozen=True)
class EvaluationConfig:
    input_domain: str = "same_as_training"
    explicit_radius: float | None = None
    selection_bound_method: str = "IC"
    bound_methods: tuple[str, ...] = ("IBP", "IC")

    def radius(self, training_radius: float) -> float:
        if self.input_domain == "same_as_training":
            return training_radius
        if self.input_domain == "clean":
            return 0.0
        if self.input_domain == "explicit" and self.explicit_radius is not None:
            return float(self.explicit_radius)
        raise ValueError(
            "evaluation.input_domain must be same_as_training, clean, or explicit "
            "(with explicit_radius)"
        )

    def validate(self) -> None:
        allowed = {"IBP", "IC"}
        if self.selection_bound_method not in allowed:
            raise ValueError("evaluation.selection_bound_method must be IBP or IC")
        if not self.bound_methods or set(self.bound_methods) - allowed:
            raise ValueError("evaluation.bound_methods must be a non-empty subset of [IBP, IC]")
        if len(set(self.bound_methods)) != len(self.bound_methods):
            raise ValueError("evaluation.bound_methods must not contain duplicates")


@dataclass(frozen=True)
class SeedSelectionConfig:
    """Deterministic candidate pool and number of accepted shared starts.

    ``stop`` is exclusive, following Python's ``range`` convention.  Candidates
    are examined in ascending order and the first ``count`` starts satisfying
    the pretraining protocol are retained before any certified method is run.
    """

    start: int
    stop: int
    count: int

    def candidates(self) -> tuple[int, ...]:
        return tuple(range(self.start, self.stop))


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    output_dir: str
    data: DataConfig
    model: ModelConfig
    pretraining: PretrainingConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    radii: tuple[float, ...]
    seed_selection: SeedSelectionConfig
    methods: tuple[str, ...] = field(
        default_factory=lambda: ("IBP", "IC", "IBP+IWAU", "IC+IWAU")
    )
    echo_epochs: bool = True

    def validate(self) -> None:
        bases = {"IBP", "AGT-Hybrid", "CROWN", "IC"}
        allowed = bases | {f"{base}+IWAU" for base in bases}
        unknown = set(self.methods) - allowed
        if unknown:
            raise ValueError(f"unknown methods: {sorted(unknown)}")
        if len(set(self.methods)) != len(self.methods):
            raise ValueError("methods must not contain duplicates")
        if self.training.horizon <= 0 or self.training.base_step <= 0:
            raise ValueError("training horizon and base step must be positive")
        if self.training.coefficient_scale < 1.0:
            raise ValueError(
                "training.coefficient_scale must be at least 1.0 to retain "
                "the finite-horizon width-control guarantee"
            )
        if self.training.max_gradient_width <= 0:
            raise ValueError("training.max_gradient_width must be positive")
        if min(
            self.data.train_size,
            self.data.calibration_size,
            self.data.test_size,
        ) <= 0:
            raise ValueError(
                "data train_size, calibration_size, and test_size must all be positive"
            )
        if self.pretraining.optimizer not in {"sgd", "adam"}:
            raise ValueError("pretraining.optimizer must be 'sgd' or 'adam'")
        if not 0 <= self.pretraining.minimum_epochs <= self.pretraining.maximum_epochs:
            raise ValueError(
                "pretraining.minimum_epochs must be between zero and maximum_epochs"
            )
        if self.pretraining.minimum_accuracy_gain < 0:
            raise ValueError("pretraining.minimum_accuracy_gain must be non-negative")
        if not self.pretraining.target_accuracy <= self.pretraining.maximum_final_accuracy <= 1.0:
            raise ValueError(
                "pretraining.maximum_final_accuracy must be between target_accuracy and one"
            )
        if not self.radii or any(radius < 0 for radius in self.radii):
            raise ValueError("radii must be a non-empty sequence of non-negative values")
        candidates = self.seed_selection.candidates()
        if not candidates:
            raise ValueError("seed_selection must define a non-empty candidate range")
        if self.seed_selection.count <= 0:
            raise ValueError("seed_selection.count must be positive")
        if self.seed_selection.count > len(candidates):
            raise ValueError("seed_selection.count exceeds the candidate-pool size")
        # An adaptive arm needs a same-bound fixed pre-run for calibration, but
        # the runner treats that pre-run as an internal dependency when the
        # corresponding fixed arm is not requested for reporting.
        self.evaluation.radius(self.radii[0])
        self.evaluation.validate()


def _construct(data: dict[str, Any]) -> ExperimentConfig:
    evaluation_data = dict(data.get("evaluation", {}))
    if "bound_methods" in evaluation_data:
        evaluation_data["bound_methods"] = tuple(evaluation_data["bound_methods"])
    config = ExperimentConfig(
        name=data["name"],
        output_dir=data["output_dir"],
        data=DataConfig(**data["data"]),
        model=ModelConfig(**data["model"]),
        pretraining=PretrainingConfig(**data.get("pretraining", {})),
        training=TrainingConfig(**data["training"]),
        evaluation=EvaluationConfig(**evaluation_data),
        radii=tuple(float(value) for value in data["radii"]),
        seed_selection=SeedSelectionConfig(**data["seed_selection"]),
        methods=tuple(data.get("methods", ("IBP", "IC", "IBP+IWAU", "IC+IWAU"))),
        echo_epochs=bool(data.get("echo_epochs", True)),
    )
    config.validate()
    return config


def load_config(path: str | Path) -> ExperimentConfig:
    configs = load_configs(path)
    if len(configs) != 1:
        raise ValueError(f"expected one experiment, found {len(configs)} variants")
    return configs[0]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_configs(path: str | Path) -> list[ExperimentConfig]:
    """Load one experiment or a base-plus-variants sweep file."""

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be a mapping")
    if "variants" not in payload:
        return [_construct(payload)]
    base = payload.get("base")
    variants = payload.get("variants")
    if not isinstance(base, dict) or not isinstance(variants, list) or not variants:
        raise ValueError("sweep configuration requires a base mapping and non-empty variants list")
    configs = []
    for variant in variants:
        if not isinstance(variant, dict) or "overrides" not in variant:
            raise ValueError("each sweep variant requires an overrides mapping")
        configs.append(_construct(_deep_merge(base, variant["overrides"])))
    return configs
