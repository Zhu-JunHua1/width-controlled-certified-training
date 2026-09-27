"""Configuration-driven experiment execution."""

from .config import ExperimentConfig, SeedSelectionConfig, load_config, load_configs
from .study_lock import experiment_config_sha256, verify_frozen_study_lock


def run_experiment(config):
    """Load the auto_LiRPA runner only when execution is requested."""

    from .runner import run_experiment as _run_experiment

    return _run_experiment(config)


def select_seeds(config, output_dir=None):
    """Load the runner only for an explicit shared-start selection request."""

    from .runner import select_seeds as _select_seeds

    return _select_seeds(config, output_dir)

__all__ = [
    "ExperimentConfig",
    "SeedSelectionConfig",
    "load_config",
    "load_configs",
    "run_experiment",
    "select_seeds",
    "experiment_config_sha256",
    "verify_frozen_study_lock",
]
