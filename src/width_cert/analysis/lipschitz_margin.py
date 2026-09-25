"""Sampled Lipschitz--margin diagnostic for binary checkpoints.

This module estimates suprema by finite sampling inside a parameter box.
Consequently its reported coverage is an optimistic diagnostic, not a sound
certificate.  The explicit ``bound_type`` field is part of the output
contract and prevents the estimate from being confused with a certified
upper bound on the Lipschitz constants.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from ..bounds.lirpa import flatten_parameters
from ..data import build_dataset
from ..experiments.config import ExperimentConfig
from ..logging.checkpoints import load_checkpoint
from ..models import layer_dimensions


def _forward_flat(theta: torch.Tensor, inputs: torch.Tensor, dimensions: list[int]) -> torch.Tensor:
    value = inputs.reshape(inputs.shape[0], dimensions[0])
    offset = 0
    for index in range(len(dimensions) - 1):
        input_dim, output_dim = dimensions[index], dimensions[index + 1]
        count = input_dim * output_dim
        weight = theta[offset : offset + count].reshape(output_dim, input_dim)
        offset += count
        bias = theta[offset : offset + output_dim]
        offset += output_dim
        value = value @ weight.transpose(-1, -2) + bias
        if index < len(dimensions) - 2:
            value = torch.relu(value)
    return value.reshape(-1)


def sampled_lipschitz_margin(
    *,
    state,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    dimensions: list[int],
    input_radius: float,
    parameter_samples: int,
    sampling_seed: int = 0,
) -> dict[str, object]:
    """Estimate the binary margin criterion by parameter-box sampling."""

    if dimensions[-1] != 1:
        raise ValueError("the sampled Lipschitz-margin diagnostic currently supports binary models")
    if parameter_samples <= 0:
        raise ValueError("parameter_samples must be positive")

    lower, upper = flatten_parameters(state, len(dimensions) - 1)
    center = (lower + upper) / 2.0
    width = upper - lower
    max_parameter_gradient = torch.zeros_like(center)
    max_input_l1_gradient = torch.tensor(0.0, device=center.device)
    generator = torch.Generator(device=center.device).manual_seed(sampling_seed)

    with torch.enable_grad():
        for sample_index in range(parameter_samples):
            if sample_index == 0:
                theta_sample = center
            else:
                draw = torch.rand(
                    center.shape,
                    generator=generator,
                    device=center.device,
                    dtype=center.dtype,
                )
                theta_sample = lower + draw * width
            for row in range(inputs.shape[0]):
                sample_inputs = inputs[row : row + 1].clone().requires_grad_(True)
                sample_theta = theta_sample.clone().requires_grad_(True)
                output = _forward_flat(sample_theta, sample_inputs, dimensions).sum()
                parameter_gradient, input_gradient = torch.autograd.grad(
                    output, (sample_theta, sample_inputs)
                )
                max_parameter_gradient = torch.maximum(
                    max_parameter_gradient, parameter_gradient.abs()
                )
                max_input_l1_gradient = torch.maximum(
                    max_input_l1_gradient, input_gradient.abs().sum()
                )

    with torch.no_grad():
        center_logits = _forward_flat(center, inputs, dimensions)
        margins = (2.0 * targets.reshape(-1) - 1.0) * center_logits
        parameter_term = 0.5 * (max_parameter_gradient * width).sum()
        input_term = max_input_l1_gradient * input_radius
        threshold = parameter_term + input_term
        coverage = 100.0 * (margins > threshold).float().mean()

    return {
        "schema_version": "1.0",
        "bound_type": "sampled_estimate",
        "sound_certificate": False,
        "parameter_samples": parameter_samples,
        "test_points": int(inputs.shape[0]),
        "input_radius": float(input_radius),
        "parameter_term": float(parameter_term),
        "input_term": float(input_term),
        "threshold": float(threshold),
        "coverage_percent": float(coverage),
        "median_center_margin": float(margins.median()),
        "maximum_parameter_gradient": float(max_parameter_gradient.max()),
        "maximum_input_l1_gradient": float(max_input_l1_gradient),
        "interpretation": (
            "Finite box sampling may underestimate the true suprema; coverage can therefore "
            "only overstate the coverage obtained with certified Lipschitz upper bounds."
        ),
        "margins": [float(value) for value in margins.detach().cpu()],
    }


def analyze_checkpoint(
    config: ExperimentConfig,
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    training_seed: int,
    input_radius: float,
    parameter_samples: int,
    sampling_seed: int = 0,
) -> tuple[Path, Path]:
    """Run the sampled diagnostic and write JSON plus a margin CDF CSV."""

    data_seed = training_seed if config.data.per_training_seed else config.data.seed
    bundle = build_dataset(
        config.data.dataset,
        train_size=config.data.train_size,
        test_size=config.data.test_size,
        seed=data_seed,
        strategy=config.data.strategy,
    )
    if bundle.output_dim != 1:
        raise ValueError("the sampled Lipschitz-margin diagnostic currently supports binary data")
    epoch, state = load_checkpoint(checkpoint)
    dimensions = layer_dimensions(
        bundle.input_dim,
        config.model.hidden_width,
        bundle.output_dim,
        config.model.hidden_layers,
    )
    device = next(iter(state.values())).device
    report = sampled_lipschitz_margin(
        state=state,
        inputs=bundle.test_inputs.to(device),
        targets=bundle.test_targets.to(device),
        dimensions=dimensions,
        input_radius=input_radius,
        parameter_samples=parameter_samples,
        sampling_seed=sampling_seed,
    )
    margins = sorted(report.pop("margins"))
    report.update(
        {
            "checkpoint": Path(checkpoint).name,
            "checkpoint_epoch": epoch,
            "experiment": config.name,
            "training_seed": training_seed,
        }
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / "lipschitz_margin.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    cdf_path = destination / "margin_cdf.csv"
    with cdf_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["center_margin", "empirical_cdf"])
        writer.writeheader()
        total = len(margins)
        for index, margin in enumerate(margins, start=1):
            writer.writerow({"center_margin": margin, "empirical_cdf": index / total})
    return report_path, cdf_path
