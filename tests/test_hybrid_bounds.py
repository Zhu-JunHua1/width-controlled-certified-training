"""Soundness smoke tests for the AGT-inspired hybrid ablation."""

from __future__ import annotations

import torch

from width_cert.bounds import AGTInspiredHybridBounder
from width_cert.bounds.lirpa import bounded_tensor
from width_cert.models import BinaryBCEGradientGraph


def _unflatten(flat: torch.Tensor):
    return (
        flat[0:6].reshape(1, 3, 2),
        flat[6:9].reshape(1, 1, 3),
        flat[9:12].reshape(1, 1, 3),
        flat[12:13].reshape(1, 1, 1),
    )


def _exact_gradient(inputs: torch.Tensor, targets: torch.Tensor, flat: torch.Tensor):
    return BinaryBCEGradientGraph(2)(inputs, *_unflatten(flat), targets)


def test_hybrid_is_exact_on_a_point_box():
    torch.manual_seed(3)
    dimensions = [2, 3, 1]
    bounder = AGTInspiredHybridBounder(dimensions, torch.device("cpu"))
    inputs = torch.randn(4, 1, 2)
    targets = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
    flat = torch.randn(13) * 0.2
    lower, upper = bounder.compute(
        bounded_inputs=bounded_tensor(inputs, inputs),
        input_lower=inputs,
        input_upper=inputs,
        targets=targets,
        flat_lower=flat,
        flat_upper=flat,
    )
    exact = _exact_gradient(inputs, targets, flat)
    assert torch.allclose(lower, exact, atol=1e-5, rtol=1e-5)
    assert torch.allclose(upper, exact, atol=1e-5, rtol=1e-5)


def test_hybrid_contains_random_samples_from_input_and_parameter_box():
    torch.manual_seed(7)
    dimensions = [2, 3, 1]
    bounder = AGTInspiredHybridBounder(dimensions, torch.device("cpu"))
    input_center = torch.randn(3, 1, 2) * 0.2
    input_radius = 0.04
    input_lower = input_center - input_radius
    input_upper = input_center + input_radius
    parameter_center = torch.randn(13) * 0.15
    parameter_radius = 0.02
    parameter_lower = parameter_center - parameter_radius
    parameter_upper = parameter_center + parameter_radius
    targets = torch.tensor([[0.0], [1.0], [1.0]])

    lower, upper = bounder.compute(
        bounded_inputs=bounded_tensor(input_lower, input_upper),
        input_lower=input_lower,
        input_upper=input_upper,
        targets=targets,
        flat_lower=parameter_lower,
        flat_upper=parameter_upper,
    )
    assert torch.all(lower <= upper)

    for _ in range(128):
        sampled_inputs = input_lower + torch.rand_like(input_lower) * (
            input_upper - input_lower
        )
        sampled_parameters = parameter_lower + torch.rand_like(parameter_lower) * (
            parameter_upper - parameter_lower
        )
        exact = _exact_gradient(sampled_inputs, targets, sampled_parameters)
        assert torch.all(exact >= lower - 1e-5)
        assert torch.all(exact <= upper + 1e-5)

