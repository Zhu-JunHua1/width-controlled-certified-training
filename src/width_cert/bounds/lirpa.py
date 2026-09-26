"""auto_LiRPA adapters for interval parameters and full gradient graphs."""

from __future__ import annotations

import torch
from auto_LiRPA import BoundedTensor, PerturbationLpNorm

from .methods import BoundMethod, IntersectionDiagnostics, intersect_bounds


def bounded_tensor(lower: torch.Tensor, upper: torch.Tensor) -> BoundedTensor:
    center = (lower + upper) / 2.0
    perturbation = PerturbationLpNorm(norm=float("inf"), x_L=lower, x_U=upper)
    return BoundedTensor(center, perturbation)


def parameter_slices(dimensions: list[int]) -> list[int]:
    boundaries = [0]
    for index in range(len(dimensions) - 1):
        boundaries.append(boundaries[-1] + dimensions[index + 1] * dimensions[index])
        boundaries.append(boundaries[-1] + dimensions[index + 1])
    return boundaries


def flatten_parameters(state: dict[str, torch.Tensor], linear_layers: int):
    lower, upper = [], []
    for index in range(1, linear_layers + 1):
        lower.extend((state[f"w{index}_L"].flatten(), state[f"b{index}_L"].flatten()))
        upper.extend((state[f"w{index}_U"].flatten(), state[f"b{index}_U"].flatten()))
    return torch.cat(lower), torch.cat(upper)


def bounded_parameters(flat_lower, flat_upper, dimensions: list[int]):
    boundaries = parameter_slices(dimensions)
    tensors, cursor = [], 0
    for index in range(len(dimensions) - 1):
        output_dim, input_dim = dimensions[index + 1], dimensions[index]
        start, end = boundaries[cursor], boundaries[cursor + 1]
        cursor += 1
        tensors.append(
            bounded_tensor(
                flat_lower[start:end].reshape(1, output_dim, input_dim),
                flat_upper[start:end].reshape(1, output_dim, input_dim),
            )
        )
        start, end = boundaries[cursor], boundaries[cursor + 1]
        cursor += 1
        tensors.append(
            bounded_tensor(
                flat_lower[start:end].reshape(1, 1, output_dim),
                flat_upper[start:end].reshape(1, 1, output_dim),
            )
        )
    return tuple(tensors)


def initialize_interval_state(model) -> dict[str, torch.Tensor]:
    state = {}
    for index, layer in enumerate(model.layers, start=1):
        weight = layer.weight.detach().clone()
        bias = layer.bias.detach().clone()
        state[f"w{index}_L"] = weight.clone()
        state[f"w{index}_U"] = weight.clone()
        state[f"b{index}_L"] = bias.clone()
        state[f"b{index}_U"] = bias.clone()
    return state


def apply_flat_interval_update(
    state: dict[str, torch.Tensor],
    gradient_lower: torch.Tensor,
    gradient_upper: torch.Tensor,
    step: float,
    dimensions: list[int],
) -> None:
    boundaries = parameter_slices(dimensions)
    cursor = 0
    for index in range(len(dimensions) - 1):
        output_dim, input_dim = dimensions[index + 1], dimensions[index]
        start, end = boundaries[cursor], boundaries[cursor + 1]
        cursor += 1
        state[f"w{index + 1}_L"] -= step * gradient_upper[start:end].reshape(output_dim, input_dim)
        state[f"w{index + 1}_U"] -= step * gradient_lower[start:end].reshape(output_dim, input_dim)
        start, end = boundaries[cursor], boundaries[cursor + 1]
        cursor += 1
        state[f"b{index + 1}_L"] -= step * gradient_upper[start:end].reshape(output_dim)
        state[f"b{index + 1}_U"] -= step * gradient_lower[start:end].reshape(output_dim)


def build_trace_inputs(dimensions: list[int], device):
    linear_layers = len(dimensions) - 1
    input_value = torch.zeros(1, 1, dimensions[0], device=device)
    gradient_inputs, evaluation_inputs = [input_value], [input_value]
    for index in range(linear_layers):
        output_dim, input_dim = dimensions[index + 1], dimensions[index]
        weight = torch.zeros(1, output_dim, input_dim, device=device)
        bias = torch.zeros(1, 1, output_dim, device=device)
        gradient_inputs.extend((weight, bias))
        evaluation_inputs.extend((weight, bias))
    gradient_inputs.append(torch.zeros(1, dimensions[-1], device=device))
    return tuple(gradient_inputs), tuple(evaluation_inputs)


def compute_gradient_bounds(
    bounded_graph,
    bounded_inputs,
    targets,
    flat_lower,
    flat_upper,
    dimensions: list[int],
    method: BoundMethod,
    diagnostics: IntersectionDiagnostics | None = None,
):
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)
    inputs = (
        bounded_inputs,
        *bounded_parameters(flat_lower, flat_upper, dimensions),
        targets,
    )
    ibp_lower, ibp_upper = bounded_graph.compute_bounds(x=inputs, method="IBP")
    if method is BoundMethod.IBP:
        return ibp_lower, ibp_upper
    crown_lower, crown_upper = bounded_graph.compute_bounds(x=inputs, method="CROWN")
    return intersect_bounds(
        ibp_lower,
        ibp_upper,
        crown_lower,
        crown_upper,
        diagnostics=diagnostics,
    )
