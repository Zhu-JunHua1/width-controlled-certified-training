"""Binary and multi-class certification criteria."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from ..bounds.lirpa import bounded_parameters, bounded_tensor, flatten_parameters
from ..bounds.methods import BoundMethod, intersect_bounds


@dataclass(frozen=True)
class CertificationMetrics:
    clean_accuracy: float
    certified_accuracy: float


def _center_logits(inputs: torch.Tensor, state, dimensions: list[int]) -> torch.Tensor:
    """Evaluate the exact midpoint-parameter model on ordinary inputs.

    This is intentionally separate from bound propagation.  For nonlinear
    networks, the midpoint of output bounds is generally not the output of the
    model at midpoint parameters.
    """

    value = inputs.reshape(inputs.shape[0], dimensions[0])
    for index in range(1, len(dimensions)):
        weight = (state[f"w{index}_L"] + state[f"w{index}_U"]) / 2.0
        bias = (state[f"b{index}_L"] + state[f"b{index}_U"]) / 2.0
        value = value @ weight.transpose(-1, -2) + bias
        if index < len(dimensions) - 1:
            value = functional.relu(value)
    return value


def evaluate_center_accuracy(data_loader, state, *, dimensions: list[int]) -> float:
    """Return clean accuracy of the exact midpoint-parameter model."""

    classes = dimensions[-1]
    correct = total = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            logits = _center_logits(inputs, state, dimensions)
            if classes == 1:
                predictions = (logits.reshape(-1) > 0).float()
                labels = targets.reshape(-1)
            else:
                predictions = logits.reshape(inputs.shape[0], classes).argmax(dim=1)
                labels = targets.long().reshape(-1)
            correct += int((predictions == labels).sum())
            total += inputs.shape[0]
    return 100.0 * correct / total


def _output_bounds(graph, inputs, method: BoundMethod):
    ibp_lower, ibp_upper = graph.compute_bounds(x=inputs, method="IBP")
    if method is BoundMethod.IBP:
        return ibp_lower, ibp_upper
    crown_lower, crown_upper = graph.compute_bounds(x=inputs, method="CROWN")
    return intersect_bounds(ibp_lower, ibp_upper, crown_lower, crown_upper)


def evaluate_certification(
    evaluation_graph,
    data_loader,
    state,
    *,
    input_radius: float,
    method: BoundMethod,
    dimensions: list[int],
) -> CertificationMetrics:
    input_dim, classes = dimensions[0], dimensions[-1]
    linear_layers = len(dimensions) - 1
    flat_lower, flat_upper = flatten_parameters(state, linear_layers)
    parameters = bounded_parameters(flat_lower, flat_upper, dimensions)
    certified = total = 0
    for inputs, targets in data_loader:
        batch = inputs.shape[0]
        center = inputs.reshape(batch, 1, input_dim)
        bounded_inputs = bounded_tensor(center - input_radius, center + input_radius)
        lower, upper = _output_bounds(evaluation_graph, (bounded_inputs, *parameters), method)
        if classes == 1:
            lower, upper = lower.reshape(-1), upper.reshape(-1)
            labels = targets.reshape(-1)
            certified += int((((labels == 1) & (lower > 0)) | ((labels == 0) & (upper < 0))).sum())
        else:
            lower, upper = lower.reshape(batch, classes), upper.reshape(batch, classes)
            labels = targets.long().reshape(-1)
            indices = torch.arange(batch, device=labels.device)
            other_upper = upper.clone()
            other_upper[indices, labels] = -torch.inf
            certified += int((lower[indices, labels] > other_upper.max(dim=1).values).sum())
        total += batch
    clean = evaluate_center_accuracy(data_loader, state, dimensions=dimensions)
    return CertificationMetrics(clean, 100.0 * certified / total)
