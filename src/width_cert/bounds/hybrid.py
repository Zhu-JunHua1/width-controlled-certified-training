"""AGT-inspired forward-CROWN/backward-IBP gradient bounds.

This is a controlled ablation, not a reimplementation claim about the AGT
tool.  Forward preactivations use auto_LiRPA CROWN bounds.  The loss derivative,
ReLU derivative boxes, backward matrix products, and parameter-gradient outer
products then use explicit interval arithmetic.
"""

from __future__ import annotations

import torch
from auto_LiRPA import BoundedModule

from .lirpa import bounded_parameters
from .operators import CUSTOM_BOUND_OPERATORS
from ..models.forward_prefix import ForwardPrefixGraph


def interval_matmul(
    left_lower: torch.Tensor,
    left_upper: torch.Tensor,
    right_lower: torch.Tensor,
    right_upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Natural interval extension of batched matrix multiplication."""

    left_l = left_lower.unsqueeze(-1)
    left_u = left_upper.unsqueeze(-1)
    right_l = right_lower.unsqueeze(-3)
    right_u = right_upper.unsqueeze(-3)
    products = torch.stack(
        (
            left_l * right_l,
            left_l * right_u,
            left_u * right_l,
            left_u * right_u,
        ),
        dim=0,
    )
    return products.min(dim=0).values.sum(dim=-2), products.max(dim=0).values.sum(dim=-2)


def interval_multiply(
    left_lower: torch.Tensor,
    left_upper: torch.Tensor,
    right_lower: torch.Tensor,
    right_upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Natural interval extension of element-wise multiplication."""

    products = torch.stack(
        (
            left_lower * right_lower,
            left_lower * right_upper,
            left_upper * right_lower,
            left_upper * right_upper,
        ),
        dim=0,
    )
    return products.min(dim=0).values, products.max(dim=0).values


def _prefix_trace_inputs(dimensions: list[int], target_layer: int, device):
    inputs = [torch.zeros(1, 1, dimensions[0], device=device)]
    for layer in range(target_layer + 1):
        inputs.append(
            torch.zeros(
                1,
                dimensions[layer + 1],
                dimensions[layer],
                device=device,
            )
        )
        inputs.append(torch.zeros(1, 1, dimensions[layer + 1], device=device))
    return tuple(inputs)


class AGTInspiredHybridBounder:
    """Forward CROWN with an explicitly interval-valued backward pass."""

    def __init__(self, dimensions: list[int], device) -> None:
        self.dimensions = list(dimensions)
        self.linear_layers = len(dimensions) - 1
        self.prefix_graphs = []
        for target_layer in range(self.linear_layers):
            graph = BoundedModule(
                ForwardPrefixGraph(target_layer).to(device),
                _prefix_trace_inputs(dimensions, target_layer, device),
                custom_ops=CUSTOM_BOUND_OPERATORS,
                bound_opts={"override_dependent_operations": True},
            )
            self.prefix_graphs.append(graph)

    def compute(
        self,
        *,
        bounded_inputs,
        input_lower: torch.Tensor,
        input_upper: torch.Tensor,
        targets: torch.Tensor,
        flat_lower: torch.Tensor,
        flat_upper: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        parameters = bounded_parameters(flat_lower, flat_upper, self.dimensions)
        preactivation_bounds = []
        for layer, graph in enumerate(self.prefix_graphs):
            preactivation_bounds.append(
                graph.compute_bounds(
                    x=(bounded_inputs, *parameters[: 2 * (layer + 1)]),
                    method="CROWN",
                )
            )

        activation_bounds = [(input_lower, input_upper)]
        for lower, upper in preactivation_bounds[:-1]:
            activation_bounds.append((lower.clamp(min=0.0), upper.clamp(min=0.0)))

        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)
        output_lower, output_upper = preactivation_bounds[-1]
        delta_lower = torch.sigmoid(output_lower) - targets.unsqueeze(1)
        delta_upper = torch.sigmoid(output_upper) - targets.unsqueeze(1)

        gradient_bounds_reversed = []
        for layer in range(self.linear_layers - 1, -1, -1):
            activation_lower, activation_upper = activation_bounds[layer]
            weight_grad_lower, weight_grad_upper = interval_matmul(
                delta_lower.transpose(-1, -2),
                delta_upper.transpose(-1, -2),
                activation_lower,
                activation_upper,
            )
            gradient_bounds_reversed.append(
                (
                    weight_grad_lower.reshape(weight_grad_lower.shape[0], -1),
                    weight_grad_upper.reshape(weight_grad_upper.shape[0], -1),
                    delta_lower.reshape(delta_lower.shape[0], -1),
                    delta_upper.reshape(delta_upper.shape[0], -1),
                )
            )
            if layer == 0:
                continue
            weight_lower = parameters[2 * layer].ptb.x_L
            weight_upper = parameters[2 * layer].ptb.x_U
            delta_lower, delta_upper = interval_matmul(
                delta_lower,
                delta_upper,
                weight_lower,
                weight_upper,
            )
            pre_lower, pre_upper = preactivation_bounds[layer - 1]
            step_lower = (pre_lower > 0).to(delta_lower.dtype)
            step_upper = (pre_upper > 0).to(delta_upper.dtype)
            delta_lower, delta_upper = interval_multiply(
                delta_lower,
                delta_upper,
                step_lower,
                step_upper,
            )

        gradient_bounds_reversed.reverse()
        lower_parts, upper_parts = [], []
        for weight_l, weight_u, bias_l, bias_u in gradient_bounds_reversed:
            lower_parts.extend((weight_l, bias_l))
            upper_parts.extend((weight_u, bias_u))
        return torch.cat(lower_parts, dim=-1), torch.cat(upper_parts, dim=-1)
