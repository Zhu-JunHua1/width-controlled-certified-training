"""Explicit full backward graphs whose outputs are per-example parameter gradients."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..bounds.operators import CustomReLU, StepActivation


class _GradientGraphBase(nn.Module):
    def __init__(self, linear_layers: int):
        super().__init__()
        self.linear_layers = linear_layers
        self.activations = nn.ModuleList(CustomReLU() for _ in range(linear_layers - 1))
        self.derivatives = nn.ModuleList(StepActivation() for _ in range(linear_layers - 1))

    def _forward_activations(self, inputs):
        value, target = inputs[0], inputs[-1]
        parameters = inputs[1:-1]
        weights = [parameters[2 * index] for index in range(self.linear_layers)]
        biases = [parameters[2 * index + 1] for index in range(self.linear_layers)]
        activations, preactivations = [value], []
        for index in range(self.linear_layers):
            current = activations[index] @ weights[index].transpose(-1, -2) + biases[index]
            preactivations.append(current)
            if index < self.linear_layers - 1:
                activations.append(self.activations[index](current))
        return value, target, weights, activations, preactivations

    def _flatten_parameter_gradients(self, input_value, gradients, activations):
        parts = []
        for index in range(self.linear_layers):
            parts.append(
                (gradients[index].transpose(-1, -2) @ activations[index]).reshape(
                    input_value.shape[0], -1
                )
            )
            parts.append(gradients[index].reshape(input_value.shape[0], -1))
        return torch.cat(parts, dim=-1)


class BinaryBCEGradientGraph(_GradientGraphBase):
    """Full gradient graph for binary or one-vs-all BCE with logits."""

    def forward(self, *inputs):
        value, target, weights, activations, preactivations = self._forward_activations(inputs)
        gradients = [None] * self.linear_layers
        gradients[-1] = torch.sigmoid(preactivations[-1]) - target.unsqueeze(1)
        for index in range(self.linear_layers - 2, -1, -1):
            upstream = gradients[index + 1] @ weights[index + 1]
            gradients[index] = upstream * self.derivatives[index](preactivations[index])
        return self._flatten_parameter_gradients(value, gradients, activations)


class BinaryHingeGradientGraph(_GradientGraphBase):
    """Full gradient graph for binary hinge loss ``max(0, 1-yz)``."""

    def __init__(self, linear_layers: int):
        super().__init__(linear_layers)
        self.margin_step = StepActivation()

    def forward(self, *inputs):
        value, target, weights, activations, preactivations = self._forward_activations(inputs)
        label = 2.0 * target.unsqueeze(1) - 1.0
        margin = 1.0 - label * preactivations[-1]
        gradients = [None] * self.linear_layers
        gradients[-1] = -label * self.margin_step(margin)
        for index in range(self.linear_layers - 2, -1, -1):
            upstream = gradients[index + 1] @ weights[index + 1]
            gradients[index] = upstream * self.derivatives[index](preactivations[index])
        return self._flatten_parameter_gradients(value, gradients, activations)

