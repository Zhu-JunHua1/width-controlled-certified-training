"""Architecture definitions without training or logging policy."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as functional

from ..bounds.operators import CustomReLU


def layer_dimensions(input_dim: int, hidden_width: int, output_dim: int, hidden_layers: int) -> list[int]:
    return [input_dim] + [hidden_width] * hidden_layers + [output_dim]


class PretrainingMLP(nn.Module):
    def __init__(self, dimensions: list[int]):
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Linear(dimensions[index], dimensions[index + 1])
            for index in range(len(dimensions) - 1)
        )

    def forward(self, value):
        for index, layer in enumerate(self.layers):
            value = layer(value)
            if index < len(self.layers) - 1:
                value = functional.relu(value)
        return value


class EvaluationGraph(nn.Module):
    def __init__(self, linear_layers: int):
        super().__init__()
        self.linear_layers = linear_layers
        self.activations = nn.ModuleList(
            CustomReLU() for _ in range(linear_layers - 1)
        )

    def forward(self, *inputs):
        value = inputs[0]
        parameters = inputs[1:]
        weights = [parameters[2 * index] for index in range(self.linear_layers)]
        biases = [parameters[2 * index + 1] for index in range(self.linear_layers)]
        for index in range(self.linear_layers):
            value = value @ weights[index].transpose(-1, -2) + biases[index]
            if index < self.linear_layers - 1:
                value = self.activations[index](value)
        return value

