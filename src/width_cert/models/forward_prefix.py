"""Prefix graphs used by the AGT-inspired hybrid gradient bounder."""

from __future__ import annotations

import torch.nn as nn

from ..bounds.operators import CustomReLU


class ForwardPrefixGraph(nn.Module):
    """Return the preactivation of one MLP layer.

    The graph receives only the parameters required by the prefix.  This avoids
    introducing unused bounded inputs into auto_LiRPA's traced graph.
    """

    def __init__(self, target_layer: int):
        super().__init__()
        if target_layer < 0:
            raise ValueError("target_layer must be non-negative")
        self.target_layer = target_layer
        self.activations = nn.ModuleList(CustomReLU() for _ in range(target_layer))

    def forward(self, value, *parameters):
        if len(parameters) != 2 * (self.target_layer + 1):
            raise ValueError("prefix graph received the wrong number of parameters")
        for layer in range(self.target_layer + 1):
            weight = parameters[2 * layer]
            bias = parameters[2 * layer + 1]
            value = value @ weight.transpose(-1, -2) + bias
            if layer < self.target_layer:
                value = self.activations[layer](value)
        return value

