"""auto_LiRPA operators used by the full gradient computation graph."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional
from auto_LiRPA.bound_ops import BoundActivation
from auto_LiRPA.operators.relu import BoundRelu


class StepFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return (value > 0).float()

    @staticmethod
    def backward(ctx, gradient):
        return gradient

    @staticmethod
    def symbolic(graph, value):
        return graph.op("StepActivation", value)


class StepActivation(nn.Module):
    def forward(self, value):
        return StepFunction.apply(value)


class BoundStepActivation(BoundActivation):
    """Sound box relaxation of the ReLU derivative (step) node."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.split_status = None

    def forward(self, value):
        return (value > 0).float()

    def _apply_split(self, lower, upper):
        if self.split_status is None:
            return lower, upper
        upper = torch.where(
            self.split_status == -1,
            torch.minimum(upper, torch.zeros_like(upper)),
            upper,
        )
        lower = torch.where(
            self.split_status == 1,
            torch.maximum(lower, torch.zeros_like(lower)),
            lower,
        )
        return lower, upper

    def interval_propagate(self, *values):
        lower, upper = self._apply_split(*values[0])
        out_lower = torch.where(lower > 0, torch.ones_like(lower), torch.zeros_like(lower))
        out_upper = torch.where(upper <= 0, torch.zeros_like(upper), torch.ones_like(upper))
        return out_lower, out_upper

    def bound_relax(self, value, init=False, **kwargs):
        original_lower, original_upper = value.lower.clone(), value.upper.clone()
        lower, upper = self._apply_split(value.lower, value.upper)
        slope_lower = torch.zeros_like(lower)
        slope_upper = torch.zeros_like(lower)
        bias_lower = torch.where(lower > 0, torch.ones_like(lower), torch.zeros_like(lower))
        bias_upper = torch.where(upper <= 0, torch.zeros_like(lower), torch.ones_like(lower))
        self.lw, self.uw, self.lb, self.ub = (
            slope_lower,
            slope_upper,
            bias_lower,
            bias_upper,
        )
        value.lower, value.upper = original_lower, original_upper
        return slope_lower, bias_lower, slope_upper, bias_upper


class CustomReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        ctx.save_for_backward(value)
        return functional.relu(value)

    @staticmethod
    def backward(ctx, gradient):
        (value,) = ctx.saved_tensors
        return gradient * (value > 0).float()

    @staticmethod
    def symbolic(graph, value):
        return graph.op("CustomReLU", value)


class CustomReLU(nn.Module):
    def forward(self, value):
        return CustomReLUFunction.apply(value)


class BoundCustomReLU(BoundRelu):
    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        self.split_status = None

    def interval_propagate(self, *values):
        lower, upper = values[0]
        if self.split_status is not None:
            upper = torch.where(
                self.split_status == -1,
                torch.minimum(upper, torch.zeros_like(upper)),
                upper,
            )
            lower = torch.where(
                self.split_status == 1,
                torch.maximum(lower, torch.zeros_like(lower)),
                lower,
            )
        return super().interval_propagate((lower, upper))


CUSTOM_BOUND_OPERATORS = {
    "StepActivation": BoundStepActivation,
    "onnx::StepActivation": BoundStepActivation,
    "CustomReLU": BoundCustomReLU,
    "onnx::CustomReLU": BoundCustomReLU,
}

