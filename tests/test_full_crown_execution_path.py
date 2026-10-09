"""Regression checks for the implemented full-gradient CROWN path.

These tests show which code paths execute. They are not a real-arithmetic
soundness proof and do not establish directed-rounding guarantees.
"""

from collections import Counter
from contextlib import ExitStack
from unittest.mock import patch

import torch
from auto_LiRPA import BoundedModule
from auto_LiRPA.operators.bivariate import BoundMul
from auto_LiRPA.operators.linear import BoundLinear

from width_cert.bounds.lirpa import bounded_parameters, bounded_tensor, build_trace_inputs
from width_cert.bounds.operators import CUSTOM_BOUND_OPERATORS, BoundStepActivation
from width_cert.models.gradient_graph import BinaryBCEGradientGraph


def _counted(counter, owner, name):
    original = getattr(owner, name)

    def wrapper(self, *args, **kwargs):
        counter[f"{owner.__name__}.{name}"] += 1
        return original(self, *args, **kwargs)

    return wrapper


def _run_case(dimensions):
    dummy, _ = build_trace_inputs(dimensions, torch.device("cpu"))
    graph = BoundedModule(
        BinaryBCEGradientGraph(len(dimensions) - 1),
        dummy,
        custom_ops=CUSTOM_BOUND_OPERATORS,
        bound_opts={"override_dependent_operations": True},
    )
    count = sum(
        input_dim * output_dim + output_dim
        for input_dim, output_dim in zip(dimensions[:-1], dimensions[1:])
    )
    parameter_lower = torch.linspace(-0.4, 0.3, count)
    inputs = (
        bounded_tensor(
            torch.full((1, 1, dimensions[0]), -0.2),
            torch.full((1, 1, dimensions[0]), 0.2),
        ),
        *bounded_parameters(parameter_lower, parameter_lower + 0.1, dimensions),
        torch.nn.functional.one_hot(
            torch.tensor([0]), num_classes=dimensions[-1]
        ).float(),
    )
    calls = Counter()
    methods = (
        (BoundLinear, "bound_backward_with_weight"),
        (BoundMul, "bound_backward_both_perturbed"),
        (BoundStepActivation, "bound_relax"),
    )
    with ExitStack() as stack:
        for owner, name in methods:
            stack.enter_context(patch.object(owner, name, _counted(calls, owner, name)))
        lower_bound, upper_bound = graph.compute_bounds(x=inputs, method="CROWN")
    parameter_nodes = [
        node for node in graph.nodes() if type(node).__name__ == "BoundInput"
    ][1:-1]
    return calls, parameter_nodes, lower_bound, upper_bound


def test_binary_full_gradient_crown_uses_weight_and_step_relaxations():
    calls, parameter_nodes, lower, upper = _run_case([2, 3, 3, 1])
    assert all(node.perturbed for node in parameter_nodes)
    assert calls["BoundLinear.bound_backward_with_weight"] > 0
    assert calls["BoundMul.bound_backward_both_perturbed"] > 0
    assert calls["BoundStepActivation.bound_relax"] > 0
    assert torch.isfinite(lower).all() and torch.isfinite(upper).all()
    assert (lower <= upper).all()


def test_multiclass_one_vs_all_graph_uses_same_full_crown_path():
    calls, parameter_nodes, lower, upper = _run_case([6, 3, 4])
    assert all(node.perturbed for node in parameter_nodes)
    assert calls["BoundLinear.bound_backward_with_weight"] > 0
    assert calls["BoundMul.bound_backward_both_perturbed"] > 0
    assert calls["BoundStepActivation.bound_relax"] > 0
    assert lower.shape == upper.shape == (1, 37)
    assert (lower <= upper).all()
