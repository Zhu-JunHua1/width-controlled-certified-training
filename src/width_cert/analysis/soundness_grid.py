"""Deterministic small-network implementation checks for gradient bounds.

This module deliberately does not call the gradient-graph implementation to
obtain reference gradients.  Instead, it evaluates the gradients of a
one-hidden-unit binary MLP analytically on a dense Cartesian grid.  At a ReLU
kink, both endpoint derivative choices in ``[0, 1]`` are checked.

The grid is finite and therefore cannot prove real-arithmetic soundness.  Its
purpose is to catch implementation errors in interval parameters, bilinear
nodes, BCE derivatives, and the box relaxation of the ReLU derivative.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from auto_LiRPA import BoundedModule

from ..bounds import (
    AGTInspiredHybridBounder,
    BoundMethod,
    CUSTOM_BOUND_OPERATORS,
    IntersectionDiagnostics,
    intersect_bounds,
)
from ..bounds.lirpa import bounded_tensor, build_trace_inputs, compute_gradient_bounds
from ..models import BinaryBCEGradientGraph


@dataclass(frozen=True)
class GridCase:
    """A scalar-input, scalar-hidden, scalar-output interval problem."""

    name: str
    input_interval: tuple[float, float]
    first_weight: tuple[float, float]
    first_bias: tuple[float, float]
    second_weight: tuple[float, float]
    second_bias: tuple[float, float]

    def intervals(self) -> tuple[tuple[float, float], ...]:
        return (
            self.input_interval,
            self.first_weight,
            self.first_bias,
            self.second_weight,
            self.second_bias,
        )


DEFAULT_GRID_CASES = (
    GridCase(
        "crossing_relu_positive_downstream",
        (-0.2, 0.4),
        (0.8, 1.2),
        (-0.25, 0.05),
        (0.6, 1.0),
        (-0.1, 0.2),
    ),
    GridCase(
        "crossing_relu_negative_downstream",
        (-0.2, 0.4),
        (0.8, 1.2),
        (-0.25, 0.05),
        (-1.1, -0.7),
        (-0.1, 0.2),
    ),
    GridCase(
        "crossing_relu_mixed_downstream",
        (-0.2, 0.4),
        (0.8, 1.2),
        (-0.25, 0.05),
        (-0.8, 0.9),
        (-0.1, 0.2),
    ),
    GridCase(
        "stable_active_relu",
        (0.2, 0.4),
        (0.8, 1.2),
        (0.1, 0.2),
        (-0.8, 0.9),
        (-0.1, 0.2),
    ),
    GridCase(
        "stable_inactive_relu",
        (-0.4, -0.2),
        (0.8, 1.2),
        (-0.2, -0.1),
        (-0.8, 0.9),
        (-0.1, 0.2),
    ),
)


def _grid_values(interval: tuple[float, float]) -> tuple[float, ...]:
    lower, upper = interval
    return lower, (lower + upper) / 2.0, upper


def _analytic_bce_gradients(
    x: float,
    w1: float,
    b1: float,
    w2: float,
    b2: float,
    target: float,
) -> list[tuple[float, float, float, float]]:
    """Return all admitted gradients, including both choices at a ReLU kink."""

    z1 = x * w1 + b1
    activation = max(z1, 0.0)
    z2 = activation * w2 + b2
    delta2 = 1.0 / (1.0 + math.exp(-z2)) - target
    if abs(z1) <= 1e-12:
        step_choices = (0.0, 1.0)
    else:
        step_choices = (1.0 if z1 > 0.0 else 0.0,)
    gradients = []
    for step in step_choices:
        delta1 = delta2 * w2 * step
        gradients.append((delta1 * x, delta1, delta2 * activation, delta2))
    return gradients


def _case_bounds(
    case: GridCase,
    target: float,
    full_graph,
    hybrid,
) -> tuple[
    dict[str, tuple[torch.Tensor, torch.Tensor]],
    dict[str, dict[str, int | float]],
]:
    dimensions = [1, 1, 1]
    input_lower = torch.tensor([[[case.input_interval[0]]]], dtype=torch.float32)
    input_upper = torch.tensor([[[case.input_interval[1]]]], dtype=torch.float32)
    flat_lower = torch.tensor(
        [
            case.first_weight[0],
            case.first_bias[0],
            case.second_weight[0],
            case.second_bias[0],
        ],
        dtype=torch.float32,
    )
    flat_upper = torch.tensor(
        [
            case.first_weight[1],
            case.first_bias[1],
            case.second_weight[1],
            case.second_bias[1],
        ],
        dtype=torch.float32,
    )
    bounded_inputs = bounded_tensor(input_lower, input_upper)
    targets = torch.tensor([target], dtype=torch.float32)

    def full(method: BoundMethod):
        lower, upper = compute_gradient_bounds(
            full_graph,
            bounded_inputs,
            targets,
            flat_lower,
            flat_upper,
            dimensions,
            method,
        )
        return lower.detach().cpu().reshape(-1), upper.detach().cpu().reshape(-1)

    bounds = {
        "IBP": full(BoundMethod.IBP),
        "AGT-inspired hybrid": tuple(
            value.detach().cpu().reshape(-1)
            for value in hybrid.compute(
                bounded_inputs=bounded_inputs,
                input_lower=input_lower,
                input_upper=input_upper,
                targets=targets,
                flat_lower=flat_lower,
                flat_upper=flat_upper,
            )
        ),
        "Full-CROWN-box": full(BoundMethod.CROWN),
    }
    diagnostics = {}
    for name, source in (
        ("Full-IC", "Full-CROWN-box"),
        ("Hybrid-IC", "AGT-inspired hybrid"),
    ):
        record = IntersectionDiagnostics()
        lower, upper = intersect_bounds(
            *bounds["IBP"],
            *bounds[source],
            diagnostics=record,
        )
        bounds[name] = lower, upper
        diagnostics[name] = record.as_dict()
    return bounds, diagnostics


def validate_small_network_grid(
    output_dir: str | Path,
    *,
    tolerance: float = 1e-5,
    cases: tuple[GridCase, ...] = DEFAULT_GRID_CASES,
) -> tuple[Path, Path]:
    """Run a deterministic dense-grid implementation check on a tiny MLP.

    Each interval variable is evaluated at its lower endpoint, midpoint, and
    upper endpoint.  This finite check is intentionally labelled as a
    diagnostic and must not be cited as a proof of real-arithmetic soundness.
    """

    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    dimensions = [1, 1, 1]
    trace_inputs, _ = build_trace_inputs(dimensions, torch.device("cpu"))
    full_graph = BoundedModule(
        BinaryBCEGradientGraph(2),
        trace_inputs,
        custom_ops=CUSTOM_BOUND_OPERATORS,
        bound_opts={"override_dependent_operations": True},
    )
    hybrid = AGTInspiredHybridBounder(dimensions, torch.device("cpu"))

    rows = []
    total_points = 0
    total_admitted_gradients = 0
    zero_kink_points = 0
    intersection_diagnostics = {}
    for case in cases:
        grid = tuple(_grid_values(interval) for interval in case.intervals())
        points = tuple(itertools.product(*grid))
        total_points += len(points) * 2
        for target in (0.0, 1.0):
            bounds, diagnostics = _case_bounds(case, target, full_graph, hybrid)
            intersection_diagnostics[f"{case.name}/target={int(target)}"] = diagnostics
            exact_gradients = []
            kink_count = 0
            for point in points:
                admitted = _analytic_bce_gradients(*point, target)
                if len(admitted) == 2:
                    kink_count += 1
                exact_gradients.extend(admitted)
            zero_kink_points += kink_count
            total_admitted_gradients += len(exact_gradients)
            exact = torch.tensor(exact_gradients, dtype=torch.float64)
            for method, (lower, upper) in bounds.items():
                lower64, upper64 = lower.double(), upper.double()
                below = exact < lower64.unsqueeze(0) - tolerance
                above = exact > upper64.unsqueeze(0) + tolerance
                violations = below | above
                lower_slack = exact - lower64.unsqueeze(0)
                upper_slack = upper64.unsqueeze(0) - exact
                width = upper64 - lower64
                rows.append(
                    {
                        "case": case.name,
                        "target": int(target),
                        "method": method,
                        "grid_points": len(points),
                        "admitted_gradients": len(exact_gradients),
                        "relu_kink_points": kink_count,
                        "checked_coordinates": exact.numel(),
                        "violations": int(violations.sum()),
                        "minimum_lower_slack": float(lower_slack.min()),
                        "minimum_upper_slack": float(upper_slack.min()),
                        "mean_bound_width": float(width.mean()),
                        "maximum_bound_width": float(width.max()),
                    }
                )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "small_network_grid_check.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    total_violations = sum(int(row["violations"]) for row in rows)
    report = {
        "schema_version": "1.0",
        "check_type": "deterministic_dense_grid_implementation_diagnostic",
        "is_real_arithmetic_soundness_proof": False,
        "network_dimensions": dimensions,
        "loss": "binary_cross_entropy_with_logits",
        "grid_values_per_interval_variable": 3,
        "cases": len(cases),
        "targets_per_case": 2,
        "parameter_and_input_grid_points": total_points,
        "admitted_gradient_evaluations": total_admitted_gradients,
        "relu_kink_points": zero_kink_points,
        "relu_kink_derivative_choices": [0.0, 1.0],
        "tolerance": tolerance,
        "total_checked_coordinates_across_methods": sum(
            int(row["checked_coordinates"]) for row in rows
        ),
        "total_violations": total_violations,
        "passed": total_violations == 0,
        "intersection_diagnostics": intersection_diagnostics,
        "limitations": (
            "Finite endpoint/midpoint enumeration can reveal implementation errors but "
            "cannot replace the operator-wise real-arithmetic proof."
        ),
    }
    json_path = output / "small_network_grid_check.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return csv_path, json_path
