"""Trace the auto_LiRPA path used by the full-gradient CROWN branch.

This tool is an implementation audit, not a soundness proof. It builds the
same ``BoundedModule`` used by the experiment runner for representative binary
and multiclass architectures, executes IBP and CROWN queries with interval
inputs and interval weights, and records graph nodes, perturbation flags, and
the relevant backward-relaxation calls.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
from auto_LiRPA import BoundedModule
from auto_LiRPA.operators.bivariate import BoundMul
from auto_LiRPA.operators.linear import BoundLinear, BoundMatMul
from auto_LiRPA.operators.relu import BoundRelu
from auto_LiRPA.operators.s_shaped import BoundSigmoid

from width_cert.bounds.lirpa import bounded_parameters, bounded_tensor, build_trace_inputs
from width_cert.bounds.operators import CUSTOM_BOUND_OPERATORS, BoundStepActivation
from width_cert.experiments.environment import environment_metadata
from width_cert.models.gradient_graph import BinaryBCEGradientGraph


CASES = {
    "binary_l1": [2, 3, 1],
    "binary_l2": [2, 3, 3, 1],
    "binary_l3": [2, 3, 3, 3, 1],
    # Multiclass training uses one-vs-all BCE in the same explicit gradient
    # graph; the output dimension is the number of classes.
    "multiclass_k4": [6, 3, 4],
}

TRACED_METHODS = (
    (BoundMatMul, "bound_backward"),
    (BoundLinear, "bound_backward_with_weight"),
    (BoundMul, "bound_backward"),
    (BoundMul, "bound_backward_both_perturbed"),
    (BoundRelu, "bound_backward"),
    (BoundSigmoid, "bound_relax"),
    (BoundStepActivation, "bound_relax"),
)


def _wrapped(counter: Counter[str], owner: type, method_name: str):
    original = getattr(owner, method_name)

    def wrapper(self, *args, **kwargs):
        counter[f"{owner.__name__}.{method_name}"] += 1
        return original(self, *args, **kwargs)

    return wrapper


def _source_record(owner: type, method_name: str) -> dict[str, object]:
    method = getattr(owner, method_name)
    _, line = inspect.getsourcelines(method)
    return {
        "owner": owner.__name__,
        "method": method_name,
        "source_module": getattr(method, "__module__", None),
        "source_line": line,
    }


def _case_inputs(dimensions: list[int]):
    parameter_count = sum(
        input_dim * output_dim + output_dim
        for input_dim, output_dim in zip(dimensions[:-1], dimensions[1:])
    )


def _tensor_record(value: torch.Tensor) -> dict[str, object]:
    array = value.detach().cpu().contiguous().numpy()
    return {
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        "minimum": float(value.min()),
        "maximum": float(value.max()),
    }
    input_lower = torch.linspace(-0.30, 0.10, dimensions[0]).reshape(1, 1, -1)
    input_upper = input_lower + 0.20
    parameter_lower = torch.linspace(-0.40, 0.30, parameter_count)
    parameter_upper = parameter_lower + 0.10
    target = torch.zeros(1, dimensions[-1])
    target[0, 0] = 1.0
    return (
        bounded_tensor(input_lower, input_upper),
        *bounded_parameters(parameter_lower, parameter_upper, dimensions),
        target,
    )


def _audit_case(case_name: str, dimensions: list[int]):
    device = torch.device("cpu")
    dummy_inputs, _ = build_trace_inputs(dimensions, device)
    graph = BoundedModule(
        BinaryBCEGradientGraph(linear_layers=len(dimensions) - 1).to(device),
        dummy_inputs,
        custom_ops=CUSTOM_BOUND_OPERATORS,
        bound_opts={"override_dependent_operations": True},
    )
    call_counter: Counter[str] = Counter()
    with ExitStack() as stack:
        for owner, method_name in TRACED_METHODS:
            stack.enter_context(
                patch.object(owner, method_name, _wrapped(call_counter, owner, method_name))
            )
        inputs = _case_inputs(dimensions)
        ibp_lower, ibp_upper = graph.compute_bounds(x=inputs, method="IBP")
        crown_lower, crown_upper = graph.compute_bounds(x=inputs, method="CROWN")

    rows: list[dict[str, object]] = []
    for node in graph.nodes():
        inputs = list(getattr(node, "inputs", []))
        rows.append(
            {
                "case": case_name,
                "name": node.name,
                "class": type(node).__name__,
                "module": type(node).__module__,
                "perturbed": getattr(node, "perturbed", None),
                "input_names": "|".join(item.name for item in inputs),
                "input_classes": "|".join(type(item).__name__ for item in inputs),
                "input_perturbed": "|".join(
                    str(getattr(item, "perturbed", None)) for item in inputs
                ),
            }
        )
    input_nodes = [row for row in rows if row["class"] == "BoundInput"]
    parameter_nodes = input_nodes[1:-1]
    report = {
        "dimensions": dimensions,
        "node_count": len(rows),
        "all_parameter_inputs_perturbed": all(
            row["perturbed"] is True for row in parameter_nodes
        ),
        "call_counts": dict(sorted(call_counter.items())),
        "outputs": {
            "shape": list(crown_lower.shape),
            "ibp_lower": _tensor_record(ibp_lower),
            "ibp_upper": _tensor_record(ibp_upper),
            "crown_lower": _tensor_record(crown_lower),
            "crown_upper": _tensor_record(crown_upper),
            "ibp_finite": bool(
                torch.isfinite(ibp_lower).all() and torch.isfinite(ibp_upper).all()
            ),
            "crown_finite": bool(
                torch.isfinite(crown_lower).all() and torch.isfinite(crown_upper).all()
            ),
            "ibp_ordered": bool((ibp_lower <= ibp_upper).all()),
            "crown_ordered": bool((crown_lower <= crown_upper).all()),
        },
    }
    return rows, report


def audit(output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    case_reports = {}
    for case_name, dimensions in CASES.items():
        rows, report = _audit_case(case_name, dimensions)
        all_rows.extend(rows)
        case_reports[case_name] = report

    with (output_dir / "bound_graph_nodes.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    report = {
        "purpose": "implementation trace; not a soundness proof",
        "environment": environment_metadata(),
        "bound_options": {"override_dependent_operations": True},
        "cases": case_reports,
        "traced_sources": [
            _source_record(owner, method_name) for owner, method_name in TRACED_METHODS
        ],
    }
    (output_dir / "execution_trace.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/audit/a0_full_crown"),
    )
    args = parser.parse_args()
    print(json.dumps(audit(args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
