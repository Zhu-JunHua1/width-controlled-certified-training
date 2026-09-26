"""Same-parameter-box audit for competing gradient-bound constructions."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as functional
from auto_LiRPA import BoundedModule

from ..bounds import (
    AGTInspiredHybridBounder,
    BoundMethod,
    CUSTOM_BOUND_OPERATORS,
    IntersectionDiagnostics,
    intersect_bounds,
)
from ..bounds.lirpa import (
    bounded_tensor,
    build_trace_inputs,
    compute_gradient_bounds,
    flatten_parameters,
)
from ..data import build_dataset
from ..experiments.config import ExperimentConfig
from ..logging.checkpoints import load_checkpoint
from ..models import BinaryBCEGradientGraph, layer_dimensions


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _unflatten_point_parameters(flat: torch.Tensor, dimensions: list[int]):
    parameters = []
    cursor = 0
    for input_dim, output_dim in zip(dimensions[:-1], dimensions[1:]):
        size = input_dim * output_dim
        parameters.append(flat[cursor : cursor + size].reshape(1, output_dim, input_dim))
        cursor += size
        parameters.append(flat[cursor : cursor + output_dim].reshape(1, 1, output_dim))
        cursor += output_dim
    return tuple(parameters)


def _measure(device: torch.device, function):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    _synchronize(device)
    started = time.perf_counter()
    result = function()
    _synchronize(device)
    elapsed = time.perf_counter() - started
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    return result, elapsed, peak_memory


def audit_same_box(
    config: ExperimentConfig,
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    training_seed: int,
    input_radius: float,
    batch_size: int = 8,
    parameter_samples: int = 32,
    sampling_seed: int = 0,
) -> tuple[Path, Path]:
    """Audit IBP, hybrid, full CROWN, and intersections on one fixed box.

    Random sampling is an implementation diagnostic only and is not a proof of
    soundness.  The audit records this limitation in its manifest.
    """

    config.validate()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle = build_dataset(
        config.data.dataset,
        train_size=config.data.train_size,
        calibration_size=config.data.calibration_size,
        test_size=config.data.test_size,
        seed=training_seed if config.data.per_training_seed else config.data.seed,
        calibration_seed=config.data.calibration_seed,
        test_seed=config.data.test_seed,
        strategy=config.data.strategy,
    )
    dimensions = layer_dimensions(
        bundle.input_dim,
        config.model.hidden_width,
        bundle.output_dim,
        config.model.hidden_layers,
    )
    linear_layers = len(dimensions) - 1
    epoch, state = load_checkpoint(checkpoint)
    state = {name: tensor.to(device) for name, tensor in state.items()}
    flat_lower, flat_upper = flatten_parameters(state, linear_layers)

    inputs = bundle.train_inputs[:batch_size].to(device).reshape(-1, 1, bundle.input_dim)
    targets = bundle.train_targets[:batch_size].to(device)
    if bundle.output_dim > 1:
        targets = functional.one_hot(targets.long(), bundle.output_dim).float()
    input_lower, input_upper = inputs - input_radius, inputs + input_radius
    bounded_inputs = bounded_tensor(input_lower, input_upper)

    dummy, _ = build_trace_inputs(dimensions, device)
    full_graph = BoundedModule(
        BinaryBCEGradientGraph(linear_layers).to(device),
        dummy,
        custom_ops=CUSTOM_BOUND_OPERATORS,
        bound_opts={"override_dependent_operations": True},
    )
    hybrid = AGTInspiredHybridBounder(dimensions, device)

    def full(method: BoundMethod):
        return compute_gradient_bounds(
            full_graph,
            bounded_inputs,
            targets,
            flat_lower,
            flat_upper,
            dimensions,
            method,
        )

    bounds = {}
    timings = {}
    memory = {}
    bounds["IBP"], timings["IBP"], memory["IBP"] = _measure(
        device, lambda: full(BoundMethod.IBP)
    )
    bounds["AGT-inspired hybrid"], timings["AGT-inspired hybrid"], memory[
        "AGT-inspired hybrid"
    ] = _measure(
        device,
        lambda: hybrid.compute(
            bounded_inputs=bounded_inputs,
            input_lower=input_lower,
            input_upper=input_upper,
            targets=targets,
            flat_lower=flat_lower,
            flat_upper=flat_upper,
        ),
    )
    bounds["Full-CROWN-box"], timings["Full-CROWN-box"], memory[
        "Full-CROWN-box"
    ] = _measure(device, lambda: full(BoundMethod.CROWN))

    full_ic_diagnostics = IntersectionDiagnostics()
    bounds["Full-IC"], full_ic_intersection_seconds, full_ic_memory = _measure(
        device,
        lambda: intersect_bounds(
            *bounds["IBP"],
            *bounds["Full-CROWN-box"],
            diagnostics=full_ic_diagnostics,
        ),
    )
    hybrid_ic_diagnostics = IntersectionDiagnostics()
    bounds["Hybrid-IC"], hybrid_ic_intersection_seconds, hybrid_ic_memory = _measure(
        device,
        lambda: intersect_bounds(
            *bounds["IBP"],
            *bounds["AGT-inspired hybrid"],
            diagnostics=hybrid_ic_diagnostics,
        ),
    )
    # End-to-end construction cost includes both source bounds and the final
    # coordinate-wise intersection; reporting intersection-only time would be
    # a misleading compute comparison.
    timings["Full-IC"] = (
        timings["IBP"] + timings["Full-CROWN-box"] + full_ic_intersection_seconds
    )
    timings["Hybrid-IC"] = (
        timings["IBP"]
        + timings["AGT-inspired hybrid"]
        + hybrid_ic_intersection_seconds
    )
    memory["Full-IC"] = (
        max(
            value
            for value in (memory["IBP"], memory["Full-CROWN-box"], full_ic_memory)
            if value is not None
        )
        if device.type == "cuda"
        else None
    )
    memory["Hybrid-IC"] = (
        max(
            value
            for value in (
                memory["IBP"],
                memory["AGT-inspired hybrid"],
                hybrid_ic_memory,
            )
            if value is not None
        )
        if device.type == "cuda"
        else None
    )

    torch.manual_seed(sampling_seed)
    exact_graph = BinaryBCEGradientGraph(linear_layers).to(device)
    exact_targets = targets.reshape(-1, 1) if targets.ndim == 1 else targets
    sampled_violations = {name: 0 for name in bounds}
    sampled_coordinates = 0
    for _ in range(parameter_samples):
        sampled_inputs = input_lower + torch.rand_like(input_lower) * (
            input_upper - input_lower
        )
        sampled_flat = flat_lower + torch.rand_like(flat_lower) * (
            flat_upper - flat_lower
        )
        exact = exact_graph(
            sampled_inputs,
            *_unflatten_point_parameters(sampled_flat, dimensions),
            exact_targets,
        )
        sampled_coordinates += exact.numel()
        for name, (lower, upper) in bounds.items():
            sampled_violations[name] += int(
                ((exact < lower - 1e-5) | (exact > upper + 1e-5)).sum()
            )

    ibp_width = bounds["IBP"][1] - bounds["IBP"][0]
    rows = []
    for name, (lower, upper) in bounds.items():
        width = upper - lower
        rows.append(
            {
                "method": name,
                "coordinates": width.numel(),
                "mean_width": float(width.mean()),
                "median_width": float(width.median()),
                "relative_mean_tightening_vs_ibp": float(
                    (ibp_width.mean() - width.mean())
                    / ibp_width.mean().clamp(min=1e-12)
                ),
                "sampled_violations": sampled_violations[name],
                "sampled_coordinates": sampled_coordinates,
                "seconds": timings[name],
                "peak_device_memory_bytes": memory[name],
            }
        )

    ibp_lower, ibp_upper = bounds["IBP"]
    crown_lower, crown_upper = bounds["Full-CROWN-box"]
    hybrid_lower, hybrid_upper = bounds["AGT-inspired hybrid"]
    selection = {
        "full_ic_crown_lower_selected_fraction": float((crown_lower > ibp_lower).float().mean()),
        "full_ic_crown_upper_selected_fraction": float((crown_upper < ibp_upper).float().mean()),
        "hybrid_ic_hybrid_lower_selected_fraction": float((hybrid_lower > ibp_lower).float().mean()),
        "hybrid_ic_hybrid_upper_selected_fraction": float((hybrid_upper < ibp_upper).float().mean()),
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "same_box_bound_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest_path = output / "same_box_bound_audit.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "checkpoint_epoch": epoch,
                "training_seed": training_seed,
                "input_radius": input_radius,
                "batch_size": len(inputs),
                "parameter_samples": parameter_samples,
                "sampling_seed": sampling_seed,
                "sampling_is_soundness_proof": False,
                "selection_fractions": selection,
                "intersection_diagnostics": {
                    "Full-IC": full_ic_diagnostics.as_dict(),
                    "Hybrid-IC": hybrid_ic_diagnostics.as_dict(),
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return csv_path, manifest_path
