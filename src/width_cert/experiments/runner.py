"""Unified configuration-driven runner for binary and multi-class experiments."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as functional
from auto_LiRPA import BoundedModule
from torch.utils.data import DataLoader, TensorDataset

from ..bounds import BoundMethod, CUSTOM_BOUND_OPERATORS
from ..bounds.lirpa import (
    apply_flat_interval_update,
    bounded_tensor,
    build_trace_inputs,
    compute_gradient_bounds,
    flatten_parameters,
    initialize_interval_state,
)
from ..certification import evaluate_center_accuracy, evaluate_certification
from ..core import CertifiedTrainer, FixedUpdate, continue_with_iwau
from ..data import DatasetBundle, build_dataset
from ..logging import ConsoleReporter, RunLogger, SCHEMA_VERSION
from ..models import BinaryBCEGradientGraph, EvaluationGraph, PretrainingMLP, layer_dimensions
from .config import ExperimentConfig
from .environment import environment_metadata
from .pretrain import PretrainingRejected, PretrainingResult, pretrain_to_accuracy


def _method_parts(name: str) -> tuple[BoundMethod, bool]:
    return (BoundMethod.IC if name.startswith("IC") else BoundMethod.IBP, name.endswith("+IWAU"))


def _write_index(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class _PreparedStart:
    seed: int
    bundle: DatasetBundle
    dimensions: list[int]
    initial_state: dict[str, torch.Tensor]
    pretraining: PretrainingResult


def _prepare_starts(
    config: ExperimentConfig,
    device: torch.device,
    output_root: Path,
    reporter: ConsoleReporter,
) -> list[_PreparedStart]:
    """Select shared starts before running any certified-training method."""

    shared_bundle = None
    if not config.data.per_training_seed:
        shared_bundle = build_dataset(
            config.data.dataset,
            train_size=config.data.train_size,
            test_size=config.data.test_size,
            seed=config.data.seed,
            strategy=config.data.strategy,
        )

    selected: list[_PreparedStart] = []
    accepted_rows: list[dict[str, object]] = []
    rejected_rows: list[dict[str, object]] = []
    candidates = config.seed_selection.candidates()
    reporter.selection_start(
        candidate_start=config.seed_selection.start,
        candidate_stop=config.seed_selection.stop,
        requested=config.seed_selection.count,
        pretraining_enabled=config.pretraining.enabled,
        target_accuracy=config.pretraining.target_accuracy,
    )
    for candidate_index, seed in enumerate(candidates, start=1):
        if len(selected) >= config.seed_selection.count:
            break
        bundle = shared_bundle or build_dataset(
            config.data.dataset,
            train_size=config.data.train_size,
            test_size=config.data.test_size,
            seed=seed,
            strategy=config.data.strategy,
        )
        train_inputs = bundle.train_inputs.to(device)
        train_targets = bundle.train_targets.to(device)
        test_inputs = bundle.test_inputs.to(device)
        test_targets = bundle.test_targets.to(device)
        train_loader = DataLoader(
            TensorDataset(train_inputs, train_targets),
            batch_size=config.data.train_size,
            shuffle=False,
        )
        dimensions = layer_dimensions(
            bundle.input_dim,
            config.model.hidden_width,
            bundle.output_dim,
            config.model.hidden_layers,
        )
        torch.manual_seed(seed)
        point_model = PretrainingMLP(dimensions).to(device)
        try:
            result = pretrain_to_accuracy(
                point_model,
                train_loader,
                test_inputs,
                test_targets,
                target_accuracy=config.pretraining.target_accuracy,
                maximum_initial_accuracy=config.pretraining.maximum_initial_accuracy,
                maximum_final_accuracy=config.pretraining.maximum_final_accuracy,
                step_size=config.pretraining.step_size,
                maximum_epochs=config.pretraining.maximum_epochs,
                minimum_epochs=config.pretraining.minimum_epochs,
                minimum_accuracy_gain=config.pretraining.minimum_accuracy_gain,
                multiclass=bundle.output_dim > 1,
                enabled=config.pretraining.enabled,
                optimizer_name=config.pretraining.optimizer,
            )
        except PretrainingRejected as error:
            rejected_rows.append(
                {
                    "seed": seed,
                    "reason": error.reason,
                    "accuracy": error.accuracy,
                }
            )
            reporter.selection_candidate(
                candidate_index=candidate_index,
                candidate_total=len(candidates),
                seed=seed,
                accepted=False,
                initial_accuracy=error.accuracy,
                reason=error.reason,
            )
            continue
        selected.append(
            _PreparedStart(
                seed=seed,
                bundle=bundle,
                dimensions=dimensions,
                initial_state=initialize_interval_state(point_model),
                pretraining=result,
            )
        )
        accepted_rows.append(
            {
                "seed": seed,
                "initial_accuracy": result.initial_accuracy,
                "pretraining_accuracy": result.final_accuracy,
                "pretraining_epochs": result.epochs,
            }
        )
        reporter.selection_candidate(
            candidate_index=candidate_index,
            candidate_total=len(candidates),
            seed=seed,
            accepted=True,
            initial_accuracy=result.initial_accuracy,
            prepared_accuracy=result.final_accuracy,
            pretraining_epochs=result.epochs,
        )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "candidate_range": asdict(config.seed_selection),
        "candidate_seeds": list(candidates),
        "selected_seeds": [item.seed for item in selected],
        "accepted": accepted_rows,
        "rejected": rejected_rows,
        "selection_completed": len(selected) == config.seed_selection.count,
    }
    (output_root / "seed_selection.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    reporter.selection_complete(
        selected=[item.seed for item in selected],
        rejected=len(rejected_rows),
    )
    if len(selected) != config.seed_selection.count:
        raise RuntimeError(
            f"only {len(selected)} of {config.seed_selection.count} requested starts "
            f"passed pretraining within candidate range [{config.seed_selection.start}, "
            f"{config.seed_selection.stop})"
        )
    return selected


def select_seeds(
    config: ExperimentConfig,
    output_dir: str | Path | None = None,
) -> Path:
    """Run only shared-start selection and return its audit JSON path."""

    config.validate()
    output_root = Path(output_dir or config.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reporter = ConsoleReporter(enabled=config.echo_epochs)
    reporter.experiment_start(
        name=f"{config.name} [seed selection only]",
        device=str(device),
        candidate_start=config.seed_selection.start,
        candidate_stop=config.seed_selection.stop,
        accepted_seeds=config.seed_selection.count,
        radii=config.radii,
        methods=config.methods,
        total_cells=0,
    )
    _prepare_starts(config, device, output_root, reporter)
    reporter.experiment_complete(name=config.name, output_dir=output_root, completed_cells=0)
    return output_root / "seed_selection.json"


def run_experiment(config: ExperimentConfig) -> Path:
    """Run every radius/seed/method cell and return the result root."""

    config.validate()
    output_root = Path(config.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index_rows: list[dict[str, object]] = []
    reporter = ConsoleReporter(enabled=config.echo_epochs)
    total_cells = config.seed_selection.count * len(config.radii) * len(config.methods)
    reporter.experiment_start(
        name=config.name,
        device=str(device),
        candidate_start=config.seed_selection.start,
        candidate_stop=config.seed_selection.stop,
        accepted_seeds=config.seed_selection.count,
        radii=config.radii,
        methods=config.methods,
        total_cells=total_cells,
    )

    prepared_starts = _prepare_starts(config, device, output_root, reporter)
    completed_cells = 0

    for seed_index, prepared in enumerate(prepared_starts, start=1):
        seed = prepared.seed
        bundle = prepared.bundle
        train_inputs = bundle.train_inputs.to(device)
        train_targets = bundle.train_targets.to(device)
        test_inputs = bundle.test_inputs.to(device)
        test_targets = bundle.test_targets.to(device)
        test_loader = DataLoader(
            TensorDataset(test_inputs, test_targets),
            batch_size=config.data.test_size,
            shuffle=False,
        )
        dimensions = prepared.dimensions
        linear_layers = len(dimensions) - 1
        pretrain_accuracy = prepared.pretraining.final_accuracy
        initial_state = prepared.initial_state
        reporter.seed_start(
            index=seed_index,
            total=len(prepared_starts),
            seed=seed,
            prepared_accuracy=pretrain_accuracy,
            pretraining_epochs=prepared.pretraining.epochs,
        )
        seed_rows: list[dict[str, object]] = []

        gradient_dummy, evaluation_dummy = build_trace_inputs(dimensions, device)
        gradient_graph = BoundedModule(
            BinaryBCEGradientGraph(linear_layers).to(device),
            gradient_dummy,
            custom_ops=CUSTOM_BOUND_OPERATORS,
            bound_opts={"override_dependent_operations": True},
        )
        evaluation_graph = BoundedModule(
            EvaluationGraph(linear_layers).to(device),
            evaluation_dummy,
            custom_ops=CUSTOM_BOUND_OPERATORS,
            bound_opts={"override_dependent_operations": True},
        )

        for radius in config.radii:
            evaluation_radius = config.evaluation.radius(radius)
            fixed_runs = {}
            fixed_timings = {}
            for method_name in config.methods:
                bound_method, adaptive = _method_parts(method_name)
                fixed_name = bound_method.name

                def gradient_bounds(state, selected=bound_method):
                    flat_lower, flat_upper = flatten_parameters(state, linear_layers)
                    total_lower = torch.zeros_like(flat_lower)
                    total_upper = torch.zeros_like(flat_upper)
                    batch_size = config.training.gradient_batch_size
                    for start in range(0, len(train_inputs), batch_size):
                        end = min(start + batch_size, len(train_inputs))
                        inputs = train_inputs[start:end].reshape(-1, 1, bundle.input_dim)
                        bounded_inputs = bounded_tensor(inputs - radius, inputs + radius)
                        targets = train_targets[start:end]
                        if bundle.output_dim > 1:
                            targets = functional.one_hot(targets.long(), bundle.output_dim).float()
                        lower, upper = compute_gradient_bounds(
                            gradient_graph,
                            bounded_inputs,
                            targets,
                            flat_lower,
                            flat_upper,
                            dimensions,
                            selected,
                        )
                        total_lower += lower.sum(dim=0).reshape(-1)
                        total_upper += upper.sum(dim=0).reshape(-1)
                    return total_lower / len(train_inputs), total_upper / len(train_inputs)

                def metrics(state, selected=bound_method, test_radius=evaluation_radius):
                    return evaluate_certification(
                        evaluation_graph,
                        test_loader,
                        state,
                        input_radius=test_radius,
                        method=selected,
                        dimensions=dimensions,
                    )

                trainer = CertifiedTrainer(
                    gradient_bounds=gradient_bounds,
                    certified_accuracy=lambda state: metrics(state).certified_accuracy,
                    clean_accuracy=lambda state: evaluate_center_accuracy(
                        test_loader, state, dimensions=dimensions
                    ),
                    apply_update=lambda state, lower, upper, step: apply_flat_interval_update(
                        state, lower, upper, step, dimensions
                    ),
                    horizon=config.training.horizon,
                    weight_width=lambda state: float(
                        (flatten_parameters(state, linear_layers)[1] - flatten_parameters(state, linear_layers)[0]).mean()
                    ),
                    max_gradient_width=config.training.max_gradient_width,
                )
                cell_dir = output_root / f"eps_{radius:g}" / f"seed_{seed}" / method_name.replace("+", "_")
                logger = RunLogger(cell_dir, echo=config.echo_epochs)
                run_index = completed_cells + 1
                logger.begin_run(
                    index=run_index,
                    total=total_cells,
                    method=method_name,
                    seed=seed,
                    training_radius=radius,
                    evaluation_radius=evaluation_radius,
                    horizon=config.training.horizon,
                )
                resolved_run_config = {
                    "experiment": config.name,
                    "dataset": asdict(config.data),
                    "model": asdict(config.model),
                    "pretraining": asdict(config.pretraining),
                    "seed_selection": asdict(config.seed_selection),
                    "selected_seeds": [item.seed for item in prepared_starts],
                    "training": asdict(config.training),
                    "evaluation": {
                        **asdict(config.evaluation),
                        "resolved_radius": evaluation_radius,
                    },
                    "training_radius": radius,
                    "seed": seed,
                    "method": method_name,
                    "role": "reported_method",
                    "loss": {
                        "pretraining": "cross_entropy" if bundle.output_dim > 1 else "binary_cross_entropy",
                        "certified_training": "one_vs_all_binary_cross_entropy" if bundle.output_dim > 1 else "binary_cross_entropy",
                    },
                    "environment": environment_metadata(),
                }
                logger.write_config(resolved_run_config)
                start_time = time.perf_counter()
                try:
                    if adaptive:
                        if fixed_name not in fixed_runs:
                            logger.echo_internal_prerun_start(
                                method=fixed_name,
                                horizon=config.training.horizon,
                            )
                            dependency_start = time.perf_counter()
                            fixed_run = trainer.run(
                                initial_state,
                                FixedUpdate(config.training.base_step),
                                capture_snapshots=True,
                            )
                            dependency_seconds = time.perf_counter() - dependency_start
                            fixed_runs[fixed_name] = fixed_run
                            fixed_timings[fixed_name] = dependency_seconds
                            dependency_logger = RunLogger(
                                cell_dir / "calibration_pre_run",
                                echo=False,
                            )
                            dependency_logger.write_config(
                                {
                                    **resolved_run_config,
                                    "method": fixed_name,
                                    "role": "iwau_calibration_pre_run",
                                    "requested_method": method_name,
                                }
                            )
                            dependency_logger.write_run(
                                fixed_run,
                                extra_summary={
                                    "reported_result": False,
                                    "requested_method": method_name,
                                    "elapsed_seconds": dependency_seconds,
                                },
                            )
                            logger.echo_internal_prerun_complete(
                                run=fixed_run,
                                elapsed_seconds=dependency_seconds,
                            )
                        continuation_start = time.perf_counter()
                        result = continue_with_iwau(
                            trainer,
                            fixed_runs[fixed_name],
                            base_step=config.training.base_step,
                            retention=config.training.retention,
                            coefficient_scale=config.training.coefficient_scale,
                        )
                        run = result.run
                        calibration = result.calibration
                        continuation_seconds = time.perf_counter() - continuation_start
                        pre_run_seconds = fixed_timings[fixed_name]
                        elapsed = pre_run_seconds + continuation_seconds
                        calibration_summary = {
                            "calibration_status": calibration.status.value,
                            "calibration_reason": calibration.reason,
                            "calibrated_coefficient": calibration.coefficient,
                            "applied_coefficient": (
                                calibration.coefficient * config.training.coefficient_scale
                            ),
                            "theoretical_condition_satisfied": (
                                config.training.coefficient_scale >= 1.0
                            ),
                            "peak_epoch": calibration.peak_epoch,
                            "reference_epoch": calibration.reference_epoch,
                            "iwau_effective": calibration.iwau_effective,
                        }
                        logger.echo_calibration(
                            source_method=fixed_name,
                            status=calibration.status.value,
                            peak_epoch=calibration.peak_epoch,
                            reference_epoch=calibration.reference_epoch,
                            coefficient=calibration.coefficient,
                        )
                    else:
                        if fixed_name in fixed_runs:
                            run = fixed_runs[fixed_name]
                            elapsed = fixed_timings[fixed_name]
                            logger.echo_run(run)
                        else:
                            run = trainer.run(
                                initial_state,
                                FixedUpdate(config.training.base_step),
                                capture_snapshots=True,
                                on_epoch=logger.epoch_callback,
                            )
                            fixed_runs[fixed_name] = run
                            elapsed = time.perf_counter() - start_time
                            fixed_timings[fixed_name] = elapsed
                        pre_run_seconds = 0.0
                        continuation_seconds = elapsed
                        calibration_summary = {
                            "calibration_status": "not_applicable",
                            "iwau_effective": False,
                        }
                    if adaptive and config.echo_epochs:
                        logger.echo_run(run)
                    logger.write_run(
                        run,
                        extra_summary={
                            **calibration_summary,
                            "pretraining_accuracy": 100.0 * pretrain_accuracy,
                            "elapsed_seconds": elapsed,
                            "pre_run_seconds": pre_run_seconds,
                            "continuation_seconds": continuation_seconds,
                            "executed_updates": (
                                len(fixed_runs[fixed_name].updates) + len(run.updates) - calibration.peak_epoch
                                if adaptive and calibration.iwau_effective
                                else len(run.updates)
                            ),
                            "seconds_per_executed_update": elapsed / max(
                                (
                                    len(fixed_runs[fixed_name].updates)
                                    + len(run.updates)
                                    - calibration.peak_epoch
                                    if adaptive and calibration.iwau_effective
                                    else len(run.updates)
                                ),
                                1,
                            ),
                            "wall_seconds_per_reported_epoch": elapsed / max(
                                config.training.horizon, 1
                            ),
                        },
                    )
                    logger.finish_run(run, elapsed_seconds=elapsed)
                except KeyboardInterrupt:
                    logger.write_terminal_status(
                        status="interrupted", reason="keyboard_interrupt", finalized=False
                    )
                    logger.fail_run(KeyboardInterrupt("keyboard_interrupt"))
                    raise
                except Exception as error:
                    logger.write_terminal_status(
                        status="failed",
                        reason=f"{type(error).__name__}:{error}",
                    )
                    logger.fail_run(error)
                    raise
                result_row = {
                    "schema_version": SCHEMA_VERSION,
                    "radius": radius,
                    "evaluation_radius": evaluation_radius,
                    "seed": seed,
                    "method": method_name,
                    "pretraining_accuracy": 100.0 * pretrain_accuracy,
                    "peak_certified_accuracy": run.peak_certified_accuracy,
                    "peak_epoch": run.peak_epoch,
                    "cert_at_T": run.cert_at_horizon,
                    "final_clean_accuracy": run.states[-1].clean_accuracy,
                    "final_weight_width": run.states[-1].weight_width,
                    "status": run.status.value,
                    "iwau_status": calibration_summary["calibration_status"],
                    "iwau_effective": calibration_summary["iwau_effective"],
                    "elapsed_seconds": elapsed,
                }
                index_rows.append(result_row)
                seed_rows.append(result_row)
                _write_index(output_root / "results.csv", index_rows)
                completed_cells += 1
        reporter.seed_summary(seed=seed, rows=seed_rows)
        reporter.seed_complete(
            index=seed_index,
            total=len(prepared_starts),
            seed=seed,
            completed_cells=completed_cells,
        )
    reporter.experiment_summary(rows=index_rows)
    reporter.experiment_complete(
        name=config.name,
        output_dir=output_root,
        completed_cells=completed_cells,
    )
    return output_root
