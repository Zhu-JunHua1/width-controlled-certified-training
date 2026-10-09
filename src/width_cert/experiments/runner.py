"""Unified configuration-driven runner for binary and multi-class experiments."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as functional
from auto_LiRPA import BoundedModule
from torch.utils.data import DataLoader, TensorDataset

from ..bounds import (
    AGTInspiredHybridBounder,
    BoundMethod,
    CUSTOM_BOUND_OPERATORS,
    IntersectionDiagnostics,
)
from ..bounds.lirpa import (
    apply_flat_interval_update,
    bounded_tensor,
    build_trace_inputs,
    compute_gradient_bounds,
    flatten_parameters,
    initialize_interval_state,
)
from ..certification import evaluate_center_accuracy, evaluate_certification
from ..core import (
    CertifiedTrainer,
    FixedUpdate,
    GlobalWidthAdaptiveUpdate,
    budget_record,
    calibrate_iwau,
    continue_from_fixed_prefix,
    continue_with_iwau,
    cumulative_matched_cosine_rule,
    matched_final_step_rules,
)
from ..data import DatasetBundle, build_dataset
from ..logging import ConsoleReporter, RunLogger, SCHEMA_VERSION
from ..models import BinaryBCEGradientGraph, EvaluationGraph, PretrainingMLP, layer_dimensions
from .config import ExperimentConfig
from .environment import environment_metadata
from .evaluation import evaluate_frozen_checkpoints
from .pretrain import PretrainingRejected, PretrainingResult, pretrain_to_accuracy
from .study_lock import (
    canonical_mapping_sha256,
    experiment_config_sha256,
    verify_frozen_study_lock,
)


def _method_parts(name: str) -> tuple[BoundMethod, str]:
    base, separator, suffix = name.partition("+")
    update_kind = suffix if separator else "Fixed"
    methods = {
        "IBP": BoundMethod.IBP,
        "AGT-Hybrid": BoundMethod.HYBRID,
        "CROWN": BoundMethod.CROWN,
        "IC": BoundMethod.IC,
    }
    try:
        return methods[base], update_kind
    except KeyError as error:
        raise ValueError(f"unknown method: {name}") from error


def _configured_bound_method(name: str) -> BoundMethod:
    try:
        return BoundMethod(name.lower())
    except ValueError as error:
        raise ValueError(f"unknown configured bound method: {name}") from error


def _write_index(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated(device))


def _maximum_memory(*values: int | None) -> int | None:
    present = [value for value in values if value is not None]
    return max(present) if present else None


@dataclass
class _PreparedStart:
    seed: int
    bundle: DatasetBundle
    dimensions: list[int]
    initial_state: dict[str, torch.Tensor]
    pretraining: PretrainingResult


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_selected_seeds(config: ExperimentConfig) -> tuple[list[int], dict | None]:
    selection = config.seed_selection
    if selection.manifest is None:
        return list(selection.candidates()), None
    path = Path(selection.manifest)
    if not path.is_file():
        raise FileNotFoundError(f"seed-selection manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "1.0":
        raise ValueError("seed-selection manifest must use schema_version 1.0")
    groups = payload.get("groups")
    if not isinstance(groups, dict) or selection.group not in groups:
        raise ValueError(
            f"seed-selection group {selection.group!r} is missing from {path}"
        )
    group = groups[selection.group]
    seeds = group.get("selected_seeds") if isinstance(group, dict) else None
    if not isinstance(seeds, list) or any(not isinstance(seed, int) for seed in seeds):
        raise ValueError("seed-selection manifest selected_seeds must be an integer list")
    if len(seeds) != selection.count or len(set(seeds)) != len(seeds):
        raise ValueError(
            "seed-selection manifest must contain exactly the requested unique starts"
        )
    allowed = set(selection.candidates())
    if any(seed not in allowed for seed in seeds):
        raise ValueError("seed-selection manifest contains a seed outside the candidate range")
    return seeds, {
        "path": selection.manifest,
        "group": selection.group,
        "sha256": _file_sha256(path),
        "rule": group.get("selection_rule"),
    }


def _shared_bundle(config: ExperimentConfig) -> DatasetBundle | None:
    if config.data.per_training_seed:
        return None
    return build_dataset(
        config.data.dataset,
        train_size=config.data.train_size,
        calibration_size=config.data.calibration_size,
        test_size=config.data.test_size,
        seed=config.data.seed,
        calibration_seed=config.data.calibration_seed,
        test_seed=config.data.test_seed,
        strategy=config.data.strategy,
        split_manifest=config.data.split_manifest,
    )


def _prepare_candidate(
    config: ExperimentConfig,
    seed: int,
    device: torch.device,
    shared_bundle: DatasetBundle | None,
) -> _PreparedStart:
    bundle = shared_bundle or build_dataset(
        config.data.dataset,
        train_size=config.data.train_size,
        calibration_size=config.data.calibration_size,
        test_size=config.data.test_size,
        seed=seed,
        calibration_seed=config.data.calibration_seed,
        test_seed=config.data.test_seed,
        strategy=config.data.strategy,
        split_manifest=config.data.split_manifest,
    )
    train_inputs = bundle.train_inputs.to(device)
    train_targets = bundle.train_targets.to(device)
    calibration_inputs = bundle.calibration_inputs.to(device)
    calibration_targets = bundle.calibration_targets.to(device)
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
    result = pretrain_to_accuracy(
        point_model,
        train_loader,
        calibration_inputs,
        calibration_targets,
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
    return _PreparedStart(
        seed=seed,
        bundle=bundle,
        dimensions=dimensions,
        initial_state=initialize_interval_state(point_model),
        pretraining=result,
    )


def _prepare_starts(
    config: ExperimentConfig,
    device: torch.device,
    output_root: Path,
    reporter: ConsoleReporter,
) -> list[_PreparedStart]:
    """Select shared starts before running any certified-training method."""

    shared_bundle = _shared_bundle(config)

    selected: list[_PreparedStart] = []
    accepted_rows: list[dict[str, object]] = []
    rejected_rows: list[dict[str, object]] = []
    candidates, selection_manifest = _manifest_selected_seeds(config)
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
        try:
            prepared = _prepare_candidate(config, seed, device, shared_bundle)
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
        selected.append(prepared)
        result = prepared.pretraining
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
        "selection_manifest": selection_manifest,
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


def select_common_seeds(
    configs: list[ExperimentConfig],
    output_path: str | Path,
    *,
    group: str,
) -> Path:
    """Select the first seeds accepted by every configuration in a group."""

    if not configs:
        raise ValueError("common seed selection requires at least one configuration")
    for config in configs:
        config.validate()
        if config.seed_selection.manifest is not None:
            raise ValueError("common seed selection inputs must not already use a manifest")
    reference = configs[0].seed_selection
    if any(
        (config.seed_selection.start, config.seed_selection.stop, config.seed_selection.count)
        != (reference.start, reference.stop, reference.count)
        for config in configs[1:]
    ):
        raise ValueError("common seed selection requires one candidate range and count")
    if len({config.name for config in configs}) != len(configs):
        raise ValueError("common seed selection configuration names must be unique")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shared_bundles = {config.name: _shared_bundle(config) for config in configs}
    selected = []
    candidate_records = []
    for seed in reference.candidates():
        if len(selected) >= reference.count:
            break
        outcomes = {}
        accepted_by_all = True
        for config in configs:
            try:
                prepared = _prepare_candidate(
                    config,
                    seed,
                    device,
                    shared_bundles[config.name],
                )
                result = prepared.pretraining
                outcomes[config.name] = {
                    "accepted": True,
                    "initial_accuracy": result.initial_accuracy,
                    "pretraining_accuracy": result.final_accuracy,
                    "pretraining_epochs": result.epochs,
                }
            except PretrainingRejected as error:
                accepted_by_all = False
                outcomes[config.name] = {
                    "accepted": False,
                    "reason": error.reason,
                    "accuracy": error.accuracy,
                }
        if accepted_by_all:
            selected.append(seed)
        candidate_records.append(
            {
                "seed": seed,
                "accepted_by_all": accepted_by_all,
                "outcomes": outcomes,
            }
        )
    if len(selected) != reference.count:
        raise RuntimeError(
            f"only {len(selected)} of {reference.count} common starts passed within "
            f"[{reference.start}, {reference.stop})"
        )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "1.0" or not isinstance(
            payload.get("groups"), dict
        ):
            raise ValueError("existing common seed manifest has an incompatible schema")
    else:
        payload = {"schema_version": "1.0", "groups": {}}
    payload["groups"][group] = {
        "selection_rule": "first_candidates_accepted_by_every_member",
        "candidate_range": asdict(reference),
        "members": [config.name for config in configs],
        "selected_seeds": selected,
        "candidates_evaluated": candidate_records,
        "test_metrics_read": False,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


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
    final_test_enabled = config.evaluation.test_access == "frozen_final"
    resolved_config_hash = experiment_config_sha256(config)
    study_lock_payload = None
    study_lock_hash = None
    protocol_lock = None
    if final_test_enabled:
        study_lock_payload = verify_frozen_study_lock(
            config.evaluation.study_lock,
            protocol_name=config.name,
            config_sha256=resolved_config_hash,
            split_manifest=config.data.split_manifest,
            seed_selection_manifest=config.seed_selection.manifest,
        )
        study_lock_hash = canonical_mapping_sha256(study_lock_payload)
        protocol_lock = study_lock_payload["protocols"][config.name]
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
        calibration_inputs = bundle.calibration_inputs.to(device)
        calibration_targets = bundle.calibration_targets.to(device)
        calibration_loader = DataLoader(
            TensorDataset(calibration_inputs, calibration_targets),
            batch_size=config.data.calibration_size,
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
        hybrid_bounder = (
            AGTInspiredHybridBounder(dimensions, device)
            if any(_method_parts(name)[0] is BoundMethod.HYBRID for name in config.methods)
            else None
        )

        for radius in config.radii:
            evaluation_radius = config.evaluation.radius(radius)
            selection_bound_method = _configured_bound_method(
                config.evaluation.selection_bound_method
            )
            test_bound_methods = tuple(
                _configured_bound_method(name)
                for name in config.evaluation.bound_methods
            )
            fixed_runs = {}
            fixed_timings = {}
            fixed_peak_memory = {}
            fixed_diagnostics = {}
            for method_name in config.methods:
                bound_method, update_kind = _method_parts(method_name)
                two_stage = update_kind != "Fixed"
                fixed_name = bound_method.name
                gradient_diagnostics = IntersectionDiagnostics()
                calibration_diagnostics = IntersectionDiagnostics()

                def gradient_bounds(state, selected=bound_method):
                    flat_lower, flat_upper = flatten_parameters(state, linear_layers)
                    total_lower = torch.zeros_like(flat_lower)
                    total_upper = torch.zeros_like(flat_upper)
                    batch_size = config.training.gradient_batch_size
                    for start in range(0, len(train_inputs), batch_size):
                        end = min(start + batch_size, len(train_inputs))
                        inputs = train_inputs[start:end].reshape(-1, 1, bundle.input_dim)
                        input_lower, input_upper = inputs - radius, inputs + radius
                        bounded_inputs = bounded_tensor(input_lower, input_upper)
                        targets = train_targets[start:end]
                        if bundle.output_dim > 1:
                            targets = functional.one_hot(targets.long(), bundle.output_dim).float()
                        if selected is BoundMethod.HYBRID:
                            if hybrid_bounder is None:
                                raise RuntimeError("hybrid bounder was not initialized")
                            lower, upper = hybrid_bounder.compute(
                                bounded_inputs=bounded_inputs,
                                input_lower=input_lower,
                                input_upper=input_upper,
                                targets=targets,
                                flat_lower=flat_lower,
                                flat_upper=flat_upper,
                            )
                        else:
                            lower, upper = compute_gradient_bounds(
                                gradient_graph,
                                bounded_inputs,
                                targets,
                                flat_lower,
                                flat_upper,
                                dimensions,
                                selected,
                                diagnostics=gradient_diagnostics,
                            )
                        total_lower += lower.sum(dim=0).reshape(-1)
                        total_upper += upper.sum(dim=0).reshape(-1)
                    return total_lower / len(train_inputs), total_upper / len(train_inputs)

                def calibration_metrics(
                    state,
                    selected=selection_bound_method,
                    calibration_radius=evaluation_radius,
                ):
                    return evaluate_certification(
                        evaluation_graph,
                        calibration_loader,
                        state,
                        input_radius=calibration_radius,
                        method=selected,
                        dimensions=dimensions,
                        diagnostics=calibration_diagnostics,
                    )

                trainer = CertifiedTrainer(
                    gradient_bounds=gradient_bounds,
                    certified_accuracy=lambda state: calibration_metrics(
                        state
                    ).certified_accuracy,
                    clean_accuracy=lambda state: evaluate_center_accuracy(
                        calibration_loader, state, dimensions=dimensions
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
                        "selection_split": "calibration",
                        "selection_bound_method": selection_bound_method.name,
                        "test_bound_methods": [method.name for method in test_bound_methods],
                        "test_accessed_during_training": False,
                        "final_test_evaluation_enabled": final_test_enabled,
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
                run_environment = resolved_run_config["environment"]
                logger.write_config(resolved_run_config)
                continued_from_prefix = False
                try:
                    if two_stage:
                        if fixed_name not in fixed_runs:
                            logger.echo_internal_prerun_start(
                                method=fixed_name,
                                horizon=config.training.horizon,
                            )
                            _reset_peak_memory(device)
                            _synchronize(device)
                            dependency_start = time.perf_counter()
                            fixed_run = trainer.run(
                                initial_state,
                                FixedUpdate(config.training.base_step),
                                capture_snapshots=True,
                            )
                            _synchronize(device)
                            dependency_seconds = time.perf_counter() - dependency_start
                            dependency_peak_memory = _peak_memory(device)
                            fixed_runs[fixed_name] = fixed_run
                            fixed_timings[fixed_name] = dependency_seconds
                            fixed_peak_memory[fixed_name] = dependency_peak_memory
                            fixed_diagnostics[fixed_name] = {
                                "gradient": gradient_diagnostics,
                                "calibration": calibration_diagnostics,
                            }
                            dependency_logger = RunLogger(
                                cell_dir / "calibration_pre_run",
                                echo=False,
                            )
                            dependency_logger.write_config(
                                {
                                    **resolved_run_config,
                                    "method": fixed_name,
                                    "role": "shared_prefix_calibration_pre_run",
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
                        # The reported continuation has independent counters;
                        # the fixed pre-run remains separately auditable.
                        gradient_diagnostics = IntersectionDiagnostics()
                        calibration_diagnostics = IntersectionDiagnostics()
                        _reset_peak_memory(device)
                        _synchronize(device)
                        continuation_start = time.perf_counter()
                        calibration = calibrate_iwau(
                            fixed_runs[fixed_name],
                            horizon=config.training.horizon,
                            retention=config.training.retention,
                        )
                        if calibration.status.value == "invalid":
                            raise RuntimeError(
                                f"invalid shared-prefix calibration: {calibration.reason}"
                            )
                        total_steps = (
                            config.training.horizon - calibration.peak_epoch
                        )
                        adaptive_kinds = {"IWAU", "GlobalIWAU"}
                        if total_steps <= 0:
                            run = fixed_runs[fixed_name]
                        elif (
                            update_kind in adaptive_kinds
                            and not calibration.iwau_effective
                        ):
                            # Lambda=0 means that width adaptation is not
                            # required.  This rule is specific to the IWAU
                            # family; scalar fairness controls still run from
                            # the same B_tp prefix independently.
                            run = fixed_runs[fixed_name]
                        elif update_kind == "IWAU":
                            result = continue_with_iwau(
                                trainer,
                                fixed_runs[fixed_name],
                                base_step=config.training.base_step,
                                retention=config.training.retention,
                                coefficient_scale=config.training.coefficient_scale,
                            )
                            run = result.run
                            calibration = result.calibration
                            continued_from_prefix = True
                        else:
                            if update_kind == "GlobalIWAU":
                                update_rule = GlobalWidthAdaptiveUpdate(
                                    base_step=config.training.base_step,
                                    coefficient=(
                                        calibration.coefficient
                                        * config.training.coefficient_scale
                                    ),
                                )
                            elif update_kind == "CumulativeCosine":
                                target_mean = config.training.cumulative_target_mean_step
                                if target_mean is None:
                                    raise RuntimeError(
                                        "cumulative schedule requires cumulative_target_mean_step"
                                    )
                                update_rule = cumulative_matched_cosine_rule(
                                    base_step=config.training.base_step,
                                    target_mean_step=target_mean,
                                    start_epoch=calibration.peak_epoch,
                                    total_steps=total_steps,
                                )
                            elif update_kind == "TunedScalar":
                                target_step = (
                                    config.training.tuned_schedule_target_final_step
                                )
                                family = config.training.tuned_schedule_family
                                if target_step is None or family is None:
                                    raise RuntimeError(
                                        "tuned scalar schedule is not fully configured"
                                    )
                                rules = matched_final_step_rules(
                                    base_step=config.training.base_step,
                                    target_final_step=target_step,
                                    start_epoch=calibration.peak_epoch,
                                    total_steps=total_steps,
                                    step_milestone_fraction=(
                                        config.training.tuned_step_decay_milestone_fraction
                                    ),
                                )
                                update_rule = rules[family]
                            else:
                                target_step = (
                                    config.training.schedule_target_final_step
                                )
                                if target_step is None:
                                    raise RuntimeError(
                                        "matched schedule requires schedule_target_final_step"
                                    )
                                rules = matched_final_step_rules(
                                    base_step=config.training.base_step,
                                    target_final_step=target_step,
                                    start_epoch=calibration.peak_epoch,
                                    total_steps=total_steps,
                                    step_milestone_fraction=(
                                        config.training.step_decay_milestone_fraction
                                    ),
                                )
                                rule_names = {
                                    "InverseTime": "inverse_time",
                                    "Cosine": "cosine",
                                    "StepDecay": "step_decay",
                                }
                                update_rule = rules[rule_names[update_kind]]
                            run = continue_from_fixed_prefix(
                                trainer,
                                fixed_runs[fixed_name],
                                start_epoch=calibration.peak_epoch,
                                update_rule=update_rule,
                            )
                            continued_from_prefix = True
                        _synchronize(device)
                        continuation_seconds = time.perf_counter() - continuation_start
                        continuation_peak_memory = _peak_memory(device)
                        pre_run_seconds = fixed_timings[fixed_name]
                        pre_run_peak_memory = fixed_peak_memory[fixed_name]
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
                            "iwau_effective": (
                                calibration.iwau_effective
                                and update_kind in adaptive_kinds
                            ),
                            "update_effective": continued_from_prefix,
                            "update_rule": update_kind,
                            "matched_final_mean_step": (
                                config.training.schedule_target_final_step
                                if update_kind
                                in {"InverseTime", "Cosine", "StepDecay"}
                                else None
                            ),
                            "matched_cumulative_mean_step": (
                                config.training.cumulative_target_mean_step
                                if update_kind == "CumulativeCosine"
                                else None
                            ),
                            "tuned_schedule_family": (
                                config.training.tuned_schedule_family
                                if update_kind == "TunedScalar"
                                else None
                            ),
                            "tuned_schedule_target_final_step": (
                                config.training.tuned_schedule_target_final_step
                                if update_kind == "TunedScalar"
                                else None
                            ),
                        }
                        logger.echo_calibration(
                            source_method=fixed_name,
                            status=calibration.status.value,
                            peak_epoch=calibration.peak_epoch,
                            reference_epoch=calibration.reference_epoch,
                            coefficient=calibration.coefficient,
                            update_rule=update_kind,
                        )
                    else:
                        if fixed_name in fixed_runs:
                            run = fixed_runs[fixed_name]
                            elapsed = fixed_timings[fixed_name]
                            logger.echo_run(run)
                        else:
                            _reset_peak_memory(device)
                            _synchronize(device)
                            fixed_start = time.perf_counter()
                            run = trainer.run(
                                initial_state,
                                FixedUpdate(config.training.base_step),
                                capture_snapshots=True,
                                on_epoch=logger.epoch_callback,
                            )
                            fixed_runs[fixed_name] = run
                            _synchronize(device)
                            elapsed = time.perf_counter() - fixed_start
                            fixed_timings[fixed_name] = elapsed
                            fixed_peak_memory[fixed_name] = _peak_memory(device)
                            fixed_diagnostics[fixed_name] = {
                                "gradient": gradient_diagnostics,
                                "calibration": calibration_diagnostics,
                            }
                        pre_run_seconds = 0.0
                        pre_run_peak_memory = None
                        continuation_seconds = elapsed
                        continuation_peak_memory = fixed_peak_memory[fixed_name]
                        calibration_summary = {
                            "calibration_status": "not_applicable",
                            "iwau_effective": False,
                            "update_effective": False,
                            "update_rule": "Fixed",
                        }
                    if two_stage and config.echo_epochs:
                        logger.echo_run(run)
                    # Development runs fail closed: test tensors are not passed
                    # to any evaluator. Final test access requires a separately
                    # validated frozen study lock.
                    if final_test_enabled:
                        test_loader = DataLoader(
                            TensorDataset(
                                bundle.test_inputs.to(device),
                                bundle.test_targets.to(device),
                            ),
                            batch_size=config.data.test_size,
                            shuffle=False,
                        )
                        test_diagnostics = {
                            method: IntersectionDiagnostics()
                            for method in test_bound_methods
                        }
                        test_diagnostics_by_name = {
                            method.name: diagnostics
                            for method, diagnostics in test_diagnostics.items()
                        }
                        _reset_peak_memory(device)
                        _synchronize(device)
                        test_start = time.perf_counter()
                        frozen_evaluations = evaluate_frozen_checkpoints(
                            run=run,
                            evaluation_graph=evaluation_graph,
                            test_loader=test_loader,
                            input_radius=evaluation_radius,
                            methods=test_bound_methods,
                            dimensions=dimensions,
                            diagnostics=test_diagnostics,
                        )
                        _synchronize(device)
                        test_seconds = time.perf_counter() - test_start
                        test_peak_memory = _peak_memory(device)
                        test_evaluations = [
                            evaluation.as_record()
                            for evaluation in frozen_evaluations
                        ]
                    else:
                        test_diagnostics = {}
                        test_diagnostics_by_name = {}
                        test_seconds = 0.0
                        test_peak_memory = None
                        test_evaluations = [
                            {
                                "evaluation_method": "not_evaluated",
                                "test_cert_at_selected": None,
                                "test_cert_at_T": None,
                                "test_clean_at_selected": None,
                                "test_clean_at_T": None,
                            }
                        ]
                    summary_test_evaluations = (
                        test_evaluations if final_test_enabled else []
                    )
                    complete_elapsed_seconds = elapsed + test_seconds
                    peak_device_memory = _maximum_memory(
                        pre_run_peak_memory,
                        continuation_peak_memory,
                        test_peak_memory,
                    )
                    result_rows = []
                    for evaluation_record in test_evaluations:
                        result_rows.append(
                            {
                                "schema_version": SCHEMA_VERSION,
                                "radius": radius,
                                "evaluation_radius": evaluation_radius,
                                "seed": seed,
                                "method": method_name,
                                "training_bound_method": bound_method.name,
                                **evaluation_record,
                                "pretraining_accuracy": 100.0 * pretrain_accuracy,
                                "peak_calibration_certified_accuracy": (
                                    run.peak_certified_accuracy
                                ),
                                "selected_epoch": run.peak_epoch,
                                "calibration_cert_at_T": run.cert_at_horizon,
                                "status": run.status.value,
                                "iwau_status": calibration_summary[
                                    "calibration_status"
                                ],
                                "update_status": calibration_summary[
                                    "calibration_status"
                                ],
                                "iwau_effective": calibration_summary[
                                    "iwau_effective"
                                ],
                                "update_effective": calibration_summary[
                                    "update_effective"
                                ],
                                "update_rule": calibration_summary["update_rule"],
                                "test_access": config.evaluation.test_access,
                                "analysis_tier": config.evaluation.analysis_tier,
                                "selection_split": "calibration",
                                "selection_metric": (
                                    f"calibration_certified_accuracy_"
                                    f"{selection_bound_method.name}"
                                ),
                                "experiment_lock_hash": study_lock_hash,
                                "code_commit": (
                                    study_lock_payload["code"]["git_commit"]
                                    if study_lock_payload is not None
                                    else run_environment.get("git_commit")
                                ),
                                "code_dirty": run_environment.get("git_dirty"),
                                "resolved_config_sha256": resolved_config_hash,
                                "data_split_sha256": (
                                    protocol_lock.get("split_manifest_sha256")
                                    if protocol_lock is not None
                                    else None
                                ),
                                "elapsed_seconds": elapsed,
                                "test_evaluation_seconds": test_seconds,
                                "complete_elapsed_seconds": complete_elapsed_seconds,
                                "peak_device_memory_bytes": peak_device_memory,
                                "intersection_fallbacks_test": (
                                    test_diagnostics_by_name.get(
                                        evaluation_record["evaluation_method"],
                                        IntersectionDiagnostics(),
                                    ).fallback_coordinates
                                ),
                            }
                        )

                    continuation_origin = calibration.peak_epoch if two_stage else 0
                    continuation_step_values = (
                        [
                            update.effective_step_mean
                            for update in run.updates
                            if update.epoch >= continuation_origin
                        ]
                        if two_stage and continued_from_prefix
                        else []
                    )
                    realized_continuation_step_sum = sum(continuation_step_values)
                    realized_continuation_step_mean = (
                        realized_continuation_step_sum / len(continuation_step_values)
                        if continuation_step_values
                        else None
                    )
                    realized_continuation_final_step = (
                        continuation_step_values[-1]
                        if continuation_step_values
                        else None
                    )
                    if two_stage and continued_from_prefix:
                        budget = budget_record(
                            fixed_runs[fixed_name],
                            run,
                            start_epoch=calibration.peak_epoch,
                        )
                        executed_updates = budget.total_bound_evaluations
                    else:
                        executed_updates = len(run.updates)
                    logger.write_run(
                        run,
                        extra_summary={
                            **calibration_summary,
                            "pretraining_accuracy": 100.0 * pretrain_accuracy,
                            "elapsed_seconds": elapsed,
                            "test_evaluation_seconds": test_seconds,
                            "complete_elapsed_seconds": complete_elapsed_seconds,
                            "pre_run_seconds": pre_run_seconds,
                            "continuation_seconds": continuation_seconds,
                            "pre_run_peak_device_memory_bytes": pre_run_peak_memory,
                            "continuation_peak_device_memory_bytes": (
                                continuation_peak_memory
                            ),
                            "test_peak_device_memory_bytes": test_peak_memory,
                            "peak_device_memory_bytes": peak_device_memory,
                            "executed_updates": executed_updates,
                            "seconds_per_executed_update": elapsed
                            / max(executed_updates, 1),
                            "wall_seconds_per_reported_epoch": elapsed
                            / max(config.training.horizon, 1),
                            "continuation_origin_epoch": continuation_origin,
                            "realized_continuation_updates": len(
                                continuation_step_values
                            ),
                            "realized_continuation_step_sum": (
                                realized_continuation_step_sum
                            ),
                            "realized_continuation_step_mean": (
                                realized_continuation_step_mean
                            ),
                            "realized_continuation_final_step": (
                                realized_continuation_final_step
                            ),
                            "primary_endpoint": (
                                "test_cert_at_calibration_selected_checkpoint"
                                if final_test_enabled
                                else "not_evaluated_in_development"
                            ),
                            "analysis_tier": config.evaluation.analysis_tier,
                            "selection_split": "calibration",
                            "selection_metric": (
                                f"calibration_certified_accuracy_"
                                f"{selection_bound_method.name}"
                            ),
                            "experiment_lock_hash": study_lock_hash,
                            "code_commit": (
                                study_lock_payload["code"]["git_commit"]
                                if study_lock_payload is not None
                                else run_environment.get("git_commit")
                            ),
                            "code_dirty": run_environment.get("git_dirty"),
                            "resolved_config_sha256": resolved_config_hash,
                            "data_split_sha256": (
                                protocol_lock.get("split_manifest_sha256")
                                if protocol_lock is not None
                                else None
                            ),
                            "test_evaluations": summary_test_evaluations,
                            "intersection_diagnostics": {
                                "fixed_pre_run": {
                                    key: value.as_dict()
                                    for key, value in (
                                        fixed_diagnostics.get(fixed_name, {})
                                        if two_stage
                                        else {}
                                    ).items()
                                },
                                "reported_trajectory": {
                                    "gradient": gradient_diagnostics.as_dict(),
                                    "calibration": calibration_diagnostics.as_dict(),
                                },
                                "test": {
                                    method.name: diagnostics.as_dict()
                                    for method, diagnostics in test_diagnostics.items()
                                },
                            },
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
                index_rows.extend(result_rows)
                seed_rows.extend(result_rows)
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
