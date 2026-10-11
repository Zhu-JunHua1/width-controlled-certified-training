"""Run the frozen official FullCert comparison through the common evaluator.

This entry point is deliberately separate from the paper's native trainer.  It
loads the audited official FullCert checkout, reproduces the already-frozen
point initialization, exports the complete parameter-box trajectory, selects
one checkpoint on calibration data, and opens test data only when a matching
frozen study lock authorizes it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import torch
import yaml
from auto_LiRPA import BoundedModule
from torch.utils.data import DataLoader, TensorDataset

from width_cert.baselines.fullcert import load_fullcert, run_fullcert_trajectory
from width_cert.bounds import BoundMethod, CUSTOM_BOUND_OPERATORS
from width_cert.bounds.lirpa import build_trace_inputs, initialize_interval_state
from width_cert.certification import evaluate_certification
from width_cert.data import build_dataset
from width_cert.data.manifests import canonical_json_sha256
from width_cert.experiments.environment import environment_metadata, sha256
from width_cert.experiments.pretrain import pretrain_to_accuracy
from width_cert.experiments.study_lock import (
    canonical_mapping_sha256,
    verify_frozen_study_lock,
)
from width_cert.models import EvaluationGraph, PretrainingMLP


METHOD = "FullCert-official"
SCHEMA_VERSION = "1.0"


def _load_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a mapping in {path}")
    return payload


def _mapping_sha256(payload: dict[str, Any]) -> str:
    return canonical_mapping_sha256(payload)


def _selected_starts(config: dict[str, Any], root: Path) -> tuple[list[int], Path]:
    starts = config["starts"]
    manifest_path = (root / starts["manifest"]).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    group = manifest["groups"][starts["group"]]
    selected = [int(value) for value in group["selected_seeds"]]
    expected = [int(value) for value in starts["expected"]]
    if selected != expected or len(selected) != len(set(selected)):
        raise ValueError("seed-selection manifest does not match the declared starts")
    if group.get("test_metrics_read") is not False:
        raise ValueError("seed selection must explicitly exclude test metrics")
    return selected, manifest_path


def _validate_config(config: dict[str, Any], root: Path) -> None:
    if config.get("official_source", {}).get("commit") != (
        "3387a8af3ab143fae238b5d3a3f5b24fa815cca6"
    ):
        raise ValueError("the comparison is not bound to the audited FullCert commit")
    source = config["official_source"]
    adapter = (root / source["adapter"]).resolve()
    if sha256(adapter) != source["adapter_sha256"]:
        raise ValueError("FullCert adapter hash mismatch")
    runner_path = source.get("runner")
    runner_hash = source.get("runner_sha256")
    if runner_path is not None or runner_hash is not None:
        if not runner_path or not runner_hash:
            raise ValueError("FullCert runner provenance is incomplete")
        if sha256((root / runner_path).resolve()) != runner_hash:
            raise ValueError("FullCert comparison runner hash mismatch")
    if config["model"] != {
        "dimensions": [2, 20, 20, 1],
        "loss": "binary_cross_entropy_with_logits",
    }:
        raise ValueError("unexpected model or loss")
    training = config["training"]
    if training["radii"] != [0.0001, 0.001, 0.01]:
        raise ValueError("unexpected radius grid")
    if training["horizon"] != 40 or training["early_stopping"] is not False:
        raise ValueError("FullCert comparison requires fixed T=40")
    expected_lr = training["paper_mean_gradient_step"] * training["full_batch_size"]
    if abs(training["official_api_learning_rate"] - expected_lr) > 1e-12:
        raise ValueError("official API learning rate does not compensate its double mean")
    evaluation = config["evaluation"]
    if evaluation["common_test_certifiers"] != ["IBP", "IC"]:
        raise ValueError("common evaluator grid must contain IBP and IC")
    starts, _ = _selected_starts(config, root)
    if evaluation["expected_summaries"] != len(starts) * len(training["radii"]):
        raise ValueError("incorrect expected summary count")
    if evaluation["expected_evaluator_rows"] != 2 * evaluation["expected_summaries"]:
        raise ValueError("incorrect expected evaluator-row count")


def _finite_state(state: dict[str, torch.Tensor]) -> bool:
    return all(bool(torch.isfinite(value).all()) for value in state.values())


def _checkout_matches_frozen_code(
    root: Path,
    *,
    current_commit: str,
    executable_commit: str,
    metadata_paths: set[str],
) -> bool:
    """Accept the executable commit or its single metadata-only lock commit."""

    if current_commit == executable_commit:
        return True
    parent = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", f"{current_commit}^"],
        text=True,
    ).strip()
    if parent != executable_commit:
        return False
    changed = set(
        subprocess.check_output(
            ["git", "-C", str(root), "diff", "--name-only", executable_commit, current_commit],
            text=True,
        ).splitlines()
    )
    return changed == metadata_paths


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _evaluator(dimensions: list[int], device: torch.device) -> BoundedModule:
    _, dummy = build_trace_inputs(dimensions, device)
    return BoundedModule(
        EvaluationGraph(len(dimensions) - 1),
        dummy,
        custom_ops=CUSTOM_BOUND_OPERATORS,
        bound_opts={"override_dependent_operations": True},
    )


def _metrics(
    graph: BoundedModule,
    loader: DataLoader,
    state: dict[str, torch.Tensor],
    *,
    radius: float,
    method: BoundMethod,
    dimensions: list[int],
) -> dict[str, float]:
    value = evaluate_certification(
        graph,
        loader,
        state,
        input_radius=radius,
        method=method,
        dimensions=dimensions,
    )
    return {
        "clean_accuracy": value.clean_accuracy,
        "certified_accuracy": value.certified_accuracy,
    }


def _assert_pretraining_manifest(
    manifest_path: Path,
    group_name: str,
    seed: int,
    *,
    initial: float,
    prepared: float,
    epochs: int,
) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    group = payload["groups"][group_name]
    candidate = next(item for item in group["candidates_evaluated"] if item["seed"] == seed)
    if not candidate["accepted_by_all"]:
        raise ValueError(f"seed {seed} was not accepted by the frozen manifest")
    outcome = next(iter(candidate["outcomes"].values()))
    checks = (
        abs(float(outcome["initial_accuracy"]) - initial) <= 1e-7,
        abs(float(outcome["pretraining_accuracy"]) - prepared) <= 1e-7,
        int(outcome["pretraining_epochs"]) == epochs,
    )
    if not all(checks):
        raise ValueError(f"seed {seed} pretraining does not reproduce its frozen audit")


def run(config_path: Path, fullcert_root: Path, repository_root: Path) -> Path:
    config = _load_mapping(config_path)
    _validate_config(config, repository_root)
    config_hash = _mapping_sha256(config)
    starts, manifest_path = _selected_starts(config, repository_root)
    final = config.get("test_access") == "frozen_final"
    if config.get("test_access") not in {"disabled", "frozen_final"}:
        raise ValueError("test_access must be disabled or frozen_final")

    lock_payload = None
    lock_hash = None
    if final:
        lock_path = (repository_root / config["study_lock"]).resolve()
        lock_payload = verify_frozen_study_lock(
            lock_path,
            protocol_name=config["protocol_name"],
            config_sha256=config_hash,
            seed_selection_manifest=manifest_path,
        )
        lock_hash = _mapping_sha256(lock_payload)

    environment = environment_metadata()
    if final:
        frozen_commit = lock_payload["code"]["git_commit"]
        metadata_paths = {
            config["study_lock"].replace("\\", "/"),
            "tests/test_fullcert_protocol.py",
        }
        if environment["git_dirty"] or not _checkout_matches_frozen_code(
            repository_root,
            current_commit=environment["git_commit"],
            executable_commit=frozen_commit,
            metadata_paths=metadata_paths,
        ):
            raise ValueError("final FullCert comparison requires its clean executable commit")

    output_root = (repository_root / config["output_dir"]).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite or rerun an existing comparison: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    api = load_fullcert(fullcert_root)
    data = config["data"]
    bundle = build_dataset(
        data["dataset"],
        train_size=data["train_size"],
        calibration_size=data["calibration_size"],
        test_size=data["test_size"],
        seed=data["train_seed"],
        calibration_seed=data["calibration_seed"],
        test_seed=data["test_seed"],
        strategy=data["strategy"],
    )
    device = torch.device("cpu")
    dimensions = [int(value) for value in config["model"]["dimensions"]]
    graph = _evaluator(dimensions, device)
    calibration_loader = DataLoader(
        TensorDataset(bundle.calibration_inputs, bundle.calibration_targets),
        batch_size=data["calibration_size"],
        shuffle=False,
    )
    test_loader = None
    if final:
        test_loader = DataLoader(
            TensorDataset(bundle.test_inputs, bundle.test_targets),
            batch_size=data["test_size"],
            shuffle=False,
        )

    rows: list[dict[str, Any]] = []
    pretraining = config["pretraining"]
    training = config["training"]
    for seed in starts:
        torch.manual_seed(seed)
        point_model = PretrainingMLP(dimensions)
        train_loader = DataLoader(
            TensorDataset(bundle.train_inputs, bundle.train_targets),
            batch_size=data["train_size"],
            shuffle=False,
        )
        prepared = pretrain_to_accuracy(
            point_model,
            train_loader,
            bundle.calibration_inputs,
            bundle.calibration_targets,
            target_accuracy=pretraining["target_accuracy"],
            maximum_initial_accuracy=pretraining["maximum_initial_accuracy"],
            maximum_final_accuracy=pretraining["maximum_final_accuracy"],
            step_size=pretraining["step_size"],
            maximum_epochs=pretraining["maximum_epochs"],
            minimum_epochs=pretraining.get("minimum_epochs", 0),
            minimum_accuracy_gain=pretraining.get("minimum_accuracy_gain", 0.0),
            multiclass=False,
            enabled=pretraining["enabled"],
            optimizer_name=pretraining["optimizer"],
        )
        _assert_pretraining_manifest(
            manifest_path,
            config["starts"]["group"],
            seed,
            initial=prepared.initial_accuracy,
            prepared=prepared.final_accuracy,
            epochs=prepared.epochs,
        )
        initial_state = initialize_interval_state(point_model)

        for radius in training["radii"]:
            run_dir = output_root / f"eps_{radius:g}" / f"seed_{seed}" / METHOD
            boxes_dir = run_dir / "trajectory"
            run_dir.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            trajectory = run_fullcert_trajectory(
                api=api,
                initial_state=initial_state,
                dimensions=dimensions,
                train_inputs=bundle.train_inputs,
                train_targets=bundle.train_targets,
                calibration_inputs=bundle.calibration_inputs,
                calibration_targets=bundle.calibration_targets,
                radius=radius,
                learning_rate=training["official_api_learning_rate"],
                epochs=training["horizon"],
                output_dir=boxes_dir,
            )
            elapsed = time.perf_counter() - started

            calibration_records: list[dict[str, float | int]] = []
            divergence_reason = None
            for epoch, state in enumerate(trajectory):
                if not _finite_state(state):
                    divergence_reason = f"nonfinite_parameter_box_at_epoch_{epoch}"
                    break
                try:
                    value = _metrics(
                        graph,
                        calibration_loader,
                        state,
                        radius=radius,
                        method=BoundMethod.IC,
                        dimensions=dimensions,
                    )
                except (RuntimeError, ValueError, FloatingPointError) as error:
                    divergence_reason = (
                        f"calibration_evaluator_failure_at_epoch_{epoch}:"
                        f"{type(error).__name__}:{error}"
                    )
                    break
                calibration_records.append({"epoch": epoch, **value})
            if not calibration_records:
                raise RuntimeError(f"no valid calibration checkpoint for seed={seed}, radius={radius}")
            selected_index = max(
                range(len(calibration_records)),
                key=lambda index: calibration_records[index]["certified_accuracy"],
            )
            selected_epoch = int(calibration_records[selected_index]["epoch"])
            status = (
                "completed"
                if divergence_reason is None and len(trajectory) == training["horizon"] + 1
                else "diverged"
            )

            test_evaluations: list[dict[str, Any]] = []
            if final:
                for method in (BoundMethod.IBP, BoundMethod.IC):
                    selected_metrics = _metrics(
                        graph,
                        test_loader,
                        trajectory[selected_epoch],
                        radius=radius,
                        method=method,
                        dimensions=dimensions,
                    )
                    terminal_metrics = None
                    if status == "completed":
                        terminal_metrics = _metrics(
                            graph,
                            test_loader,
                            trajectory[-1],
                            radius=radius,
                            method=method,
                            dimensions=dimensions,
                        )
                    record = {
                        "evaluation_method": method.name,
                        "test_cert_at_selected": selected_metrics["certified_accuracy"],
                        "test_cert_at_T": (
                            terminal_metrics["certified_accuracy"]
                            if terminal_metrics is not None
                            else None
                        ),
                        "test_clean_at_selected": selected_metrics["clean_accuracy"],
                        "test_clean_at_T": (
                            terminal_metrics["clean_accuracy"]
                            if terminal_metrics is not None
                            else None
                        ),
                    }
                    test_evaluations.append(record)
                    rows.append(
                        {
                            "schema_version": SCHEMA_VERSION,
                            "radius": radius,
                            "evaluation_radius": radius,
                            "seed": seed,
                            "method": METHOD,
                            "training_bound_method": "official_FullCert_IBP",
                            **record,
                            "pretraining_accuracy": 100.0 * prepared.final_accuracy,
                            "peak_calibration_certified_accuracy": calibration_records[selected_index]["certified_accuracy"],
                            "selected_epoch": selected_epoch,
                            "calibration_cert_at_T": (
                                calibration_records[-1]["certified_accuracy"]
                                if status == "completed"
                                else None
                            ),
                            "status": status,
                            "failure_reason": divergence_reason,
                            "test_access": config["test_access"],
                            "analysis_tier": config["analysis_tier"],
                            "selection_split": "calibration",
                            "selection_metric": "calibration_certified_accuracy_IC",
                            "experiment_lock_hash": lock_hash,
                            "code_commit": (
                                lock_payload["code"]["git_commit"] if final else environment["git_commit"]
                            ),
                            "code_dirty": environment["git_dirty"],
                            "resolved_config_sha256": config_hash,
                            "data_split_sha256": "independent_generators",
                            "elapsed_seconds": elapsed,
                            "complete_elapsed_seconds": elapsed,
                            "executed_updates": len(trajectory) - 1,
                            "seconds_per_executed_update": elapsed / max(len(trajectory) - 1, 1),
                            "peak_device_memory_bytes": None,
                        }
                    )

            summary = {
                "schema_version": SCHEMA_VERSION,
                "method": METHOD,
                "radius": radius,
                "seed": seed,
                "status": status,
                "failure_reason": divergence_reason,
                "official_source": {
                    "remote": api.source.remote,
                    "commit": api.source.commit,
                    "license_sha256": api.source.license_sha256,
                },
                "normalization": {
                    "paper_mean_gradient_step": training["paper_mean_gradient_step"],
                    "official_api_learning_rate": training["official_api_learning_rate"],
                    "factor": training["full_batch_size"],
                },
                "pretraining_accuracy": 100.0 * prepared.final_accuracy,
                "selected_epoch": selected_epoch,
                "peak_calibration_certified_accuracy": calibration_records[selected_index]["certified_accuracy"],
                "calibration_trajectory": calibration_records,
                "elapsed_seconds": elapsed,
                "executed_updates": len(trajectory) - 1,
                "seconds_per_executed_update": elapsed / max(len(trajectory) - 1, 1),
                "primary_endpoint": (
                    "test_cert_at_calibration_selected_checkpoint"
                    if final
                    else "not_evaluated_in_development"
                ),
                "analysis_tier": config["analysis_tier"],
                "selection_split": "calibration",
                "selection_metric": "calibration_certified_accuracy_IC",
                "experiment_lock_hash": lock_hash,
                "code_commit": (
                    lock_payload["code"]["git_commit"] if final else environment["git_commit"]
                ),
                "code_dirty": environment["git_dirty"],
                "resolved_config_sha256": config_hash,
                "seed_selection_manifest_sha256": canonical_json_sha256(manifest_path),
                "test_evaluations": test_evaluations,
            }
            (run_dir / "summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if final:
                _write_csv(output_root / "results.csv", rows)
            print(
                f"[FULLCERT] radius={radius:g} seed={seed} "
                f"status={status} selected_epoch={selected_epoch} "
                f"calibration_peak={calibration_records[selected_index]['certified_accuracy']:.2f}%"
            )

    if final:
        expected = config["evaluation"]
        if len(rows) != expected["expected_evaluator_rows"]:
            raise RuntimeError(f"expected {expected['expected_evaluator_rows']} rows, found {len(rows)}")
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fullcert-root", type=Path, required=True)
    args = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    print(run(args.config.resolve(), args.fullcert_root.resolve(), repository_root))


if __name__ == "__main__":
    main()
