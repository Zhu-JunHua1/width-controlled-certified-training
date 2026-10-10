"""Development-only smoke audit for the constrained official FullCert adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import torch
from auto_LiRPA import BoundedModule
from torch.utils.data import DataLoader, TensorDataset

from width_cert.baselines.fullcert import load_fullcert, run_fullcert_trajectory
from width_cert.bounds import BoundMethod, CUSTOM_BOUND_OPERATORS
from width_cert.bounds.lirpa import build_trace_inputs, initialize_interval_state
from width_cert.certification import evaluate_certification
from width_cert.data import build_dataset
from width_cert.experiments import load_config
from width_cert.experiments.pretrain import pretrain_to_accuracy
from width_cert.models import EvaluationGraph, PretrainingMLP, layer_dimensions


def _tensor_sha256(*tensors: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode("ascii"))
        digest.update(_tensor_sha256(state[key]).encode("ascii"))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fullcert-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--radius", type=float, default=0.0001)
    args = parser.parse_args()

    config = load_config(args.config)
    if config.evaluation.test_access != "disabled":
        raise ValueError("FullCert adapter smoke requires test_access=disabled")
    if args.radius not in config.radii:
        raise ValueError("smoke radius must be declared by the development config")
    api = load_fullcert(args.fullcert_root)
    bundle = build_dataset(
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
    dimensions = layer_dimensions(
        bundle.input_dim,
        config.model.hidden_width,
        bundle.output_dim,
        config.model.hidden_layers,
    )
    torch.manual_seed(args.seed)
    point_model = PretrainingMLP(dimensions)
    train_loader = DataLoader(
        TensorDataset(bundle.train_inputs, bundle.train_targets),
        batch_size=config.data.train_size,
        shuffle=False,
    )
    pretraining = pretrain_to_accuracy(
        point_model,
        train_loader,
        bundle.calibration_inputs,
        bundle.calibration_targets,
        target_accuracy=config.pretraining.target_accuracy,
        maximum_initial_accuracy=config.pretraining.maximum_initial_accuracy,
        maximum_final_accuracy=config.pretraining.maximum_final_accuracy,
        step_size=config.pretraining.step_size,
        maximum_epochs=config.pretraining.maximum_epochs,
        minimum_epochs=config.pretraining.minimum_epochs,
        minimum_accuracy_gain=config.pretraining.minimum_accuracy_gain,
        multiclass=False,
        enabled=config.pretraining.enabled,
        optimizer_name=config.pretraining.optimizer,
    )
    initial_state = initialize_interval_state(point_model)

    # FullCert's BCE derivative and LinearLayer both divide by the full-batch
    # size.  Multiplying the API learning-rate argument by N preserves the
    # paper protocol's mathematical mean-gradient step without changing the
    # official forward, backward, or update operators.
    official_learning_rate = config.training.base_step * config.data.train_size
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="fullcert-adapter-smoke-", dir=args.output.parent
    ) as temporary:
        trajectory = run_fullcert_trajectory(
            api=api,
            initial_state=initial_state,
            dimensions=dimensions,
            train_inputs=bundle.train_inputs,
            train_targets=bundle.train_targets,
            calibration_inputs=bundle.calibration_inputs,
            calibration_targets=bundle.calibration_targets,
            radius=args.radius,
            learning_rate=official_learning_rate,
            epochs=config.training.horizon,
            output_dir=temporary,
        )
        exported_boxes = sorted(Path(temporary).glob("box_epoch_*.pt"))
        export_hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in exported_boxes]

    _, evaluation_dummy = build_trace_inputs(dimensions, torch.device("cpu"))
    evaluator = BoundedModule(
        EvaluationGraph(len(dimensions) - 1),
        evaluation_dummy,
        custom_ops=CUSTOM_BOUND_OPERATORS,
        bound_opts={"override_dependent_operations": True},
    )
    calibration_loader = DataLoader(
        TensorDataset(bundle.calibration_inputs, bundle.calibration_targets),
        batch_size=config.data.calibration_size,
        shuffle=False,
    )
    calibration_ic = [
        evaluate_certification(
            evaluator,
            calibration_loader,
            state,
            input_radius=args.radius,
            method=BoundMethod.IC,
            dimensions=dimensions,
        ).certified_accuracy
        for state in trajectory
    ]
    selected_epoch = calibration_ic.index(max(calibration_ic))
    checkpoints = {
        "initial": trajectory[0],
        "selected": trajectory[selected_epoch],
        "terminal": trajectory[-1],
    }
    common_evaluator = {}
    for checkpoint, state in checkpoints.items():
        common_evaluator[checkpoint] = {
            method.name: evaluate_certification(
                evaluator,
                calibration_loader,
                state,
                input_radius=args.radius,
                method=method,
                dimensions=dimensions,
            ).certified_accuracy
            for method in (BoundMethod.IBP, BoundMethod.IC)
        }

    report = {
        "schema_version": "1.0",
        "analysis_tier": "development",
        "test_metrics_read": False,
        "official_source": {
            "remote": api.source.remote,
            "commit": api.source.commit,
            "license_sha256": api.source.license_sha256,
        },
        "protocol": {
            "dimensions": dimensions,
            "train_size": config.data.train_size,
            "calibration_size": config.data.calibration_size,
            "radius": args.radius,
            "epochs": config.training.horizon,
            "paper_mean_gradient_step": config.training.base_step,
            "official_api_learning_rate": official_learning_rate,
            "normalization_compensation": config.data.train_size,
            "early_stopping": False,
        },
        "split_isolation": {
            "adapter_accepts_test_tensors": False,
            "train_sha256": _tensor_sha256(bundle.train_inputs, bundle.train_targets),
            "calibration_sha256": _tensor_sha256(
                bundle.calibration_inputs, bundle.calibration_targets
            ),
        },
        "pretraining": {
            "seed": args.seed,
            "initial_accuracy": pretraining.initial_accuracy,
            "prepared_accuracy": pretraining.final_accuracy,
            "epochs": pretraining.epochs,
        },
        "trajectory": {
            "states": len(trajectory),
            "exported_boxes": len(export_hashes),
            "box_sha256": export_hashes,
            "state_sha256": [_state_sha256(state) for state in trajectory],
        },
        "checkpoint_selection": {
            "split": "calibration",
            "certifier": "IC",
            "rule": "first_argmax",
            "selected_epoch": selected_epoch,
            "calibration_ic_trajectory": calibration_ic,
        },
        "common_evaluator_calibration_only": common_evaluator,
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
