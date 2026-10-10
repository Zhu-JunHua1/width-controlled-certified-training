"""Constrained adapter for the official FullCert BoundFlow implementation.

The external checkout remains the implementation of record.  This module only
supplies explicit tensors, converts parameter layouts at the boundary, invokes
the official training loop unchanged, and exports exact interval endpoints.
It intentionally has no final-test or checkpoint-selection API.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Iterator

import torch
from torch.utils.data import DataLoader, TensorDataset


EXPECTED_FULLCERT_COMMIT = "3387a8af3ab143fae238b5d3a3f5b24fa815cca6"
EXPECTED_FULLCERT_REMOTE = "https://github.com/t-lorenz/FullCert"


@dataclass(frozen=True)
class FullCertSource:
    root: Path
    code_root: Path
    commit: str
    remote: str
    license_sha256: str


@dataclass(frozen=True)
class FullCertAPI:
    source: FullCertSource
    Interval: type
    LinearLayer: type
    ReLU: type
    Model: type
    BinaryCrossEntropyLoss: type
    Context: type
    perturb_entire_dataset: object
    train_model: object


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def inspect_fullcert_source(root: str | Path) -> FullCertSource:
    checkout = Path(root).resolve()
    code_root = checkout / "code"
    license_path = checkout / "LICENSE"
    if not code_root.is_dir() or not license_path.is_file():
        raise FileNotFoundError("FullCert checkout is missing code/ or LICENSE")
    commit = _git(checkout, "rev-parse", "HEAD")
    if commit != EXPECTED_FULLCERT_COMMIT:
        raise ValueError(f"unexpected FullCert commit: {commit}")
    if _git(checkout, "status", "--porcelain"):
        raise ValueError("official FullCert checkout must be clean")
    remote = _git(checkout, "remote", "get-url", "origin").removesuffix(".git")
    if remote != EXPECTED_FULLCERT_REMOTE:
        raise ValueError(f"unexpected FullCert origin: {remote}")
    return FullCertSource(
        root=checkout,
        code_root=code_root,
        commit=commit,
        remote=remote,
        license_sha256=_sha256(license_path),
    )


@contextmanager
def _official_import_path(code_root: Path) -> Iterator[None]:
    entry = str(code_root)
    sys.path.insert(0, entry)
    try:
        yield
    finally:
        try:
            sys.path.remove(entry)
        except ValueError:
            pass


def load_fullcert(root: str | Path) -> FullCertAPI:
    """Load only the audited external modules after provenance checks."""

    source = inspect_fullcert_source(root)
    with _official_import_path(source.code_root):
        interval = importlib.import_module("boundflow.interval")
        layers = importlib.import_module("boundflow.layers")
        network = importlib.import_module("boundflow.network")
        objective = importlib.import_module("boundflow.objective")
        context = importlib.import_module("boundflow.context")
        perturbation = importlib.import_module("boundflow.perturbation")
        model_functions = importlib.import_module("boundflow.model_functions")
    return FullCertAPI(
        source=source,
        Interval=interval.Interval,
        LinearLayer=layers.LinearLayer,
        ReLU=layers.ReLU,
        Model=network.Model,
        BinaryCrossEntropyLoss=objective.BinaryCrossEntropyLoss,
        Context=context.Context,
        perturb_entire_dataset=perturbation.perturb_entire_dataset,
        train_model=model_functions.train_model,
    )


def _endpoints(value, interval_type: type) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(value, interval_type):
        return value.lower.detach().clone(), value.upper.detach().clone()
    tensor = value.detach().clone()
    return tensor, tensor.clone()


def state_to_fullcert_model(
    state: dict[str, torch.Tensor], dimensions: list[int], api: FullCertAPI
):
    """Convert ``[out,in]`` public boxes to FullCert's ``[in,out]`` layout."""

    layers = []
    linear_layers = len(dimensions) - 1
    for index in range(1, linear_layers + 1):
        layer = api.LinearLayer(dimensions[index - 1], dimensions[index])
        weight_lower = state[f"w{index}_L"].detach().clone().t()
        weight_upper = state[f"w{index}_U"].detach().clone().t()
        bias_lower = state[f"b{index}_L"].detach().clone().reshape(1, -1)
        bias_upper = state[f"b{index}_U"].detach().clone().reshape(1, -1)
        if torch.equal(weight_lower, weight_upper):
            layer.weights = weight_lower
        else:
            layer.weights = api.Interval(weight_lower, weight_upper)
        if torch.equal(bias_lower, bias_upper):
            layer.biases = bias_lower
        else:
            layer.biases = api.Interval(bias_lower, bias_upper)
        layers.append(layer)
        if index < linear_layers:
            layers.append(api.ReLU())
    return api.Model(layers)


def model_to_interval_state(model, dimensions: list[int], api: FullCertAPI):
    """Export exact FullCert interval endpoints in the common evaluator layout."""

    state: dict[str, torch.Tensor] = {}
    linear_index = 0
    for layer in model.layers:
        if not isinstance(layer, api.LinearLayer):
            continue
        linear_index += 1
        weight_lower, weight_upper = _endpoints(layer.weights, api.Interval)
        bias_lower, bias_upper = _endpoints(layer.biases, api.Interval)
        state[f"w{linear_index}_L"] = weight_lower.t().contiguous()
        state[f"w{linear_index}_U"] = weight_upper.t().contiguous()
        state[f"b{linear_index}_L"] = bias_lower.reshape(-1).contiguous()
        state[f"b{linear_index}_U"] = bias_upper.reshape(-1).contiguous()
    if linear_index != len(dimensions) - 1:
        raise ValueError("FullCert model does not match the declared dimensions")
    return state


def _interval_dataset(api: FullCertAPI, inputs, targets, radius: float):
    point_dataset = TensorDataset(inputs.detach().clone(), targets.detach().clone())
    return api.perturb_entire_dataset(
        point_dataset, float(radius), data_range=(float("-inf"), float("inf"))
    )


def run_fullcert_trajectory(
    *,
    api: FullCertAPI,
    initial_state: dict[str, torch.Tensor],
    dimensions: list[int],
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    calibration_inputs: torch.Tensor,
    calibration_targets: torch.Tensor,
    radius: float,
    learning_rate: float,
    epochs: int,
    output_dir: str | Path | None = None,
) -> list[dict[str, torch.Tensor]]:
    """Run the official update and capture ``B_0,...,B_T``.

    The signature deliberately omits test tensors.  Calibration data are used
    only because the official one-epoch routine evaluates a validation loader;
    no checkpoint is selected here.
    """

    if epochs < 0:
        raise ValueError("epochs must be non-negative")
    model = state_to_fullcert_model(initial_state, dimensions, api)
    train_set = _interval_dataset(api, train_inputs, train_targets, radius)
    train_loader = DataLoader(
        train_set, batch_size=len(train_set), shuffle=False
    )
    if len(calibration_inputs) != len(calibration_targets) or not len(calibration_inputs):
        raise ValueError("a non-empty isolated calibration split is required")
    loss = api.BinaryCrossEntropyLoss()
    trajectory = [model_to_interval_state(model, dimensions, api)]
    destination = Path(output_dir) if output_dir is not None else None
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=True)
        torch.save(trajectory[0], destination / "box_epoch_000.pt")
    with torch.no_grad():
        for epoch in range(1, epochs + 1):
            x_lower, x_upper, targets = next(iter(train_loader))
            x_batch = api.Interval(
                lower=x_lower.reshape(x_lower.size(0), -1),
                upper=x_upper.reshape(x_upper.size(0), -1),
            )
            targets = targets.squeeze()
            prediction = model.forward(x_batch)
            loss_context = api.Context()
            loss.forward(prediction, targets, loss_context)
            loss_gradient = loss.backwards(loss_context)
            _, layer_gradients = model.backwards(loss_gradient)
            model.update_weights(layer_gradients, float(learning_rate))
            state = model_to_interval_state(model, dimensions, api)
            trajectory.append(state)
            if destination is not None:
                torch.save(state, destination / f"box_epoch_{epoch:03d}.pt")
    if destination is not None:
        metadata = {
            "schema_version": "1.0",
            "official_remote": api.source.remote,
            "official_commit": api.source.commit,
            "official_license_sha256": api.source.license_sha256,
            "dimensions": dimensions,
            "radius": float(radius),
            "learning_rate": float(learning_rate),
            "epochs": int(epochs),
            "boxes": len(trajectory),
            "test_tensors_accepted": False,
        }
        (destination / "trajectory_manifest.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return trajectory
