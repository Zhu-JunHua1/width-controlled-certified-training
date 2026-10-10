"""Development gates for the constrained official FullCert adapter."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
from auto_LiRPA import BoundedModule
from torch.utils.data import DataLoader, TensorDataset

from width_cert.baselines.fullcert import (
    EXPECTED_FULLCERT_COMMIT,
    load_fullcert,
    model_to_interval_state,
    run_fullcert_trajectory,
    state_to_fullcert_model,
)
from width_cert.bounds import BoundMethod, CUSTOM_BOUND_OPERATORS
from width_cert.bounds.lirpa import build_trace_inputs
from width_cert.certification import evaluate_certification
from width_cert.models import EvaluationGraph


FULLCERT_ROOT = os.environ.get("FULLCERT_ROOT")
pytestmark = pytest.mark.skipif(
    not FULLCERT_ROOT or not Path(FULLCERT_ROOT).is_dir(),
    reason="set FULLCERT_ROOT to the audited official checkout",
)


def _state(dimensions):
    generator = torch.Generator().manual_seed(17)
    state = {}
    for index, (input_dim, output_dim) in enumerate(
        zip(dimensions[:-1], dimensions[1:]), start=1
    ):
        weight = torch.randn(output_dim, input_dim, generator=generator) * 0.1
        bias = torch.randn(output_dim, generator=generator) * 0.1
        state[f"w{index}_L"] = weight - 0.01
        state[f"w{index}_U"] = weight + 0.01
        state[f"b{index}_L"] = bias - 0.01
        state[f"b{index}_U"] = bias + 0.01
    return state


def _assert_state_equal(left, right):
    assert set(left) == set(right)
    for key in left:
        assert torch.equal(left[key], right[key]), key


def test_official_source_and_interval_layout_round_trip():
    api = load_fullcert(FULLCERT_ROOT)
    assert api.source.commit == EXPECTED_FULLCERT_COMMIT
    dimensions = [2, 3, 2, 1]
    state = _state(dimensions)
    recovered = model_to_interval_state(
        state_to_fullcert_model(state, dimensions, api), dimensions, api
    )
    _assert_state_equal(state, recovered)


def test_adapter_one_step_matches_direct_official_training_call():
    api = load_fullcert(FULLCERT_ROOT)
    dimensions = [2, 3, 1]
    initial = _state(dimensions)
    # The official routine accepts an exact starting model; collapse this test
    # fixture to a point before both calls.
    for key in list(initial):
        if key.endswith("_U"):
            initial[key] = initial[key.replace("_U", "_L")].clone()
    inputs = torch.tensor([[-0.5, 0.2], [0.3, -0.1], [0.8, 0.7], [-0.4, -0.6]])
    targets = torch.tensor([0.0, 1.0, 1.0, 0.0])

    adapted = run_fullcert_trajectory(
        api=api,
        initial_state=initial,
        dimensions=dimensions,
        train_inputs=inputs,
        train_targets=targets,
        calibration_inputs=inputs,
        calibration_targets=targets,
        radius=0.001,
        learning_rate=0.1,
        epochs=1,
    )[-1]

    direct_model = state_to_fullcert_model(initial, dimensions, api)
    point = TensorDataset(inputs, targets)
    train_set = api.perturb_entire_dataset(
        point, 0.001, data_range=(float("-inf"), float("inf"))
    )
    loader = DataLoader(train_set, batch_size=len(train_set), shuffle=False)
    with torch.no_grad():
        api.train_model(
            model=direct_model,
            epochs=1,
            train_loader=loader,
            val_loader=loader,
            learning_rate=0.1,
            loss_function=api.BinaryCrossEntropyLoss(),
            early_stopping=False,
        )
    direct = model_to_interval_state(direct_model, dimensions, api)
    _assert_state_equal(adapted, direct)


def test_full_batch_learning_rate_compensates_the_official_double_mean():
    api = load_fullcert(FULLCERT_ROOT)
    dimensions = [2, 3, 1]
    initial = _state(dimensions)
    for key in list(initial):
        if key.endswith("_U"):
            initial[key] = initial[key.replace("_U", "_L")].clone()
    inputs = torch.tensor([[-0.5, 0.2], [0.3, -0.1], [0.8, 0.7], [-0.4, -0.6]])
    targets = torch.tensor([0.0, 1.0, 1.0, 0.0])

    weights = [
        initial[f"w{index}_L"].clone().requires_grad_(True)
        for index in range(1, len(dimensions))
    ]
    biases = [
        initial[f"b{index}_L"].clone().requires_grad_(True)
        for index in range(1, len(dimensions))
    ]
    value = inputs
    for index, (weight, bias) in enumerate(zip(weights, biases)):
        value = value @ weight.t() + bias
        if index < len(weights) - 1:
            value = torch.relu(value)
    torch.nn.functional.binary_cross_entropy_with_logits(
        value.reshape(-1), targets
    ).backward()
    exact = {}
    for index, (weight, bias) in enumerate(zip(weights, biases), start=1):
        exact[f"w{index}_L"] = initial[f"w{index}_L"] - 0.1 * weight.grad
        exact[f"b{index}_L"] = initial[f"b{index}_L"] - 0.1 * bias.grad

    compensated = run_fullcert_trajectory(
        api=api,
        initial_state=initial,
        dimensions=dimensions,
        train_inputs=inputs,
        train_targets=targets,
        calibration_inputs=inputs,
        calibration_targets=targets,
        radius=0.0,
        learning_rate=0.1 * len(inputs),
        epochs=1,
    )[-1]
    for index in range(1, len(dimensions)):
        assert torch.allclose(
            compensated[f"w{index}_L"], exact[f"w{index}_L"], atol=1e-7
        )
        assert torch.allclose(
            compensated[f"b{index}_L"], exact[f"b{index}_L"], atol=1e-7
        )


def test_forty_epoch_export_and_common_evaluator_acceptance(tmp_path):
    api = load_fullcert(FULLCERT_ROOT)
    dimensions = [2, 4, 1]
    initial = _state(dimensions)
    for key in list(initial):
        if key.endswith("_U"):
            initial[key] = initial[key.replace("_U", "_L")].clone()
    inputs = torch.tensor(
        [[-0.7, 0.1], [-0.4, -0.2], [-0.1, 0.4], [0.2, -0.3],
         [0.4, 0.2], [0.6, 0.7], [0.8, -0.1], [0.1, 0.8]],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.float32)
    trajectory = run_fullcert_trajectory(
        api=api,
        initial_state=initial,
        dimensions=dimensions,
        train_inputs=inputs,
        train_targets=targets,
        calibration_inputs=inputs.flip(0),
        calibration_targets=targets.flip(0),
        radius=0.0001,
        learning_rate=0.1,
        epochs=40,
        output_dir=tmp_path,
    )
    assert len(trajectory) == 41
    assert len(list(tmp_path.glob("box_epoch_*.pt"))) == 41
    assert (tmp_path / "trajectory_manifest.json").is_file()

    _, dummy = build_trace_inputs(dimensions, torch.device("cpu"))
    evaluator = BoundedModule(
        EvaluationGraph(len(dimensions) - 1),
        dummy,
        custom_ops=CUSTOM_BOUND_OPERATORS,
        bound_opts={"override_dependent_operations": True},
    )
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=len(inputs))
    for state in (trajectory[0], trajectory[-1]):
        metrics = evaluate_certification(
            evaluator,
            loader,
            state,
            input_radius=0.0001,
            method=BoundMethod.IBP,
            dimensions=dimensions,
        )
        assert 0.0 <= metrics.certified_accuracy <= 100.0
