"""Tests for exact midpoint-parameter clean accuracy."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from width_cert.certification import evaluate_center_accuracy


def test_center_accuracy_matches_direct_midpoint_model_for_binary_relu_network():
    state = {
        "w1_L": torch.tensor([[0.0], [1.0]]),
        "w1_U": torch.tensor([[2.0], [3.0]]),
        "b1_L": torch.tensor([-1.0, -1.0]),
        "b1_U": torch.tensor([1.0, 1.0]),
        "w2_L": torch.tensor([[1.0, -2.0]]),
        "w2_U": torch.tensor([[3.0, 0.0]]),
        "b2_L": torch.tensor([-0.5]),
        "b2_U": torch.tensor([0.5]),
    }
    inputs = torch.tensor([[-1.0], [0.5], [2.0]])

    w1 = (state["w1_L"] + state["w1_U"]) / 2.0
    b1 = (state["b1_L"] + state["b1_U"]) / 2.0
    w2 = (state["w2_L"] + state["w2_U"]) / 2.0
    b2 = (state["b2_L"] + state["b2_U"]) / 2.0
    logits = torch.relu(inputs @ w1.T + b1) @ w2.T + b2
    labels = (logits.reshape(-1) > 0).float()
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=2, shuffle=False)

    accuracy = evaluate_center_accuracy(loader, state, dimensions=[1, 2, 1])
    assert accuracy == 100.0


def test_center_accuracy_uses_center_parameters_not_output_bound_midpoints():
    # For x=1, the midpoint-parameter model predicts class 1.  This test fixes
    # the semantic contract independently of any bound-propagation midpoint.
    state = {
        "w1_L": torch.tensor([[-2.0]]),
        "w1_U": torch.tensor([[4.0]]),
        "b1_L": torch.tensor([-1.0]),
        "b1_U": torch.tensor([1.0]),
        "w2_L": torch.tensor([[1.0]]),
        "w2_U": torch.tensor([[1.0]]),
        "b2_L": torch.tensor([-0.25]),
        "b2_U": torch.tensor([-0.25]),
    }
    inputs = torch.tensor([[1.0]])
    labels = torch.tensor([1.0])
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=1)

    assert evaluate_center_accuracy(loader, state, dimensions=[1, 1, 1]) == 100.0
