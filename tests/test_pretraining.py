"""Shared-start pretraining acceptance tests."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from width_cert.experiments import pretrain as pretrain_module


def test_minimum_accuracy_gain_prevents_zero_update_chance_start(monkeypatch):
    observed = iter((0.50, 0.50, 0.502))
    monkeypatch.setattr(
        pretrain_module,
        "point_accuracy",
        lambda *args, **kwargs: next(observed),
    )
    model = torch.nn.Linear(1, 1)
    loader = DataLoader(
        TensorDataset(torch.tensor([[0.0], [1.0]]), torch.tensor([0.0, 1.0])),
        batch_size=2,
    )
    result = pretrain_module.pretrain_to_accuracy(
        model,
        loader,
        torch.tensor([[0.0], [1.0]]),
        torch.tensor([0.0, 1.0]),
        target_accuracy=0.50,
        maximum_initial_accuracy=0.50,
        maximum_final_accuracy=0.52,
        step_size=0.02,
        maximum_epochs=10,
        minimum_epochs=0,
        minimum_accuracy_gain=0.002,
        multiclass=False,
        enabled=True,
        optimizer_name="sgd",
    )
    assert result.initial_accuracy == 0.50
    assert result.final_accuracy == 0.502
    assert result.epochs == 2
