"""Shared-start pretraining, isolated from certified-training logic."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class PretrainingResult:
    initial_accuracy: float
    final_accuracy: float
    epochs: int


class PretrainingRejected(RuntimeError):
    """Candidate start that does not satisfy the shared-start protocol."""

    def __init__(self, reason: str, accuracy: float) -> None:
        super().__init__(f"{reason}: accuracy={accuracy:.6f}")
        self.reason = reason
        self.accuracy = float(accuracy)


def point_accuracy(model, inputs, targets, *, multiclass: bool) -> float:
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        if multiclass:
            predictions = outputs.argmax(dim=1)
            accuracy = (predictions == targets.long()).float().mean()
        else:
            predictions = (outputs.reshape(-1) > 0).float()
            accuracy = (predictions == targets.reshape(-1)).float().mean()
    model.train()
    return float(accuracy)


def pretrain_to_accuracy(
    model,
    loader,
    test_inputs,
    test_targets,
    *,
    target_accuracy: float,
    maximum_initial_accuracy: float,
    maximum_final_accuracy: float,
    step_size: float,
    maximum_epochs: int,
    minimum_epochs: int,
    minimum_accuracy_gain: float,
    multiclass: bool,
    enabled: bool,
    optimizer_name: str,
) -> PretrainingResult:
    initial = point_accuracy(model, test_inputs, test_targets, multiclass=multiclass)
    if enabled and initial > maximum_initial_accuracy:
        raise PretrainingRejected(
            "initial_accuracy_above_maximum",
            initial,
        )
    if not enabled:
        return PretrainingResult(initial, initial, 0)
    if (
        initial >= target_accuracy
        and minimum_epochs == 0
        and minimum_accuracy_gain == 0.0
    ):
        if initial > maximum_final_accuracy:
            raise PretrainingRejected("prepared_accuracy_above_maximum", initial)
        return PretrainingResult(initial, initial, 0)

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=step_size)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=step_size)
    else:
        raise ValueError(f"unknown pretraining optimizer: {optimizer_name}")
    for epoch in range(1, maximum_epochs + 1):
        for inputs, targets in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            if multiclass:
                loss = functional.cross_entropy(outputs, targets.long())
            else:
                loss = functional.binary_cross_entropy_with_logits(
                    outputs.reshape(-1), targets.reshape(-1)
                )
            loss.backward()
            optimizer.step()
        accuracy = point_accuracy(model, test_inputs, test_targets, multiclass=multiclass)
        if (
            epoch >= minimum_epochs
            and accuracy >= target_accuracy
            and accuracy + 1e-12 >= initial + minimum_accuracy_gain
        ):
            if accuracy > maximum_final_accuracy:
                raise PretrainingRejected("prepared_accuracy_above_maximum", accuracy)
            return PretrainingResult(initial, accuracy, epoch)
    raise PretrainingRejected(
        "target_not_reached_within_maximum_epochs",
        accuracy,
    )
