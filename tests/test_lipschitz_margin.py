"""Tests for the explicitly non-certified Lipschitz diagnostic."""

import torch
import pytest

from width_cert.analysis import sampled_lipschitz_margin


def test_sampled_lipschitz_margin_labels_optimistic_estimate():
    state = {
        "w1_L": torch.tensor([[1.0, -1.0]]),
        "w1_U": torch.tensor([[1.0, -1.0]]),
        "b1_L": torch.tensor([0.0]),
        "b1_U": torch.tensor([0.0]),
    }
    report = sampled_lipschitz_margin(
        state=state,
        inputs=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        targets=torch.tensor([1.0, 0.0]),
        dimensions=[2, 1],
        input_radius=0.1,
        parameter_samples=2,
    )
    assert report["bound_type"] == "sampled_estimate"
    assert report["sound_certificate"] is False
    assert report["parameter_term"] == 0.0
    assert report["input_term"] == pytest.approx(0.2)
    assert report["coverage_percent"] == 100.0
