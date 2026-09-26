"""Tests for the sound IC construction."""

import torch

from width_cert.bounds import IntersectionDiagnostics, intersect_bounds


def test_intersection_keeps_tighter_coordinates():
    ibp_lower = torch.tensor([-2.0, -1.0])
    ibp_upper = torch.tensor([3.0, 4.0])
    crown_lower = torch.tensor([-1.0, -3.0])
    crown_upper = torch.tensor([2.0, 2.0])
    lower, upper = intersect_bounds(ibp_lower, ibp_upper, crown_lower, crown_upper)
    assert torch.equal(lower, torch.tensor([-1.0, -1.0]))
    assert torch.equal(upper, torch.tensor([2.0, 2.0]))


def test_numerically_empty_coordinate_falls_back_to_ibp():
    ibp_lower = torch.tensor([1.0])
    ibp_upper = torch.tensor([1.0])
    crown_lower = torch.tensor([1.0 + 1e-6])
    crown_upper = torch.tensor([1.0 + 1e-6])
    lower, upper = intersect_bounds(ibp_lower, ibp_upper, crown_lower, crown_upper)
    assert torch.equal(lower, ibp_lower)
    assert torch.equal(upper, ibp_upper)


def test_materially_empty_intersection_falls_back_and_is_audited():
    ibp_lower = torch.tensor([0.0])
    ibp_upper = torch.tensor([1.0])
    crown_lower = torch.tensor([2.0])
    crown_upper = torch.tensor([3.0])
    diagnostics = IntersectionDiagnostics()
    lower, upper = intersect_bounds(
        ibp_lower,
        ibp_upper,
        crown_lower,
        crown_upper,
        diagnostics=diagnostics,
    )
    assert torch.equal(lower, ibp_lower)
    assert torch.equal(upper, ibp_upper)
    assert diagnostics.fallback_calls == 1
    assert diagnostics.fallback_coordinates == 1
    assert diagnostics.maximum_gap == 1.0
