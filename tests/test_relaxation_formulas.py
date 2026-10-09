"""Property checks for formulas used in the real-arithmetic proof.

Finite checks can catch transcription or implementation mistakes; they are not
a replacement for the analytic inequalities in the manuscript.
"""

import torch

from width_cert.bounds.operators import BoundStepActivation


def test_default_bilinear_planes_contain_dense_rectangles():
    endpoint_sets = (
        (-2.0, -0.2, -1.1, 0.7),
        (-0.5, 1.4, -2.0, -0.1),
        (-1.0, 2.0, -3.0, 4.0),
        (0.1, 1.3, 0.4, 2.2),
    )
    for x_lower, x_upper, y_lower, y_upper in endpoint_sets:
        x = torch.linspace(x_lower, x_upper, 41)
        y = torch.linspace(y_lower, y_upper, 43)
        x_grid, y_grid = torch.meshgrid(x, y, indexing="ij")
        product = x_grid * y_grid
        lower_plane = y_lower * x_grid + x_lower * y_grid - x_lower * y_lower
        upper_plane = y_upper * x_grid + x_lower * y_grid - x_lower * y_upper
        assert torch.all(lower_plane <= product + 1e-6)
        assert torch.all(product <= upper_plane + 1e-6)


def test_step_interval_rule_matches_implemented_point_rule():
    operator = object.__new__(BoundStepActivation)
    operator.split_status = None
    cases = (
        (-2.0, -0.1, 0.0, 0.0),
        (-2.0, 0.0, 0.0, 0.0),
        (0.1, 2.0, 1.0, 1.0),
        (-0.1, 0.1, 0.0, 1.0),
        (0.0, 0.1, 0.0, 1.0),
    )
    for lower, upper, expected_lower, expected_upper in cases:
        actual_lower, actual_upper = operator.interval_propagate(
            (torch.tensor([lower]), torch.tensor([upper]))
        )
        assert actual_lower.item() == expected_lower
        assert actual_upper.item() == expected_upper


def test_sign_aware_substitution_preserves_scalar_bounds():
    local_lower = torch.tensor([-1.2, 0.3])
    local_upper = torch.tensor([0.7, 2.1])
    concrete = torch.tensor([-0.4, 1.4])
    coefficients = torch.tensor([2.0, -3.0])
    positive = coefficients.clamp(min=0)
    negative = coefficients.clamp(max=0)
    lower = (positive * local_lower + negative * local_upper).sum()
    upper = (positive * local_upper + negative * local_lower).sum()
    exact = (coefficients * concrete).sum()
    assert lower <= exact <= upper
