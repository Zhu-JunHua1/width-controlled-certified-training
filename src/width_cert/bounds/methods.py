"""Bound-method names and the IC construction."""

from __future__ import annotations

from enum import Enum
from typing import Any


class BoundMethod(str, Enum):
    IBP = "ibp"
    IC = "ic"


def intersect_bounds(
    ibp_lower: Any,
    ibp_upper: Any,
    crown_lower: Any,
    crown_upper: Any,
) -> tuple[Any, Any]:
    """Coordinate-wise intersection of two sound interval bounds."""

    import torch

    lower = torch.maximum(ibp_lower, crown_lower)
    upper = torch.minimum(ibp_upper, crown_upper)
    # Mathematically, two sound enclosures of the same value cannot have an
    # empty intersection.  Floating-point kernels can nevertheless create a
    # tiny inconsistency at degenerate point intervals.  Falling back to IBP
    # only on those coordinates preserves soundness and the IC non-inferiority
    # guarantee relative to IBP; inventing a midpoint would not.
    conflict = lower > upper
    if torch.any(conflict):
        largest_gap = float((lower[conflict] - upper[conflict]).max())
        if largest_gap > 1e-5:
            raise RuntimeError(
                "IBP and CROWN bounds have a materially empty intersection "
                f"(maximum gap {largest_gap:.3e})"
            )
    lower = torch.where(conflict, ibp_lower, lower)
    upper = torch.where(conflict, ibp_upper, upper)
    return lower, upper
