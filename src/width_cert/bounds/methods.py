"""Bound-method names and the IC construction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class BoundMethod(str, Enum):
    IBP = "ibp"
    HYBRID = "hybrid"
    CROWN = "crown"
    IC = "ic"


@dataclass
class IntersectionDiagnostics:
    """Audit counters for floating-point IC intersection conflicts."""

    calls: int = 0
    fallback_calls: int = 0
    fallback_coordinates: int = 0
    maximum_gap: float = 0.0

    def record(self, conflict, gaps) -> None:
        self.calls += 1
        if conflict.any():
            self.fallback_calls += 1
            self.fallback_coordinates += int(conflict.sum())
            self.maximum_gap = max(self.maximum_gap, float(gaps.max()))

    def as_dict(self) -> dict[str, int | float]:
        return {
            "calls": self.calls,
            "fallback_calls": self.fallback_calls,
            "fallback_coordinates": self.fallback_coordinates,
            "maximum_gap": self.maximum_gap,
        }


def intersect_bounds(
    ibp_lower: Any,
    ibp_upper: Any,
    crown_lower: Any,
    crown_upper: Any,
    diagnostics: IntersectionDiagnostics | None = None,
) -> tuple[Any, Any]:
    """Coordinate-wise intersection with conservative floating fallback.

    A non-empty intersection is guaranteed in real arithmetic when both input
    bounds are sound enclosures of the same value.  If floating-point
    computation produces ``lower > upper``, that coordinate falls back to the
    IBP enclosure.  No endpoint swapping or midpoint fabrication is allowed.
    """

    import torch

    lower = torch.maximum(ibp_lower, crown_lower)
    upper = torch.minimum(ibp_upper, crown_upper)
    # Mathematically, two sound enclosures of the same value cannot have an
    # empty intersection.  Floating-point kernels can nevertheless create a
    # tiny inconsistency at degenerate point intervals.  Falling back to IBP
    # only on those coordinates preserves soundness and the IC non-inferiority
    # guarantee relative to IBP; inventing a midpoint would not.
    conflict = lower > upper
    if diagnostics is not None:
        diagnostics.record(conflict, lower[conflict] - upper[conflict])
    lower = torch.where(conflict, ibp_lower, lower)
    upper = torch.where(conflict, ibp_upper, upper)
    return lower, upper
