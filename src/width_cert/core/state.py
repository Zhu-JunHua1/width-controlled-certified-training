"""State helpers for interval-valued model parameters."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeAlias


IntervalState: TypeAlias = Mapping[str, Any]


def clone_state(state: IntervalState) -> dict[str, Any]:
    """Clone a mapping of tensor-like interval parameters.

    The certified-training core deliberately knows nothing about a concrete
    network architecture.  Values only need to expose ``clone()``; immutable
    scalar metadata is copied as-is.
    """

    cloned: dict[str, Any] = {}
    for name, value in state.items():
        cloned[name] = value.clone() if hasattr(value, "clone") else value
    return cloned

