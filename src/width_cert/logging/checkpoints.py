"""Checkpoint serialization kept separate from training semantics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..core.state import clone_state


def save_checkpoint(path: str | Path, *, epoch: int, state: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": int(epoch), "state": clone_state(state)}, target)


def load_checkpoint(path: str | Path) -> tuple[int, dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    return int(payload["epoch"]), payload["state"]

