"""Creation of frozen, auditable MNIST split manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import torch


def _ordered_candidates(size: int, excluded: set[int]) -> torch.Tensor:
    return torch.tensor(
        [index for index in range(size) if index not in excluded],
        dtype=torch.long,
    )


def _shuffle(values: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return values[torch.randperm(len(values), generator=generator)]


def select_mnist_indices(
    *,
    official_train_size: int,
    official_test_size: int,
    train_size: int,
    calibration_size: int,
    test_size: int,
    train_seed: int,
    calibration_seed: int,
    test_seed: int,
    excluded_train: Iterable[int] = (),
    excluded_test: Iterable[int] = (),
    legacy_consecutive_calibration: bool = False,
) -> tuple[list[int], list[int], list[int]]:
    """Select deterministic disjoint indices without reading labels or metrics."""

    blocked_train = set(int(value) for value in excluded_train)
    blocked_test = set(int(value) for value in excluded_test)
    train_candidates = _ordered_candidates(official_train_size, blocked_train)
    test_candidates = _ordered_candidates(official_test_size, blocked_test)
    if train_size + calibration_size > len(train_candidates):
        raise ValueError("not enough unexposed official-train examples")
    if test_size > len(test_candidates):
        raise ValueError("not enough unexposed official-test examples")

    train_order = _shuffle(train_candidates, train_seed)
    train_indices = train_order[:train_size]
    remaining = train_order[train_size:]
    if legacy_consecutive_calibration:
        if blocked_train or blocked_test:
            raise ValueError("legacy consecutive selection cannot exclude prior manifests")
        calibration_indices = remaining[:calibration_size]
    else:
        calibration_indices = _shuffle(remaining, calibration_seed)[:calibration_size]
    test_indices = _shuffle(test_candidates, test_seed)[:test_size]
    return (
        train_indices.tolist(),
        calibration_indices.tolist(),
        test_indices.tolist(),
    )


def canonical_json_sha256(path: str | Path) -> str:
    """Hash parsed JSON content independently of whitespace and line endings."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def create_mnist_split_manifest(
    output: str | Path,
    *,
    pair: tuple[int, int],
    train_size: int,
    calibration_size: int,
    test_size: int,
    train_seed: int,
    calibration_seed: int,
    test_seed: int,
    data_root: str | Path = "data",
    exclude_manifests: Iterable[str | Path] = (),
    legacy_consecutive_calibration: bool = False,
) -> Path:
    """Write one manifest in the filtered official MNIST-pair index spaces."""

    from torchvision.datasets import MNIST

    if len(pair) != 2 or pair[0] == pair[1]:
        raise ValueError("pair must contain two distinct MNIST digits")
    dataset_name = f"mnist:{pair[0]},{pair[1]}"

    def filtered_original_indices(train: bool) -> torch.Tensor:
        dataset = MNIST(root=str(data_root), train=train, download=True)
        mask = (dataset.targets == pair[0]) | (dataset.targets == pair[1])
        return torch.where(mask)[0]

    filtered_train = filtered_original_indices(True)
    filtered_test = filtered_original_indices(False)
    excluded_train: set[int] = set()
    excluded_test: set[int] = set()
    exclusion_records: list[dict[str, str]] = []
    for manifest in exclude_manifests:
        path = Path(manifest)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("dataset") != dataset_name:
            raise ValueError(f"excluded manifest {path} uses a different dataset")
        excluded_train.update(int(value) for value in payload["train_indices"])
        excluded_train.update(int(value) for value in payload["calibration_indices"])
        excluded_test.update(int(value) for value in payload["test_indices"])
        exclusion_records.append({"sha256": canonical_json_sha256(path)})

    train_indices, calibration_indices, test_indices = select_mnist_indices(
        official_train_size=len(filtered_train),
        official_test_size=len(filtered_test),
        train_size=train_size,
        calibration_size=calibration_size,
        test_size=test_size,
        train_seed=train_seed,
        calibration_seed=calibration_seed,
        test_seed=test_seed,
        excluded_train=excluded_train,
        excluded_test=excluded_test,
        legacy_consecutive_calibration=legacy_consecutive_calibration,
    )
    payload = {
        "schema_version": "1.0",
        "dataset": dataset_name,
        "index_space": {
            "train_and_calibration": "filtered_official_train_pair_order",
            "test": "filtered_official_test_pair_order",
        },
        "filtered_counts": {
            "official_train": len(filtered_train),
            "official_test": len(filtered_test),
        },
        "filtered_original_index_sha256": {
            "official_train": hashlib.sha256(
                filtered_train.numpy().tobytes()
            ).hexdigest(),
            "official_test": hashlib.sha256(
                filtered_test.numpy().tobytes()
            ).hexdigest(),
        },
        "selection": {
            "train_seed": train_seed,
            "calibration_seed": calibration_seed,
            "test_seed": test_seed,
            "legacy_consecutive_calibration": legacy_consecutive_calibration,
        },
        "excluded_manifests": exclusion_records,
        "train_indices": train_indices,
        "calibration_indices": calibration_indices,
        "test_indices": test_indices,
    }
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target
