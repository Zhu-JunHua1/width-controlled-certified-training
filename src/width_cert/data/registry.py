"""Single dataset registry for all public experiment protocols."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class DatasetBundle:
    train_inputs: torch.Tensor
    train_targets: torch.Tensor
    calibration_inputs: torch.Tensor
    calibration_targets: torch.Tensor
    test_inputs: torch.Tensor
    test_targets: torch.Tensor
    input_dim: int
    output_dim: int
    name: str


def _tensor(value):
    return torch.tensor(value, dtype=torch.float32)


def _twomoons(
    train_size: int,
    calibration_size: int,
    test_size: int,
    seed: int,
    calibration_seed: int,
    test_seed: int,
    strategy: str,
) -> DatasetBundle:
    if strategy == "combined_split":
        inputs, targets = make_moons(
            n_samples=train_size + calibration_size + test_size,
            noise=0.1,
            random_state=seed,
        )
        train_x, holdout_x, train_y, holdout_y = train_test_split(
            inputs,
            targets,
            train_size=train_size,
            test_size=calibration_size + test_size,
            random_state=calibration_seed,
            stratify=targets,
        )
        calibration_x, test_x, calibration_y, test_y = train_test_split(
            holdout_x,
            holdout_y,
            train_size=calibration_size,
            test_size=test_size,
            random_state=test_seed,
            stratify=holdout_y,
        )
        return DatasetBundle(
            _tensor(train_x),
            _tensor(train_y),
            _tensor(calibration_x),
            _tensor(calibration_y),
            _tensor(test_x),
            _tensor(test_y),
            2,
            1,
            "TwoMoons",
        )
    if strategy != "separate":
        raise ValueError(f"unknown TwoMoons data strategy: {strategy}")
    train_x, train_y = make_moons(n_samples=train_size, noise=0.1, random_state=seed)
    calibration_x, calibration_y = make_moons(
        n_samples=calibration_size,
        noise=0.1,
        random_state=calibration_seed,
    )
    test_x, test_y = make_moons(n_samples=test_size, noise=0.1, random_state=test_seed)
    return DatasetBundle(
        _tensor(train_x),
        _tensor(train_y),
        _tensor(calibration_x),
        _tensor(calibration_y),
        _tensor(test_x),
        _tensor(test_y),
        2,
        1,
        "TwoMoons",
    )


def _mnist(
    pair: tuple[int, int],
    train_size: int,
    calibration_size: int,
    test_size: int,
    seed: int,
    calibration_seed: int,
    test_seed: int,
    split_manifest: str | None,
) -> DatasetBundle:
    from torchvision.datasets import MNIST

    first, second = pair

    def load_all(train: bool):
        dataset = MNIST(root="./data", train=train, download=True)
        mask = (dataset.targets == first) | (dataset.targets == second)
        inputs = dataset.data[mask].float().reshape(-1, 784) / 255.0
        targets = (dataset.targets[mask] == second).float()
        return inputs, targets

    official_train_x, official_train_y = load_all(True)
    official_test_x, official_test_y = load_all(False)
    if split_manifest is not None:
        train_indices, calibration_indices, test_indices = _load_mnist_indices(
            split_manifest,
            pair=pair,
            train_size=train_size,
            calibration_size=calibration_size,
            test_size=test_size,
            official_train_size=len(official_train_x),
            official_test_size=len(official_test_x),
        )
    else:
        # Compatibility path for development/legacy configurations. Final
        # locked MNIST studies must provide an explicit split manifest.
        train_generator = torch.Generator().manual_seed(seed)
        train_order = torch.randperm(len(official_train_x), generator=train_generator)
        needed = train_size + calibration_size
        if needed > len(train_order):
            raise ValueError(
                "requested MNIST train+calibration size exceeds available data"
            )
        train_indices = train_order[:train_size]
        calibration_indices = train_order[train_size:needed]
        test_generator = torch.Generator().manual_seed(test_seed)
        test_indices = torch.randperm(
            len(official_test_x), generator=test_generator
        )[:test_size]
    return DatasetBundle(
        official_train_x[train_indices],
        official_train_y[train_indices],
        official_train_x[calibration_indices],
        official_train_y[calibration_indices],
        official_test_x[test_indices],
        official_test_y[test_indices],
        784,
        1,
        f"MNIST {first}/{second}",
    )


def _load_mnist_indices(
    path: str | Path,
    *,
    pair: tuple[int, int],
    train_size: int,
    calibration_size: int,
    test_size: int,
    official_train_size: int,
    official_test_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load and fail-closed validate a frozen MNIST index manifest."""

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"MNIST split manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_dataset = f"mnist:{pair[0]},{pair[1]}"
    if payload.get("schema_version") != "1.0":
        raise ValueError("MNIST split manifest must use schema_version 1.0")
    if payload.get("dataset") != expected_dataset:
        raise ValueError(
            f"MNIST split manifest is for {payload.get('dataset')!r}, "
            f"expected {expected_dataset!r}"
        )

    expected = {
        "train_indices": (train_size, official_train_size),
        "calibration_indices": (calibration_size, official_train_size),
        "test_indices": (test_size, official_test_size),
    }
    tensors: dict[str, torch.Tensor] = {}
    for key, (required_size, upper_bound) in expected.items():
        values = payload.get(key)
        if not isinstance(values, list) or len(values) != required_size:
            raise ValueError(
                f"MNIST split manifest {key} must contain exactly "
                f"{required_size} indices"
            )
        if any(not isinstance(value, int) for value in values):
            raise ValueError(f"MNIST split manifest {key} must contain integers")
        if len(set(values)) != len(values):
            raise ValueError(f"MNIST split manifest {key} contains duplicates")
        if any(value < 0 or value >= upper_bound for value in values):
            raise ValueError(f"MNIST split manifest {key} contains out-of-range indices")
        tensors[key] = torch.tensor(values, dtype=torch.long)

    train_set = set(payload["train_indices"])
    calibration_set = set(payload["calibration_indices"])
    if train_set & calibration_set:
        raise ValueError("MNIST train and calibration indices must be disjoint")
    return (
        tensors["train_indices"],
        tensors["calibration_indices"],
        tensors["test_indices"],
    )


def _synthetic(
    classes: int,
    train_size: int,
    calibration_size: int,
    test_size: int,
    seed: int,
) -> DatasetBundle:
    features = 6
    inputs, targets = make_classification(
        n_samples=train_size + calibration_size + test_size,
        n_features=features,
        n_informative=features,
        n_redundant=0,
        n_repeated=0,
        n_classes=classes,
        n_clusters_per_class=1,
        class_sep=2.0,
        random_state=seed,
    )
    order = np.random.RandomState(seed).permutation(len(inputs))
    inputs, targets = inputs[order].astype("float32"), targets[order]
    scaler = StandardScaler().fit(inputs[:train_size])
    train_x = _tensor(scaler.transform(inputs[:train_size]))
    calibration_end = train_size + calibration_size
    calibration_x = _tensor(scaler.transform(inputs[train_size:calibration_end]))
    test_x = _tensor(
        scaler.transform(inputs[calibration_end:calibration_end + test_size])
    )
    return DatasetBundle(
        train_x,
        torch.tensor(targets[:train_size], dtype=torch.long),
        calibration_x,
        torch.tensor(targets[train_size:calibration_end], dtype=torch.long),
        test_x,
        torch.tensor(
            targets[calibration_end:calibration_end + test_size], dtype=torch.long
        ),
        features,
        classes,
        f"Synthetic {classes}-class",
    )


def build_dataset(
    spec: str,
    *,
    train_size: int,
    calibration_size: int,
    test_size: int,
    seed: int,
    calibration_seed: int = 314159,
    test_seed: int = 42,
    strategy: str = "separate",
    split_manifest: str | None = None,
) -> DatasetBundle:
    if spec == "twomoons":
        return _twomoons(
            train_size,
            calibration_size,
            test_size,
            seed,
            calibration_seed,
            test_seed,
            strategy,
        )
    if spec.startswith("mnist:"):
        pair = tuple(int(item) for item in spec.split(":", 1)[1].split(","))
        if len(pair) != 2:
            raise ValueError("MNIST spec must be mnist:<digit>,<digit>")
        return _mnist(
            pair,
            train_size,
            calibration_size,
            test_size,
            seed,
            calibration_seed,
            test_seed,
            split_manifest,
        )
    if spec.startswith("synthetic:"):
        classes = int(spec.split(":", 1)[1])
        if classes < 3:
            raise ValueError("synthetic multi-class experiments require at least three classes")
        return _synthetic(classes, train_size, calibration_size, test_size, seed)
    raise ValueError(f"unknown dataset spec: {spec}")
