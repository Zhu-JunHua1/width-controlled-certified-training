"""Single dataset registry for all public experiment protocols."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class DatasetBundle:
    train_inputs: torch.Tensor
    train_targets: torch.Tensor
    test_inputs: torch.Tensor
    test_targets: torch.Tensor
    input_dim: int
    output_dim: int
    name: str


def _tensor(value):
    return torch.tensor(value, dtype=torch.float32)


def _twomoons(train_size: int, test_size: int, seed: int, strategy: str) -> DatasetBundle:
    if strategy == "combined_split":
        inputs, targets = make_moons(
            n_samples=train_size + test_size, noise=0.1, random_state=seed
        )
        train_x, test_x, train_y, test_y = train_test_split(
            inputs,
            targets,
            test_size=test_size / (train_size + test_size),
            random_state=42,
        )
        return DatasetBundle(
            _tensor(train_x), _tensor(train_y), _tensor(test_x), _tensor(test_y),
            2, 1, "TwoMoons",
        )
    if strategy != "separate":
        raise ValueError(f"unknown TwoMoons data strategy: {strategy}")
    train_x, train_y = make_moons(n_samples=train_size, noise=0.1, random_state=seed)
    test_x, test_y = make_moons(n_samples=test_size, noise=0.1, random_state=42)
    return DatasetBundle(_tensor(train_x), _tensor(train_y), _tensor(test_x), _tensor(test_y), 2, 1, "TwoMoons")


def _mnist(pair: tuple[int, int], train_size: int, test_size: int, seed: int) -> DatasetBundle:
    from torchvision.datasets import MNIST

    first, second = pair

    def load(train: bool, size: int):
        dataset = MNIST(root="./data", train=train, download=True)
        mask = (dataset.targets == first) | (dataset.targets == second)
        inputs = dataset.data[mask].float().reshape(-1, 784) / 255.0
        targets = (dataset.targets[mask] == second).float()
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(inputs), generator=generator)[:size]
        return inputs[indices], targets[indices]

    train_x, train_y = load(True, train_size)
    test_x, test_y = load(False, test_size)
    return DatasetBundle(train_x, train_y, test_x, test_y, 784, 1, f"MNIST {first}/{second}")


def _synthetic(classes: int, train_size: int, test_size: int, seed: int) -> DatasetBundle:
    features = 6
    inputs, targets = make_classification(
        n_samples=train_size + test_size,
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
    test_x = _tensor(scaler.transform(inputs[train_size:train_size + test_size]))
    return DatasetBundle(
        train_x,
        torch.tensor(targets[:train_size], dtype=torch.long),
        test_x,
        torch.tensor(targets[train_size:train_size + test_size], dtype=torch.long),
        features,
        classes,
        f"Synthetic {classes}-class",
    )


def build_dataset(
    spec: str,
    *,
    train_size: int,
    test_size: int,
    seed: int,
    strategy: str = "separate",
) -> DatasetBundle:
    if spec == "twomoons":
        return _twomoons(train_size, test_size, seed, strategy)
    if spec.startswith("mnist:"):
        pair = tuple(int(item) for item in spec.split(":", 1)[1].split(","))
        if len(pair) != 2:
            raise ValueError("MNIST spec must be mnist:<digit>,<digit>")
        return _mnist(pair, train_size, test_size, seed)
    if spec.startswith("synthetic:"):
        classes = int(spec.split(":", 1)[1])
        if classes < 3:
            raise ValueError("synthetic multi-class experiments require at least three classes")
        return _synthetic(classes, train_size, test_size, seed)
    raise ValueError(f"unknown dataset spec: {spec}")
