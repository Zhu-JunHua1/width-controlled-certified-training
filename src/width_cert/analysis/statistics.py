"""Predeclared paired statistics for frozen experiment results."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml


@dataclass(frozen=True)
class PairedResult:
    hypothesis: str
    radius: float
    method_a: str
    method_b: str
    evaluation_method: str
    endpoint: str
    seeds: int
    positive: int
    zero: int
    negative: int
    mean_difference: float
    ci95_lower: float
    ci95_upper: float
    p_value: float
    holm_adjusted_p: float = math.nan

    def row(self) -> dict[str, object]:
        return self.__dict__.copy()


def exact_sign_test(
    differences: Iterable[float],
    *,
    alternative: str = "greater",
    tolerance: float = 1e-12,
) -> tuple[float, int, int, int]:
    """Return exact sign-test p-value and positive/zero/negative counts."""

    values = [float(value) for value in differences]
    positive = sum(value > tolerance for value in values)
    negative = sum(value < -tolerance for value in values)
    zero = len(values) - positive - negative
    trials = positive + negative
    if trials == 0:
        return 1.0, positive, zero, negative
    if alternative == "greater":
        tail = sum(math.comb(trials, k) for k in range(positive, trials + 1))
        p_value = tail / (2**trials)
    elif alternative == "less":
        tail = sum(math.comb(trials, k) for k in range(0, positive + 1))
        p_value = tail / (2**trials)
    elif alternative == "two-sided":
        smaller = min(positive, negative)
        tail = sum(math.comb(trials, k) for k in range(0, smaller + 1))
        p_value = min(1.0, 2.0 * tail / (2**trials))
    else:
        raise ValueError("alternative must be greater, less, or two-sided")
    return p_value, positive, zero, negative


def paired_bootstrap_ci(
    differences: Iterable[float],
    *,
    confidence: float = 0.95,
    samples: int = 20000,
    seed: int = 20260927,
) -> tuple[float, float]:
    values = np.asarray(tuple(float(value) for value in differences), dtype=float)
    if values.size == 0:
        raise ValueError("paired bootstrap requires at least one difference")
    if not 0.0 < confidence < 1.0 or samples <= 0:
        raise ValueError("invalid bootstrap settings")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return tuple(float(value) for value in np.quantile(means, [alpha, 1.0 - alpha]))


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    values = [float(value) for value in p_values]
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [0.0] * len(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def _endpoint_value(row: dict[str, str], endpoint: str, missing_as_zero: bool) -> float:
    value = row.get(endpoint, "")
    if value not in (None, ""):
        return float(value)
    if missing_as_zero:
        if row.get("status") != "diverged":
            raise ValueError(
                f"missing endpoint {endpoint!r} may be assigned zero only for "
                f"status=diverged, got {row.get('status')!r}"
            )
        return 0.0
    raise ValueError(
        f"missing endpoint {endpoint!r} for seed={row.get('seed')} "
        f"method={row.get('method')}"
    )


def analyze_paired_hypotheses(
    results_path: str | Path,
    specification_path: str | Path,
    output_path: str | Path,
) -> list[PairedResult]:
    """Evaluate one frozen family across one CSV or a root of protocol CSVs."""

    result_source = Path(results_path)
    row_cache: dict[Path, list[dict[str, str]]] = {}

    def rows_for(item: dict[str, object]) -> list[dict[str, str]]:
        configured = item.get("results")
        if configured is None:
            path = result_source
        else:
            path = Path(str(configured))
            if not path.is_absolute():
                root = result_source if result_source.is_dir() else result_source.parent
                path = root / path
        if path.is_dir():
            path = path / "results.csv"
        path = path.resolve()
        if path not in row_cache:
            with path.open(newline="", encoding="utf-8") as handle:
                row_cache[path] = list(csv.DictReader(handle))
        return row_cache[path]
    specification = yaml.safe_load(Path(specification_path).read_text(encoding="utf-8"))
    hypotheses = specification.get("hypotheses") if isinstance(specification, dict) else None
    if not isinstance(hypotheses, list) or not hypotheses:
        raise ValueError("statistics specification requires a non-empty hypotheses list")
    if specification.get("correction", "Holm") != "Holm":
        raise ValueError("paired-report currently implements only Holm correction")
    bootstrap = specification.get("bootstrap", {})
    bootstrap_samples = int(bootstrap.get("samples", 20000))
    bootstrap_seed = int(bootstrap.get("seed", 20260927))

    pending: list[PairedResult] = []
    for item in hypotheses:
        rows = rows_for(item)
        radius = float(item["radius"])
        evaluation_method = str(item["evaluation_method"])
        method_a = str(item["method_a"])
        method_b = str(item["method_b"])
        endpoint = str(item["endpoint"])
        missing_as_zero = bool(item.get("missing_as_zero", False))
        selected = [
            row
            for row in rows
            if math.isclose(float(row["radius"]), radius, rel_tol=0.0, abs_tol=1e-15)
            and row["evaluation_method"] == evaluation_method
            and row["method"] in {method_a, method_b}
        ]
        by_method: dict[str, dict[int, dict[str, str]]] = {method_a: {}, method_b: {}}
        for row in selected:
            seed = int(row["seed"])
            if seed in by_method[row["method"]]:
                raise ValueError("duplicate result row in paired hypothesis")
            by_method[row["method"]][seed] = row
        common = sorted(set(by_method[method_a]) & set(by_method[method_b]))
        if not common:
            raise ValueError(f"hypothesis {item['id']} has no paired seeds")
        if set(by_method[method_a]) != set(by_method[method_b]):
            raise ValueError(f"hypothesis {item['id']} has unmatched seed sets")
        expected_pairs = item.get("expected_pairs")
        if expected_pairs is not None and len(common) != int(expected_pairs):
            raise ValueError(
                f"hypothesis {item['id']} expected {expected_pairs} paired seeds, "
                f"found {len(common)}"
            )
        differences = [
            _endpoint_value(by_method[method_b][seed], endpoint, missing_as_zero)
            - _endpoint_value(by_method[method_a][seed], endpoint, missing_as_zero)
            for seed in common
        ]
        p_value, positive, zero, negative = exact_sign_test(
            differences,
            alternative=str(item.get("alternative", "greater")),
        )
        lower, upper = paired_bootstrap_ci(
            differences,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        )
        pending.append(
            PairedResult(
                hypothesis=str(item["id"]),
                radius=radius,
                method_a=method_a,
                method_b=method_b,
                evaluation_method=evaluation_method,
                endpoint=endpoint,
                seeds=len(common),
                positive=positive,
                zero=zero,
                negative=negative,
                mean_difference=float(np.mean(differences)),
                ci95_lower=lower,
                ci95_upper=upper,
                p_value=p_value,
            )
        )
    adjusted = holm_adjust(result.p_value for result in pending)
    results = [
        PairedResult(**{**result.row(), "holm_adjusted_p": corrected})
        for result, corrected in zip(pending, adjusted)
    ]
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0].row()))
        writer.writeheader()
        writer.writerows(result.row() for result in results)
    return results
