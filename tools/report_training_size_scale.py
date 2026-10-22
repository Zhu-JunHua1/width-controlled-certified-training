"""Validate and report the one-time protocol-v6 training-size result grid."""

from __future__ import annotations

import argparse
import csv
from itertools import product
import json
from pathlib import Path
import random
from statistics import mean

import yaml

from width_cert.analysis.integrity import ResultGridSpec, validate_result_grid
from width_cert.experiments import load_configs
from width_cert.experiments.study_lock import (
    canonical_mapping_sha256,
    experiment_config_sha256,
    verify_frozen_study_lock,
)


METHODS = ("IBP", "IBP+IWAU", "IC", "IC+IWAU")
EVALUATORS = ("IBP", "IC")
SIZES = (100, 200, 400, 800)
SEEDS = tuple(range(10))


def terminal_analysis_value(row: dict[str, str]) -> float:
    """Prespecified divergence policy; never rewrite the raw result."""

    if row["status"] == "diverged":
        if row["test_cert_at_T"] != "":
            raise ValueError("diverged row has a non-null raw terminal certificate")
        return 0.0
    if row["status"] != "completed" or row["test_cert_at_T"] == "":
        raise ValueError("unhandled status or missing completed terminal certificate")
    return float(row["test_cert_at_T"])


def exact_signflip_p(differences: list[float]) -> float:
    """Two-sided exact paired sign-flip test on the mean difference."""

    if not differences:
        raise ValueError("paired differences are required")
    observed = abs(sum(differences))
    extreme = sum(
        abs(sum(sign * value for sign, value in zip(signs, differences)))
        >= observed - 1e-10
        for signs in product((-1, 1), repeat=len(differences))
    )
    return extreme / (2 ** len(differences))


def paired_bootstrap_ci(
    differences: list[float], *, draws: int = 10000, seed: int = 20261022
) -> tuple[float, float]:
    if not differences:
        raise ValueError("paired differences are required")
    generator = random.Random(seed)
    values = sorted(
        mean(generator.choice(differences) for _ in differences)
        for _ in range(draws)
    )
    return values[int(0.025 * draws)], values[int(0.975 * draws) - 1]


def _optional_mean(rows: list[dict[str, str]], field: str) -> tuple[float | None, int]:
    values = [float(row[field]) for row in rows if row[field] != ""]
    return (mean(values) if values else None), len(values)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(config_path: Path, lock_path: Path, output_dir: Path) -> None:
    configs = load_configs(config_path)
    if tuple(config.data.train_size for config in configs) != SIZES:
        raise ValueError("unexpected frozen training-size ladder")
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    lock_hash = canonical_mapping_sha256(lock)
    code_commit = lock["code"]["git_commit"]
    all_rows: list[dict[str, str]] = []
    for config in configs:
        if config.methods != METHODS or config.evaluation.bound_methods != EVALUATORS:
            raise ValueError("unexpected frozen methods or evaluators")
        if config.seed_selection.count != len(SEEDS):
            raise ValueError("unexpected frozen start count")
        verify_frozen_study_lock(
            lock_path,
            protocol_name=config.name,
            config_sha256=experiment_config_sha256(config),
            seed_selection_manifest=config.seed_selection.manifest,
        )
        validate_result_grid(
            config.output_dir,
            ResultGridSpec(
                radii=(0.001,),
                seeds=SEEDS,
                methods=METHODS,
                evaluators=EVALUATORS,
                config_sha256=experiment_config_sha256(config),
                lock_sha256=lock_hash,
                code_commit=code_commit,
            ),
        )
        with (Path(config.output_dir) / "results.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            if row["test_access"] != "frozen_final" or row["analysis_tier"] != "secondary":
                raise ValueError("result analysis tier or test-access mismatch")
            row["train_size"] = str(config.data.train_size)
            row["analysis_test_cert_at_T"] = str(terminal_analysis_value(row))
            all_rows.append(row)

    if len(all_rows) != 320:
        raise ValueError(f"expected 320 evaluator rows, found {len(all_rows)}")
    summaries: list[dict[str, object]] = []
    for size in SIZES:
        for method in METHODS:
            for evaluator in EVALUATORS:
                group = [
                    row for row in all_rows
                    if int(row["train_size"]) == size
                    and row["method"] == method
                    and row["evaluation_method"] == evaluator
                ]
                if len(group) != len(SEEDS):
                    raise ValueError("incomplete method/evaluator group")
                selected_mean, selected_n = _optional_mean(group, "test_cert_at_selected")
                clean_mean, clean_n = _optional_mean(group, "test_clean_at_selected")
                run_summaries = []
                for seed in SEEDS:
                    summary_path = (
                        Path(next(config.output_dir for config in configs if config.data.train_size == size))
                        / "eps_0.001"
                        / f"seed_{seed}"
                        / method.replace("+", "_")
                        / "summary.json"
                    )
                    item = json.loads(summary_path.read_text(encoding="utf-8"))
                    if item["primary_endpoint"] != "test_cert_at_calibration_selected_checkpoint":
                        raise ValueError("unexpected final primary endpoint metadata")
                    run_summaries.append(item)
                summaries.append(
                    {
                        "train_size": size,
                        "method": method,
                        "evaluator": evaluator,
                        "starts": len(group),
                        "completed": sum(row["status"] == "completed" for row in group),
                        "diverged": sum(row["status"] == "diverged" for row in group),
                        "iwau_not_required": sum(
                            row["iwau_status"] == "not_required" for row in group
                        ),
                        "test_selected_mean_pct": selected_mean,
                        "test_selected_n": selected_n,
                        "test_terminal_analysis_mean_pct": mean(
                            float(row["analysis_test_cert_at_T"]) for row in group
                        ),
                        "test_clean_selected_mean_pct": clean_mean,
                        "test_clean_selected_n": clean_n,
                        "complete_wall_mean_seconds": mean(
                            float(row["complete_elapsed_seconds"]) for row in group
                        ),
                        "pre_run_mean_seconds": mean(
                            float(item["pre_run_seconds"]) for item in run_summaries
                        ),
                        "seconds_per_executed_update_mean": mean(
                            float(item["seconds_per_executed_update"])
                            for item in run_summaries
                        ),
                        "wall_seconds_per_reported_epoch_mean": mean(
                            float(item["wall_seconds_per_reported_epoch"])
                            for item in run_summaries
                        ),
                        "peak_allocated_memory_mean_mib": mean(
                            float(row["peak_device_memory_bytes"]) / (1024**2)
                            for row in group
                        ),
                    }
                )

    biggest = max(SIZES)
    by_key = {
        (int(row["train_size"]), int(row["seed"]), row["method"], row["evaluation_method"]): row
        for row in all_rows
    }
    differences = [
        float(by_key[(biggest, seed, "IC+IWAU", "IC")]["analysis_test_cert_at_T"])
        - float(by_key[(biggest, seed, "IC", "IC")]["analysis_test_cert_at_T"])
        for seed in SEEDS
    ]
    low, high = paired_bootstrap_ci(differences)
    primary = {
        "status": "prespecified_blind_scale_extension",
        "train_size": biggest,
        "comparison": "IC+IWAU minus IC-fixed",
        "metric": "IC_test_certification_at_T_diverged_as_zero",
        "seeds": list(SEEDS),
        "paired_differences_percentage_points": differences,
        "mean_difference_percentage_points": mean(differences),
        "bootstrap_percentile_95_ci_percentage_points": [low, high],
        "exact_two_sided_signflip_p": exact_signflip_p(differences),
        "bootstrap_draws": 10000,
        "bootstrap_rng_seed": 20261022,
        "lock_hash": lock_hash,
        "code_commit": code_commit,
    }
    _write_csv(output_dir / "per_start_evaluator_rows.csv", all_rows)
    _write_csv(output_dir / "size_method_summary.csv", summaries)
    (output_dir / "primary_paired_result.json").write_text(
        json.dumps(primary, indent=2, sort_keys=True), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("lock", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report(args.config, args.lock, args.output)


if __name__ == "__main__":
    main()
