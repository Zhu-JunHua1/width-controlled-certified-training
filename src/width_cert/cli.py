"""Command-line interface for configuration-driven experiments."""

from __future__ import annotations

import argparse
from importlib.resources import as_file, files
from pathlib import Path

from .analysis import (
    analyze_paired_hypotheses,
    audit_same_box,
    analyze_checkpoint,
    export_report,
    summarize_results,
    validate_small_network_grid,
)
from .data import create_mnist_split_manifest
from .experiments import (
    load_config,
    load_configs,
    run_experiment,
    select_common_seeds,
    select_seeds,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="width-cert")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser(
        "run",
        help="run certified training (uses the built-in example when CONFIG is omitted)",
    )
    run_parser.add_argument(
        "config",
        nargs="?",
        help="optional YAML configuration; omit it to run the built-in TwoMoons example",
    )
    run_parser.add_argument(
        "-c",
        "--config-file",
        help="YAML configuration (equivalent to the optional positional CONFIG)",
    )
    validate_parser = subparsers.add_parser("validate-config", help="validate a YAML configuration")
    validate_parser.add_argument("config", help="path to a YAML configuration")
    select_parser = subparsers.add_parser(
        "select-seeds",
        help="optionally audit pretraining-based seed selection without certified training",
    )
    select_parser.add_argument("config", help="path to a YAML configuration")
    select_parser.add_argument(
        "--output-dir",
        help="optional directory for seed_selection.json (defaults to config output_dir)",
    )
    common_seed_parser = subparsers.add_parser(
        "select-common-seeds",
        help="select starts accepted by every named variant in a sweep",
    )
    common_seed_parser.add_argument("config", help="YAML sweep configuration")
    common_seed_parser.add_argument("output", help="shared seed-selection manifest JSON")
    common_seed_parser.add_argument("--group", required=True, help="manifest group name")
    common_seed_parser.add_argument(
        "--member",
        action="append",
        required=True,
        help="configuration name to include; repeat for every group member",
    )
    summary_parser = subparsers.add_parser("summarize", help="aggregate a results.csv over seeds")
    summary_parser.add_argument("results", help="path to results.csv")
    summary_parser.add_argument("--output", help="optional output CSV path")
    summary_parser.add_argument(
        "--diverged-as-zero",
        action="store_true",
        help="count a diverged arm as cert@T=0 while retaining its observed peak",
    )
    report_parser = subparsers.add_parser(
        "export-report",
        help="aggregate every results.csv below a results directory",
    )
    report_parser.add_argument("results_root", help="root containing experiment result directories")
    report_parser.add_argument("output_dir", help="directory for the portable report")
    report_parser.add_argument(
        "--diverged-as-zero",
        action="store_true",
        help="count a diverged arm as cert@T=0 while retaining its observed peak",
    )
    lipschitz_parser = subparsers.add_parser(
        "analyze-lipschitz",
        help="run a sampled (non-certified) Lipschitz-margin checkpoint diagnostic",
    )
    lipschitz_parser.add_argument("config", help="single resolved YAML experiment configuration")
    lipschitz_parser.add_argument("checkpoint", help="last_state.pt or selected_state.pt")
    lipschitz_parser.add_argument("output_dir", help="diagnostic output directory")
    lipschitz_parser.add_argument("--training-seed", type=int, required=True)
    lipschitz_parser.add_argument("--input-radius", type=float, required=True)
    lipschitz_parser.add_argument("--parameter-samples", type=int, default=8)
    lipschitz_parser.add_argument("--sampling-seed", type=int, default=0)
    audit_parser = subparsers.add_parser(
        "audit-bounds",
        help="compare gradient bounds on one frozen parameter box",
    )
    audit_parser.add_argument("config", help="single resolved YAML experiment configuration")
    audit_parser.add_argument("checkpoint", help="trusted interval checkpoint")
    audit_parser.add_argument("output_dir", help="audit output directory")
    audit_parser.add_argument("--training-seed", type=int, required=True)
    audit_parser.add_argument("--input-radius", type=float, required=True)
    audit_parser.add_argument("--batch-size", type=int, default=8)
    audit_parser.add_argument("--parameter-samples", type=int, default=32)
    audit_parser.add_argument("--sampling-seed", type=int, default=0)
    split_parser = subparsers.add_parser(
        "make-mnist-split",
        help="create a frozen MNIST-pair index manifest without reading metrics",
    )
    split_parser.add_argument("output", help="output JSON manifest")
    split_parser.add_argument("--digits", nargs=2, type=int, required=True)
    split_parser.add_argument("--train-size", type=int, required=True)
    split_parser.add_argument("--calibration-size", type=int, required=True)
    split_parser.add_argument("--test-size", type=int, required=True)
    split_parser.add_argument("--train-seed", type=int, required=True)
    split_parser.add_argument("--calibration-seed", type=int, required=True)
    split_parser.add_argument("--test-seed", type=int, required=True)
    split_parser.add_argument("--data-root", default="data")
    split_parser.add_argument(
        "--exclude-manifest",
        action="append",
        default=[],
        help="prior manifest whose train/calibration/test indices must be excluded",
    )
    split_parser.add_argument(
        "--legacy-consecutive-calibration",
        action="store_true",
        help="reconstruct the legacy consecutive train/calibration split",
    )
    paired_parser = subparsers.add_parser(
        "paired-report",
        help="evaluate a preregistered paired hypothesis family with Holm correction",
    )
    paired_parser.add_argument(
        "results",
        help="frozen results.csv or root for per-hypothesis relative result paths",
    )
    paired_parser.add_argument("specification", help="YAML hypothesis specification")
    paired_parser.add_argument("output", help="output paired-statistics CSV")
    grid_parser = subparsers.add_parser(
        "validate-small-bound",
        help="run a deterministic tiny-network gradient-bound implementation check",
    )
    grid_parser.add_argument("output_dir", help="directory for the CSV and JSON reports")
    grid_parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args(argv)
    if args.command == "summarize":
        summary = summarize_results(
            args.results, args.output, diverged_as_zero=args.diverged_as_zero
        )
        for row in summary:
            selected = (
                "n/a"
                if row["test_selected_mean"] is None
                else (
                    f"{row['test_selected_mean']:.3f}"
                    f"+/-{row['test_selected_std']:.3f}"
                )
            )
            print(
                f"radius={row['radius']:g} method={row['method']} n={row['seeds']} "
                f"eval={row['evaluation_method']} "
                f"test@selected={selected} "
                f"test@T={row['test_at_T_mean']} endpoint-n={row['endpoint_seeds']} "
                f"update-active={row['update_active_seeds']}"
            )
        return 0
    if args.command == "export-report":
        summary, manifest = export_report(
            args.results_root,
            args.output_dir,
            diverged_as_zero=args.diverged_as_zero,
        )
        print(f"summary written to {summary}")
        print(f"manifest written to {manifest}")
        return 0
    if args.command == "analyze-lipschitz":
        report, cdf = analyze_checkpoint(
            load_config(args.config),
            args.checkpoint,
            args.output_dir,
            training_seed=args.training_seed,
            input_radius=args.input_radius,
            parameter_samples=args.parameter_samples,
            sampling_seed=args.sampling_seed,
        )
        print(f"sampled diagnostic written to {report}")
        print(f"margin CDF written to {cdf}")
        print("note: sampled_estimate is optimistic and is not a sound certificate")
        return 0
    if args.command == "audit-bounds":
        report, manifest = audit_same_box(
            load_config(args.config),
            args.checkpoint,
            args.output_dir,
            training_seed=args.training_seed,
            input_radius=args.input_radius,
            batch_size=args.batch_size,
            parameter_samples=args.parameter_samples,
            sampling_seed=args.sampling_seed,
        )
        print(f"same-box audit written to {report}")
        print(f"audit manifest written to {manifest}")
        print("note: random sampling is an implementation check, not a soundness proof")
        return 0
    if args.command == "make-mnist-split":
        output = create_mnist_split_manifest(
            args.output,
            pair=tuple(args.digits),
            train_size=args.train_size,
            calibration_size=args.calibration_size,
            test_size=args.test_size,
            train_seed=args.train_seed,
            calibration_seed=args.calibration_seed,
            test_seed=args.test_seed,
            data_root=args.data_root,
            exclude_manifests=args.exclude_manifest,
            legacy_consecutive_calibration=args.legacy_consecutive_calibration,
        )
        print(f"MNIST split manifest written to {output}")
        return 0
    if args.command == "paired-report":
        results = analyze_paired_hypotheses(
            args.results,
            args.specification,
            args.output,
        )
        for result in results:
            print(
                f"{result.hypothesis}: mean_diff={result.mean_difference:.3f} "
                f"95%CI=[{result.ci95_lower:.3f},{result.ci95_upper:.3f}] "
                f"sign={result.positive}/{result.zero}/{result.negative} "
                f"p={result.p_value:.6g} holm={result.holm_adjusted_p:.6g}"
            )
        print(f"paired report written to {Path(args.output).resolve()}")
        return 0
    if args.command == "select-common-seeds":
        configs = load_configs(args.config)
        by_name = {config.name: config for config in configs}
        missing = [name for name in args.member if name not in by_name]
        if missing:
            raise ValueError(f"unknown common-seed members: {missing}")
        if len(set(args.member)) != len(args.member):
            raise ValueError("common-seed members must not contain duplicates")
        output = select_common_seeds(
            [by_name[name] for name in args.member],
            args.output,
            group=args.group,
        )
        print(f"common seed-selection manifest written to {output}")
        return 0
    if args.command == "validate-small-bound":
        report, manifest = validate_small_network_grid(
            args.output_dir,
            tolerance=args.tolerance,
        )
        print(f"dense-grid rows written to {report}")
        print(f"dense-grid manifest written to {manifest}")
        print("note: finite grid enumeration is an implementation check, not a proof")
        return 0
    if args.command == "run":
        if args.config and args.config_file:
            parser.error("run accepts either positional CONFIG or --config-file, not both")
        config_path = args.config_file or args.config
        if config_path:
            configs = load_configs(config_path)
            print(f"configuration source: {Path(config_path).resolve()}")
        else:
            workspace_candidates = (
                Path.cwd() / "configs" / "examples" / "example_twomoons.yaml",
                Path(__file__).resolve().parents[2]
                / "configs"
                / "examples"
                / "example_twomoons.yaml",
            )
            workspace_config = next(
                (path.resolve() for path in workspace_candidates if path.is_file()),
                None,
            )
            if workspace_config is not None:
                configs = load_configs(workspace_config)
                print("no configuration supplied; running the workspace TwoMoons example")
                print(f"configuration source: {workspace_config}")
            else:
                resource = files("width_cert").joinpath("configs", "example_twomoons.yaml")
                with as_file(resource) as default_path:
                    configs = load_configs(default_path)
                print("no configuration supplied; running the packaged TwoMoons example")
                print("configuration source: package:width_cert/configs/example_twomoons.yaml")
    else:
        configs = load_configs(args.config)
    if args.command == "validate-config":
        for config in configs:
            print(f"valid configuration: {config.name}")
        return 0
    if args.command == "select-seeds":
        if len(configs) != 1 and args.output_dir:
            raise ValueError("--output-dir is only valid for a single configuration")
        for config in configs:
            path = select_seeds(config, args.output_dir)
            print(f"seed selection written to {path}")
        return 0
    for config in configs:
        output = run_experiment(config)
        print(f"results written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
