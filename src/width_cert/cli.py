"""Command-line interface for configuration-driven experiments."""

from __future__ import annotations

import argparse
from importlib.resources import as_file, files
from pathlib import Path

from .analysis import audit_same_box, analyze_checkpoint, export_report, summarize_results
from .experiments import load_config, load_configs, run_experiment, select_seeds


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
    args = parser.parse_args(argv)
    if args.command == "summarize":
        summary = summarize_results(
            args.results, args.output, diverged_as_zero=args.diverged_as_zero
        )
        for row in summary:
            print(
                f"radius={row['radius']:g} method={row['method']} n={row['seeds']} "
                f"eval={row['evaluation_method']} "
                f"test@selected={row['test_selected_mean']:.3f}"
                f"+/-{row['test_selected_std']:.3f} "
                f"test@T={row['test_at_T_mean']} endpoint-n={row['endpoint_seeds']} "
                f"IWAU-active={row['iwau_active_seeds']}"
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
