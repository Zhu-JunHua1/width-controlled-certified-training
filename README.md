# Width-Controlled Certified Training

Reference implementation for *Provable Width Control for Certified Training against Data Perturbations*.
The package extends FullCert-style interval training with two sound components:

- **IC** intersects IBP and CROWN gradient bounds coordinate by coordinate on the full gradient graph.
- **IWAU** (interval-width-adaptive update) rescales every coordinate by
  `base_step / (1 + coefficient * gradient_width)` and starts from the state `B_tp` selected on a
  disjoint calibration split.

The repository contains the final direct-`t_p` algorithm and excludes private paper-audit data.

## Installation

The reference environment used Python 3.10, PyTorch 2.8, CUDA 12.8, auto_LiRPA 0.7.0 at commit
`9d100ec070868440b48d34e2f1dd21b97aab9172`, and an NVIDIA GeForce RTX 5070. Install PyTorch for the
target CPU/CUDA platform, install the pinned
[auto_LiRPA](https://github.com/Verified-Intelligence/auto_LiRPA) source, then install this package:

```bash
python -m venv .venv
# Activate .venv for your platform.
pip install -r requirements.txt
pip install "git+https://github.com/Verified-Intelligence/auto_LiRPA.git@9d100ec070868440b48d34e2f1dd21b97aab9172" --no-deps
pip install -e .
```

The `--no-deps` flag avoids obsolete upstream dependency pins; install a compatible PyTorch build first,
as described above.

auto_LiRPA is an external dependency and is not vendored or modified here. Its version, recognized source
commit, and Python-source-tree SHA-256 are recorded in every run configuration; absolute installation paths
are deliberately omitted. The pinned upstream commit was checked against the public CPU test suite and a
certified-training smoke run under Python 3.10 and PyTorch 2.8.

## Unified experiment entry point

Run the built-in single-seed TwoMoons example with no configuration argument:

```bash
width-cert run
```

The editable workspace example is `configs/examples/example_twomoons.yaml`. It performs actual end-to-end
certified training from a 50% shared start, prints the full trajectory, and writes structured results to
`results/example_twomoons/`. With no argument, the CLI uses this workspace file when it exists and prints
its resolved path; an installed wheel outside the source tree falls back to the packaged copy. Therefore,
editing the workspace YAML immediately changes the next default run.

Run any custom or paper experiment by supplying one validated YAML file, either positionally or with
`--config-file`:

```bash
width-cert run configs/paper/P1_twomoons_ablation.yaml
# Equivalent:
width-cert run --config-file configs/paper/P1_twomoons_ablation.yaml
```

`run` always performs shared-start seed selection internally before certified training, so no preparation
command is required. The following commands are optional diagnostics only:

```bash
width-cert validate-config configs/paper/P1_twomoons_ablation.yaml
width-cert select-seeds configs/paper/P1_twomoons_ablation.yaml
```

`select-seeds` writes the complete selection audit without launching certified training. It is useful for
inspecting a protocol, but it is not a prerequisite for `run`.

The `methods` list controls reported experiment arms. If `IC+IWAU` or `IBP+IWAU` is requested without its
fixed counterpart, the runner performs the required same-bound fixed pre-run internally for calibration,
stores it under that adaptive arm's `calibration_pre_run/` directory, and does not add it to `results.csv`,
the progress-cell count, or the terminal result tables.

For update-rule fairness studies the same entry point also accepts
`+InverseTime`, `+Cosine`, `+StepDecay`, and `+GlobalIWAU`.  Every such arm
reuses the same fixed prefix through the calibration-selected peak.  The three
time-only schedules require a predeclared
`training.schedule_target_final_step`; they reach that same final mean step and
use the same number of continuation updates.  This target must be selected on
development/calibration data and frozen in the study lock before final test
evaluation.  `+GlobalIWAU` uses the same functional form and calibrated
coefficient as IWAU, but replaces the coordinate-wise width tensor by its
global mean.

The configuration explicitly fixes disjoint train/calibration/test data, the architecture, shared-start pretraining, perturbation
radii, training horizon, base step, methods, a candidate seed range, the number of accepted starts, and
the evaluation input domain. Seeds are not hand-listed: the runner examines candidates in ascending order,
performs pretraining once, and retains the first requested number satisfying the shared-start criterion
before any certified method is run. In particular, protocol P4 sets `evaluation.input_domain: clean`; the
other paper protocols use `same_as_training`.
Sweep files use a shared `base` plus explicit `variants`; the same command executes every variant.

Test access is fail-closed. `evaluation.test_access: disabled` is the default
for examples, development gates, and calibration runs; their result rows carry
`evaluation_method=not_evaluated` and null test fields. A final run must set
`test_access: frozen_final` and name a complete, committed study lock. Draft
locks, placeholder provenance, and protocols absent from the lock are rejected
before training starts. Any test split exposed during earlier development must
be retired and cannot be reused by the final lock.

For chance-level starts, `pretraining.minimum_epochs` can require actual optimizer steps,
`pretraining.minimum_accuracy_gain` can require measurable progress beyond the untrained accuracy, and
`pretraining.maximum_final_accuracy` can reject a first-crossing overshoot. These criteria are applied
during seed selection and recorded in `seed_selection.json`; P3 uses a 0.2-point gain and a 52% delivery
ceiling so that deep or wide networks are not accepted unchanged or far above the shared start.

## Terminal progress

Set `echo_epochs: true` in a YAML configuration to print structured progress at four levels:

1. experiment metadata and the total number of training cells;
2. candidate-seed decisions and the final shared seed set;
3. seed and method/radius boundaries with a completion summary for every cell;
4. an aligned epoch table with `cal_cert`, `cal_clean`, `weight_width`, `grad_width`,
   `mean_step`, and the active update rule (`fixed` or `IWAU`).

After each seed, the terminal prints one table containing all requested method results. After the final
seed, it prints an aggregate table with mean and population standard deviation. Development runs state
that final test evaluation is disabled. Endpoint statistics use only runs that reached the requested
horizon; missing endpoints are never silently converted to zero.

The terminal names match the CSV schema semantically: `weight_width` is the mean parameter-interval
width, whereas `grad_width` is the mean gradient-bound width. Terminal output is for monitoring only;
the CSV/JSON files remain the authoritative machine-readable record.

## Output contract

The experiment root contains `seed_selection.json` (candidate pool, accepted/rejected starts, and final
shared seed set) and `results.csv` (one index row per completed arm). Each radius/seed/method cell has its
own directory containing:

| File | Meaning |
|---|---|
| `config.json` | Resolved protocol, loss definitions, environment, and dependency provenance |
| `states.csv` | State metrics: row `t` describes the interval state `B_t` |
| `updates.csv` | Update metrics: row `t` describes `B_t -> B_{t+1}`, `Omega_t`, the mean step, and the active rule |
| `summary.json` | Calibration-selected epoch, endpoints, frozen test metrics, two-stage timing/update budget, peak device memory, update status, and intersection diagnostics |
| `status.json` | Terminal state, stop reason, and last observed certified accuracy |
| `initial_state.pt` | Frozen interval state at the beginning of the reported trajectory |
| `last_state.pt` | Last valid interval state and its epoch (also written for interrupted horizons) |
| `selected_state.pt` | State selected by the calibration trajectory and used for direct continuation/final test evaluation |

Test fields are also `null` when `evaluation.test_access` is disabled. `test_cert_at_T` is `null` when the requested horizon was not reached. A diverged run is never silently encoded
as a completed run with zero certification. The configurable `max_gradient_width` is a numerical safety
stop, not the theoretical reference width used by the finite-horizon analysis.

Aggregate completed seeds without spreadsheet-specific code:

```bash
width-cert summarize results/P1_twomoons_ablation/results.csv \
  --diverged-as-zero \
  --output results/P1_twomoons_ablation/summary.csv
```

The raw log keeps `test_cert_at_T: null` for an unfinished run. `--diverged-as-zero` is an explicit analysis
policy that assigns endpoint zero without altering the raw record.

Export all experiment-level aggregates below a result root together with a path-safe provenance manifest:

```bash
width-cert export-report results public-report --diverged-as-zero
```

This writes `protocol_summary.csv` and `report_manifest.json`. The manifest records repository-relative
input paths and SHA-256 hashes, never absolute workstation paths. Plotting and typesetting can consume the
portable summary without importing the training implementation or private spreadsheets.

The supplementary Lipschitz--margin diagnostic can be run on a binary checkpoint:

```bash
width-cert analyze-lipschitz configs/paper/P2_mnist17_ablation.yaml \
  path/to/last_state.pt diagnostics/lipschitz \
  --training-seed 0 --input-radius 0.001 --parameter-samples 8
```

This command deliberately labels its constants as `sampled_estimate` and `sound_certificate: false`.
Finite box sampling can underestimate the true supremum, so the resulting coverage is an optimistic
diagnostic rather than a certified lower bound.

Compare the supported gradient-bound constructions on exactly the same frozen
parameter box with:

```bash
width-cert audit-bounds configs/development/gradient_bound_ablation.yaml \
  path/to/last_state.pt diagnostics/same-box \
  --training-seed 1 --input-radius 0.01
```

The audit reports IBP, an AGT-inspired forward-CROWN/backward-IBP hybrid,
full-gradient-graph CROWN, Full-IC, and Hybrid-IC.  It records bound width,
relative tightening, end-to-end construction time, peak device memory, and
which endpoints the intersections retain.  Its finite random sampling is an
implementation check only; it is explicitly marked as not being a soundness
proof.  The AGT-inspired arm is a controlled ablation implemented in this
repository, not a claim of bitwise equivalence to the AGT software.

Frozen-final MNIST protocols use explicit index manifests rather than relying
on a nominally new random seed. Create a manifest, optionally excluding one or
more exposed development manifests, with:

```bash
width-cert make-mnist-split data_splits/final_mnist17.json \
  --digits 1 7 --train-size 100 --calibration-size 500 --test-size 500 \
  --train-seed 55873755 --calibration-seed 261026576 --test-seed 1608498564 \
  --exclude-manifest data_splits/retired_mnist17_development.json
```

The command reads no model metric. It records only filtered official-dataset
indices and provenance; the loader validates the manifest and the frozen study
lock binds its SHA-256.

## Package layout

| Path | Responsibility |
|---|---|
| `core/` | State cloning, update rules, training loop, same-method calibration, direct peak continuation |
| `bounds/` | IBP/CROWN selection, hybrid bounds, IC intersection, auto_LiRPA adapters, custom sound operators |
| `certification/` | Binary and multi-class certified-accuracy criteria |
| `models/` | MLPs and explicit full gradient computation graphs |
| `data/` | Dataset registry for TwoMoons, MNIST pairs, and synthetic multi-class data |
| `experiments/` | Configuration schema, pretraining, environment capture, and unified runner |
| `logging/` | Structured state/update records, terminal output, JSON/CSV files, and checkpoints |
| `analysis/` | Aggregation, path-safe report export, and checkpoint-level mechanism audits |
| `configs/paper/` | Reproducible paper protocol files |
| `tests/` | CPU regression tests for indexing, calibration, capping, status, and logging |

Each module owns one concern. The mathematical core does not parse command-line arguments, construct
datasets, print progress, or choose output paths.

## Final IWAU semantics

1. Run the same bound method with fixed updates through horizon `T` and save `B_0,...,B_T`.
2. On the calibration split, select the first state `B_tp` attaining the maximum calibration certified
   accuracy, including `B_0`.
3. Let `t_ref > t_p` be the first state below `retention * peak`.
4. Compute `S_Omega = sum(Omega_t, t=t_p,...,t_ref-1)` and
   `coefficient = (T - t_p) / S_Omega`.
5. Reuse the exact fixed prefix through `B_tp`; the first IWAU transition is `B_tp -> B_{tp+1}`.
6. If no reference decline occurs within `T`, calibration returns `not_required` and the IWAU arm equals
   the fixed arm by definition.

Only a `frozen_final` run authorized by a complete study lock materializes the
test loader. The same already-selected checkpoint is then evaluated by every
configured test certifier. See
[`docs/experimental_locking.md`](docs/experimental_locking.md) for the final-study lock procedure.

The calibration result has one of three explicit states: `active`, `not_required`, or `invalid`.

## Tests

```bash
pytest
```

The CPU suite checks state/update indexing, the exact `S_Omega` summation window, calibration-selected
continuation, the coefficient-zero identity, the coordinate-wise width cap, conservative intersection
fallback, final-test isolation, non-silent divergence, and the logging schema.

## Reproducibility and citation

Paper protocols and result mappings are documented in [`docs/protocols.md`](docs/protocols.md), and
representative reference-cell checks are reported in
[`docs/reference_validation.md`](docs/reference_validation.md). GitHub
reads [`CITATION.cff`](CITATION.cff) and exposes a **Cite this repository** action. An archival DOI will be
added after the public release is frozen.

## License

This repository is released under the BSD 3-Clause License. Third-party dependencies retain their own
licenses.
