# Run log schema

All files use schema version `2.0`.

## seed_selection.json

This experiment-level record lists the configured candidate range, every examined candidate, rejection
reasons, and the accepted seeds. Selection is completed before any certified-training arm runs, so every
method and perturbation radius shares exactly the same accepted starts.

## states.csv

| Field | Meaning |
|---|---|
| `epoch` | State index `t` |
| `calibration_certified_accuracy` | Calibration certified accuracy evaluated on `B_t` |
| `weight_width` | Mean parameter-interval width of `B_t` |
| `calibration_clean_accuracy` | Calibration clean accuracy of the exact midpoint-parameter model at `B_t` |

## updates.csv

| Field | Meaning |
|---|---|
| `epoch` | Source state index `t` |
| `next_epoch` | Destination state index `t+1` |
| `gradient_width` | Mean gradient-bound width `Omega_t` on `B_t` |
| `effective_step_mean` | Mean of the scalar or coordinate-wise step used |
| `iwau_active` | Whether this transition used IWAU |

## summary.json and status.json

`summary.json` stores the calibration-selected epoch, final frozen-test evaluations, timing, structured
calibration result, and IC floating-intersection diagnostics. Timing separates the
fixed pre-run, adaptive continuation, actual executed updates, and reported horizon. In particular,
`wall_seconds_per_reported_epoch` is the complete training-pipeline time divided by `T`, whereas
`seconds_per_executed_update` uses the number of interval updates actually evaluated. The former charges
two-stage calibration overhead; the latter isolates update cost. Frozen-test time is recorded separately.
`status.json` is a
small terminal-state record intended for sweep orchestration and interrupted-run recovery; it includes the
last observed certified accuracy even when `cert_at_T` is null.

## results.csv

The experiment-level index contains one row for every explicitly requested training method and test
certifier. Its main fields are `schema_version`, `radius`, `evaluation_radius`, `seed`, `method`,
`training_bound_method`, `evaluation_method`, `test_cert_at_selected`, `test_cert_at_T`,
`test_clean_at_selected`, `test_clean_at_T`, `pretraining_accuracy`,
`peak_calibration_certified_accuracy`, `selected_epoch`, `calibration_cert_at_T`, `status`, `iwau_status`,
`iwau_effective`, `intersection_fallbacks_test`, and `elapsed_seconds`. `iwau_status` distinguishes an active calibration from
`not_required` (the fixed run did not cross the configured reference decline within the horizon) and
`not_applicable` (a non-IWAU method).

If an adaptive method is requested without its same-bound fixed counterpart, the fixed run is executed as an
internal calibration dependency and stored below that cell's `calibration_pre_run/` directory. It is marked
with `role=iwau_calibration_pre_run` and `reported_result=false`; it is deliberately absent from `results.csv`
and from the seed/experiment summary tables.

For an active adaptive arm, `calibrated_coefficient` is the coefficient derived from the same-method fixed
pre-run and `applied_coefficient` includes the configured scale. Public configurations require that scale to
be at least one, and `theoretical_condition_satisfied` records this check explicitly.

## Checkpoints

`last_state.pt` stores the last valid interval state together with its epoch. Captured runs also write
`selected_state.pt`, which is the exact calibration-selected `B_tp` used by direct IWAU continuation and
final test evaluation. These checkpoints contain
PyTorch tensors and should only be loaded from trusted experiment directories.

Test samples are not used to populate `states.csv` or `updates.csv`. They are loaded only after the
calibration-selected checkpoint and schedule are fixed. Floating-point empty IC intersections never have
their endpoints swapped: affected coordinates fall back to their IBP enclosure, and counts plus the maximum
gap are stored in `summary.json`.
