# Run log schema

All files use schema version `1.2`.

## seed_selection.json

This experiment-level record lists the configured candidate range, every examined candidate, rejection
reasons, and the accepted seeds. Selection is completed before any certified-training arm runs, so every
method and perturbation radius shares exactly the same accepted starts.

## states.csv

| Field | Meaning |
|---|---|
| `epoch` | State index `t` |
| `certified_accuracy` | Certified accuracy evaluated on `B_t` |
| `weight_width` | Mean parameter-interval width of `B_t` |
| `clean_accuracy` | Clean accuracy of the exact midpoint-parameter model at `B_t` |

## updates.csv

| Field | Meaning |
|---|---|
| `epoch` | Source state index `t` |
| `next_epoch` | Destination state index `t+1` |
| `gradient_width` | Mean gradient-bound width `Omega_t` on `B_t` |
| `effective_step_mean` | Mean of the scalar or coordinate-wise step used |
| `iwau_active` | Whether this transition used IWAU |

## summary.json and status.json

`summary.json` stores the reported metrics, timing, and structured calibration result. Timing separates the
fixed pre-run, adaptive continuation, actual executed updates, and reported horizon. `status.json` is a
small terminal-state record intended for sweep orchestration and interrupted-run recovery; it includes the
last observed certified accuracy even when `cert_at_T` is null.

## results.csv

The experiment-level index contains only methods explicitly listed in the configuration. Its main fields are
`schema_version`, `radius`, `evaluation_radius`, `seed`, `method`, `pretraining_accuracy`, `peak_certified_accuracy`,
`peak_epoch`, `cert_at_T`, `final_clean_accuracy`, `final_weight_width`, `status`, `iwau_status`,
`iwau_effective`, and `elapsed_seconds`. `iwau_status` distinguishes an active calibration from
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

`last_state.pt` stores the last valid interval state together with its epoch. Fixed-update pre-runs also
write `peak_state.pt`, which is the exact `B_tp` used by direct IWAU continuation. These checkpoints contain
PyTorch tensors and should only be loaded from trusted experiment directories.
