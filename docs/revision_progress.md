# Protocol-v2 revision progress

This file records implementation milestones for the post-review protocol.  A
completed implementation milestone is not a final experimental result; final
test values require a frozen study lock.

## Stage A1 — evaluation firewall (complete)

- Added explicit, disjoint train/calibration/test splits to every public task.
- Moved pretraining stopping and accepted-start selection from test to calibration.
- Restricted fixed pre-runs, IWAU calibration, and checkpoint selection to calibration.
- Made `test cert@calibration-selected checkpoint` the primary result field.
- Evaluated every frozen checkpoint with both IBP and IC test certifiers.
- Delayed test-loader construction until the complete calibration trajectory is frozen.
- Added a dedicated frozen-checkpoint test-evaluation API and regression tests.
- Replaced ambiguous state-log names with calibration-qualified fields.
- Added conservative IBP fallback for floating-point empty intersections and
  structured counters for training, calibration, and final test evaluation.
- Added a versioned experiment-lock template and locking procedure.

Verification at commit `6f6dce0`:

- CPU tests: 38 passed.
- End-to-end TwoMoons smoke run: completed.
- Smoke run intersection fallbacks: 0 in training, calibration, and test.
- No final protocol-v2 paper claim is made from this smoke run.

## Stage A2 — controlled methods and fairness baselines (in progress)

Completed mechanism-audit deliverables:

- Added an independently implemented AGT-inspired
  forward-CROWN/backward-IBP baseline.
- Exposed IBP, AGT-inspired hybrid, full-CROWN-box, and Full-IC as controlled
  training-bound variants.
- Added a same-parameter-box audit with end-to-end method timing, peak device
  memory, endpoint-selection fractions, conservative intersection diagnostics,
  and sampled inclusion checks.
- On one development-only TwoMoons checkpoint (328 gradient coordinates), the
  mean widths were 0.002420 (IBP), 0.002260 (hybrid), 0.002208 (full CROWN),
  and 0.002174 (Full-IC).  Relative to IBP this is 6.6%, 8.7%, and 10.2%
  tightening, respectively.  Full-IC selected a CROWN endpoint on 46.6% of
  lower and 47.9% of upper coordinates.  All five audited constructions had
  zero violations in 10,496 sampled coordinate checks; sampling is not treated
  as a soundness proof.
- CPU tests: 42 passed after the mechanism-audit additions.

Remaining fairness deliverables:

- Run the implemented fixed, inverse-time, cosine, step-decay, global-width,
  and coordinate-wise IWAU comparison on multiple development seeds before
  freezing the final target step and hypothesis set.

Implemented fairness infrastructure:

- All controlled update rules share the exact fixed prefix and continuation
  length.  Scalar schedules match one predeclared final mean step.
- Logs now identify the active update rule instead of labelling every
  continuation as IWAU.
- Complete cost includes the fixed pre-run and continuation.  Synchronized
  wall-clock timing, frozen-test time, bound-evaluation count, and peak device
  memory are recorded separately.
- On one development-only active-calibration seed, all two-stage arms used 66
  gradient-bound evaluations and selected `t_p=14`.  At an exactly matched
  final mean step of 0.056703, inverse-time, cosine, and step-decay ended at 0%
  IC-certified test accuracy; global-width control ended at 52.0%, and
  coordinate-wise IWAU at 64.5%.  This is a development gate, not a paper
  result or a statistical claim.

## Stage B — development gates (not started)

Representative seeds only; no final test claims.  Gate on bound validity,
non-vacuous calibration behavior, method ordering, and compute feasibility.

## Stage C/D — frozen representative and full experiments (not started)

These stages begin only after the lock file is complete and committed.
