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

- Fixed, inverse-time, cosine, step-decay, global-width, and coordinate-wise
  IWAU comparisons with common prefixes and explicit budget accounting.
- Peak device memory and full two-stage wall-clock reporting.

## Stage B — development gates (not started)

Representative seeds only; no final test claims.  Gate on bound validity,
non-vacuous calibration behavior, method ordering, and compute feasibility.

## Stage C/D — frozen representative and full experiments (not started)

These stages begin only after the lock file is complete and committed.
