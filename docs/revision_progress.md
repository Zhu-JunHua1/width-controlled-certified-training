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
- Final-test access now fails closed: development runs do not evaluate test
  tensors, while `frozen_final` runs require a complete lock that explicitly
  authorizes the protocol.

Verification at commit `6f6dce0`:

- CPU tests: 38 passed.
- End-to-end TwoMoons smoke run: completed before the fail-closed test gate.
- Smoke run intersection fallbacks: 0 in training, calibration, and test. Its
  exposed test split is retired and will not be used in the final study.
- No final protocol-v2 paper claim is made from this smoke run.

## Stage A2 — controlled methods and fairness baselines (complete)

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
- A second audit used the trained fixed-prefix peak checkpoint on seed 0
  (4,008 gradient coordinates). Mean widths were 0.002538 (IBP), 0.002339
  (AGT-inspired hybrid), 0.002320 (full CROWN), and 0.002292 (Full-IC), giving
  7.8%, 8.6%, and 9.7% tightening relative to IBP. Full-IC selected a CROWN
  lower endpoint on 21.7% and upper endpoint on 19.9% of coordinates. No
  sampled violation or floating-point intersection fallback occurred in
  256,512 coordinate checks. These checks validate the implementation on the
  sampled points; they do not replace the real-arithmetic soundness argument.

Implemented fairness infrastructure:

- All controlled update rules share the exact fixed prefix and continuation
  length.  Scalar schedules match one predeclared final mean step.
- Logs now identify the active update rule instead of labelling every
  continuation as IWAU.
- Complete cost includes the fixed pre-run and continuation.  Synchronized
  wall-clock timing, frozen-test time, bound-evaluation count, and peak device
  memory are recorded separately.
- A calibration-only pilot on three accepted shared starts (seeds 0, 4, and 5)
  estimated coordinate-wise IWAU final mean steps of 0.056703, 0.040261, and
  0.055975. Their frozen development median, 0.055975, was used as the common
  target for the scalar schedules. This is a population-level development
  target, not an oracle per-seed match.
- In the subsequent three-seed comparison, every arm reproduced the fixed
  prefix exactly through its calibration-selected `t_p`; all two-stage arms
  also used the same number of bound evaluations within each seed (66, 64,
  and 69). Inverse-time, cosine, and step-decay all ended at 0% calibration
  certified accuracy. Global-width control ended at 52.0%, 35.0%, and 62.0%,
  while coordinate-wise IWAU ended at 65.0%, 56.5%, and 75.5%. Mean clean
  accuracies at the horizon were 76.3%, 68.5%, 68.5%, 78.5%, and 78.8% for
  inverse-time, cosine, step-decay, global-width, and coordinate IWAU,
  respectively. Complete wall-clock time and peak device memory were nearly
  equal across the two-stage arms (about 160 seconds and 94 MiB on this
  machine).
- Test access was disabled for both pilots: all test result fields are null.
  The values above are development diagnostics only and are not paper results
  or statistical claims.

## Stage B — development gates (in progress)

Representative seeds only; no final test claims. Bound validity, non-vacuous
calibration behavior, exact common-prefix reuse, and compute feasibility have
passed. Before freezing the final study, the update-budget comparison still
needs one explicit choice between (i) the deployable population-level target
estimated above and (ii) a calibration-only per-seed oracle match whose extra
calibration cost is fully charged to the comparator.

The real-image computation gate also passed on one MNIST 1/7 shared start
(`epsilon=0.001`, five updates). IBP, AGT-inspired hybrid, full CROWN, and
Full-IC all reached the horizon with zero numerical-intersection fallbacks and
without accessing test data. Their calibration cert@T values were 48.6%,
49.0%, 48.8%, and 49.0%; the run is too short and has only one seed, so these
values are path-validation diagnostics rather than an effectiveness claim.
The measured cost difference is material: IBP and the hybrid took 1.09 and
1.33 seconds with about 0.6 GiB peak device memory, whereas full CROWN and
Full-IC took about 420 seconds and 12.0 GiB.

Same-box audits at the MNIST calibration-selected checkpoint and horizon
further delimit the mechanism. At the selected checkpoint, mean tightening
relative to IBP was 2.68% (hybrid), 2.33% (full CROWN), 3.04% (Full-IC), and
2.76% (Hybrid-IC). At the horizon, full CROWN alone was 4.70% wider than IBP,
while Full-IC remained 1.54% tighter because the intersection discarded the
looser CROWN coordinates; Hybrid-IC was 0.93% tighter. Across both audits,
there were no violations in 1,257,760 sampled coordinate checks and no
intersection fallback. This supports the intersection mechanism, but it does
not by itself establish a practically large benefit from full backward CROWN.
Final claims about that component remain conditional on the frozen multi-seed
study.

Future runs now persist `initial_state.pt` in addition to selected and horizon
checkpoints, so the preregistered initial/selected/expanded same-box audits can
be reproduced without reconstructing a state.

## Theory-scope revision (draft complete)

[`theory_revision_v2.md`](theory_revision_v2.md) now fixes the theorem scope
before manuscript edits: fixed-step IC nesting requires a common predeclared
step sequence and common certifier; separately calibrated IWAU paths are not
covered. Direct `t_p` continuation is only a calibration-trajectory property,
not an independent-test guarantee. The positive-`lambda` theorem, the
`lambda=0` identity branch, real-arithmetic enclosure proof, and finite-precision
implementation checks are explicitly separated.

## Stage C/D — frozen representative and full experiments (not started)

These stages begin only after the lock file is complete and committed.
