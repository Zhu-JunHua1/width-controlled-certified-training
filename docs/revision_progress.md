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

## Deterministic small-network implementation validation

A checkpoint-free dense-grid validator is now part of the public CLI as
`width-cert validate-small-bound`.  The default suite contains five
`[1,1,1]` BCE networks covering crossing, active, and inactive hidden ReLUs and
positive, negative, and mixed-sign downstream intervals.  Each input and
parameter interval is evaluated at its lower endpoint, midpoint, and upper
endpoint for both targets.  At an exact ReLU kink, both derivative endpoint
choices in `[0,1]` are checked using an independent analytic gradient formula.

The current development run evaluated 2,430 parameter/input grid points, 2,484
admitted gradients (including the additional kink choices), and 49,680
method-gradient coordinates across IBP, the AGT-inspired hybrid, full CROWN,
Full-IC, and Hybrid-IC.  It found zero enclosure violations at tolerance
`1e-5` and zero floating-point intersection fallbacks.  These numbers are an
implementation regression result only; the generated manifest explicitly sets
`is_real_arithmetic_soundness_proof` to false.

## Theory-scope revision (draft complete)

[`theory_revision_v2.md`](theory_revision_v2.md) now fixes the theorem scope
before manuscript edits: fixed-step IC nesting requires a common predeclared
step sequence and common certifier; separately calibrated IWAU paths are not
covered. Direct `t_p` continuation is only a calibration-trajectory property,
not an independent-test guarantee. The positive-`lambda` theorem, the
`lambda=0` identity branch, real-arithmetic enclosure proof, and finite-precision
implementation checks are explicitly separated.

## Stage C/D — frozen representative and full experiments (not started)

The prospective choices for endpoints, four primary hypotheses, Holm-corrected
paired tests, common-certifier comparisons, same-box claims, and the
development-population schedule target are recorded in
[`final_study_design_draft.md`](final_study_design_draft.md). These stages begin
only after the remaining split manifests and resolved configuration hashes are
complete and the study lock is committed.

## Protocol-v2 primary candidates (test still sealed)

The prospective P1/P2 primary configurations are now recorded under
`configs/protocol_v2_candidates/`. Both use a common IC calibration certifier,
request both IBP and IC only for later frozen checkpoint evaluation, and keep
`test_access: disabled`. The files therefore authorize calibration-only gates,
not final paper metrics.

Seed selection was rerun on the new data roles and recorded in
`study_locks/candidate_audits/`. P1 selected the first ten valid starts from
`[0,25)`: `0,4,5,6,7,9,10,11,12,13`. The initial P2 pool `[0,15)` yielded only
five valid 50% starts under the new disjoint MNIST split. Before any final-test
access, the prospective range was expanded to `[0,50)` and the first ten valid
starts were fixed as `0,5,9,11,13,17,19,21,29,30`. This is deterministic
calibration-only filtering; no seed was selected from certification results.

The TwoMoons train/calibration/test seeds were derived without reading a metric
from the SHA-256 phrase documented beside the candidate configuration. The
MNIST candidate uses the committed index manifest whose test indices have not
been evaluated by protocol-v2.

One-start calibration-only gates were then run on both candidate data roles.
Every P1/P2 arm reached its short development horizon; every result row used
`evaluation_method=not_evaluated`, and all test fields remained null. On the
three-update MNIST path gate, calibration cert@T was 58.2% for IBP and 58.6%
for IC. These values validate execution and cost only and are not paper-effect
estimates. Neither short gate exhibited a reference decline, so the IWAU
branches correctly returned `not_required` and reused their fixed trajectories.

Full-horizon one-start representative gates were then completed without test
access. On P1, fixed IBP/IC reached calibration peaks of 54.6%/64.8% and both
ended at 0%, whereas IBP+IWAU/IC+IWAU ended at 56.2%/65.0%. On P2, fixed
IBP/IC peaked at 58.2%/58.6% and ended at 0%, while the adaptive arms ended at
60.6%/61.4%. Both tasks exercised the `active` IWAU branch, all four methods
completed their horizons, and no numerical IC fallback occurred on P2. These
single-start calibration results are feasibility gates, not final effect
estimates, and will not be quoted as independent test performance.

The synthetic multi-class registry was also corrected before protocol-v2
freezing. `strategy: separate` now generates train, calibration, and test sets
from their three declared seeds and fits preprocessing on train only. The
retired P5 configurations explicitly request `combined_split`, preserving the
legacy one-pool behavior for provenance. Regression tests verify that changing
the calibration seed changes only the calibration split under the new strategy.
