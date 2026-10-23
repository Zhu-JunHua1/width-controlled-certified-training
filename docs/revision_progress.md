# Protocol-v2 revision progress

This file records implementation milestones for the post-review protocol.  A
completed implementation milestone is not a final experimental result; final
test values require a frozen study lock.

## Current publication status (protocols v4--v6 complete)

- The protocol-v4 blinded update-rule study is complete. Its six prespecified paired comparisons and
  runtime summaries are published under `artifacts/protocol_v4_final/`.
- The protocol-v5 TwoMoons multi-radius main study is complete: 120 training summaries and 240 common-
  evaluator rows across three radii, ten starts, four methods, and two test certifiers. The manuscript
  table and appendix artifacts are under `artifacts/protocol_v5_final/`.
- The separately frozen protocol-v6 training-size extension is complete: 160 summaries and 320
  evaluator rows across nested `N=100/200/400/800` TwoMoons training sets. Its aggregate, per-start,
  runtime, statistical, and raw-output archive artifacts are under `artifacts/protocol_v6_scale_final/`.
  It is not pooled with the multi-radius main study.
- P3/P6/P7 remain secondary robustness evidence; corrected P5-v3 remains post-hoc exploratory evidence.
- The constrained official FullCert adapter passed development gates. Its one-time frozen extension
  stopped after 4 of 30 cells on a non-finite interval update, so no official FullCert aggregate is
  reported and the local `IBP-fixed` arm remains labelled FullCert-style.

See `docs/protocols.md` for the current evidence hierarchy. The sections below are retained as a
chronological development record and must not be interpreted as the current run status.

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

Protocol-v4 schedule-fairness preparation:

- Added a calibration-only nine-candidate scalar-schedule search for each
  primary task and a fail-closed selector with explicit candidate/start counts.
- Added a prospective cumulative-step-matched cosine control. Its task-level
  target is frozen from previously exposed development trajectories, not from
  a paired blind-test IWAU trajectory.
- Generated `blind_mnist17_protocol_v4.json` (file SHA-256
  `3efdea77d8a697e618e00842e038df5a08f84f21295f4a4b3de2a28652a43791`).
  It contains 100/500/500 train/calibration/test indices, train and calibration
  are disjoint, and overlap with protocol-v2 is zero in each corresponding
  official MNIST namespace.
- Corrected scalar-control execution so that a `lambda=0` IWAU calibration does
  not suppress an independently specified scalar schedule. IWAU and
  GlobalIWAU alone retain the `not_required` fixed-update branch.
- A one-start development smoke run exercised both new rules from the same
  epoch-12 prefix. The cumulative cosine realized mean step
  `0.05886784227532821` for target `0.0588678422753282`; both summaries contain
  an empty `test_evaluations` list. These are implementation checks, not paper
  results.
- Full CPU regression suite: 86 passed.
- The initial nine-candidate search completed without test access. P1 selected
  inverse-time/0.04 with mean calibration `cert@T=2.87%`; all nine P2
  candidates ended at zero. Before blind-test access, the search was therefore
  extended symmetrically on both tasks with final-step targets
  `0.005/0.01/0.02`. Both initial and extension results remain reportable, and
  the final selector will rank all 18 candidates under the unchanged rule.

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

## Stage C/D — frozen primary experiments (P1/P2 complete)

The study lock was committed at `2f3a37d` before final-test access.  P1
(TwoMoons, radius 0.01) and P2 (MNIST 1/7, radius 0.001) were each executed
once under the frozen protocol.  Each experiment contains 10 selected starts,
10 declared training methods, two frozen test certifiers (IBP and IC), and
exactly 200 result rows.  The output records one lock hash, one resolved-config
hash, one data-split hash, and the locked code commit.  Completed trajectories
retain non-null selected and terminal test fields; diverged trajectories keep
their raw terminal fields null rather than silently converting the run into a
completed zero endpoint.

All four preregistered primary comparisons reject their one-sided null after
Holm correction:

- P1 IC vs IBP at the calibration-selected checkpoint under the common IBP
  test certifier: +11.38 percentage points, 95% bootstrap CI [4.72, 17.78],
  Holm-adjusted p=0.03515625.
- P2 IC vs IBP under the same endpoint/certifier policy: +0.38 points,
  95% CI [0.14, 0.70], Holm-adjusted p=0.03125.
- P1 IC+IWAU vs IC at the terminal endpoint under the common IC certifier and
  preregistered diverged-as-zero policy: +65.24 points, 95% CI [59.06, 71.12],
  Holm-adjusted p=0.00390625.
- P2 IC+IWAU vs IC under the same terminal policy: +71.22 points, 95% CI
  [60.04, 81.40], Holm-adjusted p=0.00390625.

The frozen raw outputs remain unchanged.  A portable paired-statistics CSV and
content-hashed aggregate reports are generated as derived artifacts.  The
public runtime reporter now distinguishes total training-pipeline time,
wall-clock seconds per reported epoch, and seconds per actually executed
interval update.  This separation is necessary because IWAU and the matched
two-stage controls charge their fixed pre-run to total cost.

P3--P7 remain frozen secondary/exploratory families. P6 is currently running
from an isolated worktree at the locked commit; the other families remain
pending. Their legacy results are not substitutes for the protocol-v2
evaluations.

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

The remaining P3--P7 candidate families are also encoded with final-test access
disabled. Trend axes now consume committed common-start manifests instead of
selecting a different accepted set for each architecture or horizon. The
calibration-only common sets are:

- P3 depth: `0,10,11,17,26,31,34,35,40,43`;
- P3 width: `14,23,47,58,62,80,96,101,109,120`;
- P3 step size: `0,4,7,9,10,11,12,17,18,19`;
- P6 MNIST depth: `9,17,29,30,54,59,62,74,79,82`;
- P7 horizon: `0,4,5,6,7,9,10,11,12,13`.

The selector records every member's pretraining outcome and explicitly states
that no test metric was read. Candidate ranges were widened only when the
original range contained fewer than ten starts accepted by all members.

Before freezing, the P1/P2 primary candidates were extended with the
predeclared ordinary schedule controls (`InverseTime`, `Cosine`, and
`StepDecay`), the scalar `GlobalIWAU` control, and fixed AGT-inspired hybrid
and Full-CROWN training arms. Full-horizon, one-start calibration gates passed
for every executable arm. On both tasks, coordinate-wise IWAU retained a
higher terminal calibration certificate than GlobalIWAU, while all three
matched-final-step time schedules ended at zero. The MNIST Full-CROWN gate
diverged after attaining an early calibration peak; that status is preserved
rather than rewritten as a completed zero endpoint. These are feasibility and
failure-mode checks, not final test estimates.

The complete 25-configuration frozen family is staged under
`configs/protocol_v2_frozen/`. Each configuration names the same fail-closed
study-lock path, and shared-start manifests are now themselves content-hashed
by the lock verifier. Final test access remains impossible until the frozen
lock is populated with the committed code and configuration hashes.
# 2026-10-08: A0 implementation-authenticity gate

The full-gradient CROWN execution path was audited before beginning the new
schedule-fairness study. Representative binary networks with one, two, and
three hidden layers and a four-class one-vs-all BCE network all satisfied the
following checks:

- every weight and bias input was marked perturbed;
- `BoundLinear.bound_backward_with_weight` executed;
- the doubly perturbed `BoundMul` path executed;
- the upstream ReLU and sigmoid relaxations executed; and
- the custom zero-slope Step box executed inside back-substitution.

The generated graph inventory and call trace are stored under
`artifacts/audit/a0_full_crown/`. A dense one-hidden-unit grid checked 10,080
method--coordinate inclusions without a violation; this remains an
implementation diagnostic rather than a proof. The complete Python test suite
passed at audit time.

The audit also found a provenance mismatch that had to be disclosed. The
historical runtime auto_LiRPA tree hash is `6a8d960...`, whereas the recognized
upstream `9d100ec` tree hash is `2bc6390...`; three files differ. The four A0
graphs produced byte-identical IBP and CROWN endpoints under both trees, but
the repository no longer describes the historical dependency as unmodified.
The exact three-file patch, upstream BSD license, preparation command, and
expected tree hash are now published under `third_party/auto_lirpa`; applying
the patch to the pinned upstream commit reproduces the historical hash.

The manuscript appendix now states the actual Step convention (`1{z>0}`), the
two McCormick planes selected by the default non-optimized path, the
positive/negative-coefficient substitution rule, and an operator-to-code map.
The proof claim remains limited to exact real arithmetic; ordinary floating
point is path-audited and numerically tested but not outward-rounded.
