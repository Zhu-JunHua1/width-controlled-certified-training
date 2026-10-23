# Training-size extension: staged protocol plan

This plan was fixed before running the N=800 feasibility gate. It builds on
the completed, one-start, calibration-only N=100/200/400 development run. That
run is a feasibility observation, not a blind-test or population conclusion.
No protocol-v4 or protocol-v5 frozen artifact or result may be modified.

## Stage 1: resource-only N=800 gate

Run `configs/development/train_size_N800_feasibility.yaml` once on the same
Windows CUDA host, alone on the GPU. It uses one shared start, a fresh nested
800-sample *development* pool, 500 calibration samples, T=40, epsilon=0.001,
and IBP, IBP+IWAU, IC, IC+IWAU. Test access is disabled. The result must
contain four reported summaries and four `not_evaluated` placeholder rows,
with no test metrics. Normal scientific divergence is an observed outcome, not
a reason to exclude N=800.

N=800 passes the gate only if the process exits with code zero, produces the
complete grid, has no OOM or native failure, completes within four hours of
wall time for the four methods together, and records peak PyTorch-allocated
GPU memory below 8 GiB for every method. These are resource gates only:
calibration accuracy, width, method ranking, and IWAU trigger status must not
influence the size decision. If the gate fails, archive the exact failure and
freeze the core N=100/200/400 design instead of changing T, epsilon, model,
threat model, method list, or thresholds. PyTorch allocation is not whole-GPU
use and must not be reported as such.

## Stage 2: blind protocol, frozen before test access

If N=800 passes, use strict prefixes N=100/200/400/800 of one newly generated
800-example TwoMoons training pool. The earlier 400-example pool is not
comparable: generating 800 points changes the shorter prefixes. If N=800
fails, use strict N=100/200/400 prefixes of one newly generated 400-example
pool. Either way, rerun every size in the final study on its *new common pool*.

Freeze ten consecutive shared initialization seeds without outcome filtering,
pretraining disabled, one common 500-example calibration split, and one new
unopened 500-example blind-test split. All methods and sizes share the
corresponding initial parameter state. Freeze T=40, epsilon=0.001, base step
0.1, retention 0.8, architecture [2,20,20,1], full-training-set input
perturbation, IC calibration selection by first maximum, and both IBP and IC
test evaluators at the training radius. Each IWAU arm calibrates only from its
same-bound fixed pre-run, which counts in total cost. Test samples and
metrics must not enter start selection, pre-run calibration, or checkpoint
selection. The previously used development test placeholder must be rejected.

Before blind access, commit executable code and the analysis script, freeze
data seeds and the exact ten starts, configuration, endpoint definitions,
failure policy, environment/dependency hashes, source tree, and immutable
study lock. Require clean Git status and a test-isolation dry run. Run once
without a concurrent GPU experiment. Expect 160 summaries/320 evaluator rows
with four sizes, or 120/240 with three sizes. Validate exact cells, hashes,
provenance, completed values and diverged raw-null terminal values.

## Stage 3: prespecified analysis

The principal scale comparison is the paired difference between IC+IWAU and
IC-fixed in terminal IC-certified blind-test accuracy at the largest size
frozen before test access. Report selected-checkpoint IC certification under
the common first-maximum calibration rule, the other sizes, IBP-trained
comparisons, clean accuracy, completion/divergence/no-op counts, and both
test evaluators as secondary outcomes. Include every selected start,
including IWAU `not_required` runs. Preserve raw null terminal values for
divergence; the analysis value is zero under the prespecified policy.

Report per-start differences, mean effects and 95% confidence intervals.
If p-values are used, use a two-sided exact paired sign-flip test for the one
principal contrast; specify and apply Holm correction before interpreting a
family of secondary size-wise tests. Do not infer a universal scaling law
from ten starts. Classify this as a new blinded scale-extension study whose
design was informed by development data, not as an untouched original
hypothesis.

Cost reporting includes end-to-end wall time (including fixed pre-runs),
seconds per executed update, wall time divided by declared T, pre-run time,
peak PyTorch-allocated device memory, observed process/system memory where
available, failure status, and the host/toolchain/runtime policy. Disclose
that N=200 did not trigger IC+IWAU in the one-start development run rather
than selecting only effective continuations. Keep raw per-start rows and
trajectories available for audit.

## Resource-gate decision, 2026-10-22

The N=800 development process began at 16:29:31 Asia/Shanghai and exited 0
at approximately 17:25:55. Standard error was empty. It wrote four reported
summaries and four `not_evaluated` placeholder rows, all with test access
disabled and no test metrics. All four methods produced a recorded completed
status, and the largest peak PyTorch-allocated device memory was about 75 MiB.
The elapsed process time was about 56 minutes, under the prespecified four-hour
gate. **N=800 passes the resource-only gate**; no calibration certification
value was used in this decision. The four method `complete_elapsed_seconds`
values must not be summed to estimate process time because adaptive entries
include the same fixed pre-run already reported by a fixed arm.

The forthcoming blind study therefore uses N=100/200/400/800, all drawn as
strict prefixes of a newly seeded 800-example pool. Calibration-only common
start selection accepted seeds 0 through 9 without certified-accuracy or
test-metric filtering. Its audit is
`study_locks/candidate_audits/protocol_v6_scale_seeds.json`.

## Completed blind-study record, 2026-10-23

The frozen run launched once on 2026-10-22 and exited zero on 2026-10-23. It passed the committed
four-size result-grid validator: 160 reported summaries, 320 evaluator rows, exact frozen cells,
config/lock/code hashes, clean execution provenance, and completed/diverged terminal-field semantics.
The original result tree was not modified or rerun. The prespecified analysis and runtime reports were
committed under `artifacts/protocol_v6_scale_final/` at `a4a06a8`; the complete original result tree is
also published there as `raw_results.zip`, with its SHA-256 recorded in `docs/protocols.md`.

Under the common IC evaluator, terminal certification averaged 5.58% for fixed IC and 77.32% for
IC+IWAU at N=800, a paired difference of 71.74 percentage points (paired bootstrap 95% interval
59.16--79.82; two-sided exact sign-flip p=0.001953125). Both IC arms completed all ten starts at
every size. The result is a blind extension motivated by prior development observations, not a
general scaling law. Full four-method results, clean accuracy, failure counts, and charged runtime
are in the committed reports; this completion record does not amend the earlier gate or hypotheses.
