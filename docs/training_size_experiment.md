# Training-set-size study: development stage

This stage tests whether the full-training-set perturbation implementation remains usable as the number of perturbed training examples increases. It is a development feasibility study, not an independent blinded efficacy claim. No existing frozen protocol is altered or rerun.

## Fixed design for the development run

- Task: TwoMoons; model dimensions `[2, 20, 20, 1]`; radius `0.001`; horizon `40`.
- One training pool of 400 examples is generated with seed `20261022` and permuted once. The training sets with `N = 100, 200, 400` are strict prefixes of this same pool.
- The 500-example calibration split is identical for every N. The test seed is a placeholder and `test_access=disabled`; no test accuracy or certification is computed.
- One shared initialization seed (`0`) is used across N, with point-model pretraining disabled so that initial weights do not depend on N.
- Methods: IBP-fixed, IBP+IWAU, IC-fixed, and IC+IWAU. Every adaptive arm includes its same-bound fixed pre-run in total time. The evaluation configuration retains IBP and IC for a later frozen study, but neither is run on test data here.
- Record status, executed updates, calibration-selected and terminal calibration certification, total wall time, time per executed update, peak GPU allocated memory, and whether IWAU actually continued from the fixed prefix.

Configuration: `configs/development/train_size_nested_T40.yaml`. The initial one-pass and five-epoch non-nested probes are recorded outside the publication repository at `E:\Desk\Study\lirpa_cert\tmp\scale_pilot\README.md` and must not be interpreted as paired N effects.

## Gate before any blind extension

After the development grid finishes, require all 12 method-by-N cells, exact common initialization and nested training data, no test evaluations, complete cost accounting, and explicit failure statuses. A larger study would require several shared starts, a newly generated unseen test seed, a committed config/analysis/lock, clean executable provenance, and a single test reveal. The development calibration trajectories may inform feasibility, but cannot be reported as blind-test improvement. If N=400 is infeasible, report the resource limit instead of silently changing N or the threat model.

## Development run on 2026-10-22

All 12 reported summaries and 12 **development placeholder** rows were written. These are not 24 evaluator rows: with test access disabled, each method has one `evaluation_method=not_evaluated` row and `test_evaluations=[]`. All training summaries say `completed`; the three N settings have identical SHA-256 hashes for the serialized initial model. The exact configured datasets were checked to have nested training prefixes and identical calibration/test tensors. Every adaptive summary includes its fixed pre-run in `complete_elapsed_seconds`; all rows have no test certificate values. The executable commit recorded by the run is `ed52f9749bd1636563e3957b8aa5e3cd41e2f51a` with `code_dirty=true`, as expected for an unfrozen development run.

The Python process nevertheless returned Windows exit code `-1073740791` (`0xC0000409`) **after** printing all three `results written` messages. Standard error contains no Python traceback. This is an abnormal native-process exit; the complete output grid is useful diagnostically but must not be described as a clean completed execution or as final evidence. The native exit requires investigation before a frozen scale study.

The following values are calibration-only percentages. `Seconds/T` is the full wall time divided by the declared 40 epochs and includes any IWAU fixed pre-run; it is not the mean time for a single unique gradient update. Peak memory is PyTorch-allocated device memory, not whole-process GPU usage.

| N | Method | Status | IWAU active | Peak cal. cert. % | Cal. cert. at T % | Total s | Seconds/T | Peak MiB |
|---:|:---|:---|:---:|---:|---:|---:|---:|---:|
| 100 | IBP | completed | no | 62.8 | 0.0 | 23.7 | 0.59 | 60.3 |
| 100 | IBP+IWAU | completed | yes | 63.4 | 61.4 | 38.6 | 0.96 | 60.3 |
| 100 | IC | completed | no | 64.4 | 39.0 | 316.5 | 7.91 | 79.2 |
| 100 | IC+IWAU | completed | yes | 64.6 | 62.4 | 519.3 | 12.98 | 79.2 |
| 200 | IBP | completed | no | 69.8 | 4.0 | 142.2 | 3.55 | 60.9 |
| 200 | IBP+IWAU | completed | yes | 70.0 | 69.6 | 219.9 | 5.50 | 60.9 |
| 200 | IC | completed | no | 72.4 | 60.8 | 1422.8 | 35.57 | 79.2 |
| 200 | IC+IWAU | completed | no | 72.4 | 60.8 | 1422.8 | 35.57 | 79.2 |
| 400 | IBP | completed | no | 70.6 | 1.4 | 279.6 | 6.99 | 60.9 |
| 400 | IBP+IWAU | completed | yes | 72.0 | 72.0 | 423.3 | 10.58 | 60.9 |
| 400 | IC | completed | no | 73.4 | 54.2 | 1617.5 | 40.44 | 79.2 |
| 400 | IC+IWAU | completed | yes | 74.4 | 74.4 | 2219.6 | 55.49 | 79.2 |

At N=200, IC+IWAU did not perform an adaptive continuation: calibration reported `not_required` because no reference decline occurred within T. Its equality with fixed IC is therefore a no-op, not evidence of an IWAU gain. The non-monotonic IC runtimes across N also rule out estimating a scaling law from this single run; GPU contention and warm-up were not controlled.

## Native-exit diagnosis and recovery

Windows Event Viewer identified a `ucrtbase.dll` BEX64 failure with exit code
`0xC0000409` after the original process finished writing results. Minimal CUDA
auto_LiRPA CROWN probes reproduced the failure; GPU IBP and CPU CROWN did not.
Disabling only TorchScript GPU fusion before bound-graph construction allowed
the same probe to exit normally. One IC gradient-bound probe produced the same
SHA-256 digest of 501 lower/upper float32 values with and without that setting:
`3a38a8f13668ea697fe7aafccb19341e1d4feb62147e86bece4d5b79b5f5485b`.
This is a limited implementation parity check, not a universal numerical proof.

The runtime now applies this Windows/CUDA-only policy before any bounded graph
is used, records it in run metadata, and fails closed if the PyTorch controls
are unavailable. A separate two-epoch, four-method development smoke run
completed and exited with code zero. The original T40 outputs remain untouched;
the full-grid recovery uses
`configs/development/train_size_nested_T40_recovery.yaml` and a separate result
root. This recovery is still calibration-only and is not a blind-test result.

## Clean T40 recovery outcome

The recovery run at executable commit `d1ae63530b28a732a0fc6b022875635f48a99e3f`
exited with code **0** and an empty stderr log. It wrote exactly 12 reported
summaries and 12 `not_evaluated` development placeholder rows (four methods at
each N). Every run completed. Each summary has `test_evaluations=[]` and
`primary_endpoint=not_evaluated_in_development`; all held-out test result
fields are empty. The run records a clean code tree and the
`windows_cuda_jit_gpu_fusion_disabled` runtime policy. All 12 serialized
initial parameter states are byte-identical. Regeneration verifies strict
N=100/200/400 training prefixes and identical calibration tensors.

The table below gives **one-start, calibration-only descriptive values**.
`Total s` includes the same-bound fixed pre-run for each adaptive arm;
`Seconds/T` divides that complete wall time by the declared 40 epochs and is
not the cost of one unique gradient update. Memory is peak PyTorch-allocated
GPU memory. These observations cannot establish statistical scaling or a
blind-test certification benefit.

| N | Method | IWAU active | Peak cal. cert. % | Cal. cert. at T % | Total s | Seconds/T | Peak MiB |
|---:|:---|:---:|---:|---:|---:|---:|---:|
| 100 | IBP | no | 62.8 | 0.0 | 19.4 | 0.48 | 60.3 |
| 100 | IBP+IWAU | yes | 63.4 | 61.4 | 31.7 | 0.79 | 60.3 |
| 100 | IC | no | 64.4 | 39.0 | 249.3 | 6.23 | 80.0 |
| 100 | IC+IWAU | yes | 64.6 | 62.4 | 363.8 | 9.09 | 80.0 |
| 200 | IBP | no | 69.8 | 4.0 | 34.3 | 0.86 | 60.9 |
| 200 | IBP+IWAU | yes | 70.0 | 69.6 | 53.4 | 1.34 | 60.9 |
| 200 | IC | no | 72.4 | 60.8 | 585.5 | 14.64 | 79.9 |
| 200 | IC+IWAU | no | 72.4 | 60.8 | 585.5 | 14.64 | 79.9 |
| 400 | IBP | no | 70.6 | 1.4 | 75.5 | 1.89 | 60.9 |
| 400 | IBP+IWAU | yes | 72.0 | 72.0 | 117.1 | 2.93 | 60.9 |
| 400 | IC | no | 73.4 | 54.2 | 1226.8 | 30.67 | 79.9 |
| 400 | IC+IWAU | yes | 74.4 | 74.4 | 1711.9 | 42.80 | 79.9 |

At N=200, IC+IWAU was `not_required`: its reference IC trajectory did not
meet the declared trigger, so it made no adaptive continuation and retained
the IC-fixed result and cost. Both the abnormal-exit run and this clean rerun
show the same calibration certificate percentages, but runtimes differ
substantially. Further resource-scaling claims require controlled repetitions.
