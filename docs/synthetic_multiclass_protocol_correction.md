# Synthetic multi-class protocol correction (P5-v3)

P5-v2 is retained as an immutable historical result but must **not** be used
as evidence about class-count scaling or ordinary held-out generalization.
Its `separate` strategy called `sklearn.datasets.make_classification` once
for each split, with a different random state. That regenerated the class
clusters and feature geometry instead of sampling one fixed classification
problem. The observed train/calibration/test discrepancy is therefore a
protocol defect, not evidence that IC generally underperforms IBP.

P5-v3 uses `shared_generator_split`: one generator call produces all 700
examples, after which two seeded, class-stratified partitions produce 100
training, 300 calibration, and 300 test examples. `StandardScaler` is fitted
only to training inputs. Model initialization seeds 0--9, architecture,
training horizon, step size, radii, methods, and test certifiers remain the
same as P5-v2; the generated classification problem and its partitions are
new. Calibration alone selects checkpoints and IWAU parameters. Test samples
are evaluated only after the corrected configuration and code are frozen.

The six integers in `configs/protocol_v3_frozen/P5_synthetic{3,4}_corrected.yaml`
were fixed before generating or inspecting any P5-v3 final test data. Each is
the unsigned big-endian value of the first four SHA-256 bytes of
`protocol-v3-multiclass-{classes}-{role}-2026-10-01`, for `role` equal to
`generator`, `calibration_split`, or `test_split`. These do not reuse the
exposed P5-v2 seeds. This is a post-hoc protocol repair, so its results are
**exploratory**; the four P1/P2 pre-registered hypotheses remain unchanged.

Do not edit `configs/protocol_v2_frozen/`, `study_locks/protocol_v2_frozen.yaml`,
or their result directories. Record the corrected code commit and fully
resolved P5-v3 configuration hashes in a separate study lock before final
test access. Never substitute P5-v2 cells for missing P5-v3 cells.
