# Experimental locking and final-test access

The final study uses disjoint training, calibration, and test splits. During
development, `evaluation.test_access` remains `disabled`; no reported test
result may be used to alter the method, retention ratio, schedule search space,
task list, or reporting rule.

Before a final run:

1. Copy `study_locks/experiment_lock.template.yaml` to a versioned lock file.
2. Freeze the code commit, dependency lock, auto_LiRPA source hash, data-split
   manifest, protocols, endpoints, update-budget rule, and statistical
   hypotheses. Each protocol entry contains the SHA-256 of its fully resolved
   configuration; an MNIST protocol also contains the SHA-256 of its explicit
   index manifest.
3. Set `status: frozen`, commit the lock, and record that commit identifier.
4. Give the final configuration `test_access: frozen_final` and the committed
   lock path. The runner rejects draft/incomplete locks and unlisted protocols.
5. Run calibration and checkpoint selection without evaluating test tensors.
6. Evaluate the frozen selected checkpoint and endpoint once with every declared
   test certifier on exactly the same test samples.

If the protocol changes after test access, the affected result is development
evidence only.  A new final evaluation requires a new lock and a genuinely
unused test split.  Lock files are part of the public reproducibility record;
they must not be edited retroactively.

The legacy test seed used while developing protocol-v1 and the early
protocol-v2 smoke runs is considered exposed. It must be listed under
`data.retired_development_test_splits` and may not appear as a final test seed.

The principal endpoint is defined in each frozen study lock. Earlier component and multi-radius
studies emphasize `test_cert_at_calibration_selected_checkpoint`, with `test_cert_at_T` as a
terminal outcome. The separate protocol-v6 size extension prespecifies the paired difference in
terminal IC test certification at `N=800` as its principal contrast; it does not revise the
earlier hypotheses. Parameter widths, gradient-bound widths, and effective steps are
calibration-only mechanism diagnostics.

For MNIST, a frozen-final configuration is rejected unless it supplies an
explicit split manifest. The manifest records indices in the filtered official
train and test sets, is checked for exact sizes, uniqueness, bounds, and
train/calibration disjointness, and is content-hashed in the study lock. This
prevents a nominally new seed from silently reusing an exposed development
sample.
