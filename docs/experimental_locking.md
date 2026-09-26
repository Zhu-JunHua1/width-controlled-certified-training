# Experimental locking and final-test access

The final study uses disjoint training, calibration, and test splits.  During
development, no reported test result may be used to alter the method, retention
ratio, schedule search space, task list, or reporting rule.

Before a final run:

1. Copy `study_locks/experiment_lock.template.yaml` to a versioned lock file.
2. Freeze the code commit, dependency lock, auto_LiRPA source hash, data-split
   manifest, protocols, endpoints, and statistical hypotheses.
3. Set `status: frozen`, commit the lock, and record that commit identifier.
4. Run calibration and checkpoint selection without materializing test loaders.
5. Evaluate the frozen selected checkpoint and endpoint once with every declared
   test certifier on exactly the same test samples.

If the protocol changes after test access, the affected result is development
evidence only.  A new final evaluation requires a new lock and a genuinely
unused test split.  Lock files are part of the public reproducibility record;
they must not be edited retroactively.

The primary endpoint is `test_cert_at_calibration_selected_checkpoint`.
`test_cert_at_T` is secondary.  Parameter widths, gradient-bound widths, and
effective steps are calibration-only mechanism diagnostics.
