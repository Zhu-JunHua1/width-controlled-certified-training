# Frozen protocol-v2 configurations

These files are the only configurations authorized to access the protocol-v2
test partitions. Each resolved configuration is content-hashed in
`study_locks/protocol_v2_frozen.yaml`. The lock is fail-closed: changing a
configuration, MNIST split manifest, or shared-start manifest invalidates test
access.

P1 and P2 are the predeclared primary analyses. P3, P5, P6, and P7 are
secondary; P4 is exploratory context. Checkpoint selection, IWAU calibration,
and mechanism diagnostics use calibration data only. Test data are evaluated
only at the frozen selected checkpoint and the predeclared terminal epoch.

