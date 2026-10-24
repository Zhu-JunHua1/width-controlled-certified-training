# Secondary descriptive aggregates

These two CSV files preserve the complete values removed from the compact
supplementary PDF. They are transcriptions of the previously reported aggregate
tables, not raw observations, new experiments, or a new analysis tier.
No frozen configuration, split, seed, endpoint, or result tree was changed.

- [p3_sensitivity.csv](p3_sensitivity.csv): TwoMoons width and base-step variants at
  epsilon=0.001. Each entry is mean test certification (%) over 10 shared
  starts. Checkpoints use IC-certified calibration selection. Column names
  identify the common test certifier and training method; the final column
  is the declared terminal epoch. These comparisons are descriptive.
- [p4_clean_input.csv](p4_clean_input.csv): exploratory TwoMoons external-context study at
  T=20, from-initialization training and seed-specific data splits. Each
  entry is mean calibration-selected test certification (%) over 10 starts
  under the common IBP parameter-set certifier and clean test inputs.
  The hybrid is a local AGT-inspired implementation, not official AGT.
  These values must not be pooled with the blind main study.

For study configurations and evidence tiers, see
[docs/protocols.md](../../docs/protocols.md).
The CSV files are intended for checking the displayed descriptive aggregates;
they do not replace the frozen per-start result trees or study locks.
