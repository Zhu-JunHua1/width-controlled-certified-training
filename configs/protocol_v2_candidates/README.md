# Protocol-v2 candidate configurations

These files encode the planned primary P1/P2 protocols while final test access
is still disabled. They are used for configuration review, seed-selection
audits, and calibration-only representative runs. They are **not** frozen-final
configurations and cannot produce paper test metrics.

Before final evaluation, each candidate is copied to a frozen configuration
whose `evaluation.test_access` is `frozen_final`, whose `analysis_tier` is
`primary`, and whose `evaluation.study_lock` names the committed frozen lock.
The complete resolved configuration hash is then recorded in that lock.

The final TwoMoons data seeds are the first three big-endian 32-bit words of:

```text
SHA256("width-controlled-certified-training-protocol-v2-final-twomoons")
= 35d0e402a5e5f76da62fb541cc319e37067fbe59c7421704d7793bd7e12ce29f
```

This gives train `902882306`, calibration `2783311725`, and test
`2788144449`. No metric was read while deriving them. The MNIST candidate uses
the committed explicit manifest documented in `data_splits/README.md`.

Accepted starts are never hand-picked. Candidates are scanned in ascending
order and the first ten satisfying the calibration-only pretraining rule are
retained. During the freeze gate, MNIST candidates `[0,15)` produced only five
valid starts under the new split, so the prospective pool was widened to
`[0,50)` before final-test access. This design decision and every rejection are
recorded in `seed_selection.json`.

For a trend axis, every variant uses the same accepted starts. A shared
manifest records the first ten candidate seeds that satisfy the declared
pretraining rule for every member of the axis. P3 uses separate `depth`,
`width`, and `step` groups; P6 uses one MNIST depth group; P7 uses one
training-length group. The P3 width pool was prospectively widened from
`[0,100)` to `[0,500)` after the original pool yielded only seven common
starts, and the P6 depth pool from `[0,75)` to `[0,150)` after it yielded only
eight. Both changes occurred while test access remained disabled. No
certification metric was used to accept or reject a seed.

The directory now contains all planned P1--P7 protocol families. P1/P2 are
prospective primary analyses; P3, P5, P6, and P7 are secondary; P4 is an
exploratory same-source context comparison because literature rows cannot be
treated as paired observations from this implementation. Every candidate keeps
test access disabled. Their final analysis tiers and study-lock path are added
only in separately hashed frozen configurations.

P5 uses independently generated train/calibration/test samples. Its split seeds
were derived in the same metric-free way from the phrases ending in
`p5-synthetic3` and `p5-synthetic4`; the exact values are committed in the YAML
files. The retired P5 files explicitly retain `strategy: combined_split` so the
old exploratory protocol remains reproducible.
