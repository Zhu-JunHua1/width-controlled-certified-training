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
