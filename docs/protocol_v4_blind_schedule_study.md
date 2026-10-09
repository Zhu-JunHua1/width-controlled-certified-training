# Protocol v4: blind schedule-fairness confirmation

## Status

`development-design`: implementation and calibration search are allowed; blind
test evaluation is not. This protocol does not modify or overwrite protocol-v2
or protocol-v3 artifacts.

## Purpose

The study addresses two residual fairness questions without reusing an exposed
test set as confirmation:

1. can a scalar time schedule selected with the same calibration-only search
   budget match IWAU; and
2. does IWAU remain competitive when a scalar cosine schedule is matched to a
   frozen cumulative continuation-step budget, not merely its final step?

## Data isolation

- TwoMoons seeds are the first 32 bits of SHA-256 applied to the literal labels
  `protocol-v4-twomoons-{train,calibration,blind-test}`. They are fixed before
  any run: `2268034632`, `2931978507`, and `2695485613`.
- MNIST 1/7 uses a new explicit index manifest. Every train, calibration, and
  test index appearing in protocol v2 is excluded before deterministic
  selection in its corresponding official MNIST namespace. Train and
  calibration indices address the filtered official training split; test
  indices address the separate filtered official test split. The ordering
  seeds are derived analogously and are
  `1951381023`, `1408693209`, and `1127648377`.
- Development runs set `test_access: disabled`. Their output must contain no
  frozen-test metrics. The blind test becomes accessible only after the code,
  selected schedule, configurations, endpoints, hypotheses, and hashes are
  frozen in a new study lock.

## Equal-budget scalar search

Each task evaluates exactly nine candidates on the same three accepted starts:

- inverse-time, cosine, and step-decay;
- final-step targets `0.04`, `0.06`, and `0.08`;
- step-decay uses the predeclared midpoint milestone `0.5`.

All candidates reuse the same fixed prefix through calibration-selected
`t_p`, have the same continuation length, and include the full fixed pre-run in
their cost. The winner maximizes mean calibration `cert@T`; a non-completed run
contributes zero. Ties are resolved by mean calibration peak, elapsed time,
family name, and target step in that order. Candidate deletion after observing
results is prohibited.

## Cumulative-step comparator

`IC+CumulativeCosine` uses the same prefix and continuation count as IWAU. Its
task-level target mean continuation step is frozen from already exposed
development/calibration trajectories, never from the new blind test or the
paired final-run IWAU trajectory:

- TwoMoons: `0.0588678422753282`;
- MNIST 1/7: `0.0559918078835364`.

For `n` continuation updates, the cosine minimum is solved analytically so the
mean of its `n` scalar steps equals the task target. This is a prospective,
deployable population match, not a per-seed oracle match.

## Final methods and budgets

The final blind comparison will include at least IC-fixed, IC+IWAU,
IC+GlobalIWAU, the calibration-selected `IC+TunedScalar`, and
IC+CumulativeCosine. Mechanism matching requires the same state `B_tp`, bound
method, certifier, and continuation count. Resource matching uses the same
number of expensive gradient-bound evaluations and reports complete wall-clock
time, peak memory, the realized step sum/mean/final value, and seconds per
executed update.

## Endpoints and reporting

Primary schedule endpoint: IC-certified blind-test accuracy at `T`.
Secondary endpoints: IC-certified accuracy at the calibration-selected
checkpoint, clean accuracy, trajectory AUC, first collapse epoch, complete
time, and peak memory. Paired effect sizes and confidence intervals accompany
the paired tests. All configured methods, failures, and zero effects are
reported.
