# Implementation notes

## Epoch convention

- `B_t` is the interval-valued parameter state after exactly `t` updates.
- State row `t` stores metrics evaluated on `B_t`.
- Update row `t` stores the gradient width `Omega_t` evaluated on `B_t` and the transition
  `B_t -> B_{t+1}`.
- If the fixed-update peak is `B_tp`, IWAU first acts on `B_tp -> B_{tp+1}`.

Keeping state and update records in separate files makes every index physically unambiguous.

## Same-method calibration

An adaptive arm is calibrated only from the corresponding fixed arm: IBP+IWAU uses IBP-fixed, and
IC+IWAU uses IC-fixed. For the first post-peak reference state `B_tref` below the selected retention ratio,

`S_Omega = sum(Omega_t for t_p <= t < t_ref)` and
`coefficient = (T - t_p) / S_Omega`.

No cross-method calibration is performed.

## Coordinate-wise update

IWAU computes a tensor-valued effective step

`eta_i = eta_base / (1 + coefficient * Omega_i)`.

The mean effective step is logged for diagnostics, but the interval update uses the full coordinate-wise
tensor. Therefore the implementation matches the coordinate-wise width cap used in the theory.

## Update-rule fairness

The fairness baselines do not begin at epoch zero with unrelated schedules.
They all reuse the exact fixed trajectory through the same calibration-selected
state `B_tp` and perform the same number of continuation updates.  Inverse-time,
cosine, and step-decay receive one target final mean step frozen from
development/calibration data.  `GlobalIWAU` instead evaluates
`eta_t = eta_base / (1 + coefficient * mean(Omega_t))`, isolating the value of
coordinate-wise control while retaining IWAU's functional form and coefficient.

The reported update budget counts the complete fixed pre-run plus the
continuation, not only the visible second stage.  GPU timing is synchronized;
the output separately records pre-run, continuation, and frozen-test time and
the maximum allocated device memory across those stages.  A calibration status
of `not_required` means no reference decline occurred within the horizon, so
all controlled update arms equal the fixed arm by definition and that cell is
not informative about schedule differences.

## Step-node soundness

The derivative of ReLU is represented by a step node. Stable-sign intervals are exact (`[0,0]` or
`[1,1]`); an interval crossing zero uses the sound constant box `[0,1]`, including all subgradients at
zero. Other supported operators follow auto_LiRPA's sound IBP/CROWN propagation. IC intersects the two
sound results and cannot be looser than either input bound coordinate-wise.

Floating-point kernels may produce a numerical inconsistency between the two enclosures. On every affected
coordinate IC conservatively falls back to the IBP bound, regardless of gap size. The implementation logs
the number of fallback calls, affected coordinates, and maximum gap for training, calibration selection,
and final test evaluation. It never swaps endpoints or constructs an artificial midpoint.

## Controlled gradient-bound constructions

The public runner distinguishes four training-bound methods:

- `IBP`: interval propagation on the complete gradient graph;
- `AGT-Hybrid`: an AGT-inspired controlled ablation whose forward
  preactivations use CROWN while the loss derivative, ReLU derivative, backward
  matrix products, and parameter-gradient outer products use explicit interval
  arithmetic;
- `CROWN`: CROWN/linear-relaxation bounds on the complete supported gradient
  graph, with the ReLU-derivative step node conservatively represented by its
  constant interval box;
- `IC`: the coordinate-wise intersection of the `IBP` and `CROWN` bounds.

`AGT-Hybrid` is deliberately named *inspired*: the implementation is an
independent controlled baseline and is not asserted to reproduce every detail
of the AGT reference software.  Full-CROWN and IBP differ at bilinear nodes:
CROWN retains McCormick-type affine relations before later concretization,
whereas IBP uses four-corner interval multiplication.  Their one-node marginal
ranges can coincide, but their propagated bounds need not.

The `audit-bounds` command evaluates all constructions on one identical frozen
parameter box.  This separates bound tightness from the confounding effect of
different training trajectories.  Full-IC timing includes both source-bound
computations and the final intersection; reporting intersection-only time would
understate its cost.  Random point sampling in the audit is a regression check,
not a proof of real-arithmetic soundness.

The complementary `validate-small-bound` command is deterministic and
checkpoint-free.  It uses a `[1,1,1]` binary MLP and enumerates the lower
endpoint, midpoint, and upper endpoint of every input and parameter interval.
Its reference BCE gradients are computed by an independent analytic formula.
When the hidden preactivation is exactly zero, the check evaluates both
derivative endpoint choices, 0 and 1, so the implementation is tested against
the complete constant step box used by the mathematical construction.  The
default cases also vary the sign of the downstream weight to exercise the
bilinear backward products.  This remains a finite implementation diagnostic,
not a substitute for the real-arithmetic compositional proof.

## Terminal status

Runs distinguish `completed`, `diverged`, `numerical_error`, `failed`, and `interrupted`. `cert_at_T` is
reported only for a run that actually reaches `B_T`; otherwise it is `null`. Analysis code may choose an
explicit penalty for failed runs, but the raw logger never changes their meaning.

`max_gradient_width` is a configurable numerical safety stop on the mean gradient-bound width. It prevents
unbounded or non-finite trajectories from consuming resources; it is not the theoretical reference width
used in the finite-horizon analysis. The terminal status records the stop reason and the last observed
certified accuracy so that an endpoint penalty remains auditable.

## Clean accuracy

Clean accuracy is evaluated by setting every interval parameter to its exact midpoint and running the
resulting ordinary model on unperturbed inputs. During training, `calibration_clean_accuracy` uses only the
calibration split. The frozen selected and terminal checkpoints are later reported as
`test_clean_at_selected` and `test_clean_at_T`. No value is inferred from the midpoint of propagated output
bounds, which need not equal the output of the midpoint-parameter model in a nonlinear network.
