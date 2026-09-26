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

## Step-node soundness

The derivative of ReLU is represented by a step node. Stable-sign intervals are exact (`[0,0]` or
`[1,1]`); an interval crossing zero uses the sound constant box `[0,1]`, including all subgradients at
zero. Other supported operators follow auto_LiRPA's sound IBP/CROWN propagation. IC intersects the two
sound results and cannot be looser than either input bound coordinate-wise.

Floating-point kernels may produce a numerical inconsistency between the two enclosures. On every affected
coordinate IC conservatively falls back to the IBP bound, regardless of gap size. The implementation logs
the number of fallback calls, affected coordinates, and maximum gap for training, calibration selection,
and final test evaluation. It never swaps endpoints or constructs an artificial midpoint.

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
