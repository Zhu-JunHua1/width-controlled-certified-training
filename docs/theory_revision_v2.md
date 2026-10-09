# Theory revision v2: precise scope and implementation boundary

This note freezes the mathematical scope to be used when revising the paper.
It deliberately separates real-arithmetic enclosure results, trajectory
properties on the calibration split, and empirical final-test statements.

## 1. Fixed-step IC nesting

Let `Gamma_M(B, D_t)` be the gradient enclosure produced by method `M` on
parameter box `B` and the fixed training batch/domain `D_t`.  Let
`U(B, G, eta_t)` be the interval-SGD update with a non-negative step tensor
`eta_t` that is fixed in advance and independent of the bound method.

The per-step nesting statement requires all of the following:

1. identical initial boxes;
2. identical training samples, perturbation domains, and batch order;
3. the same predeclared scalar or coordinate-wise step tensor at every step;
4. inclusion isotonicity of the IBP enclosure;
5. `Gamma_IC(B,D) subseteq Gamma_IBP(B,D)` on the same box;
6. joint inclusion isotonicity of the interval update in its state and gradient
   enclosure arguments; and
7. one common evaluation certifier whose certified set is antitone in the
   parameter box.

Under those conditions, induction gives
`B_t^IC subseteq B_t^IBP`, and the common certifier reports no lower certified
accuracy for IC at each step.  The statement does **not** compare an IBP-trained
model evaluated by IBP with an IC-trained model evaluated by IC; that mixes
training and evaluation effects.  It also does **not** apply automatically to
IBP+IWAU versus IC+IWAU, because each method computes a different width tensor
and therefore a different coordinate-wise step tensor.

## 2. Width control and the lambda=0 branch

For an active coefficient `lambda > 0`, IWAU uses

`eta_{t,i} = eta_base / (1 + lambda Omega_{t,i})`,

so the exact interval-width increment satisfies

`Delta w_{t,i} <= eta_base / lambda`.

Summation gives the coordinate-wise and mean finite-horizon bounds in the
paper.  Every formula containing `eta_base/lambda` is explicitly conditioned
on `lambda > 0` and a positive calibration sum `S_Omega`.

If no reference decline occurs in the fixed pre-run, calibration returns
`not_required` and `lambda=0`.  This is a separate identity branch: the
reported arm equals the fixed arm.  It is not obtained by substituting zero
into the positive-lambda bound.

## 3. What direct peak continuation does and does not prove

The fixed pre-run selects `t_p` on the calibration split.  Reusing the exact
prefix through `B_{t_p}` makes the calibration trajectory of the controlled
arm contain the fixed arm's selected calibration state.  Therefore calibration
peak non-inferiority is a direct construction property and should be a remark,
not a main theorem.

It gives no test-generalization theorem.  The independently held-out test set
is evaluated only after the checkpoint and update rule are frozen.  Test
certified accuracy at the calibration-selected checkpoint is the primary
empirical endpoint; test cert@T is the secondary stability endpoint.

## 4. Full-gradient-graph enclosure in real arithmetic

The soundness claim is a compositional real-arithmetic claim.  Each supported
operator has a local enclosure:

- affine map with interval parameters: natural interval/McCormick-compatible
  enclosure;
- forward ReLU: standard CROWN affine lower and upper relaxations;
- BCE loss derivative: monotone interval evaluation of `sigmoid(z)-y`;
- ReLU derivative: the point rule is `1{z > 0}`; its enclosure is `[0,0]` when
  the upper endpoint is non-positive, `[1,1]` when the lower endpoint is
  positive, and `[0,1]` when the interval crosses zero;
- matrix products and parameter-gradient outer products: sound bilinear
  relaxation (McCormick in the CROWN branch, four-corner interval arithmetic in
  the interval branch); and
- sums and reshapes: exact linear propagation.

At `z=0`, the implemented point derivative is zero. The crossing-interval box
`[0,1]` covers all point-rule values on that interval. It is not described as a
subgradient of the discontinuous step function.

Composing locally sound, inclusion-preserving relaxations on the explicit
gradient graph encloses every real gradient represented by the input and
parameter boxes.  Shared variables may be relaxed, losing tightness, but the
outer enclosure remains sound.  IC intersects two real-arithmetic sound
enclosures and is therefore sound and coordinate-wise no looser than either
input enclosure.  Strict improvement is conditional, not universal.

## 5. Floating-point implementation boundary

PyTorch/auto_LiRPA kernels do not provide directed-rounding interval
arithmetic.  The paper must therefore distinguish:

- a real-arithmetic soundness theorem for the mathematical construction; and
- implementation validation under finite precision.

If floating point yields an empty IC coordinate (`lower > upper`), the code
falls back to the complete IBP coordinate and records the event.  It never
swaps endpoints or inserts a midpoint.  This policy is conservative relative
to the two computed floating enclosures, but it does not by itself turn ordinary
floating arithmetic into a formally rounded interval implementation.

Random sampling, dense grids, high-precision checks, and small exact/MIP
comparisons are implementation tests only.  They cannot replace the
real-arithmetic proof and must not be called soundness proofs.

## 6. Width-to-certification statement

The finite-horizon theorem controls interval width, not certified accuracy.
Any general endpoint-persistence statement remains empirical/conditional.
The Lipschitz-margin proposition is a separate sufficient condition.  Its
derivation must use a supremum of (generalized) derivatives along the complete
line segment / domain, so that the mean-value integral is an inequality rather
than a first-order approximation.  Sampled estimates of those suprema can only
give optimistic diagnostic coverage and cannot produce a certified lower
bound.

## 7. Consequences for the revised manuscript

- Promote finite-horizon width control as the main theorem.
- Restrict fixed-step IC dominance to the seven conditions in Section 1.
- Remove any extension of that theorem to separately calibrated IWAU paths.
- Demote peak non-inferiority to a calibration-trajectory construction remark.
- Present independent-test results as empirical endpoints, not corollaries.
- State the `lambda=0` branch separately from all positive-lambda formulas.
- Replace broad implementation-level “sound” claims by the real-arithmetic /
  finite-precision distinction above.
- Use the training-method by test-certifier matrix to separate optimization
  effects from evaluator strength.
