# A0 full-gradient CROWN implementation audit

## Scope and status

This audit answers a narrow implementation question: does the code path used by
the experiments actually treat the data and network parameters as perturbed
variables and propagate CROWN through the full forward--backward gradient graph?
It does **not** replace the real-arithmetic soundness proof, and it does not
establish floating-point soundness.

Current decision-gate status:

- **Execution-path check: passed.** Representative binary networks with one,
  two, and three hidden layers and the four-class one-vs-all BCE graph all use
  perturbed parameter inputs, weight-perturbed matrix-product relaxations,
  doubly perturbed product relaxations, and the custom Step relaxation.
- **Dependency provenance: action required.** The historical frozen runs used
  an auto_LiRPA Python tree whose SHA-256 is
  `6a8d960cdd013cbd63cc814d2e21e30ab2f3f23bada889eaf4a91f4be3994c73`.
  It differs in three files from upstream commit
  `9d100ec070868440b48d34e2f1dd21b97aab9172`, whose corresponding tree hash is
  `2bc6390a98d9e88df4c8a2460c9e1cd2e4d7e7e7b058970f665bf7379febff2f`.
  Therefore the historical environment must not be described as an unmodified
  upstream checkout. On the four representative A0 graphs, however, the IBP
  and CROWN endpoint tensors were byte-identical between the two trees. This is
  useful regression evidence, not a proof that every training trajectory is
  identical.
- **Proof--implementation closure: not yet passed.** The executed rules have
  been identified, but the manuscript still needs an induction over the actual
  backward affine-substitution rules. Passing the tests below is validation,
  not proof.

## Actual call path

The experiment runner constructs
`BoundedModule(BinaryBCEGradientGraph(...), custom_ops=CUSTOM_BOUND_OPERATORS)`.
Training data and every weight and bias tensor are converted to `BoundedTensor`
objects with explicit lower and upper endpoints. The runner then calls
`compute_bounds(method="IBP")` and `compute_bounds(method="CROWN")`; Full-IC
intersects these results coordinate by coordinate.

The multiclass protocol uses one-vs-all BCE and the same explicit gradient
graph, with the output dimension equal to the class count. It does not take a
separate softmax-cross-entropy CROWN path.

## Operator-to-implementation map

| Gradient-graph operation | Real-arithmetic enclosure used | Executed implementation | Regression evidence |
|---|---|---|---|
| Affine map with interval weight | Bilinear relaxation for each activation--weight product, followed by affine summation | `auto_LiRPA.operators.linear.BoundLinear.bound_backward_with_weight`; `BoundMatMul.bound_backward` delegates to it | call count is positive in every audited architecture |
| Product of two perturbed quantities | For `z=xy`, the default lower plane is `y_l x + x_l y - x_l y_l`; the upper plane is `y_u x + x_l y - x_l y_u` | `auto_LiRPA.operators.bivariate.MulHelper.get_relaxation` and `BoundMul.bound_backward_both_perturbed` | call count is positive whenever a hidden-layer derivative is multiplied by an upstream gradient |
| ReLU derivative / Step | `[1,1]` if the preactivation lower endpoint is positive; `[0,0]` if its upper endpoint is non-positive; otherwise `[0,1]` | `width_cert.bounds.operators.BoundStepActivation.bound_relax` | the custom relaxation is called in every audited hidden-layer graph |
| Forward ReLU | Standard auto_LiRPA ReLU relaxation with the project's split-status wrapper | `width_cert.bounds.operators.BoundCustomReLU`, derived from `BoundRelu` | node inventory records the custom operator |
| Coordinate-wise IC | `max(l_IBP,l_CROWN)` and `min(u_IBP,u_CROWN)`; an empty floating-point intersection falls back to IBP | `width_cert.bounds.methods.intersect_bounds` | unit tests cover tighter coordinates and both numerical/material empty intersections |

The two displayed multiplication planes are valid McCormick planes: subtracting
the lower plane from `xy` gives `(x-x_l)(y-y_l) >= 0`, while subtracting `xy`
from the upper plane gives `(x-x_l)(y_u-y) >= 0` on the input box.

## Reproducible audit commands

From the repository root, with the experiment environment active:

```bash
python tools/audit_full_crown_execution.py \
  --output-dir artifacts/audit/a0_full_crown
pytest tests/test_full_crown_execution_path.py tests/test_bounds.py
```

The execution audit writes:

- `bound_graph_nodes.csv`: every traced bound node and perturbation flag;
- `execution_trace.json`: architecture-level call counts, output checks, and
  dependency metadata;
- `auto_lirpa_tree_diff.json`: path-free source-tree provenance comparison.

The existing dense-grid diagnostic remains separate. It samples exact gradients
on small networks and checks containment by IBP, the AGT-inspired hybrid,
Full-CROWN-box, Full-IC, and Hybrid-IC. Its manifest explicitly records that it
is not a soundness proof.

## Trusted computing base and claim boundary

The current implementation trusts PyTorch tensor operations, auto_LiRPA graph
tracing and backward bound propagation, the custom ReLU/Step operators, and the
project's intersection/update code. Computation uses ordinary floating-point
arithmetic. No directed outward rounding is implemented. Consequently:

1. the mathematical claim may be stated for exact real arithmetic once the
   actual substitution rules are proved;
2. the implementation may be described as path-audited and numerically tested;
3. the implementation must not be called a formally rounding-certified
   floating-point verifier.

## Required next actions

1. Preserve and publish the exact dependency delta, or rerun the final results
   against an immutable, publicly installable dependency tree. Until then,
   exact clean-room reproduction of the historical locks is incomplete.
2. Prove the actual sign-dependent backward substitution used by
   `BoundLinear`/`BoundMatMul` and `BoundMul`, then connect that induction to the
   custom Step box and the final concretization operation.
3. Run the small-network reference solver and property tests against the exact
   dependency artifact selected for release.
4. Keep the floating-point limitation explicit unless outward rounding or a
   justified error inflation scheme is implemented.
