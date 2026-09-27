# Final-study design draft (not yet a study lock)

This document records the decisions made before any protocol-v2 final test is
opened. It is intentionally marked as a draft: it cannot authorize test
access, and unresolved protocol hashes remain to be filled only after the
representative calibration runs pass.

## Scope and venue

- Target venue: *International Journal of Approximate Reasoning*.
- Scope: small, full-batch, fully connected certified-training problems.
- The paper will not claim general scalability to convolutional networks,
  full multi-class MNIST, or large data sets.
- The primary theoretical contribution is finite-horizon interval-width
  control. Certification-accuracy persistence remains conditional/empirical.

## Data roles

- All method development in the current branch uses exposed development data;
  its result fields are calibration diagnostics and all final-test fields are
  null.
- Final runs use train, calibration, and test only after the resolved
  configuration and split hashes are frozen.
- The legacy generated-data test seed 42 is retired.
- Final MNIST protocols must use a committed explicit index manifest. The
  manifest must exclude all MNIST 1/7 examples used by the legacy development
  calibration and test subsets and must be content-hashed in the study lock.
- The final test is opened once. A design change after opening retires that
  split and requires a new lock and an unused split.

## Endpoints and common evaluation

Primary endpoint:

1. Test certified accuracy at the calibration-selected checkpoint, where the
   checkpoint is the earliest epoch attaining maximum calibration certified
   accuracy under the common IC calibration certifier.

Secondary endpoint:

1. Test certified accuracy at the predeclared horizon `T`.

Safety and cost endpoints:

- clean accuracy at the selected checkpoint and at `T`;
- complete wall-clock time, including the fixed pre-run;
- bound-evaluation count and peak device memory;
- numerical-intersection fallback count and maximum gap.

Every frozen checkpoint is evaluated on exactly the same test samples with
both IBP and IC certifiers. Training-method effects are compared within one
common test-certifier column. A comparison that changes both training and
test certification is labelled only as an end-to-end pipeline comparison.

## Primary hypotheses

The primary family contains four directional, paired hypotheses across ten
predeclared accepted starts:

- **H1-TwoMoons (training-bound effect):** at `epsilon=0.01`, Full-IC fixed
  training has higher primary-endpoint test certified accuracy than IBP fixed
  training when both are evaluated by the same IBP test certifier.
- **H1-MNIST (training-bound effect):** at `epsilon=0.001`, the same comparison
  is made on MNIST 1/7.
- **H2-TwoMoons (width-control effect):** at `epsilon=0.01`, IC+IWAU has higher
  secondary-endpoint test cert@T than IC-fixed when both use the same IC test
  certifier.
- **H2-MNIST (width-control effect):** at `epsilon=0.001`, the same comparison
  is made on MNIST 1/7.

The exact one-sided paired sign test is used for these four predeclared
directions, with Holm correction across the family. The report also includes
the paired mean difference, a 95% paired bootstrap confidence interval, and
the number of positive/zero/negative paired differences. Statistical claims
are not made from individual gradient coordinates, which are dependent.

All other radii, depths, widths, class counts, alternate certifiers, and
schedule comparisons are secondary or exploratory and are labelled as such.

## Bound-method mechanism study

The controlled methods are IBP, AGT-inspired forward-CROWN/backward-IBP,
Full-CROWN-box, Full-IC, and (for same-box analysis) Hybrid-IC. “AGT-inspired”
is retained unless a formula- and implementation-level equivalence audit is
completed.

Same-box audits are preregistered at the initial, calibration-selected, and
horizon checkpoints. They report widths, endpoint-selection fractions,
runtime, memory, numerical fallbacks, and sampled inclusion checks. Sampling
is an implementation diagnostic, not a proof of soundness. A performance
claim for full backward CROWN requires both:

1. tighter same-box bounds than the hybrid on the frozen multi-seed study; and
2. better checkpoint performance under a common test certifier.

If either condition fails, the paper reports the construction as sound and
operational but does not claim a practically meaningful backward-CROWN gain.

## Update-rule fairness decision

The final study uses the deployable, prospectively frozen comparison rather
than an oracle per-seed match:

- all two-stage arms reuse the exact same fixed prefix through calibration
  `t_p` and have the same continuation length;
- inverse-time, cosine, and step-decay are analytically set to the frozen
  development-population target final mean step
  `0.055974505841732025`;
- global-width and coordinate-wise IWAU use the same calibrated coefficient
  and functional form, differing only in scalar mean width versus
  coordinate-wise width;
- complete cost includes the fixed pre-run and continuation for every arm;
- realized final/cumulative step statistics are reported rather than described
  as exactly matched per seed.

The population target is the median from development seeds 0, 4, and 5. It is
not recomputed from final calibration or test trajectories. This avoids an
oracle comparator that knows the future IWAU trajectory of each final seed.

The primary IWAU comparison remains IC+IWAU versus IC-fixed. Coordinate IWAU
versus global-width is a predeclared secondary mechanism comparison. The three
ordinary time schedules are exploratory controls for the explanation “any
late step-size reduction is sufficient.”

## Theory restrictions carried into the manuscript

- Fixed-step IC/IBP nesting requires the same initial box, data order,
  method-independent predeclared step sequence, inclusion-isotone update, and
  common inclusion-monotone certifier.
- The nesting theorem is not applied to independently calibrated IWAU paths.
- `lambda=0` (`not_required`) is an identity branch and is excluded from every
  expression that divides by `lambda`.
- Calibration peak retention is a trajectory-construction property, not an
  independent-test generalization guarantee.
- Real-arithmetic soundness and floating-point implementation validation are
  stated separately. Empty numerical intersections fall back to IBP and are
  logged; endpoints are never swapped or silently clipped.

## Remaining freeze prerequisites

1. Generate and review explicit final MNIST index manifests that exclude all
   exposed development indices.
2. Resolve final generated-data seeds and write the split-hash manifest.
3. Produce final P1/P2 representative configurations with test access still
   disabled and pass calibration-only smoke runs.
4. Freeze the full protocol list and resolved configuration hashes.
5. Record the code commit, dependency hash, auto_LiRPA source hash, hypotheses,
   and protocol hashes in a `status: frozen` study lock.
6. Only then change final configurations to `test_access: frozen_final` and run
   the test evaluation once.
