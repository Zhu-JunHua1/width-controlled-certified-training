# Protocol-v2 follow-up execution plan

This plan governs the secondary and mechanism analyses after the frozen P1/P2
confirmatory study. It does not modify the committed study lock, primary
endpoints, split manifests, accepted-start manifests, or primary hypotheses.
Legacy protocol-v1 results are not substituted for any item below.

## Stage 1: frozen secondary evaluations

Run one GPU experiment at a time from the detached locked worktree. Preserve
raw divergence and all lock/config/code hashes.

1. **P6 MNIST 1/7 depth scaling**
   - depths: 1, 2, 3;
   - radii: `1e-4`, `1e-3`;
   - 10 shared starts, methods `IBP`, `IC`, `IC+IWAU`;
   - both IBP and IC frozen-test certifiers;
   - expected: 180 per-run summaries and 360 result rows.
2. **P7 TwoMoons training-length stability**
   - horizons: 20, 40, 60, 80;
   - 10 shared starts, methods `IC`, `IC+IWAU`;
   - both test certifiers;
   - expected: 80 summaries and 160 rows.
3. **P3 TwoMoons sensitivity**
   - 13 locked depth/width/base-step variants;
   - 10 shared starts per comparison group, three methods, two certifiers;
   - expected: 390 summaries and 780 rows.
4. **P5 synthetic multi-class extension**
   - class counts 3 and 4, four radii each;
   - 10 starts, three methods, two certifiers;
   - expected across both class counts: 240 summaries and 480 rows.
5. **P4 external TwoMoons context**
   - three radii, 10 starts, methods `AGT-Hybrid`, `IC`, `IC+IWAU`;
   - clean test-input domain and both parameter certifiers;
   - expected: 90 summaries and 180 rows.

Every stage must pass its count/provenance/nullability checks before the next
one starts. No configuration may be changed in response to an observed result.

## Stage 2: same-parameter-box bound audit

The audit is a post-primary mechanism analysis and does not alter a reported
checkpoint. To avoid representative-seed selection, audit the
calibration-selected fixed-IC checkpoint for **all 10 frozen starts** in P1 and
P2. On each identical parameter box compute:

- complete IBP;
- AGT-inspired forward-CROWN/backward-IBP;
- complete CROWN;
- Hybrid-IC = IBP intersected with the hybrid bound;
- Full-IC = IBP intersected with complete CROWN.

Report coordinate validity, mean/median/quantiles of width, relative
tightening against IBP, the fraction of lower/upper endpoints selected from
each source, elapsed time, peak memory, sampled inclusion violations, and
floating-point intersection fallbacks. Sampling is an implementation
diagnostic, not a soundness proof.

## Stage 3: implementation validation

Run the deterministic small-network grid check at the locked implementation
and retain its CSV/JSON report. The grid must exercise negative-stable,
positive-stable, and zero-containing ReLU intervals; mixed signs in bilinear
backward products; both derivative endpoint choices at the ReLU kink; and the
conservative IBP fallback for a numerically empty intersection.

## Stage 4: manuscript synchronization

Only verified protocol-v2 aggregates may replace the disabled legacy-v6
secondary-result scaffold in the manuscript working copy. For every table or
figure, retain a reverse mapping to its source result files and hashes. Update
the abstract/highlights only after all secondary results are fixed; the four
P1/P2 primary hypotheses remain the confirmatory headline regardless of the
secondary outcomes.
