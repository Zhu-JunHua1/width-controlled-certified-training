# Paper protocols and evidence hierarchy

This document maps the revised manuscript to the executable configurations and derived artifacts in
this repository. It supersedes the earlier pre-revision protocol overview.

## Evidence hierarchy

| Tier | Protocol family | Purpose | Status |
|---|---|---|---|
| Main descriptive study | P1-v5 TwoMoons multi-radius | Compare IBP-fixed, IBP+IWAU, IC-fixed, and IC+IWAU across `1e-4`, `1e-3`, and `1e-2` | Frozen blind run complete |
| Update-rule controls | P1/P2-v4 | Compare coordinate IWAU with fixed IC, GlobalIWAU, a calibration-tuned scalar schedule, and a cumulative-step-matched cosine schedule | Frozen blind runs complete |
| Earlier component tests | P1/P2-v2 | Paired IC-vs-IBP selected-checkpoint test and IC+IWAU-vs-IC terminal test | Frozen, reported in appendix |
| Secondary robustness | P3/P6/P7-v2 | Depth, width, base-step, radius, and horizon sensitivity | Frozen, complete |
| Multi-class extension | P5-v3 | Corrected shared-generator split for synthetic 3/4-class tasks | Post-hoc exploratory |
| External context | P4-v2 | Clean-input comparison with local AGT-inspired and IC systems | Exploratory; not numerically pooled with the main study |
| Official FullCert extension | protocol-v5 FullCert | Matched official-source comparison through a constrained adapter | Attempted once; incomplete, no aggregate reported |

## Main P1-v5 protocol

- Configuration: `configs/protocol_v5_frozen/P1_twomoons_multiradius_main.yaml`
- Study lock: `study_locks/protocol_v5_main_frozen.yaml`
- Accepted starts: `study_locks/candidate_audits/protocol_v5_P1_main_seeds.json`
- Architecture: `[2,20,20,1]`
- Data roles: 100 training, 500 calibration, and 500 frozen-test examples
- Radii: `1e-4`, `1e-3`, and `1e-2`
- Horizon: `T=40`
- Methods: `IBP`, `IBP+IWAU`, `IC`, and `IC+IWAU`
- Checkpoint rule: first maximum of IC-certified calibration accuracy
- Test certifiers: IC primary, IBP secondary
- Expected grid: 120 training summaries and 240 evaluator rows

The run completed with the exact frozen grid. The manuscript's main table is generated from
`artifacts/protocol_v5_final/main_table.csv`; the complete evaluator matrix, per-start rows, clean
accuracy, runtime, seconds per update, and memory are retained under
`artifacts/protocol_v5_final/appendix` and `artifacts/protocol_v5_final/runtime`.

The terminal aggregate uses the prespecified `diverged_as_zero` policy only for a missing endpoint whose
raw status is `diverged`. Raw terminal fields remain null, so a diverged trajectory is never rewritten as
a completed zero-certification run.

## Update-rule controls

The protocol-v4 family evaluates five IC-trained systems on disjoint blind data:

- fixed IC;
- coordinate-wise IWAU;
- GlobalIWAU;
- a calibration-tuned scalar inverse-time schedule; and
- a cumulative-step-matched cosine schedule.

The scalar search used development/calibration data only. Final choices were frozen before test access.
The six paired comparisons and Holm-adjusted sign tests are in
`artifacts/protocol_v4_final/paired_statistics.csv`; cost summaries are in
`artifacts/protocol_v4_final/runtime`.

## Secondary and exploratory protocols

P3, P6, and P7 retain the protocol-v2 frozen code, split, seed, evaluator, and failure policies. P3
varies TwoMoons depth, width, and base step; P6 varies MNIST 1/7 depth at two radii; P7 varies
`T in {20,40,60,80}` for a two-hidden-layer TwoMoons model. These analyses are descriptive and are not
added to the multiplicity family of the earlier P1/P2 hypotheses.

P5-v3 corrects the synthetic multi-class generator so that train, calibration, and test data share one
generator distribution while remaining disjoint. Because the correction followed inspection of the
legacy P5-v2 result, P5-v3 is labelled post-hoc exploratory evidence.

P4 uses clean test inputs, `T=20`, a different base step, and seed-specific data splits to align with an
external reporting context. It is not directly comparable with P1-v5.

## Baseline labels

- `IBP-fixed` is this repository's FullCert-style interval-SGD implementation with auto_LiRPA bounds.
  It must not be called an official FullCert reproduction.
- `Hybrid-fixed` is the local AGT-inspired forward-CROWN/backward-IBP arm. It is not a bitwise AGT
  reproduction.
- `IC+IWAU` is the complete proposed system.

The official FullCert source was audited separately. The constrained adapter passed conversion,
one-step parity, split-isolation, trajectory-export, and common-evaluator development gates. Its one-time
frozen extension stopped before the declared grid completed because an interval update became non-finite.
Partial cells are excluded from means and paired comparisons; see `docs/fullcert_official_audit.md`.

## Reproduction rules

Frozen files and raw result trees are immutable. Do not modify a lock, configuration, seed manifest, or
analysis hypothesis after seeing blind results, and do not rerun a completed blind protocol to select a
more favorable outcome. Clean-room reproductions should use a new output root while preserving the
published inputs. Paper tables and figures should be regenerated from the structured CSV/JSON artifacts,
not transcribed from rounded PDF values.
