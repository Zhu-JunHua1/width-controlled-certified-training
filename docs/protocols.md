# Paper protocols and result mapping

This file describes the pre-revision task grid. Final protocol-v2 values will be frozen in a versioned
study lock before independent test evaluation. All protocols use full-batch interval SGD with 100 training samples and 10 reported seeds. The reported
seeds are not manually enumerated: each YAML file defines a deterministic candidate range and retains the
first 10 starts satisfying its pretraining criterion. Selection precedes all certified methods and radii.
For from-scratch P4/P5, every candidate is accepted. P1--P3 and P5--P7 use end-to-end certification. P4
uses clean test inputs to align its evaluation domain with MIBP-Cert Table 2; its epsilon applies to the
training-data perturbation only.

The executable YAML files are in `configs/paper/`. P3 is a 13-variant sweep; P5 and P6 use one file per
class count or depth so that each run directory remains self-describing.

| ID | Task | Architecture | Start | Epsilon | T | Main evidence |
|---|---|---|---|---|---:|---|
| P1 | TwoMoons | `[2,20,20,1]` | pretrain about 50% | `1e-3,1e-2` | 40 | 2x2 ablation and trajectories |
| P2 | MNIST 1/7 | `[784,10,1]` | pretrain about 50% | `1e-3,2e-3` | 30 | 2x2 ablation |
| P3 | TwoMoons | depth/width/base-step axes | pretrain about 50% | `1e-3` | 40 | robustness across hyperparameters |
| P4 | TwoMoons | `[2,20,1]` | from scratch | `1e-4,1e-3,1e-2` | 20 | external baseline table |
| P5 | synthetic 3/4-class | `[6,20,K]` | from scratch | `0.005,0.01,0.02,0.05` | 30 | multi-class extension |
| P6 | MNIST 1/7 | `[784,10,...,1]`, L=1/2/3 | pretrain about 50% | `1e-4,1e-3` | 40 | real-image depth scaling |
| P7 | TwoMoons | `[2,20,20,1]` | pretrain about 50% | `1e-3` | `20,40,60,80` | IWAU stability across training length |

Candidate ranges are `[0,25)` for P1, P6 L=2/3, and P7, `[0,100)` for P3, `[0,15)` for P2 and P6 L=1,
and `[0,10)` for from-scratch P4/P5. All pretrained protocols use the 50% shared-start target. P3 additionally requires a
minimum 0.2-point calibration-accuracy gain (one example on its 500-sample calibration set) and rejects prepared starts
above 52%. These rules prevent unchanged chance-level starts and large first-crossing overshoots. TwoMoons pretraining uses
SGD, whereas MNIST pretraining uses Adam, both explicitly
encoded in YAML. The exact accepted and rejected candidates are written to `seed_selection.json`.

Every executable YAML now specifies train, calibration, and test sizes and independent split seeds. Seed
acceptance, pretraining stopping, IWAU calibration, and checkpoint selection use calibration data only.
Test data are evaluated only after those decisions are frozen.

## Important reporting rules

- **Primary endpoint:** `test cert@calibration-selected checkpoint` under paired seeds.
- **Stability:** the secondary endpoint `test cert@T`; calibration trajectories explain the mechanism.
- **Training/evaluator separation:** every frozen checkpoint is evaluated by both IBP and IC.
- External rows copied from MIBP-Cert are not treated as directly comparable to local reproductions.
- P4 uses clean evaluation, base step 2.0, and reports the full cost of the same-method pre-run plus
  IWAU continuation.
- P4 generates 300 TwoMoons samples per seed and then applies the fixed train/test split used by the
  aligned baseline harness (`strategy: combined_split`).
- Binary certified training optimizes BCE with logits. Synthetic multi-class certified training uses
  one-vs-all BCE on the explicit gradient graph; its from-scratch point initialization requires no
  pretraining loss. When pretraining a multi-class model, the runner uses cross-entropy and records both
  loss definitions in `config.json`.

## Result generation

Raw runs are intentionally excluded from source control. The public `width-cert export-report` command
consumes the structured result directories produced by these YAML files and writes an aggregate CSV plus a
machine-readable provenance manifest. Paper tables and figures should consume those structured outputs
rather than values copied from rounded PDF tables.

The old peak-based figures and tables are development evidence only and are not final protocol-v2 test
results. They must not be copied into a revised manuscript without an explicit legacy label.
