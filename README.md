# Width-Controlled Certified Training

[![tests](https://github.com/Zhu-JunHua1/width-controlled-certified-training/actions/workflows/tests.yml/badge.svg)](https://github.com/Zhu-JunHua1/width-controlled-certified-training/actions/workflows/tests.yml)

Reference implementation for *Provable Width Control for Certified Training against Data Perturbations*.

The project studies certified training when all training samples may be perturbed. It contributes:

- **IC:** coordinate-wise intersection of IBP and full-gradient-graph CROWN bounds;
- **IWAU:** coordinate-wise learning rates controlled by gradient-interval width;
- frozen train/calibration/test protocols, provenance checks, and reproducible result reports.

The implementation supports TwoMoons, MNIST digit pairs, and synthetic multi-class tasks. The reported
experiments are small-scale, full-batch MLP studies; this repository does not claim large-model scalability.

## Evidence in the paper

| Evidence | Configuration family | Role |
|---|---|---|
| TwoMoons, three radii, IBP/IC with fixed/IWAU updates | `protocol_v5_frozen` | Main prespecified blind descriptive study |
| Tuned scalar, cumulative cosine, GlobalIWAU, coordinate IWAU | `protocol_v4_frozen` | Blinded update-rule fairness study |
| Depth, width, step, horizon, and MNIST studies | `protocol_v2_frozen` | Secondary robustness analyses |
| Corrected synthetic multi-class study | `protocol_v3_frozen` | Post-hoc exploratory extension |

Final aggregates are published in [`artifacts/protocol_v5_final`](artifacts/protocol_v5_final) and
[`artifacts/protocol_v4_final`](artifacts/protocol_v4_final). The complete evidence map and reporting
scope are in [`docs/protocols.md`](docs/protocols.md).

## Installation

The reference environment uses Python 3.10 and PyTorch 2.8. Install the appropriate CPU or CUDA PyTorch
build first. The frozen experiments require a small published patch to auto_LiRPA commit
`9d100ec070868440b48d34e2f1dd21b97aab9172`:

```bash
python -m venv .venv
# Activate .venv for your platform.

# Install a CPU or CUDA build suitable for your machine first.
pip install torch==2.8.0 torchvision==0.23.0
pip install -r requirements.txt

git -c core.autocrlf=false clone https://github.com/Verified-Intelligence/auto_LiRPA.git ../auto_LiRPA
git -C ../auto_LiRPA checkout 9d100ec070868440b48d34e2f1dd21b97aab9172
python tools/prepare_autolirpa.py --source ../auto_LiRPA
pip install -e ../auto_LiRPA --no-deps
pip install -e .
```

The preparation command checks the upstream revision and tree, applies the three-file patch, and verifies
the resulting source hash. The patch, license, and exact hash are documented in
[`third_party/auto_lirpa`](third_party/auto_lirpa). auto_LiRPA remains an external dependency rather than
a vendored fork.

## Quick start

Run the editable single-seed example:

```bash
width-cert run
```

Run or validate a specific configuration:

```bash
width-cert validate-config configs/paper/P1_twomoons_ablation.yaml
width-cert run configs/paper/P1_twomoons_ablation.yaml
```

`width-cert run` performs shared-start selection before certified training. The YAML `methods` list is
authoritative: required fixed pre-runs for adaptive methods are internal dependencies and are not reported
as extra experiment arms.

Frozen-final configurations are immutable provenance records. They are provided for inspection and
clean-room reproduction, not for retuning or repeated blind evaluation. Use a new output directory and
preserve their files and hashes.

## Results and analysis

Each run records resolved configuration and provenance, state/update trajectories, status, selected and
terminal checkpoints, runtime, and memory. Raw diverged endpoints remain null; treating them as zero is
an explicit reporting policy rather than a rewrite of raw results. The full schema is described in
[`docs/log_schema.md`](docs/log_schema.md).

Common commands:

```bash
# Aggregate one experiment.
width-cert summarize results/P1/results.csv --diverged-as-zero --output results/P1/summary.csv

# Export portable aggregate and runtime reports.
width-cert export-report results public-report --diverged-as-zero
width-cert runtime-report results public-runtime-report

# Same-box mechanism audit.
width-cert audit-bounds configs/development/gradient_bound_ablation.yaml \
  path/to/last_state.pt diagnostics/same-box --training-seed 1 --input-radius 0.01

# Preregistered paired statistics.
width-cert paired-report results/final study_locks/primary_hypotheses.yaml \
  results/final/primary_paired_statistics.csv
```

Test loading is fail-closed. Development configurations use `evaluation.test_access: disabled`; final
test evaluation requires a complete committed study lock. See
[`docs/experimental_locking.md`](docs/experimental_locking.md).

## Scientific scope

- **IC soundness:** the mathematical claim is for the stated real-arithmetic relaxations. The floating-point
  implementation has numerical audits but does not implement outward-rounding formal certification.
- **Finite tests:** grid, random, and reference-solver checks validate implementation behavior; they do not
  replace the operator-wise proof.
- **IBP-fixed:** this is the repository's FullCert-style local arm, not the official FullCert software.
  The attempted official adapter and its stopping condition are documented in
  [`docs/fullcert_official_audit.md`](docs/fullcert_official_audit.md).
- **AGT-inspired hybrid:** the forward-CROWN/backward-IBP arm is a controlled local ablation, not a claim
  of bitwise equivalence to the AGT software.

The implementation/theory correspondence and trusted computing base are documented in
[`docs/full_crown_execution_audit.md`](docs/full_crown_execution_audit.md) and
[`docs/theory_revision_v2.md`](docs/theory_revision_v2.md).

## Repository map

| Path | Contents |
|---|---|
| `src/width_cert/` | Training, bounds, certification, models, data, logging, and analysis |
| `configs/examples/` | Editable examples |
| `configs/paper/` | Reusable paper configurations |
| `configs/protocol_v*_frozen/` | Immutable frozen study inputs |
| `study_locks/` | Split, seed, endpoint, and provenance locks |
| `artifacts/` | Portable aggregate tables and manifests |
| `docs/` | Protocol, theory, audit, and reproducibility details |
| `tests/` | CPU regression and fail-closed protocol tests |

## Tests

```bash
python tools/prepare_autolirpa.py --source ../auto_LiRPA --check-only
pytest
```

The test suite covers indexing, calibration windows, adaptive continuation, conservative intersection
fallback, split isolation, status semantics, logging, and frozen-protocol validation.

## Citation and license

Use [`CITATION.cff`](CITATION.cff) or GitHub's **Cite this repository** action. An archival DOI will be
added after the public release is frozen.

This repository is distributed under the BSD 3-Clause License. Third-party components retain their own
licenses.
