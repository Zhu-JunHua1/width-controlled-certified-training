# Frozen data-split manifests

The JSON files in this directory contain indices only. They contain no images,
labels, model outputs, or reported metrics.

- `retired_mnist17_development.json` reconstructs the MNIST 1/7 subset used by
  the legacy/development protocol (`train_seed=10123310`, `test_seed=42`). It
  is committed solely to make prior exposure auditable.
- `final_mnist17_protocol_v2.json` is the candidate final split. Its train,
  calibration, and test indices have zero overlap with the corresponding
  exposed development index spaces. It must still be referenced by a frozen
  study lock before final-test access is permitted.

The final seeds were derived before evaluating any final example. We took the
first three big-endian 32-bit words of

```text
SHA256("width-controlled-certified-training-protocol-v2-final-split")
= 035490db0f8ef3105fdfbd842351bfb69db55a0f3d19f551964ea11239567977
```

which gives train seed `55873755`, calibration seed `261026576`, and test seed
`1608498564`. The final manifest records the SHA-256 of the retired manifest it
excludes. The complete final manifest SHA-256 is
`383a3c66ab2d565a028c9e7a64b1fc1592ca70cc91487bf70e2490fd9ee10c0c`.

To reconstruct these files after installing the package and downloading the
official MNIST files:

```bash
width-cert make-mnist-split data_splits/retired_mnist17_development.json \
  --digits 1 7 --train-size 100 --calibration-size 500 --test-size 500 \
  --train-seed 10123310 --calibration-seed 314159 --test-seed 42 \
  --legacy-consecutive-calibration

width-cert make-mnist-split data_splits/final_mnist17_protocol_v2.json \
  --digits 1 7 --train-size 100 --calibration-size 500 --test-size 500 \
  --train-seed 55873755 --calibration-seed 261026576 --test-seed 1608498564 \
  --exclude-manifest data_splits/retired_mnist17_development.json
```

The loader validates schema version, dataset identity, exact split sizes,
integer types, uniqueness, index ranges, and train/calibration disjointness.
The frozen study lock additionally binds the final manifest by content hash.
