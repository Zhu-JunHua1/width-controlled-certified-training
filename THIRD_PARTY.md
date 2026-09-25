# Third-party dependencies

This repository does not vendor third-party source code.

## auto_LiRPA

- Upstream: <https://github.com/Verified-Intelligence/auto_LiRPA>
- Reference version: `0.7.0`
- Pinned upstream commit: `9d100ec070868440b48d34e2f1dd21b97aab9172`
- Expected Python-source-tree SHA-256: `2bc6390a98d9e88df4c8a2460c9e1cd2e4d7e7e7b058970f665bf7379febff2f`
- License: BSD 3-Clause (upstream)

Every run stores the installed version, the recognized commit when the source-tree hash matches the pinned
dependency, and a SHA-256 over the package's Python source tree in `config.json`. This records the exact
code without exposing an absolute installation path.

## Datasets

- MNIST is downloaded through `torchvision`; no MNIST files are redistributed.
- TwoMoons and synthetic multi-class data are generated through scikit-learn.
