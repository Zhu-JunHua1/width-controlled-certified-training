# Historical auto_LiRPA runtime patch

The frozen experiments used auto_LiRPA 0.7.0 derived from upstream commit
`9d100ec070868440b48d34e2f1dd21b97aab9172`. The exact runtime differs from that
commit in three Python files. `historical_runtime.patch` records those differences
without vendoring the complete upstream repository.

The changes serve three narrow purposes:

1. prevent `BoundMatMul` from entering the ordinary `BoundLinear` forward-bound
   fast path;
2. propagate bounds when both the activation and matrix operand are perturbed;
3. retain the historical `BoundStep` implementation that was present in the
   frozen dependency tree.

The current project execution path uses
`width_cert.bounds.operators.BoundStepActivation`, not the historical
`auto_LiRPA.bound_ops.BoundStep`. The latter remains in the patch solely so the
published dependency can reproduce the byte-level Python-tree hash recorded by
the frozen runs.

## Prepare the dependency

From the publication repository root:

```bash
git -c core.autocrlf=false clone https://github.com/Verified-Intelligence/auto_LiRPA.git ../auto_LiRPA
git -C ../auto_LiRPA checkout 9d100ec070868440b48d34e2f1dd21b97aab9172
python tools/prepare_autolirpa.py --source ../auto_LiRPA
pip install -e ../auto_LiRPA --no-deps
```

The preparation command is fail-closed: it checks the upstream Git revision and
clean state, applies the patch, and verifies the resulting Python-source-tree
SHA-256:

`6a8d960cdd013cbd63cc814d2e21e30ab2f3f23bada889eaf4a91f4be3994c73`.

For byte-level provenance, the preparation command also restores one inert
space-only line present in the historical runtime. Keeping that normalization
in the script avoids embedding intentional trailing whitespace in the patch.

Use `--check-only` to verify an already patched tree. This patch and the copied
upstream license are distributed under auto_LiRPA's BSD 3-Clause terms. The
project's own source remains under the license in the repository root.
