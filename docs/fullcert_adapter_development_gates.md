# FullCert adapter development gates

Audit date: 2026-10-11  
Analysis tier: development only  
Blind-test metrics read: no  
Final official comparison: not run

## Frozen external source

- Official repository: `https://github.com/t-lorenz/FullCert`
- Commit: `3387a8af3ab143fae238b5d3a3f5b24fa815cca6`
- Git tree: `23067f06288ab25cdfd748bc160534e6a54b9a34`
- Working tree during all gates: clean
- License: MIT, SHA-256
  `00392331b0c8f64dae1be24673d6643252a577fc557369c904cf0bc2dbaa866a`
- Published `requirements.txt` SHA-256:
  `6b41b94711ce44896bb83deb25b5abf4772c0da96ed4a5ab922e7b50017805cd`
- Published `environment.yml` SHA-256:
  `88a948d4479887d87cc0cf6ce56d95724414a4718fdc9cab670215baa7be0336`

The external checkout was not modified. The adapter imports its `boundflow`
operators after verifying the remote, commit, clean state, and license.

## Gate results

| Gate | Result | Evidence |
|---|---|---|
| Tensor/layout round trip | Pass | Exact endpoint equality after `[out,in] -> [in,out] -> [out,in]` conversion, including a three-linear-layer box. |
| Initialization parity | Pass | Point-valued public states remain point-valued after conversion. |
| One-step update parity | Pass | One adapter step equals an unmodified call to official `train_model` on the same tensors and state. |
| Full-batch normalization | Pass with mandatory mapping | The official BCE derivative and linear layer both average. Passing `N*eta` to the API reproduces the mathematical mean-gradient step `eta` to `1e-7`. |
| Paper architecture | Pass | The adapter instantiated `[2,20,20,1]` from official `LinearLayer`, `ReLU`, and `Model` objects. |
| Fixed 40-update trajectory | Pass | A development run produced `B_0,...,B_40`, exactly 41 endpoint boxes, with early stopping disabled. |
| Common evaluator | Pass | The public evaluator consumed the initial, calibration-selected, and terminal boxes under both IBP and IC. |
| Split isolation | Pass | The training API has no test-tensor argument; the smoke report records independent train/calibration hashes and `test_metrics_read=false`. |
| Official-source immutability | Pass | All forward, backward, loss, interval, and update operations are called from the audited checkout without patches. |

The executable adapter and smoke driver hashes at the gate run were:

- `src/width_cert/baselines/fullcert.py`:
  `12675380357b6915f2e59b985953e81a5923e5019902b5cec109911ef3b50e90`
- `tools/run_fullcert_adapter_smoke.py`:
  `041dd121b1ba4699dee57c7e53c0a40729b5e654b675b811f29a36aca7a62e88`
- ignored development smoke report:
  `ccb56a8732498a54e9230d4a546764e7cca6bdb2bbe60dbb1085881c57f4e95d`

## Development smoke result

The smoke run used one accepted development start, radius `1e-4`, 100 training
examples, 500 calibration examples, architecture `[2,20,20,1]`, and `T=40`.
It used paper mean-gradient step `0.1`, represented by the audited FullCert API
argument `10.0` because `N=100`. IC calibration certification first reached
its maximum of 75.6% at epoch 25. The common evaluator returned:

| Box | IBP calibration cert. | IC calibration cert. |
|---|---:|---:|
| Initial | 50.0% | 50.0% |
| First calibration maximum | 75.2% | 75.6% |
| Terminal | 58.4% | 60.4% |

These are development diagnostics, not paper results.

## Environment and remaining constraints

The gates ran in the common publication environment: Python 3.10.20, PyTorch
2.8.0+CUDA 12.8, auto_LiRPA 0.7.0, NumPy 1.26.4, and scikit-learn 1.7.2.
This differs from FullCert's published PyTorch 1.12 environment, but is needed
for the same public evaluator. The official source behavior is protected by
one-step parity rather than claimed to be a reproduction of its historical
runtime environment.

The learning-rate mapping is now part of the matched protocol, not an optional
tuning choice. A comparison using raw FullCert API learning rate `0.1` would
have a 100-times smaller mean-gradient update and must be labelled
external-context. Runtime reporting must also separate official BoundFlow
training from post-hoc common-evaluator checkpoint selection.

## Gate decision

The constrained adapter gates pass. An official matched comparison is
technically feasible, but no final run is authorized by this document. Before
opening any blind test, the comparison still requires a separate immutable
configuration and study lock binding the accepted starts, radii, normalization
mapping, external commit and tree, adapter commit, split seeds, evaluator,
endpoints, failure policy, and row-count validator.
