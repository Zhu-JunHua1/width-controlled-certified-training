# Official FullCert source audit

Audit date: 2026-10-10  
Audit status: source path verified; matched comparison is conditionally feasible through a thin adapter  
Final comparison status: not run

## Source identity

- Repository: `https://github.com/t-lorenz/FullCert`
- Audit checkout: a separate, read-only clone of the repository above
- Audited commit: `3387a8af3ab143fae238b5d3a3f5b24fa815cca6`
- Commit date and subject: 2025-02-03, `Add acknowledgments and project logos`
- Working tree during the audit: clean
- License: MIT, copyright Tobias Lorenz (2024)
- Published environment: Python 3.10.5, PyTorch 1.12.0, torchvision 0.13.0,
  NumPy 1.23.1, scikit-learn 1.1.1, and the remaining versions recorded in
  `environment.yml` and `requirements.txt` of the audited checkout.

This identity must be retained in every result and manuscript reference to the official baseline.
The local method called `IBP-fixed` remains a FullCert-style reimplementation using auto_LiRPA;
it must not be relabelled as the official FullCert code.

## Actual execution path

The public entry point is `code/certified_training.py`. Its certified-training path is:

1. `datasets.datasets.load_dataset` constructs the pre-training, training, validation, and test sets.
2. `boundflow.perturbation.perturb_entire_dataset` converts every selected training input to an
   interval. With `--perturb-inference`, it also converts validation and test inputs to intervals.
3. `boundflow.network.Model` is built from `LinearLayer` and `ReLU` objects.
4. `boundflow.model_functions.train_model` performs forward interval propagation, evaluates the
   interval loss derivative, calls `Model.backwards`, and applies `Model.update_weights`.
5. `LinearLayer.update_weights` applies the interval SGD update directly to interval gradients.
6. `boundflow.model_functions.evaluate_model` computes certified correctness from the resulting
   interval-valued model and input intervals.

The bound implementation is the repository's own `boundflow` interval arithmetic, not auto_LiRPA.
The all-training-sample threat model is directly supported by `perturb_entire_dataset`; full-batch
training is obtained by setting the batch size equal to the training-set size.

## Parameter-box export

Parameter boxes are representable and exportable. After an interval update, every trainable
`LinearLayer` stores interval-valued `weights` and `biases`. `Model.state_dict()` serializes those
objects, and `Model.clone()` preserves their lower and upper endpoints. The public script saves only
the selected model, but a read-only adapter can record `model.state_dict()` after every epoch without
changing the forward, backward, or update operators.

The adapter must export explicit lower and upper tensors and record the epoch before invoking the
common evaluator. It must not reconstruct a box from its centre or radius after the fact.

## Match to the paper's TwoMoons protocol

| Protocol element | Official source capability | Audit decision |
|---|---|---|
| All 100 training inputs perturbed | Supported by `perturb_entire_dataset` | Exact |
| Perturbed calibration and test inputs | Supported by the same function | Exact |
| BCE objective | `BinaryCrossEntropyLoss` is provided | Exact |
| Full-batch interval SGD | Supported when batch size equals 100 | Exact |
| Constant scalar step size | Supported | Exact for the official baseline |
| Network `[2,20,20,1]` | `Model` accepts arbitrary layer lists, although the CLI builds one hidden layer | Exact only through an adapter that instantiates two hidden layers |
| Frozen train/calibration/test tensors | Core training accepts arbitrary data loaders, while the CLI regenerates TwoMoons internally | Exact only through an adapter that supplies the frozen tensors |
| Ten shared accepted starts | Initial tensors can be loaded into `LinearLayer`; transposition is required because FullCert stores linear weights as `[in,out]` | Conditional on a round-trip conversion test |
| `T=40` with no early termination | Core loop supports 40 epochs, but the CLI enables a 5-point validation-drop early stop | Exact only through an adapter that disables early stopping |
| First-maximum IC calibration checkpoint | Not implemented by official FullCert | The adapter records every box; selection is performed later by the common evaluator |
| Common IBP/IC test evaluator | Not implemented by official FullCert | Conditional on verified box conversion into the public evaluator |
| Runtime and peak memory | Runtime can be measured externally; peak memory needs adapter instrumentation | Feasible |

### Full-batch learning-rate semantics

The development adapter audit found an additional normalization detail that is
not visible from the command-line interface. `BinaryCrossEntropyLoss.backwards`
divides the logit derivative by the batch size, and `LinearLayer.backwards`
divides each parameter gradient by the batch size again. Consequently, the
raw FullCert API update is `1/N` of the usual mean-BCE gradient update for a
full batch of size `N`. A point-valued one-step parity test confirms that an
API learning-rate argument of `N * eta` reproduces the mathematical SGD step
of size `eta` without changing any official forward, backward, or update
operator. The matched adapter must therefore record both values explicitly;
for the paper protocol, `N=100`, `eta=0.1`, and the official API argument
is 10.0. A result using the raw argument 0.1 would instead be external-context
evidence because its effective update budget would differ by a factor of 100.

The CLI flags `--train-size` and `--test-size` are not passed through for the TwoMoons loader, whose
defaults are 1000 training and 200 validation/test examples. Therefore the official CLI cannot be
used as-is for the matched experiment. This is a protocol-interface limitation, not a limitation of
the BoundFlow training algorithm.

## Adapter boundary

A matched adapter may perform only the following operations:

- load the already frozen TwoMoons tensors and accepted initial parameters;
- instantiate the audited official `Model`, `LinearLayer`, `ReLU`, and BCE objective;
- transpose linear weight tensors at the input/output boundary and verify an exact round trip;
- call the official forward, backward, and interval-update operators unchanged;
- disable the CLI's early-stopping wrapper while retaining the same fixed 40-update algorithm;
- serialize every epoch's exact interval endpoints;
- measure full-pipeline time and peak memory; and
- pass exported boxes to the independently tested common evaluator.

It may not replace BoundFlow interval arithmetic, change the loss, change update equations, omit
diverged runs, or select checkpoints using test information. Any scientific change moves the result
to an external-context comparison.

## Required development gates

Before freezing a final comparison, all of the following must pass:

1. Tensor conversion round trip for every weight and bias endpoint.
2. One-step parity between the adapter and an unmodified official-code call on the same tiny batch.
3. Forty-epoch smoke run on development-only tensors with one state per epoch.
4. Common-evaluator acceptance of the initial, selected, and terminal boxes.
5. Split-isolation assertion proving that no blind-test tensor is used in training or checkpoint
   selection.
6. Provenance capture for the FullCert commit, adapter commit, environment, frozen configuration,
   split manifest, and result hashes.
7. Explicit full-batch normalization compensation, with the paper step and the
   raw FullCert API argument both preserved in the result metadata.

## Decision

An exactly matched comparison appears feasible, but only through the constrained adapter above; it
is not valid to invoke the repository's stock CLI and call that result matched. The final comparison
remains blocked until the conversion, one-step parity, trajectory export, and common-evaluator gates
pass. Failure of any gate will downgrade the official result to a separately labelled
external-context comparison.
