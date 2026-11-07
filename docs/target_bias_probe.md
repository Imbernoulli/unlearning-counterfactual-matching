# Frozen-target mismatch versus optimization residual

This CPU-only analysis finds that the late `layer4_fc` trajectories fit their
fixed training targets closely while remaining separated from current held-out
selection predictions. More training can reduce fitting residual without
eliminating that second discrepancy. This is a distribution diagnostic on seed
42, not a causal decomposition, a counterfactual validation, or an unlearning
certificate.

## Definitions and provenance

- **F**: current predictions on the 4,500 forget examples.
- **Q**: the fixed, class-conditional quantile targets assigned to those examples
  before unlearning. Targets average the Original held-out empirical quantile
  function over each forget quantile bin; they do not average classes together.
- **T0**: frozen Original predictions on the 2,500 target-estimation examples.
- **S**: current predictions on the separate 1,250 selection examples.
- **S0**: Original predictions on those same selection examples.

The scalar is true-class log odds
`s = z_y - logsumexp(z_other)`, and confidence is `sigmoid(s)`. F, T0, S and S0
use saved FP32 softmax confidences; Q confidence is computed from its saved or
reconstructed FP32 log odds. Confidence KS and log-odds KS agree up to finite
precision, as expected from their strictly monotone relationship. A confidence
threshold of 0.5 is not the multiclass error event; no UA is inferred from it.

The probe reads only exact `forget`, `target` and `selection` log-odds,
confidence and label arrays from saved distributions, the `logodds` target
array, preregistered configuration, and the existing selected epoch. It loads
no checkpoints and performs no inference. It does not open audit, official-test
or Retrain predictions, `results.json`, or retained-set predictions. Hashing an
NPZ reads its file bytes for provenance, but the probe does not extract any
unlisted array.

All runs use Original/deletion seed 42, schedule C, frozen BN, log-odds MSE,
forget weight 0.1 and no anchor. The r1 run updates all parameters; r3 updates
only `layer4` and `fc` (8,398,858 of 11,173,962 parameters). These differing scopes
and training budgets preclude treating r1/r3 as an isolated learning-rate
comparison.

| Run | Available distribution epochs | Q provenance | Existing selected epoch |
|---|---|---|---:|
| r1 all parameters, lr .003 | 0, 2, 10 only | Deterministically reconstructed from epoch-0 F/T0 | 2 |
| r2 rank-donor KL, weight 1 | None | Unavailable | Unavailable |
| r2 rank-donor KL, weight 10 | None | Unavailable | Unavailable |
| r3 layer4+fc, lr .003 | Every integer epoch 0–100 | Saved target; reconstruction agrees exactly | 3 |
| r3 layer4+fc, lr .01 | Every integer epoch 0–100 | Saved target; reconstruction agrees exactly | 80 |

The old r1 target cache and unlisted epoch NPZ files are absent. The previous
allocation was deleted, so the missing files cannot be recovered. r1 Q is
explicitly **reconstructed, not a recovered saved target**. The NumPy
reconstruction implements the frozen empirical-quantile interval-overlap
algorithm and checks its source hash against preregistration. Its output is
identical, element for element, to both available saved r3 Q arrays. No r2
trajectory or KL comparison is possible from the surviving NPZ artifacts.

## Metrics

KS is the exact two-sample empirical CDF distance. Log-odds quantile RMSE is
the Euclidean RMSE between NumPy linear quantiles at 1,001 uniformly spaced
levels from 0 to 1. Tail RMSE uses 101 levels from 0 to 0.1. These are sampled
quantile-grid diagnostics, not an asserted exact empirical W2 computation.
Pointwise RMSE pairs each F example with its own fixed Q target, preserving
the training assignment; marginal quantile RMSE can be small even when
individual target identities fit less well.

## Main result: fitting Q eventually stops explaining most of the discrepancy

Pooled log-odds quantile RMSE:

| Run / epoch | F–Q | Q–S | F–S | T0–S | S0–S | Pointwise F–Q |
|---|---:|---:|---:|---:|---:|---:|
| Common Original, 0 | 2.248 | .384 | 2.541 | .367 | .000 | 2.270 |
| r1 .003, 2 selected | .814 | .785 | 1.386 | .765 | .630 | 1.425 |
| r1 .003, 10 | .217 | .611 | .782 | .594 | .508 | .574 |
| r3 .003, 3 selected | 1.477 | 1.025 | 1.724 | 1.011 | .955 | 1.670 |
| r3 .003, 10 | .928 | .736 | 1.430 | .718 | .627 | 1.202 |
| r3 .003, 25 | .298 | .871 | 1.001 | .852 | .740 | .500 |
| r3 .003, 50 | .168 | .872 | .856 | .854 | .745 | .262 |
| r3 .003, 80 | .062 | .806 | .825 | .787 | .666 | .153 |
| r3 .003, 100 | .157 | .871 | .791 | .854 | .747 | .226 |
| r3 .01, 3 | 1.273 | .672 | 1.747 | .652 | .511 | 1.744 |
| r3 .01, 10 | .403 | .802 | 1.038 | .784 | .646 | .691 |
| r3 .01, 25 | .112 | .749 | .803 | .730 | .567 | .433 |
| r3 .01, 50 | .186 | .769 | .729 | .751 | .605 | .325 |
| r3 .01, 80 selected | .410 | .952 | 1.083 | .934 | .746 | 1.215 |
| r3 .01, 100 | .099 | .686 | .755 | .666 | .463 | .260 |

At epoch 100, F–Q marginal residual is small in both r3 runs, including
pointwise residual. Yet Q–S remains .871 and .686. This shows why interpreting
the remaining F–S discrepancy solely as insufficient target fitting would be
incorrect. At the existing selected checkpoints the diagnosis differs:
the .003 run still has large F–Q residual, and the .01 selected epoch has
pointwise residual 1.215 despite marginal residual .410. Endpoint diagnostics
do not replace the preregistered checkpoint rule or establish that epoch 100
has better utility.

The trajectories oscillate. Over epochs 25–100, F–S quantile RMSE stays within
.791–1.001 for lr .003 and .705–1.083 for lr .01, while F–Q varies within
.060–.390 and .061–.432. Thus a single small F–Q epoch should not be described
as monotone optimization convergence.

Confidence KS gives the same qualitative separation:

| Run / epoch | F–Q KS | Q–S KS | F–S KS | T0–S KS | S0–S KS |
|---|---:|---:|---:|---:|---:|
| Common Original, 0 | .1422 | .0281 | .1400 | .0288 | .0000 |
| r1 .003, 2 selected | .0609 | .1346 | .0957 | .1340 | .1576 |
| r1 .003, 10 | .0293 | .1236 | .1348 | .1228 | .1416 |
| r3 .003, 3 selected | .1920 | .2487 | .1098 | .2480 | .2752 |
| r3 .003, 100 | .0611 | .1599 | .1160 | .1592 | .1848 |
| r3 .01, 80 selected | .1224 | .2101 | .0895 | .2096 | .2320 |
| r3 .01, 100 | .0196 | .0984 | .1085 | .0988 | .1144 |

## The mismatch is present within classes and in the low-confidence tail

Epoch-100 per-class log-odds quantile RMSE:

| CIFAR-10 class index | .003 F–Q | .003 Q–S | .003 F–S | .01 F–Q | .01 Q–S | .01 F–S |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | .114 | .794 | .706 | .123 | .634 | .597 |
| 1 | .169 | .546 | .474 | .106 | .429 | .448 |
| 2 | .263 | .885 | .779 | .126 | .825 | .879 |
| 3 | .363 | 1.584 | 1.344 | .219 | .909 | 1.071 |
| 4 | .121 | 1.762 | 1.741 | .148 | 1.784 | 1.895 |
| 5 | .141 | .783 | .703 | .184 | .680 | .724 |
| 6 | .232 | .935 | .771 | .176 | .792 | .706 |
| 7 | .187 | .827 | .763 | .185 | .627 | .702 |
| 8 | .100 | .653 | .646 | .132 | .627 | .645 |
| 9 | .100 | .658 | .638 | .096 | .492 | .521 |

All classes show Q–S RMSE larger than F–Q at epoch 100. The pooled finding
therefore does not arise solely from cancellation between classes. Class 4
has the largest Q–S discrepancy in both runs, but these are descriptive
single-seed observations, not significance tests.

Lower-10% log-odds quantile RMSE:

| Run / epoch | F–Q tail | Q–S tail | F–S tail | T0–S tail |
|---|---:|---:|---:|---:|
| Common Original, 0 | 6.746 | 1.155 | 7.680 | 1.100 |
| r1 .003, 2 selected | 2.425 | 1.638 | 3.822 | 1.578 |
| r1 .003, 10 | .383 | 1.232 | 1.577 | 1.177 |
| r3 .003, 3 selected | 4.351 | 1.396 | 4.998 | 1.341 |
| r3 .003, 100 | .149 | 1.660 | 1.676 | 1.604 |
| r3 .01, 80 selected | .916 | 2.169 | 2.995 | 2.115 |
| r3 .01, 100 | .177 | 1.689 | 1.833 | 1.632 |

The tail comparison already has a substantial T0–S0 difference before any
unlearning (1.100 RMSE). It must not all be attributed to optimization drift:
the two held-out subsets differ at baseline, and this probe does not estimate
sampling uncertainty. At epoch 100, Q–S has 3.356 percentage points fewer
examples below confidence .5 for lr .003 and 2.876 points fewer for lr .01.
F–Q differs by only −.067 points in both runs. These are confidence-mass
differences, not error-rate gaps.

## What this says about target construction

Q closely reproduces its intended reference T0: pooled KS .00258,
log-odds quantile RMSE .0304 and tail RMSE .0852. Its small discrepancy is
consistent with unequal class counts and barycentric quantile discretization.
The numerical quantile construction itself is not large enough to explain
the late F–S separation.

Both T0–S and S0–S increase during training. This establishes that current
held-out behavior changes while Q remains frozen. It does **not** establish
that following current S would be correct: current S may itself have drifted
in an undesirable direction. Without a counterfactual reference, this probe
cannot choose between frozen Original unseen behavior and current unseen
behavior as the appropriate unlearning target.

For a fixed observed epoch, the triangle inequality gives
`d(F,S) >= d(Q,S) - d(F,Q)` for KS and for the sampled quantile-grid RMSE.
At r3 epoch 100 the latter lower bounds are .714 for lr .003 and .587 for
lr .01; observed F–S distances are .791 and .755. Thus perfectly fitting the
same Q, **while holding that epoch's S fixed**, would leave a substantial
discrepancy. S depends on model parameters, so these are descriptive endpoint
bounds, not immutable optimization floors or proofs of target bias.

The appropriate next research question is how to control reference behavior
and target fitting together. This analysis motivates inspecting both terms
whenever evaluating a fixed-target method. It does not launch new candidates,
change the runner, claim that rank transfer is valid, or establish a preferred
dynamic-target algorithm.

## Reproduction and audit trail

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 \
  research/unlearning_matching/probe_target_bias.py --self-test \
  --root .local-saves/unlearning-project-20261106/optimization/daytona_recovery/out/optimization \
  --output-dir .local-saves/unlearning-project-20261106/optimization/target_bias_probe
python3 -m py_compile research/unlearning_matching/probe_target_bias.py
```

Self-checks passed for tied empirical KS, identity quantiles and unequal-size
quantile-bin reconstruction. The outputs are `target_bias.json` and
`target_bias.csv` in the indicated output directory; JSON records all 215
input artifact hashes and the exact arrays or metadata fields extracted.

Probe source SHA256:
`cbd8dd16410d352327999b16bdea46b85f623577a9d3947e076aa2e7ac3c11d0`.

Frozen quantile source SHA256 checked against preregistration:
`760a2147acee205d2c34c0fbc4113be60161cea7bde74dd977bab4d81f625c79`.

Both saved r3 target NPZ files have SHA256:
`8ceb2e499c4cab5c7aa5cd509f945044b1063234c30115d7af83ca4007aca9b9`.
