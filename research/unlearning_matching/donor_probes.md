# Donor-target probes: what cached Original predictions can identify

These are CPU-only diagnostics of the already frozen Original/target caches,
not new training results. The implementation reads exactly four named arrays
from each `fixed_targets.npz`: `forget_original_logits`, `forget_labels`,
`validation_original_logits`, and `validation_labels`. It does not load any
checkpoint, result file, selection or audit prediction, official test example,
or Retrain prediction. In this section, an error is negative signed margin.

The caches are under
`.local-saves/unlearning-project-20261106/daytona/out/fixed_quantile/seed{42,43,44}_split{42,43,44}/`.
The three deletion splits share the Original and the same 2,500 target images;
they are not three independent Original training runs.

## 1. Confidence versus error compatibility

Let `s = z_y - logsumexp(z_other)` and `m = z_y - max(z_other)`. If
`r_max` is the largest Original probability conditional on a non-true class,
the I-projection target with prescribed log-odds `s*` has

    m_I = s* - log(r_max_Original).

It therefore need not share the margin of a held-out donor with the same
true-class probability. A same-donor target copies both statistics, with its
non-true logits permuted into the Original's competing-class order.

| Deletion seed | I-projection target error | Same-donor target error | Target held-out error | Target error-identity disagreement |
|---|---:|---:|---:|---:|
| 42 | 4.133% | 4.489% | 4.560% | 0.444% |
| 43 | 4.111% | 4.622% | 4.560% | 0.511% |
| 44 | 4.067% | 4.533% | 4.560% | 0.467% |

Same-donor uses midpoint donor selection, whereas I-projection uses
barycentric log-odds quantiles; this comparison includes both discretization
and non-true-shape differences. The median non-true `r_max` is .280/.286/.290
on the three forget sets versus .308 on target. That direction explains the
I-projection's somewhat more positive margins. Original within-class
log-odds/margin rank correlation averages .954--.956.

This is a real but small target discrepancy, around half a percentage point.
It cannot on its own explain large utility degradation after fitting.

## 2. Output-gradient concentration and rank sensitivity

For seed 42, only 4.489% of same-donor targets have negative margin. At the
Original outputs, these points account for the following squared output-loss
gradient norms (summing over examples and classes):

| Loss | Negative-donor group's share | Top 10% of examples' share |
|---|---:|---:|
| Forward KL | 92.53% | 99.97% |
| Centered-logit MSE | 48.30% | 68.26% |

For forward KL, the logit gradient is `p_Original - q_target`. For centered
logit MSE it is proportional to the centered-logit difference. Global loss
reduction constants cancel within each reported share. **These are output
gradients, not parameter gradients, gradient conflict, or optimizer movement.**
The network Jacobian, sample interactions, clipping, momentum, and retain
updates can change their effect. This probe motivates measuring parameter
gradient conflict separately for the negative-donor tail and the other points.

A separate sensitivity control multiplies both source and donor logits by a
common positive scale **only to compute their ranks**, while retaining the
unscaled donor target values. Scaling by .8/1.25 leaves all argmax predictions
unchanged but changes 82.07%/83.47% of donor IDs. More relevantly, the negative
target identities change on only .533%/.333% of forget points, and target
log-odds mean absolute changes are .102/.110. Donor-ID churn alone overstates
target instability. Margin ranks are exactly invariant to such global positive
scaling; `rank_by="margin"` is a small, interpretable target ablation.

## 3. Target-only pseudo-unlearning does not verify counterfactual identities

Split the 2,500 target images into two classwise halves, rank one half as a
pseudo-source and the other as donors, and repeat with ten fixed random seeds.
For both log-odds and margin rank, the error identities disagree on 2.512% of
the 1,250 pseudo-source points on average. The mean disagreement count is
31.4. But the sum of absolute **per-class error-count differences** is also
31.4: disagreement above this unavoidable count floor is zero.

This does not establish reliable semantic or counterfactual matching. With
margin ranking, errors are exactly the negative prefix of a sorted scalar;
prefix alignment is largely a consequence of the monotone sign threshold.
Log-odds has the same error-tail ordering on this cache. Both halves are also
unseen, so this exercise never observes how membership changed a point's rank.
It has **not independently identified the member-to-nonmember counterfactual
rank assumption**. Neither a positive reliability claim nor a strong identity-
mismatch claim follows from this result.

Pending the network/optimization probes, these data do not justify a more
complex feature-neighbor target model. The minimal follow-ups are the already
specified margin-rank comparison or reassignment by current full-forget rank,
with the frozen held-out target distribution unchanged.

## Reproduction

From the repository root, with NumPy installed:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 \
  research/unlearning_matching/donor_probes.py \
  .local-saves/unlearning-project-20261106/daytona/out/fixed_quantile/seed42_split42/fixed_targets.npz \
  .local-saves/unlearning-project-20261106/daytona/out/fixed_quantile/seed43_split43/fixed_targets.npz \
  .local-saves/unlearning-project-20261106/daytona/out/fixed_quantile/seed44_split44/fixed_targets.npz \
  --output-json /tmp/unlearning_donor_probes.json
```

The JSON records source and cache SHA256 hashes, array-read names, all three
diagnostics, and the interpretation limits. No model or GPU is initialized.
