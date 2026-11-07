# Three-hour calibration-only optimization

Started 2026-11-07 02:14:30 UTC (November 6 local time). Deadline approximately
05:14:30 UTC; stop GPU work before cleanup. The user requested a simple, clean
method and authorized one or two H100/H200 GPUs on Daytona.

## Frozen evidence

The first round and its separately registered global-quantile follow-up are
closed. Their raw test results are reported in full. They will not be relabelled
as validation data, reused to select a new checkpoint, or silently overwritten.
This next study is exploratory algorithm development with explicitly separate
selection and audit data. Its final test results must be disclosed as a new
round rather than an untouched test discovery.

## Questions and candidate families

1. Does a long retain pass undo a preceding forget pass? Compare stagewise
   diagnostics in schedule A with uniformly interleaved, same-data schedules B/C.
2. Does the frozen Original KL anchor inhibit the intended update? B includes
   the inherited anchor and C omits it.
3. Does BatchNorm running-state drift confound target fitting? Compare inherited
   mode handling and frozen BatchNorm through model.eval(); autograd remains on.
4. Can balanced joint updates reach the same behavior with fewer retain samples?
   Schedule D sees only 4,500 retain examples per forget epoch. Report this
   different computation budget explicitly.
5. Can same-donor targets preserve both error and confidence tails? Compare fixed
   global log-odds targets with rank-aligned held-out donors, optionally placing
   donor non-true logits according to the Original non-true class ranking.
   Keep target construction free of Retrain and official test data.

Global log-odds MSE is the starting point. Same-donor score MSE, centered-logit
MSE, and KL are mechanistic comparisons, not implicit claims of novel losses.
A direct score-ascent control can diagnose whether having an endpoint matters.
No target neural network or large architecture change is planned in this round.

## Selection rule (fixed before this study's training)

Split the unused 2,500 calibration images classwise and deterministically into
1,250 selection images and 1,250 audit images. Keep the original 2,500 target
images disjoint. Never compute audit predictions during training or model
selection. Use the same fixed 5,000-example retain subset throughout.

For a checkpoint, use its clean/eval predictions on forget F and selection S.
Among checkpoints satisfying

- retain-subset accuracy >= Original retain-subset accuracy minus 0.5 percentage
  points;
- selection accuracy >= Original selection accuracy minus 1 percentage point,

minimize

    J = max(abs(error_F - error_S), KS(confidence_F, confidence_S)).

Errors and confidence CDFs use probability units [0,1]. KS is the unconditional
empirical two-sample statistic on true-class confidence. Tie breaking is earlier
epoch, smaller learning rate, then a stable configuration identifier. If none
is feasible, return Original. Select configuration using seed 42 only, then
freeze the complete method and stopping rule before confirmation runs on seeds
43 and 44. If a stronger selection protocol is used later, record it as a new
round rather than changing this rule retrospectively.

The operational utility thresholds are preset design choices. J is a validation
proxy, not an estimator or bound for paired-Retrain AvgGap. KS bounds discrepancy
for a fixed confidence-threshold attack; an RBF SVC can have multiple decision
intervals, and changing retain/nonmember predictions can change a refitted
attack. Matching low-dimensional behavior does not certify deletion.

## Audit, test, reporting

After selection, evaluate the selected checkpoint on audit exactly once. Report
its proxy metrics; do not switch configurations in response to audit feedback.
Only then evaluate frozen confirmation checkpoints against the already available
paired Retrain references using the official test evaluator. Report all final
confirmation seeds, failures, stagewise repair evidence, utility, four raw
metrics, gap, and complete costs separately from update-loop time.

Retain at most last/best checkpoints per exploratory configuration, then remove
unselected model checkpoints after retaining metadata and diagnostics. Save the
selected models, source hashes, configurations, split identities, and raw final
results. Stop only the owned Daytona sandboxes and owned helper processes.
