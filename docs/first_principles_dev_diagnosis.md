# Why target matching under-forgot, and what fixes it (dev seed 42)

Continues the initial study; its derivations (1-D OT and I-projection in
[mathematical_connections.md](mathematical_connections.md), energy geometry,
the susceptibility view in [physics_connections.md](physics_connections.md))
and probes ([target_bias_probe.md](target_bias_probe.md),
[mechanism_findings.md](mechanism_findings.md)) are kept unchanged and are the
starting point here. Interim development evidence; nothing below is a
confirmation result.

## 0. Protocol change and its limits

The previous rule selected by `J = max(|err_F - err_S|, KS(conf_F, conf_S))`
and forbade any Retrain information. Development now uses a replica of the
benchmark's four metrics on never-trained validation images
(`research/unlearning_matching/devgap/devgap.py`): UA, RA on the fixed
5000-image retain subset, "TA" on held-out H = target (2500) + selection (1250),
and MIA with the benchmark's SVC (C=3, gamma='auto') fit on retain-subset[:|H|]
vs H. Its gap to the paired Retrain is computed **only on deletion split 42**,
which becomes a development split. Official test data and the audit fold are
not read. Confirmation must use fresh deletion splits (43-49) and independent
Originals with a frozen configuration and fixed epoch, scored with the
unchanged benchmark evaluator. Seeds 42/142..542 share split 42's forget set.

Replica check (seed 42): Original 0.00/100/94.77/0.07, Retrain 5.00/100/94.48/11.96,
SalUn 2.24/99.62/93.89/11.09 (UA/RA/TA/MIA). Dev AvgGap: Original 4.30,
SalUn 1.15 (its test AvgGap over seeds 42-44 was 0.97), frozen joint-score energy
method at its frozen epoch 23: 1.69.

## 1. Anatomy of the metric

The SVC acts on one scalar (true-class confidence) with an RBF kernel of width
~1 on [0,1]. Fitted on Original, Retrain and SalUn it is a **single cut**:

| model | cut c* | forget below c* (= MIA) | held-out below c* | retain below c* |
|---|---:|---:|---:|---:|
| Original | 0.961 | 0.07% | 10.19% | 0.11% |
| Retrain | 0.967 | 11.96% | 12.13% | 0.08% |
| SalUn | 0.847 | 11.09% | 14.19% | 4.05% |

So MIA-efficacy = P_F(confidence < c*), with c* set by the model's own
retain/held-out separation; UA = P_F(argmax wrong). Both are **lower-tail mass**
statistics of the forget confidence distribution. The extreme top
(0.99-0.9999), where most mass and most KS discrepancy lie, does not enter.
The initial probe found the KS-maximizing threshold at 0.996-0.999, so J rewarded
changes the benchmark cannot see and stopped training early (selected epochs
1-3 in 50 of 58 trials). Retrain's defining property is visible directly:
forget and held-out are exchangeable under it (11.96 vs 12.13; error 5.0 vs 5.5).

## 2. Three separable error sources of fixed-target matching

**(a) Fitting residual (optimization).** Schedule C puts a forget batch at
18 of 159 steps and clips that gradient to norm 5. The frozen energy method
reaches only UA 3.7 / MIA 7.7 at epoch 23 and still improves at 30 (1.11).
Putting a forget batch at every retain step (same loss, same clip) reaches
UA 4.2 / MIA 8.1 after one epoch. Unclipped log-odds MSE diverges (NaN): the
lowest-ranked forget samples start ~13 log-odds from their targets, so the
clip is load-bearing for MSE.

**(b) Reference bias (target).** Even a perfect fit to Original's held-out
distribution caps MIA at Original's held-out tail mass, 10.2%, below Retrain's
12.0%. Retrain was trained on 10% fewer images: its held-out log-odds are
0.4-0.9 lower in the 5-20% quantile band (slightly higher at the top, which the
metric ignores). This is the counterfactual quantity the initial probe could not
choose without a reference. A constant target shift is the crudest model of
it; on dev, shift -1 / -1.5 lift MIA to 11-12 and dev AvgGap to 0.26 / 0.22
(25 dense epochs, **over budget**; see section 4).

**(c) Generalization coupling.** Pushing forget samples down also lowers the
model's confidence on unseen look-alikes: during the shift runs the model's
own held-out tail mass rises from 10.2% to 12-13%. With self-referential
targets (forget matched to the *current* model's held-out outputs, rebuilt
every 20 steps), the loop amplifies itself: after one epoch TA fell
94.8 -> 92.4, RA 100 -> 99.3, held-out tail 14.2%, while forget/held-out
exchangeability was nearly met. Rough coupling: held-out error rose ~0.36 pp per
pp of forget error. Exchangeability is necessary but not sufficient; the
held-out behaviour itself must stay near the counterfactual.

## 3. What a good update has to do

1. Move ~5% of forget samples across the decision boundary and ~12% below c*,
   i.e. reshape the forget lower tail, not the bulk.
2. Aim at the counterfactual held-out tail (Retrain-like), not Original's.
3. Keep held-out/retain behaviour near Retrain's despite coupling.
4. Do it inside the benchmark budget (10 retain + 11 forget passes) to be
   comparable with SalUn.

## 4. Budget

The benchmark harness allows at most 10 retain and 11 forget passes. The
dense runs above use 25 retain passes and ~220 forget passes, and read
validation images, which a benchmark method never receives. They are diagnostic
only, not a SOTA comparison. Budget-matched variants:
sparse (forget every 9th step, 10 epochs, 10.2 forget passes) and mixed
(benchmark "both" pass: every 256-batch of shuffled retain+forget, 10 epochs =
10 + 10 passes). Held-out validation use remains a disclosed extra resource.

Results of the budget-matched grids are appended below when complete.

## 5. Budget-matched development results (dev split 42, AvgGap vs Retrain-42; SalUn 1.147)

| schedule (10 epochs, <= 11 forget passes) | dev AvgGap | note |
|---|---:|---|
| sparse, forget every 9th step, clip 5, shift -1.5 | 0.648 (sel 0.972) | frozen as config A |
| same, shift -1 / -2 / -2.5 | 0.82 / 1.12 / 0.68 | endpoint noise +-0.4 |
| same, tail-focused loss tau 4.6, shift -1 | 0.74 (sel 0.66) | no clear gain |
| mixed both-pass (26-sample forget slices) | 1.26-1.53 | noisy clipped steps damage RA/TA |
| front-loaded forget, then retain-only epochs | 1.29-2.11 | retain epochs relearn: MIA 10 -> 5-7 |
| unclipped MSE or Huber | diverged | frozen BN removes scale damping |

The forget/retain tug-of-war makes the endpoint phase-dependent: each forget kick
lowers the tail, ~8 retain steps then sharpen everything (including held-out)
back. Stopping right after a kick or after a long retain phase gives very
different models.

## 6. Confirmation of frozen config A (official test, unchanged evaluate_state)

Frozen before any test evaluation (`research/unlearning_matching/devgap/frozen_A_20261107.json`).
10 runs: fresh deletion splits 43-47 (canonical Original) and independent
Originals 142-542 (split 42's forget set). SalUn checkpoints from the same runs
(inherited SalUn code/hyperparameters).

| | AvgGap paired | UA gap | RA gap | TA gap | MIA gap | fixed-reference AvgGap |
|---|---:|---:|---:|---:|---:|---:|
| config A | **0.732 +- 0.259** | 0.55 | 0.08 | 0.55 | 1.74 | **0.655** |
| SalUn | 1.229 +- 0.197 | 2.23 | 0.78 | 1.14 | 0.76 | 1.238 |
| Original | 4.533 | 5.29 | 0.00 | 0.46 | 12.39 | 4.592 |

Per run, A beats SalUn in 9/10 (paired difference 0.497 +- 0.346, t = 4.55,
df 9; one-sided sign test p = 0.011). Fresh splits: 0.822 vs 1.150 (4/5);
independent Originals: 0.641 vs 1.307 (5/5). A's remaining error is MIA
(11.0 vs Retrain 12.7) and a slightly low TA (93.9 vs 94.4); SalUn's is UA
(3.1 vs 5.3) and TA (93.3). Disclosed differences: A reads 2500 never-trained
validation images (and Original's outputs on them) and uses clean forget
forwards, which benchmark methods cannot; pass counts are within the benchmark
budget (10 retain, 10.0 forget). Results file:
`/mnt/datablaze2/bohanlyu/unlearning-20261106/devgap/confirm/A_k9_sh-1.5_e10/final_test.json`.
Interim: the independent coco review is pending (pool returned 503 all night).
