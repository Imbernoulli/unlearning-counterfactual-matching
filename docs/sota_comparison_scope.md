# Scope of any competitive claim — 2026-11-07

Beating the inherited, locally rerun SalUn baseline is a necessary project milestone, not by itself evidence of field-wide SOTA. Preserve identical splits, paired Retrains, attack definitions and units, and disclose all update/search costs. Do not tune on confirmation audit or test outcomes.

## Closest additional comparison

Spectral Saliency Unlearning (SSU), current arXiv v2, reports SalUn+SSU AvgGap **1.10±0.20** versus SalUn **1.21±0.25** over five seeds; FA/RA/TA/MIA are97.06/99.38/93.26/13.38. Source: [Table1, PDF page6](https://arxiv.org/pdf/2608.15548v2#page=6), also [HTML Table1](https://arxiv.org/html/2608.15548v2#S5.T1). Its CIFAR10 random10 experiment uses5000 forget/40000 retain, LR.0113,10 epochs and spectral keep ratio.70: [AppendixB.1, PDF page19](https://arxiv.org/pdf/2608.15548v2#page=19). Our4500/40500 setting therefore needs a same-harness reproduction. Earlier secondary0.77 figures are not the current primary table.

## Other strong methods with different protocols

- Adaptive Gradient Reweighting and Multi-stage Objective Optimization, CVPR2026: RN18/CIFAR10 random10 AvgGap1.57; its SalUn5.38 and LoTUS5.33. [Table1, PDF page6](https://openaccess.thecvf.com/content/CVPR2026/papers/Lu_Machine_Unlearning_via_Adaptive_Gradient_Reweighting_and_Multi-stage_Objective_Optimization_CVPR_2026_paper.pdf#page=6). Its MIA is attack accuracy (Retrain74.98), and its200-epoch/cosine/early-stop recipe differs; these values cannot be compared directly to our confidence-SVC nonmember efficacy.
- LoTUS, CVPR2025: RN18/CIFAR10 random10 AvgGap.0350 with real heldout and.0300 with synthetic heldout. Those are probability units, or3.50/3.00 percentage points. [Table2, PDF page7](https://arxiv.org/pdf/2503.18314#page=7). Its attack-accuracy definition and optimization recipe differ.
- ReGUn, TMLR2026: RN18/CIFAR10 random10 GapAvg2.00±.18; Amun1.46±.15 and SalUn2.48±.13 in the same table. [Table1, PDF page9](https://arxiv.org/pdf/2603.11210v3#page=9). This uses RMIA/SMIA AUC, so the aggregate is not our four-metric AvgGap. ReGUn is also direct prior art for heldout-guided targets.

Primary papers were retrieved and checked under /tmp/unlearning_sota_audit/. No value above is a new local measurement. A supported claim would name the exact protocol and reproduced methods; matching these heterogeneous paper means is insufficient.
