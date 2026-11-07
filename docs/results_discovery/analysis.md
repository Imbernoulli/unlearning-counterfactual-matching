# Calibration discovery analysis

Discovery seed 42; adaptive selection-only exploration. J=max(error-gap probability, confidence KS), NOT Retrain AvgGap. No audit/test predictions were read. Single-run differences, especially below sampling variation, are not evidence of improvement.

Only completed, internally consistent runs enter plots/minima. Recovered copies are deduplicated, not treated as independent seeds. Different search counts and epoch horizons make family minima descriptive. Accuracy drifts use percentage points; positive means higher accuracy. J is in probability units. Original fallback is a valid recorded fallback, never an imputed score. Collector train_s/wall_s are full registered horizon costs, never best-checkpoint costs; no proportional best-time estimate is made.

## Method-family minima

| Family / schedule / BN | Completed search trials | Selected J | Retain drift pp | Selection drift pp | Epoch / budget | Trial |
|---|---:|---:|---:|---:|---:|---|
| PCGrad / C / frozen | 1 | 0.10347 | -0.100 | -0.640 | 1/30 | r3_geometry_symmetric_pcgrad_C |
| PCGrad / D / frozen | 1 | 0.10031 | -0.140 | -0.960 | 2/30 | r3_geometry_symmetric_pcgrad_D |
| dynamic ranks / C / frozen | 2 | 0.10178 | -0.180 | -0.800 | 1/30 | r3_dynamic_C_a03 |
| dynamic ranks / D / frozen | 2 | 0.10667 | -0.040 | -0.080 | 1/30 | r3_dynamic_D_a01 |
| energy/joint / C / frozen | 3 | 0.05222 | -0.160 | -0.720 | 23/30 | r4_energy_joint_C_a1 |
| energy/joint / D / frozen | 3 | 0.06996 | -0.240 | -0.720 | 16/30 | r4_energy_joint_D_a1 |
| fc / C / frozen | 2 | 0.13276 | +0.000 | +0.400 | 80/100 | r3_fc_C_lr01_a01_attempt2 |
| fc / D / frozen | 2 | 0.12889 | +0.000 | -0.080 | 92/100 | r3_fc_D_lr01_a01 |
| fixed quantile/kl / C / frozen | 2 | 0.09138 | -0.040 | +0.080 | 1/20 | r3_global_kl_C_a10 |
| fixed quantile/logodds_mse / A / inherited | 1 | 0.11404 | +0.000 | +0.400 | 1/10 | r1_A_inherited_lr003_a01 |
| fixed quantile/logodds_mse / B / inherited | 1 | 0.10391 | -0.080 | -0.640 | 1/10 | r1_B_inherited_lr003_a01 |
| fixed quantile/logodds_mse / C / frozen | 9 | 0.09569 | -0.240 | -0.480 | 2/10 | r1_C_frozen_lr003_a01 |
| fixed quantile/logodds_mse / C / inherited | 1 | 0.10307 | -0.100 | -0.800 | 1/10 | r1_C_inherited_lr003_a01 |
| fixed quantile/logodds_mse / D / frozen | 10 | 0.10040 | -0.100 | -0.960 | 2/20 | r1_D_frozen_lr003_a01 |
| geometry control / C / frozen | 1 | 0.09569 | -0.240 | -0.480 | 2/30 | r3_geometry_none_C |
| geometry control / D / frozen | 1 | 0.10040 | -0.100 | -0.960 | 2/30 | r3_geometry_none_D |
| layer4+fc / C / frozen | 2 | 0.08947 | -0.180 | -0.800 | 80/100 | r3_layer4_fc_C_lr01_a01 |
| layer4+fc / D / frozen | 2 | 0.11298 | -0.020 | +0.720 | 1/100 | r3_layer4_fc_D_lr01_a01 |
| mask 0.5/kl / C / frozen | 1 | 0.09369 | -0.060 | -0.160 | 1/30 | r4_mask50_C_donor_kl |
| mask 0.5/kl / D / frozen | 1 | 0.10044 | -0.240 | -0.320 | 1/30 | r4_mask50_D_donor_kl |
| mask 0.5/logodds_mse / C / frozen | 1 | 0.10147 | -0.160 | -0.560 | 2/30 | r4_mask50_C_global_mse |
| mask 0.5/logodds_mse / D / frozen | 1 | 0.09582 | -0.240 | -0.880 | 2/30 | r4_mask50_D_global_mse |
| mean-logodds ascent control / C / frozen | 1 | 0.11209 | +0.000 | +0.720 | 1/30 | r5_mean_logodds_ascent_C |
| mean-logodds ascent control / D / frozen | 1 | 0.11742 | -0.060 | -0.480 | 1/30 | r5_mean_logodds_ascent_D |
| rank donor/centered_logits_mse / C / frozen | 1 | 0.09791 | -0.080 | -0.160 | 4/20 | r2_C_rank_centered_logits_mse_lr003_a1 |
| rank donor/kl / C / frozen | 2 | 0.09369 | -0.240 | +0.240 | 2/20 | r2_C_rank_kl_lr003_a10 |
| rank donor/logodds_mse / C / frozen | 1 | 0.10240 | -0.200 | -0.800 | 1/20 | r2_C_rank_logodds_mse_lr003_a01 |
| rank donor/scores_mse / C / frozen | 2 | 0.10133 | -0.100 | +0.160 | 1/20 | r2_C_rank_scores_mse_lr003_a03 |

## Matched comparisons

Each pair matches: schedule, lr, alpha, bn_mode, loss, target_kind, seed, split_seed, anchor_weight, momentum, weight_decay, batch_size, epochs, forget_samples_per_epoch, retain_samples_per_epoch, updates_per_epoch, retain_subset, selection_size, train_sha256, original_sha256, harness_sha256, except the explicitly changed loss for target-matching versus mean-logodds-ascent controls. Selected epochs may differ because the same selection rule chooses within the matched complete horizon. The full matching values and treatment differences are in analysis.json. Same update budgets do not imply equal FLOPs: dynamic ranking adds forward passes, mask construction adds a forget gradient pass, and localized backward is cheaper. Ascent is the inherited margin_ascent CLI objective (minimize mean true-class log-odds), not signed-margin loss or standard negative-CE GA. Only equal complete horizons are paired: a completed 30-epoch ascent can match geometry-none at 30 epochs, while an all-network 100-epoch run is not retrospectively reselected from its first 30 epochs. Schedule A has 177 updates/epoch, B/C 159, D 18 in the current 256-batch setup; D uses 4,500 retain samples rather than 40,500.

The 1.36 sqrt(1/n_forget+1/n_selection) KS scale is an iid reference only, not a confidence interval for adaptive paired J changes. No difference is declared statistically significant; smaller differences require fresh splits/seeds.

| Comparison | Left | Right | Right minus left J | Retain drift delta pp | Selection drift delta pp | Selected epochs | Below iid KS reference? |
|---|---|---|---:|---:|---:|---:|---|
| all vs localized parameters (C) | r3_all_C_100epochs | r3_fc_C_lr003_a01_attempt2 | +0.03938 | +0.240 | +0.560 | 2 / 97 | True |
| all vs localized parameters (C) | r3_all_C_100epochs | r3_layer4_fc_C_lr003_a01 | +0.01413 | +0.240 | +0.000 | 2 / 3 | True |
| all vs localized parameters (D) | r3_all_D_100epochs | r3_fc_D_lr003_a01 | +0.03382 | +0.100 | +1.040 | 2 / 92 | True |
| all vs localized parameters (D) | r3_all_D_100epochs | r3_layer4_fc_D_lr003_a01 | +0.01502 | +0.100 | +0.960 | 2 / 2 | True |
| fixed vs dynamic ranks (C) | r3_geometry_none_C | r3_dynamic_C_a01 | +0.00751 | -0.060 | -0.160 | 2 / 2 | True |
| fixed vs dynamic ranks (D) | r3_geometry_none_D | r3_dynamic_D_a01 | +0.00627 | +0.060 | +0.880 | 2 / 1 | True |
| geometry none vs PCGrad (C) | r3_geometry_none_C | r3_geometry_symmetric_pcgrad_C | +0.00778 | +0.140 | -0.160 | 2 / 1 | True |
| all vs saliency mask (C) | r3_geometry_none_C | r4_mask50_C_global_mse | +0.00578 | +0.080 | -0.080 | 2 / 2 | True |
| target matching vs mean-logodds ascent (C) | r3_geometry_none_C | r5_mean_logodds_ascent_C | +0.01640 | +0.240 | +1.200 | 2 / 1 | True |
| geometry none vs PCGrad (D) | r3_geometry_none_D | r3_geometry_symmetric_pcgrad_D | -0.00009 | -0.040 | +0.000 | 2 / 2 | True |
| all vs saliency mask (D) | r3_geometry_none_D | r4_mask50_D_global_mse | -0.00458 | -0.140 | +0.080 | 2 / 2 | True |
| target matching vs mean-logodds ascent (D) | r3_geometry_none_D | r5_mean_logodds_ascent_D | +0.01702 | +0.040 | +0.480 | 2 / 1 | True |

## All attempts, including failures and incomplete trials

| Trial | Status | Selected J | Selected epoch | Reason |
|---|---|---:|---:|---|
| r1_A_inherited_lr003_a01 | complete | 0.11404 | 1 |  |
| r1_B_inherited_lr003_a01 | complete | 0.10391 | 1 |  |
| r1_C_frozen_lr003_a01 | complete | 0.095689 | 2 |  |
| r1_C_inherited_lr003_a01 | complete | 0.10307 | 1 |  |
| r1_D_frozen_lr001_a001 | complete | 0.11204 | 18 |  |
| r1_D_frozen_lr001_a01 | complete | 0.1072 | 2 |  |
| r1_D_frozen_lr001_a03 | complete | 0.10658 | 3 |  |
| r1_D_frozen_lr003_a001 | complete | 0.10191 | 13 |  |
| r1_D_frozen_lr003_a01 | complete | 0.1004 | 2 |  |
| r1_D_frozen_lr003_a03 | complete | 0.10316 | 2 |  |
| r2_C_global_lr001_a003 | complete | 0.10031 | 16 |  |
| r2_C_global_lr001_a01 | complete | 0.10178 | 3 |  |
| r2_C_global_lr001_a03 | complete | 0.10169 | 5 |  |
| r2_C_global_lr001_a1 | complete | 0.10062 | 5 |  |
| r2_C_global_lr003_a003 | complete | 0.099467 | 4 |  |
| r2_C_global_lr003_a03 | complete | 0.099822 | 2 |  |
| r2_C_global_lr003_a1 | complete | 0.1048 | 1 |  |
| r2_C_rank_centered_logits_mse_lr003_a1 | complete | 0.097911 | 4 |  |
| r2_C_rank_kl_lr003_a1 | complete | 0.099378 | 9 |  |
| r2_C_rank_kl_lr003_a10 | complete | 0.093689 | 2 |  |
| r2_C_rank_logodds_mse_lr003_a01 | complete | 0.1024 | 1 |  |
| r2_C_rank_scores_mse_lr003_a01 | complete | 0.10738 | 2 |  |
| r2_C_rank_scores_mse_lr003_a03 | complete | 0.10133 | 1 |  |
| r2_D_global_lr001_a1 | complete | 0.10631 | 3 |  |
| r2_D_global_lr003_a1 | complete | 0.1044 | 2 |  |
| r3_all_C_100epochs | complete | 0.095689 | 2 |  |
| r3_all_D_100epochs | complete | 0.1004 | 2 |  |
| r3_dynamic_C_a01 | complete | 0.1032 | 2 |  |
| r3_dynamic_C_a03 | complete | 0.10178 | 1 |  |
| r3_dynamic_D_a01 | complete | 0.10667 | 1 |  |
| r3_dynamic_D_a03 | complete | 0.10969 | 1 |  |
| r3_fc_C_lr003_a01 | failed_infrastructure | missing | missing | Same scientific configurations and source hashes as study_round3.json. The first two attempts failed due to provider root-disk exhaustion before any completed summary; replacement sandbox has 80GB disk. Those two configurations restart from Original with attempt2 IDs. Infrastructure failures are excluded from scientific trial counts. |
| r3_fc_C_lr003_a01_attempt2 | complete | 0.13507 | 97 |  |
| r3_fc_C_lr01_a01 | failed_infrastructure | missing | missing | Same scientific configurations and source hashes as study_round3.json. The first two attempts failed due to provider root-disk exhaustion before any completed summary; replacement sandbox has 80GB disk. Those two configurations restart from Original with attempt2 IDs. Infrastructure failures are excluded from scientific trial counts. |
| r3_fc_C_lr01_a01_attempt2 | complete | 0.13276 | 80 |  |
| r3_fc_D_lr003_a01 | complete | 0.13422 | 92 |  |
| r3_fc_D_lr01_a01 | complete | 0.12889 | 92 |  |
| r3_geometry_none_C | complete | 0.095689 | 2 |  |
| r3_geometry_none_D | complete | 0.1004 | 2 |  |
| r3_geometry_symmetric_pcgrad_C | complete | 0.10347 | 1 |  |
| r3_geometry_symmetric_pcgrad_D | complete | 0.10031 | 2 |  |
| r3_global_kl_C_a1 | complete | 0.096844 | 18 |  |
| r3_global_kl_C_a10 | complete | 0.091378 | 1 |  |
| r3_layer4_fc_C_lr003_a01 | complete | 0.10982 | 3 |  |
| r3_layer4_fc_C_lr01_a01 | complete | 0.089467 | 80 |  |
| r3_layer4_fc_D_lr003_a01 | complete | 0.11542 | 2 |  |
| r3_layer4_fc_D_lr01_a01 | complete | 0.11298 | 1 |  |
| r4_energy_joint_C_a01 | complete | 0.064267 | 30 |  |
| r4_energy_joint_C_a1 | complete | 0.052222 | 23 |  |
| r4_energy_joint_C_a10 | complete | 0.060178 | 23 |  |
| r4_energy_joint_D_a01 | complete | 0.071244 | 30 |  |
| r4_energy_joint_D_a1 | complete | 0.069956 | 16 |  |
| r4_energy_joint_D_a10 | complete | 0.076756 | 18 |  |
| r4_mask50_C_donor_kl | complete | 0.093689 | 1 |  |
| r4_mask50_C_global_mse | complete | 0.10147 | 2 |  |
| r4_mask50_D_donor_kl | complete | 0.10044 | 1 |  |
| r4_mask50_D_global_mse | complete | 0.095822 | 2 |  |
| r5_mean_logodds_ascent_C | complete | 0.11209 | 1 |  |
| r5_mean_logodds_ascent_D | complete | 0.11742 | 1 |  |
| smoke_D_frozen_1epoch | complete | 0.10667 | 1 |  |

Completed trials without a fully matched comparator: r1_A_inherited_lr003_a01, r1_B_inherited_lr003_a01, r1_C_frozen_lr003_a01, r1_C_inherited_lr003_a01, r1_D_frozen_lr001_a001, r1_D_frozen_lr001_a01, r1_D_frozen_lr001_a03, r1_D_frozen_lr003_a001, r1_D_frozen_lr003_a01, r1_D_frozen_lr003_a03, r2_C_global_lr001_a003, r2_C_global_lr001_a01, r2_C_global_lr001_a03, r2_C_global_lr001_a1, r2_C_global_lr003_a003, r2_C_global_lr003_a03, r2_C_global_lr003_a1, r2_C_rank_centered_logits_mse_lr003_a1, r2_C_rank_kl_lr003_a1, r2_C_rank_kl_lr003_a10, r2_C_rank_logodds_mse_lr003_a01, r2_C_rank_scores_mse_lr003_a01, r2_C_rank_scores_mse_lr003_a03, r2_D_global_lr001_a1, r2_D_global_lr003_a1, r3_dynamic_C_a03, r3_dynamic_D_a03, r3_fc_C_lr01_a01_attempt2, r3_fc_D_lr01_a01, r3_global_kl_C_a1, r3_global_kl_C_a10, r3_layer4_fc_C_lr01_a01, r3_layer4_fc_D_lr01_a01, r4_energy_joint_C_a01, r4_energy_joint_C_a1, r4_energy_joint_C_a10, r4_energy_joint_D_a01, r4_energy_joint_D_a1, r4_energy_joint_D_a10, r4_mask50_C_donor_kl, r4_mask50_D_donor_kl, smoke_D_frozen_1epoch
