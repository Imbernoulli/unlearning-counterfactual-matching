# Calibration-guided discovery evidence (closed)

This is a fixed snapshot, not new audit/test/Retrain results. The discovery used one Original and deletion seed 42. There are 58 completed scientific trials including one smoke, plus two root-disk infrastructure failures, for 60 launch attempts. Failures are not zero scores.

The preregistered feasible-selection criterion chose joint score-energy, schedule C, fixed BN, LR .003, forget weight 1, epoch 23. This selection is frozen before confirmations. J=.052222 measures a selection-only proxy and is not AvgGap or a deletion certificate. The table and full raw per-trial configurations, best metadata and full-horizon summaries are retained; see `analysis.md` for strictly matched comparisons and limits.

Original trial outputs/checkpoints remain SHA-verified in the local experiment archive. Earlier allocation cleanup lost some unbacked per-epoch r1/r2 NPZs; these are not silently reconstructed as observations. Every complete selected checkpoint and its metadata/history survived. Later allocation backups preserve all NPZs.

The original live table's timestamp kept refreshing after freeze, changing its file hash. The append-only selection receipt verifies that all 58 scientific rows in the stopped snapshot equal the rows embedded in the frozen method. No method/config/checkpoint changed.

Source scripts are in `research/unlearning_matching/` in the dev workspace. Raw task scores and confidential benchmark score transformations are not included. The PDF overview summarizes adaptive single-seed development; family minima are not causal or statistically confirmed improvements.
