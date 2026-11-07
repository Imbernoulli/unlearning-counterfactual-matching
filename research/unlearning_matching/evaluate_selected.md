# Frozen candidate evaluation

`evaluate_selected.py` is independent of the training and selection programs.
It has no model-selection operation. Audit and test are separate invocations;
each output directory is created once and cannot be overwritten, including
after an interrupted evaluation. Keep failed outputs for inspection.

First finish selection using only selection diagnostics. Prepare a readable
manifest with a `candidates` list. An entry has this form (paths below are
illustrative; freezing requires the actual files to exist):

```json
{
  "id": "selected_seed42",
  "method": "global_logodds_D",
  "seed": 42,
  "split_seed": 42,
  "epoch": 8,
  "config": {
    "schedule": "D",
    "lr": 0.003,
    "forget_weight": 0.1,
    "bn_mode": "frozen",
    "training_source_sha256": "copy from the run metadata"
  },
  "checkpoint": "/root/study/out/optimization/CHOSEN/seed42_split42/D/best.pt",
  "split_manifest": "/root/study/out/optimization/CHOSEN/seed42_split42/splits.json",
  "selection_audit_split": "/root/study/out/optimization/CHOSEN/seed42_split42/selection_audit_split.json",
  "selection_metadata": "/root/study/out/optimization/CHOSEN/seed42_split42/D/best.json",
  "preregistration": "/root/study/out/optimization/CHOSEN/seed42_split42/preregister.json",
  "paired_retrain": "/root/study/eval-only/refs/seed42_split42/retrain.pt",
  "paired_retrain_metadata": "/root/study/eval-only/refs/seed42_split42/retrain.json",
  "timing": {
    "training_loop_seconds": null,
    "target_generation_seconds": null,
    "diagnostic_seconds": null,
    "total_seconds": null
  }
}
```

Use the actual selected epoch and configuration. Do not invent runtime
components: an unmeasured component stays `null`. `training_metadata`,
`selection_metadata`, `preregistration`, and `paired_retrain_metadata` are
optional pinned files; include available source and training provenance.
For canonical Retrain metadata the test phase checks the seed, split seed,
182 epochs, and `test_used_during_training=false`.

Artifact fields accept either a path string or `{"path": "...", "sha256":
"..."}`. Relative paths resolve against the readable manifest's directory.
Freeze outputs absolute paths and verifies any supplied hashes. Prepare
paired Retrain files before freezing if the manifest will be used for test;
an audit-only manifest cannot later acquire new entries or reference files.

```bash
python research/unlearning_matching/evaluate_selected.py \
  --candidate-manifest selected.readable.json --freeze selected.frozen.json

python research/unlearning_matching/evaluate_selected.py \
  --candidate-manifest selected.frozen.json --phase audit \
  --data-dir /root/study/data --original /root/study/original/cifar10_resnet18_original.pt \
  --out-dir /root/study/final-evaluation --gpu 0 --threads 2

python research/unlearning_matching/evaluate_selected.py \
  --candidate-manifest selected.frozen.json --phase test \
  --data-dir /root/study/data --original /root/study/original/cifar10_resnet18_original.pt \
  --out-dir /root/study/final-evaluation --gpu 0 --threads 2
```

Audit reads only hash-verified training arrays. It reconstructs the canonical
4500 forget examples, the independent 1250 audit examples, and exactly the
same fixed 5000 retain examples as the calibration runner. It reports
`audit_J = max(abs(error_forget-error_audit), confidence_KS)` in probability
units, per-class KS, and utility drift against Original. Optional
`--include-selection` also reports selection summaries. Audit does not open
Retrain checkpoints or their metadata and does not choose another model.

Test requires the completed audit receipt and the identical manifest hash.
It verifies the complete dataset hash and calls the original benchmark's
`evaluate_state` for both candidate and paired Retrain. Each run's four
absolute gaps are computed before averaging. `test/summary.json` groups
equal methods/configurations and reports means and sample standard deviations
for candidate metrics, Retrain metrics, gaps, AvgGap, and measured runtime
components. Sample SD is `null` for a single run. Predictions, metadata,
source hashes, checkpoint hashes, split hashes, and evaluation receipts stay
under the evaluation directory.

Implementation verification used CPU only: canonical split reconstruction,
exact retain-subset identity, archive access trapping to reject any audit
test-array read, immutable freeze/audit outputs, absent-Retrain acceptance
during audit and rejection during test, repeated-confidence KS, same-manifest
receipt enforcement, and paired aggregation with sample SD. Integration
predictions were synthetic stubs; these checks are not scientific results.
