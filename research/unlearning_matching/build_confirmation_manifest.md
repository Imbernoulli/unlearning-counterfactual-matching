`build_confirmation_manifest.py` converts an explicit, already decided confirmation plan into the readable schema accepted by the frozen evaluator. It uses only the Python standard library. It does not search directories, compare selection scores, load models, or read prediction archives, history, summary, audit, or test outputs. Supply the actual persisted preregistration values, including wrapper-specific loss/target names; CLI placeholders are insufficient for energy methods.

All paths are relative to the plan file unless absolute. Each group supplies a method name, common reporting configuration, and nonempty `preregister_checks`: JSON pointers into the actual `preregister.json.configuration`. For dynamic rank, parameters such as the learning rate live at `/arguments/lr`. Include all frozen algorithm choices and source hashes in these checks. The emitted reporting config includes these checks and the schedule, and is identical across members of a group; seed, actual selected epoch, checkpoint paths and registered horizon remain separate candidate fields.

The following is a schema example, not a selected method or a scientific result. Replace the paths, method configuration, source hash and explicit epochs with the decisions recorded before confirmation. Canonical seed42 uses the previously selected `best`; canonical seeds43/44 use the fixed-epoch `last`. The selected epoch and the search horizon are distinct.

```json
{
  "schema": 1,
  "groups": {
    "frozen_method": {
      "method": "REPLACE_WITH_FROZEN_METHOD",
      "config": {
        "duration_rule": "seed42 selection completed before freeze; other seeds fixed at its selected epoch"
      },
      "preregister_checks": {
        "/source_sha256": "REPLACE_WITH_ACTUAL_SOURCE_SHA256",
        "/lr": 0.003,
        "/forget_weight": 1.0,
        "/target_kind": "global_logodds",
        "/loss": "logodds_mse",
        "/bn_mode": "frozen"
      }
    }
  },
  "candidates": [
    {
      "id": "canonical42",
      "group": "frozen_method",
      "schedule_dir": "search/chosen/seed42_split42/C",
      "checkpoint_kind": "best",
      "seed": 42,
      "split_seed": 42,
      "epoch": 1,
      "preregistered_epochs": 20,
      "paired_retrain": "references/seed42_split42/retrain.pt",
      "paired_retrain_metadata": "references/seed42_split42/retrain.json"
    },
    {
      "id": "canonical43",
      "group": "frozen_method",
      "schedule_dir": "final_replicates/chosen/seed43_split43/C",
      "checkpoint_kind": "last",
      "seed": 43,
      "split_seed": 43,
      "epoch": 1,
      "preregistered_epochs": 1,
      "paired_retrain": "references/seed43_split43/retrain.pt",
      "paired_retrain_metadata": "references/seed43_split43/retrain.json"
    },
    {
      "id": "canonical44",
      "group": "frozen_method",
      "schedule_dir": "final_replicates/chosen/seed44_split44/C",
      "checkpoint_kind": "last",
      "seed": 44,
      "split_seed": 44,
      "epoch": 1,
      "preregistered_epochs": 1,
      "paired_retrain": "references/seed44_split44/retrain.pt",
      "paired_retrain_metadata": "references/seed44_split44/retrain.json"
    }
  ]
}
```

Each artifact field for a paired reference also accepts `{"path": "...", "sha256": "..."}`. Optional `checkpoint_sha256` checks the exact preselected candidate bytes. The builder pins all emitted checkpoint, split, preregistration and metadata paths with SHA256. It validates seed/split/horizon, checkpoint kind/epoch, source/data/Original pins, split sizes and disjoint membership, train-only flags and the paired 182-epoch Retrain's metadata. Exact label-stratified split reconstruction remains in the frozen evaluator's audit phase.

```bash
python research/unlearning_matching/build_confirmation_manifest.py \
  --plan canonical.plan.json --out canonical.readable.json
python research/unlearning_matching/evaluate_selected.py \
  --candidate-manifest canonical.readable.json --freeze canonical.frozen.json
```

Independent Original142 and Original242 each need their own plan, readable/frozen manifest, and audit/test output root. Their unlearning seeds are142/242, their deletion split seed is42, their checkpoints are fixed-epoch `last`, and their paired Retrains have the same corresponding seed and split42. Add `"original_manifest": "independent_originals/seed142/original.json"` (or242) to every candidate entry in that Original's plan. This invokes the existing CPU-only independent Original verifier. The builder rejects a mixture of Original identities in one output. Subsequent freezing and evaluation must use the matching wrapper:

```bash
python research/unlearning_matching/build_confirmation_manifest.py \
  --plan independent142.plan.json --out independent142.readable.json
python research/unlearning_matching/evaluate_independent.py \
  --original-manifest independent_originals/seed142/original.json \
  --candidate-manifest independent142.readable.json --freeze independent142.frozen.json
```

See [evaluate_selected.md](evaluate_selected.md) and [evaluate_independent.md](evaluate_independent.md) for the one-way audit-then-test commands. The builder never runs these phases. Existing outputs are never overwritten.

For fixed `last`, `training_loop_seconds` comes from that exact checkpoint's cumulative `train_seconds`. For selected `best`, it remains `null`: the base runner did not store its cumulative loop timer, and `last.json.train_seconds` may describe many more epochs. An explicit `target_generation_seconds` in the checkpoint metadata is retained. Other absent timing fields remain `null`, including `total_seconds`. Raw `wall_seconds` is preserved separately as `confirmation.schedule_wall_seconds`; it excludes setup and target generation and is not called total time. These fields describe execution costs only; they do not assert end-to-end savings or exclude the earlier search cost.

CPU verification on2026-11-06 used `/tmp/check_unlearning_confirmation_builder.py`, with evidence at `/tmp/confirmation-builder-cpu-b2hmyl3n`. It exercised readable-to-frozen compatibility, identical grouping across seeds, the early-best timing trap, dynamic `/arguments` configuration, real historical metadata and a real independent Original142 pin. Wrong epoch/seed/config/hash, incomplete last horizon, duplicate IDs, mixed Originals and overwrites were rejected. Import traps prohibited Torch/NumPy/scikit-learn/pandas; read traps prohibited history/summary/audit/test reads. Synthetic checkpoint bytes were never evaluated as models or scientific results. Both frozen evaluator source hashes remained unchanged.
