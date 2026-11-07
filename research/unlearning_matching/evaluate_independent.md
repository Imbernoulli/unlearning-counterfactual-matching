# Independent-Original confirmation

Use `evaluate_independent.py` only after the method and hyperparameters have
been frozen. It reuses the unchanged `evaluate_selected.py` freeze, one-way
audit, and paired test operations. Independent Original seeds assess stability
across trained models; their audit and test results must not drive selection.

The wrapper requires `train_original.py`'s completed `original.json` alongside
its `original.pt`. Before importing the evaluator, it calls
`with_original_manifest.manifest_pin` and verifies the checkpoint SHA256,
canonical training-data pin, validation-split seed, common source hash,
Original-training source hash, and train-only provenance. These checks use
CPU file operations. Freeze does not import Torch or load model tensors.

For each independent Original, prepare the usual readable candidate manifest.
Use the candidate checkpoints trained from that Original, the actual fixed
seed/split manifests, and the corresponding paired Retrain references. The
candidate schema and timing fields are documented in
[evaluate_selected.md](evaluate_selected.md). Keep one frozen manifest and
output directory per Original; a manifest can contain all its prespecified
seed/split runs and baselines.

```bash
python research/unlearning_matching/evaluate_independent.py \
  --original-manifest /root/study/out/independent_originals/seed142/original.json \
  --candidate-manifest selected142.readable.json --freeze selected142.frozen.json

python research/unlearning_matching/evaluate_independent.py \
  --original-manifest /root/study/out/independent_originals/seed142/original.json \
  --candidate-manifest selected142.frozen.json --phase audit \
  --data-dir /root/study/data --out-dir /root/study/final142 --gpu 0 --threads 2

python research/unlearning_matching/evaluate_independent.py \
  --original-manifest /root/study/out/independent_originals/seed142/original.json \
  --candidate-manifest selected142.frozen.json --phase test \
  --data-dir /root/study/data --out-dir /root/study/final142 --gpu 0 --threads 2
```

`--original` is injected from the Original manifest. An explicit `--original`
must resolve to exactly that checkpoint; mismatches and duplicate options
are rejected before evaluation. The wrapper extends frozen source artifacts
with its own source, the pin helper, the Original-training source, the Original
manifest and checkpoint. The same binding is checked during freeze, audit and
test. A different Original manifest, changed weights, or changed wrapper
cannot reuse an existing audit receipt.

At audit/test runtime the original evaluator first creates its own common
module. The wrapper then logs the previous and replacement SHA256 values and
replaces only that process-local module's CIFAR-10 Original pin. Dataset pins
remain enforced, and the canonical 45,000 training-member indices are checked
against the Original manifest. No global file, default pin, training runner,
or completed evaluator source is changed.

All no-overwrite, audit-before-test, test-data isolation, manifest identity,
paired metrics and sample-SD rules remain those of `evaluate_selected.py`.
This wrapper adds provenance binding; it does not select a model or relax
the validation/test boundary.
