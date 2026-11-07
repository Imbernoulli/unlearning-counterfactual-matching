# Independent Original confirmation

These scripts train additional Original models for a final confirmation of a
previously frozen unlearning method. They do not change the benchmark, its
frozen common module, or the old runners. Independent Original seeds cover an
additional source of training variation; the old three deletion seeds shared
one Original and did not do so.

Freeze the method, coefficients, allowed updates, budgets and stopping rule
before using the new Originals. Confirmation outcomes must not trigger
parameter or method selection on these models.

## Train-only Original training

`train_original.py` lazily reads only `x_train` and `y_train` from the HF-order
`cifar10.npz`, verifies `common.SHA256_TRAIN["cifar10"]`, and uses the canonical
45,000 training members returned by `common.split_indices(...)[0]`.
The fixed validation membership is defined by `common.SPLIT_SEED=1` and does
not change with either Original-training seed or deletion seed. No validation
predictions, official test arrays, prior weights, Retrain checkpoints or
experiment results are loaded.

The recipe matches `run_experiments.reference`: CIFAR ResNet-18, SGD with
learning rate .1, momentum .9, weight decay .0005, batch size 256, 182 epochs,
and learning-rate drops by .1 after epochs 91 and 136. Augmentation is the
common GPU random crop and horizontal flip. `--amp` uses BF16 autocast,
FP32 cross-entropy/model parameters, channels-last, and the same accelerated
cuDNN settings as reference training. This acceleration is not guaranteed
bitwise deterministic even with a fixed seed. Logs are emitted every five
epochs and at the fixed final epoch; no validation chooses a checkpoint.

After the existing Daytona GPU/owner/resource preflight, examples are:

```bash
python -u research/unlearning_matching/train_original.py \
  --data-dir data --out-dir out/independent_originals/seed142 \
  --seed 142 --gpu 0 --threads 2 --amp

python -u research/unlearning_matching/train_original.py \
  --data-dir data --out-dir out/independent_originals/seed242 \
  --seed 242 --gpu 1 --threads 2 --amp
```

Use the existing detached controller and exclusive GPU checks to launch these
commands. The trainer itself does not allocate, queue, or select a free GPU.
There is no CPU-training fallback.

Each output directory contains only two final scientific artifacts:

- `original.pt`: final, CPU-contiguous state_dict in the common architecture.
- `original.json`: explicit `original_sha256`, training seed, array-data pin,
  sorted absolute training-membership SHA256 (little-endian int64 encoding),
  complete recipe, source hashes, Python/NumPy/PyTorch/CUDA versions and
  synchronized training-loop time. The time excludes data/model setup and
  final serialization.

Both artifacts are published atomically per file without replacing an
existing name. A complete pair with identical configuration and verified
checkpoint hash is skipped. A configuration mismatch, corrupt checkpoint,
partial pair, or nonfinite final state is an error. The per-directory lock
prevents concurrent writers; interrupted runs should use a fresh directory
rather than silently remove stale state or overwrite weights. No intermediate
model or optimizer checkpoints are saved.

## Explicit process-local pin wrapper

`with_original_manifest.py` is the minimal adapter for an unchanged runner.
It requires the explicit checkpoint SHA256 in `original.json` and verifies
the actual file, internal provenance consistency, training-data pin,
canonical split seed, and frozen common source hash. Only then does it set
the imported common object's `SHA256_ORIGINAL["cifar10"]` for this process.

The existing runner still executes its checkpoint hash check normally. Its
ordinary registration function reads the overridden constant and records the
new actual SHA256. No training-data or checkpoint check is disabled, and no
source file or global environment pin is edited. Wrapper logs contain the
manifest, wrapper and runner source hashes and both the old and new pins.

For example, prepare a **new** base run directory for Original seed 142:

```bash
python -u research/unlearning_matching/with_original_manifest.py \
  --manifest out/independent_originals/seed142/original.json \
  --runner run_experiments.py -- \
  --phase prepare --data-dir data --out-dir out/confirmation/orig142/base \
  --seed 42 --split-seed 42 --threads 2 --amp
```

The wrapper also supports `fixed_quantile.py` and `calibration_study.py`,
whose `common`, `base.common` and `fixed.common` aliases share the same module
instance. Supply the runner's ordinary flags after `--`; omit `--original`,
which is resolved exclusively from the verified manifest. Use new output
directories for each independent Original, even when the deletion seed is the
same. Existing registrations correctly reject a changed Original pin.

This wrapper deliberately does **not** support `evaluate_selected.py`: that
program creates a fresh common module inside `runtime()`, after import.
Final audit/test evaluation needs its own explicit manifest-aware adaptation
at that point, with the Original pin and manifest also frozen as evidence.
Patching some earlier imported common instance would not change that later
instance. Do not alter the frozen common file to work around this distinction.

## CPU checks completed

Checks used synthetic fixtures only, not scientific data or result files.
The trainer passed CLI/compile checks, canonical 45,000-member selection
independence from deletion seed, train-array-only lazy reads, rejection of a
wrong data hash, a ResNet-18 CPU forward/backward/SGD step, exact 91/136
learning-rate milestones, final state_dict/manifest serialization and hash
verification, and rejection of configuration changes, partial/corrupt
artifacts, attempted overwrite and nonfinite weights.

The wrapper passed explicit/missing/wrong SHA256 checks. CPU calls to the
registration function of **each of the three supported runners** recorded
the overridden checkpoint SHA256 and retained the pinned train-data SHA256.
The common source hash was unchanged before/after the checks.

Both check processes asserted `torch.cuda.is_initialized() == False` at the
end. Temporary CPU artifacts were removed. Full training and CUDA throughput
have not been checked by this implementation task.
