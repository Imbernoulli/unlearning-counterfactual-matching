# Matching Retrain: independent unlearning study

This project studies random-deletion machine unlearning. It
reuses the verified CIFAR-10 Original and scientific evaluation primitives,
without modifying the benchmark or its leaderboard. The old whole-class
agent logs are not evidence for this study. Historical random-forgetting
measurements and literature sources are documented in
[the research notes](../../docs/private/reports/unlearning-progress-20261106/research_notes.md).

## Fixed protocol

- ResNet-18; Original trained on 45,000 images. Each deletion split contains
  4,500 forgotten and 40,500 retained images. The 5,000 never-trained validation
  images are stratified into 2,500 target and 2,500 reserved calibration images.
- Seeds/deletion seeds 42, 43, 44 share one Original. Each has its own full
  182-epoch retrained reference. They do not cover Original-training variance.
- Exact decoded data and Original hashes must match `common.py`. Official
  CIFAR pickle ordering differs from the HF mirror used by this checkpoint;
  silently changing ordering changes training membership.
- Baselines retain their inherited implementations/settings. New methods
  have fixed settings and use additional held-out information. Their common
  scaffold uses clean/eval forget passes, augmented/train retain passes, and
  a frozen Original retain anchor. These are disclosed differences.
- Test data are opened only in final evaluation. Neither test metrics nor
  Retrain predictions choose targets, strengths, epochs, or methods.
- Calibration is reserved, not used for early stopping in this first round.
  Update-loop times exclude target construction and are not complete
  end-to-end deployment costs.

The initial comparison contains six inherited baselines, a mean-target
control (not a ReGUn reproduction), minibatch log-odds transport, and transport
of log-odds plus margin. A separately preregistered follow-up uses fixed global
quantile targets with either minimum-KL output projection or score MSE.
See [the preregistration](preregistration_20261106.md) and the derivations in
[mathematical connections](../../docs/private/reports/unlearning-progress-20261106/mathematical_connections.md).

## Reproduction

Data preparation requires NumPy, Pillow, and PyArrow; numerical training uses
PyTorch 2.5.1/CUDA 12.1, NumPy 1.26.4, and scikit-learn 1.5.2 in the measured
Daytona image. Preparation is a build step; training requires no network.

```bash
python vendor/pkg_configs/unlearn-lab/prepare_data.py --only cifar10 --out DATA
```

With the verified Original supplied and an exclusively allocated GPU, run
the following separately for each seed S. Set OMP/MKL/OpenBLAS threads to 2.
The `--amp` flag accelerates only reference training; update/evaluation remain
FP32. Use identical options in each phase so configuration checks pass.

```bash
python research/unlearning_matching/run_experiments.py --phase all \
  --data-dir DATA --original ORIGINAL.pt --out-dir RUNS \
  --seed S --split-seed S --gpu GPU --threads 2 --amp

python research/unlearning_matching/fixed_quantile.py --phase all --loss both \
  --data-dir DATA --original ORIGINAL.pt --base-runs-dir RUNS \
  --out-dir FOLLOWUP --seed S --split-seed S --gpu GPU --threads 2

python research/unlearning_matching/summarize_results.py \
  --runs-dir RUNS --followup-dir FOLLOWUP --out-dir SUMMARY
```

For Daytona allocation, `daytona_run.py` uses the existing local key-file
reader and makes one owned two-GPU allocation, with artifact collection and
deletion in `finally`. It labels its exact sandbox and never reaps other
sessions. The first deployment's continuation controls and all uploaded
source hashes are retained with the experiment evidence. No API keys are
part of the source or report.

`splits.json`, `run_config.json`, follow-up `preregister.json`, per-method
metadata/checkpoints, logs, and final raw metrics are retained. Aggregation
checks compatibility and paired references, computes gaps per run before
averaging, and reports sample SD. Include failures and avoid selecting the
best-looking test configuration as if it had been prespecified.

These measurements test behavior against a paired Retrain. They are not a
certified deletion guarantee or a complete membership-privacy evaluation.
