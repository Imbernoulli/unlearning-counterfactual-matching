# CIFAR-10 architecture comparison

This experiment uses deletion split 43 and classifier/unlearning seed 43.
The verified CIFAR-10 order, fixed validation membership, forget and retain
indices, and evaluator are shared between ResNet-18 and the small Vision
Transformer.

- The transformer has four encoder blocks, width 192, three attention heads,
  4-pixel patches, a class token, learned position embeddings, and about
  1.21 million parameters. Its Original and paired Retrain use the same 182-epoch
  AdamW recipe with five epochs of warmup and cosine learning-rate decay.
- ResNet-18 uses the original 182-epoch SGD recipe. Both architectures use
  BF16 autocast for full classifier training, 256-image batches, and the same
  crop/flip augmentation. Update and evaluation use the existing FP32 paths.
- A2 uses the frozen ResNet-18 development correction from split 42, with the
  original offset, step interval, gradient cap, and final decay. SalUn uses the
  original CIFAR-10 settings. This measures direct transfer of both updates at
  their existing settings. Original and Retrain use the exact training and
  retain sets in the split manifest, respectively.
- A2 records target construction and update-loop time separately. SalUn's time
  includes its saliency pass. The official test split is loaded in evaluation
  after all checkpoints have been produced. With one deletion split, report
  paired point estimates.

From the repository root, prepare CIFAR-10 from the train and test parquet
files of the `uoft-cs/cifar10` dataset on Hugging Face. Both decoded hashes
must match the recorded project hashes.

```sh
python research/unlearning_matching/prepare_cifar10.py \
  --train TRAIN.parquet --test TEST.parquet --out DATA/cifar10.npz
```

Run the five phases for each architecture with a CUDA-enabled PyTorch
environment and scikit-learn 1.5.2. Use separate output folders. For a given
architecture, Original and Retrain can run in parallel after `prepare`.
A2 and SalUn read Original and evaluation reads all four checkpoints.

```sh
python research/unlearning_matching/architecture_study.py PHASE \
  --architecture vit-tiny --data-dir DATA --run-dir OUTPUT
```

`PHASE` is `prepare`, `original`, `retrain`, `a2`, `salun`, or
`evaluate`. Built configurations, split manifests, weights, timings,
checkpoint hashes, and official test metrics are saved under `OUTPUT`.
