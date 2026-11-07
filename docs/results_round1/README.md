# Completed first-round paired experiments

These are new CIFAR-10/ResNet-18 measurements on two Daytona H200 GPUs.
Three deletion/retraining runs (42, 43, 44) share one Original. Each uses an
independently initialized full 182-epoch Retrain on the identical retain split.
The six raw JSON files retain all four metrics and per-run gaps. Each recorded
gap was independently recomputed from the raw metrics. The summary uses sample
standard deviations, not standard errors or confidence intervals.

| Method | UA | RA | TA | MIA | AvgGap | Update seconds |
|---|---:|---:|---:|---:|---:|---:|
| Original | 0.00 | 100.00 | 95.01 | 0.23 | 4.668 ± 0.064 | 0.0 |
| Gradient ascent | 0.00 | 100.00 | 95.05 | 0.21 | 4.681 ± 0.027 | 1.3 |
| Fine-tuning | 7.31 | 95.84 | 89.57 | 12.95 | 3.051 ± 1.230 | 22.2 |
| L1-sparse FT | 6.61 | 96.34 | 89.95 | 12.38 | 2.547 ± 0.415 | 22.5 |
| Random labelling | 2.10 | 99.81 | 94.09 | 13.56 | 1.053 ± 0.182 | 24.5 |
| SalUn | 2.73 | 99.49 | 93.72 | 12.68 | 0.970 ± 0.109 | 26.4 |
| Mean-target control | 0.10 | 99.98 | 94.69 | 1.63 | 4.218 ± 0.068 | 28.3 |
| Log-odds transport | 3.29 | 100.00 | 94.91 | 6.80 | 2.179 ± 0.154 | 29.1 |
| Log-odds + margin | 3.33 | 100.00 | 94.96 | 7.07 | 2.115 ± 0.094 | 29.9 |
| Global quantile + I-projection | 0.38 | 100.00 | 94.98 | 3.29 | 3.800 ± 0.034 | 28.3 |
| Global quantile + log-odds MSE | 3.87 | 100.00 | 94.88 | 8.11 | 1.697 ± 0.050 | 28.0 |
| Paired Retrain | 5.21 | 100.00 | 94.28 | 12.96 | 0.000 ± 0.000 | 171.7 |

Global quantile log-odds MSE improves on this study's minibatch log-odds loss
in all three paired runs (mean AvgGap reduction 0.482, 22.11%). It still falls
behind RL and SalUn. The global and minibatch implementations change target
assignment, fixed sample pairing, and class weighting; this does not isolate
a particular bias mechanism. The I-projection result illustrates a gap between
output-space optimality and successful shared-network fitting at fixed scale
and budget. It does not refute the projection derivation.

For global MSE, mean absolute UA/RA/TA/MIA gaps are respectively
1.3333 / 0.0033 / 0.6000 / 4.8519 percentage points. For SalUn they are
2.4815 / 0.5078 / 0.5633 / 0.3259. Remaining membership behavior is the primary
weakness of the new method in this protocol.

Times cover update/training loops; target generation, evaluation, environment
setup, and artifact transfer are excluded. New variants use extra held-out
information and a different update scaffold from the inherited baselines.
Historical baseline settings were selected with Retrain gaps; new variants
used fixed settings. Calibration was reserved in this round. These metrics
do not establish certified deletion or resistance to every membership attack.

The raw results are frozen. A separate three-hour calibration-only optimization
was registered in `research/unlearning_matching/optimization_3h_20261106.md`.
