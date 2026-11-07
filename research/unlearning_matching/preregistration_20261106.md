# Retrain-matching study: fixed experiments and provenance

## Initial experiment

The initial runner was frozen at commit `f199c26486` before deployment to one
Daytona sandbox with two H200 GPUs. All methods share the hash-verified Original
checkpoint and, within each run, the same deletion split. Three runs use seeds
and deletion seeds 42, 43, and 44; they do not vary the Original training seed.
The reference is independent 182-epoch retraining on the retained examples,
using the code's SGD/multistep recipe and BF16 acceleration. Evaluation is FP32.

The six inherited baselines use the existing source and hyperparameters.
The new comparison has a shared mean-target control (not an exact ReGUn
reproduction), class-conditional minibatch log-odds transport, and transport of
both log-odds and signed margin. New methods use 10 forget/retain passes,
SGD learning rate .003, momentum .9, weight decay .0005, forget weight .1, and
retain CE plus .1 times a fixed-Original KL anchor at temperature 2. Forget
updates use clean views and evaluation mode; retain updates use augmentation
and training mode. These differences from the historical baselines must be
disclosed. All new methods use the same scaffold.

The 5,000 never-trained validation examples are split classwise into 2,500
target and 2,500 calibration examples. Calibration is reserved for future
selection; this round uses fixed settings and no early stopping. Official test
data and retrained predictions do not enter an update or target. The evaluation
phase refuses to run before all initial methods and Retrain have completed.
Report all methods, including negative findings. No test-driven retuning.

## Mathematical follow-up, registered before inspecting new test comparisons

Registered on 2026-11-07 UTC during the first experiment. The independent code
review and the mathematical literature identify a concrete limitation:
minibatch OT compares approximately 25 forgotten examples per class against a
250-example target distribution; its expected loss is not the full-data OT
loss. This follow-up is motivated by that issue and the I-projection
derivation, not by selecting a favorable result from the initial experiment.

1. Evaluate the fixed Original on all forgotten examples and target-validation
   examples, without augmentation. Within each class, sort the forgotten
   true-class log-odds. Assign fixed target log-odds by averaging the validation
   empirical quantile function over the corresponding rank interval. This is
   the barycentric target of the one-dimensional empirical transport plan.
2. For target true-class probability u, project the Original distribution p
   onto q_y=u while minimizing KL(q || p). Its exact solution keeps the
   non-true conditional probabilities: q_k=(1-u)p_k/(1-p_y), k != y.
   Compute those probabilities from non-true logits for numerical stability.
3. Distill these fixed per-example targets with forward KL using the same
   10-pass, .003-learning-rate scaffold and retain anchor. A fixed log-odds
   squared-error variant is an optional prespecified ablation, not a
   configuration selected using test metrics.
4. Use the same three splits and their paired retrained models. Train before
   reading test metrics for the follow-up. Record both its configurations and
   all outcomes separately from the frozen initial experiment.

For equal-size prescribed true-class target multisets, classwise sorted
assignment followed by I-projection minimizes total output KL to Original:
KL decomposes into a Bernoulli KL and a conditional non-true KL; the only
assignment-dependent term is minus the sum of target probabilities times
Original log-odds, minimized by the rearrangement inequality. This is an
output-space statement for fixed targets. It does not prove minimal parameter
change, sample-specific counterfactual correctness, or certified deletion.

Teacher target construction is timed separately from update-loop time.
Report uncertainty across deletion/retraining runs and avoid interpreting
three runs on one Original as coverage of all training randomness.
