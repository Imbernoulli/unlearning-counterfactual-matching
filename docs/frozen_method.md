# Frozen method: class-conditional joint-score energy matching

This document defines the method frozen in [selected_method_20261106.json](../../../../research/unlearning_matching/selected_method_20261106.json), commit `007a91b6cf`, before audit or new test inspection. It adds no implementation or performance claim. The selected entrypoint is [energy_study.py](../../../../research/unlearning_matching/energy_study.py), SHA256 `63d0de85deca05542f56e7e2e1ba0f709f8b2fb0791c0a627fb71ef6a2235c7c`, around the unchanged [calibration_study.py](../../../../research/unlearning_matching/calibration_study.py), SHA256 `b25cf925c71e81932023084179cdc5910bb135db3812214b84a9fc8a610f4e7b`.

## Data and fixed reference

Start from Original ResNet-18, trained on 45,000 CIFAR-10 images. Random deletion partitions those images into 4,500 forget points F and 40,500 retain points R. A fixed class-stratified 5,000-image validation set was excluded from Original training. Its disjoint roles are target V (2,500; 250 per class), selection (1,250), and audit (1,250). Training and target construction use neither selection nor audit predictions, official test data, nor Retrain predictions.

For logits z and true class y, use the two-dimensional score

\[
T(z,y)=(s,m),\qquad
s=z_y-\log\sum_{k\ne y}e^{z_k},\quad
m=z_y-\max_{k\ne y}z_k.
\]

Thus sigmoid(s) is the true-class probability and, apart from ties, m<0 denotes an error. There is no coordinate standardization. Compute and freeze Y_c={T(f_Original(v),y_v):v∈V,y_v=c} once, with clean images and Original in eval mode. Each Y_c contains 250 vectors. The reference is a distribution, not a learned target module; there is no rank/donor/identity assignment between forget and validation examples.

## Loss and exact update schedule

For the current clean forget minibatch, let X_c={T(f_θ(x),y):y=c}, n_c=|X_c|, and m_c=|Y_c|. For each class with n_c≥2, compute the ordinary energy U-statistic using **unsquared Euclidean distance**:

\[
\widehat{\mathcal E}_{U,c}
=\frac{2}{n_cm_c}\sum_{i,j}\|X_{c,i}-Y_{c,j}\|_2
-\frac{1}{n_c(n_c-1)}\sum_{i\ne k}\|X_{c,i}-X_{c,k}\|_2
-\frac{1}{m_c(m_c-1)}\sum_{j\ne\ell}\|Y_{c,j}-Y_{c,\ell}\|_2.
\]

The self sums use ordered off-diagonal pairs. The target self term is precomputed and has zero gradient. The forget loss is the macro-average over current classes with n_c≥2; singleton/absent classes are omitted. If none qualify, its gradient is zero. Finite-sample U-statistics may be negative.

Use schedule C: at each epoch independently shuffle F and R without replacement, split each into batches of at most 256, and traverse all 159 retain batches. Place all 18 forget batches at zero-indexed retain positions

\[
\mathcal P=\{\operatorname{round}(158j/17):j=0,\ldots,17\}
=\{0,9,19,28,37,46,56,65,74,84,93,102,112,121,130,139,149,158\}.
\]

Forget batches use clean images. Retain batches use RandomCrop(32,padding=4) and RandomHorizontalFlip. Both forwards use eval mode: Original BatchNorm running mean, variance and batch count stay fixed; all learnable parameters, including BN affine parameters, remain trainable in FP32. Retain loss is mean supervised CE. The Original KL anchor has **effective weight zero**.

The actual gradient at step t is

\[
h_t=\nabla_\theta\operatorname{CE}_{R_t}^{\rm aug}(\theta_t)
+\mathbf1_{\{t\in\mathcal P\}}\operatorname{clip}_5
\!\left(\alpha\nabla_\theta\widehat{\mathcal E}_{U,F_{j(t)}}(\theta_t;Y)\right),
\quad\alpha=1,
\]
\[
b_t=.9b_{t-1}+h_t+5\!\times\!10^{-4}\theta_t,\qquad
\theta_{t+1}=\theta_t-.003b_t,\qquad b_{-1}=0.
\]

Here clip_5 is the inherited global parameter-gradient norm clipping (`torch.nn.utils.clip_grad_norm_`, max norm 5, default numerical epsilon); only the weighted forget gradient is clipped, **before** the retain gradient is added. The final sum is not clipped. This is ordinary PyTorch SGD with momentum .9, weight decay 5e-4, no Nesterov, and no dampening. The same momentum buffer persists across steps and epochs. At the other 141 retain positions there is no forget gradient; this is not a uniform CE+α·energy gradient at every step. Batch losses are separately normalized and there is no extra ratio-of-dataset-sizes correction.

Run exactly **23 epochs × 159 optimizer steps = 3,657 steps**. Each epoch processes exactly 4,500 forget and 40,500 retain examples, for totals 103,500 and 931,500 respectively. There are 414 forget backwards. The reference-target forward pass and selection diagnostics are additional costs, not counted as optimizer steps. This 23-pass method exceeds the old task's ten-retain-pass update budget and must be reported separately from budget-matched ten-pass comparisons.

## Core pseudocode

```text
θ ← Original; model.eval(); keep all learnable parameters trainable
Y[c] ← frozen clean/eval Original scores (s,m) on target-validation class c
opt ← SGD(θ, lr=.003, momentum=.9, weight_decay=5e-4)
P ← round(linspace(0,158,18)); α ← 1
for epoch = 1,…,23:
    F_batches, R_batches ← independent shuffled complete passes, batch size≤256
    for t, R_batch in enumerate(R_batches):
        opt.zero_grad(set_to_none=True)
        if t ∈ P:
            backward(α × macro_class_energy_U(clean F_batches[index_P(t)], Y))
            clip_global_parameter_gradient_norm(max_norm=5)
        backward(mean_CE(model(augment(R_batch)), retain_labels)); opt.step()
return final θ after epoch 23                         # fixed last.pt on new runs
```

## Selection history, assumptions, and limits

Epoch 23 and the complete configuration were chosen on discovery seed/deletion seed 42 using the prespecified feasible-set selection rule after 58 completed scientific trials (including smoke); two other launch attempts failed in infrastructure. That selected run originally traversed 30 epochs. The frozen deployment runs only 23 and returns final `last.pt` on every new confirmation seed; it does not reselect epochs on their calibration, audit, or test predictions. The discovery selection J=.052222 is a development statistic, not final performance or a privacy guarantee.

The scientific proxy assumes random deletion from the same class-conditional population: a retained-only model should treat forget inputs like independent nonmembers. Frozen Original held-out behavior only approximates this counterfactual; Original used more training examples. Joint (s,m) preserves a real confidence–error relationship while allowing exchange of same-class target roles, but two output statistics do not identify individual Retrain outputs or internal memory.

Energy distance and its relation to distance-induced-kernel MMD are established mathematics. This is an application of that existing distribution-matching route, not a claim to first energy-based unlearning, a new distance, or certified deletion; see [energy_prior_art.md](energy_prior_art.md) and [energy_geometry.md](energy_geometry.md). With finite first moments, population energy identifies equality of the compared score distributions. The actual U-estimator, without-replacement training, random class counts, singleton omission, shared network parameters and optimizer do not imply exact population matching or monotone improvement in KS/AvgGap. Nonzero-distance derivatives use the usual norm gradient; the implementation uses PyTorch's zero derivative at exact coincident pairs.

Selection used only 1,250 images and many adaptively designed trials. Its best value is optimistic development evidence, not an independent replicate; target/audit separation does not undo selection reuse. Independent audit/Original/deletion-seed confirmation, all registered failures, four raw metrics and complete costs remain necessary. No new final outcome is asserted here.
