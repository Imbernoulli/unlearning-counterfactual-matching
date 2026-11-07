# 固定分位数后的优化动力学与无 test 选模

日期：2026-11-07 UTC。本文只读了 runner/evaluator 源码，没有读取新的实验结果、test 指标或 Retrain 预测，没有启动 GPU。下文是实现建议与局限，不是已验证的提升。

源码依据：`research/unlearning_matching/fixed_quantile.py` 的 `train`、`fixed_targets`；`run_experiments.py` 的 `candidate`、`logodds_margin`；`tasks/security-machine-unlearning/scripts/evaluate.py` 的 `svc_mia_efficacy`。相关数学来源见同目录 `mathematical_connections.md`。本文的局部推导不依赖新文献的未经核实结论。

## 1. 现在的系数不等于整体目标中的系数

当前每一轮先遍历 4,500 个 forget 点，再遍历 40,500 个 retain 点，batch size 都是 256。因此有 18 个 forget steps，随后有 159 个 retain steps。前者的均值 loss 乘 0.1，后者为 retain CE 加 0.1 倍 Original-teacher KL。

忽略参数沿途移动、momentum、clip、augmentation/mode 差异的一阶近似，整轮位移正比于

\[
-\eta\left(18\times0.1\,\nabla L_f
+159\,\nabla L_r+159\times0.1\,\nabla A_r\right).
\]

除以 159，forget 的相对权重约为 **0.01132**，不是 0.1。这里只比较各 minibatch 均值梯度的曝光次数；不同损失的梯度范数不同，clip 也会改变实际权重，不能据此声称 forget 的实际影响一定小 88 倍。由于两个阶段的梯度不在同一参数与同一模式下求值，它甚至不严格等价于一个固定联合目标。

这也解释了为什么简单地对比最后的 forget loss 不够：日志中的 forget loss 是在 retain 段之前测到的，不一定描述最终保存的模型。至少应在同一 clean/eval 模式记录每轮 forget 段之前、之后、retain 段之后的分布差异与 utility；这是诊断，不用 test。

## 2. 为什么 retain 段可能“修回去”

假设局部 retain 目标在某点附近可写为

\[
R(\delta)\simeq R(0)+g_r^T\delta+\tfrac12\delta^TH_r\delta.
\]

先忽略 momentum。如果 forget 段引入位移 \(d_f\)，后续 \(N_r\) 次 retain 更新对这部分位移的线性响应是

\[
d_{\rm after}=(I-\eta H_r)^{N_r}d_f.
\]

在局部正曲率方向，若步长稳定，位移按 \((1-\eta h)^{N_r}\) 衰减。该式只是局部二次近似；它不说明所有方向都会衰减，也不说明当前网络实际衰减多少。

Original anchor 会加强其中一部分回复力。令

\[
A_r(\theta)=T^2\,\mathbb E_{x\in R}
D_{\rm KL}\big(p_{\theta_0}^T(x)\|p_\theta^T(x)\big).
\]

在 \(\theta_0\) 附近，其一阶项为零，二阶项是输出 softmax Fisher 经 logit Jacobian 拉回的半正定二次型。温度外的 \(T^2\) 抵消 logit 导数中的 \(1/T^2\)，但 Fisher 本身仍依赖温度。它惩罚的是相对 Original 的输出变化；如果真正 Retrain 需要的变化与之重叠，也会被惩罚。因此 anchor 不是无条件正确的目标，更不是 Retrain 的等价替代。

momentum 不会自动保留早期 forget 推力：只考察历史动量本身，在 159 步后其直接系数是 \(0.9^{159}\)，约 \(5.3\times10^{-8}\)。实际参数响应仍需与 Hessian 动力学共同分析，不能把这个数当成遗忘效果的衰减率。

此外还有独立的状态变量：forget 段用 `eval()`，retain 段用 `train()`。BatchNorm running statistics 只在 retain 段改变，最终 clean/eval 预测同时受参数和 buffers 影响。即使所有可学习参数不变，改变 BN buffers 也能改变 forget confidence。它必须与“梯度被修回”分开诊断。

## 3. 三个足够小的改动

### A. 联合 microbatch，让权重含义清楚

固定逐样本 targets 后，不再需要在一个 forget minibatch 内估计/排序分布。可以在每个 retain step 取 256 个 retain 点和约 32 个 forget 点，计算

\[
\widehat L=\frac1{B_r}\sum_{i\in B_r}\ell_{r,i}
+\lambda\frac1{B_f}\sum_{j\in B_f}\ell_{f,j}
+\beta\widehat A_r.
\]

在均匀采样下，它的梯度是 \(L_r+\lambda L_f+\beta A_r\) 的无偏 minibatch 梯度，和两个 batch 的样本数无关。159 步、forget batch 32 大约处理 5,088 个 forget views，而旧一轮处理 4,500 个；整轮增加约 588 张图像，约占原来 45,000 张的 1.3%，但会增加小 batch 的 kernel/forward 开销，运行时间仍需实测。

这同时消除“先做完 forget，再用 159 步 retain 收尾”的固定顺序偏置。它也**增加了有效 forget 权重**：相同的 \(\lambda=0.1\) 不能被描述为仅改变更新顺序。若要做纯 schedule 消融，应把联合系数设成旧的一阶权重 \(0.1\times18/159\)，或另列一个预算匹配的对照；若本轮目标是找可用的简洁改进，就清楚披露两者一起改变。

另一种 256+256 的 joint 更新更容易实现，但在相同 step 数下会增加很多 forget views；只用约 18 个 joint steps 又会少看很多 retain 点。方法、step 数、域内均值和总 view 数均应记录，不能仅写“10 epochs”就当作同预算。

### B. 所有更新使用固定 BN buffers

对于当前无 dropout 的 ResNet，训练阶段始终 `model.eval()` 是一个最小实现：**它不关闭 autograd，也不冻结参数**。卷积、线性层及 BN affine 参数仍能更新，只有 BN running mean/variance/count 保持 Original 值。retain augmentation 可以继续使用。teacher 同样保持 eval。

这样 target 构造、forget 更新与 clean 选模使用相同的 BN 语义，也去掉 retain 特有的 buffers 更新。实现应断言所有 BN buffers 未改变，并记录哪些参数允许更新。若模型包含其他依赖 train/eval 的层，应逐层说明，不能泛化这个等价说法。

固定 Original BN 也有代价：权重改变后，旧统计量可能不再适合新特征。utility 校验可发现明显损伤，不能事先保证它优于重估 BN。因此这是一个干净的、应披露的设计选择，不是理论保证。

另一个需要区分的因素是 loss 几何。非真类条件比例固定时，令当前 true-class log-odds 为 \(s\)，目标概率为 \(u=\sigma(s_*)\)，forward KL 在这个方向上就是 Bernoulli cross-entropy 加常数：

\[
\frac{\partial\ell_{\rm KL}}{\partial s}=\sigma(s)-u,
\qquad
\frac{\partial^2\ell_{\rm KL}}{\partial s^2}=\sigma(s)(1-\sigma(s)).
\]

在目标附近，KL 对 score 差的二阶权重是 \(u(1-u)/2\)；log-odds MSE 的权重则恒定。若 \(s\gg s_*\gg0\)，KL 导数接近 \(1-u\)，即使 log-odds 差很大也很小；MSE 导数 \(2(s-s_*)\) 则会继续增大。因此 KL 的最小改动性质和 MSE 的高置信尾部移动能力不应混为一谈。它可以解释一种可能的优化不足，但没有新结果证明当前 MIA 残差确由它造成。更不能只因 MSE 梯度大就认定其更好：共享参数的 utility 损伤也可能更大，应由同一独立 selection 准则约束。

### C. 用“效用约束下达到非成员行为”选择停止点

不能只最小化 forget 的 target-fitting loss：过拟合有限 targets、整体置信度下降、共享参数损伤，都可能使训练 loss 更好但实际匹配更差。下面给出只用已有 5,000 validation 点的完整准则。它不引入新网络，也不需要 Retrain。

## 4. 锁定 selection/audit，而不是反复看所有 validation

已存在的 2,500 target 点继续只用于构造固定 targets。将余下 2,500 calibration 点按类、按固定 seed 分成 selection 1,250 与 audit 1,250，写入 manifest。

- **target：**构造目标；也可用于下述 surrogate attacker 的非成员训练集。
- **selection：**只在这里选择方法/epoch，不在这里拟合 attacker。
- **audit：**选定最终模型后只计算一次并报告，不据此重选。audit 不是第二个调参集。

训练前锁定候选方法、epoch/checkpoint 集、所有系数、utility 容差、tie-break 和评分公式；保留 Original 作为零更新候选。如果 audit 不佳，就如实报告；后续新一轮实验必须另记其数据重用情况，不能称同一 audit 仍是完全独立。

### 4.1 两条硬效用约束

以下都是百分点，基线是同一 split 的 Original：

\[
\operatorname{Acc}_R(\theta)\ge\operatorname{Acc}_R(\theta_0)-0.5,
\qquad
\operatorname{Acc}_S(\theta)\ge\operatorname{Acc}_S(\theta_0)-1.0.
\]

这两个阈值是本轮预先指定的工程容差，不是统计显著性界。retain 取全部 40,500 个 clean 点；selection 取独立的 1,250 点。应同时报告成对改变数与抽样不确定性。1,250 点上 1 pp 仅约 12–13 个样本，不能把小差异解读成稳定的科学提升。

### 4.2 最简四项 surrogate

令 \(e_D\) 为错误率、\(a_D\) 为准确率，全部以百分数表示。定义

\[
U=|e_F(\theta)-e_S(\theta)|,\quad
R=[a_R(\theta_0)-a_R(\theta)]_+,\quad
T=[a_S(\theta_0)-a_S(\theta)]_+.
\]

UA 的参照是当前模型在独立新数据上的错误率。在 iid 数据、随机删除、相同类别权重的理想条件下，真正只由 retain 训练的模型在 forget 与新数据上应具有相同的预测分布。这里的条件并不自动成立于选择性删除、分布偏移或逐样本记忆残留。

RA/TA 项保护 utility，但用 Original 近似 counterfactual utility，并不知道真正 Retrain 的下降量。所以这三个量与下述 M 都是**validation surrogate，不是实际 AvgGap 的无偏估计**。

为模拟最终 confidence attacker，在每个 checkpoint 上用当前模型的下列 clean true-class probabilities 拟合 \(h_\theta\)：固定、按类平衡的 retain 2,500 点作为 members，以及 target 2,500 点作为 nonmembers。固定同一 subset，使用与最终评测相同的 RBF SVC，`C=3, gamma="auto"`，并限制 CPU 线程。selection/audit 均不得出现在 attacker 的 fit 中。target 参与过目标构造，故这仍不是独立最终攻击评估，只是一个固定验证算法。

令 \(r_D(\theta)\) 为 \(h_\theta\) 将集合 \(D\) 判为 nonmember 的比例（百分数）。最简 matching 项为

\[
M_{\rm cohort}=|r_F(\theta)-r_S(\theta)|.
\]

目标不是把 forget 的 nonmember rate 推到 100%，而是使之与同模型真正未训练数据一致。固定攻击器时，任何有界攻击的期望差都受两个输入分布的 total variation 控制；当前规则直接检测所关心的一个攻击决策区域，不声称控制所有攻击。

在满足硬约束的候选中，选择

\[
\arg\min_\theta \frac{U+R+T+M_{\rm cohort}}4,
\]

数值并列时选最早 checkpoint，然后按预先写好的方法顺序。若无更新候选可行，就回退 Original。不能在查看 test 后改用另一个 proxy、epoch 或容差。

### 4.3 攻击器重拟合的剩余问题与可选保守版

同模型 cohort matching 不会控制攻击器本身的变化。令 \(r_S(\theta_0)\) 使用 Original confidence 和同样的固定 attack-fit subsets 得到，则

\[
|r_F(\theta)-r_S(\theta_0)|
\le |r_F(\theta)-r_S(\theta)|
+|r_S(\theta)-r_S(\theta_0)|.
\]

右侧第二项能发现“整个非成员判决率一起漂移”的情况。建议至少记录它。如果希望保守地保护 Original 的非成员参考水平，可以**在任何新 selection/test 观察之前**将 M 固定为右侧两项之和，再用同一四项公式。它控制一个参考攻击率差的上界，不控制未知 Retrain 攻击率；也可能对合理的 Retrain 式漂移过于保守。

本轮应锁定最简版或保守版中的一个，不能跑完后择优汇报。最终 attacker 使用 10,000 retain + 10,000 test，而上述验证 attacker 是 2,500 + 2,500；SVC 的样本量、边界稳定性和 target 数据重用都限制其可迁移性。

### 4.4 便宜的诊断，不再增加选模自由度

额外记录 class-conditional confidence/log-odds KS、forget/selection error、clean retain CE、每阶段 target-fitting loss、BN buffer 差异。不要把均值 confidence 接近当成分布接近。KS 只直接控制阈值事件，RBF SVC 在一维上也可能有多个决策区间，因此不能把小 KS 自动解释为所有攻击成功率接近。

最终一次 audit 使用已经锁定的模型与公式；可报告同样的 utility/cohort statistics 与置信区间。最终真正 test/Retrain 评估仍只用于结果报告，不用于本轮选择。

## 5. 暂不扩张的方法与结论边界

scalar I-projection 保持非真类条件分布，故精确匹配真类 confidence 仍不保证 UA。对应的两组 logits 可具有同一 true-class probability，却有相反正确性；证明与例子见 `mathematical_connections.md`。本轮 targets agent 正在处理同 donor 的完整概率目标，应先把这一条与优化/选模因素分清，不再扩张新的模块或大网格。

一个以后可用的精确推导是：同时给定目标真类概率 \(u\) 和 signed margin \(m\)，非真类条件最大概率必须为 \(\alpha=\exp(\operatorname{logit}(u)-m)\)。在可行范围 \([1/(K-1),1]\) 内，可以用条件分布的 capped-KL projection 同时满足二者，而不是独立拟合两个不相容的统计量。本轮不建议增加该候选；同 donor 的概率向量更直接，且已有实现任务覆盖。

这些建议解决的是优化目标、状态变量和验证协议的可解释性。它们不证明删除了所有成员信息，也不证明能逼近某个具体 Retrain 的逐样本输出。最终要检验的简洁假说是：**固定全局 targets 后，合理的域均值权重、同一 BN 语义以及受 utility 约束的停止点，是否比固定的 forget-then-retain 收尾更接近独立非成员行为。**
