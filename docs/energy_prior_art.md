# Joint-score energy matching: focused prior-art check

2026-11-06；只读核对以下三篇原论文及其中两份官方代码，未新增 GPU 候选，未改已推送主文。当前 joint-energy 的改善只发生在 calibration selection 指标 \(J\)；本 memo 不读取或推断 audit/test/AvgGap 结果。有限检索不能证明“首次使用”。

**结论：应将当前方法定位为已有分布匹配路线中的一个简单设计，而非提出 energy distance 或首次用 unseen 数据指导遗忘。** A-AMU 是最直接的目标函数先例；POUR 明确把 Energy Distance 纳入表征级遗忘定义，但实际算法没有采用 forget–heldout energy loss；EUA 的 energy 则是另一种对象。

## 三篇最近邻

| 工作 | 匹配的对象与目标 | 实际距离/更新 | 与当前方法的边界 |
|---|---|---|---|
| **A-AMU**, Khan, 2025 | Forget 与类别匹配的 unseen subset 的标量分类 loss，经 log1p、smooth empirical CDF、probit 变换 | Gaussian-kernel MMD；retain 主损失平方，可叠加其它遗忘正则 | 已覆盖“forget/nonmember 分布对齐 + retain 保护”的核心思路；当前区别是 joint \(s,m\)、冻结 Original 参考、逐类匹配和 estimator |
| **POUR**, Le et al., CVPR 2026 | 理论定义比较 unlearned 与 Retrain 的特征分布；算法针对**整类**删除 | POUR-P 投影去掉该类方向；POUR-D 对同一 forget 输入做 projected-teacher 的逐样本 L2 feature distillation | 其 Energy Distance 是定义中可选 discrepancy 的例子，不是实际 forget–heldout 训练损失；本项目是随机实例删除与输出统计量匹配 |
| **EUA**, Yang et al., ICML 2026 | LLM 每个位置在整个词表上的 free energy；使 forget/retain 能量落在不同边界侧 | negative log-sum-exp logits，加 squared-hinge energy boundaries、retain CE，并用于推理拒答 | 没有使用两样本 energy distance；不是把 forget 当作正常同分布 nonmember 的目标 |

### A-AMU：纸面与官方实现应分别记录

原论文 [§3.4.1–3.4.3，式 (4)–(6)，PDF p.4](https://arxiv.org/pdf/2507.09786v3#page=4) 给出变换、Gaussian MMD 与 retain squared-loss scaffold。它把 reference 称为 class-matched test subset；我们的协议只能用独立 target-validation，不能照搬官方 test 数据的角色。

核对官方仓库 `algebraicdianuj/DC_Unlearning`，commit `6cbe1f0b410ef69ff2fa45935122f77eb5144e62`：

- [accelerated_cf.py L70–85](https://github.com/algebraicdianuj/DC_Unlearning/blob/6cbe1f0b410ef69ff2fa45935122f77eb5144e62/unlearn_methods/accelerated_cf.py#L70-L85)：Gaussian \(\sigma=1\)，三个 kernel matrix 直接 `.mean()`，self terms 包括对角线，是 V-statistic。
- [L99–119](https://github.com/algebraicdianuj/DC_Unlearning/blob/6cbe1f0b410ef69ff2fa45935122f77eb5144e62/unlearn_methods/accelerated_cf.py#L99-L119)：每个 batch 单独变换 CE；实现还在 sigmoid 前对 loss 差做 ReLU，区别于纸面式 (4) 的 signed difference。复现时应标明采用哪一版。
- [L146–178](https://github.com/algebraicdianuj/DC_Unlearning/blob/6cbe1f0b410ef69ff2fa45935122f77eb5144e62/unlearn_methods/accelerated_cf.py#L146-L178)：reference 从 forget batch 出现的类别集合中抽样后**合池**，不保证逐类数量相等；`model(test_data).detach()` 每步重算，因此 reference 是当前 student 的停止梯度分支，**并非冻结 Original**。retain 使用 batch-mean CE 的平方。[accelerated_scrub.py L225–241](https://github.com/algebraicdianuj/DC_Unlearning/blob/6cbe1f0b410ef69ff2fa45935122f77eb5144e62/unlearn_methods/accelerated_scrub.py#L225-L241) 也采用当前 student 的 detached reference，并加 teacher KL。
- [unlearn.py L114–130](https://github.com/algebraicdianuj/DC_Unlearning/blob/6cbe1f0b410ef69ff2fa45935122f77eb5144e62/unlearn.py#L114-L130)：官方脚本从其 test dataset 分出 70% 供方法使用，剩余部分评估；不能把这套数据角色直接等同于本项目的完整 official-test isolation。

### POUR：提及 energy 不等于优化 energy

[CVPR 正文 §2.1，Definition 2.1、式 (1)，PDF p.2 / proceedings p.10274](https://openaccess.thecvf.com/content/CVPR2026/papers/Le_POUR_A_Provably_Optimal_Method_for_Unlearning_Representation_via_Neural_CVPR_2026_paper.pdf#page=2) 将 MMD、Wasserstein-2、Energy Distance 并列为 unlearned/Retrain 表征分布的可选差异度。§2.2 实际以 CKA/RUS 估计表征变化。算法在 [§4.1–4.2，式 (7)–(9)、未编号 L2 loss，PDF pp.4–5 / pp.10276–10277](https://openaccess.thecvf.com/content/CVPR2026/papers/Le_POUR_A_Provably_Optimal_Method_for_Unlearning_Representation_via_Neural_CVPR_2026_paper.pdf#page=4)：

\[
P=I-\frac{w_uw_u^\top}{\|w_u\|^2},\qquad
L_{\mathrm{POUR-D}}(x)=\|\theta_s(x)-P\theta_0(x)\|_2^2,\quad x\in D_f.
\]

官方 `ale256/representation_unlearning`，commit `774ba5f885228426ec5a1b17798c62e77c901711`，核实 [methods.py L433–441](https://github.com/ale256/representation_unlearning/blob/774ba5f885228426ec5a1b17798c62e77c901711/methods.py#L433-L441) 的投影、[L519–597](https://github.com/ale256/representation_unlearning/blob/774ba5f885228426ec5a1b17798c62e77c901711/methods.py#L519-L597) 的冻结教师/forget-only loader、[L607–637](https://github.com/ale256/representation_unlearning/blob/774ba5f885228426ec5a1b17798c62e77c901711/methods.py#L607-L637) 的同输入特征目标。故它不是当前目标函数的直接实现先例；但它的表征警告适用：两个输出统计量相同，不足以保证内部记忆已删除。其最优性依赖 neural-collapse、类分布等假设，不能移植为我们的方法保证。

### EUA：log-sum-exp energy 与 energy distance 不同

[正式论文 §3.2，式 (8)–(9)，PDF p.4；§3.3，式 (12)–(13)，PDF pp.5–6](https://raw.githubusercontent.com/mlresearch/v306/main/assets/yang26dc/yang26dc.pdf#page=4) 定义 token energy 为负 logit，free energy 为

\[
E_\theta(x,y_{<i})=-T\log\sum_{v\in V}\exp(z_\theta(v\mid x,y_{<i})/T).
\]

它用能量边界分开 forget/retain，再结合 retain CE 和拒答阈值；不是两个样本群的 Euclidean distance functional。[出版页及官方代码入口](https://proceedings.mlr.press/v306/yang26dc.html)。一个直接区别：所有 logits 同加常数会改变该 free energy，却不改变当前 \(s,m\)。这是公式的代数性质，并非对其方法效果的评价。

## 当前方法可准确声称什么

本实现 [energy_study.py](../../../../research/unlearning_matching/energy_study.py) 在每个类别内比较当前 forget 与冻结 Original target-validation 的 \(X=(s,m)\)：

\[
s=z_y-\log\sum_{k\ne y}e^{z_k},\qquad m=z_y-\max_{k\ne y}z_k,
\]
\[
\mathcal E(P,Q)=2\mathbb E\|X-Y\|-\mathbb E\|X-X'\|-\mathbb E\|Y-Y'\|.
\]

它保留同一 heldout 样本的 confidence–margin 联合关系，放松固定 donor 身份配对；采用无 bandwidth 的**非平方 Euclidean**距离、off-diagonal U-statistic，有限样本值可能为负；逐类 macro-average，跳过当前 minibatch 仅一个样本的类别。retain CE、更新顺序、BN 和 optimizer 继承冻结 scaffold，不能计为新方法。冻结参考仍有 Original/nonmember 与实际 Retrain 的代理误差。

**Energy 与 MMD 也不是互斥范式。** 在一阶矩有限时，直接代入 \(k_a(x,y)=\|x-a\|+\|y-a\|-\|x-y\|\) 后，MMD 的三个期望中的 anchor terms 抵消，得到上述 \(\mathcal E\)；因此当前距离可视为距离诱导核的 squared-MMD（此处按无 \(1/2\) 的 kernel 约定）。这项代数关系解释了与 A-AMU 的共同路线，不构成我们提出的新定理。区别应落在**统计量、参考冻结方式、类条件化、配对与估计器**，不能只说“把 MMD 改成 energy”。

适合后续作者自行改写的定位：**研究联合置信度与分类 margin 的类条件分布目标，在相同 retain scaffold 下，检验分布匹配能否缓解固定逐样本目标的约束。** 当前仅可报告 calibration 指标下降；没有隔离上述因素的对照、独立 confirmation、audit 和 paired-Retrain 四指标前，不能声称其中某项机制导致提升、超过 A-AMU/POUR、首次 energy unlearning、或实现 certified deletion。

检索范围：2026-11-06 查询 `machine unlearning energy distance`、`unlearning energy distance held-out` 与 POUR 精确标题，主文核读限于以上三篇。博客与聚合页仅作发现线索，没有用来支持结论；未找到相同组合不等于不存在相同组合。
