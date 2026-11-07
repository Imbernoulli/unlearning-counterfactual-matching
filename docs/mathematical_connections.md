# 从最优传输和信息几何看 Retrain matching

日期：2026-11-06。本文只讨论数学依据与可检验算法，不包含或读取任何新 GPU/test 结果。下列小命题是对本项目的推导；所用数学原理均有既有来源，不据此声称新的 unlearning 理论或已实现的性能提升。

## 1. 最值得利用的联系

“让 forget 像非成员”可以拆成两个问题：先指定每个样本应该到达什么预测，再用共享网络参数逼近这些预测。目标构造本身不必再训练一个网络。

一个简单方案是：用完整 forget 集在 Original 下的类别内排名，为每个样本分配 held-out 真类概率分位数；随后只改变真类概率，并尽可能保持其余类别之间的相对概率。这一步有明确的最小 KL 性质。全局排名也绕开了每个 minibatch、每类大约 25 个样本时的 Wasserstein 采样偏差。

这里的保证只涉及**给定目标分布后，预测空间中的最小改变**。它不保证神经网络的参数改动最小，不保证这些目标就是逐样本 Retrain 预测，也不构成删除保证。

## 2. 一维传输：排序不是启发式

对于实线上分布 $P,Q$ 及其广义分位数函数，

\[
W_p^p(P,Q)=\int_0^1|F_P^{-1}(r)-F_Q^{-1}(r)|^p\,dr.
\]

等权、等样本数的经验分布因而按排序配对。直接依据是 Peyré–Cuturi 的 *Computational Optimal Transport*，§2.6，Remark 2.27–2.28，式 (2.31)、(2.34)、(2.37)，印刷页 35–37：[作者公开 PDF](https://optimaltransport.github.io/pdf/ComputationalOT.pdf)，[arXiv:1803.00567](https://arxiv.org/abs/1803.00567)。不同样本数时要积分两个阶梯分位数函数，不能把任意线性插值当作精确经验 $W_2$。

对本项目的意义是：在某一类别内，先给难度分数排序，再匹配 held-out 分位数，有最小平方分数改变的解释；不必引入核带宽或随机 donor。该原理本身是经典结果。

**离散化边界。** forget 与 held-out 每类数量不同，把 held-out 分位数在 forget 排名中点取值，只是固定有限目标集的构造。它一般不使两个有限经验分布精确相等。若想得到指定原子数、等权的最优 $W_2$ 离散化，每个等质量 quantile 区间应使用区间内分位数的平均值。后文最小 KL 命题对任意预先确定的目标多重集成立，不依赖选择哪一种离散化。

## 3. 信息几何：给定真类概率，怎样改动最少

令 Original 在一个 forget 样本上的预测为 $p\in\Delta^{K-1}$，各坐标严格为正。真类是 $y$，指定目标真类概率为 $u\in(0,1)$。考虑

\[
\min_{q\in\Delta^{K-1}}D_{\rm KL}(q\|p)
\quad\text{s.t.}\quad q_y=u.
\]

唯一最优解为

\[
q_y^*=u,\qquad
q_k^*=(1-u)\frac{p_k}{1-p_y},\quad k\ne y.
\]

证明不需要近似。记 $\tilde p_k=p_k/(1-p_y)$、$\tilde q_k=q_k/(1-u)$，则 KL 的链式分解给出

\[
D_{\rm KL}(q\|p)
=d_{\rm Ber}(u\|p_y)
+(1-u)D_{\rm KL}(\tilde q\|\tilde p).
\]

第一项已固定，第二项在 $\tilde q=\tilde p$ 时唯一最小为零。因此所有非真类的比值 $q_j^*/q_k^*=p_j/p_k$ 保持不变。这是线性约束上的 information projection 的一个有限维特例：Csiszár，*I-Divergence Geometry of Probability Distributions and Minimization Problems*，Annals of Probability 3(1), 146–158 (1975)，§3 式 (3.2)，印刷页 151；Theorem 3.1，页 152：[原论文扫描 PDF](https://web.mit.edu/6.291/www/handouts/csiszar_Idiv.pdf)，[DOI](https://doi.org/10.1214/aop/1176996454)。该扫描每个 PDF 页面可能含两张印刷页，引用应以印刷页为准。

它也是一个只对真类施加的指数倾斜。若 $p=\operatorname{softmax}(z)$，定义

\[
s=z_y-\log\sum_{k\ne y}e^{z_k}
=\operatorname{logit}(p_y),
\]

只需把真类 logit 改成 $z_y+\operatorname{logit}(u)-s$，其他 logits 不变，即得到 $q^*$。这里的“最小”明确指 $D_{\rm KL}(q\|p)$；不要只用 I-/M-projection 名称而省略方向。

**数值实现。** Original 的 $p_y$ 很接近 1，float32 中直接计算 $p_k/(1-p_y)$ 容易失稳。应从非真类 logits 的 softmax 计算条件分布，再乘 $1-u$；或者在 log-space 用稳定的 log-sigmoid 构造目标。此目标在训练时使用已知标签；不能在测试时用测试真标签直接改变预测，必须蒸馏进网络参数。

## 4. 更强的结论：排序配对同时最小化总 KL

给定同一类别的 Original 预测 $p_1,\ldots,p_n$，以及一个固定的目标真类概率多重集 $u_1,\ldots,u_n$，求

\[
\min_{\pi,\{q_i\}}\sum_iD_{\rm KL}(q_i\|p_i),
\qquad (q_i)_y=u_{\pi(i)}.
\]

最优构造恰好是：

1. 按 $t_i=\operatorname{logit}((p_i)_y)$ 排序。
2. 将排序后的 $u_j$ 按相同顺序配给样本。
3. 对每个配对做上一节的最小 KL 修正。

证明：先对 $q_i$ 最小化，把问题化为 $\sum_i d_{\rm Ber}(u_{\pi(i)}\|(p_i)_y)$。展开后，所有依赖排列的项只有

\[
-\sum_i u_{\pi(i)}t_i.
\]

由重排不等式，两个序列同序时 $\sum_i u_{\pi(i)}t_i$ 最大。等价地，对 $t_i\le t_j$、$u_a\le u_b$，同序配对相对交叉配对的成本差为 $(u_b-u_a)(t_i-t_j)\le0$，反复消除交叉即可。

这个推导比“排序最小化平方 margin 变化”更贴近预测保持：在所有满足同一组真类概率目标的预测中，它使相对 Original 的总 KL 最小。它仍有清楚边界：目标多重集来自 held-out，而非真实 counterfactual；Original 成员样本排名未必等于其 Retrain 难度排名；网络共享参数也未必能同时实现所有最优 $q_i$。

## 5. 为什么需要全局目标：minibatch Wasserstein 会改变优化终点

Fatras 等，*Learning with minibatch Wasserstein: asymptotic and gradient properties*，AISTATS 2020，§3.3 Proposition 3（PDF 页 4）说明其 minibatch OT 正且对称，但并不是距离，可能 $U(P,P)>0$；§3.3 后段（PDF 页 5）说明 batch size 从全数据退化到 1 时，目标从 OT 变为平均 pairwise distance。§4 的 gradient-flow 例子还明确指出 minibatch 偏差可使终点不等于目标分布：[会议原文](https://proceedings.mlr.press/v108/fatras20a/fatras20a.pdf)，[正式条目](https://proceedings.mlr.press/v108/fatras20a.html)。文中的“无偏梯度”是相对其**minibatch 平均目标**，不是说该目标等于完整数据的 Wasserstein 距离。

下面给出一个与“小 batch 缩尾”直接相关的解析例子；这是 toy 推导，不是本项目神经网络已经测出的现象。

令完整目标 $Q=\frac12\delta_{-1}+\frac12\delta_1$，候选分布 $P_\sigma=\frac12\delta_{-\sigma}+\frac12\delta_\sigma$，$\sigma\ge0$。取 $n$ 个候选样本，经验分布为 $\hat P_{\sigma,n}$。令正号比例 $R=B/n$，$B\sim\operatorname{Binom}(n,1/2)$。排序运输给出

\[
W_2^2(\hat P_{\sigma,n},Q)
=(\sigma-1)^2+4\sigma|R-1/2|.
\]

因为符号不匹配的质量为 $|R-1/2|$，该部分成本从 $(\sigma-1)^2$ 变为 $(\sigma+1)^2$。取期望得

\[
\mathbb E W_2^2=(\sigma-1)^2+4a_n\sigma,
\quad a_n=\mathbb E|B/n-1/2|,
\quad \sigma^*=1-2a_n.
\]

完整分布距离的最优值应为 $\sigma=1$，但 $n=1$ 时最优值为 0；$n=25$ 时，由有限二项和精确算得 $a_{25}=0.08059012889862062$，故 $\sigma^*=0.8388197422027588$。也就是说，这个例子在每类 25 点时会把本来正确的宽度收缩约 16.1%。它证明这种偏差可能存在，不证明当前任务必然出现同样幅度。完整 forget 集排名后固定 targets，可直接去掉这一类“每步重排小样本分布”的终点偏差。

## 6. Margin、置信度与可实现性

定义 $m=z_y-\max_{k\ne y}z_k$、$s=z_y-\log\sum_{k\ne y}e^{z_k}$。则

\[
0\le m-s\le\log(K-1).
\]

这是 log-sum-exp 与最大值的基本夹界。多分类下，单独给两个统计量独立排名、独立分配目标，可能造出违反此夹界的逐样本配对。来自同一 held-out 样本的 $(s,m)$ 对天然满足约束；用同一个 donor/rank 构造两个统计量比各自任意配对更清楚。

最小 KL 真类修正会保持 $m-s$ 不变，因为所有非真类 logits 的相对结构保持不变。故它即使精确匹配 $s$ 的边缘分布，也不一定匹配 UA。二分类时 $K-1=1$，二者相同；多分类时其区别是真实问题。可能的后续修正是按竞争结构 $a=m-s$ 做少量条件化，或直接使用同一个 held-out donor 的二维目标；但样本数有限时不能无限增加分桶。

若 $m<0$ 表示分类错误，忽略 ties，对任意耦合及 $\epsilon>0$，

\[
|\Pr(m_f<0)-\Pr(m_v<0)|
\le\Pr(|m_v|\le\epsilon)+\frac{W_1(P_{m_f},P_{m_v})}{\epsilon}.
\]

证明是将符号不一致事件分为“目标离边界很近”或“耦合后的改变量至少 $\epsilon$”，再用 Markov 不等式。若目标在边界附近没有过多质量，margin 距离小能控制 UA 差距。该结论不自动适用于仅匹配 $s$。

对 MIA，固定攻击规则 $h\in[0,1]$ 时有

\[
|\mathbb E_{P_f}h-\mathbb E_{Q_v}h|\le\operatorname{TV}(P_f,Q_v).
\]

如果用 Wasserstein，通常还需攻击分数是 Lipschitz，或硬判决边界附近有反集中条件。当前 SVC 每次由 retain/test confidence 重新拟合，因此还存在攻击器变化项。不能从 forget confidence 边缘匹配直接推出“与 Retrain 的 MIA 完全匹配”，更不能推出针对所有攻击的删除保证。

## 7. 第二个值得保留的数学候选：energy / Cramér 距离

Bellemare 等，*The Cramer Distance as a Solution to Biased Wasserstein Gradients*，2017，Theorem 1（印刷/PDF 页 4）给出 sample Wasserstein 梯度偏差及错误极小值；§4.1 Theorem 2 与随后多变量 energy 扩展（页 5）提供可替代目标：[原文](https://arxiv.org/pdf/1705.10743)，[条目](https://arxiv.org/abs/1705.10743)。

对 $X=(s,m)\sim P$、$Y\sim Q$，energy 距离为

\[
\mathcal E(P,Q)=2\mathbb E\|X-Y\|
-\mathbb E\|X-X'\|-\mathbb E\|Y-Y'\|.
\]

在有限一阶矩等标准条件下，它在两个分布相同时为零，并能识别分布差异。按类在小 batch 上用 U-statistic 估计：cross 项取均值，self 项用非对角项除以 $n(n-1)$；held-out self 项对梯度是常数。独立样本与允许交换期望/导数时，可以得到总体目标的无偏梯度。若无放回采样而目标是包含对角的有限经验分布，还应注意相应有限样本权重差异。

两个实现陷阱：不要把欧氏范数替换为平方欧氏范数，后者在该组合中会退化为只匹配均值；也不要忽略 self 项，忽略后又会成为会压缩分布的 pairwise 拟合。这里的 energy/MMD 联系及距离本身都早已有文献，最多是一个有理由加入的目标消融；本轮不需要无限扩张实验矩阵。

## 8. 建议的最小实验与结论边界

1. **全局分位数 + 最小 KL 目标。** 冻结 Original，按类别对完整 forget scores 排名；用 target-validation 构造固定概率分位数；按第 3–4 节生成逐样本 $q_i^*$，用 KL 蒸馏并保持原有 retain CE/anchor。与当前 minibatch W2 用相同更新预算比较。它检验的是“目标构造和采样偏差”，不能直接宣称学习了真实个体删除效应。
2. **同 donor 的二维目标。** 根据一个全局排名为每个样本选 held-out donor，目标 $(s,m)$ 始终来自同一个点。与各统计量独立 minibatch matching 比较，观察是否真的需要改变非真类竞争结构。这个实验可以将“只校准 confidence”与“同时改变错误率”分开。
3. **后续 energy 距离消融。** 如果全局 targets 的固定 Original 排名导致难度错配，再比较按类的二维 energy U-statistic；不引入 kernel 带宽，但应明确其统计假设和梯度估计方式。

所有配置应在 test 评估前锁定；可用独立 calibration validation 检查 utility 与分布误差。模型与 Original 的数据量差异、有限 held-out 尾部噪声、共享参数对 retain 的连带影响，都不会由上述输出空间定理消失。最终 paired Retrain 除四个 aggregate 指标外，还可在预先锁定方案后检查逐样本错误/score 关联，区分“数量对了”和“删掉这些样本后的行为接近”。

最接近的 unlearning 先例仍应明确引用 ReGUn（held-out reference）、A-AMU（unseen/forget 分布匹配）和 Datamodel Matching（counterfactual 输出估计）；详见同目录 `research_notes.md`。信息投影和排序均是经典数学，本文没有完成覆盖所有 unlearning 文献的新颖性证明。
