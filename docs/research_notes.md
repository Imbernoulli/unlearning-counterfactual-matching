# 从 Retrain 匹配出发的机器遗忘研究备忘录

2026-11-06；仅讨论已核实文献、仓库历史测量与待验证推导，不包含本轮 GPU 实验结果。下述 AvgGap 为公开文献中的原始指标差距。

## 方法演进：先明确应当忘到哪里

FT 在 retain 上最小化交叉熵；GA 在 forget 上反向优化交叉熵；RL 用随机标签破坏 forget 记忆，再用 retain 训练维护效用。这些方法提供更新方向，却没有直接规定 Retrain 应有的终点。对已高置信拟合的样本，交叉熵的 logit 梯度为 \(p-e_y\)，当 \(p_y\to1\) 时趋近零，因此短程 GA 可能几乎不动；RL 有更强梯度，但其监督目标与正常泛化预测存在偏差。

Jia 等的稀疏遗忘将 retain 微调与递减的 \(\ell_1\) 惩罚结合；SalUn 根据 forget 梯度的绝对值筛选可更新坐标，再做 RL 和 retain 训练。它们解决改哪些权重的问题，仍需选择遗忘程度。[Jia 等，2023，§4、式 (3)、表 2，p.6](https://arxiv.org/pdf/2304.04934v13)；[Fan 等，2024，§5、式 (3)–(6)，文内 pp.5–6](https://arxiv.org/pdf/2310.12508v5)。

SCRUB 用冻结 Original 作教师，在 forget 上增大教师—学生 KL，在 retain 上减小 KL 并优化 CE。SCRUB+R 用同分布验证集误差选取回退 checkpoint，已经明确处理过度遗忘。因此“用 validation 决定忘到哪里”有既有基础。[Kurmanji 等，2023，§3.1 式 (3)、§3.2，pp.4–5](https://arxiv.org/pdf/2302.09880v3)。

## 最直接的先例与结果

**ReGUn 与 proposal 的核心设想直接相关。** 它按 forget batch 的类别直方图采样 held-out 数据，平均冻结教师的预测得到一个 \(q\)，再对每个 forget 样本最小化 \(\mathrm{KL}(q\|p_\theta)\)，并加入 retain CE。该条件化发生在 batch 直方图上，同一 batch 的样本共享目标，并未保留个体难度分布。[Mirlach、Laguna、Vogt，2026，§3.2、式 (1)–(3)、算法 1–2，pp.4–6](https://arxiv.org/pdf/2603.11210v3)。官方 commit `04a0d22eb10cfe54c634ba70435b1b5846ef296d` 的 [regun.py L120–143](https://github.com/jmirlach/ReGUn/blob/04a0d22eb10cfe54c634ba70435b1b5846ef296d/mul/regun.py#L120-L143) 实现类别采样，[L245–252](https://github.com/jmirlach/ReGUn/blob/04a0d22eb10cfe54c634ba70435b1b5846ef296d/mul/regun.py#L245-L252) 明确平均并复制目标，[L289–300](https://github.com/jmirlach/ReGUn/blob/04a0d22eb10cfe54c634ba70435b1b5846ef296d/mul/regun.py#L289-L300) 实现损失；其默认 YAML 的 selector 是 `uniform`，复现论文类别设定需显式修改。

A-AMU 已用 MMD 匹配 forget 与同类 unseen 数据的变换后 loss 分布。[Khan，2025，§3.4、式 (6)，PDF p.4](https://arxiv.org/pdf/2507.09786v3)。因此 held-out 引导和 MMD 本身都不能作为新颖性主张。

以下均来自 **SalUn 同一张表**：CIFAR-10、ResNet-18、随机遗忘 10%，原始指标为百分数，AvgGap 为百分点，时间为分钟；论文报告十次独立试验。[SalUn 表 1，文内 p.7、PDF 第 9 页](https://arxiv.org/pdf/2310.12508v5#page=9)。

| 方法 | UA | RA | TA | MIA efficacy | AvgGap | 时间 |
|---|---:|---:|---:|---:|---:|---:|
| Retrain | 5.24 | 100.00 | 94.26 | 12.88 | 0.00 | 43.29 |
| FT | 0.63 | 99.88 | 94.06 | 2.70 | 3.78 | 2.37 |
| GA | 0.69 | 99.50 | 94.01 | 1.70 | 4.12 | 0.13 |
| RL | 7.61 | 99.67 | 92.83 | 37.36 | 7.15 | 2.64 |
| l1-sparse | 4.19 | 97.74 | 91.59 | 9.84 | 2.26 | 2.36 |
| SalUn | 2.85 | 99.62 | 93.93 | 14.39 | 1.15 | 2.66 |

本仓库保留的 **revision 4** 测量使用新抽取 forget split 和跨 split 的 Retrain reference。CIFAR-10 random10 的 AvgGap：Original 4.650、GA 4.641、FT 2.215、l1-sparse 2.100、RL 1.170、SalUn 1.171；记录为五个 seed 的均值。来源：[tasks/security-machine-unlearning/NOTES.md，§31 Revision-4 leaderboard](../../../../tasks/security-machine-unlearning/NOTES.md#31-revision-4-leaderboard-daytona-h100-rendered-measurement-bundle-5-fresh-split-seeds-42-46)，原始数据见 [leaderboard.csv](../../../../tasks/security-machine-unlearning/leaderboard.csv)。这些是仓库中保存的历史测量，并非本轮重新运行。

这组历史结果提示两个不同失败模式：GA 和 Original 接近，可能属于有效更新不足；RL 虽提高遗忘指标，却可能使置信度尾部偏离重训。SalUn 的优势也应拆开看，不能由平均差距推断它在每个维度都更接近。比较方法时，既要检查算法定义，也要检查训练协议。论文的随机标签约束、代码里的具体采样方式、学习率选择，以及 reference 是同一次删除的重训还是跨划分均值，都影响可解释性。

本机 RL 与论文 RL 相差明显，应核对实现、学习率、抽样和 reference 后再解释。ReGUn 的表 1 使用 RMIA/SMIA **AUC**，本任务使用 SVC 的 nonmember efficacy；其 Gap 数值也不能与上述表混排。[ReGUn 表 1，p.9](https://arxiv.org/pdf/2603.11210v3#page=9)。

## 第一性原理：分开控制分类边界和成员置信度

给定 logits，定义

\[
m=z_y-\max_{j\ne y}z_j,\qquad
s=z_y-\log\sum_{j\ne y}e^{z_j}=\operatorname{logit}(p_y).
\]

忽略概率为零的分类平局，UA 由 \(\Pr(m<0)\) 决定；本任务 confidence attack 读取的 \(p_y\) 则由 \(s\) 决定。反例：真实类均为第一类时，预测 \((0.4,0.35,0.25)\) 正确，\((0.4,0.5,0.1)\) 错误；二者 \(p_y\) 与 \(s\) 完全相同。故即使精确匹配真实类置信度分布，也不能保证多分类 UA。匹配均值的约束更弱。

一个简洁候选是对每个类别匹配 forget 与 held-out 的 \(m,s\) 分布，再以 retain CE/KL 约束扰动。两个边际匹配可分别观察边界与置信度；如需保留二者依赖，应增加二维联合匹配消融。冻结参考可避免目标随优化共同退化，但 Original 在验证集上的行为仍只是 Retrain 的代理。 对类别数 \(K\)，还有可实现性约束 \(0\le m-s\le\log(K-1)\)。若分别为每个样本指定两个独立分位数目标，可能产生不相容配对；同一 held-out 样本的联合目标能保留这一约束。

**一维 \(W_2\) 的最小配对。** 对同类别、等量样本 \(a_i,b_i\)，

\[
W_2^2(\hat P,\hat Q)=\frac1n\sum_i(a_{(i)}-b_{(i)})^2
=\min_\pi\frac1n\sum_i(a_i-b_{\pi(i)})^2.
\]

排序配对最优：当 \(a_i\le a_j,b_i\le b_j\) 时，交叉配对比顺序配对多 \(2(a_j-a_i)(b_j-b_i)\ge0\)。样本数不同时可积分经验分位数。这里最小的是**统计量空间中的配对位移**，不代表神经网络参数位移最小，也不保证梯度更新能达到该目标。

**UA 的条件性界。** 假设 margin 分布有有限二阶矩；令 \(M\) 为 forget margin、\(V\) 为参考 margin，并设 \(\delta=W_2(P_M,P_V)\)。对任意 \(\varepsilon>0\)，在最优耦合下，符号不同只能因为参考在边界附近，或位移超过 \(\varepsilon\)，故

\[
|\Pr(M<0)-\Pr(V<0)|
\le\Pr(|V|\le\varepsilon)+\delta^2/\varepsilon^2.
\]

若额外假设 \(\Pr(|V|\le\varepsilon)\le C\varepsilon\)，优化后得到 \(O(C^{2/3}\delta^{2/3})\)；百分数 UA 需乘 100。有限样本可直接计算边界附近质量，不能默认存在该密度界。上述均为本备忘录的推导，只控制相对于参考的误差；参考与 Retrain 的偏差仍需另加。匹配这些统计量**不是删除证明**，不能推出参数分布等同于重训，也不能排除其他攻击识别成员。

## 实验顺序与监督可识别性

先以一致的配对 split、更新预算和选择准则比较 FT、SalUn、SCRUB+R 与明确版本的 ReGUn，再逐步加入类均值目标、置信度分位数、margin 分位数及联合匹配。每项报告四个原始指标、各自差距、运行时间和跨 seed 波动；用独立 validation fold 选择权重和停止点，测试集与 Retrain 仅作最终评估。随后比较 head、末层 block、全网络更新，并检查更强攻击和重新学习，验证收益是否超出当前指标。 参考集还应区分目标构造与模型选择：反复依据同一批样本挑选更新强度，会让该批样本参与优化，削弱其作为独立未见数据的解释。应预先保留选择用样本，并记录选择规则。分布损失的下降、边界附近样本质量和保留集性能应同步记录，用以区分目标确实更好、更新更温和与单纯提早停止。

linear/tree/MLP 可学习条件 nonmember prior，却不能仅凭 held-out 的自预测标签识别某个训练样本删除后的反事实输出。复杂预测器不能补上缺失监督；需要明确假设，或用 retain 内构造的 shadow/留出实验产生配对监督。DMM 通过数据归因预测重训输出后蒸馏，提供了这一区别的直接先例。[Georgiev 等，ICLR 2025，摘要与引言](https://arxiv.org/abs/2410.23232)。优先验证简单分布目标是否已经解决瓶颈，再投入学习型目标模块。
