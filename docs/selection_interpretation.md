# Selection plateau: what the calibration evidence does and does not identify

本备忘录仅使用已完成的 r1/r2 selection 结果、[mechanism_findings.md](mechanism_findings.md) 和冻结实现；未读取后续轮次、audit 或新的 test/Retrain 结果，也未启动 GPU。以下排序只涉及 r1/r2，不是后续全部候选的排名。

## 1. 约 0.094 的是什么

预先固定的选择量是

\[
J=\max\{|e_F-e_S|,\;D_{\rm KS}(C_F,C_S)\},\qquad C=p_\theta(y\mid x),
\]

并要求 retain 子集准确率下降不超过 0.5 pp、selection 准确率下降不超过 1 pp。所有概率差使用 [0,1] 单位。

| 已完成候选 | 选中 epoch | J = KS | 错误率差绝对值 (pp) | retain / selection 准确率漂移 (pp) |
|---|---:|---:|---:|---:|
| r1 C/frozen，全局 log-odds MSE，LR .003，权重 .1 | 2 | .095689 | 4.8622 | −.24 / −.48 |
| r2 C/frozen，rank-donor KL，LR .003，权重 10 | 2 | .093689 | 3.2756 | −.24 / +.24 |
| r2 C/frozen，rank-donor KL，LR .003，权重 1 | 9 | .099378 | 2.9289 | −.06 / −.48 |
| r2 C/frozen，rank-donor log-odds MSE，LR .003，权重 .1 | 1 | .102400 | 6.2711 | −.20 / −.80 |

来源为 `.local-saves/unlearning-project-20261106/optimization/daytona/round1_calibration_table.json` 和 `round2_calibration_table.json`，分别有 11 行（含 smoke）和 15 行。表中所有选中项的 J 都由 KS 主导。因此 .094 表示在某个置信度阈值处，两个经验 CDF 仍相差约 9.4 个百分点；它**不是 AvgGap、实际 SVC-MIA gap，也不是已证实的不可约下界**。当前规则会优先减小这个最大 CDF 缺口，而未必奖励错误率差的进一步减小。

机制证据不支持单一解释：A 的 retain 段在早期能修回 KS 改善，但 epoch 10 反而改善 KS；梯度既有冲突也有对齐。A epoch 10 的固定批次 forget log-odds RMSE 仍为 .910（Original 为 2.424），十个置信度分位组平均残差均为正，说明至少该 checkpoint 尚未精确拟合目标。另一方面，KS 最大差对应的阈值在约 .9961–.9991 之间移动，不能仅靠正确率或一个固定攻击阈值解释。以上数字均来自上述 mechanism memo，不是独立重复实验。

## 2. 三种不同的障碍

**目标偏差。** 固定参考来自训练了 45,000 点的 Original 在 target-validation 上的预测，未知的反事实模型只用 retain 训练；两者并无相等保证。Original 在 forget 上的难度排名也未必等于删除后的排名。此外，训练中的当前模型在 selection 上会变化，而固定目标不会。

令 \(Q_*\) 为实际构造的目标真类概率经验分布（包括离散化），则三角不等式给出

\[
D_{\rm KS}(P_F^\theta,P_S^\theta)
\ge [D_{\rm KS}(Q_*,P_S^\theta)-D_{\rm KS}(P_F^\theta,Q_*)]_+.
\]

即使精确达到 \(P_F^\theta=Q_*\)，仍可能留下 \(D_{\rm KS}(Q_*,P_S^\theta)\)。这说明应区分“未达到目标”和“目标本身不是当前非成员分布”；现有表不足以判定哪项支配 .094。分位数 bin 平均和重复 donor 也不使两个有限经验分布严格相同。

**优化偏差。** 共享参数同时受 forget、retain、clip、momentum、weight decay 与 BN 约束。某个输出空间的最优目标不一定能由有限步网络更新同时实现。减少标量训练 MSE 也不保证减少 KS：KS 看最大阈值差，MSE 看平均 score 距离，二者没有无需分布假设的统一替代关系。

**不可识别性。** 即使 J 为零、utility 通过，也只匹配若干有限统计。同类 forget 样本之间交换预测，或改变非真类身份而保留真类概率和最大竞争值，都能保留这些统计，却改变逐样本反事实预测。这不是再换一个 loss 就能消除的监督缺口；不能由此声称模型等价于 Retrain 或已删除全部成员信息。

## 3. MSE 与 donor KL：可直接证明的差别

写当前预测为 \(p=(v,(1-v)r)\)，目标为 \(q=(u,(1-u)r_*)\)，真类 log-odds 为 \(s\)，\(u=\sigma(s_*)\)。在向量中令 \(r_y=r_{*,y}=0\)，则

\[
\nabla_z(s-s_*)^2=2(s-s_*)(e_y-r),
\]
\[
\nabla_zD_{\rm KL}(q\|p)
=(v-u)(e_y-r)+(1-u)(r-r_*).
\]

第一式只直接约束真类 log-odds；第二式还约束非真类条件形状。高置信区域里，KL 的 score 方向系数 \(v-u\) 可以很小，而 MSE 系数仍随 log-odds 残差线性增大。donor KL 的额外项则可改变竞争结构；因为

\[
m-s=-\log\max_{k\ne y}r_k,
\]

仅匹配 s 不保证匹配 margin/错误率。原先的最小 KL 修正取 \(r_*=r_0\)；rank-donor 目标通常不是这个同一个投影问题，而是保留 donor 统计并重排非真类身份。

这解释了两种 loss 的约束和梯度尺度为何不同，**不证明哪一种表现更好**。r2 的最低 J 来自 donor KL 权重 10，不能继续笼统称“MSE 优于 KL”；它也不能证明 KL 优于 MSE，因为与权重 .1 的 MSE 相比，系数相差 100 倍，目标形状、配对方式和选中 epoch 也不同。最小 KL 数学结论本身并未被有限预算的拟合结果反驳。

## 4. 只保留两个下一步假说

**优先：joint energy 检验固定配对是否过强。** 固定逐样本排名把一份总体非成员分布变成了身份级约束。若这些身份对应不准，网络可能为了错误配对付出 retain 代价。类条件 energy 在联合 (s,m) 空间匹配分布，允许 forget 点自行交换目标角色，又保留 margin/置信度的可实现关系。有限一阶矩下，population energy 为零当且仅当两分布相同；标准 U-statistic 在 iid 条件下估计该量，有限样本可以为负。这提供了比“再调 MSE 系数”更直接的机制检验。它仍保留 Original 参考偏差，也不保证降低 KS；当前无放回抽样及 singleton 省略还需按实现解释。建议只做一个预先固定的 joint-energy 对照，不扩展新的 LR 网格。

**第二：saliency 是几何控制，不是目标修复。** 固定 Original-forget CE 梯度的 50% 坐标掩码可以检验限制修改范围是否以更小 utility 损伤换来更充分拟合。但 Original 已近乎插值时，\(\nabla_z\mathrm{CE}=p-e_y\) 会弱化高置信点，聚合梯度还会相互抵消；大 forget 梯度也不等于 forget 独有信息。限制坐标甚至可能加剧已有拟合不足。因此它比 energy 更适合作为第二个、与全坐标设置匹配的 SalUn-inspired 控制，并应计入额外 mask pass；不能当作已知会改善 .094 的办法。

两者回答不同问题：energy 放松样本目标对应，saliency 收紧参数可行空间。不要一起改后再归因于其中一个，也不新增第三套假说或大网格。

## 5. 1250 点和多次搜索的边界

selection 只有 1,250 点，1 pp 准确率仅约 12–13 个样本。r1/r2 使用同一个 Original、同一个删除 split 和同一个 selection，26 行不是 26 次独立复现；两个 manifest 合计 480 个预定 epoch endpoints，另有 1 个 smoke endpoint。r2 又是在 r1 反馈后设计，属于适应性开发。

对一个预先固定且独立于 selection 的模型，在理想 iid 假设下，单个 selection CDF 的 95% DKW 半径为

\[
\sqrt{\log(2/.05)/(2\cdot1250)}\approx .0384.
\]

这不是 J 的置信区间，也不包括训练过 forget 的依赖或模型选择。对预先固定的 K 个模型，可用 union bound 将 2 换成 2K；K 约 481 时量级约 .063。当前自适应 r2 搜索不能直接套用这个固定族保证。r1 到 r2 的最佳 J 仅下降 .002，不能由 selection 表称为稳定泛化提升，也不能把 .094 全部归因于抽样噪声。

仍按冻结规则只用 selection 选一次完整方法/停止策略；随后 audit 只报告一次，不据此换候选。确认阶段必须保留独立删除 seeds/Originals、真实成本和全部失败，并将本轮明确称为 calibration-guided development。KS 对固定单阈值攻击的 cohort 判决率差给出上界；RBF-SVC 的多区间决策和重拟合都限制其与最终 MIA 的关系，J 不能替代最终评测。
