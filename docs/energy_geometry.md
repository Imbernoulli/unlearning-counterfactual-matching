# Energy 目标的吸引、排斥与解释边界

2026-11-06。解释已注册的 joint-energy C/D、forget weight α∈{0.1,1,10}；不增加候选，不使用新实验结果推导保证。以下明确区分经典结论、直接推导和代码事实。CPU-only；未启动 GPU。

## 1. 与实现一致的有限样本力

同一类别内，令当前 forget scores 为 xᵢ(θ)∈ℝᵈ，i=1,…,n，冻结 Original 在 target-validation 上的 scores 为 yⱼ，j=1,…,m。joint 时 d=2，x=(s,h)：s=zᵧ−log∑ₖ≠ᵧexp(zₖ)，h=zᵧ−maxₖ≠ᵧzₖ。实现没有逐坐标标准化。`energy_study.py:53–76,79–89` 使用**未平方欧氏距离**和有序非对角 self pairs：

\[
 L_U=\frac{2}{nm}\sum_{i,j}\|x_i-y_j\|
 -\frac{1}{n(n-1)}\sum_{i\ne k}\|x_i-x_k\|
 -\frac{1}{m(m-1)}\sum_{j\ne\ell}\|y_j-y_\ell\|. \tag{1}
\]

设 u(v)=v/‖v‖，v≠0。对无重合点直接求导，self 项中 xᵢ 出现在有序对的两个位置，因此

\[
 \nabla_{x_i}L_U=
 \frac{2}{nm}\sum_j u(x_i-y_j)
 -\frac{2}{n(n-1)}\sum_{k\ne i}u(x_i-x_k). \tag{2}
\]

负梯度的第一项朝各 target 吸引，第二项使 forget 点彼此排斥；排斥项保留分散程度，避免把“接近所有 target”简化为向中心聚集。target self 项完整计入数值，但已冻结，梯度为零。“力”在这里仅指负梯度，距离势并非三维 Newton 重力势。

代码对本 batch 中 n꜀≥2 的 K 个类别取宏平均（`energy_study.py:122–141`），所以未裁剪的参数梯度为

\[
 g_F=\frac{\alpha}{K}\sum_c\sum_{i\in c}
 J_i^\top\nabla_{x_i}L_{U,c},\qquad J_i=\partial x_i/\partial\theta. \tag{3}
\]

实际流程再将 g_F 全局 norm 裁剪至 5，加入 retain 梯度，并执行带 momentum/weight decay 的 SGD（`calibration_study.py:282–303,328`）。故 α 的变化不总是线性放大更新；共享 Jᵢ 也使输出点不能独立运动。即使只做 energy 梯度流，局部也有 ẋᵢ=−∑ₖJᵢJₖᵀ∇ₓₖL，而非各点单独沿式 (2) 运动。

## 2. 一维 CDF 恒等式与梯度

**经典结论。** 若 P、Q 具有有限一阶矩，X,X′ iid∼P、Y,Y′ iid∼Q 且两组独立，则

\[
 \mathcal E(P,Q)=2\mathbb E|X-Y|-\mathbb E|X-X'|-\mathbb E|Y-Y'|
 =2\int_{\mathbb R}(F(t)-G(t))^2dt. \tag{4}
\]

精确来源为 Székely–Rizzo (2013)，§2、印刷 p.1251（PDF p.4），式 (2.3) 前的恒等式；其 Definition 1、Proposition 1 给出多维定义与分布识别条件。[原文](https://pages.stat.wisc.edu/~wahba/stat860/pdf4/Energy/JSPI5102.pdf#page=4)。这里积分测度是 dt，不是 G 加权的 Cramér–von Mises 检验统计量。

**本备忘录推导。** 将 |a−b| 写成 ∫(1ₐ≤ₜ−1ᵦ≤ₜ)²dt，取独立期望并展开即得式 (4)。若另外允许微分与积分交换，则

\[
 \nabla_\theta\mathcal E(P_\theta,Q)
 =4\int(F_\theta-G)\nabla_\theta F_\theta\,dt. \tag{5}
\]

若 F、G 连续，概率质量的变分势为 φ(x)=2E|x−Y|−2E|x−X′|，故 φ′(x)=4[G(x)−F(x)]，下降方向为 4[F(x)−G(x)]：左侧质量过多时向右移动。有限样本的一维精确版本也可由式 (2) 得到：无 ties，xᵢ 的升序 rank 为 rᵢ∈{1,…,n}，则

\[
 \partial_{x_i}L_U=\frac4n\left[G_m(x_i)-\frac{r_i-1}{n-1}\right]. \tag{6}
\]

因此它使用**当前**位置的累计 target 质量与当前 rank，不绑定 Original 给某个样本分配的固定 donor。joint 版本没有一维排序公式，但同样对同类 forget 点的任意置换不变。它放松身份配对，仍未识别每个样本真实的 counterfactual retrain 输出。

Bellemare 等 (2017) §4.1、PDF p.5、式 (4) 及其后一维关系使用相同定义；Theorem 2 与 Appendix A 的证明讨论 sample-gradient 性质。[原文](https://arxiv.org/pdf/1705.10743#page=5)。其 iid/期望交换条件不能直接视为本训练流程的 SGD 收敛保证。Energy 与 MMD 的联系也早已有精确结果：Sejdinovic 等 (2013)，§5.1、Theorem 22（该 arXiv PDF p.15）为相应生成核给出 E=2 MMD²。[原文](https://arxiv.org/pdf/1207.6076#page=15)。本项目没有提出新的距离。

## 3. U-statistic、近重合点和抽样限制

**U 不等于经验 CDF-L2。** 令 Aₓ=n⁻²∑ᵢₖ‖xᵢ−xₖ‖，Aᵧ 类似，Lᵥ 为 self 项分别除 n²、m² 的经验 energy，则严格有

\[
 L_U=L_V-\frac{A_x}{n-1}-\frac{A_y}{m-1}. \tag{7}
\]

一维时 Lᵥ=2∫(Fₙ−Gₘ)²dt；式 (4) 不能原样套到 Lᵤ。例：n=m=2，x=y={0,1} 时 Lᵥ=0，Lᵤ=−1。当前 self 修正影响梯度，target 修正为常数；小 n 下额外排斥更明显。Székely–Rizzo §1、印刷 p.1250（PDF p.3）明确区分 n(n−1) 的 U 和 n² 的 V，并说明采用 V 以保留非负解释。[原文](https://pages.stat.wisc.edu/~wahba/stat860/pdf4/Energy/JSPI5102.pdf#page=3)。

**无放回要说明估计对象。** 固定 θ，从该类 N 个 forget 点均匀无放回抽 n≥2 个，任一有序 distinct pair 的入选概率为 n(n−1)/[N(N−1)]。因此式 (1) 的期望是**全体有限样本的 U 目标**，不是分母 N² 的经验 energy；cross 项对固定完整 target 无偏。随机省略 n꜀<2 的类、对剩余类宏平均会改变类别有效权重。训练中 θ 随已访问 batch 改变，random reshuffling 不满足这里固定 θ 的简单条件。独立 iid 抽样与适当可积/可微条件下的总体无偏性，也不能消除固定验证集误差。

**近重合并非无限大的力。** 每个非零 pair 的 u 长度为 1，故式 (2) 的 score-gradient norm≤4/n；但 ∂u/∂v=(I−uuᵀ)/‖v‖，近重合时方向/曲率敏感。当前 `torch.linalg.vector_norm` 在精确零处给零导数，重合 pair 的排斥为零，不会自动打破对称；margin 的 max 还会在竞争类别 ties 处不光滑。此界不约束网络参数 Jacobian，也不覆盖动量、retain 冲突或裁剪后的轨迹。joint scores 还满足 0≤h−s≤log 9，共享网络必须保持其可实现性。

## 4. 为什么不保证最小 KS 或 AvgGap

CDF 的平方积分与 sup 差异不是同一目标。一个连续反例是 P=Uniform[0,ε]、Q=Uniform[ε,2ε]：由式 (4)，E=4ε/3→0，而 KS=1。没有统一反集中条件就不存在从 energy 数值到小 KS 的普适界；本例密度随 ε 缩小而增大。理论上 pᵧ=σ(s) 为严格单调变换，KS 对这个变换不变，但 energy 对尺度/变换敏感。joint energy 与选择时 pooled confidence KS、forget error 差异也不是相同函数，更不是每步单调的保证。

即使 population joint energy 精确为零，也只匹配**冻结 Original 的 held-out 分布**：它不等于删除后 Retrain 的逐样本输出。UA 涉及 margin=0 的边界质量，RA/TA 涉及其他输入，MIA 还依赖攻击规则及其重新拟合；这些都没有由该输出分布目标同时控制。故本方法的解释是“解除固定对应关系，并用吸引/排斥匹配整体分布”，收益需由预先冻结配置后的独立 Original、audit 和 Retrain 对照验证。

**CPU 核对记录。** 在 PyTorch float64 上，式 (2) 对照实际 `energy_u_statistic` 的 autograd，最大误差 5.55×10⁻¹⁷；式 (6)、式 (7)、经验 CDF 分段积分恒等式通过；枚举 N=4,n=2 的所有子集核对有限总体 U 值与参数梯度均通过。CUDA 未初始化。对应实现 commit `9df8dbc5ff`，SHA256 `63d0de85deca05542f56e7e2e1ba0f709f8b2fb0791c0a627fb71ef6a2235c7c`；未改实验代码或配置。
