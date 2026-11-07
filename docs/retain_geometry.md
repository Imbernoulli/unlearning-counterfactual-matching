# 最小 retain 保护：先消除恢复 Original 的偏置，再考虑几何

2026-11-06。本备忘录只读实现和论文；未读取实验结果或测试集，未启动 GPU。下面所有数值例子都是明确给定函数上的解析/数值验算，不是 CIFAR 实验结果。作者只建议消融，不修改已冻结实验实现。

## 1. 三小时内的优先顺序

| 顺序 | 单独改变什么 | 新增可调标量 | 额外计算 | 要回答的问题 |
|---|---|---|---|---|
| 1 | retain 保留 CE，取消 Original KL，即现有 `anchor_weight=0` | 0，去掉一个权重 | 减少 teacher forward | Original anchor 是否阻碍正确的删除响应？ |
| 2 | 保持同样的 forget/retain 数据遍历和更新次数，把约 18 个 forget batch 均匀插入约 159 个 retain batch | 0 | 0 次额外 forward/backward | 一个完整 retain repair 是否把刚完成的遗忘压回去？ |
| 3 | 若仍需保护，采用**已有**双侧 PCGrad 作为机制对照 | 0，沿用原有 loss 权重 | 两梯度分别保存/归约；配对规则影响成本 | 冲突方向是否确实解释 trade-off？ |

1 和 2 应先作为独立消融，避免把目标、anchor、调度、BN 和优化器一起改掉。几何预条件可以后做，但在当前时间预算内不应先上全网 Fisher/CG。以上不是性能排序；尚无数据支持哪一个最终有效。

## 2. 当前实现存在两个可识别的交互

源码依据：[run_experiments.py](/srv/home/bohanlyu/MLS-Bench/research/unlearning_matching/run_experiments.py:273)。`candidate()` 每 epoch 先以 `eval()` 做整个 forget pass，再以 `train()` 做整个 retain pass；同一个 momentum=.9、weight decay=5e-4 的 SGD 穿过两阶段。retain loss 是 CE + 0.1 × 4 KL(Original at T=2 || student at T=2)。forget batch 为 256，4500 点约 18 步；retain 40500 点约 159 步。

**交互 A：长期 repair 可以恢复 Original，而不只是保护 utility。** 以局部标量线性模型说明：若一次 retain 更新将位移乘以 ρ∈(0,1)，forget 更新注入 u_t，则

  δ_T = ρ^T δ_0 + Σ_t ρ^{T−t}u_t。

这是直接展开递推的恒等式。相同的 forget 更新总和，放在开头会比均匀插入时衰减更多。它支持做调度消融，不证明更晚 forget 一定更接近 Retrain；后者也可能导致过度遗忘。用固定整数间距均匀插入，不靠最终 test 选择终止时刻。

**交互 B：BN 状态与目标不是同一个前向算子。** retain 阶段 student 为 train、Original teacher 为 eval。即使权重起点相同，student 的 batch statistics 与 teacher 的 running statistics 也可使预测不同，故初始 anchor 梯度未必为 0；retain 阶段还更新 student running statistics，之后的 forget/evaluation 使用的函数随之改变。若要检验下面的局部理论，可另设冻结 BN 的独立对照；不能直接把 train/eval 差异归因于 forgetting loss。冻结 BN 是可检验选择，并非普遍正确的设置。

## 3. 为什么 Original anchor 可能抵消正确删除

以下是本备忘录的局部推导。先假定固定相同的前向/BN 状态，θ₀ 为 Original。令 δ=θ−θ₀，保留集 CE/正则风险近似为

  L_R(θ₀+δ) ≈ const + g_Rᵀδ + ½δᵀH_Rδ。

目标是保留集最优解时，局部响应为 δ_R=−H_R⁻¹g_R。Original 蒸馏项 A 的最低点却在 δ=0，展开为 A≈½δᵀF_R,Tδ。加入它得到

  δ_anchor = −(H_R+αF_R,T)⁻¹g_R。

若某模式 F_R,T≈H_R，则该模式的正确删除位移被缩小到原来的 1/(1+α)。更一般地，在 H_R 正定子空间中，每个广义曲率模式被乘以 1/(1+αμ_j)，μ_j 是 H_R⁻¹/²F_R,T H_R⁻¹/² 的特征值。anchor 不是理论上不该使用；它是带回 Original 偏置的效用正则，需要证明它带来的 utility 收益大于该偏置。

对 A=T² KL(p₀,T || p_θ,T)，同一前向状态下在 θ₀ 的精确 Hessian 是

  F_R,T = E_R J_zᵀ [diag(p₀,T)−p₀,Tp₀,Tᵀ] J_z。

T² 抵掉求导的 1/T²，但 p₀,T 仍依赖 T。因此 α=.1 不意味着“只有 10% 曲率”。解析二分类例：logit 差为 10 时，T=1 的 logit 曲率 p(1−p)≈0.000045396，T=2 为 0.006648057，约为前者 146 倍。这里只比较 KL 的局部曲率；不能据此宣称整个 ResNet 的 anchor 作用放大 146 倍。

这与 [physics_connections.md](physics_connections.md) 的 retain inverse-curvature 删除响应一致：正确目标是移动到未见忘集的解，而保持 Original 只是一种有限步保护近似。取消 anchor 是很小且有明确可反驳理由的消融；不用先添加新的损失模块。

## 4. 梯度投影：要分清保护了哪一项

以下全部用待**最小化**的 forget 目标 Q，而不是裸 forget CE 的梯度；把 ascent 的符号混进来会反向。记 q=∇Q、r=∇L_R、a=qᵀr。

若 a<0，单侧投影 q′=q−a r/||r||² 满足 rᵀq′=0。纯 forget 步 −ηq′ 不在一阶增加当前 retain minibatch 的 loss。但组合方向 v=r+λq′ 仍可能使 Q 上升：qᵀv=a+λ(||q||²−a²/||r||²)，符号不定。**保护 retain 不代表也保护 forgetting。**

明确反例：r=(1,0)，q=(−1,.1)，λ=.1，则 q′=(0,.1)，v=(1,.01)，qᵀv=−.999。因此 −ηv 在一阶增加 Q。该例是二维向量算式，不是实验结果。

若要两任务同时一阶不增，已有双侧 PCGrad 在 a<0 时使用

  q′=q−a r/||r||²，r′=r−a q/||q||²，v=r′+λq′。

则 rᵀv=||r||²−a²/||q||²≥0，qᵀv=λ(||q||²−a²/||r||²)≥0，由 Cauchy–Schwarz 直接得到。a≥0 时保持原梯度。完全反平行会把两方向都投影为 0，近反平行也可能很慢。

这些只是一阶、同一点、当前 batch 的结论。原模型接近 retain 驻点时 r≈0，utility 损害首先由曲率体现，平均梯度投影几乎看不到；平均 retain 梯度中的样本抵消也会隐藏个别样本损伤。momentum、weight decay、预条件和 BN 更新还会改变**实际**参数步，不能给原始梯度投影附会实际步的保证。若保留原优化器，只把上述作为启发式，不宣称保证；若专门验证保证，须投影实际步或采用明确的无动量/无衰减对照并完整报告改动。

**预算：**两个梯度分别 backward 并不必然多一遍数据，但保存全网梯度需要内存和归约成本。现有每 epoch 18 forget 和 159 retain backward；若改为每一个 retain batch 都配一个完整 forget batch，就把 forget passes 放大约 9 倍。只对 18 个配对步投影时，其余 141 retain 步仍可破坏 Q。可用每个 retain step 配较小 forget microbatch，让各点总访问数不变，但这会改变 forget batch 分布、W2 估计方差和更新步数；必须有同样配对/批量的无投影对照，不能声称一切预算完全相同。

## 5. Fisher 与 head 解为什么不是第一优先级

几何上更自然的是限制 retain 预测 KL 的二阶变化 δᵀF_Rδ，再在该预算内降低 Q。近端形式给出 δ=−η(F_R+γI)⁻¹q。该方向和已有自然梯度/信赖域相连，不是新理论。predictive Fisher 使用模型标签分布，true-label empirical Fisher 使用观测标签梯度外积，两者不同；完美拟合时后一种常以更快速度退化。Martens 对该差别和 damping 的必要性有专门分析。[JMLR 21, 2020，§6、§10、§11](https://arxiv.org/pdf/1412.1193)。

若冻结 feature h，只改 K 类线性 head W，则预测 Fisher 对角可直接估计为

  d_{kj}=E_R[p_k(1−p_k)h_j²]，

bias 用 h_0=1。它免去全网 per-sample gradient，但仍要 feature forward、缓存与阻尼 γ，且只删除 head 中的适应，不能移除 feature extractor 的记忆。精确保护所有 retain logits 要求 ΔW h_i=0；如果 40500 个 retain feature 在 512 维空间张满秩，唯一解就是 ΔW=0。这解释了为什么“head 闭式零损伤遗忘”通常不能无条件成立。低秩截断或近似保护会重新引入阈值/容差，应留到更有时间时。

## 6. 文献边界与可写进报告的话

- **PCGrad 已有。** Yu et al., *Gradient Surgery for Multi-Task Learning*, NeurIPS 2020，§2.3/Algorithm 1，PDF pp.3–4，给出互相冲突梯度的双侧投影；§2.4 明确存在反平行的次优停点。[原文](https://arxiv.org/pdf/2001.06782#page=3)。
- **MU 的投影已有。** Hoang et al., *Learn to Unlearn for Deep Neural Networks: Minimizing Unlearning Interference with Gradient Projection*, WACV 2024，§3.2–3.3，PDF pp.3–4，使用 retain representation SVD 子空间保护。[原文](https://openaccess.thecvf.com/content/WACV2024/papers/Hoang_Learn_To_Unlearn_for_Deep_Neural_Networks_Minimizing_Unlearning_Interference_WACV_2024_paper.pdf#page=3)。
- **逐样本 retain 梯度投影也已有。** Shamsian et al., *Go Beyond Your Means: Unlearning with Per-Sample Gradient Orthogonalization*, arXiv:2503.02312，§4.2/Algorithm 1，PDF pp.4–5，用 retain per-sample QR 基进行投影。论文聚焦 retain 数据有限的情形，不能把其结果直接比较到本项目 40500 个 retain 点。[原文](https://arxiv.org/pdf/2503.02312#page=4)。
- **“retain 学回去”的问题已有。** Patel et al., *Learning to Unlearn while Retaining: Combating Gradient Conflicts in Machine Unlearning*, ICCV 2025，PDF p.2 的动机、§3.2–3.3/PDF pp.3–4 的双层目标与梯度积分析。[原文](https://openaccess.thecvf.com/content/ICCV2025/papers/Patel_Learning_to_Unlearn_while_Retaining_Combating_Gradient_Conflicts_in_Machine_ICCV_2025_paper.pdf#page=4)。

可以写：我们从删除的局部响应分析出发，识别出固定 Original 蒸馏与删除位移之间的偏置，并计划通过取消 anchor、相同数据遍历的交错更新和已有梯度冲突对照来区分目标构造与优化交互。不能写：提出首个正交遗忘方法、物理证明了 ResNet 精确遗忘，或未测量就断言 AvgGap 有提升。
