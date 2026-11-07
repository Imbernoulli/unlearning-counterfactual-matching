# 从统计物理到 Retrain 匹配：可用联系与不该越过的边界

2026-11-06。只读研究；未修改仓库、未启动 GPU。以下没有文献实验数值；公式标明是原文结论还是本备忘录推导。研究对象是固定 Original 的 CIFAR-10/ResNet-18，随机删除 4,500/45,000，验证集 5,000，目标为同一删除划分下 Retrain 的行为。

## 1. 最有用的物理联系是 susceptibility，而不是“让模型更无序”

统计物理中的 cavity 操作拿掉一个相互作用因子，再问其余变量如何响应；leave-one-out 拿掉一个数据损失，再问最优参数如何响应。二者共享“局部扰动 × 系统响应算子”的结构。Obuchi–Kabashima 的 LASSO 文章 §3.1 把 cavity susceptibility 写为 β 乘协方差，式 (24)，再在式 (30) 以线性响应连接 full 与 cavity 解。注意其小参数来自设计矩阵元素的尺度，不能把该渐近结论直接移到 ResNet。[原文：arXiv:1601.00881，PDF pp.6–7，式 (24)、(30)](https://arxiv.org/pdf/1601.00881#page=6)。

同作者的多项 logistic regression 文章更接近本项目：§2.1、式 (18)–(25) 从驻点方程推导 LOO 参数及 logit 修正，并用 Woodbury 避免逐样本求逆；保留 active set 是其关键假设。其附录/§3.3 还明确讨论特征异质性与近似失败，不能把每个类别压成一个相同“平均介质”。[JMLR 19(52), 2018，PDF pp.4–5，§2.1；§3.3](https://jmlr.org/papers/volume19/17-684/17-684.pdf#page=4)。

Kubo 的线性响应理论给出更广的物理背景：平衡系统受到外场后，响应由平衡关联函数表达。相关原文是 §5、印刷 pp.263–265、式 (5.5)–(5.8)。下面的静态 Gibbs 协方差恒等式可以直接求导得到，是这里使用的简洁版本；不需要假装 ResNet 的 SGD 动力学满足原文所有平衡假设。[Kubo, Reports on Progress in Physics 29, 255–284, 1966，§5](https://doi.org/10.1088/0034-4885/29/1/306)。

## 2. 删除的符号、归一化和正则项不能弄错

令 f=|F|/|D|，并对 X∈{D,R,F} 定义同一正则化强度的平均风险

  L_X(θ)=mean_{i∈X} ℓ_i(θ)+λ r(θ).

则严格有 L_D=(1−f)L_R+f L_F。令 θ₀ 为 Original，g_X=∇L_X(θ₀)，H_R=∇²L_R(θ₀)。对 retain 风险做一次 Newton 近似，得到

  δ_Newton = −H_R^{-1} g_R
            = H_R^{-1} [f g_F−g_D]/(1−f).

若 Original 确在这个目标的驻点，g_D=0，则

  δ ≈ [f/(1−f)] H_R^{-1} g_F.

**这是正号**，与 forget gradient ascent 同号，但不是普通 Euclidean ascent：H_R^{-1} 是其余数据对扰动的响应。10% 删除时系数为 1/9，不是随意省去的比例。g_F 包含上面定义的正则项；若把 g_F 改为裸 CE 梯度，必须同时处理正则化差额。Original 是有限步、增强、动量 SGD 的 checkpoint，g_D=0 也不能默认成立。

这不是新影响函数公式。Koh–Liang §2.1 式 (1) 的上权重影响为 −H_D^{-1}∇ℓ_i；删除对应负权重，所以符号翻正。其理论明确假设正定 Hessian。[ICML 2017，PDF p.2，式 (1)](https://proceedings.mlr.press/v70/koh17a/koh17a.pdf#page=2)。Guo 等已经把 Newton removal 用于强凸线性模型，并对残余梯度给界；批量删除见式 (6)。这些认证结论不能直接套到非凸全网 ResNet。[ICML 2020，§3.1、式 (3)、Theorem 1；§3.2、式 (6)](https://proceedings.mlr.press/v119/guo20c/guo20c.pdf#page=2)。

### 一个精确的反例：删除 10% 不保证“曲率也只变化 10%”

这是本备忘录的标量二次例子，不是论文实验。设 L_R(θ)=θ²/2，L_F(θ)=100(θ−1)²/2，f=.1。则 Original θ₀=10/10.9≈.91743，真正 retain 最优解是 0。用 retain 曲率 H_R=1 的上式给 δ=−.91743，二次模型下完全正确；把 H_R 偷换成 Original 曲率 H_D=10.9，则只走 δ≈−.08417，残留 θ≈.83326。少数被删点可以占据大部分曲率，比例小不等于扰动小。

## 3. Gibbs 后验删除：一个严格恒等式，和一个危险误用

假设学习对象真是 Gibbs 分布

  π_D(θ) ∝ π_prior(θ) exp[−β Σ_{i∈D}ℓ_i(θ)]，

并且相关归一化常数存在，则删掉 F 后严格有

  π_R(θ) = π_D(θ) exp[+β E_F(θ)] /
           E_{π_D}[exp(+β E_F)]， E_F=Σ_Fℓ_i.

这里假定 prior 与数据量无关。若把第一节的“平均 loss + 固定 weight decay”改写成 sum-energy，则正则能量为 nλr，删除后变为 (n−m)λr；这时倾斜中的 E_F 还要包含 mλr，不能遗漏这一归一化差异。

这就是逆似然重加权，不是“增加温度”。Nguyen 等的 Bayesian unlearning §3.1 式 (2) 已直接给出相同 Bayes 除法；§3.2 Proposition 2/式 (3) 用 posterior KL 与删集似然构造目标。[NeurIPS 2020，PDF p.3，式 (2)–(3)](https://proceedings.neurips.cc/paper_files/paper/2020/file/b8a6550662b363eb34145965d64d0cfb-Paper.pdf#page=3)。

令 π_t∝π_D exp(tβ E_F)。对任意不显含 t 的可积可微统计 A，直接求导可得本备忘录恒等式

  d E_{π_t}[A]/dt = β Cov_{π_t}(A,E_F)，
  E_{π_R}[A]−E_{π_D}[A] = ∫₀¹ β Cov_{π_t}(A,E_F) dt.

因此线性响应是用 t=0 的协方差近似整条路径积分。若仅在单一稳定模式附近做 Laplace 近似，Cov(θ)≈(βH_D)^{-1}，并线性化 E_F，则均值响应正好还原 H_D^{-1}g_F。若保留二次项，删除也改变曲率，局部精度从 H_D 变成 H_D−H_F；这是上一节反例的来源。

**不能把这个等式直接应用于一个 SGD 点估计。** 一个 checkpoint 没有提供精确 π_D、其尾部或多模式质量；把 δ_{θ₀} 当 posterior 后重加权甚至不会移动它。近似 Gaussian 漏掉的尾部会被逆似然因子放大，出现不稳定。Rawat 等专门展示 Bayesian unlearning 的此类问题，并讨论 Laplace 的单模式局限。[Challenges and Pitfalls of Bayesian Unlearning, 2022，§2、式 (1)，§2.1–2.3](https://arxiv.org/pdf/2207.03227#page=1)。所以这里的 Gibbs 表达适合做推导和诊断，不应直接作为“物理保证”。

## 4. 可落地的两项小实验

### A. 用 retain 响应度保护现有分布目标

不改现在的非成员目标，只把其梯度 g_Q 用保留集的 PSD metric M_R 预条件：

  δ = −η(M_R+γI)^{-1}g_Q.

该式严格最小化局部模型 g_Q^Tδ + δ^T(M_R+γI)δ/(2η)。若 M_R 为 predictive Fisher，它近似惩罚 retain 预测 KL；若用 centered-logit Jacobian 的 J_R^TJ_R，则直接惩罚保留集 logits 的线性化改动。它让“目标走向非成员”和“哪里改动代价小”分工明确。只需要步长 η 与 damping γ 两个标量；首轮只在 fc/最后 block 里做少量固定迭代 CG 或块对角求解。

这不是首创自然梯度/影响函数；实验价值在于检验它是否稳定当前验证集目标。预算必须包含估计曲率/HVP 的 retain passes，不要在“相同十次训练”背后悄悄加算力。不要直接逆全网不定 Hessian。另一个坑是 perfect-fit 时 true-label empirical Fisher 的 (p−e_y)² 接近零，它和模型 predictive Fisher/GGN 不是同一个矩阵；加 damping，并用独立 validation utility 设 trust region。

### B. 比较一次强删除与分段 continuation

沿 t:0→1 逐步削弱 forget 因子，使用两次半步、每步更新 retain metric；和一次整步保持总数据访问/线性求解预算相同。冻结网络大部分参数，在小子空间先验证：当真实 finite deletion curvature 改变明显时，分段更新是否能减少 retain 损伤和 reference gap。只用 calibration 选择 damping/步长，最终一次 test；报告变化是来自目标还是更温和的步长。

此想法来自上面的 thermodynamic integration/cavity expansion，仍是启发式。若收益全由多算一步带来，不能声称物理机制带来突破。

## 5. 非凸的硬边界

一个简单能量反例是 E_h(θ)=(θ²−1)²−hθ。当 h 从很小的正值变为负值时，全局最优模式从 +1 附近切到 −1 附近；单一模式内的一阶响应仍只给很小位移。小的数据扰动可以改变盆地选择，局部 Hessian 或 Fisher 不会自动复现从头训练的优化轨迹。ResNet 的缩放对称性、BatchNorm 状态、参数非识别性又会使裸参数空间响应更难解释。

因此论文可诚实写：物理视角提示要保留响应各向异性、检查删除路径，并约束效用；不能写“统计物理证明了非凸 ResNet 的精确遗忘”。当前 UA/RA/TA/confidence-MIA 也只是有限行为统计，不是 posterior 等价的检验。
