# 附录：从 KL-regularized RL 到 DPO 与 GRPO 系列（统一符号 & 从目标到算法）

> 在同一符号体系下，把 **KL-regularized RL 的闭式最优解**、**BT/PL 偏好模型→DPO**、以及 **GRPO→DAPO/GSPO**（含 KL 估计与实现坑）统一成一条从目标到算法的链条。关键等价变换、常数项消掉的原因、以及 KL 梯度的正确实现方式都会写清楚。

---

## A. 统一符号

- **输入/提示（prompt）**：$x \in \mathcal{X}$
- **输出序列（response）**：$y = (y_1, \dots, y_T) \in \mathcal{Y}$，长度 $T = |y|$
- **自回归状态**：$s_t := (x, y_{<t})$，动作（token）$a_t := y_t$
- **策略模型（policy / actor）**：$\pi_\theta(y \mid x)$

$$
\pi_\theta(y \mid x) = \prod_{t=1}^{T} \pi_\theta(y_t \mid x, y_{<t})
$$

- **参考策略（reference / SFT / pretrain）**：$\pi_{\text{ref}}(y \mid x)$
- **行为策略（采样用旧策略）**：$\pi_{\text{old}}(y \mid x)$（一般是上一轮 $\theta$ 的拷贝）
- **序列级奖励（标量）**：$R(x,y) \in \mathbb{R}$（奖励模型或规则奖励）
- **KL 系数**：$\beta > 0$
- **PPO/GRPO 裁剪阈值**：$\epsilon$（DAPO 会拆成 $\epsilon_{\text{low}}, \epsilon_{\text{high}}$）
- **组采样大小（GRPO）**：对每个 $x$，采样 $G$ 个候选 $y^{(1)}, \dots, y^{(G)}$

为方便写法，定义 token 级 logprob：

$$
\ell^\theta_t := \log \pi_\theta(y_t \mid s_t), \quad
\ell^{\text{ref}}_t := \log \pi_{\text{ref}}(y_t \mid s_t), \quad
\ell^{\text{old}}_t := \log \pi_{\text{old}}(y_t \mid s_t)
$$

以及序列级 logprob：

$$
\log \pi_\theta(y \mid x) = \sum_{t=1}^{T} \ell^\theta_t
$$

---

## B. KL-regularized RL 的闭式最优解：$\pi^*(y \mid x) \propto \pi_{\text{ref}}(y \mid x)\exp(R/\beta)$

### B.1 目标（对每个 $x$ 的条件优化）

考虑经典的 KL 正则化 RL（也可视为最大熵 / 指数倾斜）目标：

$$
\max_{\pi(\cdot \mid x)}\;
J(\pi; x)
:= \mathbb{E}_{y \sim \pi(\cdot \mid x)}[R(x,y)]
- \beta\,\mathrm{KL}\!\left(\pi(\cdot \mid x) \,\|\, \pi_{\text{ref}}(\cdot \mid x)\right)
$$

展开 KL（注意这是 **reverse KL**：$\mathrm{KL}(\pi \| \pi_{\text{ref}})$）：

$$
J(\pi; x) = \sum_{y} \pi(y \mid x) \Bigl(
R(x,y) - \beta \log\frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)}
\Bigr)
$$

约束：$\sum_y \pi(y \mid x) = 1$，且 $\pi(y \mid x) \ge 0$。

### B.2 拉格朗日法严格推导

对固定 $x$，写拉格朗日函数：

$$
\mathcal{L}(\pi, \lambda)
= \sum_y \pi_y \Bigl( R_y - \beta \log\frac{\pi_y}{\pi^{\text{ref}}_y} \Bigr)
+ \lambda \Bigl( \sum_y \pi_y - 1 \Bigr)
$$

其中 $\pi_y := \pi(y \mid x)$，$R_y := R(x,y)$，$\pi^{\text{ref}}_y := \pi_{\text{ref}}(y \mid x)$。

对 $\pi_y$ 求偏导并令 0：

$$
\frac{\partial \mathcal{L}}{\partial \pi_y}
= R_y - \beta \Bigl( 1 + \log\frac{\pi_y}{\pi^{\text{ref}}_y} \Bigr) + \lambda
= 0
$$

移项得

$$
\log\frac{\pi_y}{\pi^{\text{ref}}_y} = \frac{R_y}{\beta} + \frac{\lambda}{\beta} - 1
$$

指数化：

$$
\pi_y = \pi^{\text{ref}}_y \exp\!\Bigl(\frac{R_y}{\beta}\Bigr) \cdot \exp\!\Bigl(\frac{\lambda}{\beta} - 1\Bigr)
$$

令归一化常数

$$
Z(x) = \sum_{y} \pi_{\text{ref}}(y \mid x) \exp\!\Bigl(\frac{R(x,y)}{\beta}\Bigr)
$$

则

$$
\boxed{
\pi^*(y \mid x) = \frac{\pi_{\text{ref}}(y \mid x) \exp(R(x,y)/\beta)}{Z(x)}
}
$$

> **关键工程含义**：
>
> - $\pi^*$ 是对 $\pi_{\text{ref}}$ 的"指数倾斜"（exponential tilting）。
> - 对任意只依赖 $x$ 的常数 $c(x)$，将奖励替换为 $R'(x,y) = R(x,y) + c(x)$ 不改变 $\pi^*(y \mid x)$（因为会被 $Z(x)$ 抵消）。

这条闭式解正是统一链条的"起点"。

---

## C. 从 BT/PL 偏好模型推到 DPO loss（同一符号体系）

DPO 的核心是：**不显式训练奖励模型**，而是用偏好数据直接优化策略。DPO 论文明确把它解释为：通过变量替换，把"奖励模型的偏好似然"改写成"策略的偏好似然"[^1]。

### C.1 Bradley–Terry（BT）偏好模型（pairwise，K=2）

数据集：

$$
\mathcal{D} = \{(x,\, y_w,\, y_l)\}
$$

其中 $y_w$ 为偏好更好的（winner/chosen），$y_l$ 为更差的（loser/rejected）。

BT 模型假设存在一个（潜在）标量奖励 $r(x,y)$ 使得

$$
\mathbb{P}(y_w \succ y_l \mid x)
= \sigma\bigl(r(x,y_w) - r(x,y_l)\bigr)
$$

其中 $\sigma(z) = \frac{1}{1+e^{-z}}$。

奖励模型训练（最大化偏好似然）等价于最大化：

$$
\max_r\; \mathbb{E}_{(x,y_w,y_l) \sim \mathcal{D}}
\Bigl[\log \sigma(r(x,y_w) - r(x,y_l))\Bigr]
$$

### C.2 "变量替换"：用策略比值表达奖励（核心桥梁）

由附录 B 的闭式最优策略，我们知道：若某个奖励 $r$ 对应的 KL-regularized RL 最优策略为 $\pi^*$，则（对固定 $x$）存在

$$
\pi^*(y \mid x) \propto \pi_{\text{ref}}(y \mid x) \exp(r(x,y)/\beta)
$$

等价于（把 $Z(x)$ 吸收进 $x$-only 常数）：

$$
r(x,y) = \beta \log\frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)
$$

注意 $\beta \log Z(x)$ **只依赖 $x$**，在做差 $r(x,y_w) - r(x,y_l)$ 时会完全抵消。

因此，在 DPO 中把"未知的最优策略 $\pi^*$"用待训练策略 $\pi_\theta$ 近似，并定义**隐式奖励**：

$$
\tilde{r}_\theta(x,y) := \beta \log\frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}
$$

则

$$
\tilde{r}_\theta(x,y_w) - \tilde{r}_\theta(x,y_l)
= \beta \Bigl[
\log\frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)}
- \log\frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}
\Bigr]
$$

代回 BT 似然，得到 DPO 的二元交叉熵形式目标（最大化）：

$$
\boxed{
\max_\theta\;
\mathbb{E}_{(x,y_w,y_l) \sim \mathcal{D}}
\Bigl[
\log \sigma\Bigl(
\beta \log\frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)}
- \beta \log\frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}
\Bigr)
\Bigr]
}
$$

这就是 DPO 论文所强调的"简单 BCE 目标"[^1]。

> **工程解释**：DPO 实际在做的是把"winner 的 log-ratio"推高、把"loser 的 log-ratio"拉低；而 log-ratio 是对参考模型的"偏离幅度"。

### C.3 Plackett–Luce（PL）偏好模型（ranking，K>2）

当每个 prompt $x$ 有 $K$ 个候选 $\{y_1, \dots, y_K\}$ 且标注给出一个排序 $\tau$（$\tau_1$ 最优），PL 模型定义

$$
\mathbb{P}(\tau \mid x, \{y_i\}) =
\prod_{k=1}^K
\frac{\exp(r(x, y_{\tau_k}))}{\sum_{j=k}^K \exp(r(x, y_{\tau_j}))}
$$

同样做变量替换 $r \mapsto \tilde{r}_\theta = \beta \log(\pi_\theta / \pi_{\text{ref}})$（注意任何 $x$-only 常数在分子分母都抵消），得到 **PL-DPO**（最大化对数似然）：

$$
\boxed{
\max_\theta\;
\mathbb{E}\Bigl[
\sum_{k=1}^K
\Bigl(
\tilde{r}_\theta(x, y_{\tau_k})
- \log\sum_{j=k}^K \exp(\tilde{r}_\theta(x, y_{\tau_j}))
\Bigr)
\Bigr]
}
$$

且当 $K=2$ 时可化归为 BT→DPO。

> **与 pairwise 拆分的关系**：把排序拆成两两比较会引入统计不一致性（"胜负关系"不一定等价于"全排序"似然），PL 形式是更严格的 ranking 建模。

---

## D. GRPO：如何替代 critic、如何构造 group advantage（并连回同一目标）

GRPO 在 DeepSeekMath/DeepSeek-R1 体系里被描述为：**一种 PPO 变体，去掉 critic（价值模型），用"组内相对优势"当 baseline**[^2]。

### D.1 GRPO 的"组采样→组优势"定义（DeepSeek-R1 公式）

对每个问题/提示 $q$（即本文的 $x$），从旧策略采样一组输出：

$$
\{o_1, \dots, o_G\} \sim \pi_{\text{old}}(O \mid q)
$$

并得到组内奖励 $\{r_1, \dots, r_G\}$。DeepSeek-R1 定义组优势为标准化的组内相对分数[^3]：

$$
\boxed{
A_i = \frac{r_i - \mathrm{mean}(\{r_j\}_{j=1}^G)}{\mathrm{std}(\{r_j\}_{j=1}^G)}
}
$$

这一步的本质就是：用同一个 $x$ 下的多样本近似估计一个"状态相关 baseline"，从而降低方差、避免训练一个 $V_\psi$。

> **严谨备注（偏差/无偏）**：
>
> - 理论上，为保证 baseline 不依赖当前样本动作，最"干净"的做法是 leave-one-out baseline（排除自身样本）——很多工程实现会直接用全组 mean/std，带来一个可控的尺度偏差（常见影响是把梯度整体缩放约 $1-1/G$）。
> - 这解释了为什么在工程里 $G$ 一般不会太小：$G$ 越大，组统计越稳定、偏差越小。

### D.2 GRPO 的优化目标（从"序列级"到"token 级"的两种写法）

DeepSeek-R1 给出的 GRPO 目标（写成序列级比值）为[^3]：

$$
J_{\text{GRPO}}(\theta) =
\mathbb{E}\Bigl[
\frac{1}{G}\sum_{i=1}^G
\Bigl(
\min(\rho_i(\theta) A_i,\; \mathrm{clip}(\rho_i(\theta), 1-\epsilon, 1+\epsilon) A_i)
- \beta\,D_{\mathrm{KL}}(\pi_\theta \| \pi_{\text{ref}})
\Bigr)
\Bigr]
$$

其中序列级重要性比：

$$
\rho_i(\theta) = \frac{\pi_\theta(o_i \mid q)}{\pi_{\text{old}}(o_i \mid q)}
$$

而 DeepSeekMath 论文在 PPO→GRPO 的推导中给出了更"PPO 风格"的 **token-level surrogate**（对每个 token 做 ratio 与 clip，并对长度平均），这是工程实现更常用的形式[^2]。

二者并不冲突，因为自回归分解保证：

$$
\log\rho_i(\theta)
= \log\frac{\pi_\theta(y^{(i)} \mid x)}{\pi_{\text{old}}(y^{(i)} \mid x)}
= \sum_{t=1}^{|y^{(i)}|} \log\frac{\pi_\theta(y^{(i)}_t \mid s^{(i)}_t)}{\pi_{\text{old}}(y^{(i)}_t \mid s^{(i)}_t)}
$$

因此你可以：

- **序列级**：把同一个 $A_i$ 乘在整条序列的 $\rho_i$ 上；
- **token 级**：把同一个 $A_i$ "均摊/共享"给每个 token，做 token-level ratio & clip，再对 token 求平均（或加权平均）。

> 这也是后续 GSPO 为什么要强调"奖励是 sequence-level，就该做 sequence-level correction"的根源：**你在什么层面定义权重与裁剪，决定了方差与稳定性**。

### D.3 GRPO 的 KL 正则：是"加到 reward"还是"加到 loss"？

DeepSeekMath 明确提到一种工程选择：**GRPO 不把 KL penalty 塞进 reward，而是把 $\pi_\theta$ 与 $\pi_{\text{ref}}$ 的 KL 直接加到 loss**，以免干扰优势项 $\hat{A}$ 的计算[^2]。

DeepSeek-R1 中还给出了一个具体的 $D_{\mathrm{KL}}$ 形式[^3]：

$$
D_{\mathrm{KL}} = \frac{\pi_{\text{ref}}(o_i \mid q)}{\pi_\theta(o_i \mid q)}
- \log\frac{\pi_{\text{ref}}(o_i \mid q)}{\pi_\theta(o_i \mid q)} - 1
$$

它等价于在采样 $o_i \sim \pi_\theta$ 下对 $\mathrm{KL}(\pi_\theta \| \pi_{\text{ref}})$ 的一种无偏（带控制变量）的单样本估计形式（见下一节 KL 估计）[^4]。

---

## E. KL 估计：k1/k2/k3 的"数值估计"与"梯度实现"必须分开看

这一节是工程里最容易"看起来对、实际错"的地方：**KL 的标量估计无偏 ≠ KL 梯度估计无偏**。Tang & Munos（2025）专门指出：把 Monte-Carlo KL 估计当作 loss 直接 autograd 往往会得到错误的 KL 梯度[^4]。

### E.1 三类常见单样本 KL 估计（以 $y \sim \pi$ 为采样分布）

设我们要估计

$$
\mathrm{KL}(\pi \| \pi_{\text{ref}}) = \mathbb{E}_{y \sim \pi}\Bigl[\log\frac{\pi(y)}{\pi_{\text{ref}}(y)}\Bigr]
$$

Tang & Munos 总结了几个常用单样本估计器[^4]：

1. **Vanilla（无偏，高方差）**

$$
d_{\text{vanilla}}(y) = \log\frac{\pi(y)}{\pi_{\text{ref}}(y)}
$$

2. **Variance-reduced（控制变量，无偏，方差更低）**

$$
d_{\text{vr}}(y) = \log\frac{\pi(y)}{\pi_{\text{ref}}(y)} + \frac{\pi_{\text{ref}}(y)}{\pi(y)} - 1
$$

NormalUhr 的读书笔记把它与 Schulman 风格的 $k_1, k_2, k_3$ 关系讲得很清楚：其中 $\frac{\pi_{\text{ref}}}{\pi} - 1$ 是零均值控制变量，用来降方差[^5]。

3. **Squared（偏置但低方差，$\pi \approx \pi_{\text{ref}}$ 时 MSE 小）**

$$
d_{\text{squared}}(y) = \frac{1}{2}\Bigl(\log\frac{\pi(y)}{\pi_{\text{ref}}(y)}\Bigr)^2
$$

### E.2 "最大坑"：把 KL 估计当 loss 直接求导，往往不是你想要的 KL 梯度

Tang & Munos 给出非常关键的结论[^4]：

- 直接对 vanilla 估计 $d_{\text{vanilla}} = \log\frac{\pi}{\pi_{\text{ref}}}$ 求导，其期望梯度为 0（也就是基本**没正则效果**）。
- 对 variance-reduced 估计求导，得到的期望梯度会变成 $\nabla \mathrm{KL}(\pi_{\text{ref}} \| \pi)$（方向"交换了"），这在与 reward 组合时会改变最优解。
- 反而对 squared 估计求导，会得到正确的"vanilla KL 梯度估计"：

$$
\mathbb{E}\bigl[\nabla d_{\text{squared}}(y)\bigr]
= \mathbb{E}_{y \sim \pi}\Bigl[
\log\frac{\pi(y)}{\pi_{\text{ref}}(y)}\; \nabla \log\pi(y)
\Bigr]
= \nabla \mathrm{KL}(\pi \| \pi_{\text{ref}})
$$

并且他们指出一种工程上更常用的等价实现：用 stop-gradient（detach）把 $\log\frac{\pi}{\pi_{\text{ref}}}$ 当作系数，只对 $\log\pi$ 求导，从而得到同样的梯度[^4]：

$$
\mathcal{L}_{\text{KL-grad-correct}}
= \mathrm{sg}\!\Bigl(\log\frac{\pi(y)}{\pi_{\text{ref}}(y)}\Bigr) \cdot \log\pi(y)
\quad\Rightarrow\quad
\nabla \mathcal{L}
= \log\frac{\pi}{\pi_{\text{ref}}}\; \nabla\log\pi
$$

> **把这句话写进你的训练代码 review checklist：**
> 你想要的是 $\nabla \mathrm{KL}(\pi \| \pi_{\text{ref}})$，而不是"某个 KL 数值估计器的 autograd"。

### E.3 token-level KL shaping（与自回归因子分解的严格关系）

序列级 KL 的 integrand 是 log-ratio，利用自回归分解：

$$
\log\frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}
= \sum_{t=1}^{T} \log\frac{\pi_\theta(y_t \mid s_t)}{\pi_{\text{ref}}(y_t \mid s_t)}
$$

这就是为什么 RLHF/GRPO 工程里经常把 KL penalty 写成 **token-level shaping**：

- 要么作为"每步奖励惩罚"加入 return；
- 要么作为"每 token loss 组件"加进去（但要注意上一小节的梯度正确性问题）。

NormalUhr 的笔记把"PPO 与 GRPO 中的 KL 估计差异"映射到这三类估计器的 bias-variance 行为上，可作为实现参考[^5]。

---

## F. GRPO → DAPO → GSPO：同一链条上的"实现层"演化

下面把 DAPO/GSPO 放进同一符号体系，强调它们改的不是"目标本质"，而是**方差、长度偏置、采样浪费、以及 importance ratio 的粒度错配**。

### F.1 DAPO：在 GRPO 框架里做"四件大事"（clip / sampling / loss / length）

swift 文档给出的定义：**DAPO = Decoupled Clip and Dynamic sAmpling Policy Optimization**，基于 GRPO 加入多个工程技巧[^6]：

1. **Clip-Higher（非对称裁剪）**
   把裁剪区间从对称 $[1-\epsilon, 1+\epsilon]$ 改为非对称 $[1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}}]$，提高上界鼓励探索（文档举例上界可到 0.28，而下界保持 0.2）[^6]。

2. **Dynamic Sampling（动态重采样）**
   当组内奖励方差为 0 时，组优势 $A_i$ 全为 0，梯度消失。DAPO 会跳过这类样本并重采样直到 batch 填满[^6]。

3. **Token-level Loss（长度偏置修正）**
   文档指出：GRPO 的句子级归一化会引入响应长度偏置；DAPO 改为 token-level 归一化以避免这一问题[^6]。

4. **Overlong Filtering / Soft Overlong Punishment（过长样本处理）**
   包括过滤被截断样本、以及分段长度惩罚函数（超过阈值线性惩罚、再超过则强惩罚到 -1）[^6]。

用统一符号写出 DAPO 的核心 surrogate 形态（概念式）：

$$
\min\bigl(\rho_{i,t}\, A_i,\; \mathrm{clip}(\rho_{i,t},\, 1-\epsilon_{\text{low}},\, 1+\epsilon_{\text{high}})\, A_i\bigr)
$$

其余三项主要影响的是**采样分布与 loss 的归一化方式**，从而改变方差与 token 效率，而不改变"KL-regularized RL + group baseline"的主干。

### F.2 GSPO：把 importance sampling 从 token-level 提升到 sequence-level

swift 的 GSPO 文档给出非常明确的动机[^7]：

- GRPO 做 token-level importance weight：每个 token 只采样一次，无法有效做分布校正，反而引入高方差噪声，甚至导致训练崩溃；
- 目标单位应该与 reward 单位一致：reward 通常是 sequence-level，因此更合理做 sequence-level correction。

三种权重写法（swift 文档直接给出）[^7]：

**1. GRPO（token-level）**

$$
w^{\mathrm{GRPO}}_{i,t} =
\frac{\pi_\theta(y_{i,t} \mid x, y_{i,<t})}{\pi_{\text{old}}(y_{i,t} \mid x, y_{i,<t})}
$$

**2. GSPO（sequence-level，长度归一化）**

$$
\boxed{
w^{\mathrm{GSPO}}_{i}
= \left[\frac{\pi_\theta(y_i \mid x)}{\pi_{\text{old}}(y_i \mid x)}\right]^{\frac{1}{|y_i|}}
= \exp\Bigl(\frac{1}{|y_i|}\sum_{t=1}^{|y_i|}\log\frac{\pi_\theta(y_{i,t} \mid s_{i,t})}{\pi_{\text{old}}(y_{i,t} \mid s_{i,t})}\Bigr)
}
$$

**3. GSPO-token（sequence weight + token pathwise，带 stop-gradient）**

$$
w^{\mathrm{GSPO\text{-}token}}_{i,t}
= \mathrm{sg}[w^{\mathrm{GSPO}}_i] \cdot
\frac{\pi_\theta(y_{i,t} \mid s_{i,t})}{\mathrm{sg}[\pi_\theta(y_{i,t} \mid s_{i,t})]}
$$

swift 文档备注：当每个 token 的优势相同（GRPO 常见做法是按句子级 reward 归一化后共享优势）时，GSPO-token 与 GSPO 在梯度上等价[^7]。

> **把 GSPO 放回"从目标到算法"的链条里看**：
>
> - 目标没变：依然是在做 KL-regularized RL 的近似；
> - 改的是 off-policy correction 的粒度：从 token-level 的高方差权重，变成 sequence-level 的低方差权重（并用长度归一化稳定尺度）。

---

## G. 最终统一：一条"从目标到算法"的链

把所有东西串成一条链，你会得到一个非常清晰的"路线图"：

**1. 目标层（统一起点）**

$$
\max_{\pi}\; \mathbb{E}[R] - \beta\,\mathrm{KL}(\pi \| \pi_{\text{ref}})
$$

并且有闭式最优策略

$$
\pi^*(y \mid x) \propto \pi_{\text{ref}}(y \mid x) \exp(R/\beta)
$$

**2. 在线优化（policy gradient 近似这条目标）**

- **PPO**：用 critic/GAE 得到 $A_t$，用 ratio+clip 做多 epoch 更新（trust region 近似）
- **GRPO**：不训练 critic，改用**组内 baseline**（mean/std）构造 $A_i$，仍然用 ratio+clip 更新；在 DeepSeek-R1 中明确给出组优势与 GRPO 目标[^3]。

**3. KL 在工程中的两个"正交问题"**

- **KL 数值怎么估？**（k1/k2/k3 讨论 bias-variance）[^5]
- **KL 梯度怎么做对？**（不要直接 autograd vanilla；用 squared 或 stop-grad 构造正确梯度）[^4]

**4. GRPO 的工程演化**

- **DAPO**：clip 非对称、动态采样、token-level loss、过长惩罚/过滤——主要解决效率、长度偏置和无效样本问题[^6]。
- **GSPO**：把 importance weight 从 token 级提升到 sequence 级，与 reward 粒度对齐，降低高方差，特别针对 MoE 不稳定[^7]。

**5. 离线优化（把同一目标"换一条路走"）**

- 偏好数据 + BT/PL 假设
- 通过闭式解给出的变量替换 $r \leftrightarrow \beta\log(\pi/\pi_{\text{ref}})$
- 得到 **DPO（或 PL-DPO）**：用简单 BCE / 排序似然直接优化策略，而不显式训练奖励模型[^1]。

---

## 参考文献

[^1]: Rafailov, R., et al. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." arXiv:2305.18290. <https://arxiv.org/pdf/2305.18290>

[^2]: Shao, Z., et al. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." arXiv:2402.03300. <https://arxiv.org/pdf/2402.03300>

[^3]: DeepSeek-AI. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." arXiv:2501.12948. <https://arxiv.org/pdf/2501.12948>

[^4]: Tang, Y. & Munos, R. (2025). "On the Estimation and Optimization of KL Divergence for RL-based LLM Training." arXiv:2506.09477. <https://arxiv.org/pdf/2506.09477>

[^5]: NormalUhr. "KL Divergence Estimator in RL for LLMs." Hugging Face Blog. <https://huggingface.co/blog/NormalUhr/kl-divergence-estimator-rl-llm>

[^6]: Swift Documentation — DAPO. <https://swift.readthedocs.io/en/latest/Instruction/GRPO/AdvancedResearch/DAPO.html>

[^7]: Swift Documentation — GSPO. <https://swift.readthedocs.io/en/latest/Instruction/GRPO/AdvancedResearch/GSPO.html>