# 附录：KL 方向、估计器、梯度实现、Batch 组织与长度偏置——统一符号的完整工程参考

> 本附录在同一套符号体系下，覆盖以下五个关键模块：
>
> 1. 目标到底是哪种 KL（reverse / forward；normalized / unnormalized；token-level / sequence-level）
> 2. KL 的估计器（k1 / 控制变量 k3 / squared k2）与 bias–variance
> 3. KL 梯度怎么写才对（stop-grad / squared / 常见错误写法）
> 4. PPO / GRPO / DAPO / GSPO 的 batch 组织：prompt → group → token（以及 importance weight 的"单位"）
> 5. advantage 标量共享 + 不同归一化（sample-level vs token-level）为何会造成长度偏置（length bias）

---

## 统一符号体系（全附录通用）

- 数据集（prompt 分布）：$x \sim \mathcal{D}$，其中 $x$ 是 prompt / query / state（在 LLM 里就是上下文）。
- completion（token 序列）：$y = (y_1, \dots, y_T)$，$T = |y|$
- "历史状态"（token 级 state）：$h_t = (x, y_{<t})$
- policy（要训练的 actor）：

$$
\pi_\theta(y \mid x) = \prod_{t=1}^T \pi_\theta(y_t \mid h_t)
$$

- reference policy（对齐约束用的参考）：$\pi_{\text{ref}}$
- behavior / old policy（采样 rollout 的策略）：$\pi_{\text{old}}$，在 PPO/GRPO 中通常是"本次 rollout 时冻结的快照"。
- 序列级 reward（RM 或规则打分）：$R(x,y) \in \mathbb{R}$，通常是整条 completion 的分数（对齐/推理任务里很常见）。
- token 级 shaping reward（把目标写成逐 token 的形式时）：$r_t(x,y)$ 或 $\tilde{r}_t(x,y)$
- KL 系数：$\beta > 0$

---

## A. 目标：reverse-KL 还是 forward-KL？

### A.1 RLHF 里最常见的"KL-regularized RL"目标（reverse KL）

最标准的 KL 正则化 RL 目标是（对每个 prompt）：

$$
J(\theta) = \mathbb{E}_{x \sim \mathcal{D}}\left[
\mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)}[R(x,y)]
- \beta\,\mathrm{KL}\!\big(\pi_\theta(\cdot \mid x) \,\|\, \pi_{\text{ref}}(\cdot \mid x)\big)
\right]
$$

这里的 KL 是 **reverse KL**（policy 相对 reference）：

$$
\mathrm{KL}(\pi_\theta \| \pi_{\text{ref}})
= \mathbb{E}_{y \sim \pi_\theta}\left[\log\frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}\right]
$$

这套形式在很多 RLHF / KL-regularized policy optimization 推导里都出现，并且它有一个关键性质：**存在闭式最优解**（见 A.3），这正是 DPO 推导的起点之一[^1]。

### A.2 forward KL：什么时候会冒出来？

forward KL 是：

$$
\mathrm{KL}(\pi_{\text{ref}} \| \pi_\theta)
= \mathbb{E}_{y \sim \pi_{\text{ref}}}\left[\log\frac{\pi_{\text{ref}}(y \mid x)}{\pi_\theta(y \mid x)}\right]
$$

在 RLHF 主流目标里它**不常作为主正则项**，但它会以两种方式"偷偷出现"：

1. **你用某些 KL 单样本估计器（尤其是 control-variate 版本）并且"直接对估计器做 autograd"** 时，梯度会变成 forward KL 的梯度（详见 C.3）[^1]。
2. 你若刻意设计"覆盖 reference 模态"的约束（更 mode-covering），也可能用 forward KL（但这通常更像行为克隆/交叉熵约束，而不是 RLHF 里常见的"奖励拉动 + KL 约束"）。

**直觉差异（工程上很重要）：**

- **reverse KL**（$\pi_\theta \| \pi_{\text{ref}}$）：更"mode-seeking"。因为 KL 的期望在 $\pi_\theta$ 下，如果 $\pi_\theta$ 不去某些区域，就几乎不为"没覆盖"付出代价。好处是能更激进地追高奖励；风险是更容易 reward hacking / 跑偏（所以要 reference / KL）。

- **forward KL**（$\pi_{\text{ref}} \| \pi_\theta$）：更"mode-covering"。因为期望在 $\pi_{\text{ref}}$ 下，如果 $\pi_\theta$ 把 reference 的高概率区域压得太低，会被强惩罚。好处是更保守、更像"别忘了预训练分布"；风险是可能抑制探索，尤其在需要大幅偏离参考才能提升任务 reward 时。

### A.3 KL-regularized RL 的闭式最优解（给定 reward 时）

固定一个 prompt $x$，把 $R(x,y)$ 简写成 $R(y)$，优化：

$$
\max_{\pi(\cdot)}\; \sum_y \pi(y)\,R(y) - \beta \sum_y \pi(y)\log\frac{\pi(y)}{\pi_{\text{ref}}(y)} \quad \text{s.t. } \sum_y \pi(y)=1
$$

写拉格朗日函数：

$$
\mathcal{L}(\pi, \lambda) = \sum_y \pi(y) R(y) - \beta \sum_y \pi(y)\log\frac{\pi(y)}{\pi_{\text{ref}}(y)} + \lambda\left(\sum_y \pi(y) - 1\right)
$$

对 $\pi(y)$ 求偏导并令 0：

$$
\frac{\partial\mathcal{L}}{\partial \pi(y)} = R(y) - \beta\left(\log\frac{\pi(y)}{\pi_{\text{ref}}(y)} + 1\right) + \lambda = 0
$$

移项：

$$
\log\frac{\pi(y)}{\pi_{\text{ref}}(y)} = \frac{R(y) + \lambda - \beta}{\beta}
$$

指数化并吸收入归一化常数：

$$
\pi^*(y) \propto \pi_{\text{ref}}(y)\exp\!\left(\frac{R(y)}{\beta}\right)
$$

即

$$
\pi^*(y \mid x) = \frac{\pi_{\text{ref}}(y \mid x)\exp\!\big(R(x,y)/\beta\big)}{Z(x)}, \quad Z(x) = \sum_y \pi_{\text{ref}}(y \mid x)\exp\!\big(R(x,y)/\beta\big)
$$

这个结论在 KL-regularized RL 和 DPO 的推导里都是核心[^1]。

### A.4 normalized KL vs "k3/UKL" 形式：不是换目标，而是换写法

你会看到很多工程实现里把 KL 写成一种"永远非负"的形式（常被叫 k3 / UKL 形式）。

令

$$
u = \frac{\pi_{\text{ref}}(a \mid h)}{\pi_\theta(a \mid h)}
$$

定义

$$
k_3(u) = u - \log u - 1 \quad (\ge 0 \text{ because } \log u \le u-1)
$$

那么在 $\pi_\theta(\cdot \mid h)$ 和 $\pi_{\text{ref}}(\cdot \mid h)$ 都是规范化分布时：

- $\mathbb{E}_{a \sim \pi_\theta}[u] = \sum_a \pi_\theta(a)\frac{\pi_{\text{ref}}(a)}{\pi_\theta(a)} = \sum_a \pi_{\text{ref}}(a) = 1$

因此：

$$
\mathbb{E}_{a \sim \pi_\theta}[k_3(u)]
= \mathbb{E}_{a \sim \pi_\theta}[u] - \mathbb{E}_{a \sim \pi_\theta}[\log u] - 1
= -\mathbb{E}_{a \sim \pi_\theta}[\log u]
= \mathbb{E}_{a \sim \pi_\theta}\left[\log\frac{\pi_\theta}{\pi_{\text{ref}}}\right]
= \mathrm{KL}(\pi_\theta \| \pi_{\text{ref}})
$$

所以 **k3 不是"另一个 KL 目标"**，它只是把同一个 reverse KL 写成"可单样本输出非负、方差更小"的表达（见下一节）。这也是为什么一些工作会说"k3 等价于 UKL"，并强调它与 GRPO 的 KL penalty 的关系[^2]。

---

## B. KL 的估计器：k1 / 控制变量 k3 / squared k2

这里讨论的是：在 LLM 里你通常只拿到 sampled token 的 logprob（而不是整 vocab 分布），所以很多 KL 只能用 MC 估计。

为了统一，先定义一个"比值"：

- 设想你要估计的是 $\mathrm{KL}(p \| q) = \mathbb{E}_{x \sim p}[\log\frac{p(x)}{q(x)}]$
- 在 RLHF 里常见映射是：$p = \pi_\theta(\cdot \mid h)$，$q = \pi_{\text{ref}}(\cdot \mid h)$ 或 $q = \pi_{\text{old}}(\cdot \mid h)$

定义

$$
r = \frac{p(x)}{q(x)}, \quad \log r = \log p(x) - \log q(x)
$$

### B.1 k1：vanilla（无偏，但方差大）

$$
k_1(x) = \log r = \log\frac{p(x)}{q(x)}
$$

无偏性直接来自 KL 的定义：

$$
\mathbb{E}_{x \sim p}[k_1(x)] = \mathbb{E}_{x \sim p}\left[\log\frac{p(x)}{q(x)}\right] = \mathrm{KL}(p \| q)
$$

问题：单样本 $\log r$ 可正可负、且可能很大（heavy-tail），所以**方差常很大**[^3]。

> 工程上：PPO/TRL 默认常用 k1；你会看到很多实现把它当作"KL 估计 + KL shaping"[^4]。

### B.2 k2：squared（有偏，但低方差）

$$
k_2(x) = \frac{1}{2}(\log r)^2
$$

它通常被视作一个"更平滑、非负"的 proxy，方差往往更小，但严格来说**不是无偏 KL 估计**[^3]。

> 关键点：k2 在"估计 KL 值"这件事上是 biased 的；但在"实现正确 KL 梯度"这件事上反而很有用（见 C.5）[^1]。

### B.3 k3：控制变量（无偏 + 非负 + 方差更小）

控制变量思路：找一个 $h(x)$，使得 $\mathbb{E}_{x \sim p}[h(x)] = 0$，并且与 $k_1$ 负相关。

一个自然的零均值候选是：

$$
h(x) = \frac{q(x)}{p(x)} - 1
$$

因为

$$
\mathbb{E}_{x \sim p}\left[\frac{q(x)}{p(x)} - 1\right] = \sum_x q(x) - 1 = 0
$$

于是对任意 $\lambda$，$k(x) = k_1(x) + \lambda\, h(x)$ 都保持无偏（期望还是 KL）。

特别地，当 $\lambda = 1$ 时：

$$
k_3(x) = \log\frac{p(x)}{q(x)} + \frac{q(x)}{p(x)} - 1
$$

等价写成上节的 $u = q/p$ 形式就是 $k_3(u) = u - \log u - 1 \ge 0$。

它的好处：

- 无偏（期望仍是 KL）
- 单样本非负（更符合"KL ≥ 0"的几何直觉）
- 实证上常显著降方差（控制变量抵消波动）[^3]

在一些 RL-for-LLM 实现中，GRPO 常被描述为"用 k3 而 PPO 用 k1"[^3]。

### B.4 工程里"k2 不能选"的细节（TRL 例子）

例如 TRL 的 PPOTrainer 里暴露的 KL estimator 只允许 `"k1"` 与 `"k3"`，并注明 `"k2"` 只用于 logging[^4]。

这反映一个现实工程权衡：k2 常更像一个"稳定的 proxy/诊断"，而不是大家普遍用于训练主目标的 KL 值（当然，梯度实现上另说，见下一节）。

---

## C. KL 梯度实现：stop-grad / squared / 错误写法

这一节是很多实现最容易踩坑的地方：**"我明明把 KL penalty 加进 loss 了，怎么训练完全不受约束？"** 或者 **"我用 k3 训练怎么变成 forward KL 的行为？"**

Tang & Munos 的《Approximating KL Divergence》把这个坑系统讲清楚了：**单样本 KL 的"值"估计没问题，但你要的是"正确的 KL 梯度"，就必须用合适的 surrogate（stop-grad / squared），否则会出现 0 梯度或跑偏到 forward KL**[^1]。

### C.1 你真正想要的：$\nabla_\theta \mathrm{KL}(\pi_\theta \| \pi_{\text{ref}})$

对一个固定状态 $h$（token-level state），定义：

$$
\mathrm{KL}\big(\pi_\theta(\cdot \mid h) \| \pi_{\text{ref}}(\cdot \mid h)\big)
= \sum_a \pi_\theta(a \mid h)\log\frac{\pi_\theta(a \mid h)}{\pi_{\text{ref}}(a \mid h)}
$$

对 $\theta$ 求导：

$$
\nabla_\theta \mathrm{KL}
= \sum_a \nabla_\theta \pi_\theta(a) \left(\log\frac{\pi_\theta(a)}{\pi_{\text{ref}}(a)} + 1\right)
$$

用 $\nabla\pi = \pi\,\nabla\log\pi$：

$$
\nabla_\theta \mathrm{KL}
= \sum_a \pi_\theta(a)\left(\log\frac{\pi_\theta(a)}{\pi_{\text{ref}}(a)} + 1\right)\nabla_\theta\log\pi_\theta(a)
$$

注意 $\sum_a \pi_\theta(a)\,\nabla\log\pi_\theta(a) = \nabla\sum_a \pi_\theta(a) = 0$，所以 $+1$ 项消失：

$$
\nabla_\theta \mathrm{KL}(\pi_\theta \| \pi_{\text{ref}})
= \mathbb{E}_{a \sim \pi_\theta}\left[\log\frac{\pi_\theta(a)}{\pi_{\text{ref}}(a)}\; \nabla_\theta\log\pi_\theta(a)\right]
$$

Tang & Munos 也给出了等价的解析形式与讨论[^1]。

### C.2 错误写法 1：直接对"k1 值"做 autograd（on-policy 时梯度为 0）

很多人会写：

```python
# a ~ pi_theta is sampled (non-differentiable sampling)
logp = pi_theta.log_prob(a)          # depends on theta
logq = pi_ref.log_prob(a)            # constant wrt theta
kl_sample = logp - logq              # k1
loss = beta * kl_sample.mean()
loss.backward()
```

看起来像在最小化 KL，但实际上：

- autograd 看到的是：$\nabla_\theta(\log \pi_\theta(a)) = \nabla_\theta\log\pi_\theta(a)$
- 所以它做的是 $\mathbb{E}_{a \sim \pi_\theta}[\nabla \log \pi_\theta(a)]$
- 而 $\mathbb{E}[\nabla \log \pi] = 0$

因此这条 KL penalty **在 on-policy（样本来自同一个 $\pi_\theta$）下的期望梯度为 0**[^1]。

> 这就是你会遇到的经典"KL 加了但没用"的坑。

### C.3 错误写法 2：对 k3（控制变量）直接 autograd，会变成 forward KL 梯度

k3（或 Tang & Munos 记作 variance-reduced estimator）：

$$
k_{\text{vr}}(a) = \log\frac{\pi_\theta(a)}{\pi_{\text{ref}}(a)} + \frac{\pi_{\text{ref}}(a)}{\pi_\theta(a)} - 1
$$

如果你直接对它 autograd（同样是 on-policy sample），Tang & Munos 证明其期望梯度对应的是 **forward KL** 的梯度（而不是你原本想要的 reverse KL）[^1]。

工程含义：

- k3 作为"值估计"是无偏 KL；
- 但你若把它当作可微 loss 直接 backprop，会改变"你在优化的 KL 方向"。

### C.4 正确写法 1：stop-gradient（REINFORCE-style surrogate）

Tang & Munos 给出的一个梯度正确 surrogate 是[^1]：

$$
\mathcal{L}_{\text{sg}}(\theta) = \mathrm{sg}\!\left[\log\frac{\pi_\theta(a)}{\pi_{\text{ref}}(a)}\right] \cdot \log\pi_\theta(a)
$$

其中 $\mathrm{sg}[\cdot]$ 表示 stop-gradient / detach。

因为：

- $\mathrm{sg}[\log\frac{\pi}{\pi_{\text{ref}}}]$ 在反传中当常数
- 所以梯度是 $\log\frac{\pi}{\pi_{\text{ref}}}\;\nabla\log\pi$
- 这正是 $\nabla \mathrm{KL}$ 的无偏估计（见 C.1）

对应代码：

```python
log_ratio = (logp - logq)           # computed value
loss_kl = beta * (log_ratio.detach() * logp).mean()
```

这本质上就是把 KL 惩罚当作 **reward shaping**：在 actor 的 policy gradient 里用 $-\beta\log\frac{\pi}{\pi_{\text{ref}}}$ 当"负奖励"[^1]。

### C.5 正确写法 2：squared surrogate（k2 用来"修梯度"）

另一个梯度正确 surrogate 是 squared[^1]：

$$
\mathcal{L}_{\text{sq}}(\theta) = \frac{1}{2}\left(\log\frac{\pi_\theta(a)}{\pi_{\text{ref}}(a)}\right)^2
$$

它的梯度是：

$$
\nabla \mathcal{L}_{\text{sq}} = \log\frac{\pi}{\pi_{\text{ref}}}\; \nabla\log\pi
$$

同样是 $\nabla\mathrm{KL}$ 的无偏估计（因为 $+1$ 项期望为 0）[^1]。

对应代码：

```python
log_ratio = logp - logq
loss_kl = 0.5 * beta * (log_ratio ** 2).mean()
```

注意：这里"squared"是**梯度正确**，但它估计的"KL 值"本身是 biased（上一节 B.2）。在工程里这反而经常是可接受甚至更稳定的 tradeoff。

### C.6 重要的工程分岔：你到底在做 on-policy 还是 off-policy？

上面 C.2/C.3 的"梯度为 0 / 变 forward KL"是典型的 **on-policy sample**（样本 $a \sim \pi_\theta$）场景。

但 PPO/GRPO 实际更新常是：

- rollout 用 $\pi_{\text{old}}$ 采样
- 更新时 $\pi_\theta$ 已经不同于 $\pi_{\text{old}}$

这时你若写：

$$
\mathbb{E}_{a \sim \pi_{\text{old}}}[\log\pi_\theta(a) - \log\pi_{\text{ref}}(a)]
$$

它的梯度是 $\mathbb{E}_{a \sim \pi_{\text{old}}}[\nabla \log\pi_\theta(a)]$，**不再是 0**。但这并不等价于 $\nabla \mathrm{KL}(\pi_\theta \| \pi_{\text{ref}})$（期望分布不对）。

"On the Design …"这类工作强调：在 off-policy sampling 下，你要严格对齐目标的梯度，需要显式地处理 importance weight；否则你的 KL penalty 其实是在优化一个"偏移后的 surrogate"[^2]。

---

## D. PPO vs GRPO vs DAPO vs GSPO：batch 组织与"单位一致性"

这一节用同一个张量视角写清楚：每个算法到底按什么维度组织 batch，以及对应的 advantage / importance weight 是按 token 还是 sequence 来定义。

### D.1 PPO 的 batch：通常是 prompt → (1 rollout) → token

设 batch 有 $B$ 个 prompt：

- prompts：$\{x_b\}_{b=1}^B$
- 每个 prompt rollout 1 个 response（也可以 >1，但经典 PPO 常 1）
- 得到 $y_b$（长度 $T_b$）

数据张量常组织成：

- `input_ids`: $[B, T_{\max}]$（带 mask）
- `logp_old`: $[B, T_{\max}]$（rollout 时记录）
- `logp_ref`: $[B, T_{\max}]$（参考模型）
- `values`: $[B, T_{\max}]$（critic）
- `rewards`: $[B, T_{\max}]$（含 KL shaping 或仅末端）

优势：

- token-level advantage $\hat{A}_{b,t}$（GAE）
- PPO ratio（token-level）：

$$
r_{b,t}(\theta) = \frac{\pi_\theta(y_{b,t} \mid h_{b,t})}{\pi_{\text{old}}(y_{b,t} \mid h_{b,t})} = \exp(\log\pi_\theta - \log\pi_{\text{old}})
$$

PPO-Clip 目标（token-level）：

$$
\min\!\big(r_{b,t}\,\hat{A}_{b,t},\; \mathrm{clip}(r_{b,t}, 1\!-\!\epsilon, 1\!+\!\epsilon)\,\hat{A}_{b,t}\big)
$$

### D.2 GRPO 的 batch：prompt → group(G rollouts) → token

GRPO 的核心变化：

- 每个 prompt $x_b$ 生成 $G$ 个 responses：$\{y_{b,i}\}_{i=1}^G$
- 只用 reward（序列级）做组内 baseline，不训练 critic

因此数据形状常是：

- `input_ids`: $[B, G, T_{\max}]$
- `logp_old`: $[B, G, T_{\max}]$
- `logp_ref`: $[B, G, T_{\max}]$
- `reward_seq`: $[B, G]$

#### (1) Group advantage（标量共享）

常见的 group-relative advantage（DAPO 文档写得很明确）[^5]：

$$
\hat{A}_{b,i} = \frac{R_{b,i} - \mathrm{mean}(\{R_{b,j}\}_{j=1}^G)}{\mathrm{std}(\{R_{b,j}\}_{j=1}^G)}
$$

然后把这个标量 broadcast 到 token：

$$
\hat{A}_{b,i,t} = \hat{A}_{b,i} \cdot \mathbf{1}[t \le T_{b,i}]
$$

#### (2) token-level ratio（像 PPO 一样）

GRPO 默认 token-level importance ratio[^6]：

$$
w^{\mathrm{GRPO}}_{b,i,t} = \frac{\pi_\theta(y_{b,i,t} \mid h_{b,i,t})}{\pi_{\text{old}}(y_{b,i,t} \mid h_{b,i,t})}
$$

并用 clip 形成类似 PPO 的 surrogate（但 advantage 来自 group）。此外 GRPO 常加入 KL penalty（对 reference）并且常用 k3 形式（下一节会讲它在 off-policy 下的严格性问题）[^2]。

### D.3 DAPO：在 GRPO 框架上改"clip / sampling / loss reduction / length handling"

DAPO（Decoupled Clip + Dynamic Sampling + …）主要是**训练系统层面的稳定性 trick**[^5]：

1. **Clip Higher（非对称 clip）**：上界更大以鼓励探索（文档示例：上 0.28，下 0.2）[^5]。
2. **Dynamic Sampling**：如果一个 prompt 的 $G$ 个 rollouts reward 方差为 0，则组内 advantage 全为 0，梯度消失；DAPO 直接跳过并重采样直到填满 batch[^5]。
3. **Token-level loss（关键：改变"归一化维度"来修 length bias）**：DAPO 指出 GRPO 的 sample-level loss 会让长序列 token 贡献被稀释，导致无法有效惩罚长样本中的低质模式，从而熵和长度不健康上升；于是引入 token-level policy gradient loss[^7]。
4. **Overlong filtering / soft punishment**：处理截断样本的 reward 噪声与长度控制[^5]。

> 注意：DAPO 论文里还有一条重要设定：在某些可验证推理任务里他们甚至会**不使用 KL term**（认为模型不易偏离初始太远，KL 约束非必要），但这属于特定任务/设定的选择，不能简单类推到通用 RLHF[^7]。

### D.4 GSPO：把 off-policy correction 的"单位"从 token 改成 sequence（与 reward 一致）

GSPO 的出发点很清晰：GRPO 在 token-level 做 importance ratio，但每个 token 只有一次采样，token-level ratio 会引入高方差噪声，甚至导致训练崩溃；而 reward 多是 sequence-level，所以 correction 也应该 sequence-level[^6]。

于是定义 sequence-level weight（几何平均形式，避免随长度指数爆炸）[^6]：

$$
w_i^{\mathrm{GSPO}} = \left[\frac{\pi_\theta(y_i \mid x)}{\pi_{\text{old}}(y_i \mid x)}\right]^{1/|y_i|} = \exp\!\left(\frac{1}{|y_i|}\sum_{t=1}^{|y_i|} \log\frac{\pi_\theta(y_{i,t} \mid h_{i,t})}{\pi_{\text{old}}(y_{i,t} \mid h_{i,t})}\right)
$$

并且给出 GSPO-token 的 stop-grad 组合式（本质是"序列权重 detach + token 内部让 logp 仍可回传"）[^6]：

$$
w_{i,t}^{\text{GSPO-token}} = \mathrm{sg}[w_i^{\mathrm{GSPO}}] \cdot \frac{\pi_\theta(y_{i,t} \mid h_{i,t})}{\mathrm{sg}[\pi_\theta(y_{i,t} \mid h_{i,t})]}
$$

swift 文档还直接给出了伪代码实现（非常工程化）[^6]。

---

## E. advantage 标量共享 + 归一化维度：长度偏置为什么出现？

这是最"工程痛"的问题：同样是 GRPO / DAPO / GSPO，你只要改一下 loss reduction，模型的长度行为会完全不一样。

问题拆成两层：

1. 数学上，sequence-level reward 被 broadcast 到 token，会天然带来长度相关的梯度结构
2. 工程上，你怎么"平均 loss"，会改变每条样本/每个 token 的权重，从而引入/修复长度偏置

### E.1 先看"标量 advantage 共享"的基本梯度结构

在 GRPO 类方法中，advantage 往往是序列标量 $\hat{A}_i$（来自组内相对 reward），并广播给每个 token。

忽略 clipping、ratio ≈ 1 时，policy gradient 近似是：

$$
\nabla_\theta J \approx \mathbb{E}\left[\hat{A}_i \sum_{t=1}^{T_i} \nabla_\theta \log \pi_\theta(y_{i,t} \mid h_{i,t})\right]
$$

注意这里天然是 **sum over tokens**。

这意味着：如果两条序列的 $\hat{A}$ 一样，但一条更长，那么它的梯度项里 token 的 logprob 梯度会加更多次，**这条样本对参数更新的"总推力"更大**。

这本身不一定是"bug"：它来自序列概率分解 $\log \pi(y) = \sum_t \log \pi(y_t \mid h_t)$ 的结构。

真正决定"是否出现你不想要的长度偏置"的，是你在实现里怎么做 loss reduction。

### E.2 sample-level vs token-level normalization：两种"看起来都合理"的平均方式

swift 的 Loss Types 文档把 GRPO / BNPO / DAPO 的 reduction 写成了明确公式，这对解释 length bias 非常关键[^8]。

设每个 token 的基本损失是 $\mathcal{L}_{i,t}$（比如 PPO-Clip 的 token loss，包含 ratio 和 advantage）。

#### (1) GRPO（sample-level / sentence-level 归一化）

$$
\mathcal{L}_{\text{GRPO}} = \frac{1}{N}\sum_{i=1}^N \frac{1}{T_i}\sum_{t=1}^{T_i} \mathcal{L}_{i,t}
$$

含义：**先对每个样本内部按 token 平均，再对样本平均**[^8]。所以每条样本权重相同；长样本的每个 token 权重更小（$1/T_i$）。

这就是 DAPO 论文批评的点：长序列里的 token 贡献被稀释，导致长样本里出现的低质量模式（重复、乱码）惩罚不够，从而熵和长度不健康上涨[^7]。

#### (2) BNPO / DAPO（token-level 归一化）

BNPO[^8]：

$$
\mathcal{L}_{\text{BNPO}} = \frac{\sum_i \sum_t \mathcal{L}_{i,t}}{\sum_i T_i}
$$

DAPO（跨进程 token-level）[^8]：

$$
\mathcal{L}_{\text{DAPO}} = \frac{\sum_i \sum_t \mathcal{L}_{i,t}}{\sum_{\text{all proc}} \sum_i T_i}
$$

含义：**所有 completion tokens 等权平均**。所以长样本因为 token 多，会对总 loss 贡献更大。

DAPO 论文把这种改变称为 token-level policy gradient loss，并指出它会让长序列对梯度更新更有影响，并且对某种模式的鼓励/抑制不再依赖它出现在长样本还是短样本中[^7]。

### E.3 用一个"权重视角"统一解释长度偏置

把 reduction 写成更一般的形式：

$$
\mathcal{L}(\alpha) = \frac{1}{N}\sum_{i=1}^N \frac{1}{T_i^\alpha}\sum_{t=1}^{T_i} \mathcal{L}_{i,t}
$$

- $\alpha = 1$：sample-level（GRPO）
- $\alpha = 0$：sequence sum（每条样本按 token 求和再平均）
- token-global mean（BNPO/DAPO）不是简单 $\alpha$ 形式，但它等价于"每个 token 权重一样"，即样本权重 $\propto T_i$

若我们只关心"每条样本在总 loss 中的权重"，可以看：

- GRPO：样本权重 $\propto 1$（每条样本同权）
- BNPO/DAPO：样本权重 $\propto T_i$（长样本更重）

于是：

- **GRPO 的偏置**：长样本里的 token 被稀释 → 不容易惩罚长样本中的坏模式 → 容易让模型通过"变长/更随机"逃避惩罚（DAPO 论文的观测）[^7]。
- **DAPO/BNPO 的修正**：让 token 等权，长样本自然更重 → 长样本中的坏 token 会被更强惩罚 → 长度增长更"健康"[^7]。

### E.4 还有一个更隐蔽的长度偏置来源：KL penalty 是按 token 加还是按 sequence 加？

回到我们最初的 KL-regularized RL 目标：

$$
-\beta\,\mathrm{KL}\big(\pi_\theta(y \mid x) \| \pi_{\text{ref}}(y \mid x)\big)
= -\beta\,\mathbb{E}_{y \sim \pi_\theta}\left[\sum_{t=1}^T \log\frac{\pi_\theta(y_t \mid h_t)}{\pi_{\text{ref}}(y_t \mid h_t)}\right]
$$

它本质是 **sequence-level KL = token log-ratio 的和**。

因此如果你做 token-level shaping，理论一致的方式是：对每个 token 都加 $-\beta \log\frac{\pi_\theta}{\pi_{\text{ref}}}$，然后在策略梯度里做 sum[^1]。

但工程里如果你又做了 sample-level 平均（除以 $T_i$），那你实际上在优化更像"每 token 平均 KL"（而不是 sequence KL）。这会改变长度的 tradeoff：

- **sequence KL（sum）**：越长 KL 越大（如果平均 log-ratio 为正），更强地惩罚变长。
- **token-average KL**：惩罚不随长度线性增长，更容易出现"靠变长稀释某些 token 的 KL/奖励"的行为。

这也是为什么你会看到一些系统里需要额外做 length penalty / overlong shaping（DAPO 就专门做了）[^5]。

---

## F. 把这一切串成"从目标到算法"的链

最后用一条链把"目标 → KL 方向 → KL 估计 → 梯度实现 → batch/advantage 组织"统一起来。

### F.1 从目标出发（reverse KL 正则化）

$$
\max_\pi\; \mathbb{E}_{y \sim \pi}[R] - \beta\,\mathrm{KL}(\pi \| \pi_{\text{ref}})
\quad\Rightarrow\quad
\pi^* \propto \pi_{\text{ref}}\exp(R/\beta)
$$

### F.2 两条实现路线

#### 路线 1：在线 RL（PPO / GRPO / DAPO / GSPO）

- 你要估计 $\nabla_\theta$ 的东西是：$\nabla_\theta \mathbb{E}_{y \sim \pi_\theta}[R]$ 以及 $\nabla_\theta \mathrm{KL}(\pi_\theta \| \pi_{\text{ref}})$。
- KL 梯度正确实现：stop-grad 或 squared（避免 0 梯度/forward KL）[^1]。
- critic 的角色：估计 baseline $V$ 以得到 token-level advantage（PPO）。GRPO/DAPO 则用 **group baseline** 替代 critic：$\hat{A}_{b,i} = (R - \mu)/\sigma$[^5]。
- batch 组织：prompt→(group)→token；以及 loss reduction 决定长度偏置（GRPO vs DAPO）[^8]。
- off-policy correction 的"单位"：GRPO 用 token-level ratio $w_{i,t}$（高方差风险）[^6]；GSPO 用 sequence-level ratio $w_i^{\mathrm{GSPO}}$（与序列 reward 单位一致）[^6]。

#### 路线 2：离线偏好学习（DPO）

- 利用闭式最优解，把"需要 reward model 的 RL"变成"直接拟合偏好对"的监督式目标：DPO 将 LM 视作隐式 reward 模型，构造

$$
\log\sigma\!\left(\beta\left(\log\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right)
$$

的目标来让 chosen 比 rejected 概率更大，并对应 BT 偏好模型假设[^9]。

---

## 工程检查清单（把坑堵死）

1. **你想约束的是哪种 KL？** 绝大多数 RLHF：reverse KL $\mathrm{KL}(\pi_\theta \| \pi_{\text{ref}})$。

2. **你是在 on-policy 还是 off-policy 下实现 KL 梯度？** on-policy 下直接 autograd(k1) 会 0 梯度；autograd(k3) 会偏成 forward KL[^1]。正解：stop-grad 或 squared[^1]。

3. **你用的 KL 估计器是 k1 还是 k3？** k3（控制变量）更稳，但要注意"别直接对它做可微回传导致 KL 方向变掉"[^1]。

4. **你的 reward 是 sequence-level 还是 token-level？** sequence-level reward：GSPO 观点是 correction 也应 sequence-level；GRPO token-level ratio 易高方差[^6]。

5. **你的 loss reduction 是 sample-level 还是 token-level？** sample-level（GRPO）：长样本 token 被稀释，可能导致长度/熵异常上涨[^7]。token-level（DAPO/BNPO）：token 等权，长样本坏模式更容易被惩罚[^7]。

---

## 参考文献

[^1]: Tang, Y. & Munos, R. (2025). "Approximating KL Divergence for RL-based LLM Training." arXiv:2506.09477. <https://arxiv.org/pdf/2506.09477>

[^2]: "On the Design of KL-regularized Policy Optimization." arXiv:2505.17508. <https://arxiv.org/pdf/2505.17508>

[^3]: NormalUhr. "KL Divergence Estimator in RL for LLMs." Hugging Face Blog. <https://huggingface.co/blog/NormalUhr/kl-divergence-estimator-rl-llm>

[^4]: Hugging Face TRL — PPO Trainer. <https://huggingface.co/docs/trl/en/ppo_trainer>

[^5]: Swift Documentation — DAPO. <https://swift.readthedocs.io/en/latest/Instruction/GRPO/AdvancedResearch/DAPO.html>

[^6]: Swift Documentation — GSPO. <https://swift.readthedocs.io/en/latest/Instruction/GRPO/AdvancedResearch/GSPO.html>

[^7]: DAPO. arXiv:2503.14476. <https://arxiv.org/pdf/2503.14476>

[^8]: Swift Documentation — Loss Types. <https://swift.readthedocs.io/en/latest/Instruction/GRPO/DeveloperGuide/loss_types.html>

[^9]: Rafailov, R., et al. "Direct Preference Optimization." arXiv:2305.18290. <https://arxiv.org/html/2305.18290v3>