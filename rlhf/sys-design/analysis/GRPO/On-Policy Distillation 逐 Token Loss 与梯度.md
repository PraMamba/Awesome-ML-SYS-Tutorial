# On-Policy Distillation 逐 Token Loss 与梯度：Forward KL / Reverse KL / JSD

> 把 on-policy distillation 在 forward KL / reverse KL / (generalized) JSD 三种 divergence 下的**逐 token loss 与梯度**写到"可直接实现"的程度，并把三个关键工程点一并讲清楚：**温度**、**stop-gradient**、**teacher logits 缓存策略**。

---

## 0) 统一记号：每个 token 位置我们到底在算什么？

对一个样本（prompt）$x$，学生模型按自回归生成序列 $y=(y_1,\dots,y_L)$。在第 $t$ 个位置的"状态"是前缀

$$
s_t=(x, y_{<t})
$$

在这个状态上：

- teacher logits：$\mathbf{z}^T_t \in \mathbb{R}^{|V|}$
- student logits：$\mathbf{z}^S_t \in \mathbb{R}^{|V|}$

### 0.1 两种温度：别混了

**1. 采样温度 $\gamma$**（决定 on-policy rollouts 的随机性）

GKD 论文把生成分布写成带温度 $\gamma$ 的 softmax，并说明训练时学生温度通常取 1，用于鼓励多样性。[^1]

**2. 蒸馏温度 $\tau$**（决定对齐分布时"软化"程度）

这是 KD 里经典的"soft targets"温度。Hinton 等指出：用更高温度会让分布更软，并给出梯度随温度缩放的结论。[^3]

实务中，你可能同时用：rollout 用 $\gamma$（探索），distill loss 用 $\tau$（软标签）。两者是不同旋钮。

### 0.2 蒸馏分布定义（带温度 $\tau$）

令

$$
\mathbf{p}_t = \text{softmax}(\mathbf{z}^T_t/\tau),\qquad \mathbf{q}_t = \text{softmax}(\mathbf{z}^S_t/\tau)
$$

其中 $\mathbf{p}_t$ 是 teacher token 分布，$\mathbf{q}_t$ 是 student token 分布。

---

## 1) Forward KL（teacher → student）：逐 token loss 与梯度

### 1.1 逐 token loss

$$
\mathcal{L}^{\text{FKL}}_t = D_{\text{KL}}(\mathbf{p}_t \,\|\, \mathbf{q}_t) = \sum_{v\in V}\mathbf{p}_t(v)\Big(\log\mathbf{p}_t(v)-\log\mathbf{q}_t(v)\Big)
$$

注意 $\sum \mathbf{p}\log\mathbf{p}$ 对学生是常数，所以实现上常写成"soft targets 的交叉熵"：

$$
\mathcal{L}^{\text{FKL}}_t = -\sum_{v}\mathbf{p}_t(v)\log\mathbf{q}_t(v) \quad (+\text{const})
$$

GKD 中最标准的 on-policy KD 就是 forward KL（teacher $\|$ student）这种方向。[^1]

### 1.2 对 student logits 的梯度（最常用、最干净）

$$
\frac{\partial \mathcal{L}^{\text{FKL}}_t}{\partial \mathbf{z}^S_t} = \frac{1}{\tau}\Big(\mathbf{q}_t-\mathbf{p}_t\Big)
$$

这和 Hinton 2015 给出的 soft-target cross-entropy 梯度形式一致（他们写成 $\frac{1}{T}(q_i-p_i)$）。[^3]

### 1.3 温度缩放：为什么常见 $\tau^2$ 乘子？

Hinton 指出：soft targets 产生的梯度幅度随温度大致按 $1/\tau^2$ 缩放，因此当你把"hard label loss + soft KD loss"混合时，需要把 KD 部分乘 $\tau^2$ 来保持相对权重稳定。[^3]

如果你用

$$
\tilde{\mathcal{L}}^{\text{FKL}}_t=\tau^2\mathcal{L}^{\text{FKL}}_t
$$

则

$$
\frac{\partial \tilde{\mathcal{L}}^{\text{FKL}}_t}{\partial \mathbf{z}^S_t} = \tau\big(\mathbf{q}_t-\mathbf{p}_t\big)
$$

纯蒸馏（没有 hard loss）时，是否乘 $\tau^2$ 不再是"必须"，但它仍是一个很常见的梯度幅度旋钮。

---

## 2) Reverse KL（student → teacher）：逐 token loss 与梯度

### 2.1 "全分布版本"的逐 token loss（需要 teacher 全分布/至少 top‑k）

$$
\mathcal{L}^{\text{RKL}}_t = D_{\text{KL}}(\mathbf{q}_t \,\|\, \mathbf{p}_t) = \sum_{v}\mathbf{q}_t(v)\Big(\log\mathbf{q}_t(v)-\log\mathbf{p}_t(v)\Big)
$$

GKD 论文明确把 reverse KL 作为可选 divergence，并讨论其 mode-seeking 特性。[^1]

### 2.2 对 student logits 的梯度（全分布闭式）

令

$$
K_t = D_{\text{KL}}(\mathbf{q}_t \,\|\, \mathbf{p}_t) = \sum_v \mathbf{q}_t(v)\log\frac{\mathbf{q}_t(v)}{\mathbf{p}_t(v)}
$$

则对每个词表 token $v$：

$$
\frac{\partial \mathcal{L}^{\text{RKL}}_t}{\partial z^S_t(v)} = \frac{1}{\tau}\,\mathbf{q}_t(v)\Big(\log\frac{\mathbf{q}_t(v)}{\mathbf{p}_t(v)}-K_t\Big)
$$

向量写法：

$$
\frac{\partial \mathcal{L}^{\text{RKL}}_t}{\partial \mathbf{z}^S_t} = \frac{1}{\tau}\;\mathbf{q}_t \odot \Big(\log \mathbf{q}_t-\log \mathbf{p}_t - K_t\mathbf{1}\Big)
$$

**直观解释**（很重要）：

- 它是"乘了 $\mathbf{q}$"的梯度（不像 forward KL 的 $q-p$ 那么线性）。
- 括号里有一个 baseline $= K_t$，这让梯度满足 softmax 的"和为 0"约束（也对应 RL 里常见的 baseline 减方差结构）。

### 2.3 只用"采样到的 token 的 teacher logprob"也能做 reverse KL（RL-style estimator）

Thinking Machines 的 on-policy distillation 实现选的就是逐 token 的 reverse KL，并强调只需要 teacher 对"学生采样到的 token"的 logprob（`compute_logprobs`），不需要 full logits。[^2]

它用的是这个逐 token 量（对学生采样到的 $y_t$）：

$$
\hat{\ell}^{\text{RKL}}_t = \log \mathbf{q}_t(y_t)-\log \mathbf{p}_t(y_t)
$$

并把它当作"per-token penalty / advantage"去做 RL 风格更新。[^2]

为什么这样合理？因为：

$$
D_{\text{KL}}(\mathbf{q}_t\|\mathbf{p}_t) = \mathbb{E}_{a\sim \mathbf{q}_t}\Big[\log \mathbf{q}_t(a)-\log \mathbf{p}_t(a)\Big]
$$

所以 $\hat{\ell}^{\text{RKL}}_t$ 是一个 Monte Carlo 无偏估计器（对 loss 值）。

对应的 score-function / policy-gradient 形式梯度是：

$$
\nabla_\theta \mathbb{E}_{a\sim \mathbf{q}_t}\big[\hat{\ell}^{\text{RKL}}_t(a)\big] = \mathbb{E}_{a\sim \mathbf{q}_t} \Big[ \big(\log \mathbf{q}_t(a)-\log \mathbf{p}_t(a)-b_t\big)\;\nabla_\theta\log \mathbf{q}_t(a) \Big]
$$

其中 $b_t$ 可以取 batch mean、moving average，甚至取 $K_t$（但 $K_t$ 需要 full 分布才算得精确）。

**关键工程意义**：

- teacher 侧只要返回 sampled token 的 logprob，存储与带宽成本极低；
- 但梯度会更"RL 味儿"（估计方差更大），需要 baseline / clip / advantage normalization 等常见技巧。

---

## 3) JSD（以及 GKD 的 generalized JSD(β)）：逐 token loss 与梯度

GKD 明确给出了 generalized JSD(β) 的定义，并指出它在 $\beta\to 0$ 与 $\beta\to 1$ 时分别"接近 forward/reverse KL 的梯度行为"。[^1]

### 3.1 generalized JSD(β) 的逐 token loss

令混合分布

$$
\mathbf{m}_t = \beta \mathbf{p}_t + (1-\beta)\mathbf{q}_t,\qquad 0<\beta<1
$$

则

$$
\mathcal{L}^{\text{JSD}\beta}_t = \beta\,D_{\text{KL}}(\mathbf{p}_t\|\mathbf{m}_t) + (1-\beta)\,D_{\text{KL}}(\mathbf{q}_t\|\mathbf{m}_t)
$$

这与论文中的 $D_{\text{JSD}(\beta)}(P\|Q)$ 一一对应（把 $P=\mathbf{p}_t, Q=\mathbf{q}_t$ 代入即可）。[^1]

标准 JSD 就是 $\beta=0.5$。

### 3.2 对 student logits 的梯度（闭式、非常好用）

定义

$$
K^{(m)}_t = D_{\text{KL}}(\mathbf{q}_t\|\mathbf{m}_t) = \sum_v \mathbf{q}_t(v)\log\frac{\mathbf{q}_t(v)}{\mathbf{m}_t(v)}
$$

则：

$$
\frac{\partial \mathcal{L}^{\text{JSD}\beta}_t}{\partial z^S_t(v)} = \frac{1-\beta}{\tau}\,\mathbf{q}_t(v)\Big(\log\frac{\mathbf{q}_t(v)}{\mathbf{m}_t(v)} - K^{(m)}_t\Big)
$$

- 当 $\beta=0.5$：前面的系数就是 $1/(2\tau)$。
- 需要注意：因为 $\mathbf{m}_t$ 含 $\mathbf{q}_t$，所以实现时不能把 $\mathbf{m}_t$ detach（detach 就不是这个 objective 了）。

### 3.3 计算/缓存需求

JSD / JSD(β) 与 forward KL 类似：都需要 teacher 在同一前缀状态下给出分布信息（full logits 或 top‑k），否则 $\mathbf{m}_t$ 不好构造。

---

## 4) stop-gradient：到底 stop 在哪里？

### 4.1 stop-gradient through sampling（GKD 的核心实现选择）

GKD 在定义 on-policy loss 时明确说：不对学生的采样分布 $p_S(\cdot|x)$ 反向传播，并说明这样训练更稳定、更高效。[^1]

工程落地通常就是：

- **rollout**：用学生（或学生的 lagged copy $\theta_{\text{old}}$）采样得到 token ids $y$（这一步在 `torch.no_grad()` 下做；采样结果天然是离散的，也没法 pathwise backprop）。
- **train**：把这些 $y$ 当作"固定数据"，再跑一次 student forward 得到 $\mathbf{z}^S_t$，计算 KL/JSD loss 并反传。

这对应"DAgger / on-policy imitation"的那种"收集数据 → 监督更新"的范式，而不是"对采样过程做可微分重参数化"。

### 4.2 stop-gradient through teacher

无论 forward/reverse/JSD：

- teacher 参数固定（`teacher.eval()` + `torch.no_grad()`）
- teacher logits / logprobs 都应 detach 后缓存，否则会把 teacher 的计算图一起存下来（显存爆炸）。

Thinking Machines 也明确说他们不需要对 teacher 传播梯度，只需要 logprob 查询。[^2]

### 4.3 reverse KL 的两条路线与 stop-grad 的关系

**全分布 reverse KL**（$\text{KL}(q\|p)$ 直接算）：不需要对"采样分布"求导；因为每个位置你拿的是 $\mathbf{q}_t$ 的 full softmax（可微），梯度用上面闭式即可。

**sampled-token reverse KL**（只拿 teacher 对采样 token 的 logprob）：你是在用 score-function（RL 风格）来处理"期望对 $\mathbf{q}$ 取"的求导；这时虽然"采样本身不可微"，但 $\nabla \log \mathbf{q}_t(y_t)$ 就是正确的 policy-gradient 入口。Thinking Machines 就是把它塞进 RL 的 advantage/IS 框架去更新。[^2]

---

## 5) teacher logits 缓存策略：按 divergence 的"信息需求"来设计

### 5.1 先问：你到底需要 teacher 给什么？

- **Forward KL / JSD / 全分布 reverse KL** 需要 teacher 在每个前缀状态 $s_t$ 的**分布信息**（理想是 full logits；工程上往往用 top‑k + 近似）。
- **sampled-token reverse KL（RL-style）** 只需要 teacher 对学生采样 token 的 logprob：$\log \mathbf{p}_t(y_t)$。[^2]

这决定了缓存"长什么样"。

### 5.2 三种常见缓存形态（从省到贵）

**A) 只缓存 sampled token 的 teacher logprob（最省，适配 RL-style reverse KL）**

缓存内容（每个 token 一条标量）：`teacher_logp[t] = log p_t(y_t)`

Thinking Machines 的实现就是：对采样到的 trajectories 调 `compute_logprobs`，返回 teacher 在这些 token 上的 logprobs，再构造 per-token reverse KL。[^2]

优点：存储量 $\approx B\times L$（bf16 也就几百 KB 级），teacher 侧带宽小、CPU/GPU 内存压力小。

缺点：只直接支持"sampled-token 估计"的 reverse KL 路线（想做 forward KL/JSD 不够信息）。

**B) 缓存 teacher 的 top‑k logprobs + indices（最主流折中，适配 KL/JSD）**

veRL 的 async on-policy distill recipe 明确采用了这种方式：teacher service 返回 top‑k log-probabilities + token indices，然后 actor 侧做 KL 计算（还可以用自定义 kernel）。[^4]

缓存内容（每个 token 一组稀疏分布）：`teacher_topk_indices[t, k]` + `teacher_topk_logps[t, k]`（注意：是 log-probabilities，而不是 raw logits）[^4]

优点：能做 forward KL / JSD / 全分布 reverse KL 的近似版本；存储从 $B\times L\times |V|$ 下降到 $B\times L\times k$。

缺点：top‑k 近似会引入 bias（但通常可接受）；$\tau$ 一旦改变，你必须重新请求（因为返回的是 logprob 而不是 raw logits）。

> 小技巧：如果你担心"学生采样 token 不在 teacher top‑k 里"，可以把 sampled token 强行加入稀疏集合（k+1），避免出现"teacher 概率缺失"。

**C) 缓存 teacher full logits（最贵，但最精确）**

缓存内容：`teacher_logits[t, |V|]`

一般只在 vocab 小（例如分类）、序列极短、teacher 本身不大、或只在 CPU/disk 做离线蒸馏时可行。

原因很现实：full logits 的存储是 $B\times L\times |V|$。用 bf16（2 bytes）举例，$B=64,L=1024,|V|=50k$ 时约 6.5GB/批，很快就不可接受。

### 5.3 用 top‑k 做 KL/JSD 的一个"可控近似"（推荐至少知道）

如果你有 teacher top‑k 概率（或 logprob）集合 $K_t$，可以把 teacher 的 tail 合并成一个"other bucket"：

- teacher：$p_{\text{tail}} = 1 - \sum_{v\in K_t} p(v)$
- student：$q_{\text{tail}} = 1 - \sum_{v\in K_t} q(v)$

然后用 $K_t \cup \{\text{tail}\}$ 上的分布算近似 forward KL：

$$
D_{\text{KL}}(p\|q)\approx \sum_{v\in K_t} p(v)\log\frac{p(v)}{q(v)} + p_{\text{tail}}\log\frac{p_{\text{tail}}}{q_{\text{tail}}}
$$

这样你不会因为"忽略 tail"而把 teacher 质量很差的部分直接当 0（比"只在 top‑k 上 renorm"更稳）。

同理可以构造 JSD(β)（把 tail 也放进 $m$ 的定义里）。

---

## 6) 伪代码骨架（PyTorch 思路）

设：

- `student_logits`: `[B, L, V]`
- `mask`: `[B, L]`（padding=0）
- `tau`: distill 温度
- teacher 侧可能是：`teacher_logp_sampled`: `[B, L]`（路线 A）；或 `teacher_topk_logps, teacher_topk_idx`: `[B, L, K]`（路线 B）；或 `teacher_logits`: `[B, L, V]`（路线 C）

### 6.1 forward KL（full logits 版）

```python
with torch.no_grad():
    p = softmax(teacher_logits / tau, dim=-1)     # [B,L,V]

logq = log_softmax(student_logits / tau, dim=-1)  # [B,L,V]
loss_t = -(p * logq).sum(-1)                      # [B,L]
loss = (loss_t * mask).sum() / mask.sum()
# 可选：loss *= tau**2
loss.backward()
```

梯度对 logits 会自动是 $(q-p)/\tau$，与上面的闭式一致。

### 6.2 reverse KL（sampled-token logprob + RL-style）

你需要学生在 rollout 时就拿到 `logq_sampled = log q(y_t)`（RL 框架一般本来就算）。Thinking Machines 就是把 teacher 的 `logp_sampled` 拉回来，然后把 per-token advantage 设成 `-(logq - logp)` 去更新。[^2]

---

## 参考文献

[^1]: Agarwal, R. et al. (2024). *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes (GKD)*. ICLR 2024. 提出 Generalized Knowledge Distillation（GKD）框架，将 supervised KD 与 on-policy KD 统一为混合目标（参数 $\lambda$ 控制），系统对比 forward KL / reverse KL / generalized JSD(β) 等 divergence。

[^2]: Thinking Machines Lab (2025). *On-Policy Distillation*. 详细阐述 per-token reverse KL 作为密集信号在 on-policy distillation 中的应用，强调只需 teacher 的 sampled-token logprob（`compute_logprobs`），不需要 full logits。URL: <https://thinkingmachines.ai/blog/on-policy-distillation/>

[^3]: Hinton, G., Vinyals, O. & Dean, J. (2015). *Distilling the Knowledge in a Neural Network*. 提出 soft targets 与蒸馏温度 $\tau$，给出梯度按 $1/\tau^2$ 缩放的结论及 $\tau^2$ 补偿因子。

[^4]: veRL 框架文档. *On-Policy Distillation Recipe*. 采用 teacher service 返回 top‑k log-probabilities + token indices 的缓存方案，actor 侧做 KL 计算。