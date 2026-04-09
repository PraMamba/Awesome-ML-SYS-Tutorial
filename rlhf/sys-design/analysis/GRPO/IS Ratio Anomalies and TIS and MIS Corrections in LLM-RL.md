# 两类典型 IS Ratio 异常（训练–推理不一致）与 TIS / MIS 修正方法

---

## 1) 统一符号：这里的 IS ratio 到底是什么？

在"训练–推理不一致（training–inference mismatch）"[^1] 的工程现实里，常见结构是：

- **推理/采样端（sampler / inference engine）** 用分布

$$
\mu(\cdot)\equiv \pi_{\text{infer}}(\cdot;\theta_{\text{old}})
$$

生成 rollouts（例如 vLLM 等推理引擎与训练引擎分离的混合架构）[^1]。

- **训练端（learner / training backend）** 用分布

$$
\pi(\cdot)\equiv \pi_{\text{train}}(\cdot;\theta_{\text{old}})
$$

计算 logprob / 梯度（例如 FSDP 等训练后端）[^1]。

尽管参数都是同一个 $\theta_{\text{old}}$，但由于 **实现差异 / 精度差异 / 量化 / 内核非确定性** 等，可能出现：

$$
\pi_{\text{infer}}(\cdot;\theta_{\text{old}})\neq \pi_{\text{train}}(\cdot;\theta_{\text{old}})
$$

从而"名义 on-policy，实际变 off-policy"[^1]。

由此 IS ratio（密度比）有两种常用粒度：

### 1.1 Token 级 ratio（词元级）

$$
\rho_t \;=\;\frac{\pi_{\text{train}}(y_t\mid x,y_{<t};\theta_{\text{old}})}{\pi_{\text{infer}}(y_t\mid x,y_{<t};\theta_{\text{old}})}
$$

（Intellect-3 报告中明确以 $\pi_{\text{train}}/\pi_{\text{infer}}$ 的 token 级形式作为 mismatch 修正对象）[^2]

### 1.2 序列级 ratio（全轨迹 / 全序列）

$$
\rho(y)\;=\;\frac{\pi_{\text{train}}(y\mid x;\theta_{\text{old}})}{\pi_{\text{infer}}(y\mid x;\theta_{\text{old}})}
\;=\;\prod_{t=1}^{T}\rho_t
$$

（序列级 IS 的"单一 ratio 乘在整条轨迹上"的理论表达在 TRL 文档中被称为"正确、无偏"的估计器形式）[^3]

> **方向说明（避免歧义）：** 不同实现/文档里 ratio 也可能写成倒数（$\pi_{\text{infer}}/\pi_{\text{train}}$）。但核心都是"把在 $\mu$ 下采到的样本，换算成在 $\pi$ 下的期望/梯度"。本文统一采用 $\pi_{\text{train}}/\pi_{\text{infer}}$ 方向，与多处训练-推理 mismatch 修正文献/实现保持一致（例如 TIS 论文[^1]、IcePop 公式[^2]）。

---

## 2) **TIS** 是什么

### 2.1 全称与核心定义：TIS = Truncated Importance Sampling（截断重要性采样）

**TIS（Truncated Importance Sampling）**[^1] 做的事非常明确：

> 用 IS 修正 sampler 与 learner 分布不一致带来的 off-policy，但为了稳定性，把 IS 权重（ratio）做**上界截断**（truncate / cap）。[^1]

在 "On the Rollout-Training Mismatch in Modern RL Systems"[^1] 这篇专门讨论 rollout–training mismatch 的论文中，作者给出（以 REINFORCE 为例）从未修正到 IS 修正、再到 TIS 的关键公式：

**未修正（但在 mismatch 下会偏）**：

$$
\mathbb{E}_{a\sim \pi_{\text{vLLM}}(\theta)}
\big[R(a)\cdot \nabla_\theta \log \pi_{\text{FSDP}}(a,\theta)\big]
$$

**IS 修正（重要性比率 $\pi_{\text{FSDP}}/\pi_{\text{vLLM}}$）**：

$$
\mathbb{E}_{a\sim \pi_{\text{vLLM}}(\theta)}
\Big[\frac{\pi_{\text{FSDP}}(a,\theta)}{\pi_{\text{vLLM}}(a,\theta)}\cdot R(a)\cdot \nabla_\theta \log \pi_{\text{FSDP}}(a,\theta)\Big]
$$

**TIS（把比率截断为 $\min(\cdot,C)$）**：

$$
\mathbb{E}_{a\sim \pi_{\text{vLLM}}(\theta)}
\Big[\min\Big(\frac{\pi_{\text{FSDP}}(a,\theta)}{\pi_{\text{vLLM}}(a,\theta)}, C\Big)\cdot R(a)\cdot \nabla_\theta \log \pi_{\text{FSDP}}(a,\theta)\Big]
$$

其中 $C$ 是截断上界超参。[^1]

### 2.2 粒度精确定义：Token-TIS vs Seq-TIS

TIS 不是只能做一种粒度，它严格分两类：

**Token-level TIS（Token-TIS）**：对每个 token 的 $\rho_t$ 截断

$$
w_t=\min(\rho_t,C)
$$

（TRL 的 GRPO 配置项明确存在 `"token_truncate"` 这一模式，对应 token 级截断）[^3]

**Sequence-level TIS（Seq-TIS）**：先算全序列 ratio $\rho(y)=\prod_t\rho_t$，再截断

$$
w(y)=\min(\rho(y),C)
$$

（richardli 的 Part 3 明确给出 Seq-TIS 形式 $\min(\rho(y),C)\cdot f(y)$）[^4]

工程实现上也会明确区分 `"token_truncate"` 与 `"sequence_truncate"`。[^3]

---

## 3) **MIS** 是什么

> 重要提醒：**MIS** 在不同领域有同名缩写。
> - 在 LLM-RL / mismatch 修正语境里：**MIS = Masked Importance Sampling（掩码重要性采样）**[^3]
> - 在 Monte Carlo 渲染等语境里：MIS 常指 **Multiple Importance Sampling（多重重要性采样）**[^9]
>
> 本文**严格采用前者（Masked Importance Sampling）**，与 IcePop / GRPO 语境一致。

### 3.1 全称与核心定义：MIS = Masked Importance Sampling（掩码重要性采样）

在 TRL 文档中，**Masked Importance Sampling（MIS）** 被定义为[^3]：

> MIS 解决与 TIS 相同的训练–推理 mismatch 问题，但**用 masking 替代 clipping**；当 discrepancy 超过阈值 $C$ 时，直接丢弃更新（discard updates）。并且 TRL 说明其实现采用 **upper-side masking**（只对上侧越界做屏蔽）。

形式上，典型（上侧）mask 是：

$$
w(y)=\mathbb{I}(\rho(y)\le C)\cdot \rho(y)
$$

（Seq-MIS 的明确公式见 richardli Part 3：$\mathbb{I}(\rho(y)\le C)\rho(y)f(y)$）[^4]

### 3.2 粒度：Token-MIS vs Seq-MIS

同样地，MIS 也有粒度区分：

**Sequence-level MIS（Seq-MIS）**：

$$
\hat g_{\text{seq-mis}}(y)=\mathbb{I}(\rho(y)\le C)\cdot \rho(y)\cdot f(y)
$$

（richardli Part 3 明确标为 "Sequence-Level Masked IS (Seq-MIS)"）[^4]

**Token-level MIS（Token-MIS）**：对每个 token 的 ratio 做 mask（每个 token 是否参与梯度更新由 mask 决定）。TRL 文档明确存在 `"token_mask"` 与 `"sequence_mask"` 两种模式。[^3]

### 3.3 IcePop 里的 MIS：**双侧（double-sided）token mask** 的精准公式

在 INTELLECT-3 技术报告[^2]中，训练算法采用 **masked token-level importance sampling**，并给出 **IcePop 目标函数**与**掩码函数**：

**IcePop 目标**（报告中的式 (1)）：

$$
J_{\text{IcePop}}(\theta) = \mathbb{E}_{x\sim D,\,\{y_i\}_{i=1}^N\sim\pi_{\text{infer}}} \Big[ \frac{1}{\sum_{i=1}^N |y_i|} \sum_{i=1}^N \sum_{t=1}^{|y_i|} M\!\Big( \frac{\pi_{\text{train}}(y_{i,t}\mid x,y_{i,<t};\theta)}{\pi_{\text{infer}}(y_{i,t}\mid x,y_{i,<t};\theta_{\text{old}})} ;\alpha,\beta \Big) \hat A_{i,t} \Big]
$$

**掩码函数**（报告中的式 (2)）：

$$
M(k)=
\begin{cases}
k,& k\in[\alpha,\beta]\\
0,& \text{otherwise}
\end{cases}
$$

并且报告明确指出：**double-sided masking 对抗 trainer–inference mismatch "critical"**，并解释即便 $\pi_{\text{infer}}$ 与 $\pi_{\text{train}}$ 共享参数，仍会出现 token 概率显著差异导致分布漂移与训练崩溃风险。[^2]

报告还提到额外的 **rollout 级别屏蔽**：若任意 token 的重要性比率低于某阈值（例如 $1\mathrm{e}{-5}$），会对该 rollout 进行 masking（过滤）。[^2]

---

## 4) 两类典型 IS ratio 异常：定义与修正方向总览

两类异常可以概括为：

1. **Type A.1（词元级不一致）**：某些 Token 的 IS Ratio $\rho_t$ 极度偏离 1.0（例如 $< 10^{-30}$ 或 $> 10^5$）。
   常用应对：**TokenClip / TokenMask**（如 **Token-TIS**[^1]、IcePop 的双侧 **Token-MIS**[^2]）。

2. **Type A.2（序列级不一致）**：单个 token $\rho_t$ 看似正常，但累积的序列级 $\rho(y)=\prod_t\rho_t$ 严重偏移。
   常用应对：**Seq-TIS（截断）** 或 **Seq-MIS（拒绝/过滤）**[^4]，以及避免长度偏置的 **Geo-Mask**[^4]。

---

## 5) Type A.1（Token 级 ratio 极端异常）可以从哪些角度修正？

A.1 本质：存在 token 使得 $\rho_t$ 极端（$\ll 1$ 或 $\gg 1$），常见诱因包括实现差异、精度差异、量化、MoE 路由差异等，导致 token 概率在 train/infer 两端出现显著偏离。[^1]

下面按"从根因到补丁"的角度列修正思路（可叠加）。

### 5.1 系统层：尽量减少 $\pi_{\text{infer}}$ 与 $\pi_{\text{train}}$ 的差异（治本方向）

- **减少混合后端引入的分布差异**：rollout（推理引擎）与 training（训练后端）即便共享参数也可能产生不同 token 概率，这会把 on-policy 变成隐式 off-policy。[^1]
- **量化 rollout 会放大 mismatch**：论文指出 mismatch 在量化 rollouts（例如 INT8 / FP8）下更严重。[^1]
- **仅靠系统层修补往往不足**：论文还提到即便做一些系统级"精度对齐"等尝试，mismatch 仍可能持续存在，提示需要算法级修正兜底。[^1]

>（补充说明：这里"让推理端返回真实用于采样的概率""对齐 top-k/top-p/temperature 的实现"等属于通用工程建议；是否是你的具体根因需要结合日志与实现核对。）

### 5.2 算法层：TokenClip / TokenMask（A.1 的主线修正）

**Token-TIS（截断/clip）**：

$$
w_t=\min(\rho_t,C)
$$

用于压制极端大权重，避免梯度爆炸；TRL 中对应 `"token_truncate"`。[^3]

**Token-MIS（掩码/mask）**：

- **上侧 masking**：超阈值 token 的更新直接移除（TRL 文档将其作为 MIS 的定义性特征）。[^3]
- **双侧 masking（IcePop）**：

$$
M(k;\alpha,\beta)=
\begin{cases}
k,& k\in[\alpha,\beta]\\
0,& \text{otherwise}
\end{cases}
$$

并被报告为对 trainer–inference mismatch "critical"。[^2]

**"token 越界 ⇒ 丢整条 rollout" 的 safeguard**：IcePop 报告额外描述：若某 rollout 内任意 token ratio 过小（例如 $< 1\mathrm{e}{-5}$），会对该 rollout 进行 masking（等价于过滤/丢弃更新贡献）。[^2]

### 5.3 数值计算层：让 ratio 计算更稳（尤其针对极端值）

（这部分更偏通用数值稳定性原则，常见做法包括）

- **在 log-space 计算 ratio**：先算 $\log\rho_t=\log\pi_{\text{train}}-\log\pi_{\text{infer}}$，再做 clip/mask，避免浮点下溢/上溢。
- 对 denominator 加 **eps** 防止除 0（某端出现 0 prob 时尤其重要）。
- 检查并对齐采样策略实现（top-k/top-p/temperature）以避免一端"硬裁掉"导致 0 prob。

### 5.4 结构层（MoE 专项）：路由不一致会让 token-level ratio 失真

对 MoE 来说，即使输入序列相同，两次 forward 也可能因路由差异产生不同输出分布；在 RL 中这会给"旧策略概率"与 IS ratio 引入噪声，使训练不稳定甚至崩溃。[^5]

对应修正角度：

- **Rollout Routing Replay（R3）/ Router Replay**：核心思想是**复用推理时的 routing mask**，在训练侧重放，从源头对齐 train/infer 的 expert 选择。论文明确描述其机制：重用推理侧路由 mask $I_{\text{infer}}$，在训练侧形成 replay gating weights，并用于组合训练侧 experts 输出，从而实现"Aligning training and inference"。[^5]

---

## 6) Type A.2（Token ratio 看似正常，但序列级累积 ratio 严重偏移）可以从哪些角度修正？

A.2 关键在于：
即使每步 $\rho_t$ 只是轻微偏离 1，乘起来的 $\rho(y)=\prod_t\rho_t$ 也会随长度 $T$ 指数级放大/缩小。

richardli Part 2 给出非常直观的量级示例：1% 的 per-token mismatch 在 200 token 下会带来 $(1.01)^{200}\approx 7.3$ 的方差倍率；5% mismatch 则达到 $(1.05)^{200}\approx 17{,}292$。[^4]

因此 A.2 的修正往往更偏"序列粒度 / 信任域"。

### 6.1 直接上 Seq-TIS / Seq-MIS（SeqClip 的主线）

**Seq-TIS（序列级截断，soft clipping）**：

$$
\hat{g}_{\text{seq-tis}}(y)=\min(\rho(y),C)\cdot f(y)
$$

richardli Part 3 给出该形式，并对 "soft trust region" 做了讨论。[^4]

**Seq-MIS（序列级掩码/拒绝，hard rejection）**：

$$
\hat{g}_{\text{seq-mis}}(y)=\mathbb{I}(\rho(y)\le C)\cdot \rho(y)\cdot f(y)
$$

richardli Part 3 明确称其为 "Hard Trust Region via Rejection (Seq-MIS)"。[^4]

工程侧也经常提供 `"sequence_truncate"` / `"sequence_mask"` 这样的开关。[^3]
（TRL issue #4493 也明确讨论了"将 sequence-level TIS + MIS 作为默认行为"的动机与结论。）[^6]

### 6.2 处理"长度不公平"：几何均值/长度不变门控（Geo-Mask）

A.2 的典型副作用是：长序列更容易因乘积越界（即使每步很正常），导致 **length bias**。

richardli Part 3 提出用几何均值（等价于平均 log-ratio）构造"长度不变"的强信任域指标[^4]：

$$
\rho_{\text{geo}}(y)=\left(\prod_{t=0}^{T-1}\rho_t\right)^{1/T}=\rho(y)^{1/T}
$$

并给出 **Geo-Mask（双侧 hard trust region）**[^4]：

$$
\hat{g}_{\text{geo-mask}}(y)=\mathbb{I}\left(C_{\text{low}}\le\rho_{\text{geo}}(y)\le C_{\text{high}}\right)\cdot f(y)
$$

以及其 log-space 等价写法：

$$
\mathbb{I}\left(\log C_{\text{low}}\le \frac{1}{T}\sum_{t}\log\rho_t \le \log C_{\text{high}}\right)
$$

并且建议可与 Token-TIS 组合（先长度不变过滤，再 token 级裁剪加权）。[^4]

### 6.3 理论粒度更严格：prefix ratio / 前缀级修正（以及更稳近似）

A.2 的另一视角是：token-level ratio 是对更严格修正项（**prefix importance ratio**）的近似；当 off-policy 较大时该近似可能诱发不稳定。

arXiv:2601.22718 明确指出"理论上严格的修正项是 prefix importance ratio，而 token-level 近似在大 off-policy drift 下会导致不稳定甚至崩溃"，并提出 MinPRO 作为更稳的非累积替代。[^7]

### 6.4 训练流程/信任域角度：主动降低"有效 off-policyness"

这一类属于"从训练协议减少 A.2 的发生概率"，例如：

- 减少 sampler→learner 的延迟、降低 checkpoint staleness（异步/缓冲会增大 off-policy）。
- 收紧信任域（更强 KL 约束、更小学习率/步长），降低每次更新带来的分布漂移。（一般性工程策略）
- 缩短 rollout 或分段更新，降低乘积累积的长度效应。（一般性工程策略）

---

## 7) 将两类异常与 "TIS/MIS + Token/Seq 粒度"对齐（总结）

- **Type A.1（token 级极端 $\rho_t$）**：
  首选 **Token-TIS / Token-MIS（含 IcePop 双侧 mask）**[^1] [^2]；并结合系统层减小 mismatch（精度/量化/实现差异），MoE 场景优先考虑路由对齐（R3）[^5]。

- **Type A.2（token 级看似正常，但 $\prod\rho_t$ 序列级漂移）**：
  首选 **Seq-TIS / Seq-MIS**（软裁剪 vs 硬拒绝）[^4]，并用 **Geo-Mask（$\rho_{\text{geo}}$）**[^4] 缓解长序列 length bias；在更严格理论框架下可参考 **prefix importance ratio** 与其稳定近似（如 MinPRO）[^7]。

---

## 参考文献

[^1]: Yao, S. & Liu, Y. (2025). *On the Rollout-Training Mismatch in Modern RL Systems*. 系统分析了 vLLM+FSDP 混合架构下的训练–推理不一致问题，提出 TIS（Truncated Importance Sampling）作为修正手段。

[^2]: INTELLECT-3 Technical Report (2025). *INTELLECT-3: Decentralized Reinforcement Learning with IcePop*. 提出 IcePop 算法，采用双侧（double-sided）masked token-level importance sampling，含目标函数式 (1) 与掩码函数式 (2)；明确指出 double-sided masking 对训练–推理 mismatch 修正 "critical"。

[^3]: Hugging Face. *TRL Documentation — Paper Index*. 定义了 TIS / MIS 及其 token/sequence 级变体（`token_truncate` / `sequence_truncate` / `token_mask` / `sequence_mask`），并给出 `GRPOConfig` 配置接口。URL: <https://huggingface.co/docs/trl/main/en/paper_index>

[^4]: Li, Y. R. (2025). *When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch* — Part 2 & Part 3. Part 2 给出序列级 IS 方差指数爆炸的定量分析（$(1.01)^{200}\approx 7.3$ 等示例）；Part 3 提出 Seq-TIS、Seq-MIS 与 Geo-Mask 公式，定义硬/软信任域。URL: <https://richardli.xyz/post/rl-collapse-part3/>

[^5]: Zheng, C. et al. (2025). *Group Sequence Policy Optimization (GSPO)*. Qwen Team, Alibaba Inc. 分析了 MoE 路由差异导致 token-level IS ratio 不可靠的问题，提出 Rollout Routing Replay（R3）策略；提出序列级重要性比率与长度归一化方案。URL: <https://qwenlm.github.io/blog/gspo/>

[^6]: Hugging Face. *TRL Issue #4493 — [GRPO] Sequence-level TIS + MIS*. 讨论将 sequence-level TIS + MIS 作为 GRPO 默认行为的动机与实验结论。URL: <https://github.com/huggingface/trl/issues/4493>

[^7]: arXiv:2601.22718 (2025). 指出理论上严格的修正项是 prefix importance ratio，token-level 近似在大 off-policy drift 下导致不稳定，提出 MinPRO 作为更稳的非累积替代。

[^8]: veRL 框架文档. *Rollout Correction* 模块，支持 IS weights + rejection sampling 分离，提供 token/sequence/geo 三种粒度配置。

[^9]: Veach, E. & Guibas, L. J. (1995). *Optimally Combining Sampling Techniques for Monte Carlo Rendering*. 提出 Multiple Importance Sampling（MIS），属于 Monte Carlo 渲染领域，与本文 LLM-RL 语境的 Masked IS 同名但含义不同，列此条目以避免混淆。