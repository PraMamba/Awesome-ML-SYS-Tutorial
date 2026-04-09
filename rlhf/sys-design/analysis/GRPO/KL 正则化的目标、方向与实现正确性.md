# 附录：KL 正则化的目标、方向与实现正确性（RPG 视角）

---

## 统一设定（本文全附录默认）

- $x \sim \mathcal{D}$：prompt（或状态）分布（工程里通常就是训练集 prompts）。
- $y = (a_1, \dots, a_T) \sim \pi_\theta(\cdot \mid x)$：在 prompt $x$ 下，策略（LLM）生成的完整 completion（token 序列）。
- $R(x,y) \in \mathbb{R}$：序列级奖励（reward model / 规则奖励 / verifier 等）。
- $\pi_{\text{ref}}$：参考策略（reference/SFT），用于对齐锚定。
- $\pi_{\text{old}}$：行为/采样策略（behavior/old），用于 rollout 采样（通常是 $\theta$ 的延迟拷贝）。
- off-policy 重要性权重（序列级）

$$
w_\theta(x,y) \;\triangleq\; \frac{\pi_\theta(y \mid x)}{\pi_{\text{old}}(y \mid x)} \quad\text{where } \pi(y \mid x) = \prod_t \pi(a_t \mid h_t)
$$

---

## A. KL 正则 RL 的三种形式（Penalty / Constraint / Dual）

### A.1 Penalty 形式（primal-penalty；固定 $\beta$）

#### A.1.1 KL-penalty 的原始优化问题（最常见写法）

给定某个"锚定策略"$\pi_{\text{anchor}}$（工程里常用 $\pi_{\text{ref}}$ 或 $\pi_{\text{old}}$），KL-penalty 的 primal 问题写作：

$$
\max_{\theta}\; J_{\text{pen}}(\theta) \;\triangleq\; \mathbb{E}_{x \sim \mathcal{D}}\left[ \mathbb{E}_{y \sim \pi_\theta(\cdot|x)}[R(x,y)] \;-\; \beta\,\mathrm{KL}\!\big(\pi_\theta(\cdot|x) \,\|\, \pi_{\text{anchor}}(\cdot|x)\big) \right]
$$

- 当 $\pi_{\text{anchor}} = \pi_{\text{ref}}$ 时：这是"对参考模型的 KL 锚定"（典型 RLHF）。
- 当 $\pi_{\text{anchor}} = \pi_{\text{old}}$ 时：这是"对旧策略的 trust region"（典型 TRPO/PPO 视角）。

在 RLHF 实践中，很多工作把这一项直接写进 RL 训练目标，并把 $\beta$ 作为 **KL reward coefficient** 来调强度。比如 InstructGPT 论文把 $\beta$ 作为 KL penalty 的系数显式写在 RL 目标中[^1]。

#### A.1.2 "penalty = reward shaping"的工程等价（直觉）

当 KL 取 reverse 方向（$\mathrm{KL}(\pi_\theta \| \pi_{\text{ref}})$）且期望对 $\pi_\theta$ 取时，可以把 KL 项视为对 reward 的塑形：

$$
\tilde{R}(x,y) = R(x,y) - \beta\big(\log\pi_\theta(y|x) - \log\pi_{\text{ref}}(y|x)\big)
$$

因此在实现里常见写法是：**reward = task_reward − β·log-ratio**（token-level 则逐 token 累加）。

> **重要**：这个"看起来像 reward shaping"的写法，只有在你实现的梯度/估计器与目标一致时才成立；否则可能"形似而神不似"。（后面 B/C 专门讲这个一致性。）

### A.2 Constraint 形式（primal-constraint；$\mathrm{KL} \le \epsilon$）

#### A.2.1 KL trust region / KL budget 的原始约束问题

把 KL 从 penalty 换成硬约束：

$$
\max_{\theta}\; \mathbb{E}_{x}\,\mathbb{E}_{y \sim \pi_\theta}[R(x,y)] \quad \text{s.t.} \quad \mathbb{E}_{x \sim \mathcal{D}}\!\big[\mathrm{KL}\!\big(\pi_\theta(\cdot|x) \| \pi_{\text{anchor}}(\cdot|x)\big)\big] \le \epsilon
$$

Ziegler et al. 在"从人类偏好微调语言模型"的 RL 训练中明确提到使用 KL 约束来防止策略漂移过远[^2]。

### A.3 Dual 形式（对偶：自动调 $\beta$）

#### A.3.1 拉格朗日形式（dual Lagrangian）

将约束问题写成拉格朗日：

$$
\max_{\theta}\min_{\beta \ge 0}\;\; \mathcal{L}(\theta, \beta) = \mathbb{E}[R] - \beta\big(\mathbb{E}[\mathrm{KL}] - \epsilon\big)
$$

- 对 $\theta$ 最大化：等价于一个 KL-penalty 目标（只是多了常数 $\beta\epsilon$）。
- 对 $\beta$ 最小化：迫使 $\mathbb{E}[\mathrm{KL}]$ 接近目标 $\epsilon$。

#### A.3.2 一个"最小可用"的对偶更新（梯度法）

用随机估计 $\widehat{\mathrm{KL}}$ 做对偶更新：

$$
\beta \leftarrow \left[\beta + \eta_\beta\big(\widehat{\mathrm{KL}} - \epsilon\big)\right]_+
$$

直观：如果 KL 超过预算 $\epsilon$，就增大 $\beta$（惩罚更强）；反之减小 $\beta$。

### A.4 工程实现：KL 控制器（log-space、clip、EMA）

#### A.4.1 Ziegler et al.（2019）的 log-space proportional controller（实践证据）

Ziegler et al. 给出了一种**乘法更新**的比例控制器，用于动态调整 $\beta$ 以命中目标 KL（文中记参考模型为 $\rho$）[^2]：

$$
e_t = \mathrm{clip}\!\left(\frac{\mathrm{KL}(\pi_t, \rho) - \mathrm{KL}_{\text{target}}}{\mathrm{KL}_{\text{target}}},\, -0.2,\, 0.2\right), \qquad \beta_{t+1} = \beta_t\big(1 + K_\beta\, e_t\big)
$$

并给出其目的：让不同实验在 KL 规模上可比较[^2]。

#### A.4.2 稳定化建议（你后续写"可复现工程细节"时建议明确写出来）

1. **log-space 参数化**：令 $\beta = \exp(\alpha)$，更新 $\alpha$ 而非 $\beta$，天然保证 $\beta > 0$。
2. **EMA 平滑测量 KL**：

$$
\overline{\mathrm{KL}}_t = (1 - \lambda)\,\overline{\mathrm{KL}}_{t-1} + \lambda\,\widehat{\mathrm{KL}}_t
$$

用 $\overline{\mathrm{KL}}_t$ 替代瞬时 KL 做控制，减少长序列/高方差下的抖动。

3. **clip 相对误差**：Ziegler 的 $[-0.2, 0.2]$ clip 就是典型做法[^2]。
4. **分布式/异步**：先对 worker 的 KL 做全局聚合（mean/EMA），再更新 $\beta$，避免每个 worker 自己调导致系统级震荡。

---

## B. KL 方向（Forward vs Reverse）与梯度形态（RPG 命题）

> 本节的"forward/reverse"是 RPG 的命名：
>
> - **Forward KL**：$\mathrm{KL}(\pi_{\text{old}} \| \pi_\theta)$（期望对 old 取）
> - **Reverse KL**：$\mathrm{KL}(\pi_\theta \| \pi_{\text{old}})$（期望对 current 取）
>
> 并且讨论在 **off-policy 采样 $y \sim \pi_{\text{old}}$** 时，如何写出真梯度与一个可用的 surrogate loss。

### B.1 off-policy 采样与重要性权重

对每个 prompt $x$，采样来自 $\pi_{\text{old}}(\cdot|x)$，并定义

$$
w_\theta(x,y) = \frac{\pi_\theta(y|x)}{\pi_{\text{old}}(y|x)}
$$

RPG 统一地把正则化 policy gradient 写成

$$
\nabla_\theta J(\theta) = \mathbb{E}\big[\text{Weight}(x,y;\theta) \cdot \nabla_\theta \log\pi_\theta(y|x)\big]
$$

并据此构造 REINFORCE-style surrogate（需要 stop-gradient，后面 B.5）[^3]。

### B.2 命题：Forward KL（$\mathrm{KL}(\pi_{\text{old}} \| \pi_\theta)$）的真梯度与可微 surrogate

#### B.2.1 目标函数（normalized forward KL regularization）

定义（省略外层 $\mathbb{E}_{x \sim \mathcal{D}}$ 书写）：

$$
J_{\mathrm{FKL}}(\theta) = \mathbb{E}_{y \sim \pi_\theta}[R(x,y)] - \beta\,\mathrm{KL}\!\big(\pi_{\text{old}}(\cdot|x) \,\|\, \pi_\theta(\cdot|x)\big)
$$

#### B.2.2 真梯度（off-policy 形式）

RPG 给出的 off-policy 形式梯度为[^3]：

$$
\nabla_\theta J_{\mathrm{FKL}}(\theta) = \mathbb{E}_{y \sim \pi_{\text{old}}}\!\left[ \big(w_\theta(x,y)\,R(x,y) + \beta\big)\;\nabla_\theta \log \pi_\theta(y|x) \right]
$$

#### B.2.3 一个对应的 fully-differentiable surrogate loss

相应地，可用如下 surrogate（最小化它等价于最大化 $J$）[^3]：

$$
L_{\mathrm{FKL}}(\theta) = \mathbb{E}_{y \sim \pi_{\text{old}}}\!\left[ -w_\theta(x,y)\,R(x,y) \;-\; \beta \log\pi_\theta(y|x) \right], \qquad \nabla_\theta L_{\mathrm{FKL}} = -\nabla_\theta J_{\mathrm{FKL}}
$$

### B.3 命题：Reverse KL（$\mathrm{KL}(\pi_\theta \| \pi_{\text{old}})$）的真梯度与可微 surrogate

#### B.3.1 目标函数（normalized reverse KL regularization）

$$
J_{\mathrm{RKL}}(\theta) = \mathbb{E}_{y \sim \pi_\theta}[R(x,y)] - \beta\,\mathrm{KL}\!\big(\pi_\theta(\cdot|x) \,\|\, \pi_{\text{old}}(\cdot|x)\big)
$$

#### B.3.2 真梯度（off-policy 形式）

RPG 给出的 off-policy 形式梯度为[^3]：

$$
\nabla_\theta J_{\mathrm{RKL}}(\theta) = \mathbb{E}_{y \sim \pi_{\text{old}}}\!\left[ w_\theta(x,y)\big(R(x,y) - \beta(\log w_\theta(x,y) + 1)\big)\;\nabla_\theta\log\pi_\theta(y|x) \right]
$$

#### B.3.3 一个对应的 fully-differentiable surrogate loss

RPG 同时给出一个可微 surrogate（其梯度等于 $-\nabla J$）[^3]：

$$
L_{\mathrm{RKL}}(\theta) = \mathbb{E}_{y \sim \pi_{\text{old}}}\!\left[ w_\theta(x,y)\big(-R(x,y) + \beta\log w_\theta(x,y)\big) \right]
$$

### B.4 一句话解释（方向差异的"梯度语义"）

- **Forward KL = 在 old 样本上做 MLE（当 $R=0$）**：RPG 明确指出，当 $R(x,y) = 0$ 时，最大化 $J_{\mathrm{FKL}}$ 退化为最小化 $\beta\,\mathrm{KL}(\pi_{\text{old}} \| \pi_\theta)$，即在 $\pi_{\text{old}}$ 样本上做最大似然拟合[^3]。
- **Reverse KL = log-ratio shaping**：$\log w_\theta$ 以 $-\beta(\log w_\theta + 1)$ 的形式进入"类似 advantage 的项"，因此更像"对 log-ratio 的塑形/惩罚"，倾向于对已采样到的高质量区域做更激进的再分配[^3]。

### B.5 从"真梯度"到"工程可写的 loss"：为什么 REINFORCE-style 必须 stop-grad？

RPG 给出一类通用的 REINFORCE-style surrogate[^3]：

$$
L_{\text{REINFORCE}}(\theta) = -\mathbb{E}_{y \sim \pi_{\text{sampling}}}\!\left[\mathrm{sg}\!\big(\text{Weight}(x,y;\theta)\big) \cdot \log \pi_\theta(y|x)\right]
$$

其关键点是：**stop-gradient 必须阻断权重项里通过 $w_\theta$ 引入的 $\theta$-依赖**，否则 autograd 会把"$\nabla w_\theta$"也算进去，从而不再对应你想要的 policy-gradient 形式[^3]。

RPG 在 forward KL 的 REINFORCE-style 形式中也明确强调：需要 $\mathrm{sg}$ 防止梯度穿过 $w(x) = \pi_\theta / \pi_{\text{old}}$[^3]。

---

## C. 估计器与实现正确性（k3 = UKL；off-policy 必须带权）

### C.1 UKL（unnormalized KL）的定义：广义 KL + mass correction

当参考分布允许是"未归一化测度"（RPG 用 $\pi_{\text{old}}$ 可能 unnormalized 的情形来统一推导），UKL 定义为[^3]：

$$
\mathrm{UKL}(\pi_\theta \| \pi_{\text{old}}) = \int \pi_\theta(u)\log\frac{\pi_\theta(u)}{\pi_{\text{old}}(u)}\,du \;+\; \int\!\big(\pi_{\text{old}}(u) - \pi_\theta(u)\big)\,du
$$

第二项是 **mass correction**（质量修正）。

> 这一定义是后面"k3 $\Leftrightarrow$ UKL"的核心：k3 不仅在惩罚 log-ratio，也隐含了一个"质量差"项。

### C.2 命题：k3 估计器与 UKL 等价（RPG Appendix D）

RPG 指出 Schulman（2020）常用的 k3 函数

$$
k_3(y) = y - 1 - \log y
$$

对应的"KL-k3"实际上等价于 UKL（不只是普通 KL）[^3]。

更具体地，RPG 给出等价关系（举例）：

$$
\mathrm{KL}_{k3}(\pi_\theta \| \pi_{\text{old}}) \;\triangleq\; \mathbb{E}_{u \sim \pi_\theta}\!\left[k_3\!\left(\frac{\pi_{\text{old}}(u)}{\pi_\theta(u)}\right)\right] \;\equiv\; \mathrm{UKL}(\pi_\theta \| \pi_{\text{old}})
$$

**工程含义（必须写在"严谨版"里）**：如果你的实现里"KL penalty"写的是 k3 形式，那么你在数学上正则的其实是 UKL，而不是标准（归一化）KL；因此你在论文/代码注释里写"KL"会造成概念错配，尤其当你再叠加 off-policy 采样时，这个错配会直接反映为梯度不一致。

### C.3 GRPO 的 KL 项为什么在 off-policy 下"不对"？（missing weight）以及如何修正

RPG 对 GRPO 的一个核心观察是[^3]：

- GRPO 的 KL penalty 常用形式与 $\mathrm{UKL}(\pi_\theta \| \pi_{\text{ref}})$ 相关（k3 结构）。
- 但训练数据是用 $\pi_{\text{old}}$ 采样得到的（off-policy），此时对"目标 UKL"的无偏/一致估计一般需要乘上重要性权重 $w$。

RPG 明确指出：如果你意图用 GRPO 的 KL penalty 去近似 $\beta \cdot \mathrm{UKL}(\pi_\theta \| \pi_{\text{ref}})$，在 off-policy（从 $\pi_{\text{old}}$ 采样）下应当出现 $w$；直接不加权的做法一般不会给出目标函数的正确梯度[^3]。

并给出一个**清晰的修正估计器**（按 token 历史 $h_{i,t}$ 写）[^3]：

$$
\mathrm{KL}^{\text{GRPO-corrected}}(h_{i,t};\theta) = \mathbb{E}_{o_{i,t} \sim \pi_{\text{old}}(\cdot|h_{i,t})} \left[ w_{i,t} \cdot k_3\!\left(\frac{\pi_{\text{ref}}(o_{i,t} \mid h_{i,t})}{\pi_\theta(o_{i,t} \mid h_{i,t})}\right) \right]
$$

其中

$$
w_{i,t} = \frac{\pi_\theta(o_{i,t} \mid h_{i,t})}{\pi_{\text{old}}(o_{i,t} \mid h_{i,t})}
$$

### C.4 "闭环一致性"检查清单（建议你在严谨版里放一段）

当你写/读一个 RLHF / RFT 实现时，建议强制检查这 4 件事（否则很容易出现"看着像 PPO/GRPO，实际梯度不是那个目标"的情况）：

1. **你声称的目标里用的是 KL 还是 UKL？方向是什么？** 如果实现是 k3：优先假定你在做 UKL（见 C.2）[^3]。
2. **你的采样分布是谁？$\pi_\theta$、$\pi_{\text{old}}$、还是 $\pi_{\text{ref}}$？**
3. **若采样分布 $\neq$ 目标期望的分布，是否需要重要性权重 $w$？** GRPO KL 项的 mismatch 就是这一步漏了[^3]。
4. **你的"工程 KL 控制器"在控制哪个量？（per-token mean vs per-seq sum）并且该量与目标中的 KL 定义一致吗？** Ziegler 的 controller 是以目标 KL 为基准动态调 $\beta$ 的典型实践[^2]。

---

## 后续对接建议

如果你接下来要把这段附录与前文的 **token-level KL shaping（PPO/GRPO/DAPO/GSPO）** 直接对齐，建议做一个小改动：把上面 B/C 的 $y$ 从"整段 completion"进一步拆成 token-level，并明确写出

- token-level $w_{t} = \frac{\pi_\theta(a_t|h_t)}{\pi_{\text{old}}(a_t|h_t)}$ 与 sequence-level $w = \prod_t w_t$ 的关系；
- 以及"控制 KL"到底控制 $\sum_t \mathrm{KL}_t$ 还是 $\frac{1}{T}\sum_t \mathrm{KL}_t$。

---

## 参考文献

[^1]: Ouyang, L., et al. "Training language models to follow instructions with human feedback." (InstructGPT). <https://cdn.openai.com/papers/Training_language_models_to_follow_instructions_with_human_feedback.pdf>

[^2]: Ziegler, D. M., et al. "Fine-Tuning Language Models from Human Preferences." arXiv:1909.08593. <https://arxiv.org/pdf/1909.08593>

[^3]: "On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning." arXiv:2505.17508. <https://arxiv.org/pdf/2505.17508>