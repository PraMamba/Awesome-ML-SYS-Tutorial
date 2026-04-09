# GRPO 变体统一模板：以"每 token 标量权重 $\alpha_{i,t} \cdot \nabla \log \pi_\theta$"视角串联十余种主流方法

> **核心视角**：$\mu_{\text{old}}$ 是采样策略（行为策略），$\pi_{\text{old}}$ 是训练里的旧策略——这个"三策略"框架非常关键。很多变体本质都是在处理**两种 drift**：旧→新、采样→训练。

---

## 1. 统一模板：每 token 权重 $\alpha_{i,t}$

### 1.1 三个策略、两类 ratio

对一个 prompt（记作 $x$ 或 $q$），采样 $G$ 条 completion（序列）：

- **行为/采样策略**：$\mu_{\text{old}}$（rollout 引擎/vLLM 侧真正生成 token 的策略）
- **旧策略**：$\pi_{\text{old}}$（训练侧用来做 proximal/对照的"旧策略"，以及用于记录 old logprob）
- **当前策略**：$\pi_\theta$（正在更新的策略）

对第 $i$ 条 completion 的第 $t$ 个 token：

- **策略更新 ratio（旧→新）**

$$
r_{i,t}(\theta) = \frac{\pi_\theta(a_{i,t} \mid s_{i,t})}{\pi_{\text{old}}(a_{i,t} \mid s_{i,t})}
$$

- **训推/采样不一致 ratio（采样→训练）**（rollout correction）

$$
c_{i,t} = \frac{\pi_{\text{old}}(a_{i,t} \mid s_{i,t})}{\mu_{\text{old}}(a_{i,t} \mid s_{i,t})}
$$

这项就是"训练与推理不一致"的关键环节；很多框架把它叫 **rollout correction / off-policy correction**。verl 的数学文档明确把这类 IS 权重写成 $\text{stopgrad}(\text{weight}) \cdot \log\pi_\theta \cdot A$ 的形态（也就是权重不反传）[^1]。

### 1.2 GRPO 的"组内优势"baseline

GRPO 的特点是**不用 value critic**，而用同一个 prompt 下的 $G$ 个样本做组内 baseline：

- 序列奖励：$R_i$（通常是 rule/verifier 给整条序列的 outcome reward）
- 组内均值：$\bar{R} = \frac{1}{G}\sum_{j=1}^{G} R_j$
- 组内标准差：$\sigma_R = \mathrm{std}(R_1, \dots, R_G)$（很多实现会用）
- **组内优势**（最常见写法）：

$$
A_i = \frac{R_i - \bar{R}}{\sigma_R + \epsilon}
$$

> 这一步就是"谁做 advantage baseline"：GRPO 用 $\bar{R}$（以及可选 std）当 baseline。
> Dr.GRPO 的核心之一就是指出这里的 **std/长度归一** 会带来偏置，并给出"Done Right"的改法（后面讲）[^2]。

### 1.3 目标统一写成"每 token 权重 $\alpha_{i,t}$"

把（最大化）代理目标写成（最小化）loss 记号，token 级一般长这样：

$$
\mathcal{L} = -\mathbb{E}\left[
\sum_{i=1}^{G} \sum_{t=1}^{T_i}
\underbrace{\text{stopgrad}\big(w_{i,t}\big)}_{\text{不反传}}
\cdot
\underbrace{\ell_{\text{prox}}\big(r_{i,t}(\theta), A_i\big)}_{\text{clip/ratio/adv}}
\right]
$$

其中：

- $w_{i,t}$ 用来装"采样→训练"的 correction（比如 $c_{i,t}$ 的截断/掩码版本）
- $\ell_{\text{prox}}$ 是 PPO/GRPO 的 proximal 结构（min-clip 那套）

对梯度来说，最关键的是：**无论你怎么变体，最后都会回到**

$$
\nabla_\theta \mathcal{L} = -\mathbb{E}\left[
\sum_{i,t} \alpha_{i,t}\;\nabla_\theta \log \pi_\theta(a_{i,t} \mid s_{i,t})
\right]
$$

你要的"每 token 最终标量权重"就是

$$
\boxed{
\alpha_{i,t} = \alpha^{\text{PG}}_{i,t}
\quad(\text{如果再加 KL，就再加 } \alpha^{\text{KL}}_{i,t})
}
$$

在 GRPO/其变体里，$\alpha^{\text{PG}}_{i,t}$ 大致可看成两块相乘：

$$
\alpha^{\text{PG}}_{i,t} =
\underbrace{\tilde{c}_{i,t}}_{\text{训推不一致的修正/过滤}}
\cdot
\underbrace{\tilde{g}\big(r_{i,t}, A_i\big)}_{\text{prox: clip/ratio/adv}}
$$

---

## 2. 原始 GRPO 的两个主要不稳定源

### 2.1 不稳定源 #1：token 级 ratio + 序列级 reward 的"粒度错配"

很多 LLM-RL 是 **outcome reward**：一条 completion 只有一个 $R_i$，所以 GRPO 会把同一个 $A_i$ 分配给所有 token。

但 **ratio 却是 token 级的**：每个 token 有自己的 $r_{i,t}$。这会导致：

- 长序列里 token 多，ratio 的噪声/极端值出现概率更高
- 训练容易被少量 token 的极端比率主导（尤其是 MoE、异步 rollout、不同后端数值误差时）

GSPO/GMPO 这一支就是在修这个。

### 2.2 不稳定源 #2：$\pi_{\text{old}} / \mu_{\text{old}}$ 的训推不一致（rollout correction）

当 rollout 与训练不共享完全一致的 backend（或 MoE 路由不一致），即便"同一 checkpoint"，也会出现 $\mu_{\text{old}} \neq \pi_{\text{old}}$。这会把训练变成**隐式 off-policy**，必须做某种 IS 修正或过滤。很多工程实践都指出"单独 rollout 后端 + 训练后端"会引入分布 mismatch，需要 importance sampling/过滤来稳住[^3]。

IcePOP / TIS / MIS / WTRS / R2 / R3 这条支线就是在修这个。

---

## 3. 逐个串讲：十余种主流 GRPO 变体各自改哪一块

---

### A. DAPO：把"样本利用率"和"长 CoT"训练先救活

DAPO 是一个非常典型的"工程配方型"变体：它不是只改一条公式，而是把 **Clip-Higher、Dynamic Sampling、Token-level loss、Overlong filtering/shaping** 组合成一套[^4]。

#### A1) Clip-Higher：把上裁剪放宽，减少"探索被 clip 掐死"

标准 min-clip 的门控（对 $A > 0$）是：当 $r > 1+\epsilon$ 时梯度直接变 0（因为进入常数区间）。

DAPO 的 **Clip-Higher** 做法是把上界放宽到 $1+\epsilon_{\text{high}}$（并且常见是非对称上界更大），让正优势样本在"提高概率"方向有更大的更新空间，从而缓解熵塌缩/探索不足[^5]。

用 $\alpha$ 视角看：它只是在 $\tilde{g}(r, A)$ 里改了"何时饱和为 0"的阈值。

- $A_i > 0$：

$$
\alpha^{\text{PG}}_{i,t} = \tilde{c}_{i,t} \cdot
\begin{cases}
A_i \cdot r_{i,t} & r_{i,t} \le 1+\epsilon_{\text{high}} \\
0 & r_{i,t} > 1+\epsilon_{\text{high}}
\end{cases}
$$

- $A_i < 0$ 的下裁剪通常仍是 $1 - \epsilon$（DAPO 主打"上边放宽"）。

#### A2) Token-level loss：不再"每条序列除以长度"，让每个 token 权重更公平

原式先对序列内 token 权重取平均，会让长序列样本每 token 的"有效更新幅度"被摊薄。

DAPO 的 **Token-Level Policy Gradient Loss** 本质就是把聚合从

$$
\frac{1}{G}\sum_i \underbrace{\frac{1}{T_i}\sum_t \alpha_{i,t}}_{\text{序列内平均}}
\quad\Rightarrow\quad
\frac{1}{\sum_i T_i}\sum_i\sum_t \alpha_{i,t}
$$

也就是"每个 token 一票"，而不是"每条序列一票"。TRL 的 paper index 也把 token-level loss 作为 DAPO 组件之一[^5]。

用 $\alpha$ 看：$\alpha_{i,t}$ 本身不一定变，但**总体归一化方式**变了，长 CoT 不再天然被压缩。

#### A3) Dynamic Sampling + Overlong Filtering：把"注定 0 梯度/噪声大"的样本剔掉

- 过滤全 0、全 1（优势全为 0 或全同号、对更新贡献很小）
- 去掉因为 max_len 截断的样本（语义不完整、奖励噪声）

这些在工程上等价于：对某些序列直接设 mask=0，让它们所有 token 的 $\alpha_{i,t} = 0$。

TRL 的 paper index 也明确写到 **Dynamic Sampling、Overlong Filtering** 属于 DAPO 关键组件（并且提示 Dynamic Sampling 在 TRL 不支持）[^5]。

---

### B. GSPO / GMPO：把"连加近似连乘"的偏差拉回到序列级（sequence-first）

token 级连加 surrogate 在某些推导下可以视作对序列级连乘（真·序列 ratio）的近似；当 ratio 偏离 1 变大、序列变长时，这个近似误差会显著放大，于是出现 GSPO/GMPO 这条"sequence 化"的路线[^6]。

#### B1) GSPO：sequence-level ratio + sequence-level clip（"先连乘后裁剪"）

GSPO 的核心动机：**奖励是序列级的，但 GRPO 用 token 级 ratio 会引入高方差与不稳定**（尤其在大规模 RL 与 MoE）[^7]。

GSPO 做法：把 token 级 ratio 聚合成**序列级 ratio**（通常用对数域求和再除以长度，等价于几何平均，避免长度导致爆炸）：

$$
s_i(\theta) = \left(\frac{\pi_\theta(o_i \mid q)}{\pi_{\text{old}}(o_i \mid q)}\right)^{1/T_i}
= \exp\left(\frac{1}{T_i}\sum_{t=1}^{T_i}\log r_{i,t}\right)
$$

TRL 的 GSPO 配置说明里就用这种"序列概率比再做长度归一"的写法[^5]。

然后用 **sequence-level clip**：

$$
\tilde{s}_i = \mathrm{clip}(s_i, 1-\epsilon, 1+\epsilon)
$$

最终每个 token 的权重就变成：

$$
\alpha^{\text{PG}}_{i,t} = \tilde{c}_{i,t} \cdot
\underbrace{A_i \cdot
\begin{cases}
s_i & \text{未被 clip 饱和} \\
0 & \text{clip 饱和区（常数）}
\end{cases}}_{\text{所有 token 共用同一个权重}}
$$

**这就是"GSPO 先连乘后裁剪"的直观含义**：clip 的门控发生在序列级，因此一旦某条序列进入饱和区，可能出现"整条序列所有 token 的 $\alpha \to 0$"的强门控（稳定但更保守）。

#### B2) GMPO：先对 token 做 clip，再做几何聚合（"先裁剪后连乘"）

GMPO 的论文动机是：GRPO/PPO 的 token 级算术平均容易被 outlier token 支配；用**几何平均**能抑制极端值、提升稳定性[^8]。

和前述对齐，可以这样理解 GMPO：

1. 先对每个 token 的 $r_{i,t}$ 做裁剪（避免单 token 爆掉）
2. 再把 token 的贡献做"连乘/几何平均式"聚合成序列权重

于是 GMPO 的序列权重更像：

$$
s_i^{\text{GMPO}} = \exp\left(\frac{1}{T_i}\sum_t \log \mathrm{clip}(r_{i,t})\right)
$$

再乘上 $A_i$，得到每 token 共用的缩放（或等价的几何型 surrogate）。

**对比 GSPO：**

- GSPO：先算"真实（或更接近真实）的序列 ratio"，再 clip（门控更强、更像 trust region）
- GMPO：先把每个 token 的 ratio 截断，再聚合（门控更柔、更抗 outlier）

---

### C. 训推不一致（$\pi_{\text{old}} / \mu_{\text{old}}$）这条线：IcePOP / TIS / MIS / WTRS

核心视角：

> **所有方法都在做同一件事：把 correction 权重 $\tilde{c}_{i,t}$ 设计得"既能纠偏，又不把方差/噪声炸穿"。**

verl 的 rollout correction 数学文档把 Seq-TIS/Seq-MIS 的定义写得很明确：Seq-TIS 是 **clip 序列 ratio**，Seq-MIS 是 **超过阈值直接拒绝/掩码序列**，并且把 IS 权重作为 stopgrad 放进 REINFORCE/PG 目标里[^1]。

#### C1) TIS：Truncated Importance Sampling（截断 IS）——"都用，但别给它太大权重"

典型形式：

- **Seq-TIS（序列级）**：

$$
\rho(\tau) = \frac{\pi_{\text{old}}(\tau)}{\mu_{\text{old}}(\tau)}, \quad
\tilde{\rho}(\tau) = \min(\rho(\tau), C)
$$

verl 文档就用这句话直接定义 Seq-TIS[^1]。

- **Token-TIS（token 级）**：对每个 token 的 $c_{i,t}$ 做 clip（降低方差，但引入一定偏置），属于 verl 文档里"moderate mismatch"场景常用项[^1]。

在 $\alpha$ 模板里，就是：

$$
\tilde{c}_{i,t} = \mathrm{clip}(c_{i,t}, \cdot) \quad \text{或} \quad
\tilde{c}_i = \min(c_i, C)\;\;\forall\, t \in i
$$

然后

$$
\alpha^{\text{PG}}_{i,t} = \tilde{c}_{i,t} \cdot \tilde{g}(r_{i,t}, A_i)
$$

#### C2) MIS：Masked / Rejected IS（掩码/拒绝 IS）——"宁可不用，也别用毒尾巴"

verl 文档对 **Seq-MIS** 的定义非常直白：**与其 clip，不如直接拒绝 $\rho(\tau) > C$ 的序列**，把它当作 hard trust region filter，适合 mismatch 严重或 tail "toxic" 的场景[^1]。

对应到 $\alpha$：

$$
\tilde{c}_i =
\begin{cases}
c_i & c_i \le C \\
0 & c_i > C
\end{cases}
\quad\Rightarrow\quad
\alpha_{i,t} = 0 \;\; \forall\, t
$$

#### C3) WTRS（Worst Token Reject Sampling）/ Token Veto：一票否决整条序列

WTRS 是更极端的版本：看"最差 token"，一票否决整条序列。

verl 的 token veto 机制在一些总结里被表述为：计算 token 级重要性比率，如果轨迹中存在任何 token 使得 $\min_t \rho_t < \tau_{\text{veto}}$，整条序列丢弃[^9]。

对 $\alpha$ 来说，这就是：

$$
\mathbf{1}\Big[\min_t c_{i,t} \ge \tau_{\text{veto}}\Big]
\quad\text{作为整条序列的 mask}
$$

#### C4) IcePOP：把"训推不一致的极端 token"直接冻结/丢弃（token 级安全阀）

IcePOP 放在"类似 clip、直接舍弃显著偏离 token"这一类。在一些系统论文/卡片里，IcePOP 被作为应对训练-推理不一致（尤其高性能并行/异步/不同后端）导致的 RL 不稳定的一类策略被提到[^10]。

如果用模板描述，它最接近：

- **Token-MIS**：当 $c_{i,t}$ 太离谱（上溢/下溢）时，把该 token 的 $\alpha_{i,t}$ 置 0
- 或者更激进：触发"token veto"升级为整条序列置 0

---

### D. clip 的"硬裁剪太浪费样本"这条线：CISPO / SAPO（以及同类 GPPO / CE-GPPO）

核心矛盾：

- clip 太少：proximal 约束弱，容易崩
- clip 太多：$\alpha \to 0$ 的 token 多，探索被掐死，样本利用率下降

于是出现一类"把硬 clip 变软/变可导"的算法。

#### D1) CISPO：clip 权重但保留梯度（detach + $\log\pi$ 形式）

swift 文档给了 CISPO 的一个清晰表达：它把 $\min(r_t, \epsilon_{\text{high}})$ **detach**，然后乘 $A_t \log \pi_\theta$[^11]。

$$
\mathcal{L}_{\text{CISPO}} = -\mathbb{E}\Big[
\mathrm{detach}\big(\min(r_t, \epsilon_{\text{high}})\big) \cdot \hat{A}_t \cdot \log\pi_\theta(a_t \mid s_t)
\Big]
$$

这在"模板语言"里非常好记：

- **ratio** 还是 $r_t$
- **但 forward 用 clip 后的 ratio，当作常数权重（detach）**
- **梯度来自 $\nabla \log \pi_\theta$**，因此不会像 PPO 那样在饱和区直接"彻底断梯度"

从 $\alpha$ 角度：

$$
\alpha^{\text{PG}}_{i,t} = \tilde{c}_{i,t} \cdot
\mathrm{detach}\big(\min(r_{i,t}, \epsilon_{\text{high}})\big) \cdot A_i
$$

它相当于用"detach 后的 clipped ratio"当**重要性权重**，而不是 PPO 那种 piecewise 让梯度变 0 的门控。

#### D2) SAPO：用 Sigmoid 做"软裁剪门"，温度 $\tau$ 控制硬/软

SAPO 的描述通常是：用一个温度可调的平滑门控来衰减 off-policy 更新，同时保持在 $r \approx 1$ 附近的梯度性质[^12]。

理解方式：

- PPO 的硬 clip 是"超过阈值直接门控为 0 梯度"
- SAPO 让这个门控变成连续的：$r$ 越离谱，权重越小，但不突然断掉
- $\tau$ 越小越接近硬裁剪，$\tau$ 越大越平滑

在 $\alpha$ 模板里就是把 $\tilde{g}(r, A)$ 的门控函数换掉：

$$
\alpha^{\text{PG}}_{i,t} = \tilde{c}_{i,t} \cdot
A_i \cdot r_{i,t} \cdot \underbrace{\sigma\!\left(\frac{\text{something}(r_{i,t})}{\tau}\right)}_{\text{软门控}}
$$

（"something"可以是 $r - 1$ 或 $\log r$，不同实现略有差别；关键是"软门控 + 温度"。）

#### D3) 同类：GPPO / CE-GPPO（属于同一"保梯度裁剪"分支）

这一支线还有一些名字，比如 Gradient-Preserving Clipping Policy Optimization（GPPO）或 CE-GPPO，思路也是"硬 clip 丢梯度会抑制探索，想保留受控梯度"[^13]。

它们和 CISPO/SAPO 的关系更像"同一祖先，不同门控函数/不同推导"。

---

### E. MoE 专项：R2 / R3（Routing Replay）——把 routing 不一致这个放大器关掉

MoE 的路由会放大两种不一致性：

$$
\frac{\pi_\theta}{\mu_{\text{old}}} =
\underbrace{\frac{\pi_\theta}{\pi_{\text{old}}}}_{\text{旧→新}}
\cdot
\underbrace{\frac{\pi_{\text{old}}}{\mu_{\text{old}}}}_{\text{采样→训练}}
$$

而且 MoE 里"同权重不同后端/不同精度"会导致 router top-k 选择不同，从而 logprob 差异巨大。

一些实践/总结把 R2/R3 归为 **routing replay**[^6]：

- **R2（Vanilla Routing Replay）**：训练时复用 rollout 阶段记录的路由（通常是"训练侧"能复现的路由），降低旧→新不一致
- **R3（Rollout Routing Replay）**：更进一步，把"rollout 后端实际用的路由"记录下来并在训练中重放，从而同时降低采样→训练与旧→新不一致

从 $\alpha$ 模板看，R2/R3 不是改 $\tilde{g}(r, A)$，而是让 **$c_{i,t}$ 更接近 1、$r_{i,t}$ 更温和**——也就是从源头上让两个 ratio 不要离谱，减少 TIS/MIS/WTRS 这种"补救型"手段的触发频率。

---

### F. Dr.GRPO：把 GRPO 里的"长度与 std 归一偏置"去掉（Done Right）

Dr.GRPO（出自对 R1-Zero-like 训练的批判性分析）指出 GRPO 的优化偏置会推动"错误样本也变长"等现象，并给出一个简单修正：**去掉长度归一 + 去掉 std 归一**。论文摘要和正文都明确写到"Dr.GRPO 通过移除长度与 std 归一来修复偏置"[^2]。

用模板语言翻译就是：

- 原 GRPO 常见做法：

$$
A_i = \frac{R_i - \bar{R}}{\sigma_R}, \quad \text{并且 loss 里再除 } T_i
$$

- Dr.GRPO 倾向：

$$
A_i = (R_i - \bar{R}), \quad \text{不再除 } T_i
$$

所以 Dr.GRPO 的"每 token $\alpha$"会更直接、更不偏向"变长就是赚"。

---

## 4. 变体坐标系：各自解决什么、牺牲什么？

用四个坐标轴把十几种变体一次性定位：

### 轴 1：序列 vs token 粒度（奖励在序列、ratio 在哪一层）

| 粒度 | 代表方法 | 说明 |
|------|---------|------|
| token ratio | GRPO（原始）、DAPO | DAPO 仍然 token ratio，但改聚合/采样 |
| sequence ratio | GSPO（最典型）、GMPO | GMPO 折中：先 token clip 再 sequence 聚合 |

- **稳定性**：通常 sequence > token
- **探索/学习信号密度**：通常 token > sequence（因为 sequence clip 一旦饱和会整条断）

### 轴 2：硬裁剪 vs 软门控

| 类型 | 代表方法 |
|------|---------|
| 硬裁剪 | PPO/GRPO min-clip、IcePOP/MIS/WTRS |
| 软门控 | SAPO（sigmoid/温度）、GPPO/CE-GPPO |

- **稳定性**：硬门控更保守
- **样本利用率/探索**：软门控通常更好（但要调温度）

### 轴 3：训推不一致的处理力度（$\pi_{\text{old}} / \mu_{\text{old}}$）

| 力度 | 方法 | 特点 |
|------|------|------|
| 不处理 | — | 最省事，但最容易崩 |
| 截断 | TIS | 偏差↔方差折中 |
| 拒绝 | MIS | 拒绝"毒尾巴" |
| 一票否决 | WTRS | 最保守 |
| 源头消除 | R3 | 尤其 MoE，从 routing 层面减少 mismatch |

### 轴 4：归一化/基线是否引入偏置

| 方法 | 处理方式 |
|------|---------|
| GRPO | mean baseline + std + 长度平均 |
| Dr.GRPO | 去掉 std 与长度归一（更"无偏"，也更 token-efficient）[^2] |

---

## 5. 最终 Takeaway：十余种变体的演化链（问题驱动）

1. **GRPO**：用 group baseline 代替 critic，形成"简单可扩展"的 RLVR 方案
2. **DAPO**：发现长 CoT 下样本利用率/稳定性不够 → Clip-Higher + Dynamic Sampling + Token-level loss + overlong 处理救活大规模训练[^4] [^14]
3. **GSPO / GMPO**：发现 token ratio + sequence reward 的粒度错配/方差过大 → 走向 sequence 化（GSPO 更"trust region"、GMPO 更"抗 outlier"）[^7]
4. **IcePOP / TIS / MIS / WTRS**：发现并行 rollout / 不同后端导致 $\mu_{\text{old}} \neq \pi_{\text{old}}$，训练隐式 off-policy → rollout correction（截断/拒绝/一票否决）[^1]
5. **R2 / R3（MoE）**：发现 MoE 路由把 mismatch 放大 → routing replay，从源头把 ratio 拉回 1 附近[^6]
6. **CISPO / SAPO（及同类）**：发现硬 clip 牺牲太多梯度、探索不足 → 用 detach/soft gate 保留梯度但仍控更新幅度[^11]
7. **Dr.GRPO**：回头发现 GRPO 的 std/长度归一会引入系统性偏置（尤其推长错误回答） → 去偏置、提升 token efficiency[^2]

---

## 参考文献

[^1]: verl Documentation — Rollout Correction Math. <https://verl.readthedocs.io/en/latest/algo/rollout_corr_math.html>

[^2]: "Understanding R1-Zero-Like Training: A Critical Perspective." arXiv:2503.20783. <https://arxiv.org/html/2503.20783v1>

[^3]: LLM Data Blog — Mismatch Praxis. <https://llmdata.com/blog/mismatch-praxis/>

[^4]: DAPO. arXiv:2503.14476. <https://arxiv.org/abs/2503.14476>

[^5]: Hugging Face TRL — Paper Index (v0.22.1). <https://huggingface.co/docs/trl/v0.22.1/paper_index>

[^6]: MoE RL / R2-R3 Routing Replay. arXiv:2512.01374v3. <https://arxiv.org/html/2512.01374v3>

[^7]: GSPO. NASA ADS: 2025arXiv250718071Z. <https://ui.adsabs.harvard.edu/abs/2025arXiv250718071Z/abstract>

[^8]: GMPO. OpenReview: nCEs0tSwc2. <https://openreview.net/forum?id=nCEs0tSwc2>

[^9]: Xihuai Wang — Three-Policy Perspective. <https://xihuai18.github.io/reinforcement-learning/2025/11/15/three-policy-en.html>

[^10]: IcePOP. Hugging Face Papers: 2510.18855. <https://huggingface.co/papers/2510.18855>

[^11]: CISPO — Swift Documentation. <https://swift.readthedocs.io/en/latest/Instruction/GRPO/AdvancedResearch/CISPO.html>

[^12]: SAPO. Hugging Face Papers: 2511.20347. <https://huggingface.co/papers/2511.20347>

[^13]: GPPO / CE-GPPO. arXiv:2509.20712v2. <https://arxiv.org/html/2509.20712v2>

[^14]: DAPO (PDF). arXiv:2503.14476. <https://arxiv.org/pdf/2503.14476>