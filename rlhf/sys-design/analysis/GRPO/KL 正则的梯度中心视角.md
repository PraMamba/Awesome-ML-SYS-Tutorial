# RLHF 中 KL 正则的梯度中心视角：深入解读与工程实操指南

---

## 1. 这篇报告在"纠正"什么误区？

RLHF 常用目标（Eq.(2)）是：

$$
\mathcal{J}(\theta) = \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(\cdot|x)}[r(x,y)] - \beta \, D_{KL}(\pi_\theta(\cdot|x) \| \pi_{\rm ref}(\cdot|x))
$$

其中 KL 正则的作用是：

- 稳定训练、防止策略在奖励信号上过拟合/奖励黑客
- 防止策略过度偏离参考（SFT / ref）导致能力或风格崩坏

**行业里常见误区：**
很多人挑选 KL 的"实现形式"（比如 GRPO 用的 $k_3$）时，拿的是"KL 值估计（value estimation）好不好"的标准：无偏/低方差/更好估计 KL 数值。
但 RL 里真正起作用的是"这个项最终给到参数更新的梯度长什么样"，也就是**优化信号（gradient signal）**。

这篇报告的主张就是：

> KL 正则在 RLHF 中应该按"梯度性质"来设计与评估，而不是按"它是不是一个好的 KL 值估计器"。

并且他们给了一个非常有杀伤力的反例：**$k_1$ 作为 loss（直接反传）在 on-policy 下期望梯度为 0，等于没正则、还加噪声。**
这一下就把"无偏值估计 ⇒ 好的优化 loss"这个直觉彻底推翻了。

---

## 2. 统一框架：为什么要区分 "in reward" vs "as loss"？

文中把 KL 的实现分成两种"风格"：

### A) $k_n$ 作为 detached 系数（"in reward"）

也就是把 $k_n$ 当成一个**不传梯度的标量系数**，去乘 score function：

$$
\mathbb{E}_{y \sim \pi_{\theta}(\cdot|x)}\Big[k_n(\text{detach}) \cdot \nabla_\theta \log \pi_\theta(y|x)\Big]
$$

工程上等价于：把 KL 项塞进 advantage 或 reward shaping 里，然后走标准 policy gradient / PPO 的那条路（优势 × logprob）。

最典型是 PPO/RLHF 里常见的

$$
k_1 = \log\pi_\theta(y|x) - \log\pi_{\rm ref}(y|x) = \log\frac{\pi_\theta}{\pi_{\rm ref}}
$$

然后用 $r - \beta k_1$ 作为"合并后的优势/奖励系数"。

### B) $k_n$ 作为直接 loss（"as loss"）

也就是写一个看似"普通监督学习"的 loss：

$$
\mathbb{E}_{y \sim \pi_{\theta}(\cdot|x)}[k_n(\pi_\theta(y|x), \pi_{\rm ref}(y|x))]
$$

然后对它直接反传。

GRPO 的典型就是把 $k_3$ 当 loss 加进去。

---

## 3. 关键洞见：真正决定"这项在拉谁、往哪拉"的，是一个梯度系数 $c(y)$

文中 Figure 1 的思路非常重要：
无论你写的是"in reward"还是"as loss"，在 policy gradient 的视角里，KL 项最终都会体现为：

$$
\mathbb{E}\Big[c(x,y)\,\nabla_\theta \log \pi_\theta(y|x)\Big]
$$

其中 $c(x,y)$ 是一个标量（通常被 detach），可以理解成对每个采样动作 $y$ 的"拉回/惩罚力度"。

因此比较不同 KL 实现，本质是比较：
**它们诱导出的 $c(x,y)$ 长什么样、稳定不稳定、尾部行为如何、方差如何。**

---

## 4. 先搞清楚"真正的 Reverse KL 梯度"是什么

Reverse KL（RKL）目标：

$$
\mathcal{J}_{RKL}(\theta) = \mathbb{E}_{y \sim \pi_\theta}\Big[\log\pi_\theta(y|x) - \log\pi_{\rm ref}(y|x)\Big]
$$

文中推导得到（Eq.(12)/(42)）：

$$
\nabla_\theta \mathcal{J}_{RKL}(\theta)
= \mathbb{E}_{y \sim \pi_\theta}\Big[
\underbrace{\log\frac{\pi_\theta(y|x)}{\pi_{\rm ref}(y|x)}}_{\text{目标系数 } c^*(y)}
\,\nabla_\theta\log\pi_\theta(y|x)
\Big]
$$

中间会出现一个 $+1$，但它在期望下由于 score function 零均值恒等式消掉了。

**结论非常关键：**
Reverse KL 的"正确梯度系数"是

$$
c^*(y) = \log\frac{\pi_\theta}{\pi_{\rm ref}}
$$

你可以把它当作对 log-prob 的"线性恢复力"：

- 若 $\pi_\theta(y)$ 比 $\pi_{\rm ref}(y)$ 大（log-ratio 正），梯度会倾向于压低该动作概率
- 若 $\pi_\theta(y)$ 比 $\pi_{\rm ref}(y)$ 小（log-ratio 负），梯度会倾向于抬高该动作概率

并且强度随 log-ratio 线性增长（不是饱和，也不是爆炸式加速）。

---

## 5. 核心定理：$k_1$ in reward $\Leftrightarrow$ $k_2$ as loss（on-policy 下梯度等价）

他们证明：

- **传统 PPO 用的 $k_1$ in reward**（把 log-ratio 当 detached 系数乘 score function）
- **平方 log-ratio 的 $k_2$ as loss**：

$$
k_2 = \frac{1}{2}\left(\log\frac{\pi_\theta}{\pi_{\rm ref}}\right)^2
$$

这两个在 on-policy 采样时，给出的期望梯度都等价于真正的 RKL 梯度（即系数 $c^*(y) = \log\frac{\pi}{\pi_{\rm ref}}$）。

直觉原因很简单但很重要：

$$
\nabla_\theta \frac{1}{2} t^2 = t\,\nabla_\theta t, \quad t = \log\frac{\pi_\theta}{\pi_{\rm ref}}
$$

而 $\nabla_\theta t = \nabla_\theta \log\pi_\theta$。

所以它产生的梯度就是

$$
t\,\nabla_\theta\log\pi_\theta
$$

恰好就是正确系数 $c^*(y)$ 乘 score function。

### 这意味着什么（工程意义很大）？

如果你不喜欢"把 KL 塞进 advantage 再 detach"的写法（in reward），你也可以写成一个"干净的" penalty loss：平方 log-ratio（$k_2$），它在 on-policy 语境下同样是**理论上正确的 RKL 梯度实现**。

---

## 6. 反例：为什么"$k_1$ as loss"会彻底失效？

这是全文最"打脸 value-estimation 视角"的点。

若你把

$$
k_1 = \log\pi_\theta - \log\pi_{\rm ref}
$$

直接当 loss 做反传（as loss），在 on-policy + detached sampling 的实现里，梯度变成：

$$
\nabla_\theta \mathbb{E}_{y \sim \pi_\theta(\text{detached})}[\log\pi_\theta(y|x) - \log\pi_{\rm ref}(y|x)]
= \mathbb{E}[\nabla_\theta \log\pi_\theta(y|x)]
$$

但由于 score function 的零均值恒等式：

$$
\mathbb{E}_{y \sim \pi_\theta}[\nabla_\theta \log\pi_\theta(y|x)] = 0
$$

所以它的**期望梯度恒为 0**。

### 解释一下"为什么会这样"

- 真正的 KL 目标在数学上是"分布 $\pi$ 的函数"，梯度要同时考虑"密度变了导致期望权重变化"的那部分
- 但你用 sampled batch + 直接反传的方式，实际上在做的是 $\mathbb{E}[\nabla(\cdot)]$ 而不是 $\nabla\mathbb{E}[\cdot]$，并且由于你没显式引入 score-function 权重项，就把最关键那部分丢了
- 结果只剩一个"无信息、零均值"的梯度噪声项

所以它不仅没约束，还会**增加梯度方差**（文中也提到：这等价于把 baseline 技巧反过来用）。

---

## 7. GRPO 用的 $k_3$ as loss：它到底在优化什么？

$k_3$（常写成）：

$$
k_3 = \delta - 1 - \log\delta, \quad \delta = \frac{\pi_{\rm ref}}{\pi_\theta}
$$

他们算出：在 on-policy 语境下，$k_3$ as loss 的梯度等价于一个"in reward 的系数"：

$$
c_{k3}(y) = 1 - \delta = 1 - \frac{\pi_{\rm ref}}{\pi_\theta}
$$

而真正的 RKL 系数是：

$$
c^*(y) = -\log\delta = \log\frac{\pi_\theta}{\pi_{\rm ref}}
$$

### 7.1 为什么说它是一阶近似？

在 $\delta \approx 1$ 附近：

$$
-\log\delta \approx 1 - \delta
$$

所以 $k_3$ 的系数就是把 $-\log\delta$ 在 $\delta = 1$ 处做泰勒展开取一阶。

> 只在"策略很接近 reference"时近似成立。

### 7.2 三个具体问题

#### (1) 系数偏差（bias）

除非 $\delta = 1$，否则 $1 - \delta \neq -\log\delta$，更新方向相对真正 RKL 梯度是偏的。

#### (2) 尾部行为病态（asymmetry）

把 $\delta = \pi_{\rm ref}/\pi$ 代进去看两端：

- **Over-coverage：** $\delta \to 0$（即 $\pi \gg \pi_{\rm ref}$）
  - 真正系数：$-\log\delta \to +\infty$（持续增强的拉回力）
  - $k_3$ 系数：$1 - \delta \to 1$（饱和）
  - → 这会导致后期"策略大量抬高某些 token 概率"时，$k_3$ 的 KL 约束明显偏弱，更容易 drift。

- **Under-coverage：** $\delta \to \infty$（即 $\pi \ll \pi_{\rm ref}$）
  - 真正系数：$-\log\delta$ 只对数级变大（温和）
  - $k_3$ 系数：$1 - \delta \to -\infty$（线性爆炸）
  - → 一旦某些动作概率被压得很低，$k_3$ 会产生极端大梯度，带来不稳定甚至数值炸裂风险。

#### (3) 方差问题（statistical instability）

在 on-policy 下 $\mathbb{E}[\delta] = 1$，因此 $\mathbb{E}[1 - \delta] = 0$；但它的方差变成了与卡方散度相关：

$$
\mathrm{Var}[1 - \delta] = \chi^2(\pi_{\rm ref} \| \pi_\theta)
$$

卡方散度在 tail mismatch 时非常不稳定，意味着你得到的 KL 梯度会更"抖"。

> 这点非常反直觉：
> 有人用 $k_3$ 的理由是"它作为 KL 值估计器方差小"，但作为**梯度系数**时却可能引入更糟的方差结构。

---

## 8. Off-policy（尤其 PPO 多 epoch）时，"as loss"实现为什么会有系统性偏差？

这是工程上最容易踩的大坑之一：**很多实现只给 reward head 做 PPO ratio/clip，却把 KL loss 当普通监督 loss 加上去，没有 IS/clip。**

但如果你是在 PPO 的典型设置里：

- 用 $\pi_{\theta_k}$ rollout 采样（行为策略）
- 用 $\pi_\theta$ 训练多轮（目标策略在变）

那么 KL 项的期望本来应该是 under $\pi_\theta$，你却在用来自 $\pi_{\theta_k}$ 的样本去估它。正确做法需要重要性采样（IS）：

$$
\rho_k(\theta) = \frac{\pi_\theta(y|x)}{\pi_{\theta_k}(y|x)}
$$

因此 KL head（无论你把它写成 reward coefficient 还是 loss）在 off-policy 更新时都应该带上 $\rho$ 并进入 PPO 的 clip 机制，否则会产生系统偏差。

### 报告给的"原则性修复"：把 as-loss 先转换为 in-reward 形式

他们给了一个非常干净的通用公式：

$$
k_n'(x,y) = \frac{\partial k_n}{\partial \log\pi_\theta}\bigg|_{\pi = \pi_{\theta}(\text{snapshot})}
$$

也就是说：

- 你用 $k_n$ as loss 时，它诱导的系数其实是 $k_n'$
- 先把它换成一个 detached 系数 head（in reward head），再像 PPO 一样对这个 head 做 ratio + clip

例如：

- $k_2 = \frac{1}{2}\left(\log\frac{\pi}{\pi_{\rm ref}}\right)^2 \Rightarrow k_2' = \log\frac{\pi}{\pi_{\rm ref}}$（回到正确系数）
- $k_3 = \delta - 1 - \log\delta \Rightarrow k_3' = 1 - \delta$

然后你就可以把 KL head 和 reward head：

- 合并成一个 head（combined）一起 clip
- 或分成两个 head（decoupled）分别 clip（甚至可以用不同 clip 超参）

---

## 9. 实验结果怎么解读？

### 9.1 $k_1$ as loss ≈ 没 KL

因为它期望梯度为 0，本质上不提供有效"拉回 reference"的信号，只加噪声。
所以曲线几乎跟"RL w/o KL"重合，这正是理论预测。

### 9.2 $k_2$ as loss 更稳定、更强约束

因为它实现的就是 RKL 的正确系数 $c^* = \log\frac{\pi}{\pi_{\rm ref}}$，并且在两侧都不会出现 $k_3$ 那种"一个方向饱和、另一个方向爆炸"的不对称。
因此更稳、更贴 ref。

但约束强也会带来典型 trade-off：
**固定 $\beta$ 下**，更强的 KL 拉回会压制探索和奖励驱动的移动，导致性能可能更低——这不代表它"不正确"，而是超参/训练策略要配合（例如更小 $\beta$、adaptive $\beta$、或 ref reset 等）。

### 9.3 $k_3$ as loss 更"松"、可能早期更好但更容易漂与不稳

因为在 $\pi \gg \pi_{\rm ref}$（后期常见）时它的系数饱和在 1，拉回变弱 → 更容易 drift；
在 $\pi \ll \pi_{\rm ref}$ 时又可能产生非常大的负系数 → 容易造成局部不稳定。

这也解释了描述中的现象：
$k_3$ 可能在某些阶段 reward 更高，但 KL/logprob diff 更大，方差也更大。

---

## 10. 实操建议：工程 Checklist

### 10.1 首选：用"正确系数"的实现（$k_1$ in reward 或 $k_2$ as loss）

**推荐方案 A（最常见/最稳）：$k_1$ in reward（PPO 经典做法）**

- 计算 `log_ratio = logπ - logπ_ref`
- **detach log_ratio**（关键）
- `advantage_combined = advantage_reward - β * log_ratio_detached`
- 用 PPO ratio/clip 去优化 `advantage_combined * logπ`

**推荐方案 B（写法更像普通 loss）：$k_2$ as loss**

- `loss_kl = 0.5 * (log_ratio)²`
- 反传会自然产生 `log_ratio * ∇logπ`（等价正确系数）
- 但注意：一旦你在 PPO 多 epoch/off-policy 下用它，**也要对它做 IS/clip**（下一条）

### 10.2 只要你是 off-policy（PPO 多 epoch），KL head 也必须 IS/clip

很多代码库只对 reward head 用 ratio/clip，对 KL penalty 直接加 loss，这是不对的（报告说这是普遍 bug）。

更稳的做法是：

- 把 KL head 写成一个"in-reward 的 detached 系数 head"
- 然后像 reward head 一样进入 PPO $\min(\rho c,\, \mathrm{clip}(\rho) c)$

尤其是 GRPO / "去掉 critic 的 PPO-like 方法"，最容易犯这个错。

### 10.3 如果你坚持用 $k_3$（GRPO 风格），至少要知道它在干什么，并做防护

如果你用 $k_3$ 的动机是"更松一点、更像 trust region 的弱约束"，那你需要明确接受：

- 它是 RKL 梯度的一阶近似，有 bias
- 会出现尾部病态（饱和/爆炸）

工程防护建议（基于文中问题性质）：

- 对 $\delta = \pi_{\rm ref}/\pi$ 做 clip（防止极端比值）
- 或直接对系数 $1 - \delta$ 做 clamp（防止梯度爆）
- 并且最好仍然走 PPO ratio+clip 的方式去集成（off-policy 时必须）

### 10.4 关于 reward 的 normalization：别"顺手"把 KL 一起标准化导致 $\beta$ 失真

如果你采用 combined form，把 `reward - β KL` 合并后再做 batch/group normalization，会改变 KL 项在整体中的尺度占比（等价于动态改变 $\beta$）。
是否可接受取决于你目的：

- 追求严格语义：尽量别让 normalization 改变 $\beta$ 的含义（可只对 reward advantage 处理）
- 追求经验稳定：合并后再 normalize 可能也能用，但要知道它不再是你设定的那个固定 $\beta$

### 10.5 GRPO 的 group normalization：std 很小会放大噪声，需要 std floor/clip

文末 Appendix J 的建议很实用：
当同一 prompt 的 $G$ 个 reward 几乎一样时，group std 接近 0，会把极小差异放大成很大 advantage。
解决：对 std 加下限（std floor）或 clip 到区间。

---

## 11. 额外补充理解与可延伸点

### 11.1 统一框架的本质：把所有 KL 正则归约成"score function 的系数设计"

一旦你接受这个视角，你会意识到：

- "KL 作为值"其实不是你真正关心的东西
- 你真正关心的是 $c(y)$ 这个"拉回力"函数的形状：线性？饱和？有界？是否对称？尾部是否爆炸？

这也自然引出他们提到的 bounded 替代（如 MiniMax-01 / MSE regularizer）：
让系数变成 $\pi - \pi_{\rm ref}$ 这种有界量（$[-1,1]$），稳定性会更好，但约束语义变了（在概率空间而不是 log 空间）。

### 11.2 "$k_2$ 太强压性能"并不否定它正确

很多人看到"更强正则 → 性能差"就会下意识改正则形式，其实更合理的可能是：

- 用正确梯度（$k_1$/$k_2$）
- 调整 $\beta$（甚至 adaptive $\beta$）
- 或者用 ref reset / periodic update（如 Prorl 的思路）

---

## 12. 一句话总结：最重要的结论

- **别用 $k_1$ as loss**：期望梯度为 0，等于没 KL，还加噪声。
- **Reverse KL 的"正确梯度系数"是 $\log(\pi/\pi_{\rm ref})$**。
- **$k_1$ in reward 与 $k_2$ as loss 在 on-policy 下梯度等价且是 principled 的 RKL 实现**。
- **GRPO 用的 $k_3$ as loss 只是 $-\log\delta$ 的一阶近似，尾部不对称且可能高方差**。
- **只要 off-policy（PPO 多 epoch），KL 项也必须做 IS/clip；很多实现漏了，这是系统性偏差来源**。