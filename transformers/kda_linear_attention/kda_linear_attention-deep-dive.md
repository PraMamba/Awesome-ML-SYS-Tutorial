# KDA 线性注意力：从 Delta Rule 快权重到 Kimi Linear 生产内核

> **检索与源码状态截至：2026 年 8 月 30 日。**
> 本文中的 **KDA 指 Kimi Delta Attention**（Kimi Linear 的注意力算子），不是某种 Kernel Design Agent。
> 数学记号全文统一：状态矩阵 $S_t\in\mathbb{R}^{d_k\times d_v}$、输出 $o_t=S_t^\top q_t$（与 [Kimi Linear 报告](https://arxiv.org/abs/2510.26692) 的 Eq. 1 一致）；遗忘门（log 空间）$g_t\in\mathbb{R}^{HV\times K}$ 按 value-head 逐 key 维存储；$\beta_t\in\mathbb{R}^{HV}$ 按 value-head 存储。GDN 才是标量门 $g_t\in\mathbb{R}^{H}$。

第一次打开 Kimi Linear 官方 `modeling_kimi.py`，我产生了三个困惑，它们最终成了这篇文章的骨架：

1. KDA 明明被称为线性注意力，为什么训练时调用的是 `chunk_kda`，解码时却又切换成 `fused_recurrent_kda`？
2. 它不是固定大小的 RNN 状态吗，为什么缓存里同时还有卷积状态、循环状态，以及 MLA 层的 KV cache？
3. 论文只写了一条状态更新公式，为什么 SGLang 的实现却需要模型层、状态池、prefill/decode/verify 调度器和多套 GPU 内核？

官方的模型代码确实会在短序列或解码阶段走递归内核，而训练只允许 chunk 模式；缓存对象分别维护 KDA 的卷积状态、循环状态，以及全注意力层的 K/V 缓存。换句话说，**KDA 的"算法公式"只有三行，但一个可用的 Kimi Linear 系统至少包含：模型参数化、并行训练算法、推理状态管理和硬件内核四层。**

这篇文章的路线是四条主线（对应"概念 → 模型/算法 → 工程代码"的顺序）：

1. **概念**：为什么需要线性注意力、delta rule 为什么"先读再写"、遗忘门为什么要逐通道（§1–§3）；
2. **模型**：Kimi Linear 全貌与计算特征（§5–§6）；
3. **算法**：KDA 状态机的代数结构（受约束 DPLR）与 chunkwise 分解（§7–§8）；
4. **代码**：官方 checkpoint 参数化、FLA `fla.ops.kda` 内核走读、SGLang/vLLM/Megatron/verl 生产集成（§9–§12）。

致谢：这篇文章的大部分内容来自 Moonshot 公开展出的论文、checkpoint 与内核实现，以及 Moonshot 工程师在知乎写的并行计算解析（本地草稿 `draft-2` 存档）；数学推导的交叉验证参考了 Zhiyuan Li 的两篇博客。有把握的判断来自我对照本地资料的逐式核验，没把握的地方我都标了来源边界。

阅读本文只需要：Transformer 里的 $Q,K,V$；外积 $kv^\top$ 与矩阵状态；RNN 的逐步状态更新；以及"GPU 更擅长大矩阵乘法，不擅长很长的逐 token 循环"这一个直觉。所有公式都标注了出处（论文章节或源码文件 + 行号），所有代码引用都锁定了 commit。

## 一、动机：Softmax 注意力之外的另一条路

### 1.1 两个随上下文长度增长的量

因果 Softmax 注意力可以写成：

$$
o_t=\sum_{i\le t}\operatorname{softmax}_i\left(\frac{q_t^\top k_i}{\sqrt d}\right)v_i.
$$

它有两个随上下文长度 $T$ 增长的问题：

- **训练时**，显式或隐式处理 $T\times T$ 的 token 交互矩阵；
- **推理时**，自回归解码必须保留历史 token 的 K/V，缓存随 $T$ 线性增长。

FlashAttention 能显著减少中间张量的 HBM 读写，但它没有改变"当前 query 必须访问全部历史 K/V"这一语义，复杂度还是 $O(T^2)$ 的量级，KV cache 还是 $O(T)$。早在 2020 年，[Transformers are RNNs](https://arxiv.org/abs/2006.16236) 就把前向/反向的计算需求随序列长度的增长画成了一条对比曲线：Transformer 前向（蓝点）比 Linear（黑叉）上涨得更陡，两者在 2¹⁶ 长度处拉开近两个数量级：

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/tar-fig1-compute-scaling.jpg" alt="Transformers are RNNs 论文的 Fig. 1（前向计算需求 vs 序列长度，双对数坐标）：蓝色点线与红色虚线为 Reformer/LSH 与 Softmax 类方案，黑色叉线为 Linear Attention；随长度从 2^9 到 2^16，Linear 的斜率明显更缓（图例以论文原图为准）" style="width: 60%;">
</div>

> **图片来源**：Transformers are RNNs（arXiv 2006.16236）Figure 1（前后向 pass 的计算需求对比，双对数轴）。本地副本：`references/papers/transformers-are-rnns/transformers-are-rnns.md`。这张图说明的是**渐进复杂度**；而 KDA 面临的挑战恰恰是同样的图里不显示的东西，常数项与硬件利用率（§7.1）。

Kimi Linear 报告把长轨迹、工具调用和 RL test-time scaling 下的 KV cache 与解码成本列为核心工程动机，并在论文开头用一张散点图直接亮出结果（图 1）：这类"长上下文 + RL"的动机在我仓库里其实早有伏笔，[Kimi K1.5 partial rollout 笔记](../../rlhf/partial-rollout/readme.md)（published）记录过 Kimi 系模型的长上下文/部分 rollout 策略，但那篇讲的是"怎么用长上下文"，本文补上"注意力底层为什么变"这一环。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/kimi-fig1a-teaser.jpg" alt="Kimi Linear 与 MLA、GDN-H 在 RULER(128k) 与 MMLU-Pro(4k) 上的性能-解码加速散点图（图 1a）：RULER 上 MLA 81.3@1x、GDN-H 80.5@约 3.8x、Kimi Linear 84.3@3.98x；MMLU-Pro 上 MLA 47.2@1x、GDN-H 47.9@约 1.2x、Kimi Linear 51.0@3.98x" style="width: 60%;">
</div>

> **图片来源**：Kimi Linear 报告（arXiv 2510.26692）Figure 1(a)，§1 Introduction（图注原文：MMLU-Pro (4k context length)、RULER (128k context length)，Kimi Linear 取得 84.3/51.0 与 3.98× 加速）。本地副本：`references/papers/kimi-linear/kimi-linear.md` 第 15 行引用图。

图 1a 展示的是"性能和解码加速都要"的双目标；图 1b 则给出了随解码长度展开的具体代价曲线：

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/kimi-fig1b-tpot.jpg" alt="解码 TPOT 随解码长度变化的折线图（图 1b）：MLA 在 1M token 时约 11.4ms，GDN-H 与 Kimi Linear 曲线几乎重合、约 1.8ms，标注 4.8x(256K)、5.7x(512K)、6.3x(1M) 的相对 MLA 加速倍数" style="width: 60%;">
</div>

> **图片来源**：Kimi Linear 报告（arXiv 2510.26692）Figure 1(b)，§1 Introduction。本地副本：`references/papers/kimi-linear/kimi-linear.md` 第 18 行引用图。图中标注的 **6.3× 是 Kimi Linear 相对 MLA 在 1M 处的 TPOT 实测比值**（约 11.4ms vs 约 1.8ms），不是"理论提速上限"；论文 §6.3 另有一段"理论上限 6.3×"的表述，其推理链是"省下的 KV 内存转投更大 batch"，与图线是两个口径（§6.1 展开）。论文关于"1M 上下文解码 TPOT 提升约 6.3 倍"的表述即来自此图；**下文（§6.1）会专门区分"理论上的混合效率比"与"端到端实测"。**

Kimi Linear 的目标不是再优化一次 Softmax kernel，而是**改掉状态表示本身**：不保存全部 token，而是把历史压缩进一个固定大小的矩阵状态。

### 1.2 把历史压缩成矩阵：线性注意力的 RNN 视角

忽略特征映射和归一化项，最简单的因果线性注意力为：

$$
S_t=S_{t-1}+k_tv_t^\top,\qquad o_t=S_t^\top q_t,
\qquad S_t\in\mathbb{R}^{d_k\times d_v}.
$$

每个 $k_tv_t^\top$ 是一次**外积写入**（附录 C 会专门讲外积）。把所有历史展开：

$$
S_t=\sum_{i=1}^{t}k_iv_i^\top,\qquad
o_t=\sum_{i=1}^{t}(q_t^\top k_i)v_i.
$$

这正是 **Linear Transformer（[Transformers are RNNs](https://arxiv.org/abs/2006.16236)，ICML 2020）** 给出的视角：利用矩阵乘法结合律，把"每个 query 扫描全部历史"改写为"维护固定维度的状态"，这是线性复杂度的来源，也揭示了它与 RNN 的等价关系。

> **"线性"指的是相对于序列长度 $T$ 的增长方式，不代表每个 token 的计算量是常数级小数。**一次状态读写仍然是 $d_k\times d_v$ 的矩阵运算（本节先埋下这个伏笔，§7.1 会回来算这笔账）。

Kimi Linear 报告还给出一个更有数学味道的解读：线性注意力可以看作在"无界相关目标" $\mathcal{L}_t(S)=-\langle S^\top k_t,v_t\rangle$ 上对 $S$ 做**梯度上升**，最近写入的 key-value 对被不断强化，但**没有任何判定标准决定该擦除哪一段记忆**，语义上等价于"记忆无限膨胀"的 Hebbian 累加（[论文 §2.2](https://arxiv.org/abs/2510.26692)，Table 7 第一行）。

### 1.3 加法写入为什么会发生记忆干扰

假设连续出现两个相同 key：$k_1=k_2=k$，但它们对应不同 value：$v_1\neq v_2$。加法更新得到：

$$
S_2=kv_1^\top+kv_2^\top=k(v_1+v_2)^\top.
$$

再次用 $k$ 查询时，读到的更接近 $v_1+v_2$ 的混合，而不是"最新映射" $v_2$。这正是关联记忆的容量冲突：**它知道怎样增加一段关联，却不知道应当怎样替换旧关联。**

[Fast Weight Programmers 论文](https://arxiv.org/abs/2102.11174)（*Linear Transformers Are Secretly Fast Weight Programmers*）把线性注意力解释为一个慢网络不断"编程"的快权重矩阵，并用实验量化了纯加法写入在键值回忆任务上的损失，写入规则每升级一步，评估损失就显著下降一步：

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/fwp-fig2-associative-recall.jpg" alt="softmax memory 与多种线性注意力机制的最终评估损失对比（FWP Figure 2）：相同的键值对数量下，普通线性注意力（纯外积加法）的损失明显高于 delta rule 类机制" style="width: 60%;">
</div>

> **图片来源**：Fast Weight Programmers（arXiv 2102.11174）Figure 2。本地副本：`references/papers/linear-transformers-fast-weight-programmers/linear-transformers-fast-weight-programmers.md`。

顺带一提，这篇 2021 年的工作已经给出"用 delta rule 替换外积加法"的解（他们称之为 *Delta Rule*，直接继承自 Widrow-Hoff）：

$$
S_t=S_{t-1}+\beta_tk_t\left(v_t-S_{t-1}^\top k_t\right)^\top.
$$

当时的规模限制在合成任务上；把它真正规模化、并配齐可训练 chunk kernel，是 2024 年的 DeltaNet（下一节与 §8 的主角之一）完成的。

## 二、Delta Rule：先看看自己写错了多少

### 2.1 从重构目标推导更新

我们希望状态矩阵"记住"的是一条映射：$S^\top k_t\approx v_t$。据此定义**在线重构损失**：

$$
\mathcal{L}_t(S)=\frac12\left\|S^\top k_t-v_t\right\|_2^2.
$$

对矩阵 $S$ 求梯度（$S^\top k_t-v_t$ 是 $d_v$ 维列向量，$k_t$ 是 $d_k$ 维列向量）：

$$
\nabla_S\mathcal{L}_t=k_t\left(S^\top k_t-v_t\right)^\top.
$$

这个梯度是**外积**：在本文约定（$S\in\mathbb{R}^{d_k\times d_v}$）下，是**地址 $k_t$ 乘误差行向量 $e_t^\top$**，即 $\nabla_S\mathcal{L}_t=k_te_t^\top$，形状 $d_k\times d_v$。做一步学习率为 $\beta_t$ 的梯度下降：

$$
S_t=S_{t-1}-\beta_t\nabla_S\mathcal{L}_t(S_{t-1})
=S_{t-1}+\beta_tk_t\left(v_t-S_{t-1}^\top k_t\right)^\top.
$$

定义预测值与误差：$\hat v_t=S_{t-1}^\top k_t$、$e_t=v_t-\hat v_t$，于是：

$$
\boxed{\;S_t=S_{t-1}+\beta_tk_te_t^\top=(I-\beta_tk_tk_t^\top)S_{t-1}+\beta_tk_tv_t^\top\;}
$$

这就是 delta rule。**它不再盲目写入 $k_tv_t^\top$，而是只写入当前状态尚未解释的残差。** Kimi Linear 报告 §2.2 把这个推导完整写了一遍，注意它与"线性注意力 = 梯度上升"的区别：线性注意力是对"无约束的相关目标"做上升，而 delta rule 是对"带误差信号的重构目标"做下降，两者唯一的区别是目标函数里有没有当前记忆的预测值。

> 这里的每一步（损失 → 梯度 → 一步 SGD → 展开合并）与 [Kimi Linear 报告 §2.2](https://arxiv.org/abs/2510.26692) 逐式核对过；"记忆 = 一个推理时在线被训练的小模型"这一更深层的解读（测试时训练 / TTT 视角）放在**附录 A**。

### 2.2 为什么 delta rule 能"覆盖旧值"

假设 $\|k_t\|_2=1$、$\beta_t=1$，把更新式转置后对同一个 key 读出：

$$
\begin{aligned}
S_t^\top k_t
&=\left[S_{t-1}^\top\left(I-\beta_tk_tk_t^\top\right)+\beta_tv_tk_t^\top\right]k_t\\
&=S_{t-1}^\top k_t-\beta_tS_{t-1}^\top k_t\left(k_t^\top k_t\right)+\beta_tv_t\left(k_t^\top k_t\right)\\
&\overset{\beta_t=1,\ \|k_t\|_2=1}{=}v_t.
\end{aligned}
$$

也就是说，对当前精确 key 而言，旧映射被直接改写为新映射，**这正是附录 B 讲的正交投影 $\left(I-\beta_tk_tk_t^\top\right)$ 的几何含义：擦掉 $k$ 方向上的旧内容、保留垂直方向，再写入新值。**

从"学习曲线"看，这一步升级的效果非常直接（Fast Weight Programmers Figure 3：同一条合成任务上，delta rule 的收敛远快于纯 Hebbian 加法）：

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/fwp-fig3-update-rules.jpg" alt="不同更新规则的学习曲线（FWP Figure 3）：序列长 40、20 组唯一键值对，delta rule 类规则显著优于纯加法外积" style="width: 60%;">
</div>

> **图片来源**：Fast Weight Programmers Figure 3（§6.1.2，SETTING 2: COMPARING UPDATE RULES）。本地副本：`references/papers/linear-transformers-fast-weight-programmers/linear-transformers-fast-weight-programmers.md` 第 351 行引用图。

但 DeltaNet 仍有一个问题：**它会修改被新 key 触达的旧关联，却没有一个独立机制主动淘汰"已经过时、但暂时没被同类 key 覆盖"的记忆。** 一段记忆一旦写入，只要没有同 key 的后续写入，它就永远留在状态里。这引导出遗忘门，以及本文真正的主题：遗忘门应该以什么粒度存在。

## 三、从标量遗忘到逐通道遗忘

### 3.1 相关论文的图注（图 2：DeltaNet 的块设计）

在展开门控之前，先看一眼两条技术谱系的块级设计。DeltaNet 论文（arXiv 2406.06484）给出了它的标准块：

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/deltanet-fig2-architecture.jpg" alt="DeltaNet 神经网络架构示意图（DeltaNet Figure 2）：左侧为 N 层堆叠的 DeltaNet 块（DeltaNet + RMSNorm + SwiGLU + RMSNorm + Linear），右侧为块内细节：q/k 经 Linear→Conv→L2Norm，v 经 Linear→Conv，标量 β 由 Linear 生成，共同进入 Delta Rule 模块，再经 RMSNorm 与 Linear 输出" style="width: 70%;">
</div>

> **图片来源**：DeltaNet（arXiv 2406.06484）Figure 2，§3.3（DeltaNet Transformer）。本地副本：`references/papers/delta-rule/delta-rule.md`。

注意右侧的块设计：**q/k 有 L2Norm + ShortConv，v 有 ShortConv，β 是标量（每 head 一个），没有遗忘门。** KDA 的块（§9.1 的官方数据流）会在这个骨架上做三处手术：把 L2Norm 下沉到 kernel 内、把 gate 从标量升级为逐通道向量、并在输出端加 sigmoid 输出门与 RMSNorm。

### 3.2 GDN：一个 head 一个遗忘速度

[Gated DeltaNet](https://arxiv.org/abs/2412.06464)（GDN，arXiv 2412.06464）在 delta update 外增加**标量衰减**：

$$
S_t=\alpha_t\left(I-\beta_tk_tk_t^\top\right)S_{t-1}+\beta_tk_tv_t^\top,
\qquad \alpha_t\in[0,1].
$$

它对**旧状态项**施加标量衰减（$\alpha_t$ 是标量，与投影 $(I-\beta_tk_tk_t^\top)$ 可交换，因此"先衰减还是先投影"不影响旧状态项；**新写入项 $\beta_tk_tv_t^\top$ 不会被 $\alpha_t$ 缩放**）。论文把 $\alpha_t$ 解读为 fast-weights 上的 data-dependent $L_2$ 正则（[Kimi Linear §2.2](https://arxiv.org/abs/2510.26692)）。GDN 的块设计长这样：

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/gdn-fig1-architecture.jpg" alt="Gated DeltaNet 混合架构与块设计（GDN Figure 1）：左为中 Gated DeltaNet-H1 堆叠（Gated DeltaNet + SWA + MLP），中为 H2（Mamba2 + Gated DeltaNet + SWA + MLP），右为块内细节：q/k 经 Linear→Conv→L2，v 经 Linear→Conv，标量 α（记作 alpha）与 β 由两组 Linear 生成，进入 Gated Delta Rule 块后经 Norm 与 Linear 输出" style="width: 90%;">
</div>

> **图片来源**：Gated Delta Networks（arXiv 2412.06464）Figure 1。本地副本：`references/papers/gated-delta-networks/gated-delta-networks.md`。

问题在于：一个 head 中可能同时存在需要快速遗忘的局部格式、需要中期保留的实体信息、需要长时间保留的任务状态。**单个 $\alpha_t$ 迫使这些方向共享同一记忆寿命。**

### 3.3 KDA：每个 key channel 自己决定遗忘速度

KDA 把标量衰减扩展为向量：$\alpha_t\in[0,1]^{d_k}$，并定义 $D_t=\operatorname{Diag}(\alpha_t)$。核心状态更新（[Kimi Linear 报告 Eq. 1](https://arxiv.org/abs/2510.26692)）为：

$$
\boxed{\;
S_t=\underbrace{\left(I-\beta_tk_tk_t^\top\right)}_{\text{delta rule 投影}}
\ \underbrace{D_tS_{t-1}}_{\text{逐通道衰减}}
+\underbrace{\beta_tk_tv_t^\top}_{\text{写入新值}}
\;}
\qquad
o_t=S_t^\top q_t.
$$

把它改写成更容易看懂的**四步**：

$$
\bar S_t=D_tS_{t-1},\qquad
\hat v_t=\bar S_t^\top k_t,\qquad
e_t=v_t-\hat v_t,\qquad
S_t=\bar S_t+\beta_tk_te_t^\top.
$$

即：**先逐通道遗忘，再用当前 key 读旧映射，最后只写预测误差。** 注意 $\hat v_t$ 用的是衰减后的状态 $\bar S_t$ 而不是 $S_{t-1}$，这也是"KDA 与 GDN 的差别不只是多一个 gate"的支点：**GDN 的标量 $\alpha_t$ 只作用在旧状态项上（且与投影 $(I-\beta_tk_tk_t^\top)$ 可交换，"先衰减还是先投影"没有可观察差异），当前写入 $\beta_tk_tv_t^\top$ 同样不被 $\alpha_t$ 缩放**；而 **KDA 的逐通道 $D=\operatorname{Diag}(\alpha_t)$ 一般不与投影可交换**，必须先形成 $D_tS_{t-1}$、再在衰减后的状态上做校正（即 $D$ 在内、$S$ 在外）。

#### 最容易写错的地方

KDA 是：

$$
\left(I-\beta_tk_tk_t^\top\right)D_tS_{t-1}+\beta_tk_tv_t^\top,
$$

**不是**：

$$
D_t\left[\left(I-\beta_tk_tk_t^\top\right)S_{t-1}+\beta_tk_tv_t^\top\right].
$$

前者是"先遗忘、再以未衰减的目标 $v$ 修正"；后者还会额外衰减当前写入。两者不是同一个状态机，这正是 KDA 的 DPLR 对应式：$(\mathbf D-\boldsymbol a_t\boldsymbol b_t^\top)\mathbf S_{t-1}+\beta_t\boldsymbol k_t\boldsymbol v_t^\top$，其中 $\mathbf D=\operatorname{Diag}(\alpha_t)$、$\boldsymbol a_t=\beta_t\boldsymbol k_t$、$\boldsymbol b_t=\boldsymbol k_t\odot\boldsymbol\alpha_t$（写入项 $\beta_t\boldsymbol k_t\boldsymbol v_t^\top=\boldsymbol a_t\boldsymbol v_t^\top$，与 Eq. 1 一致）。**需要注意**：[论文 §6.2](https://arxiv.org/abs/2510.26692) 同一小节给出的通用 DPLR 对应式写作 $+\boldsymbol k_t\boldsymbol v_t^\top$，按 $\boldsymbol a_t=\beta_t\boldsymbol k_t$ 读会少一个 $\beta_t$（论文 §3.2 的重写式与 Eq. 1 都带 $\beta_t$），这是论文该行的省略/内部不一致；本文一律采用与 Eq. 1 一致的形式（以下简述该节对应式时写作 $+\boldsymbol a_t\boldsymbol v_t^\top$）。**衰减作用在 $S_{t-1}$ 上，修正作用在其后。** 这一点在 §7.2 推导 chunkwise 分解时会再次出现并决定矩阵结构的形状。

### 3.4 一个可计算的 2×2 例子

设 $S_0=I$，$\alpha=[0.8,\,0.2]^\top$（两个 channel 的寿命相差很大），$k=[1,\,0]^\top$，$v=[0,\,1]^\top$，$\beta=0.5$。

先衰减：

$$
\bar S=\operatorname{Diag}(0.8,0.2)S_0=\begin{bmatrix}0.8&0\\0&0.2\end{bmatrix}.
$$

当前 key 读到（$k$ 精确命中第一个 channel）：

$$
\hat v=\bar S^\top k=\begin{bmatrix}0.8\\0\end{bmatrix},\qquad
e=v-\hat v=\begin{bmatrix}-0.8\\1\end{bmatrix}.
$$

delta 写入：

$$
\beta ke^\top=\begin{bmatrix}-0.4&0.5\\0&0\end{bmatrix},\qquad
S_1=\begin{bmatrix}0.4&0.5\\0&0.2\end{bmatrix}.
$$

用 $q=[1,\,1]^\top$ 查询：$o=S_1^\top q=[0.4,\,0.7]^\top$，第一个 channel 被修改为"读新映射"，第二个 channel 保持了它自己的衰减速度（0.2 而不是 0.8）。

如果像 GDN 一样使用统一标量 $\alpha=0.8$，同样的计算给出 $o_{\text{GDN}}=[0.4,\,1.3]^\top$，与 key 无关的第二个 channel 被迫共享 0.8 的寿命。这个例子并不证明 KDA 一定更好（它只是多了一个自由度，这个自由度用不用得好看训练），但它把"逐通道"的含义显式化了：**与当前 key 无关的记忆方向，可以拥有和第一个方向完全不同的寿命。**
## 四、模型演进：每一代具体增加了什么

把前三章压缩成一张表，可以看到这条谱系连续回答的四个问题（**注意表中 KDA 行的 gate 记作 $\operatorname{Diag}(\alpha_t)\in\mathbb{R}^{d_k}$，是逐 key 维的；GDN 的 $\alpha_t$ 是标量**）：

| 方法 | 状态更新核心 | 新增能力 | 仍未解决的问题 |
| --- | --- | --- | --- |
| 线性注意力 | $S\leftarrow S+kv^\top$ | 固定状态、线性序列复杂度 | 只能叠加，关联容易冲突 |
| Delta Rule | $S\leftarrow S+\beta k(v-S^\top k)^\top$ | 可以修正或覆盖旧映射 | 没有主动全局遗忘 |
| DeltaNet | 同上 + Householder/WY chunk 算法 | 现代 GPU 上可训练 | 旧关联可能长期残留 |
| GLA | $S\leftarrow\operatorname{Diag}(\alpha_t)S+kv^\top$ | channel-wise 遗忘（但没有 error correction） | 写入仍是纯加法 |
| Gated DeltaNet | $S\leftarrow\alpha_t(I-\beta_tk_tk_t^\top)S_{t-1}+\beta_tk_tv_t^\top$ | 遗忘与定向修改结合 | 一个 head 只能共享一个寿命 |
| KDA | $S\leftarrow(I-\beta_tk_tk_t^\top)\operatorname{Diag}(\alpha_t)S_{t-1}+\beta_tk_tv_t^\top$ | 每个 key channel 独立遗忘和修正 | 固定状态仍有有限容量，需混合全注意力 |

这条链的四个关键转折是：**历史怎么保存**（矩阵状态）→ **错误关联怎么改**（delta rule）→ **过期信息怎么删**（forget gate）→ **不同记忆方向要不要同时删**（不要，因此用 channel-wise gate）。其中"逐通道遗忘但不加 error correction"的中间体是 [GLA](https://arxiv.org/abs/2312.06635)（channel-wise gating + I/O-aware 训练的那一支），KDA 可以理解成"GLA 的门控 × delta rule 的修正"，但它并不是把两个公式机械相加：**channel-wise diagonal decay 会改变整个 chunkwise transition 的代数结构，必须重新设计并行算法。** 这正是本文下一章要解决的事。


## 五、Kimi Linear 模型全貌：配置、架构与为什么要混合 MLA


### 5.1 架构全貌：KDA、MLA 与 MoE 的堆叠

[论文 Figure 3](https://arxiv.org/abs/2510.26692)（§4）给出了完整架构图：

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/kimi-fig3-architecture.jpg" alt="Kimi Linear 模型架构图（Figure 3）：左侧为两种残差块，N 层的 KDA 块（Norm→KDA→Norm→MoE）与 1 层的 MLA 块（Norm→MLA→Norm→MoE），N 设为 3；右上为 MoE 通道细节（1 个 Shared Expert + Nr 个 Routed Expert，Router 输出 topk 加权）；右下为 KDA 块细节：输入经 Linear 投影后，q/k 走 ShortConv→L2 归一化，v 走 ShortConv，另有两组低秩投影（梯形）生成门控信号，进入 Kimi Delta Attention 模块，输出经 Norm 与 Linear，带残差连接" style="width: 90%;">
</div>

> **图片来源**：Kimi Linear 报告 Figure 3，§4。本地副本：`references/papers/kimi-linear/kimi-linear.md` 第 179 行引用图。

左半是宏观堆叠：$N=3$ 层 KDA 块配 1 层 MLA 块（论文明确说 N 设为 3，是实现里的经验选择）；右上 MoE 细节；**右下的 KDA 块细节才是本文 §9.1 单层数据流的图像版**：Linear → Conv → L2（q/k）、Linear → Conv（v）、低秩梯形（门控），四路汇入 Kimi Delta Attention → Norm → Linear。与 DeltaNet/GDN 的块设计（§3.1/§3.2 两张图）对比可以直观看到三代块设计的差异：GDN 的 $\alpha$ 是标量（一个 Linear 输出），KDA 的忘门是低秩投影 `f_a/f_b` 输出的 `[HV,K]` 通道向量，并且输出端多了 sigmoid 门 + RMSNorm。

为什么纯 KDA 还不够？**固定状态把任意长度历史压缩进 $S\in\mathbb{R}^{H\times d_k\times d_v}$，这是效率的来源，也是信息瓶颈。** 对于精确检索（从大量近似文本中找回一个 token、多个相似 key 对应不同 value、复制长序列、保持大量独立实体），有限状态尤其困难。论文 §4 的说法很直接：**"长上下文检索仍是纯线性注意力的主要瓶颈"，因此做 KDA 与全局 MLA 的 layerwise hybrid**；并强调选 layerwise（整层交替）而非 headwise（层内混合 head）是出于基础设施简洁性与训练稳定性，**而 3:1 是质量-吞吐的经验折中**，论文 §5.2 的直接消融给出：3:1（3 个 KDA 层配 1 个 MLA 层）训练/验证损失最低；更高的比值（7:1）训练损失相当但验证显著变差（说明线性层的信息瓶颈真实存在）；更低的比值（1:1）验证损失接近但推理开销更大；纯全注意力（0:1）则明显更差。所以：

> **KDA 负责把大部分 token mixing 变成固定状态计算，MLA 负责为无法可靠压缩的信息保留稀疏但精确的逃生通道。** 周期性全注意力层不是装饰，而是架构闭环的一部分，图 1a 里 Kimi Linear 在 RULER(128k) 上反超 MLA 3.0 分（RULER 84.3 vs 81.3），这一结果与论文 §4/§6 的混合设计解释一致（"KDA 承担大部分 token mixing、MLA 保留精确访问能力"）；注意论文给出的是**设计解释而非因果隔离消融**，本文不作进一步归因。

混合是否划算，最终是"质量-吞吐"的帕累托问题。论文的缩放律（Figure 5）显示：在相同训练量下 Kimi Linear 的曲线整体位于 MLA **之下**（纵轴是 loss，越低越好），即同等算力下混合架构的损失更低（论文表述为约 1.16× 计算效率优势）；两条拟合曲线几乎平行（指数 −0.0536 vs −0.0527），效率差主要来自常数项：

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/kimi-fig5-scaling-law.jpg" alt="MLA 与 Kimi Linear 的拟合缩放律曲线（Kimi Linear Figure 5）：横轴为 PFLOP/s-days（对数轴）、纵轴为损失，Kimi Linear（红）曲线整体低于 MLA（蓝），标注约 1.16x 计算效率优势" style="width: 60%;">
</div>

> **图片来源**：Kimi Linear 报告 Figure 5，§5.3（Fitted scaling law curves for MLA and Kimi Linear）。本地副本：`references/papers/kimi-linear/kimi-linear.md` 第 250 行引用图。论文原文："Kimi Linear achieves ∼1.16× computational efficiency compared to the MLA baselines with compute optimal training"，**读法是"同等 FLOPs 下损失更低"（曲线在下方），不是"同损失下省 1.16× FLOPs"的数字游戏**；拟合指数（−0.0536 vs −0.0527）说明两者缩放斜率接近，效率差主要来自常数项。


### 5.2 NoPE：位置编码去哪了

论文对 MLA 层**不用任何位置编码（NoPE）**，位置信息完全交给 KDA。理由在论文 §6.1：gated delta recurrence 的转移矩阵乘积是**数据相关的乘法式位置机制**，历史 token 对当前输出的影响含其写入时刻到当前时刻的状态转移矩阵连乘，时间间隔与中间内容都会改变权重；KDA 又把不同 channel 提供不同 decay trajectory（类似 RoPE 的不同旋转频率），因此可以替代一部分位置感知。

这个判断的定量证据是合成任务（论文 Figure 4）：

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/kimi-fig4-palindrome.jpg" alt="Palindromic 任务训练曲线（Kimi Linear Figure 4a 一组）：KDA 约 7.5K 步即收敛到接近 100% 精度，GDN 到约 15K 步才追上，Mamba2 全程低于 20%" style="width: 60%;">
</div>

> **图片来源**：Kimi Linear 报告 Figure 4 (a)（Palindrome 子图），图位于论文 §4 末尾（合成任务结果，被 §5.1 引用）。同一组还有 (b) MQAR、(c) Stack。本地副本：`references/papers/kimi-linear/kimi-linear.md`。

但我必须给这个论证画两条边界线（这些都是论文自己的措辞边界）：

- **论文给出的是"可学习的乘法式位置编码"等价解释**，它有启发性，但不是"任意模型都可以删掉 RoPE"的证明；
- NoPE 的可行性依赖 KDA/MLA 比例、训练长度、数据和模型规模，且**周期性 MLA 本身仍参与全局信息传递**。论文同时指出 NoPE 有两个工程红利：MLA 推理时可转成纯 MQA、长上下文训练免去 RoPE 频率基调整（如 YaRN）。 论文还做了 NoPE vs RoPE 的对照（Table 5 与 §5.5）："Kimi Linear 在长上下文评估上持续领先，而 Kimi Linear (RoPE) 在短上下文任务上分数接近"，论文的解释是**位置偏置的层级分布**：RoPE 版本把强显式相对位置信号集中在全局注意力层，线性层只提供弱的隐式位置归纳偏置，这种不匹配让全局层过度强调短程顺序、削弱了向长上下文调整的灵活性；NoPE 版本则把位置偏置更均衡地分配到跨层。这也再次说明 NoPE 不是"删掉一个东西"，而是"换了一种位置信息的分配方式"。

> **误区预警（对应 §14 的误区 9）**：NoPE 是"混合架构 + 训练 recipe"的选择结论，不是从 KDA 公式自动推出的免费附赠。


### 5.3 官方 checkpoint 的关键配置

官方 [Kimi-Linear-48B-A3B-Instruct](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct)（本地锁定副本：`references/checkpoints/kimi-linear-48b-official/{config.json, modeling_kimi.py}`，2026-08-30 拉取）的配置如下：

| 配置 | 数值 |
| --- | ---: |
| hidden size | 2304 |
| Transformer 层数 | 27 |
| KDA `num_heads` / `head_dim` | 32 / 128 |
| ShortConv kernel size | 4 |
| KDA 层数 | 20 |
| MLA 层数 | 7 |
| 全注意力层位置 | 4, 8, 12, 16, 20, 24, 27 |
| 最大上下文 | 1,048,576 |
| MoE experts / routed per token / shared | 256 / 8 / 1 |
| 全局 `head_dim`（属 MLA） | 72 |

> **容易踩的坑：** `config.json` 里全局的 `head_dim=72` 属于 **MLA** 层；KDA 层的 head dim 在 `linear_attn_config.head_dim=128`。两个 128×128 的 KDA 状态矩阵是 $K=V=128$，而 MLA 的 KV 是低秩（`kv_lora_rank=512`）+ NoPE/MQA 压缩后的形态，两者不要混算。

因此不是机械 3:1 重复到结尾：按 `full_attn_layers = [4, 8, 12, 16, 20, 24, 27]` 排布，前 24 层是六个完整的"KDA KDA KDA MLA"组，**尾段（第 25–27 层）是"KDA KDA MLA"（该组只有 2 个 KDA 层）**，合计 20 个 KDA 层 + 7 个 MLA 层。这组配置还说明一个常被忽略的事实：**Kimi Linear 的最终效果来自 KDA、MLA、MoE、训练数据和 recipe 的共同作用，不能把完整 checkpoint 的性能差异全部归因于一条 KDA 公式。**


## 六、KDA 的计算特征与效率口径


### 6.1 三个数字特征与效率口径

抛开架构设计，从计算模式看 KDA 有三个数字特征（[论文 §3.2、§6.3](https://arxiv.org/abs/2510.26692)）：

1. **训练 FLOPs**：$\operatorname{FLOPs}_{\text{KDA}}=6Td_h^2+3TCd_h+TC^2$（Eq. 13，$C=64$）。对比全注意力 $2T^2d_h$（Eq. 14），KDA 对 $T$ 线性；但 **$Td_h^2$ 项意味着 head_dim 每翻倍，状态更新的直接成本翻四倍**。
2. **decode 每 token**：$O(d_kd_v)$，与上下文长度无关，但若上下文很短、并发序列很多、state 用 FP32、GPU 带宽不足，固定状态的读写也可能成为显著开销。
3. **kernel 耗时**：论文的算子基准（Figure 2，batch=1、16 heads）比较的是 **KDA kernel 与一般 DPLR kernel**，两者都随输入长度上升，但 KDA 在 64K 处约为 DPLR 的一半（约 30ms vs 60ms），这正是 §7.3 说的"算子效率提升约 100%"：

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/kimi-fig2-kernel-time.jpg" alt="KDA 与一般 DPLR 的 kernel 执行时间对比（Kimi Linear Figure 2）：横轴输入长度 2K 至 64K，纵轴执行时间（ms），DPLR 曲线在 64K 处约 60ms，KDA 约 30ms，即约 2x 差距" style="width: 60%;">
</div>

> **图片来源**：Kimi Linear 报告 Figure 2，§3.2（Execution time of kernels，batch size 1、16 heads）。本地副本：`references/papers/kimi-linear/kimi-linear.md` 第 164 行引用图。

**这四个数字全部是"论文在特定模型、特定配置下报告的"，不能推广成"KDA 让一切快 6 倍"。** 论文自己的效率口径是（§5.6 / §6.3）：解码进入 I/O 受限区后，混合模型趋近于"理论混合效率比 3:1"（论文 §6.3 只给出该上限表述，未给可逐项验证的成本分解，本文不另行拆解）。**这里有两处论文内部口径差异值得记录**：(1) §5.6 正文写"decoding at 1M，Kimi Linear 比全注意力快 6×"（指向 Figure 1b，其标注为 6.3×，一致）；(2) §6.3 正文写"Figure 7b 中 Kimi Linear 在 1M 上下文实现 2.3× 提速"，但 **Figure 7b 图面标注是 1.8×（512K）/ 2.2×（1M）**，论文文字与图面数据不一致，本文以**图面标注为准**（2.2×@1M），引 §6.3 时只取"3:1 理论混合效率比"这个结构性结论。两套图表的数据语境也不同：Figure 1b 是论文开头 teaser 的 decode TPOT（256K/512K/1M 标注 4.8×/5.7×/6.3×，Kimi Linear 1M 约 1.8ms），Figure 7b 是 §5.6 同配置 48B 模型的 decode TPOT（1M 约 8ms vs MLA 17ms）：

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/kimi-fig7a-prefill.jpg" alt="prefill 延迟对比（Kimi Linear Figure 7a）：横轴 prefill 长度 4K 至 1M，纵轴延迟（秒）；MLA 在 1M 处约 65s，Kimi Linear 约 23s（与 GDN-H 曲线几乎重合），标注 2.3x（512K）与 2.9x（1M）" style="width: 55%;">
    <img src="./pics/kimi-fig7b-decode.jpg" alt="decode 阶段 TPOT 对比（Kimi Linear Figure 7b）：横轴解码长度，纵轴 TPOT（ms）；MLA 在 1M 处约 17ms，Kimi Linear 约 8ms，标注 1.8x（512K）与 2.2x（1M）" style="width: 55%;">
</div>

> **图片来源**：Kimi Linear 报告 §5.6（Efficiency Comparison，Figure 7 a/b 两联）。本地副本：`references/papers/kimi-linear/kimi-linear.md` 第 343/346 行引用图。论文的两条效率结论在本文中分开陈述：**"75% KV cache 减少"** 来自 3:1 混合缓存结构本身（20/27 层为固定状态线性注意力）；**"6.3× 理论解码提速"** 是"KV 内存释放 → 可以容纳更大 batch → 总体吞吐提高"的推论链（论文 §6.3）；两者不是同一条因果链，**不能只读倍数、不看前提**。


## 七、为什么一个"线性"RNN 还需要 Chunkwise 算法


> **这是本文的驱动问题。** 前面几章已经建立了完整的推导链：KDA 的状态更新只有三行，总运算量对 $T$ 是线性的，状态大小恒定。读者自然会问：既然是线性的，为什么不能直接逐 token 递推？为什么论文要花一个大章节推导 chunkwise 分解、FLA 还要写上一千多行 Triton？这个问题的答案一半在 GPU 的执行模型里（§7.1），一半在 KDA 转移矩阵特有的代数结构里（§7.2–7.3）。

### 7.1 递推在 GPU 上为什么慢

递归形式写出来就这么长：

```text
for token in sequence:
    state = update(state, token)
```

它有两个特征：总运算量随 $T$ 线性增长（好事）；第 $t$ 步依赖第 $t-1$ 步，无法把全部 token 一次送入大 GEMM（坏事）。现代 GPU 更喜欢

```text
large matrix × large matrix
```

而不是

```text
几万次小矩阵状态更新
```

**问题不只是"串行依赖"，还有"算术强度"**：每个 token 的状态更新是一组 $d_k\times d_v$ 的矩阵读写与矩阵-向量乘（Moonshot 工程师的知乎解析（本地草稿 `kda_linear_attention-deep-dive_draft-2.md` 保留其技术内容）据此估算"约 6 次 head-dim × head-dim 的 CUDA core 乘加"，并进一步推断"head_dim=128 时 prefill 偏 compute bound、训练需间隔保存 FP32 中间状态"（**这是该文的口述直觉，本文不作断言**）；可从本地源码确认的事实是：论文 Eq. 13 的 $6Td_h^2$ 项与 §3.2 的矩阵化动机；FLA chunk 路径默认 `disable_recompute=False`（前向释放 `w,u,qg,kg,v_new,h`、反传重算，`chunk_fwd.py` L127–134），中间 `h` 也非一律 FP32（只有 `final_state` 明确 FP32，`chunk_delta_h.py` L714–719）。DeltaNet 论文对"纯递归 vs chunkwise"做过直接测速（§3.2 的速度对比图）：

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/deltanet-fig1-chunk-speed.jpg" alt="DeltaNet 论文的 Triton kernel 测速：横轴为序列长度（0.5K 至 16K），纵轴为 chunkwise 并行实现相对纯递归实现的加速比，三条曲线对应 head dim 64/128/256，head dim 越大、序列越长加速比越高（16K/256 时约 35x）" style="width: 60%;">
</div>

> **图片来源**：DeltaNet（arXiv 2406.06484）§3.2（Chunkwise Parallel Form for DeltaNet）的 "Speed comparison"（chunkwise vs 纯递归 Triton 实现）。本地副本：`references/papers/delta-rule/delta-rule.md`。**这张图的纵轴是"加速比"而非耗时**：它量化了"递推形式即使总 FLOPs 线性，在 GPU 上仍然慢多少"，收益随 head_dim 与序列长度放大，与 §7.1 的"CUDA core bound"直觉和 §6.1 的 $Td_h^2$ 项方向一致（**该图证明 chunkwise 相对递归有更高加速，但不构成"compute bound"硬件归因的证明**）。

Kimi Linear 报告 §6.3 给出了正式的理论口径：**对固定 chunk $C$、单 head、head-dim $d_h$，KDA 的训练 FLOPs 为（[论文 Eq. 13](https://arxiv.org/abs/2510.26692)）：**

$$
\operatorname{FLOPs}_{\text{KDA}}(T;C,d_h)=6Td_h^2+3TCd_h+TC^2,
$$

而全注意力的主导项是 $2T^2d_h$（Eq. 14）。注意：**$6Td_h^2$ 那一项是 $d_h^2$ 而不是 $d_h$**，head_dim 变大时，矩阵状态读写和状态更新并不便宜，这也是后面 FlashKDA 选择更小 CHUNK 的动机之一（§12.5）。

### 7.2 KDA 的转移矩阵是受约束的 DPLR

现在看代数结构。令 $D_t=\operatorname{Diag}(\alpha_t)$，KDA 的**状态转移部分**是：

$$
A_t=\left(I-\beta_tk_tk_t^\top\right)D_t.
$$

展开：

$$
A_t=D_t-\beta_tk_t\left(k_t^\top D_t\right)=D_t-\beta_tk_t\left(D_tk_t\right)^\top.
$$

于是 $A_t=D_t-a_tb_t^\top$，其中 $a_t=\beta_tk_t$、$b_t=D_tk_t=k_t\odot\alpha_t$。**它是对角 + 秩 1 的：DPLR（Diagonal-Plus-Low-Rank）转移矩阵。** 这与 Kimi Linear 报告 §6.2 的对应式逐项吻合（论文写的是 $\mathbf D-\boldsymbol a_t\boldsymbol b_t^\top$，$\mathbf D=\operatorname{Diag}(\alpha_t)$，$\boldsymbol a_t=\beta_t\boldsymbol k_t$，$\boldsymbol b_t=\boldsymbol k_t\odot\boldsymbol\alpha_t$）。

一般 DPLR 里的 $a_t,b_t$ 可以互相独立；**KDA 把低秩修正的左右两个方向都绑定到当前 key**。这个约束牺牲了一部分自由度，换来两个收益：（1）语义上更贴近经典 delta rule 的"沿当前 key 修正映射"；（2）大幅简化 chunkwise 计算。这正是 [DPLR 数学原理](https://zhiyuan1i.github.io/posts/dplr-mathematics/) 那篇博客反复强调的框架：DPLR 是"对角衰减 + 低秩修正"的统一形式，KDA 是从它特化出来的，用它的记号（$P_t=\operatorname{diag}(\exp(g_t))+b_ta_t^\top$）对照，KDA 就是把 $P_t$ 的分解中两个低秩向量都固定为 key（博客与论文的记号方向相反：博客按 SSM 习惯写成"$\operatorname{diag}+$低秩"，论文按"$\mathbf D-$低秩"，两者互为同一结构的符号选择，本文统一用论文记号）。

### 7.3 一般 DPLR 为什么难：baseline、二次分块与关键绑定

既然 KDA 只是 DPLR 的特例，理论上可以用一般 DPLR 的 chunkwise 算法来算 KDA，**先看这条 baseline 为什么不够好**（这一节是"为什么不用 X"式的演进分析，来源是论文 §3.2 与 §6.2 的并排伪代码，即论文 Listing 8a/8b）：

1. **一般 DPLR 的 chunkwise 分解需要两套 WY/UT 处理。** DPLR 的 chunk 内转移矩阵是 $\operatorname{Diag}(a)-u_tw_t^\top$，而 chunkwise 需要把一串"对角 + 秩 1"因子的乘积压缩成"对角衰减 + 低秩修正"（WY 表示，§8.1）；$u$、$w$ 是自由向量时，二者的衰减路径不同，需要分别维护两套辅助矩阵，关键 chunk 矩阵从 2 个变成 4 个，并额外多出约 3 次矩阵乘法。
2. **一般 DPLR 还要二次分块。** 论文 §3.2 点明：细粒度衰减（fine-grained decay）在除法运算中带来数值精度问题（比如 §8.2 中 Eq. 9 的分母项 $K/\Gamma$），GLA 的解法是对数域计算 + **全精度二次分块**（secondary chunking），但这会阻止半精度矩阵乘法，显著降低算子速度。KDA 通过绑定 $a=b=k$，把二次分块的 chunk 矩阵计算从 4 个降到 2 个，并省去 3 次额外矩阵乘（[论文 §3.2](https://arxiv.org/abs/2510.26692)）。
3. **最终方案的收益是多少**：论文报告算子效率平均提升约 100%（相比一般 DPLR 实现），注意这是 **kernel 级**的效率差异，不是模型端到端。

> 这个"一般 DPLR → 二次分块 → 绑定 key"的演进路径并不是凭空对比：论文的 Listing 8a/8b 给了两者并排的 chunkwise PyTorch 伪代码，[Zhiyuan Li 的 DPLR 文章](https://zhiyuan1i.github.io/posts/dplr-mathematics/)则用同一框架把 DPLR、KDA、IPLR（Independent-Projection Low-Rank）放在一起比较。**本文只引用其中与 KDA 直接相关的一支；IPLR 等其它特化不属于本文范围。**

于是驱动问题的完整答案可以这样表述：**KDA 的逐 token 递推既有"串行依赖"又有"低算术强度"两个工程弱点；而它的转移矩阵恰好是受约束的 DPLR（$a_t=\beta_tk_t$、$b_t=k_t\odot\alpha_t$），这个"恰好"让"把一串秩 1 修正压缩成稠密表示"的 WY/UT 分解只需要一套矩阵与一次三角求解，数学结构允许，硬件才用得上。** 下面的章节就推导这套分解。

## 八、Chunkwise 分解：chunk 内并行、chunk 间递归


把序列切成长度 $C$ 的块（$T=N_{\text{chunk}}\cdot C$），chunkwise 算法的目标是一句话：**在数学上完全等价的前提下重排计算顺序，块间仍串行传递状态（只剩 $T/C$ 步），块内 $C$ 个 token 全部改写成 $C\times C$、$C\times d$ 的矩阵乘法一次算完。** 下面按三个子挑战展开：先建立 WY 表示（§8.1），再解决 chunk 内的打包（§8.2）与 chunk 间的传播（§8.3），最后是数值稳定性（§8.4）。

### 8.1 子挑战 1：WY 表示，先把一串秩 1 修正压缩成紧凑表示

回顾 KDA 递推（块内位置 $r$，块首状态 $S_{[t]}^0$）：

$$
S_{[t]}^r=\underbrace{\left(\prod_{i=1}^r\left(I-\beta_{[t]}^ik_{[t]}^ik_{[t]}^{i\top}\right)\operatorname{Diag}(\alpha_{[t]}^i)\right)}_{\textstyle:=P_{[t]}^r}S_{[t]}^0+\sum_{i=1}^r\underbrace{\left(\prod_{j=i+1}^r\left(I-\beta_{[t]}^jk_{[t]}^jk_{[t]}^{j\top}\right)\operatorname{Diag}(\alpha_{[t]}^j)\right)\beta_{[t]}^ik_{[t]}^iv_{[t]}^{i\top}}_{\textstyle:=H_{[t]}^r}.
$$

这是[论文 Eq. 2](https://arxiv.org/abs/2510.26692)的"部分展开"：把 $S_{[t]}^r$ 拆成"块首状态 $S^0$ 的贡献（$P^r$）"与"块内各次写入的历史贡献（$H^r$）"。直接计算 $P^r$ 需要连乘 $r$ 个非对角的 $d_k\times d_k$ 矩阵，但每个因子都是"对角 + 秩 1"（§7.2），**$r$ 个秩 1 修正的乘积，其总修正的秩最多是 $r$**，因此必然可以写成紧凑形式：

$$
P_{[t]}^r=\operatorname{Diag}(\gamma_{[t]}^r)-\sum_{i=1}^r\operatorname{Diag}(\gamma_{[t]}^{i\to r})k_{[t]}^iw_{[t]}^{i\top},
\qquad
H_{[t]}^r=\sum_{i=1}^r\operatorname{Diag}(\gamma_{[t]}^{i\to r})k_{[t]}^iu_{[t]}^{i\top}.
$$

[论文 Eq. 3](https://arxiv.org/abs/2510.26692)（**WY 表示**，经典 Householder 数值线性代数技巧，DeltaNet 与 Comba 都用过；论文称其遵循 Comba 的 $P$ 表述以减少后续矩阵求逆）。这里 $\gamma$ 是**逐通道累积衰减**：

$$
\gamma_{[t]}^r=\alpha_{[t]}^1\odot\cdots\odot\alpha_{[t]}^r\in(0,1]^{d_k},\qquad
\gamma_{[t]}^{i\to r}=\gamma_{[t]}^r\oslash\gamma_{[t]}^i.
$$

为什么衰减是对角的就能直接连乘合并：$\operatorname{Diag}(a_1)\operatorname{Diag}(a_2)=\operatorname{Diag}(a_1\odot a_2)$，逐通道独立连乘。辅助向量 $w_{[t]}^r\in\mathbb{R}^{d_k}$、$u_{[t]}^r\in\mathbb{R}^{d_v}$ 由递推（[论文 Eq. 4/5](https://arxiv.org/abs/2510.26692)，即 Appendix B 的 Proposition 1 的证明过程）给出：

$$
w_{[t]}^r=\beta_{[t]}^r\left(\gamma_{[t]}^r\odot k_{[t]}^r-\sum_{i=1}^{r-1}w_{[t]}^i\left(k_{[t]}^{i\top}\operatorname{Diag}(\gamma_{[t]}^{i\to r})k_{[t]}^r\right)\right),
$$

$$
u_{[t]}^r=\beta_{[t]}^r\left(v_{[t]}^r-\sum_{i=1}^{r-1}u_{[t]}^i\left(k_{[t]}^{i\top}\operatorname{Diag}(\gamma_{[t]}^{i\to r})k_{[t]}^r\right)\right).
$$

**先别被下标吓到** ，这两式在说：位置 $r$ 的有效修正（$w^r$/$u^r$）由当前位置的原始 key/value 减去**块内更早修正经过衰减后的贡献**。它把"块内 $r$ 次 delta 更新串行相互影响"编码进了两个递推向量。

在动手推导矩阵化之前，先看 GLA 论文是怎么画 chunkwise 计算图的，KDA 与它在"哪块用 TensorCore、哪块必须串行"上是同一张图（差别只在二阶分块的数值处理）：

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/gla-fig3-chunkwise-map.jpg" alt="GLA 论文的 chunkwise 计算注意力样式图：对角块（level 2, 浅色）为 chunk 内因果部分，非对角块（level 1, 灰色）为跨 chunk 部分；level 2 对角块中的第一行子块与第一列子块不能用 TensorCore（标 x），其余块可用（标勾）" style="width: 70%;">
</div>

> **图片来源**：Gated Linear Attention Transformers（arXiv 2312.06635）Figure 3，§4.2（Chunkwise Parallel Form of GLA，Attention-style map）。本地副本：`references/papers/gated-linear-attention/gated-linear-attention.md`。

> 图为 GLA 的 chunkwise 结构，**level 1（灰）= 跨 chunk 的行（用 TensorCore），level 2（彩）= chunk 内对角块（部分用 TensorCore）**。KDA 同样遵循"块间递归 + 块内并行"的划分；区别在于 KDA 的对角块内部还嵌着 delta 修正的三角求解（§8.2 的 UT transform），这正是图里打勾/打叉那几块的工程故事。

### 8.2 子挑战 1 收尾：UT transform 把递推打包成一次三角求解

设块内已经算好累积对数衰减 $G_i=\sum_{l\le i}\log\alpha_l$（逐通道），$\gamma_i=\exp(G_i)$（$e^G$ 逐元素）。定义三组"预处理"向量：

$$
\tilde q_i=\gamma_i\odot q_i,\qquad \tilde k_i=\gamma_i\odot k_i,\qquad \hat k_j=k_j\oslash\gamma_j .
$$

（$\oslash$ 表示逐通道除法，$k_j$ 与 $\gamma_j$ 逐元素相除；下文的 $U$ 与 $\gamma$ 类似。）关键代数技巧，[Zhiyuan Li 的 KDA 数学文章](https://zhiyuan1i.github.io/posts/kda-mathematics/) 称之为"Affine 变换"的准备工作：

$$
\operatorname{Diag}(\gamma^{j\to i})k_j=\gamma_i\odot(k_j\oslash\gamma_j)=\gamma_i\odot\hat k_j\qquad(j\le i) .
$$

其中 $\gamma^{j\to i}=\gamma_i\oslash\gamma_j$（与 §8.1 的 $\gamma^{i\to r}$ 同义，只是下标字母不同；$j\le i$ 时衰减系数 $\le 1$）。**所有跨时间衰减都被"拆"成"目标时刻的 $\gamma$ 乘当前时刻的 $1/\gamma$"**。于是可以用归纳法证明（完整推导放**附录 D**，这里给结论）：存在向量组 $u_1,\dots,u_C$，使

$$
S_i=\operatorname{Diag}(\gamma_i)\left(S_0+\sum_{j\le i}\hat k_j\,u_j^\top\right),
\qquad
u_i=\beta_i\left(v_i-S_0^\top\tilde k_i-\sum_{j<i}\left\langle\tilde k_i,\hat k_j\right\rangle u_j\right).
$$

矩阵化：取严格下三角矩阵 $A_{ij}=\beta_i\langle\tilde k_i,\hat k_j\rangle\ (j<i)$ 与右端项 $B_i=\beta_i(v_i-S_0^\top\tilde k_i)$，上面的递推恰好是

$$
(I+A)U=B\quad\Longrightarrow\quad U=(I+A)^{-1}B .
$$

这步就是 **UT transform**（论文称 UT transform 把非矩阵乘的 FLOPs 换给 TensorCore，[论文 Eq. 6/7](https://arxiv.org/abs/2510.26692)）：

$$
M=\left(I+\operatorname{StrictTril}\left(\operatorname{Diag}(\beta)\left(\Gamma^{1\to C}\odot K\right)\left(K/\Gamma^{1\to C}\right)^\top\right)\right)^{-1}\operatorname{Diag}(\beta),
$$

$$
W=M\left(\Gamma^{1\to C}\odot K\right),\qquad U=MV .
$$

两个漂亮的性质：**$I+A$ 是单位对角的下三角矩阵，永远可逆**（用前向替换求解，代价 $O(C^2d_v)$，且严格下三角幂零，$A^C=0$，$U$ 还可以用分块 Neumann 级数高度并行地算）；**构造 $A$ 本身就是一个 $C\times C$ 的 GEMM**（$(\operatorname{Diag}(\beta)\tilde K)\hat K^\top$ 取严格下三角）。

> **与教学版记号的对账**：这里 $U=MV$ 已经吸收了对块首状态的修正（$W S_0$），所以论文把 $\left(U-W S_{[t]}\right)$ 整体称为 **"pseudo-value" term**（Eq. 9 下划线标注）。教学推导（附录 D）写的是"未吸收版" $U_{\text{teach}}=M\left(V-\tilde K S_0\right)$，两者关系为 $U_{\text{paper}}=U_{\text{teach}}+WS_0$。**写作时容易混淆的就是这个，论文的伪代码里 $U$ 不是"纯写入量"，$W$ 也不是纯衰减键，二者合起来才是有效写入。**

**物理直觉**：块内 $C$ 次串行的 rank-1 更新，其相互干扰全部被浓缩进这个 $C\times C$ 的小三角矩阵 $A$；解一次三角方程组，就等价于按顺序做完了 $C$ 次 delta 更新。串行性没有消失，而是**从"沿着 $d_k\times d_v$ 大状态做 $C$ 步串行"压缩成"解一个 $C\times C$ 小三角阵"**，后者便宜得多，且本身可并行。

### 8.3 子挑战 2：状态传播与输出

拿到 $W,U$ 之后，"delta 修正"这个困难就彻底被打包了，剩下的是标准门控线性注意力的 chunkwise 流程。状态传播（[论文 Eq. 8](https://arxiv.org/abs/2510.26692)）：

$$
S_{[t+1]}=\underbrace{\operatorname{Diag}(\gamma_{[t]}^C)S_{[t]}}_{\text{块首状态衰减到块尾}}+\underbrace{\left(\Gamma_{[t]}^{i\to C}\odot K_{[t]}\right)^\top\left(U_{[t]}-W_{[t]}S_{[t]}\right)}_{\text{块内各次有效写入累加}}\in\mathbb{R}^{d_k\times d_v}.
$$

块内每个位置的输出（[论文 Eq. 9](https://arxiv.org/abs/2510.26692)）：

$$
O_{[t]}=\underbrace{\left(\Gamma_{[t]}^{1\to C}\odot Q_{[t]}\right)S_{[t]}}_{\text{inter-chunk：读块前状态}}+\underbrace{\operatorname{Tril}\left(\left(\Gamma_{[t]}^{1\to C}\odot Q_{[t]}\right)\left(K_{[t]}/\Gamma_{[t]}^{1\to C}\right)^\top\right)\left(U_{[t]}-W_{[t]}S_{[t]}\right)}_{\text{intra-chunk：块内因果注意力}}.
$$

> 行文习惯：论文 Eq. 8/9 用 $S_{[t]}$ 表示"第 $t$ 个 chunk 的块首状态"，$S_{[t+1]}$ 是块尾状态（即下一块的块首）。FLA 源码里这个量叫 `h`（chunk 边界状态），初始状态叫 `initial_state`，中间状态数组 `h` 的形状是 `[B, NT, HV, K, V]`（`NT = ceil(T/C)`），`v_new` 就是 $\left(U-WS_{[t]}\right)$，见 [FLA `fla/ops/common/chunk_delta_h.py`](https://github.com/fla-org/flash-linear-attention/blob/6ec09887145d2f51bf203304777770b041374b65/fla/ops/common/chunk_delta_h.py) 的 `chunk_gated_delta_rule_fwd_h`（L687 起）。该内核逐 token 做 `b_v = u − w·h`（w·h 部分在 L212–L229，减法则在 L237–L238）、把衰减后的键 `kg = k·exp(g_last − g_t)` 与 `b_v` 外积累加进 `h`（L244–L301）、结尾对 `h` 乘上 `exp2(gk_last)` 得到块尾状态，与 Eq. 8 逐项对应。

**至此 chunkwise 的数学闭环完成**。对照一下 FLA 真正的教学参考实现（`fla/ops/kda/naive.py::naive_chunk_kda`，[commit `6ec09887`，L69–166](https://github.com/fla-org/flash-linear-attention/blob/6ec09887145d2f51bf203304777770b041374b65/fla/ops/kda/naive.py#L69-L166)）可以确认这套公式的全部实现细节：

```python
# fla/ops/kda/naive.py  L120-163 的要点（删去注释与 padding 后）
g = g.cumsum(-2)                                  # 块内累积对数衰减 G（逐通道）
A = ...                                          # A[c,i] = beta_c * <k_c, k_i> * exp(g_c - g_i)，严格下三角
A = -A.masked_fill(mask, 0)                      # 负号后为严格下三角 A := -(beta <k~, k^>)
for i in range(1, BT):                           # 就地累加 Neumann 级数：A ← A + A^2 + ...
    A[..., i, :i] = A[..., i, :i] + (A[..., i, :, None] * A[..., :, :i]).sum(-2)
A = (A + torch.eye(BT)) * beta[..., None, :]     # +I 补齐 (I-A)^{-1}，再逐列乘回 beta（即论文的 M）
w = A @ (g.exp() * k)                            # W = M (Gamma ⊙ K)
u = A @ v                                        # U = M V
# ... 每个 chunk:
v_i = u_i - w_i @ S                              # pseudo-value = U - W S
o = (q_i * g_i.exp()) @ S + Aqk @ v_i            # inter + intra
S = S * g_i[:, :, -1].exp()                        # Diag(gamma_C) S
S += ((g_i[:, :, -1:] - g_i).exp() * k_i).transpose(-1, -2) @ v_i  # (Gamma^{i->C} ⊙ K)^T (U - W S)
```

（`naive.py` L162–163 的原始写法：`rearrange((g_i[:, :, -1:] - g_i).exp() * k_i, 'b h c k -> b h k c') @ v_i`，即"按 token 的衰减权重乘上 key 后，转置成 $d_k\times C$ 再与 $U$ 相乘"。）

> **行列记号的最终检定**：论文 Eq. 8/9 里 $S_{[t]}$ 就是本节开头的块首状态 $S_{[t]}^0$（论文在部分展开时用上标 $0$，在 Eq. 8/9 里省略了上标；$S_{[t+1]}$ 是下一块的块首状态）。FLA 源码里块边界状态叫 `h`，$S_{[t+1]}$ 对应 `final_state`；`h` 的存储布局 `[B, NT, HV, K, V]` 与 `state_v_first=True/False` 两种转置在 [`chunk_delta_h.py` L714–719](https://github.com/fla-org/flash-linear-attention/blob/6ec09887145d2f51bf203304777770b041374b65/fla/ops/common/chunk_delta_h.py#L714-L719) 显式分配。

> 注意教学版把"矩阵化"做到位、但把"硬件效率"留在注释里：`Akk`/`Aqk` 都用了逐元素循环而非一次性 GEMM，实际 Triton kernel（`chunk_intra.py`）的做法是 16×16 分块 + 前节/前列专属处理（§11.2）。**教学版存在的意义是数值校验，不是性能。**

### 8.4 子挑战 3：数值稳定性，同一组公式的三次起义

把公式"翻译"成 kernel 时，最容易被忽略的是：**g 在对数域，$g\le 0$，但 $e^g$、$e^{-g}$、$e^{g/2}$ 的数值行为完全不同。** Moonshot 工程师在知乎解析（本地草稿 draft-2）里把这整段故事称为"从循环到 TensorCore：forget gate 的数值雷区"，剧情分四幕：

1. **第一幕：朴素写法就溢出。** 循环版里写 `decay = (g[j+1:i+1]).sum().exp()` 是安全的（$g\le0\Rightarrow e^g\in(0,1]$）。但要把带门控的块内 Gram 矩阵 $A_{ij}=\beta_i\langle k_i,k_j\rangle e^{G_i-G_j}$ 拍平成两个矩阵乘、喂给 TensorCore，公式必须改写为

$$
A_{ij}=\underbrace{\left(\beta_i k_i\odot e^{G_i}\right)^\top}_{\text{left factor}}\cdot\underbrace{\left(k_j\odot e^{-G_j}\right)}_{\text{right factor}} .
$$

$e^{-G_j}$ 里的负号把符号翻转：bf16 的最大有限值约为 $3.39\times10^{38}$，$e^{-G_j}$ 在 $-G_j>\ln(3.39\times10^{38})\approx88.7$（即 $G_j<-88.7$）时直接溢出为 $\inf$（随后才可能在运算中变成 NaN）；而在 $G_j$ 只是"较负"（如 $-50$ 上下）的区间，左侧 $e^{G_i}$ 与右侧 $e^{-G_j}$ 的量级已相差数十个数量级（例如 $G_i=G_j=-50$ 时，$e^{-G_j}/e^{G_i}$ 约 $2.7\times10^{43}$ 倍；因 $i\ge j$，实际比值不低于该量级）、乘法结果同样不可靠，**数值未定义/不可信**。不过该危险只出现在"未钳制"的 gate 上：`safe_gate` + `lower_bound=-5` 把单步衰减钳到 $[e^{-5},1)$，chunk 内累积衰减幅度受控，这也是 FlashKDA 选 CHUNK=16 的理由之一（§12.5）。

2. **第二幕：相对偏移 + clamp 能救，但引入 loss spike。** 利用 $e^x$ 的性质显式引入相对偏移并立即 clamp，左右因子就都数值安全了，但实践训练发现这会造成 loss spike，说明"每步都 clamp"改变了衰减路径的可微训练行为。

3. **第三幕：FLA 的实际做法不是"只有对角块需要 $e^{-g/2}$"，而是在子块内做相对衰减。** 看 [`chunk_intra.py` 的内核](https://github.com/fla-org/flash-linear-attention/blob/6ec09887145d2f51bf203304777770b041374b65/fla/ops/kda/chunk_intra.py#L748-L756)：以子块中位（`BC//2`）处的 gate `b_gn` 为参考，算相对值 `b_gm = b_g - b_gn`，再用 `exp2(+b_gm)`/`exp2(-b_gm)` 表达衰减，源码注释明言"subtracting the left boundary less than 85 to avoid overflow in exp2"。也就是说**真实内核从不物化裸的 $e^{-g_j}$**（更非 $e^{-g_j/2}$）：它等价于把 $e^{G_i-G_j}$ 拆成"相对参考点的衰减比"，与 $i\ge j$ 时 $G_i-G_j\le 0$（永不爆炸）的对数差融合（[GLA 论文的做法](https://arxiv.org/abs/2312.06635)，也是 §7.3 说"一般 DPLR 需要二次分块"的原因）。

4. **第四幕：KDA 的解法是"干掉问题本身"。** 论文把 $a,b$ 都绑定为 $k$，使关键 chunk 矩阵从四个减为两个、免去两次二次分块（见 §7.3）；而生产实现（FLA 与 FlashKDA）再配合三招：**log 域 gate 激活 + chunk-local cumsum（乘以 $1/\ln 2$ 后用硬件 $exp2$ 指令，见 [FLA `gate.py`](https://github.com/fla-org/flash-linear-attention/blob/6ec09887145d2f51bf203304777770b041374b65/fla/ops/kda/gate.py) 的 `kda_gate_chunk_cumsum` 与 `utils/constant.py` 的 `RCP_LN2`）**；**gate 下界（`safe_gate` + `lower_bound=-5`，把 $g$ 钳到 $[-5,0)$，每步衰减 $\exp(-5)\approx0.0067$，并可启用 M=16 TensorCore 快速路径）**；**Q/K L2 归一化**（$\|k\|=1$ 保证转移矩阵特征值稳定，论文引 DeltaNet 的建议）。

> **这一幕（draft-2 的口述演进）与 FLA 实际实现的对应关系值得说清**：草稿讲的是"朴素 `exp(-g/2)` → 相对偏移+clamp → 二次分块回退 → 干掉问题本身"的迭代；而本地 FLA 源码采用的就是"相对偏移"这一步的具体化（子块内相对 gate 差 + `exp2`），并配套架构级绑定（$a=b=k$，§7.3）与 kernel 级 `safe_gate` 下界。**不要把草稿中"只有对角块需要 $e^{-g/2}$/需要回退 CUDA cores"的表述当成 FLA 的事实**；可核实的事实是 §5.3 的"绑定 key 免去一般 DPLR 的二次分块"与这里的相对差实现。

涉及 $\beta$ 的还有两个必须显式化的开关（[FLA `chunk.py` L60–62、L403–404](https://github.com/fla-org/flash-linear-attention/blob/6ec09887145d2f51bf203304777770b041374b65/fla/ops/kda/chunk.py#L60-L62)）：`use_beta_sigmoid_in_kernel=True` 时 kernel 内做 `sigmoid(beta)`，`allow_neg_eigval=True` 时再乘 2（$\beta\in[0,2)$，允许负特征值增强状态追踪能力，参考 [*Unlocking State-Tracking in Linear RNNs Through Negative Eigenvalues*](https://arxiv.org/abs/2411.12537)）；`use_qk_l2norm_in_kernel=True` 时 kernel 内做 L2 归一（反向自动链接 `l2norm_bwd`）。


## 九、完整 KDA Layer：参数化与公式到源码的对应


### 9.1 单层数据流：从数学对象到官方代码

设输入 $x\in\mathbb{R}^{B\times T\times 2304}$。官方 [`KimiDeltaAttention`](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct)（`modeling_kimi.py` L444–605，本地副本行号）的数据流是（**下图为本文按官方代码自绘的流程图，非论文原图**）：

```mermaid
flowchart TD
    A["x: [B,T,2304]"] --> B["q/k/v 投影<br/>q_proj/k_proj: 2304→4096, v_proj: 2304→4096"]
    B --> C["ShortConv + SiLU<br/>q/k/v_conv1d, kernel=4"]
    C --> D["reshape 多头<br/>Q,K,V: [B,T,32,128]"]
    A --> E["忘门（从 x 直接低秩投影）<br/>r=f_b_proj∘f_a_proj: 2304→128→4096<br/>激活 g = -exp(A_log)·softplus(r + dt_bias) → α=exp(g)"]
    A --> F["b_proj（从 x）：2304→32 → sigmoid → beta: [B,T,32]"]
    E --> G["KDA core: chunk_kda / fused_recurrent_kda<br/>(use_qk_l2norm_in_kernel=True)"]
    F --> G
    G --> H["输出门（从 x 低秩投影）g_b_proj∘g_a_proj<br/>o = o_norm(o, g)  # FusedRMSNormGated(sigmoid)"]
    A -.-> H
    H --> I["o_proj: 4096→2304"]
    I --> J["[B,T,2304]"]
```

第一步，Q/K/V 投影（[`modeling_kimi.py` L465–469](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct)）：

$$
Q_{\text{flat}},K_{\text{flat}},V_{\text{flat}}\in\mathbb{R}^{B\times T\times 4096},\qquad 4096=32\times128 .
$$

第二步，ShortConv（kernel=4，SiLU），不承担长程记忆，只做短距离局部混合（L471–485）。

第三步，reshape 多头 $Q,K,V\in\mathbb{R}^{B\times T\times 32\times 128}$（L563–565）。

第四步，生成逐通道遗忘门（**低秩**投影：`f_a_proj: hidden→head_dim(128)`、`f_b_proj: 128→4096`，L490–491；`A_log` 为每 head 参数，初始化为 $\log\mathcal{U}(1,16)$ 形状 `[1,1,32,1]`，L487–488；`dt_bias: [4096]`，L493–494）：

$$
r_t=f_b\left(f_a(x_t)\right),\qquad g_t=-\exp(A_{\log})\odot\operatorname{softplus}\left(r_t+\operatorname{dt\_bias}\right),\qquad \alpha_t=\exp(g_t),
$$

$$
g_t\le 0,\quad 0<\alpha_t\le 1 .
$$

> **$g$ 是 log decay，不是直接的 $\alpha$！** 且它的 shape 是 `[B,T,HV,K]`（逐 value-head、逐 key 维；官方 checkpoint $HV=H=32$，但 GVA 架构下 `[B,T,HV,K]` 才是通用形式，[FLA `naive.py` L30–31 与 `chunk.py` L414 的 shape 断言](https://github.com/fla-org/flash-linear-attention/blob/6ec09887145d2f51bf203304777770b041374b65/fla/ops/kda/chunk.py#L414-L415) 均如此）。**GDN 才是标量 `[B,T,H]`。**

第五步，生成 delta 学习率 $\beta_t=\sigma\left(W_\beta x_t\right)\in\mathbb{R}^{32}$（`b_proj: 2304→32`，L496；**官方代码在模型侧就做了 sigmoid**，L561：`beta = self.b_proj(hidden_states).float().sigmoid()`）。

第六步，KDA core：输入 $Q,K,V,g,\beta$，输出 $O_{\text{core}}\in\mathbb{R}^{B\times T\times 32\times128}$，循环状态 $S\in\mathbb{R}^{B\times 32\times128\times128}$（L567–590）。**注意官方在调用内核时只传了 `use_qk_l2norm_in_kernel=True`（L576/L588），没有传 `state_v_first`，也没有传 `use_gate_in_kernel` / `use_beta_sigmoid_in_kernel`**，即 gate 由模型外 `fused_kda_gate` 激活、$\beta$ 在模型外 sigmoid、状态布局用默认 `[N,HV,K,V]`。

第七步，输出门与归一化（另一组低秩投影 `g_a_proj/g_b_proj`，L498–499；`o_norm = FusedRMSNormGated(head_dim, activation='sigmoid')`，L501–502）：

$$
o_t=W_o\left(\operatorname{Sigmoid}\left(W_g^\uparrow W_g^\downarrow x_t\right)\odot\operatorname{RMSNorm}\left(\operatorname{KDA}(q_t,k_t,v_t,\alpha_t,\beta_t)\right)\right).
$$

这是[论文 Eq. 10](https://arxiv.org/abs/2510.26692)（全部项与官方代码逐项对应）。

### 9.2 公式到源码的逐项对应（官方 checkpoint vs FLA 高层 layer）

| 数学对象 | 官方 checkpoint（`modeling_kimi.py`） | FLA 高层（`fla/layers/kda.py`，commit `6ec09887`） |
| --- | --- | --- |
| $q_t$ | `q_proj` → `q_conv1d`（SiLU，kernel 4） | 同名；`ShortConvolution`（SiLU） |
| $k_t$ | `k_proj` → `k_conv1d` | 同名 |
| $v_t$ | `v_proj` → `v_conv1d` | 同名（`expand_v` 可支持 GVA） |
| log decay $g_t$ | `f_a_proj`→`f_b_proj`→`fused_kda_gate`（模型外激活） | `f_proj` + `A_log` + `dt_bias`（`use_gate_in_kernel=True` 时 kernel 内激活） |
| $\alpha_t$ | kernel 内 `exp(g)` | kernel 内 `exp(g)`（$1/\ln2$ 缩放后 `exp2`） |
| $\beta_t$ | `b_proj` → 模型外 `.sigmoid()` | `b_proj`；`use_beta_sigmoid_in_kernel=True` 时 kernel 内 sigmoid |
| $S_t$ | `recurrent_state`（`KimiDynamicCache`，L540/L592） | `initial_state`/`final_state` |
| KDA core | `chunk_kda` / `fused_recurrent_kda` | 同名算子（`mode="chunk"/"fused_recurrent"`） |
| 状态布局 | 默认 `[N,HV,K,V]`（未传 `state_v_first`） | 传 `state_v_first=True`（`[N,HV,V,K]`） |
| 输出门 | `g_a_proj`→`g_b_proj` | `g_proj`（低秩，`bias=True` — 官方同） |
| 输出归一化 | `FusedRMSNormGated` | 同名 |
| 最终投影 | `o_proj` | 同名 |

**这是"算法相同、接线不同"的活例子**：官方模型把 gate 激活与 $\beta$ sigmoid 放在 Python 侧（顺便说一句，官方 `modeling_kimi.py` 本身依赖 FLA，`from fla.ops.kda.gate import fused_kda_gate`，L26），FLA 高层则是"三合一"（L2norm/gate/beta-sigmoid 全下沉到 kernel）；**注意：本表是源码语义对照，不保证 HF remote code 与本 FLA pin 的运行时 API 兼容**（例如官方 L560 调用 `fused_kda_gate(g, A_log, head_dim, g_bias=dt_bias)`，与 `6ec09887` 的签名 `fused_kda_gate(g, A_log=None, dt_bias=None, lower_bound=None, output_dtype=...)` 形状不同，实际共装运行需锁定兼容的 `fla` 版本或加 adapter）；而 FLA 高层的 `KimiDeltaAttention.forward` 还会自动切模式（[`fla/layers/kda.py` L211–218](https://github.com/fla-org/flash-linear-attention/blob/6ec09887145d2f51bf203304777770b041374b65/fla/layers/kda.py#L211-L218)）：`torch.is_grad_enabled()` 强制 `chunk`，否则 `q_len <= 64` 用 `fused_recurrent`，与官方 `mode = 'fused_recurrent' if q_len <= 64 else self.mode`（L523）的策略一致，但官方更激进：`self.mode` 恒为 `'chunk'`（L448），训练时断言 `mode == 'chunk'`（L524–525）。

### 9.3 状态到底占多少内存

对官方 checkpoint 的一个 KDA 层：$H=32,\ K=V=128$，状态元素数 $32\times128\times128=524{,}288$。每条序列、每个 KDA 层：BF16 约 **1 MiB**、FP32 约 **2 MiB**；20 个 KDA 层合计 BF16 约 **20 MiB/序列**、FP32 约 40 MiB。这还没算：三组 ShortConv 状态（每层 3×4096×4 窗口）、7 个 MLA 层的 KV cache、scheduler 元数据与 speculative 中间状态。

这个状态不随上下文长度增长，但随并发序列数增长：$\text{memory}=O(BHV\cdot d_kd_v)$。所以：

> **固定状态不等于零成本，也不意味着短上下文一定比 KV cache 小。** 一个普通 32-head、128-dim 的全注意力层每 token 的 BF16 K/V 约为 $2\times32\times128\times2=16\text{ KiB}$；一个 BF16 KDA state 约等于 64 个 token 的这类普通 KV。这个比较只是为了建立量级直觉，Kimi 的 MLA KV 本身经过低秩压缩（`kv_lora_rank=512` + NoPE 下可转纯 MQA），实际对比以论文 Fig. 1b 的口径（混合带宽效率比 3:1）为准（§6.1）。

## 十、三条执行路径：训练、Prefill 与 Decode


回顾 §9.2 的结论：KDA 的"公式"只有一行，但使用它的系统需要**三条执行路径**，它们不是同一个函数的三个参数，而是三种不同的资源优化目标。关于 prefill/decode 调度与 KV pool 的通用背景，仓库里的 [SGLang Scheduler 学习笔记](../../sglang/scheduler/readme.md)（published）已经讲全；本章只聚焦"KDA 与全注意力在状态管理上的差异"。

### 10.1 训练：`chunk_kda`

训练要一次处理长序列、并为反向传播保留或重算中间量，因此只能使用 chunk 模式（官方 `modeling_kimi.py` L524–525 直接断言：`Only chunk mode is supported in training`；[FLA `fla/layers/kda.py` L217–218](https://github.com/fla-org/flash-linear-attention/blob/6ec09887145d2f51bf203304777770b041374b65/fla/layers/kda.py#L217-L218) 同）。`chunk_kda` 的特点：chunk 内并行、块边界状态递归、自定义反向（`ChunkKDAFunction.backward`，§11.3）、支持 packed variable-length（`cu_seqlens`）、可选 context parallel（`cp_context`）、chunk size 32 或 64、初始状态与部分累计倾向更高精度（`initial_state` 必须是 float32，[`chunk.py` L384–385](https://github.com/fla-org/flash-linear-attention/blob/6ec09887145d2f51bf203304777770b041374b65/fla/ops/kda/chunk.py#L384-L385)）。

### 10.2 Prefill：仍然主要使用 chunk

长 prompt 的 prefill 需要把所有 token 压缩成最终循环状态，并为每个 token 产生输出。逐 token recurrent 虽然总 FLOPs 线性，但 GPU 利用率太低（§7.1），所以生产框架仍用 chunk kernel；在满足条件的 Hopper/Blackwell 上，FLA 会自动分派到 FlashKDA（CUTLASS 后端，§12.5）。

**会变化的是 kernel 选择，不会变化的是"必须同时产出三个状态"**：ShortConv 的卷积状态、KDA 的循环状态、（MLA 层的）KV cache。vLLM 的固定快照（§12.2）里 prefill 分支调 `chunk_kda` 并把 `final_state` 写回对应 sequence slot，这提醒我们：**prefill 的产出物不是输出 logits，而是一整套"可继续 decode 的状态"**。

### 10.3 Decode：`fused_recurrent_kda`

自回归解码每次只有一个（或少量）token，没必要构造 chunk 矩阵，直接走逐 token 递归内核：[`fused_recurrent_kda`](https://github.com/fla-org/flash-linear-attention/blob/6ec09887145d2f51bf203304777770b041374b65/fla/ops/kda/fused_recurrent.py)。kernel 主循环（[`fused_recurrent_kda_fwd_kernel` L170–205](https://github.com/fla-org/flash-linear-attention/blob/6ec09887145d2f51bf203304777770b041374b65/fla/ops/kda/fused_recurrent.py#L170-L205)）只有五行：

```python
b_h *= exp(b_gk)          # ① 衰减循环状态（g 逐通道）
b_v -= ...                # ② 读当前 key 的旧映射：v_hat = h^T k
b_v *= b_beta             # ③ 误差乘写入强度（可选 sigmoid / ×2）
b_h += b_v ⊗ b_k          # ④ 外积修正
b_o = ...                 # ⑤ o = q^T h
```

生产内核比这五行多的东西恰恰是"状态管理"：`ssm_state_indices`（continuous batching 中定位各 sequence 的循环状态槽位）、`num_accepted_tokens`（speculative decoding 的接受 token 数）、`inplace_final_state`（把终态原地写回传入的 `initial_state`，零拷贝；公共 API 默认 `False`，kernel 层默认 `True`，vLLM 集成路径用 `True`）、`state_v_first`（状态布局与 chunk 路径对齐）。

### 10.4 Verify 不是普通 decode

Speculative decoding 的 **verify** 一次可能要验证多个候选 token，状态不能在每个候选分支上无条件永久推进，否则被拒绝 token 会污染主状态。生产系统因此需要：**临时中间状态、`num_accepted_tokens` 语义、tree/chain speculation 的限制、正确的 commit/rollback 语义**。这正是 SGLang 为 prefill、decode、verify 分别设置 backend dispatch 的原因，而不是让所有路径都调用同一个 `forward()`（§12.1）。


## 十一、FLA `fla.ops.kda` 源码走读


这一章以 [FLA 仓库](https://github.com/fla-org/flash-linear-attention/tree/6ec09887145d2f51bf203304777770b041374b65)的 commit [`6ec09887145d2f51bf203304777770b041374b65`](https://github.com/fla-org/flash-linear-attention/commit/6ec09887145d2f51bf203304777770b041374b65)（v0.6.0 前后）为准。三个重要版本点：KDA 通过 [commit `b5d48b7d`（#621，Add KimiDeltaAttention）](https://github.com/fla-org/flash-linear-attention/commit/b5d48b7d2376b7c9b344d603591cb06d93c13aea) 合入；FlashKDA backend 通过 [commit `97aaf767`（#852）](https://github.com/fla-org/flash-linear-attention/commit/97aaf76714afa5c5b443b25a5d5daac3ccc1ff54) 接入；[commit `19b5a3f4`（#919）](https://github.com/fla-org/flash-linear-attention/commit/19b5a3f411ecea6cdda62c6cc65cdae55ed2dec5) 增加 beta-sigmoid kernel fusion 与负特征值选项。

> **复现纪律**：复现实验不能只记录 `pip install flash-linear-attention`，必须记录 commit，并记录激活是在模型外完成还是 kernel 内完成（§9.2 那张表）。

### 11.1 文件结构与阅读顺序

```bash
fla/ops/kda/
├── __init__.py                    # 导出 chunk_kda / fused_recurrent_kda
├── naive.py                       # 纯 PyTorch 参考（naive_recurrent_kda / naive_chunk_kda）★数学真值
├── chunk.py                       # 公共 API + 参数校验 + ChunkKDAFunction（autograd 入口）
├── chunk_fwd.py                   # 前向编排：gate → intra → WY → inter-chunk h → output
├── chunk_bwd.py                   # 反向编排 + dAv / wy_dqkg_fused 内核
├── chunk_intra.py                 # intra：Aqk/Akk 构造、(I−tril)⁻¹、反向 intra
├── chunk_intra_token_parallel.py  # token 并行版 intra（对角 16×16 块，fp32 精度）
├── wy_fast.py                     # 由 A 重算 w/u/qg/kg（前向重计算 + 反向 prepare_wy_repr_bwd）
├── gate.py                        # 忘门激活（标准 / lower_bound）+ chunk-local cumsum
├── fused_recurrent.py             # 逐 token 递归融合内核（decode）
└── backends/                      # 后端注册（triton_ascend / flash_kda / tilelang）
```

推荐的阅读顺序：`naive.py`（数学真值）→ `chunk.py`（API 与校验）→ `chunk_fwd.py`（编排）→ `chunk_intra.py`/`wy_fast.py`（核心算法）→ `common/chunk_delta_h.py`（状态传播）→ `fused_recurrent.py`（decode）。另有 `fla/ops/precond_kda/`（同一套结构的"预条件"变体，含独立的 `initial_A_state`/`g_atk` 等参数，是 KDA 谱系的另一个家族成员；本文不展开，只提醒你在 `fla/ops/` 下不要把它当 KDA 本体）。

### 11.2 前向编排

`ChunkKDAFunction.forward`（[`chunk.py` L24–119](https://github.com/fla-org/flash-linear-attention/blob/6ec09887145d2f51bf203304777770b041374b65/fla/ops/kda/chunk.py#L24-L119)）先做 `l2norm_fwd(q/k)` 与 `fused_beta_sigmoid(beta)`（可选），再进入 [`chunk_kda_fwd`](https://github.com/fla-org/flash-linear-attention/blob/6ec09887145d2f51bf203304777770b041374b65/fla/ops/kda/chunk_fwd.py)（**下图为本文按源码行为自绘的编排图，非论文图**）：

```mermaid
flowchart TD
    A["chunk_kda(q,k,v,g,beta, ...)"] --> B{"use_gate_in_kernel?"}
    B -- True --> C["gate.py::kda_gate_chunk_cumsum<br/>g = -exp(A_log)·softplus(g+dt_bias) 或 lower_bound·sigmoid(...)<br/>+ chunk-local cumsum (× RCP_LN2)"]
    B -- False --> D["utils::chunk_local_cumsum(g) (× RCP_LN2)<br/># g 为已激活的 log decay"]
    C --> E
    D --> E
    E["chunk_intra.py::chunk_kda_fwd_intra<br/>① token_parallel 求对角 16x16 Aqk/Akk（fp32）<br/>② inter_solve_fused 求 off-diagonal + 前代法合成 A=(I−tril)⁻¹"]
    E --> F["wy_fast.py::recompute_w_u_fwd<br/>w = A@(exp(g)·k), u = A@v<br/>qg = q·exp(g), kg = k·exp(g_last−g)"]
    F --> G["cp_context? → cp::chunk_gated_delta_rule_fwd_h_pre_process（跨卡初始状态通信）"]
    G --> H["common::chunk_gated_delta_rule_fwd_h<br/>v_new = u − w@h; h_{n+1} = h·exp(g_last) + Σ decayed_k ⊗ v_new"]
    H --> I["gla::chunk_gla_fwd_o_gk<br/>o = (q·exp(g))@h + Aqk@v_new"]
    I --> J{"disable_recompute?"}
    J -- False --> K["释放 w,u,qg,kg,v_new（省显存；反传重算）"]
    J -- True --> L["保留全部中间量（省算力，反传直接复用）"]
    K --> M["返回 o, final_state"]
    L --> M
```

几个值得一读的细节：

1. **gate 的两次"cumsum"**：`chunk_fwd.py` L43–64 里，`use_gate_in_kernel=True` 时走 `kda_gate_chunk_cumsum`（gate 激活与 chunk-local cumsum 融合，阈值 `lower_bound` 可走 M=16 快速路径）；否则 `g` 视为已激活的 log decay，只做 `chunk_local_cumsum`。两者都乘以 `RCP_LN2`，之后的 kernel 里用 `exp2` 而非 `exp`（[`utils/constant.py`](https://github.com/fla-org/flash-linear-attention/blob/6ec09887145d2f51bf203304777770b041374b65/fla/ops/utils/constant.py)）。
2. **intra 分两段**（[`chunk_intra.py` L794–880](https://github.com/fla-org/flash-linear-attention/blob/6ec09887145d2f51bf203304777770b041374b65/fla/ops/kda/chunk_intra.py#L794-L880)）：先在 `chunk_kda_fwd_intra_token_parallel` 里算对角 16×16 块的 Aqk/Akk（**fp32，保证 `solve_tril` 精度**），再由 `chunk_kda_fwd_kernel_inter_solve_fused` 一次算出 off-diagonal 的 Aqk/Akk 并就地做前代法合并出完整 `Akk_inv=(I−tril)⁻¹`。`Akk` 初始化为零（kernel 只写下三角）。
3. **"策略 A"的内存选择**：`disable_recompute=False`（默认）时前向释放 `w,u,qg,kg,v_new,h`（`chunk_fwd.py` L127–134），反传用 `wy_fast.py::recompute_w_u_fwd` 重算；`True` 时全保留。对小模型训练是"省算力、费显存"的取向，大模型相反。

### 11.3 反向编排

`ChunkKDAFunction.backward`（`chunk.py` L121–173）→ [`chunk_kda_bwd`](https://github.com/fla-org/flash-linear-attention/blob/6ec09887145d2f51bf203304777770b041374b65/fla/ops/kda/chunk_bwd.py)（`chunk_bwd.py` 头部），路线是"**先算注意力矩阵梯度，再解状态链反传，最后按算子逆序补偿**"：

1. `chunk_kda_bwd_dAv`：由输出梯度 `do` 得 `dAqk = do @ v_newᵀ`（intra 部分）与 `dv`；
2. `chunk_gated_delta_rule_bwd_dhu`（`common/chunk_delta_h.py`）：沿 chunk 间状态链反传，得各 chunk 起始状态梯度 `dh`、初始状态梯度 `dh0`，并**更新 `dv`**（inter 部分的贡献）；
3. `chunk_kda_bwd_wy_dqkg_fused`：一个融合内核同时算 `dq, dk, dv, db, dg, dAkk`，把 WY 表示 $w,u$ 与 inter-chunk `h` 的梯度一次性吸收（避免多次访存）；
4. `chunk_kda_bwd_intra`：用 `dAqk, dAkk` 补 intra 部分对 `dq/dk/db/dg` 的贡献（`safe_gate` 时走 M=16 路径，否则逐 token 循环）；
5. **GVA 归约**：`dq/dk` 在 value-head 维度算出后，对 group 维 `sum` 回 qk-head；
6. 门控反传：`dg` 先做反向 chunk-local cumsum，再经 `kda_gate_bwd` 还原到 `g_raw`、`A_log`、`dt_bias` 梯度（`use_gate_in_kernel=True` 时；否则 `dA/dbias` 为 `None`）。

返回值顺序与 `chunk_kda` 参数顺序对齐（见 `chunk.py` L172–173）。

### 11.4 关键约定

| 张量/参数 | 形状/取值 | 说明 |
| --- | --- | --- |
| q,k | `[B, T, H, K]` | qk-head；`K ≤ 256`（[`chunk.py` L409](https://github.com/fla-org/flash-linear-attention/blob/6ec09887145d2f51bf203304777770b041374b65/fla/ops/kda/chunk.py#L409)） |
| v | `[B, T, HV, V]` | value-head；GVA 时 `HV ≥ H` 且 `HV % H == 0` |
| g | `[B, T, HV, K]` | 逐 value-head、逐 key 维的忘门；**`use_gate_in_kernel=True` 时是 raw preactivation（未激活），`False` 时才是已激活的 log decay（恒 ≤0）**（不是 `[B,T,H,K]`） |
| beta | `[B, T, HV]` | 写入门；`use_beta_sigmoid_in_kernel=True` 时传 logits |
| A_log | `[HV]` | 每 value-head（GVA 原生支持）；`use_gate_in_kernel=True` 时必传（`lower_bound` 未设时） |
| dt_bias | `[HV × K]` | 门控偏置 |
| initial_state | `[N, HV, K, V]`，必须 float32 | `N = B`（等长）或 `len(cu_seqlens)−1`（变长） |
| 状态布局 | 默认 `[K,V]`；`state_v_first=True` → `[V,K]` | 旧参数名 `transpose_state_layout` 已废弃（[`chunk.py` L354–362](https://github.com/fla-org/flash-linear-attention/blob/6ec09887145d2f51bf203304777770b041374b65/fla/ops/kda/chunk.py#L354-L362)） |
| chunk_size | 32 或 64；sub-chunk `BC=16` | KDA 不支持其它 chunk 大小 |
| cu_seqlens | `[N+1]`，变长时 `B=1` | FlashAttention 风格（[`chunk.py` L373–383](https://github.com/fla-org/flash-linear-attention/blob/6ec09887145d2f51bf203304777770b041374b65/fla/ops/kda/chunk.py#L373-L383)） |
| safe_gate + lower_bound | `lower_bound ∈ [-5, 0)` | 钳门到 `[lower_bound, 0)` 并启用 M=16 加速（参数合法性见 [`chunk.py` L397–401](https://github.com/fla-org/flash-linear-attention/blob/6ec09887145d2f51bf203304777770b041374b65/fla/ops/kda/chunk.py#L397-L401)，M=16 加速本身见 [`chunk.py` L248–251](https://github.com/fla-org/flash-linear-attention/blob/6ec09887145d2f51bf203304777770b041374b65/fla/ops/kda/chunk.py#L248-L251)） |
| allow_neg_eigval | 需 `use_beta_sigmoid_in_kernel=True` | `β ← 2·sigmoid(β)`（`chunk.py` L403–404） |

### 11.5 后端机制

FLA 通过 `fla/ops/backends` 的 `BackendRegistry` 按条件自动选择实现：默认 Triton（NVIDIA），`triton_ascend`（昇腾，`IS_NPU=True`），`flash_kda`（CUTLASS，verifier 逐项检查：未启用梯度 + `FLA_FLASH_KDA≠0` + bf16 + `K=V=128` + 无 GVA（`HV==H`）+ `safe_gate=True` + `state_v_first=True` + `cp_context=None` + `return_intermediate_states=False` + 三个 `*_in_kernel` 全开；仅前向），`tilelang`（安装且无 GQA 时用于反向的 `chunk_kda_bwd_wy_dqkg_fused`）。每个 `@dispatch('kda')` 函数都会先跑对应后端的 `*_verifier`，通过才使用，否则回退默认 Triton，**调用方无感知，但激活路径与数值路径可能不同**，这正是"复现必须记录版本与配置"的原因。

## 十二、生产集成：SGLang、vLLM、Megatron、verl、FlashKDA


> 本章所有类名与行号均来自对应仓库的固定 commit：SGLang [`7399c2b5587e1559f3e5a26566ed322e81e1433a`](https://github.com/sgl-project/sglang/tree/7399c2b5587e1559f3e5a26566ed322e81e1433a)（主流 main）；Megatron KDA 相关代码在 **`upstream/dev`** 分支、merge commit [`788e9e1711e23bdfc39b27438e3786ba60fc232b`](https://github.com/NVIDIA/Megatron-LM/commit/788e9e1711e23bdfc39b27438e3786ba60fc232b)；vLLM 快照 [`0384aa7150c4c9778efca041ffd1beb3ad2bd694`](https://github.com/vllm-project/vllm/tree/0384aa7150c4c9778efca041ffd1beb3ad2bd694)；FlashKDA [`1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b`](https://github.com/MoonshotAI/FlashKDA/tree/1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b)；verl [`9ff05e323020a51ed3c22af9f67bdd2c6a6fa50a`](https://github.com/volcengine/verl/tree/9ff05e323020a51ed3c22af9f67bdd2c6a6fa50a)。**禁止引用 main 分支行号**，这里全部用 commit。

### 12.1 SGLang：真正的生产实现不止一个 kernel

SGLang 的模型层在 [`python/sglang/srt/models/kimi_linear.py`](https://github.com/sgl-project/sglang/blob/7399c2b5587e1559f3e5a26566ed322e81e1433a/python/sglang/srt/models/kimi_linear.py)（文件头注明改编自 vLLM commit `0384aa7…`，L3），backend 层在 [`python/sglang/srt/layers/attention/linear/kda_backend.py`](https://github.com/sgl-project/sglang/blob/7399c2b5587e1559f3e5a26566ed322e81e1433a/python/sglang/srt/layers/attention/linear/kda_backend.py)。

**类名先纠正一个常见错误**：backend 的真实类名是 **`KDAAttnBackend`**（[`kda_backend.py` L370](https://github.com/sgl-project/sglang/blob/7399c2b5587e1559f3e5a26566ed322e81e1433a/python/sglang/srt/layers/attention/linear/kda_backend.py#L370)，继承 `MambaAttnBackendBase`），不是 `KDAAttentionBackend`；kernel 调度器是 **`KDAKernelDispatcher`**（[L40](https://github.com/sgl-project/sglang/blob/7399c2b5587e1559f3e5a26566ed322e81e1433a/python/sglang/srt/layers/attention/linear/kda_backend.py#L40)）；昇腾 NPU 子类是 **`AscendKDAAttnBackend`**（[`python/sglang/srt/hardware_backend/npu/attention/ascend_kda_backend.py` L115](https://github.com/sgl-project/sglang/blob/7399c2b5587e1559f3e5a26566ed322e81e1433a/python/sglang/srt/hardware_backend/npu/attention/ascend_kda_backend.py#L115)）。实际的 KDA 分派发生在 [`attention_registry.py` L494–504](https://github.com/sgl-project/sglang/blob/7399c2b5587e1559f3e5a26566ed322e81e1433a/python/sglang/srt/layers/attention/attention_registry.py#L494-L504)：`kimi_linear_config(...) is not None` 时 CUDA 走 `KDAAttnBackend(runner)`、NPU 走 `AscendKDAAttnBackend(runner)`。`linear_attn_model_registry.py` 是给外部模型注册用的通用 API（其 L9–15 的 `KDAAttnBackend` 只出现在 docstring 示例中，不是 Kimi 的实际注册）。

模型层负责：tensor-parallel 投影、Q/K/V 与忘门/输出门 GEMM 的融合、低秩投影、ShortConv、`RadixLinearAttention`（[`python/sglang/srt/layers/radix_linear_attention.py`](https://github.com/sgl-project/sglang/blob/7399c2b5587e1559f3e5a26566ed322e81e1433a/python/sglang/srt/layers/radix_linear_attention.py) L38 起）与输出门/RMSNorm/row-parallel 输出投影。backend 层负责：sequence slot 与循环状态的对应、prefill/decode/verify 三套 kernel 的 dispatch、continuous batching、speculative decoding、NPU/CPU/CUDA 平台差异、全注意力 KV pool 与 KDA recurrent-state pool 的协同。

所以 SGLang 的核心工程结论是：

> **KDA serving 不是"把 FlashAttention 换成 `chunk_kda`"，而是需要一种新的混合缓存管理模型。** 一张图概括（自绘示意图，逻辑与 `kda_backend.py` 的调度分层一致）：

```mermaid
flowchart LR
    subgraph Model["KimiLinearForCausalLM (kimi_linear.py)"]
        KDA["KimiDeltaAttention (KDA 层)"] --> KDA2["..."]
        MLA["KimiMLAAttention (MLA 层)"]
    end
    subgraph Backend["KDAAttnBackend (+ KDAKernelDispatcher)"]
        P["prefill: chunk_kda / FlashKDA"]
        D["decode: fused_recurrent_kda"]
        V["verify: chunk/recurrent + accepted-token 语义"]
        DP{"阶段 + 硬件 + 平台 dispatch"}
        DP --> P & D & V
    end
    subgraph Pools["混合状态池"]
        R["recurrent state pool [N,HV,K,V] (fp32)"]
        C["conv state (3×short_conv)"]
        KV["MLA KV pool (paged)"]
    end
    KDA --> Backend
    Backend --> Pools
    Pools --> KDA
    MLA --> KV
    D == "inplace 写回" ==> R
```

> 这张图是本文按源码行为自绘的**逻辑示意图**（非论文图）：重点不是画每个模块，而是说明"三套状态、按阶段切换内核"这一与全注意力模型本质不同的结构。

### 12.2 vLLM：一个状态布局 bug 为什么能破坏长文本检索

SGLang 的 Kimi Linear 实现改编自 vLLM 快照 [`0384aa7`](https://github.com/vllm-project/vllm/tree/0384aa7150c4c9778efca041ffd1beb3ad2bd694)；该快照里 KDA 实现本体在 `vllm/model_executor/layers/kda.py`（`KimiDeltaAttention`），prefill 用 `chunk_kda`、decode 用 `fused_recurrent_kda`，并通过 Mamba-style state manager 管理三组卷积状态与一个循环状态。

2026 年 4 月的 [vLLM issue #41292](https://github.com/vllm-project/vllm/issues/41292)（已 closed，2026-04-30）是一个极为典型的"KDA 工程脆弱性"案例：状态布局从 `[N,H_V,K,V]` 切换为 `[N,H_V,V,K]`（PR #33291 的改动）后，chunk prefill 与后续路径没有保持一致，导致长上下文 needle retrieval 失败；短 prompt 仍能输出流畅文本，普通生成 smoke test 很难发现。报告称 v0.15.1 正常，而 v0.16.0 及之后的受测版本出现错误。

更隐蔽的是：**官方 checkpoint 的 KDA 是 $K=V=128$，`[K,V]` 与 `[V,K]` 的 shape 完全相同，错误 transpose 不会触发任何 shape assert，只会静默改变状态语义。**

> **对 KDA，shape 相同不等于 layout 相同。** FLA 因此把状态布局显式化为 `state_v_first` 参数（默认 `False` → `[N,HV,K,V]`），并废弃了旧名 `transpose_state_layout`；而对"官方 checkpoint 的 K=V=128"这一特殊情况，任何开发者在迁移/融合 kernel 时都必须用 $K\neq V$ 的小型测试做布局 oracle（§15）。

### 12.3 Megatron Core：基础 KDA 已有，完整 Kimi 仍在建设

**先说一个与常见介绍不同的事实**：在 `upstream/dev` 分支、merge commit `788e9e1`（[PR #6556](https://github.com/NVIDIA/Megatron-LM/pull/6556)，2026-08-25 合入，标题 *[Dev] Add Ling-V3 Tiny hybrid model training support*）上，KDA 位于 [`megatron/core/ssm/gated_delta_net/kda.py`](https://github.com/NVIDIA/Megatron-LM/blob/788e9e1711e23bdfc39b27438e3786ba60fc232b/megatron/core/ssm/gated_delta_net/kda.py)，类名 `KimiDeltaAttention`（L52，继承 `_GDNBase`）；**标量门（per-head）的对应实现不在 `gated_delta_product.py`，而在同目录 [`gdn.py`](https://github.com/NVIDIA/Megatron-LM/blob/788e9e1711e23bdfc39b27438e3786ba60fc232b/megatron/core/ssm/gated_delta_net/gdn.py)（`GatedDeltaNet`，L33）**，引用时务必区分：

- `kda.py::KimiDeltaAttention._compute_gates`（L167 起）：`raw_g = raw_g.reshape(batch, seq_len, num_key_heads, key_head_dim)`，得到的是**逐 channel** 的 `[B,T,H,K]` 衰弱信号（`beta` 按 key-head 数 sigmoid），真 KDA；
- `gdn.py::GatedDeltaNet._compute_gates`（L78 起）：`g = -A_log.exp() * softplus(alpha + dt_bias)`，`alpha` 是 **per-head 标量门**（GDN/GDP 家族）。

PR #6556 自身明确限定范围为 **Megatron Core training**（不含 checkpoint conversion 与 KDA inference；`KimiDeltaAttention.forward` 里还有"不支持 dynamic inference batching / inference"的断言）；它覆盖 packed THD、distributed tests 与 checkpoint-resume smoke test。**完整 Kimi 风格组件（low-rank F-decay projection、low-rank output-gate projection、per-layer NoPE MLA、hybrid spec 扩展）仍在 [PR #6877](https://github.com/NVIDIA/Megatron-LM/pull/6877)（检索时 open**）中。三个结论级别的区分（均以 2026-08-30 为准）：

| 表述 | 是否准确 |
| --- | --- |
| Megatron Core 已有基础 KDA hybrid 训练能力 | 是，已进入 `dev` |
| Megatron 已完整复现官方 Kimi Linear checkpoint 架构 | 否 |
| Megatron 已提供成熟 KDA 推理 | 否（基础 PR 明确不含 inference） |
| 可直接用 Megatron 转换并训练官方 48B-A3B checkpoint | 仍需检查后续 PR、转换工具与权重映射 |

### 12.4 verl：它不负责实现 KDA

verl（[`9ff05e32`](https://github.com/volcengine/verl/tree/9ff05e323020a51ed3c22af9f67bdd2c6a6fa50a)）是 RL 后训练编排框架，官方文档（[`docs/index.rst` L10](https://github.com/volcengine/verl/blob/9ff05e323020a51ed3c22af9f67bdd2c6a6fa50a/docs/index.rst#L10)）的定位是"解耦计算与数据依赖，允许无缝集成现有 LLM 基础设施（PyTorch FSDP、Megatron-LM、vLLM 和 SGLang）"；[`verl/workers/engine/base.py` 的 `BaseEngine`](https://github.com/volcengine/verl/blob/9ff05e323020a51ed3c22af9f67bdd2c6a6fa50a/verl/workers/engine/base.py#L30) 定义了引擎接口（训练侧 `BaseEngine` + `EngineRegistry`；rollout 侧由独立 worker 承载，如 `docs/index.rst` 列出的 `sglang_worker`）。verl 的设计意图是训练/推理后端均可替换，**但"rollout 可替换"的代码证据在 worker 侧而非 `base.py`，本文表述收窄为"verl 把训练与 rollout 后端设计为可替换引擎"**。**因此不存在一个独立的"verl KDA kernel"**。在 verl 里跑 Kimi Linear 至少需要：

1. **Actor 训练后端**能构建 KDA + MLA + MoE（`chunk_kda` 训练与对应 checkpoint 参数）；
2. **Rollout 后端**（vLLM/SGLang）能 serving 同一 checkpoint，并维护 conv/recurrent/KV 混合缓存、多轮与 speculative decoding；
3. **权重同步**两侧参数名与 TP/EP 切分一致。

截至检索日，端到端仍有明显缺口：SGLang 已有较完整的 rollout 实现；Megatron 完整 Kimi 风格训练支持仍在演进（§12.3）；官方 HF remote model 的 `KimiSparseMoeBlock` 在 training mode 会抛 `NotImplementedError`（[`modeling_kimi.py` L733](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct)，本地副本同），所以 `trust_remote_code=True` **不等于**开箱即用的 FSDP 训练。对 RL 还有一个额外风险：**rollout 用 recurrent decode、actor logprob 重算可能用 chunk training，两条路径在 beta sigmoid、state layout、gate lower bound、精度上不一致时，产生的不是普通推理误差，而是 rollout policy 与训练 policy 的数值漂移。**

### 12.5 FlashKDA：算法没有变，硬件映射变了

[FlashKDA](https://github.com/MoonshotAI/FlashKDA)（[`1ce47ea`](https://github.com/MoonshotAI/FlashKDA/tree/1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b)）是 Moonshot 的 CUTLASS 高性能 KDA forward kernel。两个事实口径（README 与 deep-dive 文档的措辞有张力，**本文以 README 为准：要求 SM90 及以上、CUDA 12.9+**，deep-dive 文档内部另有一处"SM80-only MMA path"的表述，两者存在措辞张力，引用时一律采用 README 的 SM90+ 口径）：

- **为什么 CHUNK=16**（deep-dive 文档 §1）：论文/通用 FLA 常用 64，FlashKDA v1 选 16。三个理由：`lower_bound=-5` 时 16 步累积衰减的数值范围恰好落在 bf16 表示内（**省去较大 chunk 所需的二级分块与额外 rescaling；gate/log 域运算本身仍保留**）；$16\times16$ 三角求逆便宜且可并入 Neumann 级数；所有 chunk 内数学能干净映射到 SM80 系 MMA。
- **两阶段 kernel 拆分**（deep-dive 文档 §2）：K1（token/chunk 并行：gate 激活 → L2 归一化 → 衰减 → `L`/`Mqk` 构造 → 矩阵求逆）与 K2（仅 head 并行：逐 chunk delta-rule 递推 → 输出投影 → 运行状态累积）。文档报告：早期单 fused kernel 中 K1 的 token 并行被 K2 的低并行度卡死，大量 SM 空闲；**拆成两个 kernel 至少带来 15% 端到端收益**，且两阶段各自可调优。**注意这是 FlashKDA 内部 deep-dive 报告给出的结论（未附完整复现脚本与机器配置基线），引用时应标注为"作者内部报告"，不应作跨硬件外推。**
- **Benchmark 口径**（[`BENCHMARK_GB200.md`](https://github.com/MoonshotAI/FlashKDA/blob/1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b/BENCHMARK_GB200.md)、[`BENCHMARK_H20.md`](https://github.com/MoonshotAI/FlashKDA/blob/1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b/BENCHMARK_H20.md)，2026-04/05 生成）：相对 `fla_chunk_kda`（配置 `use_gate_in_kernel=True, use_qk_l2norm_in_kernel=True, use_beta_sigmoid_in_kernel=True, lower_bound=-5`、旧参数名 `transpose_state_layout=True`），T=8192、D=128 下 GB200 为 **1.70–3.27×**（H=96/64），H20 为 **1.85–2.31×**。它只说明：**对满足支持条件的 KDA core，专门的 CUTLASS 实现显著快于当时的通用 Triton 路径**，不等价于 Kimi Linear 快 3 倍、端到端 serving 快 3 倍、训练自动快 3 倍，或任意 head_dim 有同等收益。


## 十三、复杂度、表达能力与数值稳定性


### 13.1 三个维度的权衡表

| 维度 | KDA 的优点 | 边界与代价 |
| --- | --- | --- |
| 训练计算 | 对 $T$ 线性（$6Td_h^2+3TCd_h+TC^2$） | $Td_h^2$ 项：head_dim 翻倍成本 ×4；chunk 内三角求解是额外开销 |
| decode | 每 token $O(d_kd_v)$，不随上下文增长 | 短上下文 + 高并发 + FP32 state 时，固定状态读写也是显著开销 |
| 表达能力 | 定向覆盖旧映射、逐 channel 时间尺度、状态大小恒定 | 精确 verbatim retrieval 仍是瓶颈；相近 key 共享状态方向；Kimi Linear 为缓解该边界而混合 MLA |
| 数值稳定性 | log 域 gate、`exp2`、`safe_gate` 下界、QK L2norm | 多步衰减连乘会在 bf16 下溢出；chunk/recurrent 路径数值不严格一致；`exp(-g/2)` 类写法不能用（draft-2 口述反例，非 FLA 实现） |

### 13.2 为什么这些工程选择是"演进而来"的

把三章的内容回看一遍，几乎所有生产决策都是"baseline → 中间 → 最终"的形状：

1. **忘门数值**（draft-2 口述反例：其"朴素 `exp(-g/2)` 写法"并非 FLA 实现，被 FLA 以相对 gate 差实现取代，§8.4）：相对偏移 + clamp（有 loss spike）→ 二次分块回退 CUDA cores → 绑定 key（架构级）+ log 域 + `safe_gate`（§8.4）；
2. **并行化**：逐 token 递归（利用率低）→ chunk 内核（§8，数学等价 + 块间串行）→ CUTLASS 两阶段内核（§12.5，K1/K2 拆分让 SM 不再空闲）；
3. **状态布局**：`[K,V]` 默认 → `state_v_first` 显式参数（`[V,K]`）→ 旧名废弃（§11.4）；而 vLLM 的教训是"只改布局不加测试"会静默破坏语义（§12.2）；
4. **门控参数化**：GDN 标量 → KDA 逐通道低秩（`f_a/f_b`）→ Mega、FLA 的 kernel 内融合（`use_gate_in_kernel`），每一步都在"表达能力 / 参数效率 / kernel 效率"三者之间重新切分。


## 十四、常见误区与失败模式


1. **KDA 是一种低秩 Softmax。** 不是。KDA 没有保留逐 query 对历史 token 的 softmax 概率分布，它是一个递归状态机。"低秩"只是它转移矩阵的结构性质（DPLR），不是对 Softmax 的近似。
2. **KDA 与 Kimi Linear 是同一个概念。** 不是：KDA = token-mixing 算子；Kimi Linear = KDA + MLA + ShortConv + 输出门 + MoE + 混合缓存。评估单层算子不能直接推断完整模型质量。
3. **线性复杂度 = 所有长度都更快。** 短序列下投影、ShortConv、gate 投影、$d_k d_v$ 状态更新、MoE、kernel launch 都可能占主导；论文自己的曲线也显示显著优势主要在长上下文（§6.1）。
4. **tensor shape 对了 state layout 就对了。** `[K,V]` 与 `[V,K]` 在 $K=V=128$ 时 shape 相同；必须测数值语义（§12.2，vLLM 反例）。
5. **忘记或重复执行 beta sigmoid。** FLA 0.5.0 曾存在上层传 `use_beta_sigmoid_in_kernel=True` 但旧接口通过 `**kwargs` 静默吞掉未知参数、导致 beta 没做 sigmoid 却不报错不警告的情况（[issue #1119](https://github.com/fla-org/flash-linear-attention/issues/1119)），**该问题属于 0.5.0 旧版**；当前 commit（`6ec09887`）已把 `A_log/dt_bias/chunk_size` 显式处理并加校验，但"未知语义参数被静默忽略"这一类风险仍值得主动防御：启动时检查签名、更重要的是做数值 oracle（§15）。
6. **只测试短 prompt 的语言流畅度。** 状态错误时模型仍可能生成语法流畅的通用回答；应测长 needle retrieval、cache resume、chunk boundary、多 sequence 隔离、prefix 后继续 decode。
7. **把 chunk kernel benchmark 当成端到端吞吐。** 完整模型还包括 QKV/gate 投影、MoE、TP/EP 通信、sampling、scheduler、MLA 层、状态池访问；算子快 2 倍不代表模型快 2 倍（§12.5 的 15% 与 1.7–3.27× 都是 kernel 级口径）。
8. **纯 KDA 必然适合所有长上下文任务。** 固定状态适合状态压缩，不保证精确随机访问；Kimi 团队自己混合 MLA 恰好说明纯线性注意力的边界（§5.1）。
9. **NoPE 是 KDA 的免费附赠。** 是完整 hybrid 架构与训练 recipe 的选择，不是从 KDA 公式自动推出的普遍结论（§5.2）。
10. **框架显示"支持 KDA"就等于支持官方 Kimi checkpoint。** 至少还应核对：low-rank forget gate、low-rank output gate、ShortConv、NoPE MLA、MoE routing、checkpoint 参数名、state layout、prefill/decode parity、inference/conversion 支持。Megatron 的现状（§12.3）正好展示了"基础 KDA 支持"与"完整 Kimi 架构支持"的差别。


## 十五、最小正确性测试集与选型建议


### 15.1 建议的最小测试集

1. **单步公式 oracle（FP64/FP32）**：`S_t = D_t S_{t-1} + β_t k_t (v_t - (D_t S_{t-1})^T k_t)^T` 逐项比较。
2. **Recurrent 与 Chunk parity**：`naive_recurrent` / `fused_recurrent` / `chunk` 三者比较所有 token 输出与最终状态。
3. **任意分块不变性**：全量 prefill vs 前 100 + 后续 vs 前 64 + 后续 vs 随机多段，最终状态与 logits 一致。
4. **State-layout oracle**：显式构造 $K\neq V$（如 $K=3,V=5$），让 `[K,V]` 与 `[V,K]` 无法靠相同 shape 蒙混过关。
5. **Variable-length 隔离**：多序列打包验证 padding 不更新状态、序列间不污染、`cu_seqlens` 边界正确。
6. **Cache resume**：一次性 prompt+continuation vs 先 prompt 存状态再 continuation。
7. **精度漂移**：FP32 state / BF16 state / FP32 累计 + BF16 存储，看长序列最终状态与 retrieval，单步误差说明不了问题。
8. **Speculative verify**：rejected token 不永久写入主状态、accepted 数与最终状态对应。

### 15.2 选型建议

| 场景 | 建议 |
| --- | --- |
| 学习 KDA 数学 | 先读 FLA `naive.py`，再对照论文 Eq. 1（本文 §2–§8 的顺序即来源于此） |
| 手写研究原型 | 纯 PyTorch recurrent oracle 起步，不要一开始写 Triton |
| 训练新 KDA 模型 | 固定 commit 的 FLA `chunk_kda`，先建 recurrent/chunk parity |
| 复现官方 Kimi layer | 对照官方 `modeling_kimi.py`，不要只实例化通用 FLA 默认层（§9.2 的接线差异） |
| 部署官方 checkpoint | 优先经过该 checkpoint 验证的 SGLang/vLLM 版本 |
| Hopper/Blackwell 长 prefill | 满足形状与环境条件时评估 FlashKDA，同时保留 FLA fallback |
| 短上下文模型 | 不应只因 $O(T)$ 就采用 KDA；FlashAttention 通常更简单成熟 |
| 精确长距离检索 | 用 KDA/MLA hybrid，而不是纯 KDA |
| Megatron 大规模预训练 | 视为快速演进中的能力：锁定 `dev` commit 并核对待合入 PR |
| verl RL | 先验证 actor 训练后端，再验证 SGLang/vLLM rollout；verl 本身不会补齐模型支持 |
| 高并发 serving | 重点测每 sequence 固定 state、state dtype 与 state-pool 的显存占用 |
| 修改 kernel | 必须跑长 retrieval、state layout、cache resume 与分块不变性测试 |

一个务实判断：**KDA 最有吸引力的场景，不是"所有 Transformer 都换掉 Softmax"，而是长生成、长轨迹、状态持续更新、KV cache 已成为主要瓶颈、且可以接受混合少量全注意力层来维持精确检索的模型。**


## 十六、全文逻辑链


把全文压缩成一条因果链：**Softmax 注意力的精确访问随上下文增长 → 线性注意力用固定矩阵压缩历史（但纯加法只会叠加）→ delta rule 按预测误差定向修改（但不会主动遗忘）→ GDN 加标量遗忘（但一个 head 只能有一个寿命）→ KDA 逐通道遗忘 + delta 修正（但并行结构被破坏）→ 转移矩阵是受约束 DPLR（$a_t=\beta_tk_t$, $b_t=k_t\odot\alpha_t$）→ 一套 WY/UT 即可 chunkwise 化（块间递归、块内并行；省 3 次矩阵乘与二次分块）→ 生产实现四层：参数化（低秩忘门/Sigmoid 输出门/ShortConv）、内核（chunk/recurrent/FlashKDA）、状态管理（conv + recurrent + MLA KV/混合池/布局）、框架集成（SGLang/vLLM/Megatron/verl）→ 固定状态仍不擅长精确任意检索 → Kimi Linear = 3:1 KDA/MLA hybrid。**

最终最重要的认识不是"KDA 把复杂度从二次变成线性"，而是：

> **KDA 把注意力问题重新表述成了在线记忆管理问题：模型不再保留全部历史，而是学习每一步应该遗忘哪些状态方向、修正哪一条 key-value 映射，以及何时借助周期性的全注意力层恢复精确访问能力。**
---

## 附录 A：DeltaNet 的记忆更新 = 在线梯度下降

这是理解 KDA 谱系最深的一个视角：**记忆矩阵 $S$ 不是一个被动的缓存，而是一个在推理时不断被"训练"的小模型，delta rule 就是对它做的一步梯度下降。** 全文用本文约定（$S\in\mathbb{R}^{d_k\times d_v}$，$\hat v=S^\top k$）。

### A.1 把记忆看成一个待训练的模型

线性注意力的功能是"给一个 key，回忆出对应的 value"：$\hat v=S^\top k$。这在形式上就是一个**无偏置的单层线性网络**（$S$ 是权重）。于是"记住 $(k_t,v_t)$"可以表述为一个学习目标：

$$
\mathcal{L}_t(S)=\frac12\left\|S^\top k_t-v_t\right\|^2 .
$$

每来一个新 token，就产生一条新的训练样本 $(k_t,v_t)$。

### A.2 对这个损失做一步梯度下降

求梯度（对矩阵 $S$）：

$$
\nabla_S\mathcal{L}_t=k_t\left(S^\top k_t-v_t\right)^\top .
$$

注意这又是一个**外积**：在本文约定（$S\in\mathbb{R}^{d_k\times d_v}$）下是"地址 $k_t$ 乘误差行向量" $k_t(S^\top k_t-v_t)^\top$。以学习率 $\beta_t$ 做一步随机梯度下降：

$$
S_t=S_{t-1}-\beta_t k_t\left(S_{t-1}^\top k_t-v_t\right)^\top
=S_{t-1}+\beta_tk_t\left(v_t-S_{t-1}^\top k_t\right)^\top.
$$

展开合并：

$$
\boxed{\;S_t=\left(I-\beta_tk_tk_t^\top\right)S_{t-1}+\beta_tk_tv_t^\top\;}
$$

**这正是 delta rule**（Widrow-Hoff LMS / 感知机学习规则的连续版本，DeltaNet 的名字正来源于此）。"先擦除再写入"与"对记忆做一步梯度下降"是同一件事的两种读法：

| 梯度下降视角 | 记忆操作视角 |
| --- | --- |
| $S\leftarrow S-\beta_t\,k_t\,(S^\top k_t-v_t)^\top$ | 先擦掉 $k_t$ 槽位的旧值，再按 $\beta_t$ 写入 $v_t$ |

**逐项语义**：更新量正比于 $v_t-S_{t-1}^\top k_t$（新信息与已有记忆的差距）。记忆里已存对时更新量 $\approx 0$（不做无用功）；存的是过时值则精确修正差额。对比普通线性注意力的"盲写"（同一个 key 出现 10 次就把 10 个 value 全堆在一起），delta 版会收敛到最新值，这就是它在关联回忆类任务上大幅胜出的根本原因。$\beta_t\in(0,1]$ 是**数据依赖的学习率**：大 $\beta$ 快速改写，小 $\beta$ 几乎不动记忆；$\beta_t\to1$ 对应完整的正交投影擦除（附录 B）。

### A.3 谱系：内层学习问题的不同设定

Kimi Linear 报告 §6.3 的 Table 7 用 TTT（Test-Time Training）框架把整个谱系放在一张表里（下面摘录与本文相关的行，记号已统一为本文件约定；$\tilde S$ 表示衰减后的状态）：

| 架构 | 内层目标函数 $\mathcal{L}$（示意） | 更新规则 |
| --- | --- | --- |
| 线性注意力 | $-\langle S_{t-1}^\top k_t, v_t\rangle$ | $S_t=S_{t-1}+k_tv_t^\top$ |
| RetNet | $-\beta_t\langle S_{t-1}^\top k_t,v_t\rangle+\frac12\|\sqrt{1-\alpha}S_{t-1}\|_F^2$ | $S_t=\alpha S_{t-1}+\beta_tk_tv_t^\top$ |
| Mamba2 | 同上（$\alpha_t$ 逐头） | 同上（逐头 $\alpha_t$） |
| GLA | $-\langle S_{t-1}^\top k_t,v_t\rangle+\frac12\|\sqrt{\operatorname{Diag}(1-\alpha_t)}S_{t-1}\|_F^2$ | $S_t=\operatorname{Diag}(\alpha_t)S_{t-1}+k_tv_t^\top$ |
| GDN | $\frac{\beta_t}{2}\|\tilde S_{t-1}^\top k_t-v_t\|^2$ | $S_t=(I-\beta_tk_tk_t^\top)\alpha_tS_{t-1}+\beta_tk_tv_t^\top$ |
| **KDA** | $\frac{\beta_t}{2}\|\tilde S_{t-1}^\top k_t-v_t\|^2$（$\tilde S$ 逐通道衰减） | $S_t=(I-\beta_tk_tk_t^\top)\operatorname{Diag}(\alpha_t)S_{t-1}+\beta_tk_tv_t^\top$ |

其中"权重衰减"的解读很有意思：优化里 weight decay 对应损失中的正则项 $\frac{\lambda}{2}\|S\|^2$，作用是让不再被更新的权重逐渐归零；映射到记忆语境，**长期没有被新 key 触碰的记忆会慢慢淡忘**，这正是遗忘门的功能。于是：

- Gated DeltaNet = 内层做"SGD + 全局统一的 weight decay"；
- KDA = 内层做"SGD + **每个特征通道独立的 weight decay**"，不同类型的信息可以有不同遗忘半衰期。

**一条工程推论**：梯度下降是串行的（$S_t$ 依赖 $S_{t-1}$），看起来必须逐 token 计算；DeltaNet 系列能实用化的关键突破，是发现这些秩 1 更新可以用 **WY 表示**改写，推导出**分块并行（chunkwise）算法**，块内矩阵乘并行、块间串行传递状态（§8 与附录 D）。KDA 的贡献之一是把这套并行化技术扩展到"对角门控 + delta rule"的受约束 DPLR 转移矩阵上。同样的视角还统一了 TTT、Titans 等近期工作：区别只在于内层模型的结构（线性/MLP）、损失与优化器选择。**本文只把这一句谱系关系写在这里作为知识边界，不再展开。**

## 附录 B：$I-kk^\top$ 是向 $k^\perp$ 的正交投影

设 $k$ 为单位向量（$\|k\|=1$）。任何向量 $v$ 可分解为沿 $k$ 的分量与垂直分量：

$$
v=\underbrace{(kk^\top)v}_{\text{沿 }k\text{ 的分量}}+\underbrace{(I-kk^\top)v}_{\text{垂直于 }k\text{ 的分量}} .
$$

两个关键验证：

1. **结果确实垂直于 $k$**：$k^\top\left[(I-kk^\top)v\right]=k^\top v-\underbrace{(k^\top k)}_{=1}k^\top v=0$；
2. **投影两次等于一次**（投影算子的定义性质 $P^2=P$）：$(I-kk^\top)^2=I-2kk^\top+k\underbrace{(k^\top k)}_{=1}k^\top=I-kk^\top$。

在 KDA 中这个投影的角色正是"先擦后写"：delta rule 项 $(I-\beta_tk_tk_t^\top)S_{t-1}$ **左乘**状态矩阵（本文约定 $S\in\mathbb{R}^{d_k\times d_v}$，投影作用在 $S$ 的 key 索引维度上；由于该矩阵对称，也可理解为对 $S$ 各行的 $k$ 方向分量做擦除）。它保证：（1）**精准擦除**，只删除与当前 key $k_t$ 相关的记忆方向，不误伤存储在垂直方向的信息；（2）**无损保留**，与 $k_t$ 垂直的存量经过矩阵后完全不变。

两个容易混淆的补注：

- **$\beta_t<1$ 时 $(I-\beta kk^\top)$ 不是投影**（$P^2\neq P$），它在 $k$ 方向上是"收缩"（擦除力度 $\beta_t$，$\beta_t=1$ 才彻底替换）。$\beta_t$ 扮演"擦除力度"而非"投影"的参数；
- 本文的"读一读再写"次序与"投影后写入"次序等价性：$S_t=(I-\beta kk^\top)S+\beta kv^\top$ 的转置读出给出 $S_t^\top k=(1-\beta)S^\top k+\beta v$，正是"读出旧值 $\to$ 插值到 $v$"。

## 附录 C：$kk^\top$ 是什么，外积（Outer Product）

若 $k$ 是 $d$ 维**列向量**，它和自己有两种乘法，结果完全不同：

| 写法 | 形状 | 结果 | 名字 |
| --- | --- | --- | --- |
| $k^\top k$ | $(1\times d)(d\times 1)$ | 标量 | 内积/点积 |
| $kk^\top$ | $(d\times 1)(1\times d)$ | $d\times d$ 矩阵 | 外积 |

$k^\top k=\|k\|^2$ 是长度平方；$kk^\top$ 生成一个**秩 1**、**对称**、$\|k\|=1$ 时**幂等**（$(kk^\top)^2=kk^\top$）的矩阵。关键是把 $kk^\top$ 看成"作用在其他向量上的算子"：$(kk^\top)v=k(k^\top v)$，先算内积（标量），再乘 $k$，所以它是"投影到 $k$ 方向"的算子。$I-kk^\top$ 正好相反："删掉 $k$ 分量，剩下垂直部分"。

**两个外积的用途要分清**：

- 写入记忆的是 $k_t v_t^\top$（与本文 $S_t=\sum k_iv_i^\top$ 的约定一致；$v$、$k$ 不同向量）：制造一条"键值绑定"，以 $k$ 为地址把 $v$ 存进去；之后查询 $q$ 经 $S^\top q$ 读出时，$q$ 与地址 $k$ 内积越大，读出的 $v$ 越完整；
- delta rule 用的是 $k_t k_t^\top$（地址对地址）：它不存东西，而是构造一个"只认这个地址的探测器/擦除器"，这就是 $(I-\beta kk^\top)$ 定点清空旧记忆的数学基础（附录 B）。KDA 的更新 $\beta_tk_te_t^\top$ 则是第三种：**地址 $k_t$ × 误差向量 $e_t$**，写入的是"当前状态没有解释的部分"。

> DPLR 中的 "Low-Rank" 正是来源于此：$kk^\top$ 的秩为 1，$D_t-\beta_tk_t(D_tk_t)^\top$ 是"对角矩阵 + 秩 1 修正"（§7.2）。

## 附录 D：Chunkwise 并行计算原理详解（完整归纳推导）

### D.1 问题

KDA 定义是逐 token 递推（§3.3）。它的两个致命工程问题是"串行依赖"（$S_t$ 依赖 $S_{t-1}$，$T$ 步串行）与"计算形态差"（每步是向量级 rank-1 更新，算术强度低，喂不饱 TensorCore）。chunkwise 的目标：**数学完全等价的前提下重排计算顺序**。

### D.2 热身：没有 delta 修正时怎么并行

先看简化版（GLA 风格的门控线性注意力）：$S_i=\operatorname{Diag}(a_i)S_{i-1}+\beta_ik_iv_i^\top$。设块首状态 $S_0$，块内位置 $i=1,\dots,C$。定义逐通道累积衰减

$$
\gamma_i=a_i\odot a_{i-1}\odot\cdots\odot a_1\in(0,1)^{d_k},
$$

因为衰减是对角的：$\operatorname{Diag}(a_i)\cdots\operatorname{Diag}(a_1)=\operatorname{Diag}(\gamma_i)$。展开递推：

$$
S_i=\operatorname{Diag}(\gamma_i)S_0+\sum_{j\le i}\operatorname{Diag}\!\left(\frac{\gamma_i}{\gamma_j}\right)k_j\,(\beta_jv_j)^\top .
$$

**关键代数技巧**：$\operatorname{Diag}(\gamma_i/\gamma_j)k_j=\gamma_i\odot(k_j\oslash\gamma_j)$（逐通道）。定义

$$
\tilde q_i=q_i\odot\gamma_i,\qquad\tilde k_i=k_i\odot\gamma_i,\qquad\hat k_j=k_j\oslash\gamma_j .
$$

输出 $o_i=S_i^\top q_i$ 变成 $o_i^\top=\tilde q_i^\top S_0+\sum_{j\le i}\langle\tilde q_i,\hat k_j\rangle\,\beta_jv_j^\top$。堆成矩阵（$\tilde Q,\hat K\in\mathbb{R}^{C\times d_k}$，$V\in\mathbb{R}^{C\times d_v}$）：

$$
O=\tilde Q S_0+\operatorname{tril}\!\left(\tilde Q\hat K^\top\right)\operatorname{Diag}(\beta)V .
$$

这已经是"注意力"的样子了：$\tilde Q\hat K^\top$ 是 $C\times C$ 的块内分数矩阵，tril 保因果，第一项是块外全部历史经 $S_0$ 传入的贡献。所有位置一次算完，没有串行。

### D.3 核心难点：delta 修正让转移矩阵不再对角

加回 $(I-\beta_ik_ik_i^\top)$ 后，从 $S_0$ 到 $S_i$ 的转移矩阵是一串**非对角矩阵的乘积**。但每个因子都是"对角 + 秩 1"（§7.2），$i$ 个秩 1 修正的乘积总修正秩最多为 $i$，**WY 表示**保证可以写成"对角衰减 + 低秩修正"的紧凑形式。具体地，**归纳假设**：存在待定向量 $u_1,\dots,u_i\in\mathbb{R}^{d_v}$，使

$$
S_i=\operatorname{Diag}(\gamma_i)\left(S_0+\sum_{j\le i}\hat k_j\,u_j^\top\right).
$$

### D.4 归纳推导：$u_i$ 满足一个下三角线性方程组

设 $S_{i-1}=\operatorname{Diag}(\gamma_{i-1})(S_0+\sum_{j<i}\hat k_ju_j^\top)$ 成立。先作用衰减（$a_i\odot\gamma_{i-1}=\gamma_i$）：

$$
\operatorname{Diag}(a_i)S_{i-1}=\operatorname{Diag}(\gamma_i)\left(S_0+\sum_{j<i}\hat k_ju_j^\top\right)\;\triangleq\;\bar S .
$$

再作用 delta 修正并写入：

$$
S_i=\bar S-\beta_ik_i\left(k_i^\top\bar S\right)+\beta_ik_iv_i^\top=\bar S+k_i\left[\beta_i\left(v_i-\bar S^\top k_i\right)\right]^\top .
$$

计算读出项 $\bar S^\top k_i$。注意 $k_i^\top\operatorname{Diag}(\gamma_i)=\tilde k_i^\top$，所以：

$$
\bar S^\top k_i=S_0^\top\tilde k_i+\sum_{j<i}\left\langle\tilde k_i,\hat k_j\right\rangle u_j .
$$

再利用 $k_i=\operatorname{Diag}(\gamma_i)\hat k_i$，把新增的 rank-1 项也吸收进 $\operatorname{Diag}(\gamma_i)(\cdots)$ 括号（$k_i=\operatorname{Diag}(\gamma_i)\hat k_i$），保持 $S_i$ 的形式不变，得到新的写入向量：

$$
\boxed{\;u_i=\beta_i\left(v_i-S_0^\top\tilde k_i-\sum_{j<i}\left\langle\tilde k_i,\hat k_j\right\rangle u_j\right)\;}
$$

物理意义：$u_i$ 是位置 $i$ 的**真实预测误差写入量**，从原始 $v_i$ 中减去块首旧状态 $S_0$ 的预测（第二项），再减去**块内更早写入对当前预测的贡献**（第三项，编码了块内 delta 更新之间的串行依赖）。

**矩阵化**。定义严格下三角 $A\in\mathbb{R}^{C\times C}$ 与右端项 $B$：

$$
A_{ij}=\beta_i\left\langle\tilde k_i,\hat k_j\right\rangle\ (j<i),\qquad B_i=\beta_i\left(v_i-S_0^\top\tilde k_i\right),
$$

则递推恰好是 $(I+A)U=B$，即 **UT transform**：$U=(I+A)^{-1}B$。$I+A$ 是单位对角下三角阵，永远可逆（前向替换，代价 $O(C^2d_v)$；严格下三角幂零 $A^C=0$，逆可写成有限级数，支持分块高度并行）；构造 $A$ 本身是 $C\times C$ 的 GEMM（$(\operatorname{Diag}(\beta)\tilde K)\hat K^\top$ 取严格下三角），完全 TensorCore 友好。

**直觉总结**：块内 $C$ 次串行 rank-1 更新的相互干扰全部浓缩进这个 $C\times C$ 小三角阵；解一次三角方程组就等价于按序做完 $C$ 次 delta 更新。

### D.5 拿到 $U$ 之后：三个 GEMM 收尾

- **块内输出**（每个位置 = 历史贡献 + 块内因果注意力）：$O=\underbrace{\tilde Q\,S_0}_{\text{块外历史}}+\underbrace{\operatorname{tril}\!\left(\tilde Q\hat K^\top\right)U}_{\text{块内因果部分}}$；
- **跨块状态传播**（$C$ 次衰减 + $C$ 次 rank-1 写入一次完成）：$S_C=\operatorname{Diag}(\gamma_C)\left(S_0+\hat K^\top U\right)$；
- 然后 $S_C$ 作为下一块的 $S_0$，块间循环 $T/C$ 次。

**复杂度**：每块计算全部是稠密矩阵乘（构造 $A$ 与 $\tilde Q\hat K^\top$ 是 $O(C^2d)$，与 $S$ 交互是 $O(Cd_kd_v)$，三角求解 $O(C^2d_v)$），整个序列约 $O(TCd+Td^2)$，对 $T$ 严格线性，与逐 token 递推相比只是多了 $TCd$ 项，换来块内 64 token 完全并行 + 全部热点 GEMM + 状态每 $C$ 步才读写一次。

### D.6 数值稳定性：为什么教学版不能直接上生产

推导中出现 $\hat k=k/\gamma$；$\gamma_i$ 是 $(0,1)$ 内数值的连乘，会指数级衰减到接近 0，**直接物化除法会上溢**。真正需要的量都是 $\langle\tilde k_i,\hat k_j\rangle$ 形式，衰减因子是 $j\le i$ 时的 $\gamma_i/\gamma_j=\exp(G_i-G_j)$，在对数域 $G_i-G_j\le 0$，**指数永远不会爆炸**。真实 kernel 从不单独物化 $k/\gamma$，而是在计算注意力分数时以对数差融合衰减（§8.4）。一般 DPLR（$a,b$ 自由）做 chunkwise 时需要对 $u,w$ 各做一套 WY/UT，通常还要二次分块控制 $1/\Gamma$ 范围，KDA 把两个低秩向量都绑定为 $k_t$，使本附录只需一套 $A$ 与一次三角求解（§7.3）。"一图流"五步总结：

1. 预处理：$G=\operatorname{cumsum}(\log a)$，$\tilde Q=Q\odot e^G$，$\tilde K=K\odot e^G$，$\hat K=K\odot e^{-G}$；
2. UT transform：$A=\operatorname{stril}\!\left(\operatorname{Diag}(\beta)\tilde K\hat K^\top\right)$，$U=(I+A)^{-1}\operatorname{Diag}(\beta)\left(V-\tilde KS_0\right)$；
3. 块内输出：$O=\tilde QS_0+\operatorname{tril}\!\left(\tilde Q\hat K^\top\right)U$；
4. 状态更新：$S_C=\operatorname{Diag}(e^{G_C})\left(S_0+\hat K^\top U\right)$；
5. $S_0\leftarrow S_C$，进入下一块。

## 附录 E：FLA `chunk_kda` / `fused_recurrent_kda` API 速查

> 以下 API 签名与说明以 commit `6ec09887`（[`fla/ops/kda/chunk.py`](https://github.com/fla-org/flash-linear-attention/blob/6ec09887145d2f51bf203304777770b041374b65/fla/ops/kda/chunk.py) L176–200、[`fused_recurrent.py`](https://github.com/fla-org/flash-linear-attention/blob/6ec09887145d2f51bf203304777770b041374b65/fla/ops/kda/fused_recurrent.py) L339 起）为准；注意 `A_log`、`dt_bias` 经 `**kwargs` 传入并在 `chunk.py` L387–391 用 **`kwargs.get` 取出后显式校验**（`A_log` 未提供且未设 `lower_bound` 时报错）；`chunk_size` 则是 **`kwargs.pop("chunk_size", 64)`**（L393）并在 L394–395 校验取值。也就是说：**A_log/dt_bias 是"get + 校验"、chunk_size 是"pop + 校验"**，相比 0.5.0 时代"未知 kwargs 静默吞掉"的行为已显式化（见 §14 误区 5），但这三个参数**仍不出现在 `chunk_kda` 的签名里**，因此直接 `inspect.signature(chunk_kda)` 并不能确认它们存在。

### E.1 `chunk_kda`（训练/长 prefill）

```text
o, final_state = chunk_kda(
    q, k, v, g, beta,
    scale=None,                       # 默认 K ** -0.5
    initial_state=None,               # [N, HV, K, V] 必须 fp32
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,           # 需 use_beta_sigmoid_in_kernel=True；β ← 2·sigmoid
    safe_gate=False, lower_bound=None,  # safe_gate 时 lower_bound ∈ [-5, 0)
    disable_recompute=False,          # True: 前向保留中间量、反传不重算
    return_intermediate_states=False, # 仅 torch.inference_mode()；返回 (o, final_state, h)
    state_v_first=False,              # 状态布局 [V,K] vs 默认 [K,V]
    cu_seqlens=None, cu_seqlens_cpu=None,
    cp_context=None,                  # 此时禁用 initial_state/output_final_state
    A_log=None, dt_bias=None,         # A_log 在 use_gate_in_kernel=True 且未设 lower_bound 时必传；dt_bias 可选
    chunk_size=64,                    # 仅 32/64
)
```

主要标志语义见 [§11.4 关键约定](#114-关键约定)。

### E.2 `fused_recurrent_kda`（decode）

```text
o, final_state = fused_recurrent_kda(
    q, k, v, g, beta,
    A_log=None, dt_bias=None, scale=None,
    initial_state=None, output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False, lower_bound=None,
    state_v_first=False, cu_seqlens=None,
)
```

kernel 层另有 `inplace_final_state`（默认 `True`：终态原地写回 `initial_state`）、`ssm_state_indices`（continuous batching 定位各 sequence 的状态槽）、`num_accepted_tokens`（speculative decoding 接受 token 数），公共 API 不暴露 `inplace_final_state`，vLLM 集成走单独封装。

### E.3 示例（GVA / 变长 / safe_gate → FlashKDA）

> **维度提醒**：FLA 的 KDA 内核目前只支持 `K <= 256`（[`chunk.py` L409](https://github.com/fla-org/flash-linear-attention/blob/6ec09887145d2f51bf203304777770b041374b65/fla/ops/kda/chunk.py#L409)），下面的示例统一取 `K = V = 128`（也是官方 checkpoint 的配置；FlashKDA 后端进一步要求 `K = V = 128`）。

```python
import torch
from fla.ops.kda import chunk_kda, fused_recurrent_kda

B, T, H, K, V = 4, 2048, 4, 128, 128
q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda')      # use_beta_sigmoid_in_kernel=True 时为 logits
g = torch.rand(B, T, H, K, dtype=torch.bfloat16, device='cuda')       # use_gate_in_kernel=True 时为 raw preactivation
h0 = torch.randn(B, H, K, V, dtype=torch.float32, device='cuda')
A_log = torch.randn(H, dtype=torch.float32, device='cuda')
dt_bias = torch.randn(H * K, dtype=torch.float32, device='cuda')

# 基础（无 GVA，HV == H，内核内算门控与 beta sigmoid）
o, ht = chunk_kda(q, k, v, g, beta, A_log=A_log, dt_bias=dt_bias,
                  use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                  use_beta_sigmoid_in_kernel=True,
                  initial_state=h0, output_final_state=True)
# o: [B, T, H, V], ht: [B, H, K, V]

# GVA（HV > H）：q/k 仍是 H 个 head，v/g/beta 扩展为 HV 个
HV = 8
q2 = q  # [B, T, H, K]
v2 = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device='cuda')
g2 = torch.rand(B, T, HV, K, dtype=torch.bfloat16, device='cuda')
b2 = torch.rand(B, T, HV, dtype=torch.bfloat16, device='cuda')
h02 = torch.randn(B, HV, K, V, dtype=torch.float32, device='cuda')
A2 = torch.randn(HV, dtype=torch.float32, device='cuda')
d2 = torch.randn(HV * K, dtype=torch.float32, device='cuda')
o2, ht2 = chunk_kda(q2, k, v2, g2, b2, A_log=A2, dt_bias=d2,
                    use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                    initial_state=h02, output_final_state=True)

# 变长（B=1 拍平 + cu_seqlens）
from einops import rearrange
qf, kf, vf, bf, gf = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta, g))
cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
o3, ht3 = chunk_kda(qf, kf, vf, gf, bf, A_log=A_log, dt_bias=dt_bias,
                    use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                    initial_state=h0, output_final_state=True, cu_seqlens=cu_seqlens)

# FlashKDA 高吞吐模式：需满足全部条件（推理模式（grad 关闭） + bf16 + K=V=128 + 无 GVA（H==HV）
# + safe_gate=True + state_v_first=True + cp_context=None + return_intermediate_states=False
# + 三个 *_in_kernel 全开 + 环境变量 FLA_FLASH_KDA!=0），满足时自动分派到 CUTLASS 内核。
with torch.inference_mode():
    # state_v_first=True 时状态布局为 [B, H, V, K]，与默认 [K, V] 不同（K=V=128 时 shape 相同但语义不同）：
    h0_vk = h0.transpose(-1, -2).contiguous()   # [B, H, V, K]
    o4, ht4 = chunk_kda(q, k, v, g, beta, A_log=A_log, dt_bias=dt_bias,
                        use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                        use_beta_sigmoid_in_kernel=True,
                        safe_gate=True, lower_bound=-5.0,
                        state_v_first=True,
                        initial_state=h0_vk, output_final_state=True)
    # ht4 同样为 [B, H, V, K] 布局

# 解码
o5, ht5 = fused_recurrent_kda(q, k, v, g, beta, A_log=A_log, dt_bias=dt_bias,
                              use_gate_in_kernel=True,
                              initial_state=h0, output_final_state=True)
```

正确性校验：`naive_recurrent_kda` / `naive_chunk_kda`（`fla/ops/kda/naive.py`）用于与 `chunk_kda` / `fused_recurrent_kda` 数值比对（CI 见仓库 `tests/ops/`）。

---

## 参考资料

检索与锁定日期：**2026 年 8 月 30 日**。

| 资料 | 类型与版本 | 本文用途 |
| --- | --- | --- |
| [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692) | 官方技术报告，arXiv v2（2025-11-01），本地全文 `references/papers/kimi-linear/kimi-linear.md`（含 60 张图） | KDA 公式（Eq. 1–14）、chunkwise 算法、混合架构、实验与复杂度 |
| [Kimi-Linear-48B-A3B-Instruct](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct) | 官方 checkpoint（本地锁定副本 `references/checkpoints/kimi-linear-48b-official/`） | `config.json` 配置、`modeling_kimi.py` 完整层与缓存实现（文中行号均指本地副本，与 2026-08-30 拉取的 main 一致） |
| [Transformers are RNNs](https://arxiv.org/abs/2006.16236) | ICML 2020 | 线性注意力结合律与 RNN 视角；Fig. 1 |
| [Linear Transformers Are Secretly Fast Weight Programmers](https://arxiv.org/abs/2102.11174) | 2021 | fast-weight memory、容量干扰；delta rule 的更新规则对比（Fig. 2/3） |
| [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484) | DeltaNet 论文 | Householder/WY chunkwise 并行源头；Fig. 1/2 |
| [Gated Linear Attention Transformers with Hardware-Efficient Training](https://arxiv.org/abs/2312.06635) | GLA 论文 | channel-wise gating、I/O-aware 训练；Fig. 3（chunkwise map） |
| [Gated Delta Networks](https://arxiv.org/abs/2412.06464) | GDN 论文 | 标量忘门；Fig. 1（块设计） |
| [FLA `fla/ops/kda/`](https://github.com/fla-org/flash-linear-attention/tree/6ec09887145d2f51bf203304777770b041374b65/fla/ops/kda) | commit `6ec09887`（v0.6.0 附近），`fla/ops/precond_kda/` 同基线 | 全部内核事实（naive/chunk/fused_recurrent/gate/intra/wy_fast/chunk_delta_h） |
| [FLA KDA 合入 #621](https://github.com/fla-org/flash-linear-attention/commit/b5d48b7d2376b7c9b344d603591cb06d93c13aea) | commit `b5d48b7d` | KDA 进入 FLA 的版本点 |
| [FLA FlashKDA backend #852](https://github.com/fla-org/flash-linear-attention/commit/97aaf76714afa5c5b443b25a5d5daac3ccc1ff54) | commit `97aaf767` | FlashKDA 后端接入版本点 |
| [FLA beta-sigmoid / 负特征值 #919](https://github.com/fla-org/flash-linear-attention/commit/19b5a3f411ecea6cdda62c6cc65cdae55ed2dec5) | commit `19b5a3f4` | `use_beta_sigmoid_in_kernel`/`allow_neg_eigval` 行为版本点 |
| [FLA issue #1119](https://github.com/fla-org/flash-linear-attention/issues/1119) | 0.5.0 旧版问题（0.6.0 已显式化） | `**kwargs` 静默吞 flag 反例（§14 误区 5） |
| [SGLang `kimi_linear.py`](https://github.com/sgl-project/sglang/blob/7399c2b5587e1559f3e5a26566ed322e81e1433a/python/sglang/srt/models/kimi_linear.py) | commit `7399c2b5` | 模型层：TP 投影融合、RadixLinearAttention |
| [SGLang `kda_backend.py`](https://github.com/sgl-project/sglang/blob/7399c2b5587e1559f3e5a26566ed322e81e1433a/python/sglang/srt/layers/attention/linear/kda_backend.py) | 同上 | `KDAAttnBackend`(L370)、`KDAKernelDispatcher`(L40) |
| [vLLM 快照](https://github.com/vllm-project/vllm/tree/0384aa7150c4c9778efca041ffd1beb3ad2bd694) | commit `0384aa7` | SGLang 改编来源；KDA 实现与 Mamba-style state manager |
| [vLLM issue #41292](https://github.com/vllm-project/vllm/issues/41292) | 已 closed（2026-04-30） | 状态布局 bug 案例 |
| [Megatron PR #6556](https://github.com/NVIDIA/Megatron-LM/pull/6556) | merge commit `788e9e1`（2026-08-25 入 `dev`，状态经 GitHub API 复核） | 基础 KDA hybrid 训练支持 |
| [Megatron PR #6877](https://github.com/NVIDIA/Megatron-LM/pull/6877) | open（经 GitHub API 复核，2026-08-30） | 低秩忘门/输出门/NoPE MLA |
| [Megatron `gated_delta_net/kda.py`](https://github.com/NVIDIA/Megatron-LM/blob/788e9e1711e23bdfc39b27438e3786ba60fc232b/megatron/core/ssm/gated_delta_net/kda.py) | `upstream/dev` @ `788e9e1` | 真 KDA（per-channel） |
| [FlashKDA](https://github.com/MoonshotAI/FlashKDA/tree/1ce47ea3bb22c84eb9cc665028399cf35e8ffb0b) | commit `1ce47ea` | CUTLASS 内核；deep-dive 文档与 GB200/H20 benchmark |
| [verl 文档](https://verl.readthedocs.io/) | commit `9ff05e32` | `docs/index.rst`（引擎解耦、FSDP/Megatron/vLLM/SGLang 集成）、`workers/engine/base.py` |
| [KDA 数学原理（Zhiyuan Li）](https://zhiyuan1i.github.io/posts/kda-mathematics/) | 个人技术博客（KaTeX 数学文，无正文插图），本地 `references/articles/kda-mathematics/` | chunk-wise / Affine / WY / CP/SM 推导的交叉验证 |
| [DPLR 数学原理（Zhiyuan Li）](https://zhiyuan1i.github.io/posts/dplr-mathematics/) | 同上，本地 `references/articles/dplr-mathematics/` | DPLR vs KDA vs IPLR 统一框架 |
| [hwilner 教学复现](https://github.com/hwilner/kimi-delta-attention/tree/4b08a62c6ed7c94a8a3a730cd8beb2847da4fe5e)（本地 `references/community/kimi-delta-attention/`，commit `4b08a62c`） | 社区 MIT 教学实现，**非官方** | 未作为公式来源；仅可用于概念教学对照（本文未引用其代码） |
| Moonshot 工程师知乎解析（KDA 并行计算） | 口述式素材，本地草稿 `kda_linear_attention-deep-dive_draft-2.md` 保留其技术内容 | §7.1 算术直觉、§8.4 数值雷区叙事（所有公式均已回论文/源码复核后再采用） |

**图片来源说明**：正文复用的 15 张图全部来自上述论文目录（`references/papers/<slug>/images/`），均已复制到本文 `pics/` 下并以相对路径引用（`./pics/<name>.jpg`），每张图配图注与出处；三张自绘 Mermaid（§9.1 单层数据流、§11.2 前向编排、§12.1 混合缓存）均在正文标明为自绘。经核验的三处来源边界：两篇 Zhiyuan Li 博客**无正文插图**；draft-2/4 中的知乎在线图片未随仓库保存（不引用）；论文图片目录中未被 `.md` 引用的哈希文件为 MinerU 转换产生的公式/版面渲染资产（公式、表格渲染等），**不属于正文插图，不纳入 61 张正文图的复用核对**。

> 社区复现（hwilner）、博客（Zhiyuan Li）、论文、官方 checkpoint 四类来源在正文中分开标注；冲突时以"锁定 commit 的源码 / 官方论文"为准。

<!-- /learn-write 自动检查报告
本稿已完成四轮独立审核（作者侧 + 外部 REVIEW-deep / REVIEW-deep-r3 / REVIEW-deep-r4 / REVIEW-deep-r5），全部 P0 已闭环（处理记录见 REVIEW.md / REVISION.md）。
双轨：概念(§1–§4) → 模型/算法(§5–§8) → 代码(§9–§12)；源码引用锁定 commit（FLA 6ec09887 + b5d48b7d/97aaf767/19b5a3f4；SGLang 7399c2b5；Megatron upstream/dev@788e9e1；vLLM 0384aa7；FlashKDA 1ce47ea；verl 9ff05e32），无 main 分支行号。
格式：15 张复用图（pics/，相对路径可解析，均有 alt、图注与出处）；3 张自绘 Mermaid（§9.1/§11.2/§12.1，已标注）；正文无 ASCII 字符画、无破折号；附录与参考资料为二级标题。
-->


