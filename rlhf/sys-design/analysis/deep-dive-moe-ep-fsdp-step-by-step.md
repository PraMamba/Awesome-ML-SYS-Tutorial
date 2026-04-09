# 逐步理解：DeepSeek MoE、Expert Parallelism 与 FSDP 二次开发

> 本文是对 [readme-4.md](../readme-4.md) 的逐步拆解式学习笔记。
> 原文涉及大量分布式系统和 MoE 架构的交叉知识，本文将其拆成 8 个循序渐进的学习步骤，每一步只解决一个问题。

---

## 学习路线图（ToDo List）

| 步骤 | 主题 | 核心问题 | 前置知识 |
|------|------|---------|---------|
| Step 1 | Dense 模型的计算方式 | 一个 Token 在普通模型里是怎么被处理的？ | 无 |
| Step 2 | MoE 的基本思想 | 为什么不让所有参数都参与计算？ | Step 1 |
| Step 3 | DeepSeek MoE 的创新 | "小而多"的专家设计有什么好处？ | Step 2 |
| Step 4 | 为什么需要 Expert Parallelism | 256 个专家放不下一张卡怎么办？ | Step 3 |
| Step 5 | EP 的三阶段流程 | Token 如何跨 GPU 找到自己的专家？ | Step 4 |
| Step 6 | EP vs TP：通信量与计算效率 | 为什么不直接用 TP 切专家？ | Step 5 |
| Step 7 | FSDP 是什么，为什么要在上面做二次开发 | FSDP 只管 DP，EP 要自己加 | Step 6 |
| Step 8 | 三大框架的工程实现对比 | VeOmni / TorchTitan / Automodel 怎么做的？ | Step 7 |

---

## Step 1：Dense 模型的计算方式——理解基线

### 一句话总结

在普通的 Dense（稠密）模型中，每个 Token 经过每一层时，**所有参数都参与计算**。

### 详细解释

想象一个 Transformer 模型有 40 层，每层有两个主要部件：
- **Attention 层**：让 Token 之间互相"看"对方
- **FFN 层（Feed-Forward Network）**：对每个 Token 独立做一次非线性变换

```
输入 Token
    ↓
┌─────────────────┐
│  Attention 层    │  ← 所有参数参与
├─────────────────┤
│  FFN 层          │  ← 所有参数参与（这里是重点！）
│  [Hidden → 4H → Hidden] │
└─────────────────┘
    ↓
输出 Token
```

FFN 层的结构通常是：
- 一个 **上投影矩阵**（$W_{up}$）：把维度从 $H$ 扩大到 $4H$
- 一个激活函数（如 ReLU / SiLU）
- 一个 **下投影矩阵**（$W_{down}$）：把维度从 $4H$ 缩回 $H$

**关键点**：不管输入什么 Token，这个 FFN 的全部参数都要参与运算。如果模型有 405B 参数（如 Llama 3.1 405B），那处理每个 Token 都要动用全部 405B 参数——计算量巨大。

### 这引出了什么问题？

模型越大，能力越强，但计算开销也线性增长。有没有办法**让模型很大（参数多），但每个 Token 只用其中一小部分参数**？

---

## Step 2：MoE 的基本思想——用"专家分工"降低计算量

### 一句话总结

MoE 把一个巨大的 FFN 拆成很多个小 FFN（称为"专家"），每个 Token 只挑几个专家来算。

### 详细解释

MoE 的核心改动**只发生在 FFN 层**，Attention 层不变：

```
Dense 模型的 FFN：                    MoE 模型的 FFN：
┌──────────────────┐               ┌──────────────────┐
│                  │               │  Gate（路由网络）  │ ← 新增！决定去哪个专家
│  一个巨大的 FFN   │               ├──────────────────┤
│  所有 Token 共用  │               │ Expert 0 (小FFN)  │
│                  │               │ Expert 1 (小FFN)  │ ← 每个 Token 只选 Top-k 个
└──────────────────┘               │ Expert 2 (小FFN)  │
                                   │ ...               │
                                   │ Expert 7 (小FFN)  │
                                   └──────────────────┘
```

**工作流程**（5 步）：

1. **Gate 计算**：Token 经过一个小型网络（Gate），算出它与每个专家的"匹配分数"
2. **专家选择**：根据分数选出 Top-k 个专家（比如 k=2，选 2 个最匹配的）
3. **Token 分发**：把 Token 发给选中的专家
4. **专家计算**：被选中的专家各自独立计算
5. **结果合并**：把各专家的输出按 Gate 给出的权重加权求和

### 为什么这样能省计算？

假设原来一个 FFN 有参数量 $P$，现在拆成 8 个专家，每个专家参数量 $P/8$。每个 Token 只激活 2 个专家，那实际计算量只有 $2 \times P/8 = P/4$，是原来的 **1/4**。

但模型的**总参数量没变**（还是 $P$），甚至可以更多！这就是 MoE 的核心优势：**用更少的计算获得更大模型的表达能力**。

### 直觉类比

想象一个医院：
- **Dense 模式**：每个病人（Token）来了，所有医生（参数）都要会诊——效率低
- **MoE 模式**：前台（Gate）先判断病人的症状，然后分配给 2 个最相关的专科医生（Experts）——效率高，还能养更多医生

---

## Step 3：DeepSeek MoE 的创新——"小而多"与"共享专家"

### 一句话总结

DeepSeek 把专家拆得**更细更多**（从 8 个到 256 个），还留了一些"永远在线的通识专家"。

### 创新点 1：细粒度专家（Fine-grained Experts）

传统 MoE（如 GShard）：8 个专家，每个很大
DeepSeek MoE：64 个专家（V3 是 256 个），每个很小

```
传统 MoE：                          DeepSeek MoE：
┌────────┐                         ┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐┌──┐
│Expert 0│  ← 每个专家很大          │E0││E1││E2││E3││E4││E5││E6││E7│  ← 每个专家很小
├────────┤                         ├──┤├──┤├──┤├──┤├──┤├──┤├──┤├──┤
│Expert 1│  共 8 个                 │E8││E9││..│...│...│...│...│...│  共 256 个
├────────┤  每个 Token 选 Top-2     └──┘└──┘└──┘└──┘└──┘└──┘└──┘└──┘
│...     │                         每个 Token 选 Top-8
│Expert 7│
└────────┘
```

**为什么拆更细更好？**

- 更多组合可能性：8 个里选 2 个 = $C(8,2)$ = 28 种组合；256 个里选 8 个 = $C(256,8)$ ≈ 48 亿种组合
- 更精细的知识分工：大专家可能混杂了多种知识，小专家可以更"专"

### 创新点 2：共享专家（Shared Experts）

```
每层的结构：
┌──────────────────────────────┐
│  Shared Expert (永远激活)     │  ← 存储"常识"，每个 Token 都经过
├──────────────────────────────┤
│  Router Expert 0 ~ 255       │  ← 256 个路由专家，每个 Token 只选 8 个
│  (Top-8 激活)                │
└──────────────────────────────┘
```

共享专家的作用：有些知识是所有 Token 都需要的（比如语法规则、常见词义），不需要"挑选"，直接让共享专家处理。

### DeepSeek V3 的具体数字

| 指标 | 数值 |
|------|------|
| 总参数量 | 671B |
| 每个 Token 激活参数量 | ~37B |
| 路由专家数 | 256 |
| 共享专家数 | 1 |
| 每 Token 激活路由专家数 | Top-8 |

**关键结论**（原文强调）：总参数 $X$、激活参数 $Y$ 的 MoE 模型，其表现 > 总参数为 $Y$ 的 Dense 模型，而计算开销 < 总参数为 $X$ 的 Dense 模型。这几乎是 2024 年以来 MoE 架构统治时代的开端之作。

---

## Step 4：为什么需要 Expert Parallelism——单卡装不下

### 一句话总结

256 个专家的总参数太大，一张 GPU 放不下，必须把专家**分散到多张 GPU 上**。

### 问题的本质

以 DeepSeek V3 为例：
- 总参数 671B，即便用 FP16 存储也需要 ~1.2TB 显存
- 目前最大的单卡（H100 80GB）完全不够
- 即使你有 8 张 H100（共 640GB），还是塞不下

所以必须把模型切分到多张卡上。但怎么切？有几种思路：

```
切分策略大全：

1. Data Parallelism (DP)：每张卡存完整模型，切分数据
   → 前提是单卡能装下完整模型，671B 显然不行

2. Tensor Parallelism (TP)：把每一层的矩阵横切/竖切
   → 可以，但对 MoE 有严重效率问题（后面 Step 6 详解）

3. Pipeline Parallelism (PP)：不同层放不同卡
   → 可以，但和本文关系不大

4. Expert Parallelism (EP)：不同专家放不同卡 ← 本文主角！
   → 天然适合 MoE 的结构
```

### EP 的核心思想

假设有 256 个专家和 8 张 GPU：

```
GPU 0: Expert 0 ~ 31     (32 个专家)
GPU 1: Expert 32 ~ 63    (32 个专家)
GPU 2: Expert 64 ~ 95    (32 个专家)
...
GPU 7: Expert 224 ~ 255  (32 个专家)
```

每张卡只需要存 32 个专家的权重，显存压力降为 1/8。

### 但这带来了新问题

在没有 EP 时，一个 Token 要去 Expert 42 计算，直接在本地调用就行。现在 Expert 42 在 GPU 1 上，但这个 Token 可能在 GPU 5 上——需要**跨 GPU 传输 Token 数据**。

这就引出了 EP 的核心通信模式：**All-to-All**。

---

## Step 5：EP 的三阶段流程——Token 如何跨 GPU 找专家

### 一句话总结

EP 的工作流程是：Token 先跨 GPU 发到专家那里（Dispatch），专家算完后再发回来（Combine），中间需要两次 All-to-All 通信。

### All-to-All 通信是什么？

先理解几种常见的通信模式（以 4 张 GPU 为例）：

```
Broadcast（广播）：一个人说话，所有人听
  GPU 0: [A] ──→ 所有 GPU 都得到 [A]

All-Reduce（全规约）：所有人把数据加起来，每人得到总和
  GPU 0: [1]  ──→  所有 GPU 都得到 [1+2+3+4] = [10]
  GPU 1: [2]  ──→
  GPU 2: [3]  ──→
  GPU 3: [4]  ──→

All-to-All（全交换）：每人把不同的数据发给不同的人
  GPU 0: [A0, A1, A2, A3] ──→ GPU 0 得到 [A0, B0, C0, D0]
  GPU 1: [B0, B1, B2, B3] ──→ GPU 1 得到 [A1, B1, C1, D1]
  GPU 2: [C0, C1, C2, C3] ──→ GPU 2 得到 [A2, B2, C2, D2]
  GPU 3: [D0, D1, D2, D3] ──→ GPU 3 得到 [A3, B3, C3, D3]
```

**All-to-All 就是"分布式转置"**——数据原来按"来源"排列，通信后按"目的地"排列。

### EP 的三阶段

```
阶段 1: Dispatch（分发）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
每张 GPU 上有一批 Token，每个 Token 通过 Gate 计算出要去哪个专家。
然后通过 All-to-All，把 Token 发到专家所在的 GPU。

       通信前（按序列位置排列）          通信后（按专家索引排列）
GPU 0: [Token A→Exp5, Token B→Exp1]    GPU 0: [Token D, Token F]  ← 发给 Exp 0~31
GPU 1: [Token C→Exp0, Token D→Exp0]    GPU 1: [Token B, Token E]  ← 发给 Exp 32~63
GPU 2: [Token E→Exp33, Token F→Exp2]   GPU 2: [Token C, Token A]  ← 发给 Exp 64~95
...                                     ...


阶段 2: Expert Compute（专家计算）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
每张 GPU 用自己本地的专家，对收到的 Token 做 FFN 计算。
这个阶段完全并行，无需通信！

GPU 0: Expert 0~31 处理收到的 Token → 输出结果
GPU 1: Expert 32~63 处理收到的 Token → 输出结果
...

⚠️ 潜在问题：如果大量 Token 涌向同一个专家（热点专家），
   那个 GPU 算得慢，其他 GPU 只能等着 → 负载不均衡


阶段 3: Combine（合并）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
再做一次 All-to-All，把计算结果发回 Token 原来所在的 GPU。
然后按 Gate 权重加权求和，继续往下一层传。
```

### 为什么需要两次 All-to-All？

一次发 Token 过去（Dispatch），一次把结果发回来（Combine）。数据必须回到原来的位置，因为下一层的 Attention 还要按序列位置做计算。

---

## Step 6：EP vs TP——为什么不直接用 TP 切专家？

### 一句话总结

TP 把每个专家"切碎"，虽然也能减少显存，但对 MoE 的小专家来说，**切碎后的矩阵乘法效率极低**。EP 保持专家完整，效率更高。

### 理解 TP（Tensor Parallelism）

TP 的思路完全不同于 EP——它不是按"专家"拆分，而是把每个专家的矩阵本身切开：

```
EP 的切法（按专家维度切）：
  GPU 0 持有完整的 Expert 0, 1, 2, 3
  GPU 1 持有完整的 Expert 4, 5, 6, 7
  → 每个 GPU 上的专家矩阵是完整的

TP 的切法（按矩阵维度切）：
  GPU 0 持有 Expert 0~7 的左半部分矩阵
  GPU 1 持有 Expert 0~7 的右半部分矩阵
  → 每个 GPU 上的每个专家矩阵都是不完整的碎片
```

### 通信量对比

定义变量：
- $N$ = GPU 数量
- $S$ = 总激活数据量（$Batch \times SeqLen \times HiddenSize$）
- $k$ = 每 Token 激活的专家数

| | TP | EP |
|---|---|---|
| 通信量 | $\approx 2S$（固定，和 $k$ 无关） | $\approx \frac{2k}{N} S$（随 $k/N$ 缩放） |
| 通信模式 | All-Reduce | All-to-All |

**数值例子**：假设 $N=256$, $k=8$

- TP 通信量：$2S$
- EP 通信量：$\frac{2 \times 8}{256} S = \frac{1}{16} S$

EP 的通信量只有 TP 的 $1/32$！

### 但为什么 EP 的通信量小，不代表 EP 一定通信更快？

原文指出两个关键因素：

```
通信带宽差异：

TP 通常在单机 8 卡内通信（NVLink）
  ┌─────────────────────────┐
  │  GPU0 ←NVLink→ GPU1     │  带宽：~900 GB/s
  │  GPU2 ←NVLink→ GPU3     │  延迟：极低
  │  ...                    │
  └─────────────────────────┘

EP 通常需要跨机器通信（RDMA/InfiniBand）
  ┌──────────┐  RDMA   ┌──────────┐
  │ 机器 A    │ ←────→  │ 机器 B    │  带宽：~50-100 GB/s
  │ GPU 0~7  │         │ GPU 8~15 │  延迟：较高
  └──────────┘         └──────────┘

带宽差了 10 倍！
```

### 那 EP 到底凭什么赢？——计算效率（MFU）

**这是整篇文章最核心的论点**。原文的结论是：EP 在**计算效率**上的优势远远弥补了通信上的劣势。

```
TP 的问题：矩阵太"瘦"

原始专家矩阵：[H × 4H] ← 比如 [7168 × 28672]，形状饱满，GPU 算得快

TP 8 切分后：[H × 4H/8] ← 变成 [7168 × 3584]，形状变瘦

如果是 DeepSeek 的细粒度专家（单个专家本来就小）：
原始：[H × 256] ← 已经很小了
TP 8：[H × 32]  ← 极其瘦长！

GPU 的 Tensor Core 需要矩阵的维度足够大才能充分利用，
太瘦的矩阵 → 流水线填不满 → 实际算力利用率暴跌
```

```
EP 的优势：矩阵完整

EP 不切专家矩阵，只是把 Token 搬到专家所在的 GPU 上。
到了目标 GPU 后，专家矩阵是完整的 → GEMM 效率高。
```

### 为什么"小而多"的专家设计让 EP 碾压 TP？

这是原文的深层洞察：

```
GShard 时代（2020）：                    DeepSeek 时代（2024+）：
- 8 个大专家                            - 256 个小专家
- 单个专家参数量大                        - 单个专家参数量极小
- TP 切分后，矩阵仍然够大                 - TP 切分后，矩阵小到无法忍受
- → TP 是主流方案                        - → EP 成为必然选择
```

**结论**：EP 的崛起是被**算法设计**（细粒度专家）驱动的。

### DeepEP 如何进一步解决 EP 的通信瓶颈

DeepSeek 开发了 DeepEP 库，两个关键优化：

1. **计算-通信重叠（Stream-K）**：不等所有 Token 到齐就开始计算
   ```
   传统方式：[等 Token 全到齐] → [开始计算]  ← 通信和计算串行
   Stream-K：[第1批 Token 到] → [立即计算第1批] → [第2批到，计算第2批] ...
             ← 通信和计算并行，互相掩盖
   ```

2. **RDMA 直驱**：绕过 NCCL 协议栈，用 PTX 级优化实现低延迟跨节点通信

### EP vs TP 总结表

| 维度 | TP | EP |
|------|----|----|
| 通信量 | 固定 ($\approx 2S$)，与 $k$ 无关 | 随 $k/N$ 缩放 ($\approx 2kS/N$) |
| 通信带宽 | NVLink ~900 GB/s（快） | RDMA ~50-100 GB/s（慢） |
| 计算效率(MFU) | 低（矩阵切碎，算子不饱满） | 高（矩阵完整，硬件充分利用） |
| 扩展性 | 局限于单机 | 支持万卡集群 |
| 适用场景 | 大而少的专家（GShard 时代） | 小而多的专家（DeepSeek 时代） |

### 补充：ETP 是什么？TP 和 EP 可以同时开吗？

原文提到 SGLang 启动 DeepSeek R1 的命令：

```bash
python3 -m sglang.launch_server --tp 8 --ep 8
```

这**不是** ETP（先 EP 再对每个 expert 做 TP）。实际含义是：
- `--ep 8`：MoE 部分的 256 个专家分到 8 张卡上
- `--tp 8`：非 MoE 部分（如 Attention 的线性层）按 TP 分到 8 张卡上

两个参数作用于模型的**不同子模块**，并不冲突。

---

## Step 7：FSDP 是什么？为什么需要二次开发？

### 一句话总结

FSDP 是 PyTorch 官方的分布式训练方案，但它只支持最基础的 DP（数据并行），不支持 EP。要在 FSDP 上用 EP，得自己动手改。

### FSDP 基础

FSDP = Fully Sharded Data Parallelism（全切分数据并行），本质是 DeepSpeed ZeRO 的 PyTorch 原生实现。

```
普通 DP（数据并行）：
  GPU 0: 完整模型副本 + 数据 Batch 0
  GPU 1: 完整模型副本 + 数据 Batch 1
  → 每张卡都存完整模型，浪费显存

FSDP（全切分数据并行）：
  GPU 0: 模型参数的 1/N + 数据 Batch 0
  GPU 1: 模型参数的 1/N + 数据 Batch 1
  → 参数切碎分散存储，需要时再临时拼起来
```

FSDP 的核心动作：
- **Forward 前**：`all-gather` 把其他卡的参数碎片拼回完整参数
- **Forward 后**：释放其他卡的参数碎片（省显存）
- **Backward 后**：`reduce-scatter` 聚合梯度并重新切分

### FSDP 的"隐式切分" vs EP 的"显式切分"

这是理解本文工程部分的**最关键区别**：

```
FSDP 隐式切分（动态逻辑切分）：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
存储时：参数被切碎成 N 份，每卡存 1 份
计算时：自动 all-gather 拼回完整参数

上层代码完全不知道参数被切过！
model.weight 看起来就是完整的 [128, H, I]
FSDP 在背后偷偷做通信


EP 显式切分（静态物理切分）：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
存储时：专家被物理分开，每卡只有部分专家
计算时：就是那么多专家，不会临时拼回去

上层代码必须知道"我只有 32 个专家"！
model.experts 的 shape 确实从 [128, H, I] 变成了 [32, H, I]
模型代码必须感知并适配这个变化
```

### 为什么 FSDP 不原生支持 EP？

FSDP 的设计哲学是"对上层透明的参数切分"——它切分参数的方式不需要上层代码知道。但 EP 需要上层代码明确感知专家被分到了哪些卡上，这和 FSDP 的设计哲学冲突。

所以要在 FSDP 上加 EP，就需要**二次开发**：
1. 先手动做 EP 切分（显式的，改变 Tensor shape）
2. 然后在 EP 切分后的模块上再套 FSDP（隐式的，切 hidden 维度）

### EP + FSDP 的协同流程

```
原始参数 shape: [128 experts, H, I]
         ↓ EP 切分 (dim 0)
每卡参数 shape: [32 experts, H, I]
         ↓ FSDP 切分 (dim 1)
每卡实际存储: [32 experts, H/fsdp_size, I]

Forward 时：
  FSDP all-gather → [32, H, I] ← FSDP 自动拼回完整的 hidden 维度
  All-to-All Dispatch → Token 跨卡搬运 ← EP 的通信
  Expert FFN 计算
  All-to-All Combine → Token 搬回来 ← EP 的通信
  FSDP 释放参数

Backward 时：
  类似但多了 reduce-scatter（FSDP 的梯度聚合）
```

**一个重要细节**：EP 组内的专家**不需要**做梯度聚合！因为每个专家是独立的模块，各自优化各自的参数。FSDP 的 reduce-scatter 只在 FSDP 自己切分的维度上做。

---

## Step 8：三大框架的工程实现——VeOmni / TorchTitan / Automodel

### 一句话总结

三个框架的整体思路一致（先 EP 再 FSDP），但在通信后端（NCCL vs DeepEP）和预取策略（prefetch）上各有特色。

### 共同架构

```
三个框架都是这个流程：

1. 对 MoE 层的 expert 做 EP 切分 (dim 0)
2. 对切分后的 expert 做 FSDP (dim 1)
3. 对非 MoE 的模块做常规 FSDP (dim 0)
4. 配置 prefetch 来掩盖通信延迟
```

### 框架 1：VeOmni（字节跳动 Seed）

**特色：清晰的分层架构 + PyTorch 原生通信**

核心代码逻辑（简化版）：

```python
# 第 1 步：EP 切分——把 128 个专家分到 4 张卡上
# shape 从 [128, H, I] 变成每卡 [32, H, I]
parallel_plan.apply(model, device_mesh)

# 第 2 步：对每层循环，由内而外做 FSDP
for layer in decoder_blocks:
    if layer 有专家模块:
        fully_shard(layer.experts, ...)  # 先切专家（dim 1）
    fully_shard(layer, ...)              # 再切整层（dim 0）

# 第 3 步：切根模型
fully_shard(model, ...)

# 第 4 步：配置预取——计算第 n 层时，提前拉取第 n+1 层参数
for cur, nxt in 相邻层对:
    cur.set_modules_to_forward_prefetch([nxt 的模块列表])
```

**通信实现**（基于 PyTorch 原生 `dist.all_to_all`）：

```
Preprocess → Dispatch → Expert Compute → Combine
     ↓            ↓                          ↓
all_gather    all_to_all                 all_to_all
(交换元数据)  (Token 跨卡发送)          (结果跨卡回传)
```

三个关键函数：
- `preprocess()`：先通过 all_gather 交换元数据（"我要发多少 Token 给你，你要发多少给我"）
- `token_pre_all2all()`：本地 Permute 重排 → All-to-All 发送 → Sort 排序
- `tokens_post_all2all()`：逆向操作，把结果发回原来的 GPU

### 框架 2：Automodel（NVIDIA NeMo）

**特色：深度集成 DeepEP，绕过 NCCL**

Automodel 不用 PyTorch 原生的 `dist.all_to_all`，而是用 DeepSeek 开源的 DeepEP 库：

```
调用链：
MoEFlexTokenDispatcher
    → _DeepepManager.dispatch()
        → fused_dispatch()          ← DeepEP 的 RDMA 直驱
            → buffer.get_dispatch_layout()  ← 计算通信布局
            → buffer.dispatch()             ← 执行通信
```

核心设计：**有状态的上下文管理**

```python
class _DeepepManager:
    def dispatch(self, hidden_states):
        # DeepEP 返回一个 handle，包含通信布局信息
        ..., handle = fused_dispatch(hidden_states, ...)
        self.handle = handle  # 保存 handle！
        return hidden_states

    def combine(self, hidden_states):
        # combine 时复用 dispatch 的 handle
        result = fused_combine(hidden_states, self.handle)
        self.handle = None
        return result
```

为什么需要 handle？因为 Dispatch 和 Combine 是一对逆操作——Combine 需要知道 Dispatch 时 Token 是怎么分发的，才能把结果发回正确的 GPU。handle 保存了这个"分发记录"。

**`FusedDispatch` 的自动微分**：

```python
class FusedDispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ...):
        ..., handle = buffer.dispatch(x, ...)
        ctx.handle = handle  # 保存给 backward 用
        return recv_x, ...

    @staticmethod
    def backward(ctx, grad_output):
        # backward 的反向通信就是 combine
        grad_x = buffer.combine(grad_output, ctx.handle)
        return grad_x, ...
```

这意味着：forward 的 dispatch 对应 backward 的 combine，反之亦然。这是因为 All-to-All 的反向传播本身就是另一个 All-to-All（方向相反）。

### 框架 3：TorchTitan（PyTorch 官方）

**特色：最全面的 prefetch 策略**

大多数框架只预取"下一层 Block"，TorchTitan 更进一步：

```
普通框架的 prefetch：
  计算第 n 层时 → 预取第 n+1 层的参数

TorchTitan 的 prefetch：
  计算第 n 层时 → 预取第 n+1 层的参数 AND 第 n+1 层的 Expert 参数

而且覆盖全链路：
  Embedding → Block 0 → Block 1 → ... → Block N → Norm → Output
  ↑ prefetch ↑ prefetch ↑         ...    ↑ prefetch ↑ prefetch
  每一步都在提前拉取下一步需要的数据
```

前向 prefetch 代码逻辑：
```python
# 从 Embedding 就开始预取
tok_embeddings.set_modules_to_forward_prefetch([第一个 Block])

for 当前Block, 下一个Block in 相邻层:
    if 下一个Block 有 MoE:
        # 预取 Block + Block 内的 Expert 模块
        当前Block.prefetch([下一个Block, 下一个Block.moe.experts])
    else:
        当前Block.prefetch([下一个Block])

# 最后一个 Block 预取 Norm 和 Output
最后Block.prefetch([model.norm, model.output])
```

反向传播也有对应的 prefetch，从 Output 反向到 Embedding，密不透风。

**条件切分策略**：当 `efsdp_size * ep_degree > num_experts` 时，FSDP 切换到 `Shard(1)` 沿 hidden 维度切分，而不是默认的 dim 0。这避免了在专家数量不够分的情况下出错。

### 三大框架对比总结

| 维度 | VeOmni | Automodel | TorchTitan |
|------|--------|-----------|------------|
| 开发方 | 字节跳动 Seed | NVIDIA NeMo | PyTorch 官方 |
| 通信后端 | PyTorch 原生 dist.all_to_all | DeepEP (RDMA 直驱) | 支持原生 + DeepEP |
| Prefetch | 前向+反向，手动配置 | 未在本文详述 | 全链路密不透风 |
| EP 切分 | parallel_plan.apply() | ExpertParallel.apply_ep() | fully_shard + edp_mesh |
| 代码清晰度 | 高（分层清晰） | 中（封装较深） | 中（条件逻辑多） |
| 模型支持 | Qwen3-MoE | Megatron 兼容模型 | Llama 4 |
| GitHub Star（原文时） | - | ~200（被低估） | 高（官方项目） |

---

## 全文核心观点回顾

读完这 8 个步骤，回顾原文想传达的核心信息：

1. **MoE 的本质**是用稀疏激活换取更大的参数空间，DeepSeek 把这条路走到了极致（256 个极小专家）

2. **EP 的必要性**：细粒度专家设计 → 单卡放不下 → 必须把专家分散到多卡 → EP

3. **EP 优于 TP 的根本原因**：不是通信量（EP 虽少但带宽低），而是**计算效率**——TP 把小专家切更碎导致 GEMM 效率暴跌

4. **EP 是算法驱动的并行策略**：是"小而多"的专家设计推动了 EP 的崛起，而不是反过来

5. **FSDP 的局限**：只支持 DP，要做 EP 必须二次开发。社区的做法是：EP 切 dim 0（专家数量），FSDP 切 dim 1（hidden 维度），两者正交协作

6. **三大框架的共识**：先 EP 再 FSDP，配合 prefetch 掩盖通信。差异主要在通信后端（NCCL vs DeepEP）和 prefetch 覆盖范围

---

## 延伸阅读

- [DeepSeek MoE 论文](https://arxiv.org/abs/2401.06066)
- [DeepSeek V3 技术报告](https://arxiv.org/pdf/2412.19437)
- [DeepEP GitHub](https://github.com/deepseek-ai/DeepEP)
- [VeOmni](https://github.com/ByteDance-Seed/VeOmni) / [TorchTitan](https://github.com/pytorch/torchtitan) / [Automodel](https://github.com/NVIDIA-NeMo/Automodel)
- [DeepEP 代码解读](https://www.cnblogs.com/CQzhangyu/p/18741625)
- [EPLB 深度解析](https://zhuanlan.zhihu.com/p/29963005584)
