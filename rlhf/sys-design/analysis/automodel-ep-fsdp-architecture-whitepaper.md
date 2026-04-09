# Automodel 架构设计白皮书：EP + FSDP 混合并行系统

> 本文从架构设计的角度，解析 Automodel（NVIDIA NeMo）如何在 Mixture-of-Experts（MoE）训练中实现 Expert Parallelism（EP）与 Fully Sharded Data Parallelism（FSDP）的正交叠加。不涉及具体代码实现，聚焦于**设计约束、决策逻辑和模块交互**。

---

## 目录

1. [问题定义与设计目标](#1-问题定义与设计目标)
2. [双 DeviceMesh 架构：物理 GPU 到逻辑维度的映射](#2-双-devicemesh-架构物理-gpu-到逻辑维度的映射)
3. [EP 与 FSDP 的正交性设计](#3-ep-与-fsdp-的正交性设计)
4. [并行化编排的依赖链](#4-并行化编排的依赖链)
5. [通信后端选型：DeepEP 与 NCCL 的取舍](#5-通信后端选型deepep-与-nccl-的取舍)
6. [Token 路由的系统设计](#6-token-路由的系统设计)
7. [两种专家计算范式的架构权衡](#7-两种专家计算范式的架构权衡)
8. [梯度同步的分层控制](#8-梯度同步的分层控制)
9. [检查点系统的格式抽象](#9-检查点系统的格式抽象)
10. [总结：约束驱动的设计哲学](#10-总结约束驱动的设计哲学)

---

## 1. 问题定义与设计目标

### 1.1 MoE 训练的本质矛盾

MoE 模型引入了一种根本性的异质性：**模型中同时存在两类具有截然不同分布语义的参数**。

- **普通参数**（Attention、LayerNorm、Embedding、LM Head）：每个 GPU 上都需要完整的副本或按统一规则切分。它们参与所有 token 的计算，梯度在所有数据并行 rank 之间归约。
- **专家参数**（MoE 层中的 FFN 权重）：每个 GPU 只需要持有部分专家的权重。每个专家只处理被路由到它的 token 子集，不同专家之间的梯度无需归约。

这种异质性使得"对所有参数使用同一套并行策略"变得不可行。具体来说：

| 维度 | 普通参数 | 专家参数 |
|------|---------|---------|
| **切分语义** | 按 hidden dimension 切分（FSDP Shard） | 按 expert ID 切分（EP Shard） |
| **归约语义** | 全局 All-Reduce 或 Reduce-Scatter | 无需跨专家归约 |
| **通信模式** | All-Gather + Reduce-Scatter（FSDP 通信） | All-to-All（Token 路由通信） |
| **切分维度** | 参数张量的 dim=0 或 dim=1 | 参数张量的 dim=0（专家维度） |
| **Device Mesh** | 覆盖 DP + CP + TP 的组合网格 | 覆盖 EP + EP Shard 的独立网格 |

### 1.2 设计目标

Automodel 的 EP + FSDP 系统需要同时满足以下约束：

1. **正交切分**：EP 和 FSDP 必须切不同的维度，互不干扰
2. **独立管理**：专家参数和普通参数必须由不同的 FSDP 实例管理，因为它们的通信组、切分维度和同步策略都不同
3. **统一编排**：尽管管理独立，所有并行策略必须在一个统一的编排流程中按正确顺序施加
4. **通信高效**：Token 路由的 All-to-All 通信是 MoE 训练的性能瓶颈，必须选择高效的通信后端
5. **梯度正确性**：在梯度积累和 Pipeline 并行场景下，梯度同步的时机必须精确控制
6. **检查点兼容**：内部表示与外部模型格式（HuggingFace）之间需要无损转换

---

## 2. 双 DeviceMesh 架构：物理 GPU 到逻辑维度的映射

### 2.1 为什么需要两个 DeviceMesh

**核心问题**：同一组物理 GPU，需要承担两种不同的逻辑角色。

在 PyTorch 的 DeviceMesh 抽象中，每个维度代表一种并行语义。对于普通参数，GPU 被组织为 `(Pipeline, DataParallel_Replicate, DataParallel_Shard, ContextParallel, TensorParallel)` 的五维网格；对于专家参数，同一批 GPU 需要被重新组织为 `(Pipeline, EP_Shard, Expert_Parallel)` 的三维网格。

**约束分析**：

- 不能在同一个 DeviceMesh 中混合这两种维度划分。PyTorch 的 DTensor 要求每个参数绑定到一个 DeviceMesh 上，并在该 mesh 的各维度上声明其分布策略（Shard / Replicate / Partial）。
- 如果强行将 EP 维度塞入 world_mesh（例如增加一个 `ep` 维度），则 world_mesh 变成六维甚至七维，且 EP 维度和 DP 维度之间存在复杂的互斥关系——EP 占用的 GPU 不能同时参与 DP Shard，反之亦然。
- 因此，Automodel 的设计选择是：**创建两个独立的 DeviceMesh，覆盖相同的物理 GPU 集合，但定义不同的逻辑维度**。

### 2.2 world_mesh：普通参数的五维逻辑空间

world_mesh 定义了普通参数的分布语义：

```
world_mesh 五维结构：

维度名称          │ 含义                      │ 参与的通信操作
─────────────────┼──────────────────────────┼───────────────────────
pp               │ Pipeline 并行             │ 跨 stage 的激活/梯度传递
dp_replicate     │ 数据并行（梯度归约组）      │ All-Reduce 梯度
dp_shard         │ 数据并行（参数切分组）      │ All-Gather / Reduce-Scatter
cp               │ Context 并行             │ 序列维度切分通信
tp               │ Tensor 并行              │ 矩阵乘法切分通信
```

系统还从这五个基础维度中派生出三个**扁平化子网格**：

- **dp 网格** = `dp_replicate × dp_shard`：用于数据加载器的分片
- **dp_shard_cp 网格** = `dp_shard × cp`：参数实际 FSDP 切分时使用的网格，因为 CP 维度的 GPU 也参与参数切分
- **dp_cp 网格** = `dp_replicate × dp_shard × cp`：loss All-Reduce 时使用的网格

**设计决策**：将 `dp_shard` 和 `cp` 合并为参数切分维度是一个关键选择。其原因是 Context Parallelism 中，每个 GPU 处理序列的不同片段，但它们共享相同的模型参数。在 FSDP 的视角下，CP 维度上的 GPU 可以像 DP Shard GPU 一样参与参数切分，从而最大化参数分散度、减少每个 GPU 的显存占用。

### 2.3 moe_mesh：专家参数的三维逻辑空间

moe_mesh 定义了专家参数的分布语义：

```
moe_mesh 三维结构：

维度名称          │ 含义                      │ 分布策略
─────────────────┼──────────────────────────┼──────────────
pp               │ Pipeline 并行             │ 与 world_mesh 的 pp 一致
ep_shard         │ 专家参数的 FSDP 切分       │ Shard(1)
ep               │ Expert Parallelism        │ Shard(0)
```

### 2.4 ep_shard_size 的推导逻辑

`ep_shard_size` 不是一个独立配置项，而是由其他并行度参数**推导**出来的。这是因为参与 EP 和参与 FSDP 的 GPU 之间存在一个约束关系：

**约束**：在数据并行 + Context 并行构成的 GPU 池中，EP 占用了一部分 GPU 来切分专家。剩余的 GPU 需要用 FSDP 来进一步切分每个专家的参数。

```
推导公式：

  dp_cp_size = dp_size × cp_size    （数据并行 + 上下文并行的 GPU 总数）
  ep_shard_size = dp_cp_size / ep_size  （EP 占用后剩余的 GPU 数量）

前提约束：dp_cp_size 必须是 ep_size 的整数倍
边界条件：如果 ep_size ≥ dp_cp_size，则 ep_shard_size = 1（无需进一步 FSDP 切分）
```

### 2.5 物理 GPU 到双 Mesh 的映射示意

以 8 GPU、EP=4、DP=8、CP=1 为例：

```
物理 GPU：
  GPU 0   GPU 1   GPU 2   GPU 3   GPU 4   GPU 5   GPU 6   GPU 7

world_mesh 视角（dp_replicate=1, dp_shard=8）：
  ←————————————— dp_shard = 8 ——————————————→
  所有 8 个 GPU 共同切分普通参数

moe_mesh 视角（ep_shard=2, ep=4）：
  ←— ep=4 —→   ←— ep=4 —→
   0  1  2  3    4  5  6  7

  ep group A: {GPU 0, 1, 2, 3}  ← 4 个 GPU 各持有不同专家子集
  ep group B: {GPU 4, 5, 6, 7}  ← 4 个 GPU 各持有不同专家子集

  ep_shard group 0: {GPU 0, GPU 4}  ← 同一个专家在这 2 个 GPU 上做 FSDP
  ep_shard group 1: {GPU 1, GPU 5}
  ep_shard group 2: {GPU 2, GPU 6}
  ep_shard group 3: {GPU 3, GPU 7}

关键观察：
  - GPU 0 和 GPU 4 持有"相同的专家子集"，但各自只持有这些专家参数的一半（FSDP Shard）
  - GPU 0 和 GPU 1 持有"不同的专家子集"（EP Shard）
  - 从 world_mesh 看，8 个 GPU 是一个 FSDP 组
  - 从 moe_mesh 看，8 个 GPU 被切成 4 个 ep_shard 对 + 2 个 ep 组
```

**设计洞察**：这种"同一批 GPU，两套逻辑视图"的设计之所以可行，是因为普通参数和专家参数**从不在同一个通信操作中混合**。普通参数的 All-Gather/Reduce-Scatter 走 world_mesh 的进程组；专家参数的 FSDP 通信走 moe_mesh 的 ep_shard 子组；Token 路由走 moe_mesh 的 ep 子组。三者互不干扰。

---

## 3. EP 与 FSDP 的正交性设计

### 3.1 正交性的数学含义

EP 和 FSDP 对专家参数的切分是**正交**的，体现在它们切的是参数张量的不同维度：

```
专家权重张量的形状（以 gate_and_up_projs 为例）：

  [n_experts, hidden_dim, intermediate_dim]
       ↑            ↑
   EP 切 dim=0   FSDP 切 dim=1

EP 切分：将 n_experts 个专家分配到 ep_size 个 GPU 上
  → 每个 GPU 持有 n_experts / ep_size 个完整专家

FSDP 切分（在 EP 之后）：将每个本地专家沿 hidden_dim 切分
  → 每个 GPU 持有的专家参数量再减少 ep_shard_size 倍
```

**为什么 FSDP 不能也切 dim=0？**

因为 dim=0 已经被 EP 使用了。EP 把 64 个专家分配到 4 个 GPU 上，每个 GPU 持有 16 个本地专家。如果 FSDP 也沿 dim=0 切分，就会把这 16 个专家进一步拆分——但这在语义上是错误的：FSDP 的 All-Gather 会收集所有 ep_shard rank 上的切片来重建完整参数，而不同 ep_shard rank 持有的是**同一组专家的不同 hidden dim 切片**。如果 FSDP 沿 dim=0 切，All-Gather 后会错误地把不同专家的参数拼接在一起。

### 3.2 三重 FSDP 逻辑

Automodel 在对 Transformer Block 施加 FSDP 时，实现了三重互锁的逻辑：

**第一重：专家的独立 FSDP Wrap**

```
问题���专家参数需要使用 moe_mesh 的 ep_shard 子网格做 FSDP，而非 world_mesh。
约束：PyTorch FSDP2 的 fully_shard 接受一个 mesh 参数，决定了参数在哪组 GPU 上切分。
解决：对 MoE 模块的 experts 子模块单独调用 fully_shard，传入 ep_shard 子网格，
      并指定 Shard(1) 作为切分策略（沿 hidden 维度而非专家维度）。
```

**第二重：参数排除**

```
问题：如果不做处理，后续对整个 Transformer Block 做 fully_shard 时，
      FSDP 会尝试再次管理专家参数，导致双重切分冲突。
约束：PyTorch FSDP2 会自动排除已经被子模块 fully_shard 管理的参数，
      但这只在专家有独立 FSDP Wrap 时才生效。
解决：对于有 EP 但无 ep_shard FSDP 的场景（即 ep_shard_size=1），
      需要显式将专家参数加入 ignored_params 集合。
```

**第三重：Block 级 FSDP**

```
问题：Attention、LayerNorm、Gate 等普通参数需要使用 world_mesh 做标准 FSDP。
约束：FSDP 以模块为粒度做 wrap，通常以 Transformer Block 为单位。
解决：对整个 Block 调用 fully_shard，使用 world_mesh 的 dp_shard_cp 子网格，
      同时通过 ignored_params 排除已被 EP/ep_shard 管理的专家参数。
```

### 3.3 Wrap 层次结构

```
Model
├── embed_tokens ──── FSDP(world_mesh)
├── Transformer Block[i] ──── FSDP(world_mesh, ignored_params=专家参数)
│   ├── Attention      → 由 Block 的 FSDP 管理
│   ├── LayerNorm      → 由 Block 的 FSDP 管理
│   └── MoE
│       ├── Gate        → 由 Block 的 FSDP 管理（Gate 权重对所有 rank 可见）
│       └── Experts ──── FSDP(ep_shard_mesh, Shard(1))  [独立管理]
├── LM Head ──── FSDP(world_mesh)
└── 最外层 ──── FSDP(world_mesh)  [根级 wrap]
```

**设计决策：为什么 Gate 不参与 EP 切分？**

Gate 模块（Router）负责计算每个 token 应该被路由到哪些专家。这个决策必须在所有 rank 上保持一致——否则不同 rank 会对同一个 token 做出不同的路由选择，导致通信不一致。因此，Gate 的权重在所有 rank 上是完整复制的，由 world_mesh 的 FSDP 统一管理。

---

## 4. 并行化编排的依赖链

### 4.1 四步编排及其严格顺序

Automodel 的并行化遵循 `CP → EP → AC → FSDP` 的固定顺序。**这个顺序不是约定俗成，而是由技术依赖关系强制决定的**。

```
Step 1: Context Parallelism (CP)
  ↓ CP 不改变参数形状，只设置通信组
  ↓ 必须最先做：AC 和 FSDP 会修改模块结��，之后 CP 无法正确注入

Step 2: Expert Parallelism (EP)
  ↓ 将专家参数转为 DTensor，沿 dim=0 切分
  ↓ 必须在 FSDP 之前：FSDP 的 Shard(1) 和 ignored_params 依赖 EP 的结果
  ↓ 对于 DeepEP 后端，还会初始化通信调度器

Step 3: Activation Checkpointing (AC)
  ↓ 用 checkpoint wrapper 包裹每个 Block
  ↓ 必须在 EP 之后：EP 改变了模块内部结构（DTensor 参数）
  ↓ 必须在 FSDP 之前：FSDP 的 fully_shard 不兼容已被 checkpoint 包裹的模块

Step 4: FSDP
  ↓ 最后做：需要知道哪些参数被 EP 管理了（构造 ignored_params）
  ↓ FSDP wrap 改变参数存储方式（FlatParam），之后不能再做 EP 切分
```

### 4.2 关键依赖关系详解

**EP → FSDP 依赖**

EP 切分产生了两个关键信息，FSDP 必须使用：
1. **哪些参数是专家参数**：FSDP 需要知道这些参数的集合，才能构造 ignored_params
2. **专家参数已经是 DTensor**：FSDP 对专家做 Shard(1) 时，参数已经是 Shard(0) 的 DTensor，FSDP 是在其基础上叠加第二层切分

如果颠倒顺序——先做 FSDP 再做 EP——FSDP 会将专家参数沿默认维度切分并转为 FlatParam，之后 EP 无法再对其做 Shard(0) 切分。

**AC → FSDP 依赖**

PyTorch FSDP2 的 fully_shard 内部会注册前向和反向的钩子函数。如果先 FSDP wrap 再做 AC，checkpoint wrapper 会改变模块的 forward 调用路径，可能导致 FSDP 的钩子无法正确触发。反过来——先 AC 再 FSDP——FSDP 在已被 checkpoint 包裹的模块外层做 wrap，钩子注册在正确的位置。

### 4.3 Tensor Parallelism 的禁用

Automodel 的 MoE 并行化器**显式禁用了 TP**。

**原因**：EP 和 TP 的组合（通常称为 EP×TP）需要创建一个二维通信组：TP 组内的 GPU 协同处理同一个专家的矩阵乘法切分，EP 组内的 GPU 各持有不同的专家。这种组合的通信模式极为复杂——需要在 TP All-Reduce 和 EP All-to-All 之间做精确的交错调度。

Automodel 选择不实现 EP×TP，而是依赖 EP + FSDP 来分摊参数显存。如果模型需要 TP（例如非 MoE 的 Attention 层），走的是另一条独立的代码路径。

---

## 5. 通信后端选型：DeepEP 与 NCCL 的取舍

### 5.1 MoE All-to-All 的通信挑战

MoE 训练中的 Token 路由通信具有以下特征，使其不同于常规的 All-Reduce 或 All-Gather：

```
通信特征分析：

特征                    │ 描述                                │ 对通信后端的要求
───────────────────────┼─────────────���──────────────────────┼────────────────────
不规则数据量            │ 每个 GPU 发往不同 rank 的 token      │ 需要高效处理非均匀数据分布
                       │ 数量不同，由 router 动态决定          │
───────────────────────┼────────────────────────────────────┼────────────────────
低延迟要求              │ All-to-All 位于前向和反向的关键路径上  │ kernel launch 开销必须最小化
───────────────────────┼────────────────────────────────────┼────────────────────
双向通信                │ dispatch（发送 token）和              │ 需要 dispatch/combine 配对
                       │ combine（收集结果）成对出现           │ 且 combine 依赖 dispatch 的路由信息
───────────────────────┼────────────────────────────────────┼────────────────────
与排列操作融合            │ token 需要先按专家分组排列，           │ 排列和通信如果分开，额外的 kernel
                       │ 再发送；接收后需要反向排列             │ launch 和内存拷贝成为瓶颈
───────────────────────┼────────────────────────────────────┼────────────────────
跨节点通信               │ 专家通常分布在多个节点上              │ 需要同时利用 NVLink 和 RDMA
```

### 5.2 NCCL 方案的局限性

PyTorch 原生的 All-to-All 通信通过 NCCL 实现，存在以下局限：

1. **排列与通信分离**：使用 NCCL 时，token 排列（permute）是一个独立的 kernel，All-to-All 是另一个 kernel，反向排列（unpermute）又是一个 kernel。三个 kernel 的 launch 开销和中间 buffer 的显存占用不可忽略。

2. **双通道无法差异化**：NCCL 对节点内和节点间通信使用统一的协议栈，无法针对 NVLink（高带宽、低延迟）和 RDMA（相对低带宽、高延迟）做差异化优化。

3. **SM 占用不可控**：NCCL kernel 的 SM 使用量由 NCCL 内部决策，无法精确控制其对计算 kernel 的影响。

### 5.3 DeepEP 方案的优势

DeepEP（DeepSeek 开发的专用 MoE 通信库）针对上述问题做了三项关键优化：

**融合 Kernel**

```
NCCL 方案（3 个独立步骤）：
  Token Permute ──→ NCCL All-to-All ──→ Token Unpermute
  (GPU kernel)      (NCCL kernel)       (GPU kernel)

DeepEP 方案（1 个融合步骤）：
  Fused Dispatch = Permute + All-to-All（一个 kernel 完成排列和发送）
  Fused Combine  = All-to-All + Unpermute（一个 kernel 完成接收和反排列）
```

这种融合消除了中间 buffer 和额外的 kernel launch 开销。

**双通道通信**

DeepEP 的 Buffer 抽象显式区分了 NVLink Buffer 和 RDMA Buffer：

```
Buffer 架构：

  Buffer
  ├── NVLink Buffer（节点内通信）
  │   └── 利用 NVLink 的高带宽做节点内 GPU 间的 token 交换
  │   └── buffer 大小由 dispatch/combine config 的 hint 函数动态计算
  └── RDMA Buffer（节点间通信）
      └── 绕过 CPU，GPU 直接通过 RDMA 网卡与远端 GPU 通信
      └── buffer 大小同样由 config hint 动态计算
```

**SM 数量控制**

DeepEP 允许通过环境变量精确指定分配给通信操作的 SM 数量（默认 20 个）。这使得训练系统可以在通信延迟和计算吞吐之间做精确的权衡：分配更多 SM 给通信可以降低 All-to-All 延迟，但会减少可用于矩阵乘法的 SM 数量。

### 5.4 Buffer 的生命周期管理

DeepEP 的 Buffer 采用**全局单例 + 惰性分配 + 按需扩展**的策略：

```
Buffer 生命周期：

首次调用：
  1. 根据 group size 和 hidden size 计算所需的 NVLink/RDMA buffer 大小
  2. 分配 Buffer 实例，注册到全局单例
  3. 后续所有 MoE 层共用此 Buffer

后续调用：
  4. 检查现有 Buffer 是否满足当前需求（group、大小）
  5. 如果满足，直接复用
  6. 如果不满足（例如 hidden size 变大），重新分配更大的 Buffer

设计约束：
  - 所有 MoE 层共享一个 Buffer → 减少显存占用
  - 但这意味着不同 MoE 层不能并行通信 → 它们必须串行使用 Buffer
  - Buffer 大小取 dispatch 和 combine 两种 config 的最大值 → 一个 Buffer 同时满足两种操作
```

**设计权衡**：共享 Buffer 节省了显存，但引入了串行约束。对于当前的 MoE 训练（MoE 层按顺序执行），这个权衡是合理的。如果未来需要层间并行（如 interleaved pipeline），则需要重新评估此设计。

### 5.5 适配网络配置的注意事项

DeepEP 有一个重要的运维约束：**网络的自适应路由（Adaptive Routing）必须关闭**。

原因：DeepEP 的 RDMA 通信依赖固定的路由路径。自适应路由会改变数据包的传输路径，导致 DeepEP 的 buffer 管理和同步机制失效。这是一个架构层面的约束，要求部署 Automodel 的集群必须在网络交换机上禁用自适应路由功能。

---

## 6. Token 路由的系统设计

### 6.1 路由数据流的全景视图

Token 路由是 MoE 层的核心数据流，涉及多个系统组件的协作：

```
Token 路由全景数据流：

  Input Tokens [batch × seq_len, hidden_dim]
       │
       ▼
  ┌─────────────────┐
  │   Gate (Router)  │  → 计算路由概率，选出 top-k 专家
  └────────┬────────┘
           │ 产出：token_indices [n_tokens, topk]
           │       token_probs   [n_tokens, topk]
           ▼
  ┌─────────────────────┐
  │  Dispatch Phase      │  → 把 token 发送到持有目标专家的 GPU
  │  (All-to-All 发送)   │
  └────────┬────────────┘
           │ 产出：handle（记录路由映射）
           │       dispatched_tokens（到达目标 GPU 的 token）
           ▼
  ┌─────────────────────┐
  │  Expert Compute      │  → 每个 GPU 用本地专家处理收到的 token
  │  (Grouped GEMM)      │
  └────────┬────────────┘
           │ 产出：expert_output（专家计算结果）
           ▼
  ┌─────────────────────┐
  │  Combine Phase       │  → 把结果发回 token 的源 GPU
  │  (All-to-All 接收)   │     消费 handle 中的路由映射信息
  └────────┬────────────┘
           │
           ▼
  Output Tokens [batch × seq_len, hidden_dim]
```

### 6.2 Handle 机制的设计逻辑

Handle 是 Dispatch 和 Combine 之间的**状态桥梁**。

**问题**：Dispatch 阶段将 token 从源 GPU 发送到目标 GPU。Combine 阶段需要把专家计算的结果发回源 GPU。但 Combine 操作本身并不知道"哪个结果应该发回哪个 GPU"——这个信息只有 Dispatch 阶段知道。

**解决方案**：Dispatch 操作产生一个 Handle 对象，记录了完整的路由映射：

```
Handle 记录的信息：

  - 哪些 token 发给了哪个 rank（正向映射）
  - 每个 rank 发送/接收了多少 token
  - token 在源 GPU 和目标 GPU 上的排列顺序

Dispatch（产生 Handle）──→ [Expert 计算] ──→ Combine（消费 Handle）

Handle 的生命周期：
  1. Dispatch.forward 创建 handle，保存到 autograd context
  2. Expert 计算期间，handle 持有路由状态
  3. Combine.forward 使用 handle 做反向路由
  4. Combine 完成后，handle 被释放
```

### 6.3 前向-反向的对称性

MoE 路由的自动微分设计利用了 All-to-All 的一个数学性质：**All-to-All 的反向传播就是方向相反的 All-to-All**。

```
前向传播：
  Dispatch.forward = buffer.dispatch  （源 GPU → 目标 GPU）
  Combine.forward  = buffer.combine   （目标 GPU → 源 GPU）

反向传播：
  Dispatch.backward = buffer.combine  （梯度从目标 GPU → 源 GPU）
  Combine.backward  = buffer.dispatch （梯度从源 GPU → 目标 GPU）

对称关系：
  ┌──────────────────────────────────────────────────────┐
  │  Forward:   dispatch ──→ expert ──→ combine          │
  │  Backward:  combine  ←── expert ←── dispatch         │
  │                                                      │
  │  dispatch 的反向 = combine （收集梯度回源）            │
  │  combine 的反向  = dispatch（分发梯度到专家）          │
  └──────────────────────────────────────────────────────┘
```

这种对称性意味着框架只需要实现两个基本操作（dispatch 和 combine），就能覆盖前向和反向的全部四个通信步骤。每个操作都封装为 PyTorch 的 `autograd.Function`，确保梯度自动正确传播。

### 6.4 异步完成与流同步

DeepEP 支持异步通信模式，通过 CUDA Event 机制实现通信与计算的重叠：

```
同步模式：
  dispatch() ────等待完成────→ expert_compute()

异步模式：
  dispatch() ──→ 立即返回 ──→ expert_compute()
       │                           ↑
       └── Event 信号 ─────────────┘
           （GPU 自动在 Event 完成后开始计算）

设计考虑：
  - 异步模式减少了 CPU 端的空转等待
  - 但需要在正确的时机做 stream 同步（current_stream_wait）
  - 框架在每次异步操作后立即插入同步点，确保后续操作的正确性
```

### 6.5 通信管理器的共享设计

所有 MoE 层共享一个通信管理器实例。

**动机**：如果每个 MoE 层各自创建独立的管理器实例，每个实例都会持���自己的 Buffer 引用和 Handle 状态，显存占用将线性增长。

**实现策略**：通过类级别的共享变量（而非实例变量），第一个创建的管理器实例被注册为共享实例，后续所有 MoE 层复用同一个实例。

```
共享管理器的约束：

  ┌─────────────────────────────────────────────────────┐
  │  MoE Layer 0  ─┐                                    │
  │  MoE Layer 1  ──┤── 共享 CommManager ── 共享 Buffer  │
  │  MoE Layer 2  ──┤   (同一个实例)       (同一个实例)   │
  │  ...          ──┘                                    │
  │                                                      │
  │  约束：不同层必须串行使用管理器                          │
  │  原因：管理器内部的 handle 状态一次只能服务一个层         │
  │  影响：MoE 层不能并行通信（但可以并行计算本地专家）       │
  └─────────────────────────────────────────────────────┘
```

### 6.6 索引格式转换的必要性

DeepEP 的 dispatch 返回的路由信息使用 **indices 格��**（每个 token 的 top-k 专家 ID 列表），而后续的 token 排列/反排列操作使用 **multihot 格式**（每个 token 对每个本地专家的 0/1 标记矩阵）。

```
格式对比：

Indices 格式（紧凑，DeepEP 输出）：
  token 0: [expert_2, expert_5]      ← top-2 专家 ID
  token 1: [expert_1, expert_3]

Multihot 格式（稀疏，permute/unpermute 需要）：
  token 0: [0, 0, 1, 0, 0, 1, 0, 0]  ← 每个本地专家一个 bit
  token 1: [0, 1, 0, 1, 0, 0, 0, 0]

为什么需要转换？
  - DeepEP 内部使用 indices 格式（compact，适合通信）
  - permute/unpermute 使用 multihot 格式（scatter/gather 操作需要按专家维度索引）
  - 两种格式各有优势，转换是不可避免的桥接操作
```

转换操作通过 GPU 上的并行 kernel 实现，每个线程处理一个 token 的转换，避免了 CPU 端的瓶颈。同时生成一个 **position_map**（位置映射），用于在反向传播中将梯度从 multihot 格式正确映射回 indices 格式。

---

## 7. 两种专家计算范式的架构权衡

Automodel 提供两种专家计算实现，适用于不同的部署场景。选择不在运行时动态发生，而是在模型初始化时由配置决定。

### 7.1 范式一：All-Gather / Reduce-Scatter（标准实现）

**通信模式**：

```
每个 GPU 的视角：

  1. All-Gather：收集所有 GPU 上的 token
     本地 token → [All-Gather] → 全局 token（每个 GPU 有完整副本）

  2. 本地专家计算：
     全局 token + 本地专家权重 → 遍历本地专家 → 计算结果

  3. Reduce-Scatter：聚合结果并分发
     计算结果 → [Reduce-Scatter] → 每个 GPU 得到属于自己 token 的最终结果
```

**设计特点**：

- **通信与计算完全分离**：All-Gather 和 Reduce-Scatter 使用 DTensor 的语义操作实现，与专家的矩阵乘法是独立的 kernel
- **逐专家 for 循环**：对每个本地专家，先找出路由到该专家的 token，再做矩阵乘法。当本地专家数量多时（例如 16-32 个），循环开销不可忽略
- **利用 PyTorch 编译器优化**：激活函数（如 SwiGLU）可以通过编译器融合来减少 kernel 数量

**适用场景**：

- 单机训练或小规模集群
- 调试和验证阶段
- 不需要 RDMA 支持的环境
- 需要最大化与 PyTorch 生态的兼容性

### 7.2 范式二：Fused All-to-All + Grouped GEMM（DeepEP 实现）

**通信模式**：

```
每个 GPU 的视角：

  1. Fused Dispatch：排列 + 发送（一个 kernel）
     本地 token → [按目标专家分组 + All-to-All 发送] → 到达目标 GPU

  2. Grouped GEMM 计算：
     收到的 token（已按专家分组）→ 批量矩阵乘法 → 计算结果

  3. Fused Combine：接收 + 反排列（一个 kernel）
     计算结果 → [All-to-All 接收 + 恢复原始顺序] → 返回源 GPU
```

**设计特点**：

- **通信与排列融合**：Dispatch 在一个 kernel 中完成 token 的按专家分组排列和跨 GPU 发送，消除了中间 buffer
- **Grouped GEMM**：不再逐专家循环，而是利用分组矩阵乘法一次性处理所有本地专家。输入按 `tokens_per_expert` 划分，每个分组使用对应专家的权重
- **显式处理空专家**：当某个 GPU 上的某些本地专家没有收到任何 token 时，需要一个 **dummy 计算**来保持梯度图的连通性。不执行这个 dummy 计算会导致反向传播中 reduce-scatter 的 hang（因为部分 rank 没有梯度贡献）

**适用场景**：

- 大规模多节点训练
- 需要最大化通信效率的生产环境
- GPU 集群具备 RDMA 网络支持

### 7.3 架构权衡对比

```
┌────────────────────┬─────────────────────────┬─────────────────────────────┐
│     维度           │ 标准实现                  │ DeepEP 实现                  │
├────────────────────┼─────────────────────────┼─────────────────────────────┤
│ 通信抽象           │ DTensor 语义              │ DeepEP Buffer 直接操作       │
│                    │ (high-level，可移植)       │ (low-level，高性能)          │
├────────────────────┼─────────────────────────┼─────────────────────────────┤
│ Kernel 数量        │ 多（排列、通信、计算分离）   │ 少（排列+通信融合，grouped   │
│                    │                          │ GEMM 批处理）               │
├────────────────────┼─────────────────────────┼─────────────────────────────┤
│ 显存峰值           │ 较高（All-Gather 产生       │ 较低（只发送目标 token，      │
│                    │ 全局副本）                 │ 无全局副本）                 │
├────────────────────┼─────────────────────────┼─────────────────────────────┤
│ 通信量             │ O(n × hidden) 全量广播     │ O(n × hidden / ep_size)     │
│                    │ + O(n × hidden) 归约       │ 只发送目标 token             │
├────────────────────┼─────────────────────────┼─────────────────────────────┤
│ 计算效率           │ 逐专家循环，矩阵较小         │ Grouped GEMM，矩阵批量       │
│                    │ （无法充分利用 GPU SM）      │ （充分利用 GPU 计算资源）     │
├────────────────────┼─────────────────────────┼─────────────────────────────┤
│ 依赖项             │ 仅 PyTorch                │ PyTorch + DeepEP +          │
│                    │                          │ RDMA 网络 + 关闭自适应路由    │
├────────────────────┼───────���─────────────────┼─────────────────────────────┤
│ 回退策略           │ 无需回退（默认实现）         │ 单 GPU 时自动回退到标准实现   │
└────────────────────┴─────────────────────────┴─────────────────────────────┘
```

**关键设计决策**：当使用 DeepEP 后端但只有单个 GPU 时，系统自动回退到标准实现。这是因为 DeepEP 的通信管理器需要多 GPU 的进程组才能初始化——在单 GPU 上强行使用 DeepEP 既无通信收益，又引入不必要的初始化开销。

---

## 8. 梯度同步的分层控制

### 8.1 问题：为什么 MoE 的梯度同步更复杂

标准 FSDP 的梯度同步逻辑相对简单：每个 backward 结束后，FSDP 自动执行 Reduce-Scatter 归约梯度并释放全量参数。但 MoE 训练引入了两个复杂因素：

**因素一：梯度积累**

训练中经常使用梯度积累来模拟更大的 batch size。在积累的中间步（非最后一步），执行梯度同步是浪费的——通信开销不产生任何收益，因为中间步的梯度还不完整。

**因素二：双 FSDP 实例**

MoE 模型有两个 FSDP 管理域：普通参数（world_mesh）和专家参数（ep_shard_mesh）。这两个域的梯度同步必须同时启用或禁用，否则会导致状态不一致。

### 8.2 分层控制架构

Automodel 设计了一个两级控制机制来管理梯度同步：

```
┌─────────────────────────────────────────────────────────────┐
│                   训练循环（外层控制）                         │
│                                                             │
│   for step in gradient_accumulation_steps:                  │
│       if step == 0:                                         │
│           prepare_for_grad_accumulation()                   │
│           → 禁用所有 FSDP 模块的 sync/reshard               │
│       if step == last:                                      │
│           prepare_for_final_backward()                      │
│           → 启用所有 FSDP 模块的 sync/reshard               │
│       backward()                                            │
│   optimizer.step()                                          │
│                                                             │
├──��──────────────────────────────────────────────────────────┤
│              FSDP 模块级别（内层控制）                        │
│                                                             │
│   每个 FSDP 模块有三个独立控制位：                             │
│                                                             │
│   ① requires_gradient_sync:                                │
│      控制 backward 结束后是否执行 Reduce-Scatter 梯度归约      │
│      禁用 → 梯度只在本地 GPU 上累加，不通信                   │
│                                                             │
│   ② reshard_after_backward:                                │
│      控制 backward 结束后是否重新切分参数（释放全量参数的显存） │
│      禁用 → 参数保持全量状态，下一步 forward 无需 All-Gather   │
│                                                             │
│   ③ is_last_backward:                                      │
│      标记当前是否是最后一次 backward                          │
│      影响 FSDP 内部的 post_backward 清理逻辑                  │
└─────────────────────────────────────────────────────────────┘
```

### 8.3 遍历逻辑的全覆盖

控制梯度同步状态时，必须遍历模型中**所有**被 FSDP wrap 的模块。遍历逻辑覆盖四类模块：

```
遍历目标：

1. 模型根模块   → 如果被 FSDP wrap，控制其同步状态
2. Embedding     → 独立 FSDP wrap 的 embed_tokens
3. LM Head      → 独立 FSDP wrap 的 lm_head
4. 专家模块     → 每个 Transformer Block 中 MoE 的 experts（如果被独立 FSDP wrap）

为什么需要覆盖专家模块？

  专家有独立的 FSDP 实例（ep_shard_mesh）。如果只遍历 Block 级 FSDP 模块
  而忽略专家的 FSDP 模块，会导致：
  - Block 的 FSDP 已禁用同步
  - 但专家的 FSDP 仍在做同步
  - 结果：中间步的 backward 中，专家参数的梯度被意外归约，破坏了梯度积累
```

### 8.4 Pipeline 并行的特殊处理

当模型使用 Pipeline Parallelism（PP）时，梯度同步的控制变得更加复杂。原因是 PP 中每个 micro-batch 的 backward 都是独立调度的，PyTorch 的 Pipeline 引擎需要在每个 backward 调用中决定是否做 FSDP 同步。

**问题**：PyTorch 的 Pipeline Stage 内部的 backward 处理函数不了解 MoE 的梯度同步需求。

**解决方案**：Automodel 通过函数注入（monkey-patch）替换了 Pipeline Stage 的 backward 处理逻辑，注入 MoE 感知的梯度同步控制。

```
PP 感知的梯度同步流程：

1. 每次 backward 开始前：
   - 禁用所有 FSDP 模块的 sync/reshard（默认不同步）

2. 执行 backward

3. backward 完成后，判断是否需要同步：
   - 条件 A：这是 PP 调度中的最后一个 micro-batch 的 backward（last_backward）
   - 条件 B：这是梯度积累的最后一步（is_optim_step 全局标志）
   - 只有 A ∧ B 同时满足时，才触发梯度同步和 post_backward 清理

全局标志机制：
  - 训练循环在 optimizer.step() 之前的最后一个 backward 之前设置 is_optim_step = True
  - backward 完成后设置 is_optim_step = False
  - Pipeline Stage 的 patched backward 检查此标志来决定同步时机
```

**为什么需要两个条件的合取（AND）？**

考虑一个 4 步梯度积累 + 2 stage PP 的场景：

```
Step 1, Stage 0: backward (micro 0) → 不同步（既不是最后 micro-batch，也不是最后积累步）
Step 1, Stage 0: backward (micro 1) → 不同步（是最后 micro-batch，但不是最后积累步）
Step 2, Stage 0: backward (micro 0) → 不同步
Step 2, Stage 0: backward (micro 1) → 不同步
Step 3, Stage 0: backward (micro 0) → 不同步
Step 3, Stage 0: backward (micro 1) → 不同步
Step 4, Stage 0: backward (micro 0) → 不同步（是最后积累步，但不是最后 micro-batch）
Step 4, Stage 0: backward (micro 1) → 同步！（是最后 micro-batch 且是最后积累步）
```

只有在梯度积累完成 + micro-batch 调度完成的那个确切时刻，才触发同步。过早同步会浪费通信，过晚同步会导致 optimizer 使用不完整的梯度。

### 8.5 优化收益

```
梯度积累优化的通信节省：

假设 gradient_accumulation_steps = 4：

未优化：
  Step 1: backward → Reduce-Scatter(梯度) + All-Gather(重建参数) → 2 次通信
  Step 2: backward → Reduce-Scatter(梯度) + All-Gather(重建参数) → 2 次通信
  Step 3: backward → Reduce-Scatter(梯度) + All-Gather(重建参数) → 2 次通信
  Step 4: backward → Reduce-Scatter(梯度) + All-Gather(重建参数) → 2 次通信
  总通信：8 次集合通信

优化后：
  Step 1: backward → 无通信
  Step 2: backward → 无通信
  Step 3: backward → 无通信
  Step 4: backward → Reduce-Scatter(梯度) + All-Gather(重建参数) → 2 次通信
  总通信：2 次集合通信

节省率：75%（4 步积累时）
通用公式：节省率 = 1 - 1/gradient_accumulation_steps
```

---

## 9. 检查点系统的格式抽象

### 9.1 格式差异的根源

Automodel 内部使用**合并格式**存储专家权重，而 HuggingFace 生态使用**分离格式**。这两种格式的差异来自不同的设计优先级：

```
合并格式（Automodel Native）：

  参数名: gate_and_up_projs
  形状: [n_experts, hidden_dim, 2 × intermediate_dim]
  设计优先级: 计算效率
  原因: 合并后可以做一次 GEMM 同时计算 gate 和 up projection，
        减少 kernel launch 次数。且 dim=0 是专家维度，便于 EP Shard(0)

分离格式（HuggingFace）：

  参数名: experts.{i}.gate_proj.weight
          experts.{i}.up_proj.weight
          experts.{i}.down_proj.weight
  形状: 各自为 [intermediate_dim, hidden_dim] 或 [hidden_dim, intermediate_dim]
  设计优先级: 模型可读性和模块化
  原因: 每个专家的每个投影是独立的参数，便于分析、调试和单独替换
```

### 9.2 双向转换的设计

检查点系统提供了透明的双向格式转换：

**保存时（Native → HuggingFace）**：

```
转换流程：

合并的 gate_and_up_projs [n_experts, hidden_dim, 2×inter_dim]
  │
  ├─ 第一步：DTensor 感知切分
  │   → 如果参数是 DTensor（有 EP 切分），只处理本 rank 持有的专家
  │   → 产出：按专家拆分的本地权重列表 + 全局专家 ID 列表
  │
  ├─ 第二步：维度拆分
  │   → 将 [hidden_dim, 2×inter_dim] 沿最后一维拆为两部分：
  │     gate_proj: [hidden_dim, inter_dim]  →  转置为 [inter_dim, hidden_dim]
  │     up_proj:   [hidden_dim, inter_dim]  →  转置为 [inter_dim, hidden_dim]
  │
  └─ 第三步：重命名
      → gate_and_up_projs → experts.{id}.gate_proj.weight
                            experts.{id}.up_proj.weight

合并的 down_projs [n_experts, inter_dim, hidden_dim]
  │
  ├─ 第一步：同上
  ├─ 第二步：转置 → [hidden_dim, inter_dim]
  └─ 第三步：→ experts.{id}.down_proj.weight
```

**加载时（HuggingFace → Native）**：

```
转换流程：

experts.{i}.gate_proj.weight [inter_dim, hidden_dim]
experts.{i}.up_proj.weight   [inter_dim, hidden_dim]
  │
  ├─ 第一步：按 rank 过滤
  │   → 根据 ep_mesh 计算本 rank 的专家 ID 范围
  │   → 跳过不属于本 rank 的专家（节省内存和 I/O）
  │
  ├─ 第二步：转置并合并
  │   → gate: [inter_dim, hidden_dim] → 转置 → [hidden_dim, inter_dim]
  │   → up:   [inter_dim, hidden_dim] → 转置 → [hidden_dim, inter_dim]
  │   → 沿���后一维 concat → [hidden_dim, 2×inter_dim]
  │
  ├─ 第三步：按专家堆叠
  │   → stack 所有本地专家 → [n_local_experts, hidden_dim, 2×inter_dim]
  │
  └─ 第四步：DTensor 包装
      → 用 moe_mesh 创建 DTensor，placement 为 Shard(0) + Shard(1)
```

### 9.3 DTensor 感知的专家切分

在保存检查点时，专家参数可能是一个**多维 DTensor**——在 EP 维度上 Shard(0)，在 FSDP 维度上 Shard(1)。将其拆分为单个专家的权重需要精细的 placement 调整：

```
DTensor 感知的切分逻辑：

原始参数：
  DTensor on (ep_shard, ep) mesh
  Placements: [Shard(1), Shard(0)]
  含义: dim=0 按 ep 切分，dim=1 按 ep_shard 切分

拆分后的单个专家权重：
  DTensor on (ep_shard,) mesh  ← 移除了 ep 维度
  Placements: [Shard(0)]       ← 原来的 Shard(1) 变成 Shard(0)
  含义: 原来的 dim=1（现在是 dim=0）仍然按 ep_shard 切分

维度调整规则：
  1. 移除 ep 维度对应的 mesh 维度和 placement
  2. 所有 Shard(dim) 中 dim > 0 的，减 1（因为专家维度被移除了）
  3. Shard(0) 变成 Replicate（因为切出的单个专家不再有专家维度的切分）
  4. 如果移除 ep 后没有剩余的有意义的 mesh 维度，退化为普通 Tensor
```

这种精细的 placement 管理确保了即使在复杂的多维 DTensor 场景下，检查点的保存和加载也能保持正确的分布语义。

### 9.4 加载时的 Rank 感知过滤

加载 HuggingFace 格式检查点时，一个重要的优化是**只加载属于本 rank 的专家**：

```
Rank 感知的加载策略：

全量加载（无优化）：
  每个 GPU 读取所有 64 个专家的权重文件 → 只保留本地 16 个 → 丢弃其余 48 个
  问题：浪费 75% 的 I/O 和临时显存

按需加载（优化后）：
  1. 根据 moe_mesh 计算本 rank 的专家范围 [start_expert, end_expert)
  2. 遍历 state_dict 中的 key
  3. 解析出 expert_id
  4. 如果 expert_id 不在 [start_expert, end_expert) 范围内，跳过
  5. 只加载和转换属于本 rank 的专家权重

收益：I/O 量减少为 1/ep_size，临时显存占用同比减少
```

---

## 10. 总结：约束驱动的设计哲学

Automodel 的 EP+FSDP 架构是一系列**约束驱动的设计决策**的产物。每个设计选择都可以追溯到一个具体的技术约束：

### 10.1 核心约束 → 设计决策映射

```
约束 1: 普通参数和专家参数有不同的分布语义
  → 决策: 创建双 DeviceMesh（world_mesh + moe_mesh）
  → 原因: PyTorch DTensor 要求每个参数绑定到唯一的 DeviceMesh

约束 2: EP 已经使用了 dim=0 来切分专家维度
  → 决策: FSDP 切 dim=1（hidden 维度）
  → 原因: 不能在同一维度上做两种不同语义的切分

约束 3: 专家参数不能被两个 FSDP 实例管理
  → 决策: 通过 ignored_params 将专家参数从 Block 级 FSDP 中排除
  → 原因: 双重 FSDP wrap 会导致参数被切分两次

约束 4: FSDP wrap 需要知道哪些参数属于 EP
  → 决策: EP 必须在 FSDP 之前施加（CP → EP → AC → FSDP 顺序）
  → 原因: ignored_params 集合依赖 EP 切分的结果

约束 5: Token 路由通信是 MoE 的性能瓶颈
  → 决策: 使用 DeepEP 替代 NCCL All-to-All
  → 原因: DeepEP 通过排列+通信融合和双通道 buffer 显著降低延迟

约束 6: DeepEP 的 dispatch 和 combine 需要路由元信息
  → 决策: Handle 机制在 dispatch 和 combine 之间传递状态
  → 原因: combine 操作需要知道 token 的来源 GPU，这个信息只有 dispatch 知道

约束 7: All-to-All 的反向传播是反向的 All-to-All
  → 决策: FusedDispatch.backward = buffer.combine, FusedCombine.backward = buffer.dispatch
  → 原因: 数学性质决定，无需额外实现反向通信操作

约束 8: 梯度积累的中间步不需要通信
  → 决策: 分层的 FSDP 同步控制（prepare_for_grad_accumulation / prepare_for_final_backward）
  → 原因: 中间步的梯度同步浪费通信且不影响最终结果

约束 9: PP 中的 backward 调度由 Pipeline 引擎控制
  → 决策: 通过函数注入（monkey-patch）在 Pipeline Stage 的 backward 中注入 MoE 感知逻辑
  → 原因: Pipeline 引擎不了解 MoE 的梯度同步需求，必须在正确的时机（last_backward ∧ is_optim_step）触发同步

约束 10: 内部合并格式与外部分离格式不兼容
  → 决策: 双向检查点格式转换 + DTensor 感知的切分/合并
  → 原因: 需要同时支持高效训练（合并格式）和生态兼容（HuggingFace 分离格式）
```

### 10.2 架构全景图

```
                     物理 GPU 集群
                          │
           ┌──────────────┴──────────────┐
           │                             │
      world_mesh                    moe_mesh
  (pp, dp_rep, dp_shard,         (pp, ep_shard, ep)
       cp, tp)                        │
           │                    ┌─────┴──────┐
           │                    │            │
      普通参数 FSDP           EP 切分      EP FSDP
    Attention/LN/Gate      Shard(0)     Shard(1)
    Embedding/LM Head     专家维度       Hidden 维度
           │                    │            │
           │               ┌───┴────────────┘
           │               │
    ┌──────┴───────────────┴──────┐
    │    parallelize_model         │
    │    CP → EP → AC → FSDP       │
    └──────────────┬───────────────┘
                   │
         ┌─────────┴──────────┐
         │                    │
    标准专家实现          DeepEP 专家实现
   All-Gather/RS         Fused A2A + GMEM
    DTensor 语义          DeepEP Buffer
         │                    │
         │              ┌─────┴──────┐
         │              │            │
         │         Dispatch      Combine
         │         (发送 token)  (收集结果)
         │              │            │
         │         Handle 传递路由状态
         │                    │
    ┌────┴────────────────────┴────┐
    │   梯度同步分层控制             │
    │   无 PP: prepare_for_xxx     │
    │   有 PP: patched_backward    │
    │          + is_optim_step     │
    └──────────────┬───────────────┘
                   │
    ┌──────────────┴───────────────┐
    │   检查点格式双向转换           │
    │   Native ↔ HuggingFace       │
    │   DTensor 感知切分/合并        │
    │   Rank 感知按需加载            │
    └──────────────────────────────┘
```

### 10.3 设计原则提炼

1. **分离关注点**：普通参数和专家参数由不同的 DeviceMesh 和 FSDP 实例管理，互不干扰
2. **正交切分**：EP 切 dim=0，FSDP 切 dim=1，在参数张量的不同维度上独立操作
3. **依赖驱动的执行顺序**：并行策略的施加顺序由技术依赖关系决定，而非人为约定
4. **通信融合**：将可以合并的通信操作（排列+传输）融合为一个 kernel，减少 launch 开销
5. **惰性共享**：Buffer 和管理器采用惰性分配 + 全局共享，在节省显存和简化生命周期之间取得平衡
6. **格式透明**：检查点系统在两种格式之间做透明转换，对上层训练循环和下层存储系统都不暴露格式细节
7. **最小化通信**：通过精确控制梯度同��时机，将梯度积累场景下的通信量减少到 `1/gradient_accumulation_steps`
