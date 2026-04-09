# LLM 分布式训练/推理：通信模式完全图解

> 本文系统梳理 LLM 分布式场景中最常用的通信原语（Communication Primitives）。
> 这些模式源自 MPI 的集体通信语义，在 GPU 场景由 NCCL 实现，在 PyTorch 中由 `torch.distributed` 暴露。
> 每个模式都配有 ASCII 图示，帮助直观理解数据在多 GPU 间的流动方式。

---

## 目录

- [0. 统一术语](#0-统一术语)
- [1. Broadcast（广播）](#1-broadcast广播)
- [2. Reduce（归约）](#2-reduce归约)
- [3. All-Reduce（全归约）](#3-all-reduce全归约)
- [4. Scatter（分发）](#4-scatter分发)
- [5. Gather（汇聚）](#5-gather汇聚)
- [6. AllGather（全汇聚）](#6-allgather全汇聚)
- [7. Reduce-Scatter（归约分发）](#7-reduce-scatter归约分发)
- [8. All-to-All（全交换）](#8-all-to-all全交换)
- [9. Barrier（栅栏同步）](#9-barrier栅栏同步)
- [10. Scan（前缀归约）](#10-scan前缀归约)
- [11. Send / Recv（点对点通信）](#11-send--recv点对点通信)
- [12. LLM 中的通信模式组合速查表](#12-llm-中的通信模式组合速查表)
- [附录：关键等价关系](#附录关键等价关系)

---

## 0. 统一术语

在阅读任何通信模式之前，先对齐这几个核心概念：

```
┌─────────────────────────────────────────────────────────────┐
│                   通信组 (Process Group)                     │
│                                                             │
│   rank 0        rank 1        rank 2        rank 3          │
│   ┌─────┐       ┌─────┐       ┌─────┐       ┌─────┐        │
│   │GPU 0│       │GPU 1│       │GPU 2│       │GPU 3│        │
│   └─────┘       └─────┘       └─────┘       └─────┘        │
│                                                             │
│   world_size = 4                                            │
│   root = 指定的某个 rank（部分操作需要）                       │
└─────────────────────────────────────────────────────────────┘
```

| 术语 | 含义 |
|------|------|
| **rank** | 通信组中每个进程/设备的编号，从 0 到 world_size - 1 |
| **world_size** | 通信组内 rank 的总数 |
| **通信组** | 参与同一集体通信的一组 rank。MPI 叫 communicator，PyTorch 叫 ProcessGroup |
| **root** | 某些操作（Broadcast / Reduce / Gather / Scatter）需要指定一个特殊的 rank |

**关键工程约束**：对 NCCL 而言，组内每个 rank 都**必须**参加同一个 collective，且参数必须一致（count、datatype 等），否则会**卡死（hang）**——这是分布式训练中最常见的调试噩梦之一。

---

## 1. Broadcast（广播）

### 一句话

一个人说话，所有人都听到**完全相同**的内容。

### 语义

- root 持有数据 X
- 调用后，**所有 rank** 都得到 X 的完整拷贝

### 图示

```
root = rank 0

通信前：                              通信后：
rank 0: [AAAA]                       rank 0: [AAAA]
rank 1: [    ]         Broadcast     rank 1: [AAAA]
rank 2: [    ]       ──────────→     rank 2: [AAAA]
rank 3: [    ]                       rank 3: [AAAA]
          ↑                                    ↑
     只有 root 有数据               所有人都有一样的数据
```

### LLM 中的典型场景

- **模型初始化**：rank 0 加载好权重后 broadcast 到所有 rank，确保初始参数一致
- **超参数/控制信号同步**：比如广播 "是否提前停止训练" 的布尔标志

### PyTorch 示例

```python
import torch.distributed as dist

if dist.get_rank() == 0:
    tensor = torch.tensor([1.0, 2.0, 3.0], device="cuda")
else:
    tensor = torch.zeros(3, device="cuda")

dist.broadcast(tensor, src=0)
# 现在所有 rank 的 tensor 都是 [1.0, 2.0, 3.0]
```

---

## 2. Reduce（归约）

### 一句话

所有人把自己的数据"汇总"到一个人手里（通常是求和）。

### 语义

- 每个 rank i 有数据 $X_i$
- 用某个算子 $\oplus$（如 SUM / MAX / MIN）做逐元素归约
- **只有 root** 得到结果 $Y = X_0 \oplus X_1 \oplus \cdots \oplus X_{k-1}$

### 图示

```
root = rank 0,  op = SUM

通信前：                              通信后：
rank 0: [1, 2]                       rank 0: [10, 20]  ← 1+2+3+4, 2+4+6+8
rank 1: [2, 4]         Reduce       rank 1: [ ?, ? ]  ← 结果未定义
rank 2: [3, 6]       ──────────→     rank 2: [ ?, ? ]
rank 3: [4, 8]                       rank 3: [ ?, ? ]
          ↑                                    ↑
     每人一份数据                    只有 root 拿到归约结果
```

### LLM 中的典型场景

- 在只需要一个 rank 汇总指标（如 loss / accuracy）时使用
- 实践中 All-Reduce 更常见（因为通常每个 rank 都需要结果）

---

## 3. All-Reduce（全归约）

### 一句话

所有人把数据汇总，然后**每个人**都拿到汇总结果。

### 语义

- 每个 rank i 有数据 $X_i$
- 计算 $Y = X_0 \oplus X_1 \oplus \cdots \oplus X_{k-1}$
- **所有 rank** 都得到同一个 $Y$

### 图示

```
op = SUM

通信前：                              通信后：
rank 0: [1, 2]                       rank 0: [10, 20]
rank 1: [2, 4]       All-Reduce      rank 1: [10, 20]
rank 2: [3, 6]     ──────────────→   rank 2: [10, 20]
rank 3: [4, 8]                       rank 3: [10, 20]
          ↑                                    ↑
     每人一份数据                    每个人都拿到相同的归约结果
```

### 等价分解

```
All-Reduce  =  Reduce（汇总到 root） + Broadcast（root 广播给所有人）
            =  Reduce-Scatter + AllGather    ← 工程中更常用的分解！
```

### LLM 中的典型场景——数据并行梯度同步

这是 LLM 训练中**最高频**的通信模式：

```
数据并行（DP）训练流程：

1. 每张卡拿不同的数据 batch，用相同的模型做 forward + backward
2. 每张卡得到各自的梯度 grad_i
3. All-Reduce(SUM) 得到全局梯度之和
4. 除以 world_size 得到平均梯度
5. 每张卡用相同的平均梯度更新参数 → 模型保持同步

   rank 0: grad_0 ──┐
   rank 1: grad_1 ──┤  All-Reduce(SUM)    所有 rank 得到
   rank 2: grad_2 ──┤ ──────────────────→  grad_0 + grad_1 + grad_2 + grad_3
   rank 3: grad_3 ──┘
```

### 常见实现算法

```
Ring All-Reduce（环形）：

rank 0 → rank 1 → rank 2 → rank 3 → rank 0
  ↑                                      │
  └──────────────────────────────────────┘

数据沿环传递，每步做局部归约。
总通信量：2 × (N-1)/N × 数据大小  ≈ 2 × 数据大小（N 很大时）
优点：带宽利用率高，与 GPU 数量无关
```

---

## 4. Scatter（分发）

### 一句话

一个人把一块大蛋糕**切成等份**，每人分一块。

### 语义

- root 有一个大数据 S，切成 k 个等大小的块
- Scatter 后，rank i 收到第 i 块 S[i]

### 图示

```
root = rank 0

通信前：                                   通信后：
rank 0: [A, B, C, D]  ← 大缓冲区          rank 0: [A]
rank 1: [            ]      Scatter        rank 1: [B]
rank 2: [            ]    ──────────→      rank 2: [C]
rank 3: [            ]                     rank 3: [D]
              ↑                                 ↑
       root 有完整数据                    每人只拿到自己的那一块
```

### 变体：Scatterv（不等块分发）

```
root = rank 0

通信前：                                   通信后：
rank 0: [AA, B, CCC, D]  ← 不等长块       rank 0: [AA ]  ← 2 个元素
rank 1: [              ]    Scatterv       rank 1: [B  ]  ← 1 个元素
rank 2: [              ]  ──────────→      rank 2: [CCC]  ← 3 个元素
rank 3: [              ]                   rank 3: [D  ]  ← 1 个元素

每个 rank 收到的大小可以不同，由 sendcounts[] 和 displs[] 指定
```

### LLM 中的典型场景

- 把一个大 batch 的数据分发给各 rank
- FSDP 中把完整参数的梯度切块分回各 rank（概念上）

---

## 5. Gather（汇聚）

### 一句话

每人交一份作业，**一个人**收齐所有作业。是 Scatter 的逆操作。

### 语义

- 每个 rank i 有数据 $X_i$
- Gather 后，root 得到拼接结果 $Y = [X_0, X_1, \cdots, X_{k-1}]$

### 图示

```
root = rank 0

通信前：                              通信后：
rank 0: [A]                          rank 0: [A, B, C, D]  ← 收齐了！
rank 1: [B]           Gather         rank 1: [B]  ← 不变
rank 2: [C]         ──────────→      rank 2: [C]  ← 不变
rank 3: [D]                          rank 3: [D]  ← 不变
       ↑                                    ↑
  每人一小块                          root 按 rank 顺序拼接
```

### 变体：Gatherv（不等块汇聚）

允许每个 rank 贡献的数据大小不同，root 通过 `recvcounts[]` + `displs[]` 指定接收布局。

### Scatter 与 Gather 的互逆关系

```
       Scatter
root ──────────→ 各 rank 各拿一块
       Gather
各 rank ────────→ root 拼回完整数据

Scatter 和 Gather 互为逆操作
```

---

## 6. AllGather（全汇聚）

### 一句话

每人交一份作业，然后**每个人都拿到所有人的作业合集**。

### 语义

- 每个 rank i 有数据 $X_i$
- AllGather 后，**每个 rank** 都得到 $Y = [X_0, X_1, \cdots, X_{k-1}]$

### 图示

```
通信前：                              通信后：
rank 0: [A]                          rank 0: [A, B, C, D]
rank 1: [B]         AllGather        rank 1: [A, B, C, D]
rank 2: [C]       ──────────────→    rank 2: [A, B, C, D]
rank 3: [D]                          rank 3: [A, B, C, D]
       ↑                                       ↑
  每人一小块                           每人都拿到完整拼接结果
```

### 与 Gather 的区别

```
Gather：    各 rank → root 一人收齐
AllGather： 各 rank → 每个人都收齐    （= Gather + Broadcast）
```

### 变体：AllGatherv

允许每个 rank 贡献的块大小不同。

### LLM 中的典型场景

- **FSDP forward 前的参数恢复**：每个 rank 只存参数碎片，forward 前 AllGather 拼回完整参数
- **张量并行（TP）**：某些分片计算完成后，AllGather 拼回完整激活

```
FSDP 中的 AllGather：

存储状态（省显存）：                    计算状态（forward 时临时恢复）：
rank 0: [参数块 0]                     rank 0: [参数块 0,1,2,3] ← 完整参数
rank 1: [参数块 1]     AllGather       rank 1: [参数块 0,1,2,3]
rank 2: [参数块 2]   ──────────→       rank 2: [参数块 0,1,2,3]
rank 3: [参数块 3]                     rank 3: [参数块 0,1,2,3]

计算完成后立即释放其他 rank 的参数，回到存储状态
```

---

## 7. Reduce-Scatter（归约分发）

### 一句话

所有人的数据**先汇总求和，再切块分给每个人**。每人只拿到结果的一部分。

### 语义

- 每个 rank i 有数据 $X_i$（可视为由 k 个等长块组成）
- 先归约：$Z = X_0 \oplus X_1 \oplus \cdots \oplus X_{k-1}$
- 再切块：rank r 得到 $Z[r]$（Z 的第 r 个等长块）

### 图示

```
op = SUM,  每个 rank 的输入有 4 个元素

通信前：                                    通信后：
rank 0: [a0, a1, a2, a3]                  rank 0: [a0+b0+c0+d0]  ← 第 0 块的 SUM
rank 1: [b0, b1, b2, b3]  ReduceScatter   rank 1: [a1+b1+c1+d1]  ← 第 1 块的 SUM
rank 2: [c0, c1, c2, c3]  ─────────────→  rank 2: [a2+b2+c2+d2]  ← 第 2 块的 SUM
rank 3: [d0, d1, d2, d3]                  rank 3: [a3+b3+c3+d3]  ← 第 3 块的 SUM
              ↑                                         ↑
     每人完整数据（4 个元素）                每人只拿到归约结果的 1/4
```

### 与 All-Reduce 的关键关系

```
All-Reduce = Reduce-Scatter + AllGather

拆解过程：
                   Reduce-Scatter                    AllGather
rank 0: [a0..a3] ──────────────→ rank 0: [sum_0] ──────────→ rank 0: [sum_0,1,2,3]
rank 1: [b0..b3] ──────────────→ rank 1: [sum_1] ──────────→ rank 1: [sum_0,1,2,3]
rank 2: [c0..c3] ──────────────→ rank 2: [sum_2] ──────────→ rank 2: [sum_0,1,2,3]
rank 3: [d0..d3] ──────────────→ rank 3: [sum_3] ──────────→ rank 3: [sum_0,1,2,3]

先各拿一块归约结果 → 再 AllGather 拼回完整结果 = All-Reduce
```

**这个分解极其重要**，因为它让通信可以和计算 **overlap（重叠）**——在 FSDP / ZeRO 中被大量使用。

### LLM 中的典型场景

- **FSDP backward 后的梯度聚合**：计算出完整梯度后，Reduce-Scatter 将梯度归约并切块分回各 rank（每个 rank 只保留自己负责的那块梯度）
- **张量并行 + 序列并行**：Megatron-LM 在某些层用 Reduce-Scatter 替代 All-Reduce，配合序列并行的激活分片

```
FSDP 中的 Reduce-Scatter（backward 阶段）：

各 rank 独立算出完整梯度：              Reduce-Scatter 后：
rank 0: [grad 完整]                    rank 0: [聚合后的梯度块 0] ← 只存自己负责的
rank 1: [grad 完整]  ReduceScatter     rank 1: [聚合后的梯度块 1]
rank 2: [grad 完整]  ─────────────→    rank 2: [聚合后的梯度块 2]
rank 3: [grad 完整]                    rank 3: [聚合后的梯度块 3]

效果：梯度被归约（求和/平均）的同时完成了分片，省显存
```

---

## 8. All-to-All（全交换）

### 一句话

每个人给每个人发**不同的东西**，所有人同时收到来自所有人的专属包裹。是一种"分布式转置"。

### 语义

- 每个 rank i 把自己的数据切成 k 块：$[S_{i \to 0}, S_{i \to 1}, \cdots, S_{i \to k-1}]$
- $S_{i \to j}$ 表示 rank i 发给 rank j 的那一块
- All-to-All 后，rank j 收到 $[S_{0 \to j}, S_{1 \to j}, \cdots, S_{k-1 \to j}]$

MPI 的经典一句话定义：**"进程 i 发出的第 j 块，会被进程 j 接收，并放在接收缓冲区的第 i 块位置。"**

### 图示

```
每个 rank 有 4 块数据，分别要发给 4 个不同的 rank：

通信前（按"我要发给谁"排列）：          通信后（按"谁发给我的"排列）：
rank 0: [A→0, A→1, A→2, A→3]         rank 0: [A→0, B→0, C→0, D→0]
rank 1: [B→0, B→1, B→2, B→3]         rank 1: [A→1, B→1, C→1, D→1]
rank 2: [C→0, C→1, C→2, C→3]         rank 2: [A→2, B→2, C→2, D→2]
rank 3: [D→0, D→1, D→2, D→3]         rank 3: [A→3, B→3, C→3, D→3]

换个方式理解——这就是矩阵转置！

         发送矩阵（行=发送者）                接收矩阵（行=接收者）
         to_0  to_1  to_2  to_3              from_0 from_1 from_2 from_3
rank 0 [ A→0   A→1   A→2   A→3 ]    rank 0 [ A→0   B→0    C→0    D→0 ]
rank 1 [ B→0   B→1   B→2   B→3 ]    rank 1 [ A→1   B→1    C→1    D→1 ]
rank 2 [ C→0   C→1   C→2   C→3 ]    rank 2 [ A→2   B→2    C→2    D→2 ]
rank 3 [ D→0   D→1   D→2   D→3 ]    rank 3 [ A→3   B→3    C→3    D→3 ]

接收矩阵 = 发送矩阵的转置
```

### 与其他 collective 的对比

```
AllGather：每个 rank 给所有人发同一块数据（"我给你们每人一份相同的东西"）
All-to-All：每个 rank 给不同人发不同的数据（"我给你们每人一份不同的东西"）

AllGather 是 All-to-All 的特例（所有发送块相同时）
```

### 变体

| 变体 | 区别 |
|------|------|
| **Alltoallv** | 每对 (i, j) 之间传输的数据大小可以不同，由 `sendcounts[]` + `sdispls[]` 指定。MoE 中 Token 路由不均衡时必须用这个 |
| **Alltoallw** | 最通用形式：不仅大小可变，每对之间的 datatype 也可以不同，位移以字节为单位。MPI 称之为 "most general form of complete exchange" |

### LLM 中的核心场景——MoE 的 Token Dispatch

这是 All-to-All 在 LLM 中**最重要**的用途：

```
MoE Expert Parallelism 的 Token 流动：

假设 4 张 GPU，每张负责不同的 experts：
  GPU 0: Expert 0,1    GPU 1: Expert 2,3
  GPU 2: Expert 4,5    GPU 3: Expert 6,7

Gate 路由结果（每个 Token 选了 Top-2 个 expert）：
  GPU 0 上的 Token A → 需要 Expert 2（在 GPU 1）和 Expert 5（在 GPU 2）
  GPU 1 上的 Token B → 需要 Expert 0（在 GPU 0）和 Expert 7（在 GPU 3）
  GPU 2 上的 Token C → 需要 Expert 1（在 GPU 0）和 Expert 3（在 GPU 1）
  GPU 3 上的 Token D → 需要 Expert 4（在 GPU 2）和 Expert 6（在 GPU 3）

Step 1: Dispatch (All-to-All)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  数据从"按序列位置排列"变成"按专家位置排列"
  GPU 0 把 Token A 发给 GPU 1 和 GPU 2
  GPU 1 把 Token B 发给 GPU 0 和 GPU 3
  ...

Step 2: Expert Compute
━━━━━━━━━━━━━━━━━━━━━━
  每张 GPU 用本地 expert 计算收到的 Token（完全并行，无通信）

Step 3: Combine (All-to-All)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  计算结果通过反向 All-to-All 发回 Token 原来的 GPU
  GPU 0 收回 Token A 的计算结果
  GPU 1 收回 Token B 的计算结果
  ...
```

### 为什么 MoE 的 All-to-All 很有挑战性？

```
挑战 1：N^2 连接数
  4 张 GPU → 16 条连接
  256 张 GPU → 65536 条连接！
  握手开销和长尾延迟成为瓶颈

挑战 2：通信量是数据依赖的
  All-Reduce 的通信量 = 固定（由参数大小决定）
  All-to-All 的通信量 = 取决于 Token 路由分布（数据依赖！）
  如果大量 Token 涌向同一个 expert → 某些 GPU 通信量暴增

挑战 3：跨节点带宽低
  EP 通常需要跨机器通信（RDMA ~50-100 GB/s）
  远不及机内 NVLink（~900 GB/s）

应对方案（DeepEP）：
  - RDMA 直驱，绕过 NCCL 协议栈
  - 计算-通信重叠（Stream-K）
  - EPLB 专家负载均衡
```

---

## 9. Barrier（栅栏同步）

### 一句话

所有人到齐了才能继续走，**不传输任何数据**，只做同步。

### 语义

- 每个 rank 调用 Barrier
- 所有 rank 都到达 Barrier 后，才允许任何 rank 继续执行
- 不涉及数据传输

### 图示

```
时间线 →

rank 0: ──work──|████ Barrier 等待 ████|──继续──→
rank 1: ──work work──|██ Barrier 等 ██|──继续──→
rank 2: ──work work work──| Barrier  |──继续──→  ← 最后到达
rank 3: ──work──|████ Barrier 等待 ████|──继续──→

                                       ↑
                              所有人到齐才放行
```

### LLM 中的典型场景

- 确保所有 rank 完成某个阶段（如 checkpoint 保存）后再进入下一阶段
- 调试时同步所有进程的执行进度
- **注意**：过度使用 Barrier 会严重损害性能，应尽量用隐式同步（集体通信本身就是同步点）

---

## 10. Scan（前缀归约）

### 一句话

每个人拿到"从第 0 个人到自己"的累积结果。

### 语义（Inclusive Scan）

- 每个 rank i 有数据 $X_i$
- rank i 的输出为前缀归约：$Y_i = X_0 \oplus X_1 \oplus \cdots \oplus X_i$

### 图示

```
op = SUM (Inclusive Scan)

通信前：                              通信后：
rank 0: [3]                          rank 0: [3]          = 3
rank 1: [1]           Scan          rank 1: [4]          = 3+1
rank 2: [4]         ──────────→      rank 2: [8]          = 3+1+4
rank 3: [2]                          rank 3: [10]         = 3+1+4+2
       ↑                                      ↑
  每人一个值                         每人拿到从 rank 0 到自己的累加和
```

```
Inclusive vs Exclusive Scan 的区别：

                  rank 0    rank 1    rank 2    rank 3
输入：               3         1         4         2
Inclusive Scan：     3         4         8        10      ← 包含自身
Exclusive Scan：    0         3         4         8      ← 不包含自身
```

### LLM 中的典型场景

- 在 LLM 主干训练中不如其他模式常见
- 用于计算**变长序列拼接时的偏移量**（offset），例如把多条不等长序列 pack 到一个 batch 时，需要知道每条序列的起始位置
- 某些自定义 CUDA kernel 内部会用到 parallel scan

---

## 11. Send / Recv（点对点通信）

### 一句话

两个人之间**直接传话**，不经过集体通信。

### 语义

- `send(tensor, dst)`：当前 rank 把 tensor 发给 rank dst
- `recv(tensor, src)`：当前 rank 从 rank src 接收数据到 tensor

### 图示

```
Send / Recv（阻塞式）：

rank 0: ──send(T, dst=2)──|阻塞，等 rank 2 接收|──继续──→
rank 1: ──────────────────────────────────────────────────→  (不参与)
rank 2: ──recv(T, src=0)──|阻塞，等 rank 0 发送|──继续──→
rank 3: ──────────────────────────────────────────────────→  (不参与)

       rank 0 ═══════════════════→ rank 2
               直接传输 tensor T
```

### 非阻塞变体：Isend / Irecv

```
Isend / Irecv（非阻塞式）：

rank 0: ──isend(T, dst=2)──继续干别的事──wait()──确认完成──→
rank 2: ──irecv(T, src=0)──继续干别的事──wait()──确认完成──→

返回一个 handle，调用 wait() 前不能读写相关 buffer
```

### 与集体通信的区别

```
集体通信（Collective）：            点对点通信（P2P）：
- 组内所有 rank 都必须参与          - 只涉及两个 rank
- 有固定的通信模式                  - 完全灵活，任意 rank 对可通信
- All-Reduce, AllGather 等         - Send/Recv, Isend/Irecv
```

### LLM 中的核心场景——流水线并行（Pipeline Parallelism）

```
Pipeline Parallelism 中的 Send/Recv：

Stage 0 (rank 0)     Stage 1 (rank 1)     Stage 2 (rank 2)     Stage 3 (rank 3)
┌──────────┐         ┌──────────┐         ┌──────────┐         ┌──────────┐
│ Layer 0-9│  send   │Layer10-19│  send   │Layer20-29│  send   │Layer30-39│
│          │────────→│          │────────→│          │────────→│          │
│          │  recv   │          │  recv   │          │  recv   │          │
│          │←────────│          │←────────│          │←────────│          │
└──────────┘ 激活值   └──────────┘ 激活值   └──────────┘ 激活值   └──────────┘
              forward →                                         (计算 loss)
              ← backward                                       ← 梯度回传

每对相邻 stage 之间用 P2P send/recv 传递激活值（forward）和梯度（backward）
```

---

## 12. LLM 中的通信模式组合速查表

不同的并行策略使用不同的通信原语组合：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        LLM 并行策略 × 通信模式                          │
├─────────────────┬───────────────────────────────────────────────────────┤
│                 │                                                       │
│  数据并行 (DP)   │  All-Reduce（梯度同步）                                │
│                 │  = Reduce-Scatter + AllGather                         │
│                 │                                                       │
├─────────────────┼───────────────────────────────────────────────────────┤
│                 │                                                       │
│  FSDP / ZeRO    │  AllGather（forward 前恢复完整参数）                    │
│                 │  Reduce-Scatter（backward 后梯度聚合+分片）             │
│                 │                                                       │
├─────────────────┼───────────────────────────────────────────────────────┤
│                 │                                                       │
│  张量并行 (TP)   │  All-Reduce / AllGather / Reduce-Scatter              │
│                 │  （取决于切分方式：行切/列切/序列并行）                    │
│                 │                                                       │
├─────────────────┼───────────────────────────────────────────────────────┤
│                 │                                                       │
│  专家并行 (EP)   │  All-to-All × 2（Dispatch + Combine）                 │
│                 │  通常是 Alltoallv（变长 Token 路由）                     │
│                 │                                                       │
├─────────────────┼───────────────────────────────────────────────────────┤
│                 │                                                       │
│  流水线并行 (PP) │  P2P Send / Recv（相邻 stage 间传激活/梯度）            │
│                 │                                                       │
├─────────────────┼───────────────────────────────────────────────────────┤
│                 │                                                       │
│  通用场景        │  Broadcast（初始化参数同步）                            │
│                 │  Barrier（阶段同步，如 checkpoint）                     │
│                 │                                                       │
└─────────────────┴───────────────────────────────────────────────────────┘
```

### 一张图看全部通信模式

```
                    ┌─────────────┐
                    │  Broadcast  │  1 → 全部（相同数据）
                    └──────┬──────┘
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
     ┌──────────┐   ┌───────────┐   ┌──────────┐
     │  Scatter  │   │  Reduce   │   │  Gather  │
     │ 1→全部    │   │ 全部→1    │   │ 全部→1   │
     │(不同块)   │   │(归约求和) │   │(拼接)    │
     └────┬─────┘   └─────┬─────┘   └────┬─────┘
          │               │              │
          ▼               ▼              ▼
  ┌──────────────┐ ┌────────────┐ ┌────────────┐
  │Reduce-Scatter│ │ All-Reduce │ │ AllGather   │
  │全部→全部     │ │ 全部→全部  │ │ 全部→全部  │
  │(归约+切块)   │ │(归约,人人有)│ │(拼接,人人有)│
  └──────────────┘ └────────────┘ └────────────┘
          │               ↑              │
          └───────────────┘──────────────┘
          Reduce-Scatter + AllGather = All-Reduce

                    ┌─────────────┐
                    │ All-to-All  │  全部→全部（每人给每人发不同的东西）
                    └─────────────┘

                    ┌─────────────┐
                    │ Send / Recv │  1 ↔ 1（点对点直连）
                    └─────────────┘
```

---

## 附录：关键等价关系

理解这些等价关系对于分析系统性能至关重要：

```
1. All-Reduce = Reduce + Broadcast
   语义上等价，但直接实现通常比拆开更快。

2. All-Reduce = Reduce-Scatter + AllGather          ★ 工程中最重要的分解
   FSDP/ZeRO 正是利用这个分解，把梯度同步拆成两步，
   中间可以插入计算（overlap）。

3. AllGather = Gather + Broadcast
   先收集到 root，再广播给所有人。

4. Reduce-Scatter = Reduce + Scatter
   先归约到 root，再切块分发。

5. All-to-All 是 AllGather 的推广
   AllGather：每人给所有人发相同的块
   All-to-All：每人给不同人发不同的块

6. Scatter 和 Gather 互为逆操作
   Scatter：root 切块分发
   Gather：各 rank 贡献块，root 拼接
```

---

## 参考资料

- [MPI Standard (MPI Forum)](https://www.mpi-forum.org/docs/)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)
- [PyTorch Distributed Documentation](https://pytorch.org/docs/stable/distributed.html)
- [PyTorch Distributed Tutorial](https://pytorch.org/tutorials/intermediate/dist_tuto.html)
- [Megatron-LM 论文 (Shoeybi et al.)](https://arxiv.org/abs/1909.08053)
- [DeepSpeed ZeRO 论文](https://arxiv.org/abs/1910.02054)
- [DeepEP GitHub](https://github.com/deepseek-ai/DeepEP)
- [Megatron Core MoE Documentation](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-reference/distributed.html)
