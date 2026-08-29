# Stage B: 通信直觉

> 本阶段目标：理解 FSDP 背后的三个核心 collective 操作，建立对 FSDP 通信模式的直觉。

## 1. 三个核心 Collective 操作

### AllReduce

```
Rank 0: [1, 2]          Rank 0: [4, 6]
                  SUM
Rank 1: [3, 4]    →     Rank 1: [4, 6]
```

- **语义**：所有 rank 的数据先 Reduce（求和/求平均），结果广播到每个 rank
- **用途**：DDP 梯度同步 — 每个 rank 独立 backward 得到本地梯度，AllReduce 后每个 rank 拿到一样的全局梯度
- **通信量**：2(N-1)/N × 数据量（接近 2×数据量）

### AllGather

```
Rank 0: [A]              Rank 0: [A, B]

Rank 1: [B]       →      Rank 1: [A, B]
```

- **语义**：每个 rank 有一个分片，AllGather 后每个 rank 拿到所有分片拼接的完整数据
- **用途**：FSDP Forward — 从权重分片恢复完整权重
- **通信量**：(N-1)/N × 数据量

### ReduceScatter

```
Rank 0: [A0, A1]         Rank 0: [A0+B0]    (Reduce 第 0 分片)

Rank 1: [B0, B1]   →     Rank 1: [A1+B1]    (Reduce 第 1 分片)
```

- **语义**：先 Reduce（求和），再 Scatter（分发），每个 rank 只拿到结果的一个分片
- **用途**：FSDP Backward — 梯度求和后分发给对应的权重分片持有者
- **通信量**：(N-1)/N × 数据量

## 2. 关键洞察：AllReduce = ReduceScatter + AllGather

这不仅是数学等价，NCCL 内部实现 AllReduce 时就是这样分解的（Ring AllReduce）。

理解这个等价关系对理解 FSDP 至关重要：
- **DDP** 直接使用 AllReduce 同步梯度
- **FSDP** 将 AllReduce 拆成两步，分别插入 forward 和 backward：
  - Forward: AllGather 权重
  - Backward: ReduceScatter 梯度

## 3. FSDP 的通信模式

```
训练一个 step:

Forward:
  Layer 1: AllGather(W1_shard) → W1_full → matmul → [释放 W1_full]
  Layer 2: AllGather(W2_shard) → W2_full → matmul → [释放 W2_full]
  ...

Backward:
  Layer N: AllGather(WN_shard) → WN_full → 计算梯度 → ReduceScatter(grad) → [释放 WN_full]
  Layer N-1: ...
  ...

Update:
  每个 rank 只更新自己的权重分片（使用对应的梯度分片）
```

**显存节省的关键**：完整权重只在计算时临时恢复，计算后立即释放。每个 rank 常驻的显存只有 `1/N` 的权重 + `1/N` 的梯度 + `1/N` 的优化器状态。

> 更深入的通信量分析，参见 [Deep Thoughts on RL Systems: FSDP Training Backend](../../../rlhf/sys-design/readme-2-en.md)

## 4. 脚本说明

| 脚本 | 演示内容 |
|------|---------|
| [01_allreduce_basics.py](codes/01_allreduce_basics.py) | 模拟 DDP：每 rank 有"梯度"张量，AllReduce 求和，打印 before/after |
| [02_allgather_reducescatter.py](codes/02_allgather_reducescatter.py) | 分步演示 ReduceScatter → AllGather → 验证等价于 AllReduce |
| [03_fsdp_comm_intuition.py](codes/03_fsdp_comm_intuition.py) | 手动模拟 FSDP：shard weight → AllGather → matmul → ReduceScatter grad |

### 运行示例

```bash
# 需要 2+ GPU
torchrun --nproc_per_node=2 codes/01_allreduce_basics.py
torchrun --nproc_per_node=2 codes/02_allgather_reducescatter.py
torchrun --nproc_per_node=2 codes/03_fsdp_comm_intuition.py
```
