# Stage G: 组合并行

> 本阶段目标：理解为什么需要组合并行，掌握 FSDP + TP 的 2D 并行实现。

## 1. 为什么要组合并行

每种并行策略解决不同的瓶颈：

| 并行策略 | 解决的问题 | 局限性 |
|---------|-----------|--------|
| Data Parallel (DDP) | 加速训练（多 GPU 处理更多数据） | 每 GPU 需持有完整模型 |
| FSDP | 减少每 GPU 显存（参数分片） | 模型的单层仍需放入一个 GPU |
| Tensor Parallel (TP) | 单层参数拆分到多 GPU | 通信频繁（每层每次 forward/backward） |
| Pipeline Parallel (PP) | 层间拆分到多 GPU | 引入 bubble，实现复杂 |

**单一策略不够时，需要组合**：

- 模型太大，单层放不下一个 GPU → 需要 TP
- 总参数太多，FSDP 分片后仍显存不够 → 需要 TP + FSDP
- 跨节点通信带宽有限 → 节点内 TP，节点间 FSDP

## 2. 2D 并行：FSDP + TP

2D 并行使用 2D DeviceMesh 统一管理：

```python
# 4 GPU → 2D mesh (dp=2, tp=2)
mesh_2d = init_device_mesh("cuda", (2, 2), mesh_dim_names=("dp", "tp"))
```

```
         tp=0  tp=1
dp=0  [ GPU0  GPU1 ]  ← TP 组 0: linear1 按列拆分，linear2 按行拆分
dp=1  [ GPU2  GPU3 ]  ← TP 组 1: 同上

        ↑      ↑
       FSDP   FSDP     ← 数据并行: GPU0,2 同步梯度; GPU1,3 同步梯度
```

### 应用顺序

**先 TP，后 FSDP**（由内到外）：

```python
# 1. 先做 TP: 层内参数拆分
tp_mesh = mesh_2d["tp"]
for layer in model.layers:
    parallelize_module(layer, tp_mesh, {
        "linear1": ColwiseParallel(),
        "linear2": RowwiseParallel(),
    })

# 2. 再做 FSDP: 数据并行分片
dp_mesh = mesh_2d["dp"]
for layer in model.layers:
    fully_shard(layer, mesh=dp_mesh)
fully_shard(model, mesh=dp_mesh)
```

### ColwiseParallel vs RowwiseParallel

```
ColwiseParallel (linear1):
  权重 W [d_model, 4*d_model] → 按列拆分
  GPU0: W[:, :2*d_model]
  GPU1: W[:, 2*d_model:]
  输出: 每 GPU 得到一半特征 → 无需通信

RowwiseParallel (linear2):
  权重 W [4*d_model, d_model] → 按行拆分
  GPU0: W[:2*d_model, :]
  GPU1: W[2*d_model:, :]
  输出: AllReduce → 每 GPU 得到完整输出
```

**关键搭配**：ColwiseParallel 的输出拼接后 = RowwiseParallel 的输入拆分。两者配合可以减少一次 AllGather。

## 3. 3D 并行概念：FSDP + TP + PP

3D 并行在 2D 基础上增加 Pipeline Parallelism（流水线并行）：

```
3D mesh (pp=2, dp=2, tp=2) → 8 GPU

PP Stage 0:              PP Stage 1:
         tp=0  tp=1              tp=0  tp=1
dp=0  [ GPU0  GPU1 ]   dp=0  [ GPU4  GPU5 ]
dp=1  [ GPU2  GPU3 ]   dp=1  [ GPU6  GPU7 ]
```

3D 并行的实现较复杂，推荐参考 [TorchTitan](https://github.com/pytorch/torchtitan) 项目，它提供了生产级的 3D 并行实现。

## 4. DeviceMesh 统一抽象

DeviceMesh 的优雅之处在于：
- **同一套 API** 支持 1D（FSDP/DDP）、2D（FSDP+TP / HSDP）、3D（FSDP+TP+PP）
- 通过 **子 mesh 提取** 将物理设备映射到逻辑并行维度
- 各并行策略的**通信组自动隔离**，互不干扰

```python
# 同一个 mesh，不同视角
mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("dp", "tp"))
mesh["dp"]  # FSDP 在这个子 mesh 上通信
mesh["tp"]  # TP 在这个子 mesh 上通信
```

## 5. 脚本说明

| 脚本 | GPU 需求 | 演示内容 |
|------|---------|---------|
| [01_fsdp2_tp_2d.py](codes/01_fsdp2_tp_2d.py) | 4+ | 2D mesh → TP(ColwiseParallel/RowwiseParallel) + FSDP fully_shard → train |
| [02_multi_dim_mesh.py](codes/02_multi_dim_mesh.py) | 4+ | 多种 mesh 配置 (4,), (2,2), (1,4)，打印子 mesh 分组 |

### 运行示例

```bash
# 需要 4 GPU
torchrun --nproc_per_node=4 codes/01_fsdp2_tp_2d.py
torchrun --nproc_per_node=4 codes/02_multi_dim_mesh.py
```

## 6. 进阶参考

- [TorchTitan](https://github.com/pytorch/torchtitan): PyTorch 官方的 3D 并行训练框架
- [Support FSDP2 as A Training Backend for slime](../../rlhf/slime/fsdp/readme_en.md): 生产级 FSDP2 用法
- [Deep Thoughts on RL Systems: FSDP Training Backend](../../rlhf/sys-design/readme-2-en.md): FSDP 原理深度分析
