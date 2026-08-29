# Stage F: DTensor 与 DeviceMesh 基础

> 本阶段目标：理解 FSDP2 底层的两个核心抽象 — DTensor 和 DeviceMesh，为组合并行打下基础。

## 1. DTensor：带 Placement 标注的 Tensor

DTensor 是 PyTorch 的分布式张量抽象，它在普通 Tensor 之上增加了两个关键属性：

- **Placement**：描述张量如何分布在设备上
- **DeviceMesh**：描述设备的逻辑组织

```python
from torch.distributed.tensor import DTensor, Shard, Replicate

# DTensor 看起来像普通 Tensor，但实际上分布在多个设备上
dt = DTensor.from_local(local_tensor, mesh, placements=[Shard(0)])
```

**FSDP2 的 `fully_shard` 就是将普通参数转换为 DTensor**。

## 2. Placement 类型

### Shard(dim)

```
完整张量:  [[1, 2, 3, 4],    Shard(0):  Rank 0: [[1, 2, 3, 4]]
            [5, 6, 7, 8]]               Rank 1: [[5, 6, 7, 8]]

                              Shard(1):  Rank 0: [[1, 2],    Rank 1: [[3, 4],
                                                   [5, 6]]            [7, 8]]
```

- 张量沿 `dim` 维度被切分到各设备
- **FSDP 使用 Shard(0) 来分片参数**

### Replicate()

```
完整张量: [1, 2, 3, 4]   →   Rank 0: [1, 2, 3, 4]
                              Rank 1: [1, 2, 3, 4]
```

- 每个设备持有完整副本
- **DDP 的语义：参数全部 Replicate**

### Partial()

```
Rank 0: [1, 2]    Partial(SUM)    完整结果: [4, 6]
Rank 1: [3, 4]       →           (需要 reduce 才能得到)
```

- 每个设备持有部分结果，需要 reduce 操作才能得到完整结果
- **Backward 时梯度的中间状态**

## 3. DeviceMesh

DeviceMesh 是设备的逻辑组织，定义了哪些设备在哪个维度上协作：

```python
from torch.distributed.device_mesh import init_device_mesh

# 1D mesh: 所有设备在一个维度上
mesh_1d = init_device_mesh("cuda", mesh_shape=(4,))
# → 标准 FSDP 或 DDP

# 2D mesh: 设备组织成矩阵
mesh_2d = init_device_mesh("cuda", mesh_shape=(2, 2),
                           mesh_dim_names=("replicate", "shard"))
# → HSDP 或 FSDP+TP
```

### 子 Mesh 提取

```python
mesh_2d = init_device_mesh("cuda", (2, 2), mesh_dim_names=("dp", "tp"))

# 提取子 mesh
dp_mesh = mesh_2d["dp"]   # 数据并行维度的子 mesh
tp_mesh = mesh_2d["tp"]   # 张量并行维度的子 mesh
```

## 4. HSDP：节点内 Shard + 节点间 Replicate

HSDP (Hybrid Sharded Data Parallel) 使用 2D mesh 实现：

```python
# 假设 2 个节点，每节点 2 GPU → 共 4 GPU
mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("replicate", "shard"))
fully_shard(model, mesh=mesh)
```

```
节点 0:  GPU0 ←shard→ GPU1    ← 节点内: 参数分片（高带宽 NVLink）
              ↕ replicate ↕
节点 1:  GPU2 ←shard→ GPU3    ← 节点内: 参数分片（高带宽 NVLink）
         ↑                ↑
         节点间 replicate    ← 梯度同步（低带宽 IB/网络）
```

**HSDP 的优势**：
- 节点内使用 shard 减少每 GPU 显存（利用高带宽 NVLink）
- 节点间使用 replicate 减少跨节点通信量（只需 AllReduce 梯度，不需 AllGather 参数）

## 5. 脚本说明

| 脚本 | GPU 需求 | 演示内容 |
|------|---------|---------|
| [01_dtensor_basics.py](codes/01_dtensor_basics.py) | 2+ | DTensor 创建、Shard/Replicate placement、redistribute |
| [02_device_mesh.py](codes/02_device_mesh.py) | 4+ | 1D 和 2D DeviceMesh 创建、子 mesh 提取 |
| [03_hsdp_example.py](codes/03_hsdp_example.py) | 4+ | 2D mesh + fully_shard = HSDP，train 验证 |

### 运行示例

```bash
# 01 需要 2+ GPU
torchrun --nproc_per_node=2 codes/01_dtensor_basics.py

# 02, 03 需要 4 GPU
torchrun --nproc_per_node=4 codes/02_device_mesh.py
torchrun --nproc_per_node=4 codes/03_hsdp_example.py
```
