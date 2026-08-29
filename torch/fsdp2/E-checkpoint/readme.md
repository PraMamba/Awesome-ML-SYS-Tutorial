# Stage E: 检查点与弹性恢复

> 本阶段目标：掌握 FSDP2 的分布式检查点机制，理解如何保存、加载、以及跨 GPU 数量恢复。

## 1. 为什么需要分布式检查点

### Naive 方式的问题

```python
# 错误示范: 在 FSDP 环境下直接 torch.save
torch.save(model.state_dict(), "model.pt")  # 每个 rank 只有参数分片！
```

**问题**：
- FSDP 下每个 rank 只持有参数的分片，直接 `torch.save` 保存的是不完整的
- 如果所有 rank 先 AllGather 到完整参数再保存 → 大模型会 OOM
- 不支持弹性恢复（N GPU 保存 → M GPU 加载）

### 正确方式：分布式检查点

FSDP2 使用 sharded state dict + `torch.distributed.checkpoint` (DCP)：
- 每个 rank 保存自己的分片 → 不需要 AllGather → 不会 OOM
- DCP 处理 resharding → 支持不同 GPU 数量加载

## 2. FSDP2 的 State Dict

```python
# FSDP2 的 state_dict() 默认返回 sharded state dict
state_dict = model.state_dict()
# 每个 rank 的 state_dict 包含的是 DTensor（分片的）
```

FSDP2 直接使用 `model.state_dict()` 即可，返回的参数已经是 DTensor 类型，保留了分片信息。

## 3. DCP (Distributed Checkpoint) API

```python
import torch.distributed.checkpoint as dcp

# 保存
dcp.save(
    {"model": model.state_dict()},
    checkpoint_id="/path/to/checkpoint",
)

# 加载
state_dict = {"model": model.state_dict()}  # 需要先创建模型 + fully_shard
dcp.load(state_dict, checkpoint_id="/path/to/checkpoint")
model.load_state_dict(state_dict["model"])
```

**DCP 的优势**：
- 每个 rank 并行保存自己的分片（IO 并行）
- 支持 resharding：N GPU 保存 → M GPU 加载
- 保存的是目录而非单个文件，包含元数据 + 各 rank 的分片文件

## 4. 弹性恢复：N GPU → M GPU

```bash
# 用 2 GPU 训练并保存
torchrun --nproc_per_node=2 codes/03_resharding_checkpoint.py --mode save

# 用 1 GPU 加载（DCP 自动 resharding）
torchrun --nproc_per_node=1 codes/03_resharding_checkpoint.py --mode load

# 用 4 GPU 加载（同样可以）
torchrun --nproc_per_node=4 codes/03_resharding_checkpoint.py --mode load
```

DCP 在加载时自动根据当前 world_size 重新分配分片，不需要手动处理。

## 5. 脚本说明

| 脚本 | 演示内容 |
|------|---------|
| [01_sharded_state_dict.py](codes/01_sharded_state_dict.py) | 训练 → state_dict() 保存 → 重新加载 → 验证参数一致 |
| [02_dcp_save_load.py](codes/02_dcp_save_load.py) | DCP save/load，验证模型输出一致 |
| [03_resharding_checkpoint.py](codes/03_resharding_checkpoint.py) | 支持 --mode save/load，演示不同 world_size 的弹性恢复 |

### 运行示例

```bash
# 需要 2+ GPU
torchrun --nproc_per_node=2 codes/01_sharded_state_dict.py
torchrun --nproc_per_node=2 codes/02_dcp_save_load.py

# Resharding 示例（分步运行）
torchrun --nproc_per_node=2 codes/03_resharding_checkpoint.py --mode save
torchrun --nproc_per_node=2 codes/03_resharding_checkpoint.py --mode load
```
