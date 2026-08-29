# Stage D: 四个生产级调参旋钮

> 本阶段目标：掌握 FSDP2 的四个关键配置参数，理解它们的工程权衡。

## 1. reshard_after_forward

```python
fully_shard(layer, reshard_after_forward=True)   # 默认
fully_shard(layer, reshard_after_forward=False)
```

| 设置 | Forward 后 | Backward 时 | Peak Memory | 通信量 |
|------|-----------|-------------|-------------|--------|
| `True` (默认) | 释放完整参数，只保留分片 | 需要再次 AllGather | 低 | 高（2× AllGather） |
| `False` | 保留完整参数 | 直接使用，无需 AllGather | 高 | 低（1× AllGather） |

**推荐**：大模型默认 `True`（省显存）；显存充裕时用 `False`（省通信、提速度）。

## 2. MixedPrecisionPolicy

```python
from torch.distributed.fsdp import MixedPrecisionPolicy

mp_policy = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,    # 参数存储和计算精度
    reduce_dtype=torch.float32,    # 梯度归约精度
)
fully_shard(layer, mp_policy=mp_policy)
```

| 参数 | 作用 | 典型值 |
|------|------|--------|
| `param_dtype` | 参数的存储和计算精度 | `torch.bfloat16`（比 fp16 更稳定，不需 loss scaling） |
| `reduce_dtype` | 梯度 AllReduce/ReduceScatter 的精度 | `torch.float32`（保证梯度数值精度） |

**典型策略**：`param_dtype=bf16 + reduce_dtype=fp32`，兼顾速度和精度。

## 3. CPUOffloadPolicy

```python
from torch.distributed.fsdp import CPUOffloadPolicy

offload = CPUOffloadPolicy(pin_memory=True)
fully_shard(layer, offload_policy=offload)
```

- 将参数 offload 到 CPU 内存，GPU 只在计算时临时加载
- `pin_memory=True`：使用 pinned memory 加速 CPU↔GPU 数据传输
- **代价**：PCIe 传输延迟，step time 显著增加
- **适用**：模型太大无法完全放入 GPU 显存

## 4. Sharding 粒度

粒度由 `fully_shard` 的调用位置决定：

```python
# 粗粒度: 整个模型一个 FSDP unit
fully_shard(model)

# 推荐粒度: 每层一个 FSDP unit
for layer in model.layers:
    fully_shard(layer)
fully_shard(model)

# 细粒度: 每个子模块一个 FSDP unit
for layer in model.layers:
    fully_shard(layer.attn)
    fully_shard(layer.ff)
    fully_shard(layer)
fully_shard(model)
```

| 粒度 | Peak Memory | 通信 Overhead | 适用场景 |
|------|-------------|--------------|---------|
| 粗 | 最高 | 最低 | 小模型、显存充裕 |
| 中（推荐） | 适中 | 适中 | 生产环境默认选择 |
| 细 | 最低 | 最高 | 超大模型、显存极度紧张 |

## 5. 脚本说明

| 脚本 | 演示内容 |
|------|---------|
| [01_reshard_after_forward.py](codes/01_reshard_after_forward.py) | 同模型 True vs False，测量 peak memory + step time |
| [02_mixed_precision.py](codes/02_mixed_precision.py) | MixedPrecisionPolicy(bf16/fp32)，打印参数 dtype 变化 + memory 节省 |
| [03_cpu_offload.py](codes/03_cpu_offload.py) | CPUOffloadPolicy，测量 GPU peak memory 和速度开销 |
| [04_sharding_granularity.py](codes/04_sharding_granularity.py) | 8 层 Transformer，coarse vs per_layer 粒度实验 |

### 运行示例

```bash
# 需要 2+ GPU
torchrun --nproc_per_node=2 codes/01_reshard_after_forward.py
torchrun --nproc_per_node=2 codes/02_mixed_precision.py
torchrun --nproc_per_node=2 codes/03_cpu_offload.py
torchrun --nproc_per_node=2 codes/04_sharding_granularity.py
```
