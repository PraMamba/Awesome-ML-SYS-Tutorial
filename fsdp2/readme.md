# FSDP2 系统学习教程

> 从分布式训练基础到组合并行（FSDP+TP），七个阶段带你系统掌握 PyTorch FSDP2。

## 前置要求

- **硬件**：至少 2 GPU（Stage F, G 需要 4 GPU）
- **软件**：PyTorch >= 2.4（FSDP2 API 最低版本要求）
- **知识**：基本的 PyTorch 训练流程（model → loss → backward → step）

## 学习路线

```
A 分布式基础 (无依赖)
 └─→ B 通信直觉 (依赖 A)
      └─→ C 最小 FSDP2 (依赖 A, B)
           ├─→ D 生产级调参 (依赖 C)
           ├─→ E 检查点 (依赖 C)
           ├─→ F DTensor/DeviceMesh (依赖 C)
           │    └─→ G 组合并行 (依赖 C, F)
```

## 各阶段概览

### [Stage A: 分布式训练生存技能](A-distributed-basics/readme.md)

torchrun 启动、环境变量、init_process_group、DistributedSampler、故障排查。

| 脚本 | 演示内容 |
|------|---------|
| `01_torchrun_hello.py` | 打印每个 rank 的分布式信息 |
| `02_ddp_training.py` | DDP 完整训练 loop (10 steps) |
| `03_distributed_sampler.py` | DistributedSampler 数据切分 |
| `04_fault_diagnosis.py` | 不对称 collective 调用导致 hang |

### [Stage B: 通信直觉](B-communication/readme.md)

AllReduce、AllGather、ReduceScatter，以及 FSDP 的通信模式。

| 脚本 | 演示内容 |
|------|---------|
| `01_allreduce_basics.py` | AllReduce = DDP 梯度同步 |
| `02_allgather_reducescatter.py` | AllReduce ≡ ReduceScatter + AllGather |
| `03_fsdp_comm_intuition.py` | 手动模拟 FSDP 通信 |

### [Stage C: 最小 FSDP2 实践](C-minimal-fsdp2/readme.md)

`fully_shard` API、自底向上包裹、DTensor 属性检查、粒度对比。

| 脚本 | 演示内容 |
|------|---------|
| `01_fully_shard_mlp.py` | MLP + fully_shard 训练 |
| `02_fully_shard_transformer.py` | Transformer + fully_shard 训练 |
| `03_dtensor_verification.py` | 检查参数 DTensor 属性 |
| `04_wrapping_granularity.py` | coarse/medium/fine 粒度对比 |

### [Stage D: 四个生产级调参旋钮](D-production-knobs/readme.md)

reshard_after_forward、MixedPrecisionPolicy、CPUOffloadPolicy、Sharding 粒度。

| 脚本 | 演示内容 |
|------|---------|
| `01_reshard_after_forward.py` | True vs False 显存/速度权衡 |
| `02_mixed_precision.py` | bf16 参数 + fp32 梯度归约 |
| `03_cpu_offload.py` | CPU Offload 显存/速度权衡 |
| `04_sharding_granularity.py` | coarse vs per_layer 粒度实验 |

### [Stage E: 检查点与弹性恢复](E-checkpoint/readme.md)

Sharded state dict、DCP save/load、N GPU 保存 → M GPU 加载。

| 脚本 | 演示内容 |
|------|---------|
| `01_sharded_state_dict.py` | 保存/加载 sharded state dict |
| `02_dcp_save_load.py` | DCP save + load |
| `03_resharding_checkpoint.py` | 弹性恢复 (--mode save/load) |

### [Stage F: DTensor 与 DeviceMesh](F-dtensor-devicemesh/readme.md)

DTensor placement (Shard/Replicate)、1D/2D DeviceMesh、HSDP。

| 脚本 | GPU 需求 | 演示内容 |
|------|---------|---------|
| `01_dtensor_basics.py` | 2+ | DTensor 创建 + redistribute |
| `02_device_mesh.py` | 4+ | 1D/2D mesh + 子 mesh 提取 |
| `03_hsdp_example.py` | 4+ | HSDP 训练验证 |

### [Stage G: 组合并行](G-combined-parallelism/readme.md)

FSDP + TP 2D 并行、DeviceMesh 多维配置。

| 脚本 | GPU 需求 | 演示内容 |
|------|---------|---------|
| `01_fsdp2_tp_2d.py` | 4+ | FSDP + TP 2D 并行训练 |
| `02_multi_dim_mesh.py` | 4+ | 多种 mesh 配置对比 |

## 快速开始

```bash
# 确认 PyTorch 版本
python -c "import torch; print(torch.__version__)"  # 需要 >= 2.4

# 确认 GPU 数量
python -c "import torch; print(f'GPU 数量: {torch.cuda.device_count()}')"

# 从 Stage A 开始
cd A-distributed-basics/
torchrun --nproc_per_node=2 codes/01_torchrun_hello.py
```

## 参考资料

### 仓库内

- [FSDP Training Backend 原理分析](../rlhf/sys-design/readme-2-en.md)
- [Support FSDP2 as A Training Backend for slime](../rlhf/slime/fsdp/readme_en.md)
- [PyTorch Distributed 通信实践](../torch/torch-distributed/readme.md)
- [Deep Dive into DeepSeek MoE with EP on FSDP](../rlhf/sys-design/readme-4-en.md)

### 外部

- [PyTorch FSDP2 官方教程](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [PyTorch DTensor 文档](https://pytorch.org/docs/stable/distributed.tensor.html)
- [TorchTitan: 3D 并行训练](https://github.com/pytorch/torchtitan)
