# Stage A: 分布式训练生存技能

> 本阶段目标：掌握 PyTorch 分布式训练的基础概念和工具，为后续学习 FSDP2 打下坚实基础。

## 1. 启动方式对比：torchrun vs mp.spawn

PyTorch 提供两种启动多进程的方式：

### torchrun（推荐）

```bash
torchrun --nproc_per_node=2 my_script.py
```

- torchrun 自动设置 `MASTER_ADDR`、`MASTER_PORT`、`RANK`、`LOCAL_RANK`、`WORLD_SIZE` 等环境变量
- 支持弹性训练（elastic training），节点可以动态加入/离开
- 代码中只需 `dist.init_process_group(backend="nccl")` 即可，无需手动传 rank

### mp.spawn

```python
mp.spawn(worker_fn, args=(world_size,), nprocs=world_size, join=True)
```

- 需要手动设置 `MASTER_ADDR` 和 `MASTER_PORT`
- rank 作为参数传给 worker 函数
- 适合本地调试、教学演示，不支持弹性训练

**本教程中**：Stage A 的 `04_fault_diagnosis.py` 使用 mp.spawn（方便单命令运行故障演示），其余脚本均使用 torchrun。

## 2. 环境变量

| 变量 | 含义 |
|------|------|
| `MASTER_ADDR` | 主节点 IP，所有进程通过它进行 rendezvous |
| `MASTER_PORT` | 主节点端口，需确保未被占用 |
| `RANK` | 全局进程编号（0, 1, ..., WORLD_SIZE-1） |
| `LOCAL_RANK` | 节点内进程编号（单机时等于 RANK） |
| `WORLD_SIZE` | 进程总数 |

## 3. init_process_group 做了什么

```python
dist.init_process_group(backend="nccl")
```

这一行代码背后做了两件事：

1. **Rendezvous（会合）**：所有进程通过 MASTER_ADDR:MASTER_PORT 互相发现，交换网络信息
2. **创建 NCCL Communicator**：建立 GPU 间的通信通道，后续所有 collective 操作（AllReduce、AllGather 等）都通过它完成

`backend="nccl"` 表示使用 NVIDIA NCCL 库（GPU 通信首选）。CPU 场景可用 `"gloo"`。

## 4. DistributedSampler

在 DDP 训练中，每个 GPU 应该处理不同的数据子集：

```python
sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
dataloader = DataLoader(dataset, batch_size=8, sampler=sampler)

for epoch in range(num_epochs):
    sampler.set_epoch(epoch)  # 每个 epoch 必须调用，否则 shuffle 不变
    for batch in dataloader:
        ...
```

**关键点**：
- DistributedSampler 将数据集按 rank 均匀切分，每个 rank 只看到 `1/world_size` 的数据
- `set_epoch(epoch)` 保证每个 epoch 的 shuffle 顺序不同
- 如果总样本数不能被 world_size 整除，Sampler 会自动补齐（pad）

## 5. 常见故障排查

### 5.1 进程 hang（卡死）

**症状**：训练突然停住，没有任何输出

**常见原因**：
- 不对称的 collective 调用（如 rank 0 调用了 barrier，rank 1 跳过了）
- 数据加载不均匀导致某些 rank 提前结束

**诊断方法**：
```bash
TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun --nproc_per_node=2 my_script.py
```

### 5.2 NCCL 超时

**症状**：`RuntimeError: NCCL communicator was aborted ... NCCL timeout`

**诊断方法**：
```bash
NCCL_DEBUG=INFO torchrun --nproc_per_node=2 my_script.py
```

### 5.3 端口冲突

**症状**：`RuntimeError: Address already in use`

**解决**：
```bash
# 使用不同端口
torchrun --nproc_per_node=2 --master_port=29501 my_script.py

# 或清理残留进程
kill $(lsof -t -i:29500)
```

### 5.4 GPU 不可见

**症状**：`RuntimeError: CUDA error: invalid device ordinal`

**解决**：
```bash
# 检查可用 GPU
python -c "import torch; print(torch.cuda.device_count())"

# 指定可见 GPU
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 my_script.py
```

## 6. 脚本说明

| 脚本 | 启动方式 | 演示内容 |
|------|---------|---------|
| [01_torchrun_hello.py](codes/01_torchrun_hello.py) | `torchrun` | 打印每个 rank 的 LOCAL_RANK, RANK, WORLD_SIZE, GPU 信息 |
| [02_ddp_training.py](codes/02_ddp_training.py) | `torchrun` | DDP 训练完整 loop：model → DDP wrap → train 10 steps → print loss |
| [03_distributed_sampler.py](codes/03_distributed_sampler.py) | `torchrun` | TensorDataset + DistributedSampler，打印每 rank 拿到的 indices |
| [04_fault_diagnosis.py](codes/04_fault_diagnosis.py) | `mp.spawn` | 故意制造 hang（rank 0 调 barrier，rank 1 跳过），演示超时行为 |

### 运行示例

```bash
# 需要 2+ GPU
torchrun --nproc_per_node=2 codes/01_torchrun_hello.py
torchrun --nproc_per_node=2 codes/02_ddp_training.py
torchrun --nproc_per_node=2 codes/03_distributed_sampler.py
python codes/04_fault_diagnosis.py  # mp.spawn，直接 python 运行
```
