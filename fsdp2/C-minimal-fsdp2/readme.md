# Stage C: 最小 FSDP2 实践

> 本阶段目标：用最少的代码跑通 FSDP2 训练，理解 `fully_shard` 的核心机制。

## 1. fully_shard 不是 wrapper

与 FSDP1 的 `FullyShardedDataParallel(module)` 不同，FSDP2 的 `fully_shard` 是一个 **in-place 函数**：

```python
# FSDP1 (旧): wrapper 风格
model = FullyShardedDataParallel(model)

# FSDP2 (新): in-place 风格
fully_shard(model)  # model 本身被修改，类型不变
```

`fully_shard` 做了什么：
1. 将模块的参数转换为 `DTensor`（带 Shard placement）
2. 注册 forward/backward hooks 来自动执行 AllGather 和 ReduceScatter
3. 不改变模块的类型（`model` 仍然是原来的类）

## 2. 自底向上的包裹模式

FSDP2 要求自底向上地调用 `fully_shard`：

```python
# 正确: 先子模块，再 root
for layer in model.layers:
    fully_shard(layer)
fully_shard(model)  # root 最后

# 错误: 先 root 再子模块
fully_shard(model)  # 这样 root 的参数已经 shard 了
fully_shard(model.layer1)  # 再 shard 子模块会出问题
```

**直觉**：每次 `fully_shard(module)` 将 module 的直属参数变成一个分片单元。自底向上保证子模块先独立分片，root 只处理自己的直属参数。

## 3. 一个 fully_shard 调用 = 一个 FSDP unit

每次调用 `fully_shard(module)` 定义了一个通信单元：

- Forward 时，这个 unit 的所有参数一起 AllGather
- Backward 时，这个 unit 的所有梯度一起 ReduceScatter
- Forward 结束后（如果 `reshard_after_forward=True`），这个 unit 的参数被释放

因此，包裹粒度直接影响 peak memory：
- 粒度太粗（如整个模型一个 unit）→ AllGather 时所有参数同时在显存中 → peak memory 高
- 粒度太细（如每个 Linear 一个 unit）→ 通信次数多 → overhead 大
- **推荐**：每个 Transformer layer 一个 unit（medium 粒度）

## 4. model(input) vs model.forward(input)

**必须使用 `model(input)` 而非 `model.forward(input)`**。

`model(input)` 会触发 PyTorch 的 hook 机制（`__call__` → forward pre hooks → forward → forward hooks）。FSDP2 依赖这些 hooks 来执行通信。直接调用 `model.forward(input)` 会跳过 hooks，导致参数没有被 AllGather。

## 5. Optimizer 的创建时机

```python
# 正确: fully_shard 之后创建 optimizer
fully_shard(model)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# 错误: fully_shard 之前创建 optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
fully_shard(model)  # 参数引用已变，optimizer 持有的是旧引用
```

> 更多关于 FSDP 原理和生产级用法，参见 [FSDP Training Backend](../../rlhf/sys-design/readme-2-en.md) 和 [Support FSDP2 as A Training Backend for slime](../../rlhf/slime/fsdp/readme_en.md)。

## 6. 脚本说明

| 脚本 | 演示内容 |
|------|---------|
| [01_fully_shard_mlp.py](codes/01_fully_shard_mlp.py) | 3 层 MLP → 自底向上 fully_shard → train 5 steps → print loss |
| [02_fully_shard_transformer.py](codes/02_fully_shard_transformer.py) | 2 层 TransformerEncoder → 每层 fully_shard → root fully_shard → train |
| [03_dtensor_verification.py](codes/03_dtensor_verification.py) | fully_shard 后检查 type(param)=DTensor, param.placements, param.device_mesh |
| [04_wrapping_granularity.py](codes/04_wrapping_granularity.py) | 同模型 3 种粒度（coarse/medium/fine），对比 peak memory 和 step time |

### 运行示例

```bash
# 需要 2+ GPU
torchrun --nproc_per_node=2 codes/01_fully_shard_mlp.py
torchrun --nproc_per_node=2 codes/02_fully_shard_transformer.py
torchrun --nproc_per_node=2 codes/03_dtensor_verification.py
torchrun --nproc_per_node=2 codes/04_wrapping_granularity.py
```
