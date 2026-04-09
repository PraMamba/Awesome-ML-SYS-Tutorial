# TorchTitan EP+FSDP 深度源码分析

> 本文基于 TorchTitan 的 `source_code_analysis` 分支，对 Expert Parallelism（EP）与 FSDP 的协同实现进行逐行级别的源码分析。涵盖 DeviceMesh 设计、四种 EP 策略、双层 FSDP 包装、全链路 Prefetch、DeepEP 后端、DualPipeV 以及 torch.compile 适配等全部核心模块。

---

## 目录

- [第 1 节：概述与动机](#第-1-节概述与动机)
- [第 2 节：模型架构 — Llama4 MoE](#第-2-节模型架构--llama4-moe)
- [第 3 节：DeviceMesh 设计 — Dense Mesh vs Sparse Mesh](#第-3-节devicemesh-设计--dense-mesh-vs-sparse-mesh)
- [第 4 节：四种 Expert Parallelism 策略](#第-4-节四种-expert-parallelism-策略)
- [第 5 节：Token Dispatch 与 Combine 详解](#第-5-节token-dispatch-与-combine-详解)
- [第 6 节：FSDP 二次开发 — 双层包装策略](#第-6-节fsdp-二次开发--双层包装策略)
- [第 7 节：全链路 Prefetch 策略](#第-7-节全链路-prefetch-策略)
- [第 8 节：DeepEP 后端详解](#第-8-节deepep-后端详解)
- [第 9 节：torch.compile 与 MoE](#第-9-节torchcompile-与-moe)
- [第 10 节：DualPipeV — PP 与 EP 通信重叠](#第-10-节dualpipev--pp-与-ep-通信重叠)
- [第 11 节：完整并行化流程](#第-11-节完整并行化流程)
- [第 12 节：数值示例](#第-12-节数值示例)

---

## 第 1 节：概述与动机

> **一句话总结**：TorchTitan 为 Llama4 MoE 模型设计了一套完整的 EP+FSDP 协同训练框架，通过四种 EP 策略、双 Mesh 架构、双层 FSDP 包装和全链路 Prefetch 解决了 MoE 模型分布式训练中的核心挑战。

### 1.1 核心挑战

Llama4 MoE 模型采用 **交替 dense/MoE 层** 的设计（详见第 2 节），这意味着同一个模型中同时存在两类需要不同并行策略的层：

1. **Dense 层**（标准 FFN）：参数在 FSDP mesh 上切分，通信模式为 all-gather / reduce-scatter
2. **MoE 层**（路由专家）：专家参数需要在 EP mesh 上按专家维度切分，通信模式为 all-to-all

这两类层共存于同一个 `TransformerBlock` 中，导致以下技术难题：

- **Mesh 不兼容**：Dense 层的 FSDP mesh 维度为 `[dp_replicate, fsdp, tp]`，MoE 层需要 `[dp_replicate, efsdp, ep, etp]`，两者的切分维度不同
- **FSDP 包装粒度**：标准 FSDP 只在 TransformerBlock 级别包装，但 MoE 层内部的 `GroupedExperts` 需要在不同的 mesh 上单独包装
- **Prefetch 失效**：EP 的 all-to-all 通信涉及 D2H（device-to-host）同步，会阻塞 CUDA stream，破坏 FSDP 的隐式 prefetch 机制
- **编译兼容性**：MoE 层的 FSDP 钩子会导致 graph break，需要子模块级编译策略

### 1.2 TorchTitan 的解决方案

TorchTitan 的核心架构贡献包括：

| 模块 | 解决的问题 | 核心文件 |
|------|-----------|---------|
| 四种 EP 策略 | 灵活的专家并行方式 | `expert_parallel.py` |
| 双 Mesh 架构 | Dense/Sparse 层分离 | `parallel_dims.py` |
| 双层 FSDP | 内层专家/外层 Block | `parallelize.py` |
| 全链路 Prefetch | 解决 D2H sync 阻塞 | `parallelize.py:432-497` |
| DeepEP 集成 | 高效通信后端 | `deepep.py` |
| DualPipeV | PP+EP 通信重叠 | `dual_pipe_v.py` |
| 子模块编译 | 避免 graph break | `parallelize.py:593-690` |

### 1.3 入口函数概览

整个并行化从 `parallelize_llama()` 开始（`parallelize.py:69-226`），按以下顺序执行六大步骤：

```
parallelize_llama()
├── 1. apply_non_moe_tp()         # 非 MoE 层的 TP
├── 2. apply_moe_ep_tp()          # MoE 层的 EP/TP
├── 3. apply_cp_to_attention()    # Context Parallelism
├── 4. apply_ac()                 # Activation Checkpointing
├── 5. apply_compile()            # per-block 编译
└── 6. apply_fsdp()              # 双层 FSDP + prefetch
```

> **NOTE**：这个顺序至关重要。TP/EP 必须在 FSDP 之前应用，因为 FSDP 需要感知已经被 TP/EP 切分过的参数。compile 必须在 AC 之后、FSDP 之前，以确保编译单元与 FSDP 包装单元对齐。

---

## 第 2 节：模型架构 — Llama4 MoE

> **一句话总结**：Llama4 通过 `interleave_moe_layer_step` 控制 dense/MoE 层的交替频率，MoE 层由 `TokenChoiceTopKRouter`、`TokenReorderer`、`GroupedExperts` 和可选的共享专家组成。

### 2.1 TransformerBlock 的 MoE 交替设计

`model.py:344-457` 中的 `TransformerBlock` 通过 `moe_enabled` 标志决定使用 MoE 还是 dense FFN：

```python
# model.py:386
self.moe_enabled = (layer_id + 1) % model_args.interleave_moe_layer_step == 0
if self.moe_enabled:
    self.moe = MoE(moe_args, dim=dim, hidden_dim=hidden_dim)
else:
    self.feed_forward = FeedForward(...)
```

**设计含义**：
- 当 `interleave_moe_layer_step = 4` 时，layer 3, 7, 11, ... 是 MoE 层
- 其余层是 dense 层，使用标准 FFN
- 这意味着并行化策略必须 **逐层** 判断，不能一刀切

在 forward 中的分支逻辑（`model.py:443-446`）：

```python
if self.moe_enabled:
    out = h + self.moe(self.ffn_norm(h))
else:
    out = h + self.feed_forward(self.ffn_norm(h))
```

### 2.2 MoE 类层次结构

`moe.py:419-553` 定义了 `MoE` 类，它的子模块层次如下：

```
MoE
├── router: TokenChoiceTopKRouter
│   └── gate: nn.Linear(dim, num_experts)     # 路由门控
├── reorderer: TokenReorderer                  # token 重排序
├── experts: GroupedExperts                     # 路由专家
│   ├── w1: Parameter(num_experts, hidden_dim, dim)
│   ├── w2: Parameter(num_experts, dim, hidden_dim)
│   └── w3: Parameter(num_experts, hidden_dim, dim)
├── shared_experts: FeedForward | None         # 共享专家
│   ├── w1: Linear(dim, hidden_dim * num_shared)
│   ├── w2: Linear(hidden_dim * num_shared, dim)
│   └── w3: Linear(dim, hidden_dim * num_shared)
├── expert_bias: Buffer                        # 负载均衡偏置
└── tokens_per_expert: Buffer                  # 专家使用统计
```

### 2.3 GroupedExperts — 三维权重张量

`GroupedExperts`（`moe.py:131-178`）是 EP 的核心切分对象。它的权重不是标准的 `nn.Linear`，而是 **三维参数张量**：

```python
# moe.py:141-143
self.w1 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))   # (E, H, D)
self.w2 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))   # (E, D, H)
self.w3 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))   # (E, H, D)
```

其中：
- **dim 0**：专家维度 `num_experts`，EP 在此维度切分
- **dim 1/2**：hidden/input 维度，TP 在此维度切分

**DTensor→local 转换**（`moe.py:151-158`）：

```python
def forward(self, x, num_tokens_per_expert):
    if isinstance(self.w1, DTensor):
        # 将 DTensor 转为普通 Tensor，因为 EP 的动态 shape 不易用 DTensor 表达
        w1 = self.w1.to_local()
        w2 = self.w2.to_local()
        w3 = self.w3.to_local()
    else:
        w1, w2, w3 = self.w1, self.w2, self.w3
```

> **NOTE**：这里的 `to_local()` 是关键设计——EP/TP 通过 DTensor 进行参数切分和注册，但在实际计算时转为普通 Tensor，因为 `grouped_mm` 操作需要精确控制 token 的动态分配，DTensor 的自动通信语义在此场景下反而是障碍。

### 2.4 TokenChoiceTopKRouter — 路由与负载均衡

`TokenChoiceTopKRouter`（`moe.py:186-353`）实现了 token-choice top-K 路由：

```python
# moe.py:310-331
scores = self.gate(x)                              # (bs*slen, num_experts)
if self.score_func == "sigmoid":
    scores = torch.sigmoid(scores.to(torch.float32))
elif self.score_func == "softmax":
    scores = F.softmax(scores.to(torch.float32), dim=1)

scores_for_choice = scores if expert_bias is None else scores + expert_bias
_, selected_experts_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1)
top_scores = scores.gather(dim=1, index=selected_experts_indices)
```

**Auxiliary-loss-free 负载均衡**（`moe.py:450-463`）：

TorchTitan 使用了 `expert_bias` 机制实现无辅助损失的负载均衡（参考 [论文](https://arxiv.org/abs/2408.15664)）：

- `expert_bias` 是一个 `(num_experts,)` 的 buffer，**仅影响路由选择，不影响 gate 值**
- `tokens_per_expert` 在前向过程中累积，用于在 optimizer step pre-hook 中更新 `expert_bias`
- gate 值 `top_scores` 始终从原始 `scores` 中提取，而非从 `scores + expert_bias` 中提取

### 2.5 score_before_experts 机制

`MoE.forward()` 中有一个关键的 `score_before_experts` 标志（`moe.py:515-519`）：

```python
if self.score_before_experts:
    routed_input = (
        routed_input.to(torch.float32) * top_scores_experts_sorted.reshape(-1, 1)
    ).to(x.dtype)
```

- 当 `score_before_experts=True`：在专家计算 **之前** 乘以路由分数，结果直接 `sum`
- 当 `score_before_experts=False`：在专家计算 **之后** 使用 `bmm` 加权求和

前者可以与 DeepEP 的异步 combine 更好地重叠（因为 combine 之后不需要额外的加权操作）。

### 2.6 TokenReorderer

`TokenReorderer`（`moe.py:361-416`）将 token 按专家分组排序，为 `GroupedExperts` 的批量计算做准备：

```python
# moe.py:406-410
token_indices_experts_sorted = torch.argsort(
    selected_experts_indices.view(-1), stable=True
)
top_scores_experts_sorted = top_scores.view(-1)[token_indices_experts_sorted]
```

> **NOTE**：为什么 `num_tokens_per_expert` 计算了两次？（`moe.py:500-504` 注释说明）第一次在 router 中用于更新 `self.tokens_per_expert`（全局统计，所有 TP rank 一致），第二次在 reorderer 中用于实际路由（当 `expert_tensor_parallel_degree==1` 时，被 ReordererSequenceParallel 分片后不同 TP rank 看到不同值）。

---

## 第 3 节：DeviceMesh 设计 — Dense Mesh vs Sparse Mesh

> **一句话总结**：TorchTitan 构建三套全局 Mesh（dataloading、dense、sparse），通过 `efsdp = fsdp * tp // (etp * ep)` 公式将 FSDP 和 TP 维度重新映射到 EP 空间。

### 3.1 三套全局 Mesh

`ParallelDims.build_mesh()`（`parallel_dims.py:68-175`）创建三套从同一个 world mesh 展开的全局 Mesh：

```python
# parallel_dims.py:96-98 (注释中的说明)
# dataloading_mesh: ["pp", "batch",         "cp", "tp"]
# dense_mesh:       ["pp", "dp_replicate",  "fsdp", "tp"]
# sparse_mesh:      ["pp", "dp_replicate",  "efsdp", "ep", "etp"]
```

每套 Mesh 覆盖 **全部 GPU**，只是维度的划分方式不同：

```
所有 GPU
├── dataloading_mesh: pp × batch × cp × tp
│   用途：数据加载，决定每个 rank 读哪部分数据
│
├── dense_mesh: pp × dp_replicate × fsdp × tp
│   用途：Dense 层（包括 attention、FFN、embedding 等）的 FSDP/TP
│
└── sparse_mesh: pp × dp_replicate × efsdp × ep × etp
    用途：MoE 层（路由专家）的 FSDP/EP/TP
```

### 3.2 核心公式推导

`parallel_dims.py:133-135` 中的计算：

```python
batch = self.dp_replicate * self.dp_shard
fsdp = self.dp_shard * self.cp
efsdp = fsdp * self.tp // (self.etp * self.ep)
```

**推导过程**：

从 dense mesh 出发：
```
world_size = pp × dp_replicate × fsdp × tp
           = pp × dp_replicate × (dp_shard × cp) × tp
```

从 sparse mesh 出发：
```
world_size = pp × dp_replicate × efsdp × ep × etp
```

两者必须相等（覆盖全部 GPU），因此：
```
pp × dp_replicate × fsdp × tp = pp × dp_replicate × efsdp × ep × etp

化简（两边除以 pp × dp_replicate）：
fsdp × tp = efsdp × ep × etp

求解：
efsdp = fsdp × tp / (ep × etp)
      = (dp_shard × cp × tp) / (ep × etp)
```

> **NOTE**：`efsdp` 的物理含义是——在 EP 空间中，专家参数被 FSDP 切分的程度。Dense mesh 中 FSDP 切分了 `fsdp` 份，TP 切分了 `tp` 份；Sparse mesh 中 EP 切分了 `ep` 份，ETP 切分了 `etp` 份。剩余的切分维度就归给 `efsdp`。

### 3.3 约束条件

`parallel_dims.py:58-59`：

```python
if ep > 1:
    assert etp == tp or etp == 1, "Currently we only support ETP=TP or ETP=1"
```

**两种模式**：
- `etp == tp`：ExpertTensorParallel 模式，EP 和 TP 组合成二维切分。此时 `efsdp = fsdp * tp / (tp * ep) = fsdp / ep = dp_shard * cp / ep`
- `etp == 1`：标准 EP 模式，TP 维度全部借给了 EP + efsdp。此时 `efsdp = fsdp * tp / ep`

### 3.4 `_mesh_exist()` 的特殊处理

`parallel_dims.py:61-66`：

```python
def _mesh_exist(self, name: str, degree: int) -> bool:
    if name == "efsdp":
        # EP > 1 时 efsdp 始终存在，即使 size 为 1
        return True if self.ep > 1 else False
    return degree > 1
```

**为什么 `efsdp` 在 EP>1 时始终存在？** 因为 MoE 层需要 FSDP 包���来处理 **混合精度训练**（`MixedPrecisionPolicy`）。即使 `efsdp=1`（即没有数据并行切分），FSDP 仍然负责参数的 dtype 转换（param_dtype / reduce_dtype）。

### 3.5 fsdp_gradient_divide_factor

`parallel_dims.py:350-355`：

```python
@property
def fsdp_gradient_divide_factor(self) -> int:
    return self.dp_replicate * self.dp_shard * self.cp
```

这个因子与所有数据并行维度的乘积一致，确保梯度的平均计算跨越所有数据并行 rank——无论参数是在 dense mesh 还是 sparse mesh 上 FSDP 切分。

### 3.6 Mesh 映射可视化

以 8 GPU 为例，`pp=1, dp_replicate=1, dp_shard=2, cp=1, tp=4, ep=4, etp=1`：

```
计算：
  batch = 1 * 2 = 2
  fsdp = 2 * 1 = 2
  efsdp = 2 * 4 / (1 * 4) = 2

dense_mesh: [pp=1, dp_replicate=1, fsdp=2, tp=4]
  GPU 布局: [[0,1,2,3], [4,5,6,7]]
            |-- tp=4 --| |-- tp=4 --|
            |---------- fsdp=2 ----------|

sparse_mesh: [pp=1, dp_replicate=1, efsdp=2, ep=4, etp=1]
  GPU 布局: [[0,1,2,3], [4,5,6,7]]
            |--ep=4---|  |--ep=4---|
            |------ efsdp=2 ------|

Dense 层看到：GPU 0-3 是一个 TP group，GPU 0,4 是一个 FSDP group
MoE  层看到：GPU 0-3 是一个 EP group，GPU 0,4 是一个 efsdp group
```

---

## 第 4 节：四种 Expert Parallelism 策略

> **一句话总结**：TorchTitan 提供四种 EP 策略——TensorParallel（无 EP）、ExpertParallel（标准 all-to-all）、ExpertTensorParallel（EP+TP 二维切分）和 DeepEPExpertParallel（DeepEP 后端），通过统一的 `ParallelStyle` 接口实现。

### 4.1 TensorParallel（仅 TP，无 EP）

**条件**：`ep_mesh is None`（`parallelize.py:559-563`）

```python
if ep_mesh is None:
    assert ep_etp_mesh is None
    experts_mesh = tp_mesh
    experts_plan = TensorParallel()
```

**权重切分**（`expert_parallel.py:61-77`）：

```python
class TensorParallel(ParallelStyle):
    def _partition_fn(self, name, module, device_mesh):
        # w1: (E, H, D) → Shard(1) 即按 hidden_dim 切分 → ColwiseParallel
        module.register_parameter(
            "w1", nn.Parameter(distribute_tensor(module.w1, device_mesh, [Shard(1)]))
        )
        # w2: (E, D, H) → Shard(2) 即按 hidden_dim 切分 → RowwiseParallel
        module.register_parameter(
            "w2", nn.Parameter(distribute_tensor(module.w2, device_mesh, [Shard(2)]))
        )
        # w3: (E, H, D) → Shard(1) 即按 hidden_dim 切分 → ColwiseParallel
        module.register_parameter(
            "w3", nn.Parameter(distribute_tensor(module.w3, device_mesh, [Shard(1)]))
        )
```

**输入处理**（`expert_parallel.py:50-57`）：

```python
def _prepare_input_fn(self, mod, inputs, device_mesh):
    routed_input, num_tokens_per_expert = inputs
    routed_input = DTensor.from_local(
        routed_input, device_mesh, (Replicate(),)
    ).to_local(grad_placements=(Partial(),))
    return routed_input, num_tokens_per_expert
```

这里 `from_local(Replicate()).to_local(grad_placements=(Partial(),))` 的含义：
- **前向**：输入被标记为 `Replicate`（每个 TP rank 拥有完整副本），`to_local()` 返回普通 Tensor
- **反向**：梯度被标记为 `Partial`，表示需要 all-reduce 归约

### 4.2 ExpertParallel（标准 EP，all-to-all）

**条件**：`ep_mesh is not None` 且 `etp_mesh is None` 且 `use_deepep=False`（`parallelize.py:564-577`）

```python
elif tp_mesh is None or etp_mesh is None:
    assert ep_etp_mesh is None
    experts_mesh = ep_mesh
    experts_plan = ExpertParallel()
```

**权重切分**（`expert_parallel.py:97-100`）：

```python
class ExpertParallel(BaseExpertParallel):
    def _partition_fn(self, name, mod, device_mesh):
        for param_name, param in mod.named_parameters(recurse=False):
            dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
            mod.register_parameter(param_name, dist_param)
```

所有参数在 **dim 0**（专家维度）上切分。如果有 16 个专家和 4 个 EP rank，每个 rank 持有 4 个专家的参数。

**dispatch/combine 流程**在第 5 节详细分析。

### 4.3 ExpertTensorParallel（EP+TP 二维切分）

**条件**：`ep_mesh` 和 `etp_mesh` 都存在（`parallelize.py:578-580`）

```python
else:
    experts_mesh = ep_etp_mesh
    experts_plan = ExpertTensorParallel()
```

**权重切分**（`expert_parallel.py:218-238`）——二维 DTensor 切分：

```python
class ExpertTensorParallel(ExpertParallel):
    def _partition_fn(self, name, mod, device_mesh):
        # w1: (E, H, D) → [Shard(0), Shard(1)] → EP 切专家维，TP 切 hidden_dim
        mod.register_parameter(
            "w1", nn.Parameter(distribute_tensor(mod.w1, device_mesh, [Shard(0), Shard(1)]))
        )
        # w2: (E, D, H) → [Shard(0), Shard(2)] → EP 切专家维，TP 切 hidden_dim
        mod.register_parameter(
            "w2", nn.Parameter(distribute_tensor(mod.w2, device_mesh, [Shard(0), Shard(2)]))
        )
        # w3: (E, H, D) → [Shard(0), Shard(1)]
        mod.register_parameter(
            "w3", nn.Parameter(distribute_tensor(mod.w3, device_mesh, [Shard(0), Shard(1)]))
        )
```

**dispatch 的两层处理**（`expert_parallel.py:197-216`）：

```python
def _token_dispatch(self, mod, inputs, device_mesh):
    routed_input, num_tokens_per_expert = inputs

    # 1. 先在 etp mesh 上标记 Replicate → 反向时梯度 Partial 归约
    routed_input = DTensor.from_local(
        routed_input, device_mesh["etp"], (Replicate(),)
    ).to_local(grad_placements=(Partial(),))

    inputs = (routed_input, num_tokens_per_expert)

    # 2. 再在 ep mesh 上执行标准 EP dispatch（all-to-all）
    return super()._token_dispatch(mod, inputs, device_mesh["ep"])
```

> **NOTE**：注释（`expert_parallel.py:204-208`）指出，这里理论上应该使用 `dense_mesh["tp"]` 而非 `device_mesh["etp"]`，但因为 ETP 约束为 `etp == tp or etp == 1`，两者几乎总是相同的，所以简化了接口。

### 4.4 DeepEPExpertParallel（DeepEP 后端）

**条件**：`use_deepep=True`（`parallelize.py:567-574`）

```python
if use_deepep:
    score_before_experts = transformer_block.moe.score_before_experts
    experts_plan = DeepEPExpertParallel(
        score_before_experts=score_before_experts,
    )
```

**不兼容 ETP**（`parallelize.py:121-125`）：

```python
if parallel_dims.etp_enabled:
    raise NotImplementedError(
        "DeepEP with Expert Tensor Parallelism (ETP) is not supported yet."
    )
```

**权重切分**（`expert_parallel.py:358-365`）——与标准 EP 相同，`Shard(0)`：

```python
@staticmethod
def _partition_fn(name, mod, device_mesh):
    for param_name, param in mod.named_parameters(recurse=False):
        mod.register_parameter(
            param_name,
            nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)])),
        )
```

**dispatch/combine**委托给 `deepep.py` 中的 `dispatch_tokens()` / `combine_tokens()`（详见第 8 节）。

### 4.5 策略选择决策树

`apply_moe_ep_tp()` 的完整决策逻辑（`parallelize.py:500-590`）：

```
apply_moe_ep_tp(tp_mesh, ep_mesh, etp_mesh, ep_etp_mesh, use_deepep)
│
├── tp_mesh is not None?
│   ├── YES: 创建 MoE 层级 TP plan (moe_layer_plan)
│   │   ├── moe: PrepareModuleInputOutput(Shard(1)→Replicate, Partial→Shard(1))
│   │   ├── moe.router.gate: NoParallel()
│   │   ├── ep_mesh is not None and etp_mesh is None?
│   │   │   └── YES: moe.reorderer → ReordererSequenceParallel()
│   │   └── shared_experts: ColwiseParallel / RowwiseParallel
│   └── NO: 跳过 MoE TP plan
│
├── 选择 experts_plan:
│   ├── ep_mesh is None? → TensorParallel() on tp_mesh
│   ├── ep_mesh is not None, etp_mesh is None?
│   │   ├── use_deepep? → DeepEPExpertParallel() on ep_mesh
│   │   └── else → ExpertParallel() on ep_mesh
│   └── etp_mesh is not None? → ExpertTensorParallel() on ep_etp_mesh
│
├── dual_pipe_v and isinstance(experts_plan, BaseExpertParallel)?
│   └── YES: experts_plan = DualPipeExpertParallel(experts_plan)
│
└── parallelize_module(transformer_block.moe.experts, experts_mesh, experts_plan)
```

### 4.6 ReordererSequenceParallel — ETP=1 时的特殊处理

当 EP 借用了全部 TP 维度（`etp==1`），TP rank 之间不再对专家权重做 TP 切分，而是 **token 级别的序列并行**。`ReordererSequenceParallel`（`expert_parallel.py:256-317`）在 Reorderer 的输入输出上进行分片/还原：

```python
# 输入：将 (bs*slen, top_k) 按 token 维度分片到各 TP rank
def _prepare_inputput_fn(self, mod, inputs, device_mesh):
    top_scores, selected_experts_indices = inputs
    num_tokens, _ = top_scores.shape
    local_num_tokens = num_tokens // device_mesh.size()
    local_rank = device_mesh.get_local_rank()
    offset = local_rank * local_num_tokens
    top_scores = top_scores[offset : offset + local_num_tokens]
    selected_experts_indices = selected_experts_indices[offset : offset + local_num_tokens]
    return top_scores, selected_experts_indices

# 输出：调整 token_indices 为全局索引
def _prepare_output_fn(self, mod, outputs, device_mesh):
    top_scores, token_indices_experts_sorted, num_tokens_per_expert = outputs
    local_rank = device_mesh.get_local_rank()
    token_indices_experts_sorted = (
        token_indices_experts_sorted + top_scores.shape[0] * local_rank
    )
    return top_scores, token_indices_experts_sorted, num_tokens_per_expert
```

---

## 第 5 节：Token Dispatch 与 Combine 详解

> **一句话总结**：标准 EP 的 dispatch 通过两次 `all_to_all_single` 和一个 Triton kernel `generate_permute_indices` 将 token 从"按 rank 排列"重排为"按专家排列"并对齐填充。

### 5.1 Dispatch 完整流程

`ExpertParallel._token_dispatch()`（`expert_parallel.py:102-166`）的执行步骤：

```
Step 1: all_to_all(num_tokens_per_expert)  → 交换 token 计数信息
Step 2: 计算 input_splits / output_splits   → CPU 上的同步点
Step 3: all_to_all_single_autograd(routed_input)  → 主数据通信
Step 4: _permute(routed_input, ...)         → rank 顺序 → 专家顺序 + 对齐填充
```

#### Step 1：交换 token 计数

```python
# expert_parallel.py:112-122
with torch.no_grad():
    num_tokens_per_expert_group = all_to_all_single(
        num_tokens_per_expert,   # shape: (num_total_experts,)
        None, None,
        group=device_mesh.get_group(),
    )
    num_tokens_per_expert_group = torch.ops._c10d_functional.wait_tensor(
        num_tokens_per_expert_group
    )
```

**数值示例**：假设 4 个 EP rank，每 rank 2 个专家（共 8 个专家），某个 rank 发送/接收的情况：

```
Rank 0 的 num_tokens_per_expert = [10, 5, 8, 3, 12, 7, 6, 4]
                                   ↑  ↑  ↑  ↑   ↑  ↑  ↑  ↑
                                  E0 E1 E2 E3  E4 E5 E6 E7

all_to_all 后，Rank 0 收到每个 rank 发给自己（E0, E1）的 token 数：
num_tokens_per_expert_group = [10, 5,   # 来自 Rank 0
                                9, 3,   # 来自 Rank 1
                               11, 7,   # 来自 Rank 2
                                8, 4]   # 来自 Rank 3
```

#### Step 2：计算 splits（D2H 同步点）

```python
# expert_parallel.py:123-135
input_splits = (
    num_tokens_per_expert.view(ep_degree, -1)
    .sum(dim=1)
    .to(torch.device("cpu"), non_blocking=True)
)
# NOTE: this would incur a device-to-host sync
output_splits = (
    num_tokens_per_expert_group.view(ep_degree, -1)
    .sum(dim=1)
    .to(torch.device("cpu"), non_blocking=False)  # ← 同步等待！
)
self.input_splits = input_splits.tolist()
self.output_splits = output_splits.tolist()
```

**关键性能影响**：`non_blocking=False` 导致 CPU 必须等待 GPU 上的计算完成才能拿到 output_splits 的值。这是一个 **D2H 同步点**，会阻塞 CUDA stream。

> **NOTE**：这个 D2H 同步是 EP 破坏 FSDP 隐式 prefetch 的根本原因（详见第 7 节）。`input_splits` 使用 `non_blocking=True` 是因为它可以延迟到主 all-to-all 之前才需要，但 `output_splits` 必须立刻知道才能确定接收 buffer 大小。

**数值示例**（续）：

```
input_splits = [10+5, 8+3, 12+7, 6+4] = [15, 11, 19, 10]
（Rank 0 发给各 rank 的 token 总数）

output_splits = [10+5, 9+3, 11+7, 8+4] = [15, 12, 18, 12]
（Rank 0 从各 rank 接收的 token 总数）
```

#### Step 3：主数据通信

```python
# expert_parallel.py:138-143
routed_input = all_to_all_single_autograd(
    routed_input,
    self.output_splits,   # 接收 splits
    self.input_splits,    # 发送 splits
    device_mesh.get_group(),
)
```

`all_to_all_single_autograd` 支持自动微分——前向做 all-to-all 发送，反向自动做反向 all-to-all。

#### Step 4：_permute — 从 rank 顺序到专家顺序

all-to-all 之后，数据的布局是 **按 rank 排列** 的：

```
收到的 routed_input 排列���序：
[来自 Rank0 给 E0 的 tokens, 来自 Rank0 给 E1 的 tokens,
 来自 Rank1 给 E0 的 tokens, 来自 Rank1 给 E1 的 tokens,
 来自 Rank2 给 E0 的 tokens, 来自 Rank2 给 E1 的 tokens,
 来自 Rank3 给 E0 的 tokens, 来自 Rank3 给 E1 的 tokens]
```

但 `grouped_mm` 需要 **按专家排列**：

```
期望的排列顺序：
[给 E0 的所有 tokens (来自 R0, R1, R2, R3), padding,
 给 E1 的所有 tokens (来自 R0, R1, R2, R3), padding]
```

`_permute()` 函数（`utils.py:42-59`）完成这个重排：

```python
def _permute(x, num_tokens_per_expert, ep_degree, num_local_experts):
    x_padded_per_expert = x.shape[0] + num_local_experts * TOKEN_GROUP_ALIGN_SIZE_M
    padded_max_len = _round_up(x_padded_per_expert, TOKEN_GROUP_ALIGN_SIZE_M)

    with torch.no_grad():
        (permuted_indices, num_tokens_per_expert, _offsets) = generate_permute_indices(
            num_tokens_per_expert, num_local_experts, ep_degree,
            padded_max_len, TOKEN_GROUP_ALIGN_SIZE_M,
        )

    x = torch.vstack((x, x.new_zeros((x.shape[-1]))))  # 追加一行零（用于填充索引）
    input_shape = x.shape
    x = x[permuted_indices, :]  # 使用索引重排

    return input_shape, x, permuted_indices, num_tokens_per_expert
```

### 5.2 generate_permute_indices — Triton Kernel

`kernels.py:143-216` 中的 `generate_permute_indices` 是排列索引生成的核心：

```python
def generate_permute_indices(
    tokens_per_expert_group, experts_per_rank, num_ranks, max_len, alignment
):
    # 1. 前缀和：计算每个 (rank, expert) 对的 token 起始位置
    start_index_values = torch.cumsum(tokens_per_expert_group, 0) - tokens_per_expert_group

    # 2. 按专家聚合：计算每个本地专家的总 token 数（跨所有 rank）
    total_tokens_per_expert = tokens_per_expert_group.view(num_ranks, -1).sum(0)

    # 3. 对齐填充：确保每个专家的 token 数是 alignment 的倍数
    total_tokens_per_expert = torch.clamp_min(total_tokens_per_expert, alignment)
    m_sizes = ((total_tokens_per_expert + alignment - 1) // alignment * alignment).to(torch.int32)

    # 4. 计算写入偏移
    m_offsets = torch.cumsum(m_sizes, 0)
    write_offsets = m_offsets - m_sizes

    # 5. 调用 Triton kernel 并行填充索引
    permuted_indices = fill_indices_wrapper(
        tokens_per_expert_group, start_index_values, write_offsets,
        experts_per_rank, num_ranks, max_len,
    )

    return permuted_indices, m_sizes, m_offsets.to(torch.int32)
```

**数值示例**：

```
tokens_per_expert_group = [10, 5, 9, 3, 11, 7, 8, 4]
                           E0 E1  E0 E1  E0 E1  E0 E1
                          |rank0| |rank1| |rank2| |rank3|

experts_per_rank = 2, num_ranks = 4, alignment = 8

Step 1: start_index_values = [0, 10, 15, 24, 27, 38, 45, 53]

Step 2: total_tokens_per_expert = [10+9+11+8, 5+3+7+4] = [38, 19]

Step 3: m_sizes = [ceil(38/8)*8, ceil(19/8)*8] = [40, 24]

Step 4: write_offsets = [0, 40]

Step 5: Triton kernel 为每个专家生成索引：
  E0: 位置 0-9 ← rank0 的 token（start=0），
      位置 10-18 ← rank1 的 token（start=15），
      位置 19-29 ← rank2 的 token（start=27），
      位置 30-37 ← rank3 的 token（start=45）
      位置 38-39 ← padding（-1 → 指向追加的零行）
  E1: 位置 40-44 ← rank0 的 token（start=10），...
```

### 5.3 TOKEN_GROUP_ALIGN_SIZE_M

`utils.py:15-16`：

```python
TOKEN_GROUP_ALIGN_SIZE_M = 8
ValidTokenGroupAlignmentSize = Literal[8, 16, 32]
```

对齐大小根据训练精度选择：
- **bf16**：8（16 bytes / 2 bytes per elem = 8）
- **fp8**：16（16 bytes / 1 byte per elem = 16）
- **mxfp8**：32（scaling block size）

### 5.4 Combine 流程

`ExpertParallel._token_combine()`（`expert_parallel.py:168-181`）是 dispatch 的逆操作：

```python
def _token_combine(self, mod, routed_output, device_mesh):
    # 1. 反排列：从专家顺序恢复到 rank ���序
    routed_output = _unpermute(routed_output, self.input_shape, self.permuted_indices)

    # 2. 反向 all-to-all：将结果发回各 rank
    routed_output = all_to_all_single_autograd(
        routed_output,
        self.input_splits,     # 注意：splits 交换了
        self.output_splits,
        device_mesh.get_group(),
    )
    return routed_output
```

`_unpermute()`（`utils.py:62-66`）：

```python
def _unpermute(out, input_shape, permuted_indices):
    out_unpermuted = out.new_empty(input_shape)
    out_unpermuted[permuted_indices, :] = out  # scatter 回原位
    out = out_unpermuted[:-1]                   # 去掉追加的零行
    return out
```

---

## 第 6 节：FSDP 二次开发 — 双层包装策略

> **一句话总结**：MoE 层需要双层 FSDP 包装——内层在 `edp_mesh` 上包装 `experts`，外层在 `dp_mesh` 上包装整个 `TransformerBlock`，并通过条件 Shard 策略和手动梯度除法解决 mesh 不一致问题。

### 6.1 为什么需要双层 FSDP

标准 FSDP 在 `dp_mesh`（dense mesh 的 `[dp_replicate, fsdp]`）上包装每个 `TransformerBlock`。但 MoE 层的 `experts` 参数已经被 EP 在 dim 0 上切分过了，它需要在 **不同的 mesh**（`edp_mesh` = sparse mesh 的 `[dp_replicate, efsdp]`）上做 FSDP。

```
TransformerBlock (MoE enabled)
├── attention_norm, attention, ffn_norm  → dp_mesh 上 FSDP
├── moe.router, moe.shared_experts      → dp_mesh 上 FSDP（作为 block 的一部分）
└── moe.experts                          → edp_mesh 上 FSDP（独立包装）
    ├── w1: DTensor([Shard(0)]) on ep_mesh → 进一步在 edp_mesh 上 FSDP 切分
    ├── w2: DTensor([Shard(0)]) on ep_mesh → ...
    └── w3: DTensor([Shard(0)]) on ep_mesh → ...
```

### 6.2 apply_fsdp 的双层包装实现

`parallelize.py:324-427` 中 `apply_fsdp()` 的核心逻辑：

```python
def apply_fsdp(model, dp_mesh, ..., ep_degree, edp_mesh, ...):
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}

    # 1. 包装 tok_embeddings
    if model.tok_embeddings is not None:
        fully_shard(model.tok_embeddings, **fsdp_config, reshard_after_forward=...)

    # 2. 遍历每个 TransformerBlock
    for layer_id, transformer_block in model.layers.items():
        # 2a. MoE 层的内层 FSDP（experts 单独在 edp_mesh 上包装）
        if transformer_block.moe_enabled and ep_degree > 1:
            fsdp_mod_ep_config = fsdp_config.copy()
            fsdp_mod_ep_config["mesh"] = edp_mesh  # ← 关键���使用不同的 mesh

            # 条件 Shard 策略
            _experts_shard_placement_fn = None
            if edp_mesh["efsdp"].size() * ep_degree > transformer_block.moe.experts.num_experts:
                _experts_shard_placement_fn = lambda param: Shard(1)

            fully_shard(
                transformer_block.moe.experts,
                **fsdp_mod_ep_config,
                reshard_after_forward=...,
                shard_placement_fn=_experts_shard_placement_fn,  # ← 条件切分维度
            )

        # 2b. 外层 FSDP（整个 TransformerBlock 在 dp_mesh 上包装）
        fully_shard(transformer_block, **fsdp_config, reshard_after_forward=...)

    # 3. 包装 norm + output
    if model.norm is not None and model.output is not None:
        fully_shard([model.norm, model.output], **fsdp_config, reshard_after_forward=...)

    # 4. 包装 model 根
    fully_shard(model, **fsdp_config)

    # 5. 禁用自动梯度除法
    disable_fsdp_gradient_division(model)

    # 6. 设置 prefetch（如果 EP 启用）
    if ep_degree > 1:
        # ... (详见第 7 节)
```

### 6.3 条件 Shard 策略

`parallelize.py:395-402`：

```python
if (
    edp_mesh["efsdp"].size() * ep_degree
    > transformer_block.moe.experts.num_experts
):
    _experts_shard_placement_fn = lambda param: Shard(1)
```

**为什么需要条件切分维度？**

- EP 已经在 dim 0 上按 `ep_degree` 切分了专家
- 每个 EP rank 持有 `num_experts // ep_degree` 个专家
- FSDP 默认也在 dim 0 上切分
- 如果 `efsdp * ep > num_experts`，dim 0 上已经没有足够的切分空间

**数值示例**：
```
num_experts = 16, ep_degree = 8, efsdp = 4
每个 EP rank 持有 16/8 = 2 个专家
efsdp * ep = 4 * 8 = 32 > 16

此时如果 FSDP 还在 dim 0 切分，2 个专家要切 4 份 = 每份 0.5 个专家 → 无意义
所以切换到 dim 1（hidden_dim），在 hidden_dim 上切 4 份
```

### 6.4 包装顺序的重要性

完整的包装顺序：

```
1. fully_shard(tok_embeddings)          # 独立 FSDP 单元
2. 对每个 TransformerBlock:
   a. fully_shard(block.moe.experts)    # 内层：edp_mesh（仅 MoE 层）
   b. fully_shard(block)                # 外层：dp_mesh
3. fully_shard([norm, output])          # 联合 FSDP 单元
4. fully_shard(model)                   # 根节点
```

> **NOTE**：内层必须在外层之前包装。`fully_shard` 是递归的——它会自动跳过已经被 `fully_shard` 过的子模块。所以先包装 `experts`，再包装整个 `block` 时，`experts` 会被识别为独立的 FSDP 单元，不会被外层重复切分。

### 6.5 disable_fsdp_gradient_division

`llama3/infra/parallelize.py:272-284`：

```python
def disable_fsdp_gradient_division(model):
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.set_gradient_divide_factor(1.0)
```

**为什么要禁用？** 因为内层和外层 FSDP 的 mesh 大小不同：

- 外层（dp_mesh）：大小为 `dp_replicate * fsdp`
- 内层（edp_mesh）：大小为 `dp_replicate * efsdp`

FSDP 默认会按自己 mesh 的大小除以梯度，但我们需要 **统一的** 梯度除法因子（`dp_replicate * dp_shard * cp`）。所以先禁用 FSDP 的自动除法，在训练循环中用全局 token count 手动控制。

### 6.6 MoE 层 vs Dense 层的 FSDP 对比

```
Dense TransformerBlock:
┌─────────────────────────────────┐
│ fully_shard(block, dp_mesh)     │  ← 单层 FSDP
│ ├── attention_norm              │
│ ├── attention (wq,wk,wv,wo)    │
│ ├── ffn_norm                    │
│ └── feed_forward (w1,w2,w3)    │
└─────────────────────────────────┘

MoE TransformerBlock:
┌───────────────────��─────────────────────────────┐
│ fully_shard(block, dp_mesh)        ← 外层 FSDP  │
│ ├── attention_norm                               │
│ ├── attention (wq,wk,wv,wo)                     │
│ ├── ffn_norm                                     │
│ ├── moe.router, moe.reorderer                   │
│ ├── moe.shared_experts                           │
│ └── moe.experts                                  │
│     ┌───────────────────────────────────┐        │
│     │ fully_shard(experts, edp_mesh)    │ ← 内层  │
│     │ ├── w1: (E/ep, H, D)             │        │
│     │ ├── w2: (E/ep, D, H)             │        │
│     │ └── w3: (E/ep, H, D)             │        │
│     └───────────────────────────────────┘        │
└─────────────────────────────────────────────────┘
```

---

## 第 7 节：全链路 Prefetch 策略

> **一句话总结**：EP 的 D2H 同步会破坏 FSDP 的隐式 prefetch，TorchTitan 通过显式设置前向和反向的 prefetch 链（包括 MoE 层的"双重 prefetch"）来恢复通信-计算重叠。

### 7.1 EP 为何破坏隐式 Prefetch

FSDP 的隐式 prefetch 机制依赖于 CUDA stream 的顺序执行：

```
正常 FSDP（无 EP）：
  Block N forward → [FSDP 自动 all-gather Block N+1 参数]
  Block N+1 forward → ...

隐式 prefetch 工作原理：
  FSDP 在 Block N 的 forward 完成后，在同一 CUDA stream 上
  提交 Block N+1 的 all-gather。由于 CUDA stream 是 FIFO，
  Block N+1 的 all-gather 会在 Block N 的 forward 完成后立即开始。
```

但 EP 引入了 D2H 同步（`non_blocking=False`）：

```
有 EP 的 MoE 层：
  Block N forward:
    ├── all_to_all(num_tokens)     ← GPU 通信
    ├── output_splits.to("cpu", non_blocking=False)  ← CPU 等待 GPU！
    ├── all_to_all_single(data)    ← GPU 通信
    └── experts compute
  [此时 CUDA stream 被 D2H sync 阻塞，FSDP 的 prefetch 无法提交]
```

### 7.2 显式 Prefetch 策略

`parallelize.py:432-497` 设置了完整的显式 prefetch 链：

#### 前向 Prefetch

```python
# parallelize.py:437-467
transformer_blocks = list(model.layers.values())
next_transformer_blocks = transformer_blocks[1:] + [None]

# tok_embeddings 预取第���个 block
if model.tok_embeddings is not None and len(model.layers) > 0:
    model.tok_embeddings.set_modules_to_forward_prefetch([transformer_blocks[0]])

# 每个 block 预取下一个 block
for transformer_block, next_transformer_block in zip(
    transformer_blocks, next_transformer_blocks
):
    if next_transformer_block is not None:
        if next_transformer_block.moe_enabled:
            # MoE 层需要"双重 prefetch"
            transformer_block.set_modules_to_forward_prefetch(
                [next_transformer_block, next_transformer_block.moe.experts]
            )
        else:
            transformer_block.set_modules_to_forward_prefetch(
                [next_transformer_block]
            )
    elif model.norm is not None and model.output is not None:
        transformer_block.set_modules_to_forward_prefetch(
            [model.norm, model.output]
        )
```

**前向 Prefetch 链**：

```
tok_embeddings
  → prefetch: [Block 0]
Block 0 (dense)
  → prefetch: [Block 1]              # 下一层是 dense
Block 1 (dense)
  → prefetch: [Block 2, Block 2.moe.experts]  # 下一层是 MoE → 双重 prefetch
Block 2 (MoE)
  → prefetch: [Block 3]              # 下一层是 dense
Block 3 (dense)
  → prefetch: [Block 4, Block 4.moe.experts]  # 下一层是 MoE
...
Block N-1 (last)
  → prefetch: [norm, output]
```

**为什么 MoE 层需要双重 prefetch？** 因为 MoE 层有 **两个独立的 FSDP 单元**：
1. `TransformerBlock` 本身（外层 FSDP，包含 attention、router、shared_experts 等）
2. `moe.experts`（内层 FSDP，独立的 all-gather）

两者的参数需要分别预取。

#### 反向 Prefetch

```python
# parallelize.py:470-497
reversed_transformer_blocks = list(reversed(model.layers.values()))
prev_transformer_blocks = reversed_transformer_blocks[1:] + [None]

# output 预取最后一个 block
if model.norm is not None and model.output is not None and len(model.layers) > 0:
    model.output.set_modules_to_backward_prefetch([reversed_transformer_blocks[0]])

# 每个 block 预取前一个 block（反向顺序）
for transformer_block, prev_transformer_block in zip(
    reversed_transformer_blocks, prev_transformer_blocks
):
    if prev_transformer_block is not None:
        if prev_transformer_block.moe_enabled:
            transformer_block.set_modules_to_backward_prefetch(
                [prev_transformer_block, prev_transformer_block.moe.experts]
            )
        else:
            transformer_block.set_modules_to_backward_prefetch(
                [prev_transformer_block]
            )
    elif model.tok_embeddings is not None:
        transformer_block.set_modules_to_backward_prefetch([model.tok_embeddings])
```

**反向 Prefetch 链**：

```
output
  → backward_prefetch: [Block N-1]
Block N-1
  → backward_prefetch: [Block N-2, Block N-2.moe.experts]  # 如果 N-2 是 MoE
Block N-2 (MoE)
  → backward_prefetch: [Block N-3]
...
Block 0
  → backward_prefetch: [tok_embeddings]
```

### 7.3 Prefetch 的触发时机

```
Forward:
  Block N forward 开始
    → FSDP all-gather Block N 参数（如果还没 prefetch 到）
    → 执行 forward 计算
    → FSDP 触发 prefetch: all-gather Block N+1 参数
                           + all-gather Block N+1.moe.experts 参数（如果 MoE）
    → Block N forward 结束

Backward:
  Block N backward 开始
    → FSDP all-gather Block N 参数（如果已 reshard）
    → 执行 backward 计算
    → FSDP 触发 prefetch: all-gather Block N-1 参数
                           + all-gather Block N-1.moe.experts 参数（如果 MoE）
    → Block N backward 结束
```

---

## 第 8 节：DeepEP 后端详解

> **一句话总结**：DeepEP 后端通过自定义 Op 注册实现 SAC 兼容，通过 Handle 缓存机制桥接 tensor-only 返回值与非 tensor 状态，通过异步 combine + `sync_combine()` 实现通信-计算重叠。

### 8.1 整体架构

DeepEP 后端涉及三个文件的协同：

```
expert_parallel.py          deepep.py                    moe_deepep.py
┌─────────────────────┐     ┌──────────────────────┐     ┌────────────────────┐
│ DeepEPExpertParallel │     │ dispatch_tokens()    │     │ DeepEPMoE          │
│ ._token_dispatch()  │ ──→ │ combine_tokens()     │     │ .forward()         │
│ ._token_combine()   │ ──→ │ torch.ops.deepep.*   │     │ (通信-计算重叠)    │
│ ._partition_fn()    │     │ _handle_cache        │     │ sync_combine()     │
└─────────────────────┘     │ _pending_combine_evt │     └────────────────────┘
                            └──────────────────────┘
```

### 8.2 自定义 Op 注册（SAC 兼容）

`deepep.py:57-73` 使用 `torch.library` 注册自定义 Op：

```python
_lib = torch.library.Library("deepep", "DEF")

_lib.define(
    "dispatch(Tensor x, Tensor topk_idx, Tensor topk_weights, "
    "Tensor num_tokens_per_rank, Tensor num_tokens_per_rdma_rank, "
    "Tensor is_token_in_rank, Tensor num_tokens_per_expert) "
    "-> (Tensor, Tensor, Tensor, Tensor, Tensor)"
)

_lib.define("combine(Tensor x, Tensor handle_id) -> Tensor")
```

**为什么要注册为自定义 Op？**

1. **SAC 兼容**：Selective Activation Checkpointing 需要识别哪些 Op 的结果需要保存。自定义 Op 可以被添加到 `_op_sac_save_list`（`parallelize.py:132-133`）
2. **torch.compile 兼容**：编译器需要知道 Op 的签名才能正确追踪
3. **Autograd 兼容**：可以注册自定义的前向/反向实现

### 8.3 Handle 缓存机制

DeepEP 的 `buffer.dispatch()` 返回一个 `handle` 对象（非 Tensor），但 `torch.library` 的 Op 签名只支持 Tensor 输入输出。解决方案是 Handle 缓存：

```python
# deepep.py:38-39
_handle_cache: dict = {}   # handle_id → handle
_handle_counter: int = 0

# 在 dispatch Op 中：
def _dispatch_op_impl(x, topk_idx, ...):
    ..., handle, after_event = buffer.dispatch(...)
    handle_id = _get_next_handle_id()              # 生成唯一 CPU tensor
    _handle_cache[handle_id.item()] = handle       # 缓存 handle
    return recv_x, recv_indices, recv_scores, recv_num, handle_id  # 返回 tensor

# 在 combine Op 中：
def _combine_op_impl(x, handle_id):
    handle = _handle_cache.get(handle_id.item())   # 从缓存取出 handle
    combined, _, after_event = buffer.combine(x=x, handle=handle, ...)
    return combined
```

**handle_id 的生命周期**：

```
dispatch forward  → _handle_cache[id] = handle       (创建)
dispatch setup_context → ctx.saved_handle = handle    (保存到 autograd context)
combine forward  → handle = _handle_cache.get(id)     (使用)
combine setup_context → _handle_cache.pop(id)         (从缓存中移除)
combine backward → handle = ctx.saved_handle           (从 context 中使用)
dispatch backward → handle = ctx.saved_handle          (从 context 中使用)
```

### 8.4 Autograd 注册

`deepep.py:238-243`：

```python
torch.library.register_autograd(
    "deepep::dispatch", _dispatch_backward, setup_context=_dispatch_setup_context
)
torch.library.register_autograd(
    "deepep::combine", _combine_backward, setup_context=_combine_setup_context
)
```

**反向的对称性**：
- dispatch 的反向是 combine（把梯度从专家端收集回来）
- combine 的反向是 dispatch（把梯度分发到专家端）

```python
# dispatch backward (deepep.py:132-168):
def _dispatch_backward(ctx, grad_recv_x, ...):
    grad_x, grad_scores, after_event = _buffer.combine(
        x=grad_recv_x, handle=ctx.saved_handle, ...
    )
    return grad_x, None, grad_topk_weights, None, None, None, None

# combine backward (deepep.py:211-235):
def _combine_backward(ctx, grad_combined):
    grad_x, _, _, _, _, after_event = _buffer.dispatch(
        x=grad_combined, handle=ctx.saved_handle, ...
    )
    return grad_x, None
```

### 8.5 异步 Combine 与 sync_combine()

`deepep.py:171-202` 中 combine Op 的关键设计——**异步返回**：

```python
@torch.library.impl(_lib, "combine", "CUDA")
def _combine_op_impl(x, handle_id):
    combined, _, after_event = buffer.combine(
        x=x, handle=handle, ...,
        async_finish=True,           # ← 异步完成
        allocate_on_comm_stream=True,
    )
    # 不立即同步！而是存储 event 等待后续手动同步
    _pending_combine_event = after_event
    return combined
```

`sync_combine()`（`deepep.py:247-285`）在需要时手动同步：

```python
@torch.compiler.disable()
def sync_combine():
    global _pending_combine_event
    if _pending_combine_event is not None:
        _pending_combine_event.current_stream_wait()
        _pending_combine_event = None
```

**通信-计算重叠的时序**：

```
时间 →
CUDA compute stream:  [experts compute] [shared_experts] [sync] [add + reshape]
CUDA comm stream:     [dispatch]                [combine........]
                                         ↑ combine 异步返回
                                                          ↑ sync_combine() 等待
```

### 8.6 DeepEPMoE — 通信-计算重叠

`moe_deepep.py:16-81` 中的 `DeepEPMoE` 继承自 `MoE` 并重写了 `forward()`：

```python
class DeepEPMoE(MoE):
    def __init__(self, moe_args, dim, hidden_dim):
        super().__init__(moe_args, dim, hidden_dim)
        self.reorderer = None  # DeepEP 不需要 reorderer

    def forward(self, x):
        bs, slen, dim = x.shape
        x = x.view(-1, dim)

        # 1. Router
        top_scores, selected_experts_indices, num_tokens_per_expert = self.router(x, self.expert_bias)

        # 2. Experts（hooks 处理 dispatch/combine）
        #    combine 异步返回，routed_output 可能还没准备好
        routed_output = self.experts(
            x, num_tokens_per_expert,
            selected_experts_indices, top_scores, self.experts.num_experts,
        )

        # 3. Shared experts（与 combine 通信重叠）
        out = self.shared_experts(x) if self.shared_experts is not None else None

        # 4. 同步 combine
        sync_combine()

        # 5. 结果相加
        if out is None:
            return routed_output.reshape(bs, slen, dim)
        return (out + routed_output).reshape(bs, slen, dim)
```

**关键差异对比**（标准 MoE vs DeepEPMoE）：

| 方面 | 标准 MoE | DeepEPMoE |
|------|---------|-----------|
| 输入给 experts 的参数 | `(routed_input, num_tokens_per_expert)` | `(x, num_tokens_per_expert, indices, scores, num_experts)` |
| Reorderer | 使用 | 不使用（`self.reorderer = None`） |
| Combine 同步 | 同步（阻塞） | 异步 + 手动 `sync_combine()` |
| 通信-计算重叠 | 无 | shared_experts 与 combine 重叠 |

### 8.7 Buffer 管理

`deepep.py:288-317` 中的 `get_buffer()`：

```python
def get_buffer(group, hidden_bytes):
    global _buffer
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (
        Buffer.get_dispatch_config(group.size()),
        Buffer.get_combine_config(group.size()),
    ):
        num_nvl_bytes = max(
            config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes
        )
        num_rdma_bytes = max(
            config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes
        )

    if (_buffer is None or _buffer.group != group
        or _buffer.num_nvl_bytes < num_nvl_bytes
        or _buffer.num_rdma_bytes < num_rdma_bytes):
        _buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)
    return _buffer
```

**Buffer 自动计算**：
- NVLink buffer 大小：根据 `dispatch_config` 和 `combine_config` 的 hint 取最大值
- RDMA buffer 大小：同上
- Buffer 在 group 不变且大小足够时复用，避免重复分配

### 8.8 _permute_tokens / _unpermute_tokens（DeepEP 版）

DeepEP 使用自己的 permute 逻辑（`deepep.py:320-380`），与标准 EP 的 `_permute` 不同：

```python
def _permute_tokens(hidden_states, dispatched_indices, dispatched_scores):
    mask = dispatched_indices != -1                    # -1 表示 masked
    valid_expert_ids = dispatched_indices[mask]         # 1d tensor
    valid_scores = dispatched_scores[mask]

    sort_order = torch.argsort(valid_expert_ids, stable=True)
    permuted_indices = torch.arange(len(hidden_states), device=...) \
        .repeat_interleave(mask.sum(dim=1))[sort_order]
    permuted_hidden_states = hidden_states.index_select(0, permuted_indices)
    permuted_scores = valid_scores[sort_order]

    return permuted_hidden_states, permuted_scores, permuted_indices
```

**差异**：
- 标准 EP 的 `_permute` 使用 Triton kernel 生成对齐填充的索引
- DeepEP 的 `_permute_tokens` 使用 PyTorch 原生操作，不做对齐填充（DeepEP 内部处理对齐）
- DeepEP 需要处理 `top-k` 的 token 复制（`repeat_interleave`），而标准 EP 在 MoE.forward 中已经展开

### 8.9 dispatch_tokens 完整流程

`deepep.py:393-481`：

```python
def dispatch_tokens(hidden_states, selected_experts_indices, top_scores,
                    num_local_experts, num_experts, group, score_before_experts):
    # 1. Mask zero-score tokens
    selected_experts_indices = selected_experts_indices.masked_fill(top_scores == 0, -1)

    # 2. Get buffer
    buffer = get_buffer(group, get_hidden_bytes(hidden_states))

    # 3. Calculate dispatch layout (metadata exchange)
    (num_tokens_per_rank, num_tokens_per_rdma_rank,
     num_tokens_per_expert_dispatch, is_token_in_rank, _) = \
        buffer.get_dispatch_layout(topk_idx=selected_experts_indices, num_experts=num_experts)

    # 4. Execute dispatch via custom op
    (hidden_states, dispatched_indices, dispatched_expert_scores,
     num_tokens_per_expert, handle_id) = torch.ops.deepep.dispatch(
        hidden_states, selected_experts_indices, top_scores,
        num_tokens_per_rank, num_tokens_per_rdma_rank,
        is_token_in_rank, num_tokens_per_expert_dispatch,
    )

    # 5. Permute to grouped_mm format
    hidden_states, permuted_scores, permuted_indices = _permute_tokens(
        hidden_states, dispatched_indices, dispatched_expert_scores
    )

    # 6. Apply scores before experts if configured
    if score_before_experts and permuted_scores is not None:
        hidden_states = hidden_states * permuted_scores.to(hidden_states.dtype).reshape(-1, 1)
        permuted_scores_for_state = None
    else:
        permuted_scores_for_state = permuted_scores

    # 7. Build state for combine
    state = DispatchState(handle_id=handle_id, permuted_indices=permuted_indices,
                          num_recv_tokens=num_recv_tokens,
                          permuted_scores=permuted_scores_for_state)

    return hidden_states, num_tokens_per_expert, state
```

---

## 第 9 节：torch.compile 与 MoE

> **一句话总结**：Dense block 整块编译，MoE block 采用子模块级编译（跳过 experts 避免 graph break），`_run_experts_grouped_mm` 全局函数单独编译，EP 启用时通过 `mark_dynamic` 处理动态 token 数。

### 9.1 编译策略分支

`parallelize.py:593-690` 中 `apply_compile()` 对 dense 和 MoE block 使用不同策略：

```python
def apply_compile(model, compile_config, ep_enabled):
    torch._dynamo.config.capture_scalar_outputs = True

    for layer_id, transformer_block in model.layers.named_children():
        if transformer_block.moe_enabled:
            # MoE block：子模块级编译
            block = transformer_block._checkpoint_wrapped_module \
                    if isinstance(transformer_block, CheckpointWrapper) \
                    else transformer_block

            for attr_name, submod in block.named_children():
                if isinstance(submod, moe_module.MoE):
                    moe = submod
                    for attr_name, submod in moe.named_children():
                        if attr_name == "experts":
                            continue  # ← 跳过 experts！
                        setattr(moe, attr_name,
                                torch.compile(submod, backend=..., fullgraph=True))
                else:
                    setattr(block, attr_name,
                            torch.compile(submod, backend=..., fullgraph=True))
        else:
            # Dense block：整块编译
            transformer_block = torch.compile(
                transformer_block, backend=..., fullgraph=True
            )
```

**为什么跳过 experts？** 因为 `FSDP(GroupedExperts)` 作为独立的 FSDP 单元，其 all-gather/reduce-scatter hooks 会导致 graph break。编译 experts 会让 AC 回退到全 eager 模式。

> **NOTE**：注释（`parallelize.py:627-629`）说明还有一个 B200 相关的 issue 导致 token dispatch/combine 也不能编译。

### 9.2 _run_experts_grouped_mm 的全局编译

`parallelize.py:658-685`：

```python
# 只 patch 一次
already_patched = (
    "_run_experts_grouped_mm_dynamic"
    in moe_module._run_experts_grouped_mm.__qualname__
)
if not already_patched:
    moe_module._run_experts_grouped_mm = torch.compile(
        moe_module._run_experts_grouped_mm,
        backend=compile_config.backend, fullgraph=True,
    )

    if ep_enabled:
        compiled_fn = moe_module._run_experts_grouped_mm

        def _run_experts_grouped_mm_dynamic(w1, w2, w3, x, num_tokens_per_expert):
            torch._dynamo.mark_dynamic(x, 0)  # ← 标记第 0 维为动态
            return compiled_fn(w1, w2, w3, x, num_tokens_per_expert)

        moe_module._run_experts_grouped_mm = _run_experts_grouped_mm_dynamic
```

**为什么需要 `mark_dynamic`？** EP 的 all-to-all 通信后，每个 rank 收到的 token 数量是动态的（取决于路由结果）。`mark_dynamic(x, 0)` 告诉编译器第 0 维（token 维度）是动态的，不要对此维度做特化。

**`capture_scalar_outputs = True`**（`parallelize.py:600`）：这是 `torch._dynamo.config` 的一个实验性标志，允许编译器捕获标量输出（如 `num_tokens_per_expert` 的 item()），避免 graph break。

### 9.3 MoE block 的编译结构

```
MoE TransformerBlock 编译后：
├── attention_norm   → compiled
├── attention        → compiled
├── ffn_norm         → compiled
├── moe
│   ├── router       → compiled
│   ├── reorderer    → compiled
│   ├── shared_experts → compiled
│   └── experts      → NOT compiled (FSDP hooks 会 graph break)
│       └── _run_experts_grouped_mm → 全局编译（独立的编译单元）
```

---

## 第 10 节：DualPipeV — PP 与 EP 通信重叠

> **一句话总结**：`DualPipeExpertParallel` 通过 SyncHook 包装 dispatch/combine，配合 `HookCoordinator` 的 `threading.Barrier(2)` 和 `overlap_callback`，实现 PP 的前向/反向两个线程与 EP 通信的交织执行。

### 10.1 动机

当 PP 和 EP 同时启用时，存在两类通信：
1. **PP 通信**：pipeline stage 之间的 P2P send/recv
2. **EP 通信**：专家之间的 all-to-all dispatch/combine

这两类通信和计算之间可以重叠——当一个 stage 在等待 EP 通信时，另一个 stage 可以执行计算。

### 10.2 DualPipeExpertParallel 的 SyncHook 包装

`dual_pipe_v.py:54-98`：

```python
class DualPipeExpertParallel(BaseExpertParallel):
    def __init__(self, inner_ep: BaseExpertParallel):
        self.inner_ep = inner_ep

    def _token_dispatch(self, mod, inputs, device_mesh):
        """A -> dispatch -> B"""
        inputs = (SyncHook.apply(inputs[0], "A"),) + inputs[1:]
        outputs = self.inner_ep._token_dispatch(mod, inputs, device_mesh)
        outputs = (SyncHook.apply(outputs[0], "B"),) + outputs[1:]
        return outputs

    def _token_combine(self, mod, routed_output, device_mesh):
        """C -> combine -> D"""
        routed_output = SyncHook.apply(routed_output, "C")
        combine_output = self.inner_ep._token_combine(mod, routed_output, device_mesh)
        combine_output = SyncHook.apply(combine_output, "D")
        return combine_output
```

执行顺序变为：

```
A → [barrier] → dispatch → B → [barrier] → module → C → [barrier] → combine → D → [barrier]
```

每个 `[barrier]` 是一个 2 线程同步点，让前向线程和反向线程互相等待。

### 10.3 HookCoordinator

`dual_pipe_v.py:101-141`：

```python
class HookCoordinator:
    def __init__(self):
        self._execution_barrier = threading.Barrier(2)
        self._coordination_enabled = False
        self._cycle_count = 0
        self._num_layers = None

    def barrier(self):
        if not self.is_coordination_enabled():
            return
        try:
            self._execution_barrier.wait()
        except threading.BrokenBarrierError:
            pass

    def enable_coordination(self, num_layers=None):
        if num_layers is not None and num_layers > 0:
            self._coordination_enabled = True
            self._cycle_count = 0
            self._execution_barrier = threading.Barrier(2)
            self._num_layers = num_layers
```

`threading.Barrier(2)` 确保两个线程交替执行——一个线程到达 barrier 后等待另一个线程也到达，然后两者同时继续。

### 10.4 SyncHook — 自定义 Autograd Function

`dual_pipe_v.py:148-173`：

```python
class SyncHook(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, hook_name=""):
        ctx.hook_name = hook_name
        # 在 "D" hook 处递增 cycle 计数，超过层数时禁用协调
        if _hook_coordinator._coordination_enabled and hook_name == "D":
            _hook_coordinator._cycle_count += 1
            if not _hook_coordinator.check_should_continue_coordination():
                _hook_coordinator.disable_coordination()
                return x
        _hook_coordinator.barrier()
        return x

    @staticmethod
    def backward(ctx, grad_output):
        hook_name = ctx.hook_name
        # 反向的第一个 "D" hook 跳过 barrier（避免死锁）
        if hook_name == "D" and _hook_coordinator._cycle_count == 0:
            return grad_output, None
        _hook_coordinator.barrier()
        return grad_output, None
```

### 10.5 overlap_callback — 前向/反向并行执行

`dual_pipe_v.py:190-324` 实现了 `overlap_callback`，它是 DualPipeV 调度的核心：

```python
def overlap_callback(action, ctx):
    # 获取前向和反向的 stage
    fwd_action = action.sub_actions[0]
    bwd_action = action.sub_actions[1]

    # 计算 MoE 层数（取两个 stage 的最小值）
    min_num_layers = min(
        _count_moe_modules(forward_stage.submod),
        _count_moe_modules(backward_stage.submod),
    )

    # 启用协调
    _hook_coordinator.enable_coordination(num_layers=min_num_layers)
    main_stream = torch.accelerator.current_stream(device_type)

    def run_backward():
        device_module.set_stream(main_stream)  # 共享同一个 CUDA stream
        # ... 执行反向计算 ...

    def run_forward():
        # ... 执行前向计算 ...

    # 两个线程并行执行，通过 SyncHook 的 barrier 交替同步
    thread = threading.Thread(target=run_backward, daemon=True)
    thread.start()
    run_forward()
    thread.join()

    _hook_coordinator.disable_coordination()
```

**关键设计**：两个线程共享同一个 CUDA stream。这意味着��
- 它们 **不是真正的 GPU 并行**，而是 **CPU 线程交织提交 CUDA kernel**
- 当一个线程在等待 barrier 时（比如前向线程在等待 dispatch 完成），另一个线程可以提交其 CUDA kernel
- 效果是 EP 通信和 PP 计算在 CUDA stream 上交织排列

```
CUDA Stream 时序（理想情况）：
[fwd_attn] [dispatch_a2a] [bwd_attn] [fwd_experts] [combine_a2a] [bwd_experts] ...
           ← fwd 的 EP →  ← bwd →    ← fwd →       ← fwd EP →   ← bwd →
```

### 10.6 不兼容 AC

`dual_pipe_v.py:45-49`：

```python
if dual_pipe_v and job_config.activation_checkpoint.mode != "none":
    raise NotImplementedError(
        "Expert Parallel with DualPipeV and Activation Checkpointing "
        "cannot be used together."
    )
```

**原因**：AC 的 recompute 会重新执行前向，此时 SyncHook 的 barrier 计数会混乱——recompute 阶段没有对应的反向线程来配对。

---

## 第 11 节：完整并行化流程

> **一句话总结**：`parallelize_llama()` 按严格顺序执行六步并行化——TP → EP → CP → AC → Compile → FSDP，顺序至关重要因为每步都依赖前一步的结果。

### 11.1 六步流程详解

```python
def parallelize_llama(model, parallel_dims, job_config):
```

#### Step 1：apply_non_moe_tp()（`parallelize.py:91-110`）

```python
if parallel_dims.tp_enabled:
    tp_mesh = parallel_dims.get_mesh("tp")
    apply_non_moe_tp(model, tp_mesh, loss_parallel=..., enable_float8_tp=...)
    maybe_enable_async_tp(job_config, tp_mesh)
```

处理的模块：
- `tok_embeddings`：`RowwiseParallel(Replicate→Shard(1))`
- `norm`：`SequenceParallel()`
- `output`：`ColwiseParallel(Shard(1)→Replicate or Shard(-1))`
- 每个 TransformerBlock 的 attention：`wq/wk/wv` ColwiseParallel, `wo` RowwiseParallel
- Dense 层的 FFN：`w1/w3` ColwiseParallel, `w2` RowwiseParallel
- **跳过** MoE 层的 FFN（由 Step 2 处理）

#### Step 2：apply_moe_ep_tp()（`parallelize.py:137-148`）

```python
if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
    dual_pipe_v = get_dual_pipe_v_flag(job_config, parallel_dims)
    apply_moe_ep_tp(model, tp_mesh, ep_mesh, etp_mesh, ep_etp_mesh, dual_pipe_v, use_deepep)
```

处理的模块：
- MoE 层级的 TP：`moe` 的 input/output Shard, `router.gate` NoParallel, `shared_experts` ColwiseParallel/RowwiseParallel
- Experts 的 EP/TP：根据策略选择（详见第 4 节）
- 可选的 DualPipeV 包装

#### Step 3：Context Parallelism（`parallelize.py:150-157`）

```python
if parallel_dims.cp_enabled:
    apply_cp_to_attention_module(
        [block.attention.inner_attention for block in model.layers.values()],
        parallel_dims.get_mesh("cp"),
        attn_type,
    )
```

#### Step 4：Activation Checkpointing（`parallelize.py:162-175`）

```python
if job_config.activation_checkpoint.mode != "none":
    apply_ac(model, job_config.activation_checkpoint, ...)
```

#### Step 5：apply_compile()（`parallelize.py:178-179`）

```python
if model_compile_enabled:
    apply_compile(model, job_config.compile, parallel_dims.ep_enabled)
```

> **NOTE**：compile 必须在 AC 之后（AC 会包装模块，compile 需要看到包装后的结构），但在 FSDP 之前（FSDP 的钩子在 MoE block 中会导致 graph break）。

#### Step 6：apply_fsdp()（`parallelize.py:181-216`）

```python
if parallel_dims.fsdp_enabled or parallel_dims.ep_enabled:
    dp_mesh = parallel_dims.get_mesh(dp_mesh_names)
    edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)
    apply_fsdp(model, dp_mesh, ..., ep_degree, edp_mesh, gradient_divide_factor)
```

### 11.2 顺序依赖分析

```
Step 1 (TP)
  │ 原因：TP 切分权重为 DTensor，后续步骤需要感知切分状态
  ↓
Step 2 (EP/TP for MoE)
  │ 原因：EP 切分专家权重为 DTensor，FSDP 需要感知已切分的维度
  ↓
Step 3 (CP)
  │ 原因：CP 修改 attention 模块的行为，与 TP/EP 无冲突
  ↓
Step 4 (AC)
  │ 原因：AC 会用 CheckpointWrapper 包装 TransformerBlock
  │       compile 需要看到包装后的结构
  ↓
Step 5 (Compile)
  │ 原因：compile 会替换模块的 forward 为编译版本
  │       FSDP 需要在编译后的模块上注册钩子
  ↓
Step 6 (FSDP)
  │ 原因：FSDP 是最外层包装，需要知道所有内部切分
  │       prefetch 需要知道哪些模块是独立的 FSDP 单元
  ↓
Done
```

### 11.3 DeepEP 的初始化时机

`parallelize.py:113-135` 中 DeepEP 的初始化发生在 Step 2 之前：

```python
if job_config.parallelism.expert_parallel_comm_backend == "deepep":
    # 验证兼容性
    if not parallel_dims.ep_enabled:
        raise ValueError("DeepEP requires ep_degree > 1")
    if parallel_dims.etp_enabled:
        raise NotImplementedError("DeepEP with ETP is not supported")

    use_deepep = True

    # 导入模块以注册自定义 op
    import torchtitan.distributed.deepep  # noqa: F401

    # 将 DeepEP ops 加入 SAC save list
    _op_sac_save_list.add(torch.ops.deepep.dispatch.default)
    _op_sac_save_list.add(torch.ops.deepep.combine.default)
```

---

## 第 12 节：数值示例

> **一句话总结**：通过三个具体的 GPU 配置示例，展示 Dense Mesh 和 Sparse Mesh 的完整计算过程以及 GPU 分配方式。

### 12.1 示例 1：8 GPU（仅 TP，无 EP）

**配置**：
```
pp=1, dp_replicate=1, dp_shard=2, cp=1, tp=4, ep=1, etp=1
world_size = 1 * 1 * 2 * 1 * 4 * 1 = 8  ✓ (但实际验证用 pp*dp_rep*dp_shard*cp*tp)
```

**Mesh 计算**：
```python
batch = dp_replicate * dp_shard = 1 * 2 = 2
fsdp  = dp_shard * cp = 2 * 1 = 2
efsdp = fsdp * tp / (etp * ep) = 2 * 4 / (1 * 1) = 8
```

**三套 Mesh**：
```
dataloading: [pp=1, batch=2, cp=1, tp=4]
dense:       [pp=1, dp_replicate=1, fsdp=2, tp=4]
sparse:      [pp=1, dp_replicate=1, efsdp=8, ep=1, etp=1]
```

**GPU 布局**：
```
Dense Mesh (fsdp=2, tp=4):
  FSDP group 0: [GPU 0, 1, 2, 3]  ← TP group
  FSDP group 1: [GPU 4, 5, 6, 7]  ← TP group
  FSDP pairs: (GPU0, GPU4), (GPU1, GPU5), (GPU2, GPU6), (GPU3, GPU7)
```

**EP 策略**：`TensorParallel()`（因为 `ep=1`，ep_mesh 不存在）

**FSDP**：单层包装（所有层都在 dp_mesh 上）

### 12.2 示例 2：64 GPU（EP=8, ETP=1）

**配置**：
```
pp=1, dp_replicate=1, dp_shard=8, cp=1, tp=8, ep=8, etp=1
world_size = 1 * 1 * 8 * 1 * 8 = 64
```

**Mesh 计算**：
```python
batch = 1 * 8 = 8
fsdp  = 8 * 1 = 8
efsdp = 8 * 8 / (1 * 8) = 8
```

**三套 Mesh**：
```
dataloading: [pp=1, batch=8, cp=1, tp=8]
dense:       [pp=1, dp_replicate=1, fsdp=8, tp=8]
sparse:      [pp=1, dp_replicate=1, efsdp=8, ep=8, etp=1]
```

**GPU 布局**（以 rank 视角）：
```
Dense Mesh (fsdp=8, tp=8):
  每 8 个 GPU 组成一个 TP group：
    TP group 0: [0, 1, 2, 3, 4, 5, 6, 7]
    TP group 1: [8, 9, 10, 11, 12, 13, 14, 15]
    ...
    TP group 7: [56, 57, 58, 59, 60, 61, 62, 63]
  FSDP 跨 TP group 切分：GPU 0 和 GPU 8, 16, ..., 56 是一个 FSDP group

Sparse Mesh (efsdp=8, ep=8, etp=1):
  每 8 个 GPU 组成一个 EP group（与 TP group 相同的 GPU）：
    EP group 0: [0, 1, 2, 3, 4, 5, 6, 7]
    EP group 1: [8, 9, 10, 11, 12, 13, 14, 15]
    ...
  efsdp 跨 EP group 切分：GPU 0 和 GPU 8, 16, ..., 56 是一个 efsdp group
```

**EP 策略**：`ExpertParallel()`
- experts 权重 `Shard(0)` 在 ep_mesh 上
- 同时启用 `ReordererSequenceParallel()`（因为 EP 借用了全部 TP）

**FSDP**：
- Dense 层：单层在 dp_mesh（fsdp=8）上
- MoE 层：双层——内层在 edp_mesh（efsdp=8）上，外层在 dp_mesh（fsdp=8）上

**条件 Shard 检查**（假设 num_experts=128）：
```
efsdp * ep = 8 * 8 = 64 < 128 → 使用默认 Shard(0)
```

### 12.3 示例 3：64 GPU（EP=8, ETP=8）

**配置**：
```
pp=1, dp_replicate=1, dp_shard=1, cp=1, tp=8, ep=8, etp=8
world_size = 1 * 1 * 1 * 1 * 8 = 8...
```

> **NOTE**：这个配置在 64 GPU 下不成立（`1*1*1*1*8=8`），让我们修正为实际可行的配置。

**修正配置**：
```
pp=1, dp_replicate=1, dp_shard=8, cp=1, tp=8, ep=8, etp=8
world_size = 1 * 1 * 8 * 1 * 8 = 64
```

**Mesh 计算**：
```python
batch = 1 * 8 = 8
fsdp  = 8 * 1 = 8
efsdp = 8 * 8 / (8 * 8) = 1
```

**三套 Mesh**：
```
dataloading: [pp=1, batch=8, cp=1, tp=8]
dense:       [pp=1, dp_replicate=1, fsdp=8, tp=8]
sparse:      [pp=1, dp_replicate=1, efsdp=1, ep=8, etp=8]
```

**GPU 布局**：
```
Dense Mesh 与示例 2 相同

Sparse Mesh (efsdp=1, ep=8, etp=8):
  EP+ETP 组合 mesh 大小 = 8 * 8 = 64（覆盖全部 GPU）
  但由于 efsdp=1，没有跨 EP 的 FSDP 切分
  EP 和 ETP 共同构成一个 64 GPU 的二维 mesh：
    ep 维度：8 个 rank group
    etp 维度：每个 ep group 内 8 个 TP rank

  注意：efsdp=1 但 ep>1，所以 efsdp mesh 仍然存在（_mesh_exist 的特殊处理）
  edp_mesh 大小为 1，FSDP 只负责混合精度转换
```

**EP 策略**：`ExpertTensorParallel()`
- experts 权重在 ep_etp_mesh 上使用 `[Shard(0), Shard(1)]` / `[Shard(0), Shard(2)]`
- EP 在 dim 0 切 8 份，ETP 在 dim 1/2 切 8 份

**FSDP**：
- Dense 层：单层在 dp_mesh（fsdp=8）上
- MoE 层：双层——内层在 edp_mesh（efsdp=1）上（仅混合精度），外层在 dp_mesh（fsdp=8）上

**条件 Shard 检查**（假设 num_experts=128）：
```
efsdp * ep = 1 * 8 = 8 < 128 → 使用默认 Shard(0)
```

### 12.4 配置对比总结

| 参数 | 示例 1 | 示例 2 | 示例 3 |
|------|--------|--------|--------|
| GPU 数 | 8 | 64 | 64 |
| pp | 1 | 1 | 1 |
| dp_replicate | 1 | 1 | 1 |
| dp_shard | 2 | 8 | 8 |
| cp | 1 | 1 | 1 |
| tp | 4 | 8 | 8 |
| ep | 1 | 8 | 8 |
| etp | 1 | 1 | 8 |
| **fsdp** | **2** | **8** | **8** |
| **efsdp** | **8** | **8** | **1** |
| EP 策略 | TensorParallel | ExpertParallel | ExpertTensorParallel |
| FSDP 层数 | 单层 | 双层 | 双层 |
| Prefetch | 隐式 | 显式 | 显式 |
| 通信模式 | TP all-gather/reduce-scatter | EP all-to-all + TP sequence-parallel | EP all-to-all + ETP reduce |

---

## 附录：关键文件与行号索引

| 文件路径 | 行号范围 | 内容 |
|---------|---------|------|
| `parallelize.py` | 69-226 | `parallelize_llama()` 主入口 |
| `parallelize.py` | 229-321 | `apply_non_moe_tp()` |
| `parallelize.py` | 324-497 | `apply_fsdp()` + prefetch |
| `parallelize.py` | 500-590 | `apply_moe_ep_tp()` |
| `parallelize.py` | 593-690 | `apply_compile()` |
| `expert_parallel.py` | 49-87 | `TensorParallel` |
| `expert_parallel.py` | 89-192 | `ExpertParallel` |
| `expert_parallel.py` | 196-253 | `ExpertTensorParallel` |
| `expert_parallel.py` | 256-317 | `ReordererSequenceParallel` |
| `expert_parallel.py` | 320-385 | `DeepEPExpertParallel` |
| `parallel_dims.py` | 19-27 | `ParallelDims` 字段 |
| `parallel_dims.py` | 68-175 | `build_mesh()` |
| `parallel_dims.py` | 350-355 | `fsdp_gradient_divide_factor` |
| `moe.py` | 131-178 | `GroupedExperts` |
| `moe.py` | 186-353 | `TokenChoiceTopKRouter` |
| `moe.py` | 361-416 | `TokenReorderer` |
| `moe.py` | 419-553 | `MoE` |
| `utils.py` | 42-66 | `_permute()` / `_unpermute()` |
| `kernels.py` | 143-216 | `generate_permute_indices()` |
| `deepep.py` | 57-73 | 自定义 Op 注册 |
| `deepep.py` | 75-168 | dispatch Op + backward |
| `deepep.py` | 171-243 | combine Op + backward |
| `deepep.py` | 247-285 | `sync_combine()` |
| `deepep.py` | 320-380 | `_permute_tokens()` / `_unpermute_tokens()` |
| `deepep.py` | 393-481 | `dispatch_tokens()` |
| `deepep.py` | 484-501 | `combine_tokens()` |
| `moe_deepep.py` | 16-81 | `DeepEPMoE` |
| `dual_pipe_v.py` | 54-98 | `DualPipeExpertParallel` |
| `dual_pipe_v.py` | 101-141 | `HookCoordinator` |
| `dual_pipe_v.py` | 148-173 | `SyncHook` |
| `dual_pipe_v.py` | 190-324 | `overlap_callback()` |
| `model.py` | 344-457 | `TransformerBlock` |
| `model.py` | 459-601 | `Transformer` |
