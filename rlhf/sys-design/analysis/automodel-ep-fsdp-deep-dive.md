# 深入分析 Automodel 框架 EP + FSDP 二次开发实现

> 本文是 [deep-dive-moe-ep-fsdp-step-by-step.md](./deep-dive-moe-ep-fsdp-step-by-step.md) Step 8 中 Automodel 框架部分的深度展开。原文仅概述了调用链和有状态上下文管理的概念，本文将逐行剖析 Automodel（NVIDIA NeMo）的源码实现，覆盖从 DeviceMesh 创建到检查点管理的完整链路。

---

## 目录

1. [概述与定位](#1-概述与定位)
2. [双 DeviceMesh 架构](#2-双-devicemesh-架构)
3. [EP 切分的实现](#3-ep-切分的实现)
4. [FSDP 二次开发的核心](#4-fsdp-二次开发的核心)
5. [parallelize_model 编排](#5-parallelize_model-编排)
6. [DeepEP 通信后端](#6-deepep-通信后端)
7. [两种专家实现对比](#7-两种专家实现对比)
8. [梯度同步与 Pipeline 并行适配](#8-梯度同步与-pipeline-并行适配)
9. [检查点与 State Dict 管理](#9-检查点与-state-dict-管理)

---

## 1. 概述与定位

Automodel 是 NVIDIA NeMo 团队推出的训练框架，在 MoE 训练场景中做了三个关键决策：

1. **双 DeviceMesh**：为普通参数和专家参数分别创建独立的 DeviceMesh（`world_mesh` 和 `moe_mesh`），避免维度冲突
2. **DeepEP 替代 NCCL**：不使用 PyTorch 原生的 `dist.all_to_all`，而是集成 DeepSeek 的 DeepEP 库实现 RDMA 直驱的 All-to-All 通信
3. **FSDP 二次开发**：在 PyTorch FSDP2 的基础上，针对 MoE 专家参数做了精细化控制——专家参数有独立的 FSDP wrap、独立的梯度同步策略

**源码文件结构**：

```
nemo_automodel/
├── components/
│   ├── distributed/
│   │   └── fsdp2.py                    # FSDP2Manager：双 DeviceMesh 创建
│   └── moe/
│       ├── parallelizer.py             # EP+FSDP 编排核心
│       ├── layers.py                   # 专家实现 + Gate + MoE 模块
│       ├── fsdp_mixin.py               # MoE 梯度同步管理
│       ├── state_dict_mixin.py         # 检查点格式转换
│       ├── state_dict_utils.py         # DTensor 感知的专家切分
│       └── megatron/
│           ├── token_dispatcher.py     # DeepEP Token 调度
│           ├── fused_a2a.py            # 融合 All-to-All 自动微分
│           ├── moe_utils.py            # Token permute/unpermute
│           └── fused_indices_converter.py  # Triton indices↔multihot
└── recipes/
    └── llm/
        └── train_ft.py                 # 训练循环集成
```

---

## 2. 双 DeviceMesh 架构

Automodel 的核心设计之一是为普通参数和专家参数分别创建两个 DeviceMesh。这不是因为"好看"，而是因为 EP 的切分维度和 FSDP 的切分维度在数学上不同——普通参数按 `(dp_replicate, dp_shard)` 切，而专家参数按 `(ep_shard, ep)` 切。

### 2.1 world_mesh 的创建

> 源码路径：`nemo_automodel/components/distributed/fsdp2.py` — `FSDP2Manager._get_device_mesh()`

```python
def _get_device_mesh(self):
    mesh_shape = (self.pp_size, self.dp_replicate_size, self.dp_shard_size, self.cp_size, self.tp_size)
    mesh_names = ("pp", "dp_replicate", "dp_shard", "cp", "tp")

    self.device_mesh = init_device_mesh(
        device_type="cuda" if self.backend == "nccl" else "cpu",
        mesh_shape=mesh_shape,
        mesh_dim_names=mesh_names,
    )

    # 创建扁平化子网格：
    dp_mesh_dim_names = ["dp_replicate", "dp_shard"]
    dp_shard_cp_mesh_dim_names = ["dp_shard", "cp"]
    dp_cp_mesh_dim_names = ["dp_replicate", "dp_shard", "cp"]

    self.device_mesh[tuple(dp_mesh_dim_names)]._flatten(mesh_dim_name="dp")
    self.device_mesh[tuple(dp_shard_cp_mesh_dim_names)]._flatten(mesh_dim_name="dp_shard_cp")
    self.device_mesh[tuple(dp_cp_mesh_dim_names)]._flatten(mesh_dim_name="dp_cp")
    return self.device_mesh
```

world_mesh 是一个 5 维的 DeviceMesh：`(pp, dp_replicate, dp_shard, cp, tp)`。同时通过 `_flatten` 创建了三个扁平化子网格：
- `dp` = `dp_replicate × dp_shard`（数据并行，用于 dataloader 分片）
- `dp_shard_cp` = `dp_shard × cp`（参数切分的实际维度）
- `dp_cp` = `dp_replicate × dp_shard × cp`（loss all-reduce 的维度）

### 2.2 moe_mesh 的创建

> 源码路径：`nemo_automodel/components/distributed/fsdp2.py` — `FSDP2Manager._get_moe_mesh()`

```python
def _get_moe_mesh(self):
    mesh_shape = (self.pp_size, self.ep_shard_size, self.ep_size)
    mesh_names = ("pp", "ep_shard", "ep")

    self.moe_mesh = init_device_mesh(
        device_type="cuda" if self.backend == "nccl" else "cpu",
        mesh_shape=mesh_shape,
        mesh_dim_names=mesh_names,
    )
    return self.moe_mesh
```

moe_mesh 是一个独立的 3 维 DeviceMesh：`(pp, ep_shard, ep)`。注意它和 world_mesh 完全独立——它们覆盖的 GPU 集合相同，但维度划分不同。

### 2.3 ep_shard_size 的推导

> 源码路径：`nemo_automodel/components/distributed/fsdp2.py` — `FSDP2Manager._setup_distributed()`

```python
dp_cp_size = self.dp_size * self.cp_size
assert dp_cp_size % self.ep_size == 0, f"{dp_cp_size=} must be a multiple of {self.ep_size=}"
if self.ep_size < dp_cp_size:
    self.ep_shard_size = dp_cp_size // self.ep_size
else:
    self.ep_shard_size = 1
```

`ep_shard_size` 的计算逻辑：在 DP+CP 构成的所有 GPU 中，EP 占了 `ep_size` 个 GPU 用来切专家。剩下的 `dp_cp_size / ep_size` 个 GPU 构成了 `ep_shard`——这些 GPU 上的专家参数需要用 FSDP 做进一步切分。

**举例**：假设 `dp_size=8, cp_size=1, ep_size=4`：
- `dp_cp_size = 8`
- `ep_shard_size = 8 / 4 = 2`
- moe_mesh 形状为 `(pp, 2, 4)`，即每 4 个 GPU 做 EP 切分，每 2 个 GPU 做 FSDP 切分

```
世界布局示意图（8 GPUs, EP=4, ep_shard=2）:

world_mesh (dp_replicate=1, dp_shard=8):
  GPU: [0, 1, 2, 3, 4, 5, 6, 7]
        ←———— dp_shard=8 ————→

moe_mesh (ep_shard=2, ep=4):
  GPU: [0, 1, 2, 3, 4, 5, 6, 7]
        ←ep_shard→  ←ep_shard→
        ← ep=4 →   ← ep=4 →

  ep_shard group 0: {GPU 0, GPU 4}  ←— 这两个 GPU 上的专家做 FSDP
  ep_shard group 1: {GPU 1, GPU 5}
  ep_shard group 2: {GPU 2, GPU 6}
  ep_shard group 3: {GPU 3, GPU 7}

  ep group 0: {GPU 0, 1, 2, 3}  ←— 这 4 个 GPU 各持有不同的专家子集
  ep group 1: {GPU 4, 5, 6, 7}
```

---

## 3. EP 切分的实现

### 3.1 ExpertParallel ParallelStyle

> 源码路径：`nemo_automodel/components/moe/parallelizer.py:51-74`

```python
class ExpertParallel(ParallelStyle):
    """
    ExpertParallel class is used to shard the MoE parameters on the EP mesh.
    Dim `0` of each parameter is sharded since that is the expert dimension.
    """

    def _partition_fn(self, name, module, device_mesh):
        assert device_mesh.ndim == 1

        for name, param in module.named_parameters(recurse=False):
            dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
            dist_param.requires_grad = param.requires_grad
            module.register_parameter(name, dist_param)

        if isinstance(module, GroupedExpertsDeepEP):
            module.init_token_dispatcher(ep_mesh=device_mesh)

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            self._partition_fn,
        )
```

关键设计：

1. **`Shard(0)` 切分**：专家参数的 shape 是 `[n_experts, dim, inter_dim]`（或 `[n_experts, inter_dim, dim]`），dim 0 就是专家维度。`Shard(0)` 意味着每个 EP rank 只持有 `n_experts / ep_size` 个专家。

2. **DTensor 转换**：`distribute_tensor` 把普通 Tensor 转成 DTensor，自动处理切片和元数据。转换后，`param.to_local()` 返回本地切片，`param.full_tensor()` 返回全量。

3. **条件初始化 Token Dispatcher**：只有 `GroupedExpertsDeepEP`（使用 DeepEP 后端）才需要初始化 `MoEFlexTokenDispatcher`。这是因为 DeepEP 需要知道 EP mesh 的进程组信息来建立通信通道。

### 3.2 apply_ep 函数

> 源码路径：`nemo_automodel/components/moe/parallelizer.py:77-95`

```python
def apply_ep(model: nn.Module, ep_mesh: DeviceMesh):
    """Applies EP to MoE module."""
    assert ep_mesh.size() > 1

    if hasattr(model, "model") and model.model is not None:
        _model = model.model
    else:
        _model = model
    _model = get_text_module(_model)

    for _, block in _model.layers.named_children():
        moe_module = block.moe if hasattr(block, "moe") else block.mlp
        if isinstance(moe_module, MoE):
            parallelize_module(
                module=moe_module.experts,
                device_mesh=ep_mesh,
                parallelize_plan=ExpertParallel(),
            )
```

`apply_ep` 遍历所有 transformer block，找到 `MoE` 模块，然后对其 `experts` 子模块应用 `ExpertParallel`。注意：**只切 experts，不切 gate**——Gate 的权重在所有 rank 上是完整的（因为每个 rank 都需要完整的路由信息）。

### 3.3 init_token_dispatcher 详解

> 源码路径：`nemo_automodel/components/moe/layers.py:525-549`

```python
def init_token_dispatcher(self, ep_mesh: DeviceMesh):
    self.ep_size = ep_mesh.size()
    self.ep_rank = ep_mesh.get_local_rank()

    config = MegatronMoEConfig(
        moe_router_topk=self.config.n_activated_experts,
        num_moe_experts=self.config.n_routed_experts,
        moe_permute_fusion=True,
        moe_enable_deepep=True,
    )

    self.n_routed_experts = self.config.n_routed_experts
    num_local_experts = self.config.n_routed_experts // self.ep_size
    local_expert_indices_offset = self.ep_rank * num_local_experts
    local_expert_indices = [local_expert_indices_offset + i for i in range(num_local_experts)]

    self.token_dispatcher = MoEFlexTokenDispatcher(
        num_local_experts=num_local_experts,
        local_expert_indices=local_expert_indices,
        config=config,
        ep_group=ep_mesh.get_group(),
    )
```

这个方法在 EP 切分时被调用，创建 `MoEFlexTokenDispatcher`。它计算出本 rank 持有的专家 ID 范围（例如 rank 0 持有 expert 0-15，rank 1 持有 expert 16-31），并把 EP 进程组传给 dispatcher。

---

## 4. FSDP 二次开发的核心

这是 Automodel 最精妙的部分。`apply_fsdp` 实现了三重逻辑：(1) 专家的 FSDP Shard(1)，(2) ignored_params 排除专家参数，(3) block 级 FSDP wrap。

> 源码路径：`nemo_automodel/components/moe/parallelizer.py:154-255`

### 4.1 完整的 apply_fsdp 函数

```python
def apply_fsdp(
    model: torch.nn.Module,
    fsdp_mesh: DeviceMesh,
    ep_enabled: bool,
    ep_shard_enabled: bool,
    ep_shard_mesh: DeviceMesh | None = None,
    mp_policy: MixedPrecisionPolicy | None = None,
    offload_policy: OffloadPolicy | None = None,
    reshard_after_forward: bool = False,
    lm_head_precision: str | torch.dtype | None = None,
    wrap_outer_model: bool = True,
):
    # 默认混合精度策略
    if mp_policy is None:
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=torch.bfloat16,
            cast_forward_inputs=True,
        )

    # 构造 fully_shard 的默认偏函数
    fully_shard_default = functools.partial(
        fully_shard,
        mesh=fsdp_mesh,
        reshard_after_forward=reshard_after_forward,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
    )

    _model = get_text_module(model.model if hasattr(model, "model") else model)

    for _, block in _model.layers.named_children():
        moe_module = block.moe if hasattr(block, "moe") else block.mlp

        # ——— 第一重：专家的独立 FSDP ———
        if isinstance(moe_module, MoE) and ep_shard_enabled:
            fully_shard(
                moe_module.experts,
                mesh=ep_shard_mesh,                    # 用 moe_mesh 的子网格！
                shard_placement_fn=lambda _: Shard(1), # 沿 dim=1 切！
                reshard_after_forward=reshard_after_forward,
            )

        # ——— 第二重：排除专家参数 ———
        ignored_params = None
        if isinstance(moe_module, MoE) and ep_enabled:
            ignored_params = set(moe_module.experts.parameters())

        # ——— 第三重：block 级 FSDP ———
        fully_shard_default(block, ignored_params=ignored_params)

    # Embedding 和 LM Head 的 FSDP
    if hasattr(_model, "embed_tokens") and _model.embed_tokens is not None:
        fully_shard_default(_model.embed_tokens)

    lm_head = getattr(_model, "lm_head", None) or getattr(model, "lm_head", None)
    if lm_head is not None:
        if lm_head_precision == torch.float32:
            # LM Head 使用 FP32 精度
            lm_head_mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
                output_dtype=torch.float32,
            )
            fully_shard(lm_head, mesh=fsdp_mesh, reshard_after_forward=reshard_after_forward,
                        mp_policy=lm_head_mp_policy, offload_policy=offload_policy)
        else:
            fully_shard_default(lm_head)

    fully_shard_default(_model)

    if wrap_outer_model and model is not _model:
        fully_shard_default(model)
```

### 4.2 三重逻辑详解

**第一重：专家的 Shard(1)**

```python
if isinstance(moe_module, MoE) and ep_shard_enabled:
    fully_shard(
        moe_module.experts,
        mesh=ep_shard_mesh,                    # ← 注意：用的是 moe_mesh 的子网格
        shard_placement_fn=lambda _: Shard(1), # ← 注意：切 dim=1，不是 dim=0
        reshard_after_forward=reshard_after_forward,
    )
```

为什么切 dim=1 而不是 dim=0？因为 dim=0 已经被 EP 切了（专家维度）。如果 `ep_shard_size > 1`，说明在 EP 切分之后，每个 EP group 内还有多余的 GPU 需要进一步切分参数。此时只能沿 dim=1（hidden 维度）来切。

```
专家权重 gate_and_up_projs 的 shape:
[n_local_experts, dim, 2*inter_dim]
      ↑              ↑
   EP 已切 dim=0    FSDP 切 dim=1
```

**第二重：ignored_params**

```python
ignored_params = None
if isinstance(moe_module, MoE) and ep_enabled:
    ignored_params = set(moe_module.experts.parameters())
```

当 EP 启用时，专家参数已经被 EP（和可选的 ep_shard FSDP）管理了。必须把它们从 block 级 FSDP 中排除，否则会出现双重切分。

注释中的解释很精确：
- 如果 `ep_shard_enabled`（专家有独立 FSDP），PyTorch FSDP 会自动排除已经 wrap 过的子模块参数
- 如果只有 EP 没有 ep_shard（即 PP × EP 已经把参数完全切完了），就需要手动用 `ignored_params` 排除

**第三重：block 级 FSDP**

```python
fully_shard_default(block, ignored_params=ignored_params)
```

对整个 transformer block 做 FSDP wrap，但排除专家参数。这意味着 attention、layernorm、gate 权重等由 `world_mesh` 的 FSDP 管理，专家权重由 `moe_mesh` 的 FSDP 管理。

### 4.3 FSDP wrap 顺序示意

```
Model
├── embed_tokens ←──── fully_shard(fsdp_mesh)
├── layers
│   └── Block[i]  ←──── fully_shard(fsdp_mesh, ignored_params=expert_params)
│       ├── self_attn   (由 block 的 FSDP 管理)
│       ├── layernorm   (由 block 的 FSDP 管理)
│       └── MoE
│           ├── gate    (由 block 的 FSDP 管理)
│           └── experts ←──── fully_shard(ep_shard_mesh, Shard(1))  [独立！]
├── lm_head ←──── fully_shard(fsdp_mesh)
└── _model ←──── fully_shard(fsdp_mesh)  [最外层 wrap]
```

---

## 5. parallelize_model 编排

> 源码路径：`nemo_automodel/components/moe/parallelizer.py:278-332`

```python
def parallelize_model(
    model: torch.nn.Module,
    world_mesh: DeviceMesh,
    moe_mesh: DeviceMesh | None,
    *,
    dp_axis_names: tuple[str, ...],
    cp_axis_name: str | None = None,
    tp_axis_name: str | None = None,
    ep_axis_name: str | None = None,
    ep_shard_axis_names: tuple[str, ...] | None = None,
    activation_checkpointing: bool = False,
    ignore_router_for_ac: bool = False,
    reshard_after_forward: bool = False,
    lm_head_precision: str | torch.dtype | None = None,
    wrap_outer_model: bool = True,
):
    # 1. Context Parallelism
    cp_enabled = cp_axis_name is not None and world_mesh[cp_axis_name].size() > 1
    if cp_enabled:
        apply_cp(model, world_mesh[cp_axis_name])

    # 2. Expert Parallelism
    ep_enabled = ep_axis_name is not None and moe_mesh is not None and moe_mesh[ep_axis_name].size() > 1
    if ep_enabled:
        assert model.model.moe_config.n_routed_experts % moe_mesh[ep_axis_name].size() == 0
        apply_ep(model, moe_mesh[ep_axis_name])

    # 3. Activation Checkpointing
    if activation_checkpointing:
        apply_ac(model, ignore_router=ignore_router_for_ac)

    # 4. FSDP
    ep_shard_mesh = moe_mesh[ep_shard_axis_names] if ep_shard_axis_names is not None else None
    fsdp_enabled = dp_axis_names is not None and world_mesh[dp_axis_names].size() > 1
    fsdp_mesh = world_mesh[tuple(dp_axis_names)] if fsdp_enabled else None
    if fsdp_enabled:
        apply_fsdp(
            model, fsdp_mesh,
            ep_enabled=ep_enabled,
            ep_shard_enabled=ep_shard_mesh is not None and ep_shard_mesh.size() > 1,
            ep_shard_mesh=ep_shard_mesh,
            reshard_after_forward=reshard_after_forward,
            lm_head_precision=lm_head_precision,
            wrap_outer_model=wrap_outer_model,
        )
```

### 5.1 严格的执行顺序：CP → EP → AC → FSDP

这个顺序不是随��的，每一步都依赖前一步的结果：

```
Step 1: CP (Context Parallelism)
  └─ 设置 TransformerEngine 的 CP group，不改变参数形状
  └─ 必须最先做，因为 AC 和 FSDP 会修改模块结构

Step 2: EP (Expert Parallelism)
  └─ 把专家参数转成 DTensor，沿 dim=0 切分
  └─ 必须在 FSDP 之前，因为 FSDP 的 Shard(1) 和 ignored_params 都依赖 EP 的结果
  └─ 如果是 DeepEP 后端，还会初始化 token_dispatcher

Step 3: AC (Activation Checkpointing)
  └─ 用 checkpoint_wrapper 包裹每个 block
  └─ 必须在 EP 之后（EP 改变了模块内部结构）
  └─ 必须在 FSDP 之前（FSDP 不能 wrap 已经被 checkpoint_wrapper 修改过的模块）

Step 4: FSDP
  └─ 最后做，因为需要知道哪些参数被 EP 管理了（用于 ignored_params）
  └─ FSDP wrap 会改变参数的存储方式（FlatParam），之后不能再做 EP 切分
```

### 5.2 TP 的禁用

```python
assert tp_axis_name is None or world_mesh[tp_axis_name].size() == 1, (
    "Tensor parallelism not supported for custom MoE models"
)
```

Automodel 的 MoE 并行化器不支持 TP。这是因为 EP + TP 的组合需要复杂的 `TPxEP` 通信组管理，目前 Automodel 选择不实现这个组合。如果模型需要 TP，走的是另一条代码路径（通过 `FSDP2Manager.parallelize` 中的 HF TP plan）。

---

## 6. DeepEP 通信后端

Automodel 使用 DeepEP 替代 PyTorch 原生的 `dist.all_to_all` 做专家间的 Token 通信。这涉及四个层次的组件。

### 6.1 Buffer 管理

> 源码路径：`nemo_automodel/components/moe/megatron/fused_a2a.py:22-77`

```python
try:
    from deep_ep import Buffer
    from deep_ep.utils import EventHandle, EventOverlap
    HAVE_DEEP_EP = True
    Buffer.set_num_sms(int(os.environ.get("DEEP_EP_SM_NUMS", 20)))
except ImportError:
    HAVE_DEEP_EP = False

_buffer = None

def get_buffer(group: torch.distributed.ProcessGroup, hidden_bytes: int):
    global _buffer
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (
        Buffer.get_dispatch_config(group.size()),
        Buffer.get_combine_config(group.size()),
    ):
        num_nvl_bytes = max(config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes)
        num_rdma_bytes = max(config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes)

    if (
        _buffer is None
        or _buffer.group != group
        or _buffer.num_nvl_bytes < num_nvl_bytes
        or _buffer.num_rdma_bytes < num_rdma_bytes
    ):
        _buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)
    return _buffer
```

关键细节：
- **Buffer 是全局单例**：模块级别的 `_buffer` 变量，所有 MoE 层共用同一个 Buffer 实例
- **惰性分配**：只在 Buffer 不存在或大小不够时才重新分配
- **双通道**：DeepEP 区分 NVLink（节点内）和 RDMA（节点间）两种通信通道，分别计算所需 buffer 大小
- **SM 数量配置**：通过环境变量 `DEEP_EP_SM_NUMS` 控制分配给 DeepEP 的 SM 数量，默认 20

### 6.2 FusedDispatch 自动微分

> 源码路径：`nemo_automodel/components/moe/megatron/fused_a2a.py:80-177`

```python
class FusedDispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, token_indices, token_probs, num_experts, group,
                async_finish=False, allocate_on_comm_stream=False):
        # 如果 async_finish，创建事件用于流同步
        previous_event = None
        if async_finish:
            previous_event = EventOverlap(EventHandle())

        buffer = get_buffer(group, get_hidden_bytes(x))

        # Step 1: 计算通信布局
        (num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert,
         is_token_in_rank, event) = buffer.get_dispatch_layout(
            token_indices, num_experts,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

        # Step 2: 执行 dispatch（融合了 permute + all-to-all）
        (recv_x, recv_token_indices, recv_token_probs,
         num_recv_tokens_per_expert_list, handle, after_event_overlap
        ) = buffer.dispatch(
            x, topk_idx=token_indices, topk_weights=token_probs,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=event,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

        # 确保当前 stream 同步完成
        if async_finish:
            after_event_overlap.current_stream_wait()

        # 保存 handle 给 backward
        ctx.group = group
        ctx.handle = handle
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        tokens_per_expert = torch.tensor(num_recv_tokens_per_expert_list)
        return (recv_x, recv_token_indices, recv_token_probs, tokens_per_expert, handle)

    @staticmethod
    def backward(ctx, grad_output, grad_token_indices, grad_token_probs,
                 grad_tokens_per_expert, grad_handle):
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output))
        # backward 的 dispatch → forward 的 combine
        grad_x, grad_token_probs, after_event = buffer.combine(
            grad_output.contiguous(), ctx.handle,
            topk_weights=grad_token_probs.float(),
            previous_event=None,
            async_finish=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )
        if ctx.async_finish:
            after_event.current_stream_wait()
        return grad_x, None, grad_token_probs, None, None, None, None
```

**核心对称性**：
- `FusedDispatch.forward` 调用 `buffer.dispatch`（发送 token 到专家）
- `FusedDispatch.backward` 调用 `buffer.combine`（把梯度收回来）
- `FusedCombine.forward` 调用 `buffer.combine`（收集专家输出）
- `FusedCombine.backward` 调用 `buffer.dispatch`（把梯度发出去）

这是因为 All-to-All 的反向传播就是方向相反的 All-to-All。

### 6.3 FusedCombine 自动微分

> 源码路径：`nemo_automodel/components/moe/megatron/fused_a2a.py:179-223`

```python
class FusedCombine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group, handle, async_finish=False, allocate_on_comm_stream=False):
        previous_event = None
        if async_finish:
            previous_event = EventOverlap(EventHandle())
        buffer = get_buffer(group, get_hidden_bytes(x))
        combined_x, _, after_event = buffer.combine(
            x, handle=handle,
            async_finish=async_finish,
            previous_event=previous_event,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        if async_finish:
            after_event.current_stream_wait()
        ctx.handle = handle  # 保存 handle 给 backward
        ctx.group = group
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        return combined_x, None

    @staticmethod
    def backward(ctx, grad_output, previous_event=None):
        previous_event = None
        if ctx.async_finish:
            previous_event = EventOverlap(EventHandle())
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output))
        # backward 的 combine → forward 的 dispatch
        grad_x, _, _, _, _, after_event = buffer.dispatch(
            grad_output.contiguous(),
            handle=ctx.handle,
            previous_event=previous_event,
            async_finish=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )
        if ctx.async_finish:
            after_event.current_stream_wait()
        return grad_x, None, None, None, None
```

### 6.4 Handle 机制

Handle 是 DeepEP 的核心概念——它记录了 dispatch 时的通信布局：哪些 token 发给了哪个 rank、每个 rank 收到了多少 token 等。combine 时需要这些信息来把结果发回正确的源 GPU。

```
Forward 数据流：

dispatch 阶段：                    combine 阶段：
Input tokens ─── dispatch ──→ Expert compute ─── combine ──→ Output tokens
                 ↓ 产生 handle                    ↑ 消费 handle
              handle 保存了                    handle 指导了
              token→rank 的映射               result→rank 的反向映射
```

### 6.5 _DeepepManager 生命周期

> 源码路径：`nemo_automodel/components/moe/megatron/token_dispatcher.py:90-308`

```python
class _DeepepManager(_DispatchManager):
    def __init__(self, group, router_topk, ...):
        self.group = group
        self.handle = None           # 每次 dispatch 后保存
        self.token_indices = None
        self.token_probs = None

    def dispatch(self, hidden_states, ...):
        (hidden_states, dispatched_indices, dispatched_probs,
         num_tokens_per_expert, handle) = fused_dispatch(
            hidden_states, self.token_indices, self.token_probs,
            self.num_experts, self.group, ...)
        self.handle = handle          # 保存 handle
        self.tokens_per_expert = num_tokens_per_expert
        self.dispatched_indices = dispatched_indices
        self.dispatched_probs = dispatched_probs
        return hidden_states

    def combine(self, hidden_states, ...):
        hidden_states, _ = fused_combine(
            hidden_states, self.group, self.handle, ...)
        self.handle = None            # 释放 handle
        return hidden_states
```

**共享管理器**：`MoEFlexTokenDispatcher` 使用类变量 `shared_comm_manager` 实现跨层共享：

```python
class MoEFlexTokenDispatcher:
    shared_comm_manager: _DeepepManager = None  # 所有实例共享

    def __init__(self, ...):
        if SHARING_DEEPEP_MANAGER:
            if MoEFlexTokenDispatcher.shared_comm_manager is None:
                MoEFlexTokenDispatcher.shared_comm_manager = _DeepepManager(...)
            self._comm_manager = MoEFlexTokenDispatcher.shared_comm_manager
```

这意味着模型的所有 MoE 层共享同一个 `_DeepepManager` 实例（进而共享同一个 DeepEP Buffer）。这减少了内存开销，但也意味着不同层不能并行通信（它们复用同一个 handle 状态）。

### 6.6 Triton Indices ↔ Multihot 转换

> 源码路径：`nemo_automodel/components/moe/megatron/fused_indices_converter.py`

DeepEP dispatch 返回的路由信息是 indices 格式（每个 token 的 top-k 专家 ID），但 permute/unpermute 操作需要 multihot 格式（每个 token 对每个专家的 0/1 标记）。Automodel 用 Triton kernel 做高效转换：

```python
@triton.jit
def _indices_to_multihot_kernel(
    indices_ptr, probs_in_indices_ptr,
    multihot_indices_ptr, probs_in_multihot_ptr, position_map_ptr,
    num_of_local_experts, ..., topk, ...,
):
    # 每个 Triton program 处理一个 token
    row_idx = tl.program_id(0)

    # 加载这个 token 的 top-k indices
    indices_row = tl.load(indices_ptr + row_idx * topk + topk_row, mask=topk_row_mask)
    probs_row = tl.load(probs_in_indices_ptr + row_idx * topk + topk_row, mask=topk_row_mask)

    # 初始化 multihot 输出为 0
    tl.store(multihot_indices_ptr + row_idx_offset + num_exp_row, 0, mask=num_exp_row_mask)
    tl.debug_barrier()

    # 在对应位置写入 1 和 prob
    mask = (indices_row != -1) & (indices_row < num_of_local_experts)
    tl.store(multihot_indices_ptr + row_idx_offset + indices_row, 1, mask)
    tl.store(probs_in_multihot_ptr + row_idx_offset + indices_row, probs_row, mask)
    tl.store(position_map_ptr + row_idx_offset + indices_row, position_row, mask)
```

转换示例：
```
输入 (indices 格式):
  indices = [[0, 1], [1, 2]]         # 每个 token 选了 2 个专家
  probs   = [[0.1, 0.2], [0.3, 0.4]]

输出 (multihot 格式, 4 个本地专家):
  multihot = [[1, 1, 0, 0], [0, 1, 1, 0]]
  probs    = [[0.1, 0.2, 0, 0], [0, 0.3, 0.4, 0]]
```

`position_map` 是反向传播用的——它记录了 multihot 中的每个位置对应 indices 中的第几个 top-k 位，用于在 backward 中把梯度从 multihot 格式转回 indices 格式。

---

## 7. 两种专家实现对比

Automodel 提供两种专家实现，选择取决于 `backend.enable_deepep` 配置：

```python
# nemo_automodel/components/moe/layers.py:1009-1020
if backend.enable_deepep and get_world_size_safe() == 1:
    self.experts = GroupedExperts(config)       # 单 GPU 回退
elif backend.enable_deepep:
    self.experts = GroupedExpertsDeepEP(config) # DeepEP 后端
else:
    self.experts = GroupedExperts(config)        # 标准后端
```

### 7.1 GroupedExperts：All-Gather / Reduce-Scatter

> 源码路径：`nemo_automodel/components/moe/layers.py:238-399`
> 注：以下代码为简化后的关键逻辑摘要，省略了辅助变量、异常处理和边界检查，完整实现请参阅源码。

```python
class GroupedExperts(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        # 参数 shape: [n_experts, dim, 2*inter_dim] (gated) 或 [n_experts, dim, inter_dim]
        up_proj_dim = config.moe_inter_dim * 2 if self.is_gated else config.moe_inter_dim
        self.gate_and_up_projs = nn.Parameter(
            torch.empty(config.n_routed_experts, config.dim, up_proj_dim, dtype=config.dtype))
        self.down_projs = nn.Parameter(
            torch.empty(config.n_routed_experts, config.moe_inter_dim, config.dim, dtype=config.dtype))

    def forward(self, x, token_mask, weights, indices):
        # 获取 EP 信息
        if isinstance(self.gate_and_up_projs, DTensor):
            ep_mesh = self.gate_and_up_projs.device_mesh
            ep_size = ep_mesh.size()
            ep_rank = ep_mesh.get_local_rank()
        else:
            ep_size, ep_rank = 1, 0

        # EP 模式下：先 all-gather 所有 token
        if ep_size > 1:
            x = DTensor.from_local(x, device_mesh=ep_mesh, placements=[Shard(0)]).full_tensor()
            weights = DTensor.from_local(weights.float(), device_mesh=ep_mesh, placements=[Shard(0)]).full_tensor()
            indices = DTensor.from_local(indices, device_mesh=ep_mesh, placements=[Shard(0)]).full_tensor()

        # 逐专家计算
        n_local_experts = self.n_routed_experts // ep_size
        for i in range(experts_start_idx, experts_end_idx):
            indices_mask = torch.logical_and(indices == i, token_mask.unsqueeze(-1))
            idx, top = torch.where(indices_mask)
            if idx.numel() == 0:
                continue

            gate_and_up_proj = get_local_proj(self.gate_and_up_projs, i)
            down_proj = get_local_proj(self.down_projs, i)
            expert_out = self.expert_activation(x_idx, gate_and_up_proj=gate_and_up_proj,
                                                 down_proj=down_proj) * weights[idx, top, None]
            y.scatter_add_(dim=0, index=idx_b, src=expert_out.to(x.dtype))

        # EP 模式下：reduce-scatter 结果
        if ep_size > 1:
            y = DTensor.from_local(y, device_mesh=ep_mesh, placements=[Partial()])
            y = y.redistribute(placements=[Shard(0)]).to_local()
        return y
```

通信模式：
```
All-Gather tokens → 每个 rank 有完整 token → 计算本地专家 → Reduce-Scatter 结果
```

这是"经典"的 EP 实现，使用 DTensor 的 `full_tensor()`（all-gather）和 `Partial → Shard`（reduce-scatter）来实现通信。

### 7.2 GroupedExpertsDeepEP：Fused All-to-All + grouped_gemm

> 源码路径：`nemo_automodel/components/moe/layers.py:450-617`

```python
class GroupedExpertsDeepEP(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        # 参数 shape 相同，但不指定 dtype（由 DeepEP 管理）
        self.gate_and_up_projs = nn.Parameter(
            torch.empty(config.n_routed_experts, config.dim, up_proj_dim))
        self.down_projs = nn.Parameter(
            torch.empty(config.n_routed_experts, config.moe_inter_dim, config.dim))

    def forward(self, x, token_mask, weights, indices):
        # 1. 屏蔽无效 token
        indices = indices.masked_fill(~token_mask.unsqueeze(-1), -1)

        # 2. Token permutation（融合了 all-to-all + permute）
        (permuted_local_hidden_states, tokens_per_expert,
         permuted_probs) = self.token_dispatcher.token_permutation2(
            hidden_states=x, num_local_tokens=x.size(0),
            token_probs=weights, token_indices=indices)

        # 3. Grouped GEMM 计算
        gate_and_up_projs = self.gate_and_up_projs.to_local()
        down_projs = self.down_projs.to_local()

        if torch.count_nonzero(tokens_per_expert) > 0:
            output1 = ops.gmm(permuted_local_hidden_states, gate_and_up_projs,
                              tokens_per_expert, trans_b=False)
            output1 = self.expert_activation(output1, permuted_probs)
            output2 = ops.gmm(output1, down_projs, tokens_per_expert, trans_b=False)
        else:
            # 空专家的 dummy 计算（保证梯度流）
            output1 = torch.matmul(x[0] * 0, gate_and_up_projs[0])
            output1_ = self.expert_activation(output1, permuted_probs)
            output2 = torch.matmul(output1_, down_projs[0])

        # 4. Token unpermutation（融合了 permute + all-to-all）
        y = self.token_dispatcher.token_unpermutation(output2)
        return y
```

通信模式：
```
Fused Dispatch (permute + all-to-all) → grouped_gemm → Fused Combine (all-to-all + unpermute)
```

### 7.3 对比总结

```
┌───────────────────┬─────────────────────────┬───────────────────────────────┐
│                   │ GroupedExperts           │ GroupedExpertsDeepEP          │
├───────────────────┼─────────────────────────┼───────────────────────────────┤
│ 通信方式          │ DTensor all-gather +    │ DeepEP fused dispatch/combine │
│                   │ reduce-scatter          │ (RDMA 直驱)                   │
├───────────────────┼─────────────────────────┼───────────────────────────────┤
│ 计算方式          │ 逐专家 for 循环 +       │ grouped_gemm (ops.gmm)        │
│                   │ torch.compile 的 swiglu │ 批量矩阵乘                    │
├───────────────────┼─────────────────────────┼───────────────────────────────┤
│ Token 路由        │ DTensor 的 Shard/Partial│ MoEFlexTokenDispatcher +      │
│                   │ 语义处理                │ DeepEP 融合 kernel            │
├───────────────────┼─────────────────────────┼───────────────────────────────┤
│ 参数存储          │ 指定 dtype (config.dtype)│ 不指定 dtype (运行时确定)     │
├───────────────────┼─────────────────────────┼───────────────────────────���───┤
│ 空专家处理        │ dummy 计算 (保证梯度流) │ dummy 计算 (保证梯度流)       │
├───────────────────┼─────────────────────────┼───────────────────────────────┤
│ 性能特点          │ 通信与计算分离          │ 通信与排列融合，减少 kernel   │
│                   │                         │ launch 开销                    │
├───────────────────┼─────────────────────────┼───────────────────────────────┤
│ 适用场景          │ 调试/小规模/单机        │ 大规模多机训练                │
└───────────────────┴─────────────────────────┴───────────────────────────────┘
```

---

## 8. 梯度同步与 Pipeline 并行适配

### 8.1 MoEFSDPSyncMixin

> 源码路径：`nemo_automodel/components/moe/fsdp_mixin.py:95-156`

MoE 模型在梯度积累时需要特殊处理：非最后一步的 backward 不需要做梯度同步和 reshard，只有最后一步才需要。

```python
class MoEFSDPSyncMixin:
    """
    控制 FSDP 的梯度同步和 resharding 行为：
    - 无 PP：prepare_for_grad_accumulation() 在积累开始时禁用同步/reshard
              prepare_for_final_backward() 在最后一步重新启用
    - 有 PP：由 patched_backward_maybe_with_nosync 管理
    """

    def prepare_for_grad_accumulation(self, pp_enabled: bool = False):
        if not self.backend.enable_fsdp_optimizations:
            return
        for fsdp_module in _iter_fsdp_modules(self):
            _configure_fsdp_module(
                fsdp_module,
                is_last_backward=False,
                reshard_after_backward=False,       # 不 reshard → 省通信
                requires_gradient_sync=False,        # 不同步 → 省 reduce
            )

    def prepare_for_final_backward(self, pp_enabled: bool = False):
        if not self.backend.enable_fsdp_optimizations:
            return
        for fsdp_module in _iter_fsdp_modules(self):
            _configure_fsdp_module(
                fsdp_module,
                is_last_backward=True,
                reshard_after_backward=True,         # 最后一步要 reshard
                requires_gradient_sync=True,          # 最后一步要同步
            )
```

### 8.2 _iter_fsdp_modules 遍历逻辑

> 源码路径：`nemo_automodel/components/moe/fsdp_mixin.py:45-73`

```python
def _iter_fsdp_modules(module: torch.nn.Module) -> Iterator[FSDPModule]:
    _model = module.model if hasattr(module, "model") else module
    if isinstance(_model, FSDPModule):
        yield _model

    if hasattr(_model, "embed_tokens") and isinstance(_model.embed_tokens, FSDPModule):
        yield _model.embed_tokens

    if hasattr(module, "lm_head") and isinstance(module.lm_head, FSDPModule):
        yield module.lm_head

    # 遍历每层的 experts（如果它们被独立 FSDP wrap 了）
    if hasattr(_model, "layers"):
        for _, block in _model.layers.named_children():
            if hasattr(block, "mlp") and hasattr(block.mlp, "experts"):
                experts = block.mlp.experts
                if isinstance(experts, FSDPModule):
                    yield experts
```

这个函数遍历所有需要控制 FSDP 状态的模块——注意它同时覆盖了普通模块（`_model`, `embed_tokens`, `lm_head`）和专家模块（`block.mlp.experts`），确保梯度同步的控制是全局一致的。

### 8.3 Pipeline 并行适配：patched_backward_maybe_with_nosync

> 源码路径：`nemo_automodel/components/moe/fsdp_mixin.py:194-287`

当使用 Pipeline 并行（PP）时，PyTorch 的 `_PipelineStageBase` 需要知道如何处理 FSDP 的梯度同步。Automodel 通过 monkey-patch 实现：

```python
def patched_backward_maybe_with_nosync(
    self, backward_type, bwd_kwargs: dict, last_backward: bool = False,
) -> tuple[...]:
    # 标准 FSDP 模块
    if isinstance(self.submod, FSDPModule):
        # 先禁用所有同步
        self.submod.set_is_last_backward(False)
        self.submod.set_reshard_after_backward(False)
        self.submod.set_requires_gradient_sync(False)

        result = perform_backward(backward_type)()

        if last_backward:
            # 最后一步：手动触发 post_backward
            def run_post_backward(fsdp_module):
                fsdp_module.set_is_last_backward(True)
                fsdp_module.set_reshard_after_backward(True)
                fsdp_module.set_requires_gradient_sync(True)
                fsdp_state = fully_shard.state(fsdp_module)
                for state in fsdp_state._state_ctx.all_states:
                    if state._fsdp_param_group:
                        state._fsdp_param_group.post_backward()
                fsdp_state._root_post_backward_final_callback()

            run_post_backward(self.submod)

    # MoE ���块（使用 MoEFSDPSyncMixin）
    elif isinstance(self.submod, MoEFSDPSyncMixin):
        _disable_fsdp_for_moe_module(self.submod)
        result = perform_backward(backward_type)()
        if last_backward and get_is_optim_step():
            _run_post_backward_for_moe_module(self.submod)
```

关键点：
- **MoE 的额外条件**：`last_backward and get_is_optim_step()`。这是因为在 PP 中，每个 micro-batch 的最后一个 backward 不一定是真正的"最后"——只有当这个 backward 之后紧接着 optimizer step 时，才需要做梯度同步
- **IS_OPTIM_STEP 全局标志**：由训练循环设置（`set_is_optim_step(True/False)`），用来告诉 FSDP 模块何时触发梯度同步

### 8.4 梯度积累优化的整体流程

```
假设 gradient_accumulation_steps = 4:

Step 1: prepare_for_grad_accumulation()  →  禁用 sync/reshard
Step 1: backward()                       →  梯度积累，不通信
Step 2: backward()                       →  梯度积累，不通信
Step 3: backward()                       →  梯度积累，不通信
Step 4: prepare_for_final_backward()     →  启用 sync/reshard
Step 4: backward()                       →  梯度同步 + reshard
Step 4: optimizer.step()
```

---

## 9. 检查点与 State Dict 管理

Automodel 的专家参数在内部使用"合并格式"（`gate_and_up_projs`, `down_projs`），但 HuggingFace 模型使用"分离格式"（`experts.{i}.gate_proj.weight`, `experts.{i}.up_proj.weight`, `experts.{i}.down_proj.weight`）。检查点管理需要在两种格式间转换。

### 9.1 MoESplitExpertsStateDictMixin

> 源码路径：`nemo_automodel/components/moe/state_dict_mixin.py:31-432`

这个 Mixin 提供了完整的双向转换能力：

**Native → HuggingFace（保存时）**：`_to_hf_w_split_experts`

```python
def _convert_single_merged_expert_to_hf_split_experts(self, fqn, tensor):
    n_experts = self.moe_config.n_routed_experts
    inter_dim = self.moe_config.moe_inter_dim

    if f".{expert_segment}.gate_and_up_projs" in fqn:
        layer_num = re.search(r"layers\.(\d+)", fqn).group(1)
        splits = self._split_experts_weights(tensor, n_experts)
        result = []
        for i, w in enumerate(splits):
            expert_id = self._last_expert_ids[i]
            if self._is_gated_moe:
                # [dim, 2*inter_dim] → gate_proj [inter_dim, dim] + up_proj [inter_dim, dim]
                w_gate = w[:, :inter_dim].transpose(0, 1).contiguous()
                w_up = w[:, inter_dim:].transpose(0, 1).contiguous()
                result.append((f"...{expert_id}.gate_proj.weight", w_gate))
                result.append((f"...{expert_id}.up_proj.weight", w_up))
            else:
                # ReLU²: [dim, inter_dim] → up_proj [inter_dim, dim]
                w_up = w.transpose(0, 1).contiguous()
                result.append((f"...{expert_id}.up_proj.weight", w_up))
        return result

    elif f".{expert_segment}.down_projs" in fqn:
        splits = self._split_experts_weights(tensor, n_experts)
        result = []
        for i, w in enumerate(splits):
            expert_id = self._last_expert_ids[i]
            # [inter_dim, dim] → [dim, inter_dim]
            result.append((f"...{expert_id}.down_proj.weight", w.transpose(0, 1).contiguous()))
        return result
```

**HuggingFace → Native（加载时）**：`_from_hf_w_merged_experts`

```python
def _from_hf_w_merged_experts(self, hf_state_dict, device_mesh=None):
    n_experts = self.moe_config.n_routed_experts

    # 只加载本 rank 的专家
    if device_mesh is not None:
        start_expert, end_expert = get_expert_range_for_rank_from_mesh(device_mesh, n_experts)
        expected_experts_per_rank = end_expert - start_expert

    for key, value in hf_state_dict.items():
        m = expert_pattern.match(key)
        expert_num = int(m.group(3))

        # 跳过不属于本 rank 的专家
        if not should_load_expert_for_rank(expert_num, device_mesh, n_experts):
            continue

        if which in ["gate_proj", "up_proj"]:
            # 收集 gate 和 up 的权重
            if is_gated:
                expert_weights[expert_num]["gate_proj"] = value
                expert_weights[expert_num]["up_proj"] = value
            # 当所有专家齐全时，合并
            if all_complete:
                for expert_id in sorted_ids:
                    gate_t = gate_weight.transpose(0, 1)  # [inter_dim, dim] → [dim, inter_dim]
                    up_t = up_weight.transpose(0, 1)
                    tensors.append(torch.cat([gate_t, up_t], dim=-1))  # [dim, 2*inter_dim]
                stacked = torch.stack(tensors, dim=0)  # [n_local_experts, dim, 2*inter_dim]
                state_dict[native_key] = create_dtensor_from_local(stacked, device_mesh)
```

### 9.2 DTensor 感知的专家切分

> 源码路径：`nemo_automodel/components/moe/state_dict_utils.py:92-164`

保存检查点时，专家参数可能是 DTensor。`split_experts_weights_dtensor_aware` 正确处理了这种情况：

```python
def split_experts_weights_dtensor_aware(weight, n_experts):
    local_tensor, start_expert, end_expert = get_expert_slice_for_rank(weight, n_experts)
    local_n_experts = end_expert - start_expert

    if is_dtensor(weight):
        device_mesh = weight.device_mesh
        original_placements = weight.placements
        mesh_dim_names = list(weight.device_mesh.mesh_dim_names)

        # 移除 'ep' 维度
        ep_dim_idx = mesh_dim_names.index("ep")
        remaining_mesh_dims = mesh_dim_names[:ep_dim_idx] + mesh_dim_names[ep_dim_idx + 1:]

        # 构建不含 'ep' 的新 device mesh
        if remaining_mesh_dims and any_non_trivial_dim:
            new_device_mesh = get_submesh(device_mesh, tuple(remaining_mesh_dims))
        else:
            new_device_mesh = None

        # 调整 placement：移除 ep 对应的 Shard(0)
        new_placements_template = original_placements[:ep_dim_idx] + original_placements[ep_dim_idx + 1:]

    for i in range(local_n_experts):
        expert_weight = local_tensor[i]  # 移除 expert 维度
        if is_weight_dtensor:
            # 调整 Shard 维度号（因为移除了 dim 0）
            new_placements = []
            for placement in new_placements_template:
                if isinstance(placement, Shard) and placement.dim > 0:
                    new_placements.append(Shard(placement.dim - 1))
                else:
                    new_placements.append(placement)
            expert_weight = DTensor.from_local(expert_weight, new_device_mesh, new_placements)
        split_weights.append(expert_weight)
    return split_weights, expert_ids
```

这段代码做了一件很精细的事：当专家权重是一个多维 DTensor（例如在 `(ep, ep_shard)` mesh 上分布）时，拆分出单个专家后，需要：
1. 从 device_mesh 中移除 `ep` 维度（因为单个专家不再需要 EP 切分）
2. 调整所有 `Shard` placement 的维度号（因为 dim 0 被移除了，原来的 `Shard(1)` 变成 `Shard(0)`）
3. 在剩余的 mesh 上重新构造 DTensor

### 9.3 create_dtensor_from_local

> 源码路径：`nemo_automodel/components/moe/state_dict_utils.py:202-253`

加载检查点时，需要把本地 Tensor 转成 DTensor 放到 moe_mesh 上：

```python
def create_dtensor_from_local(local_tensor, device_mesh, rank=None):
    if device_mesh is None:
        return local_tensor

    placements = []
    for dim_name in mesh_dim_names:
        if dim_name == "ep":
            placements.append(Shard(0))        # EP 维度：按 dim 0 切
        elif dim_name == "ep_shard":
            placements.append(Shard(1))        # ep_shard 维度：按 dim 1 切
        elif dim_name == "ep_replicate":
            placements.append(Replicate())     # 复制维度

    dtensor = DTensor.from_local(local_tensor, get_submesh(device_mesh, tuple(dim_names)), placements)
    return dtensor
```

这确保了加载的专家权重在 moe_mesh 上有正确的分布语义：
- `ep` 维度上 `Shard(0)`：每个 EP rank 持有不同的专家子集
- `ep_shard` 维度上 `Shard(1)`：在 EP group 内，进一步沿 hidden 维度 FSDP 切分
- `ep_replicate` 维度上 `Replicate()`：在复制组内保持一致

---

## 总结

Automodel 的 EP+FSDP 实现可以用一张图概括：

```
                        FSDP2Manager
                       /            \
              world_mesh            moe_mesh
         (pp,dp_rep,dp_shard,      (pp,ep_shard,ep)
              cp,tp)
                |                       |
                |                       |
         parallelize_model ─────────────┘
          CP → EP → AC → FSDP
                     |
              ┌──────┴──────┐
              │             │
         apply_ep      apply_fsdp
              │             │
    ExpertParallel    三重逻辑：
    DTensor Shard(0)  1. experts: Shard(1) on ep_shard_mesh
    init_dispatcher   2. ignored_params 排除专家
                      3. block: fully_shard on fsdp_mesh
              │
    ┌─────────┴─────────┐
    │                   │
GroupedExperts    GroupedExpertsDeepEP
(all-gather/       (DeepEP fused a2a
 reduce-scatter)    + grouped_gemm)
                        │
              MoEFlexTokenDispatcher
                        │
                  _DeepepManager
                   (共享单例)
                        │
              ┌─────────┴─────────┐
              │                   │
        FusedDispatch       FusedCombine
        (autograd.Function)
              │                   │
         Buffer.dispatch    Buffer.combine
         (DeepEP RDMA)     (DeepEP RDMA)
```

核心设计决策：
1. **双 DeviceMesh** 解耦了普通参数和专家参数的分布语义
2. **Shard(1) FSDP** 在 EP 已经切了 dim=0 的前提下，沿 dim=1 继续切分
3. **ignored_params** 防止 block 级 FSDP 和专家级 FSDP 冲突
4. **DeepEP 融合通信** 把 permute + all-to-all 合并成一个 kernel，减少开销
5. **共享 _DeepepManager** 跨层复用通信 buffer，节省内存
6. **DTensor 感知的检查点** 在 EP 切分状态下正确序列化和加载专家权重
