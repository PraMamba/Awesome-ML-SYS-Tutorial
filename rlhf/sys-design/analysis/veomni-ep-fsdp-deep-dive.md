# VeOmni EP+FSDP 深度源码分析

> 本文基于 VeOmni（ByteDance Seed team）的 `source_code_analysis` 分支，对 Expert Parallelism（EP）与 FSDP 的协同实现进行逐行级别的源码分析。涵盖 Qwen3-MoE 模型适配、Device Mesh 设计、ParallelPlan EP 切分声明与执行、FSDP1/FSDP2 双路径集成、MoE 前向传播全链路、All-to-All 通信原语、GroupGemm 融合计算、反向传播梯度同步、Prefetch 策略、Checkpoint 管理以及端到端部署示例。

---

## 目录

- [第 1 节：概述与定位](#第-1-节概述与定位)
- [第 2 节：模型架构 -- Qwen3-MoE](#第-2-节模型架构----qwen3-moe)
- [第 3 节：Device Mesh 设计](#第-3-节device-mesh-设计)
- [第 4 节：ParallelPlan -- EP 切分声明与执行](#第-4-节parallelplan----ep-切分声明与执行)
- [第 5 节：FSDP1 路径 -- EP + FSDP1 集成](#第-5-节fsdp1-路径----ep--fsdp1-集成)
- [第 6 节：FSDP2 路径 -- EP + FSDP2 集成](#第-6-节fsdp2-路径----ep--fsdp2-集成)
- [第 7 节：MoE 前向传播 -- EP 通信全流程](#第-7-节moe-前向传播----ep-通信全流程)
- [第 8 节：All-to-All 通信原语](#第-8-节all-to-all-通信原语)
- [第 9 节：GroupGemm 融合计算](#第-9-节groupgemm-融合计算)
- [第 10 节：反向传播与梯度同步](#第-10-节反向传播与梯度同步)
- [第 11 节：Prefetch 策略](#第-11-节prefetch-策略)
- [第 12 节：Checkpoint 管理](#第-12-节checkpoint-管理)
- [第 13 节：端到端示例 -- Qwen3-MoE-30B on 16 GPUs](#第-13-节端到端示例----qwen3-moe-30b-on-16-gpus)
- [第 14 节：总结与框架对比](#第-14-节总结与框架对比)

---

## 第 1 节：概述与定位

> **一句话总结**：VeOmni 采用 monkey-patch + ParallelPlan 的方式，在不修改 HuggingFace 模型定义的前提下注入 EP 能力，通过独立的 EP mesh 与 FSDP mesh 协同工作，同时支持 FSDP1 和 FSDP2 两条路径。

### 1.1 VeOmni 框架简介

VeOmni 是 ByteDance Seed 团队开发的多模态训练框架，支持多模态（Vision-Language）和 MoE 模型的分布式训练。与 TorchTitan 等框架不同，VeOmni 的核心设计理念是 **复用 HuggingFace 生态的模型定义**，通过 monkey-patch 机制注入分布式并行能力，而非从零编写自定义模型。

### 1.2 核心挑战：EP + FSDP 如何协同

MoE 模型中同时存在两类参数，它们需要在不同的维度上切分：

1. **普通参数**（attention、embedding 等）：在 FSDP mesh 上做数据并行切分
2. **专家参数**（gate_proj、up_proj、down_proj）：先在 EP mesh 的 dim 0（专家维度）上切分，再在 ep_fsdp mesh 上做数据并行切分

两套 mesh 覆盖的 GPU 集合相同，但维度划分方式不同。核心难点在于：

- 专家模块必须从主 FSDP 中排除（`ignored_states`），然后单独用不同的 mesh 再包一层 FSDP
- 梯度归约的除法因子需要手动修正，因为专家参数的 FSDP group 与普通参数不同
- checkpoint 保存/加载时，专家参数需要 EP-aware 的切片逻辑

### 1.3 与 TorchTitan 的定位差异

| 方面 | VeOmni | TorchTitan |
|------|--------|------------|
| 模型来源 | 复用 HuggingFace 模型，monkey-patch 注入 | 自定义模型（`model.py`） |
| EP 策略 | 统一的 `ParallelPlan` 声明式接口 | 四种 `ParallelStyle`（TP/EP/ETP/DeepEP） |
| Token dispatch | 不涉及（模型内部处理） | 自实现 all-to-all + Triton permute kernel |
| FSDP 版本 | 同时支持 FSDP1 和 FSDP2 | 仅 FSDP2 |
| EP mesh | 独立的 2D `(ep, ep_fsdp)` mesh | 从三套全局 mesh 的 sparse mesh 中提取 |
| 适配新模型 | 编写 ~16 行 `parallel_plan.py` + monkey-patch | 需要在自定义模型中实现完整的 MoE 类 |

### 1.4 源码文件结构

```
veomni/
├── distributed/
│   ├── torch_parallelize.py        # 核心入口：build_parallelize_model()
│   │                                #   ├── parallelize_model_fsdp1()  (L84-234)
│   │                                #   └── parallelize_model_fsdp2()  (L237-435)
│   ├── parallel_plan.py             # ParallelPlan：EP 切分声明与执行 (L44-170)
│   ├── parallel_state.py            # ParallelState + DeviceMesh 构建 (L78-580)
│   ├── utils.py                     # check_fqn_match, set/get_module_from_path
│   ├── fsdp/
│   │   ├── clip_grad_norm.py        # EP-aware 梯度裁剪
│   │   └── ...                      # checkpoint extension, parallel_init 等
│   └── fsdp2/
│       └── clip_grad_norm.py        # FSDP2 梯度裁剪
│
└── models/transformers/qwen3_moe/
    ├── modeling_qwen3_moe.py        # Monkey-patch: PatchQwen3MoeExperts 等 (L67-364)
    └── parallel_plan.py             # 模型特定的 EP plan (16 行)
```

### 1.5 入口函数 `build_parallelize_model` 概览

`build_parallelize_model()`（`torch_parallelize.py:438-523`）是整个并行化的总入口，按以下顺序执行：

```python
# torch_parallelize.py:438-523
def build_parallelize_model(model, weights_path=None, ...):
    # 0. 获取 parallel_state
    parallel_state = get_parallel_state()

    # 1. 混合精度：upcast to float32
    if enable_mixed_precision:
        model = model.float()

    # 2. Gradient Checkpointing
    if enable_gradient_checkpointing:
        model.gradient_checkpointing_enable(...)

    # 3. Tensor Parallelism（如果启用）
    if parallel_state.tp_enabled:
        model = parallelize_module(model, device_mesh=parallel_state.tp_mesh)

    # 4. FSDP（包含 EP 的处理）
    if parallel_state.fsdp_enabled:
        if parallel_state.dp_mode == "fsdp2":
            model = parallelize_model_fsdp2(model, ...)    # FSDP2 路径
        elif parallel_state.dp_mode == "fsdp1":
            model = parallelize_model_fsdp1(model, ...)    # FSDP1 路径
        else:
            model = DDP(model, ...)                         # DDP fallback

    return model
```

执行流程概览：

```
build_parallelize_model()
├── 1. model.float()                    # 混合精度预处理
├── 2. gradient_checkpointing_enable()  # 激活检查点
├── 3. parallelize_module() [TP]        # Tensor Parallelism（可选）
└── 4. parallelize_model_fsdp1/fsdp2()  # FSDP + EP
       ├── parallel_plan.apply()        #   4a. EP 切分专家参数
       ├── get_fsdp_no_shard_info()     #   4b. 获取专家模块 FQN
       ├── FSDP(model, ignored=experts) #   4c. 主 FSDP（排除专家）
       ├── FSDP(experts, ep_fsdp_mesh)  #   4d. 专家 FSDP（独立 mesh）
       ├── register_checkpoint_ext()    #   4e. 注册 checkpoint hook
       └── clip_grad_norm_ patch        #   4f. EP-aware 梯度裁剪
```

> **NOTE**：与 TorchTitan 不同，VeOmni 的 EP 处理不是独立的步骤，而是嵌入在 FSDP 包装流程内部。TP 在 FSDP 之前执行（如果启用），但 EP 的切分和 FSDP 的包装是交织在同一个函数中完成的。

---

## 第 2 节：模型架构 -- Qwen3-MoE

> **一句话总结**：VeOmni 通过 monkey-patch 将 HuggingFace 的 `Qwen3MoeSparseMoeBlock` 替换为自定义的 `PatchQwen3MoeSparseMoeBlock`，使用 3D 参数张量 `[num_experts, dim, dim]` 存储所有专家权重，并通过 16 行的 `parallel_plan.py` 声明 EP 切分维度。

### 2.1 PatchQwen3MoeExperts -- 三维权重张量

`PatchQwen3MoeExperts`（`modeling_qwen3_moe.py:67-124`）是 EP 的核心切分对象。它将所有专家的权重合并为三维参数张量，而非为每个专家创建独立的 `nn.Linear`：

```python
# modeling_qwen3_moe.py:67-78
class PatchQwen3MoeExperts(nn.Module):
    """Collection of expert weights stored as 3D tensors."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        # 三维参数：[num_experts, intermediate_dim, hidden_dim]
        self.gate_proj = nn.Parameter(torch.empty(self.num_experts, self.intermediate_dim, self.hidden_dim))
        self.up_proj   = nn.Parameter(torch.empty(self.num_experts, self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]
        self._moe_implementation = getattr(config, "_moe_implementation", "eager")
```

其中：
- **dim 0**：专家维度 `num_experts`，EP 在此维度切分
- **dim 1/2**：模型维度（intermediate_dim / hidden_dim），FSDP2 路径可能在 dim 1 上做 FSDP 切分

> **NOTE**：这种 3D 张量设计与 TorchTitan 的 `GroupedExperts` 完全一致。两者都将所有专家权重打包为 `(E, H, D)` 形状的单个参数，区别在于 TorchTitan 使用 DTensor 进行切分，而 VeOmni 使用 `DTensor.from_local → redistribute → to_local` 的显式流程。

### 2.2 Eager vs Fused 前向分支

`PatchQwen3MoeExperts.forward()`（`modeling_qwen3_moe.py:81-124`）支持两种计算模式：

```python
# modeling_qwen3_moe.py:81-124
def forward(self, hidden_states, top_k_index, top_k_weights):
    final_hidden_states = torch.zeros_like(hidden_states)

    if self._moe_implementation == "eager":
        # 逐专家循环计算
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            # 使用 F.linear 对单个专家做计算
            gate = F.linear(current_state, self.gate_proj[expert_idx])
            up   = F.linear(current_state, self.up_proj[expert_idx])
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

    elif self._moe_implementation == "fused":
        # 调用 fused_moe_forward（Triton 实现）
        top_k_weights = top_k_weights.to(final_hidden_states.dtype)
        final_hidden_states = fused_moe_forward(
            module=self, num_experts=self.num_experts,
            routing_weights=top_k_weights, selected_experts=top_k_index,
            hidden_states=hidden_states,
            fc1_1_weight=self.gate_proj, fc1_2_weight=self.up_proj, fc2_weight=self.down_proj,
        )
    return final_hidden_states
```

**两种模式的对比**：

| 方面 | eager | fused |
|------|-------|-------|
| 计算方式 | 逐专家循环 `F.linear` | 一次 Triton kernel 批量计算 |
| 性能 | 较慢（Python 循环开销） | 较快（GPU kernel fusion） |
| 适用场景 | 调试、小规模 | 生产训练 |
| EP 兼容性 | 均兼容（EP 在 dim 0 切分后，`self.gate_proj[expert_idx]` 自动索引本地专家） |

### 2.3 PatchQwen3MoeTopKRouter -- 路由门控

`PatchQwen3MoeTopKRouter`（`modeling_qwen3_moe.py:127-145`）实现了 softmax + top-k 路由：

```python
# modeling_qwen3_moe.py:127-145
class PatchQwen3MoeTopKRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)        # (seq_len, num_experts)
        router_logits = F.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
        if self.norm_topk_prob:
            router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(router_logits.dtype)
        return router_logits, router_top_value, router_indices
```

> **NOTE**：与 TorchTitan 的 `TokenChoiceTopKRouter` 不同，VeOmni 的 router 使用 softmax 评分（而非 sigmoid），且不包含 auxiliary-loss-free 的 `expert_bias` 机制。VeOmni 的负载均衡依赖 HuggingFace 模型原生的 `load_balancing_loss`。

### 2.4 PatchQwen3MoeSparseMoeBlock -- 组合专家与路由

```python
# modeling_qwen3_moe.py:148-167
class PatchQwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.experts = PatchQwen3MoeExperts(config)
        self.gate = PatchQwen3MoeTopKRouter(config)
        # 初始化权重
        _init_weight(self.experts.gate_proj)
        _init_weight(self.experts.up_proj)
        _init_weight(self.experts.down_proj)
        _init_weight(self.gate.weight)

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        router_logits, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        final_hidden_states = self.experts(hidden_states_reshaped, selected_experts, routing_weights)
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim), router_logits
```

模块层次结构：

```
PatchQwen3MoeSparseMoeBlock
├── gate: PatchQwen3MoeTopKRouter
│   └── weight: Parameter(num_experts, hidden_dim)       # 路由权重
└── experts: PatchQwen3MoeExperts
    ├── gate_proj: Parameter(num_experts, intermediate_dim, hidden_dim)  # EP dim=0
    ├── up_proj:   Parameter(num_experts, intermediate_dim, hidden_dim)  # EP dim=0
    └── down_proj: Parameter(num_experts, hidden_dim, intermediate_dim)  # EP dim=0
```

### 2.5 `get_parallel_plan()` -- 模型特定的 EP 声明

这是 VeOmni 适配新 MoE 模型的关键接口。对于 Qwen3-MoE，整个文件只有 16 行：

```python
# models/transformers/qwen3_moe/parallel_plan.py (完整文件)
from torch.distributed._tensor import Shard
from ....distributed.parallel_plan import ParallelPlan

def get_parallel_plan():
    ep_plan = {
        "model.layers.*.mlp.experts.gate_proj": Shard(0),
        "model.layers.*.mlp.experts.up_proj":   Shard(0),
        "model.layers.*.mlp.experts.down_proj":  Shard(0),
    }
    parallel_plan = ParallelPlan(ep_plan=ep_plan)
    return parallel_plan
```

**设计要点**：

- `ep_plan` 是一个字典，key 是参数的 FQN（Fully Qualified Name）模式，value 是 `Shard` placement
- `*` 是通配符，匹配任意层号（如 `model.layers.0.mlp.experts.gate_proj`、`model.layers.1.mlp.experts.gate_proj`...）
- `Shard(0)` 表示在 dim 0（专家维度）上切分
- `ParallelPlan` 会自动从 `ep_plan` 的 key 中提取：
  - `ep_param_suffix`：`{"gate_proj", "up_proj", "down_proj"}`
  - `fsdp_no_shard_module`：`{"model.layers.*.mlp.experts"}`（去掉最后一个 `.xxx` 后的前缀）

### 2.6 Monkey-Patching 机制

`apply_veomni_qwen3_moe_patch()`（`modeling_qwen3_moe.py:341-363`）通过替换 HuggingFace 模块的类属性实现注入：

```python
# modeling_qwen3_moe.py:341-363
def apply_veomni_qwen3_moe_patch():
    logger.info_rank0("Apply VeOmni patch to qwen3_moe.")

    # 替换 SparseMoeBlock 类 → 使用 3D 参数张量版本
    hf_qwen3_moe.Qwen3MoeSparseMoeBlock = PatchQwen3MoeSparseMoeBlock

    # 注入 get_parallel_plan() 方法
    from .parallel_plan import get_parallel_plan
    hf_qwen3_moe.Qwen3MoeForCausalLM.get_parallel_plan = lambda self: get_parallel_plan()

    # 替换 forward 方法（支持 SP 和 fused cross_entropy）
    hf_qwen3_moe.Qwen3MoeModel.forward = qwen3_moe_model_forward
    hf_qwen3_moe.Qwen3MoeForCausalLM.forward = qwen3_moe_forcausal_lm_forward

    # 平台特定 patch
    if IS_CUDA_AVAILABLE:
        from .gpu_patch import apply_veomni_qwen3_moe_gpu_patch
        apply_veomni_qwen3_moe_gpu_patch()
    elif IS_NPU_AVAILABLE:
        from .npu_patch import apply_qwen3_moe_npu_patch
        apply_qwen3_moe_npu_patch()
```

**关键行为**：

1. `hf_qwen3_moe.Qwen3MoeSparseMoeBlock = PatchQwen3MoeSparseMoeBlock`：当 HuggingFace 的 `from_pretrained()` 或 `from_config()` 创建模型时，会自动使用替换后的类
2. `Qwen3MoeForCausalLM.get_parallel_plan = lambda self: get_parallel_plan()`：给模型实例注入 `get_parallel_plan()` 方法，供 `parallelize_model_fsdp1/fsdp2` 调用

> **NOTE**：这种 monkey-patch 设计意味着适配一个新的 MoE 模型只需要：(1) 实现类似的 `PatchXxxExperts` 类，将独立专家权重合并为 3D 张量；(2) 编写 `parallel_plan.py` 声明哪些参数需要 EP 切分；(3) 实现 `apply_patch()` 函数完成替换。框架的 EP+FSDP 逻辑完全通用，不需要修改。

---

## 第 3 节：Device Mesh 设计

> **一句话总结**：VeOmni 构建两套 Device Mesh——主 mesh 用于 DP/FSDP/TP/SP/PP，独立的 EP mesh 用于专家并行。EP mesh 是一个 2D `(ep, ep_fsdp)` 矩阵，通过 `ep_inside`（默认，交错排列）和 `ep_outside`（连续排列）两种布局策略支持不同的硬件拓扑。

### 3.1 主 Device Mesh 构建

`init_parallel_state()`（`parallel_state.py:449-568`）负责构建全局 Device Mesh。主 mesh 的维度构成如下：

```python
# parallel_state.py:485-499
mesh_shape = []
mesh_dim_names = []
for d, name in zip(
    [pp_size, dp_replicate_size, dp_shard_size, ulysses_size, cp_size, tp_size],
    ["pp",    "dp_replicate",    "dp_shard",    "ulysses",    "cp",    "tp"],
):
    if d > 1 or name in ["dp_shard"]:   # dp_shard 始终包含
        mesh_shape.append(d)
        mesh_dim_names.append(name)

device_mesh = init_device_mesh(
    device_type=device_type,
    mesh_shape=tuple(mesh_shape),
    mesh_dim_names=tuple(mesh_dim_names),
)
```

**设计要点**：
- 只有 `size > 1` 的维度才会被包含，**除了** `dp_shard` 始终存在
- 这意味着在最简单的单机训练中，mesh 可能只有 `["dp_shard"]` 一个维度

随后，通过 `_flatten()` 创建多个复合子 mesh（`parallel_state.py:501-536`）：

```python
# parallel_state.py:502-536
# dp = dp_replicate + dp_shard
dp_mesh_dim_names = []                    # 数据加载用
dp_shard_sp_mesh_dim_names = []           # 参数切分用（FSDP + SP）
dp_sp_mesh_dim_names = []                 # loss all-reduce 用
sp_mesh_dim_names = []                    # 序列并行用

if dp_replicate_size > 1:
    dp_mesh_dim_names.append("dp_replicate")
    dp_sp_mesh_dim_names.append("dp_replicate")
if dp_shard_size >= 1:
    dp_mesh_dim_names.append("dp_shard")
    dp_shard_sp_mesh_dim_names.append("dp_shard")
    dp_sp_mesh_dim_names.append("dp_shard")
if ulysses_size > 1:
    dp_shard_sp_mesh_dim_names.append("ulysses")
    sp_mesh_dim_names.append("ulysses")
    dp_sp_mesh_dim_names.append("ulysses")
if cp_size > 1:
    dp_shard_sp_mesh_dim_names.append("cp")
    sp_mesh_dim_names.append("cp")
    dp_sp_mesh_dim_names.append("cp")

# 展平为命名子 mesh
device_mesh[tuple(dp_mesh_dim_names)]._flatten(mesh_dim_name="dp")
device_mesh[tuple(dp_shard_sp_mesh_dim_names)]._flatten(mesh_dim_name="dp_shard_sp")
device_mesh[tuple(dp_sp_mesh_dim_names)]._flatten(mesh_dim_name="dp_sp")
device_mesh[tuple(sp_mesh_dim_names)]._flatten(mesh_dim_name="sp")
```

子 mesh 的含义：

| 子 Mesh 名称 | 组成维度 | 用途 |
|-------------|---------|------|
| `dp` | dp_replicate + dp_shard | 数据加载、sampler |
| `dp_shard_sp` | dp_shard + ulysses + cp | FSDP 参数切分（含 SP） |
| `dp_sp` | dp_replicate + dp_shard + ulysses + cp | loss all-reduce |
| `sp` | ulysses + cp | 序列并行通信 |

### 3.2 EP Mesh 构建

EP mesh 的构建独立于主 mesh（`parallel_state.py:538-548`）：

```python
# parallel_state.py:538-548
if ep_size > 1:
    world_size = dist.get_world_size()
    assert world_size % ep_size == 0, "ep_size must be a factor of world_size"
    ep_fsdp_size = world_size // ep_size

    mesh = init_ep_mesh_matrix(ep_size=ep_size, ep_fsdp_size=ep_fsdp_size, ep_outside=ep_outside)
    ep_fsdp_device_mesh = DeviceMesh(
        device_type=device_type,
        mesh=mesh,
        mesh_dim_names=("ep", "ep_fsdp"),
    )
```

核心函数 `init_ep_mesh_matrix()`（`parallel_state.py:57-75`）根据 `ep_outside` 参数生成两种不同的 GPU 排列矩阵：

```python
# parallel_state.py:57-75
def init_ep_mesh_matrix(ep_size: int, ep_fsdp_size: int, ep_outside: bool = False):
    if ep_outside:
        # EP ranks 连续排列
        mesh = torch.arange(ep_size * ep_fsdp_size).view(ep_size, ep_fsdp_size)
    else:
        # EP ranks 交错排列（默认）
        mesh = torch.arange(ep_size * ep_fsdp_size).view(ep_fsdp_size, ep_size).transpose(0, 1)
    return mesh
```

### 3.3 两种 EP Mesh 布局

**数值示例：16 GPU，ep_size=4**

计算 `ep_fsdp_size = 16 / 4 = 4`

#### ep_inside（默认，`ep_outside=False`）

先 `view(ep_fsdp_size=4, ep_size=4)` 得到按行排列的矩阵，再 `transpose(0, 1)` 使 EP 维度成为行：

```
                ep_fsdp_rank
ep_rank     0     1     2     3
  0       [ 0,    4,    8,   12]
  1       [ 1,    5,    9,   13]
  2       [ 2,    6,   10,   14]
  3       [ 3,    7,   11,   15]
```

**读法**：
- **EP group**（同一行）：GPU {0,4,8,12} 持有相同编号的专家 shard → 它们之间做 ep_fsdp（FSDP 切分）
- **EP peers**（同一列）：GPU {0,1,2,3} 各持有不同的专家 → 它们组成一个 EP group
- ep_inside 的特点：同一个 EP group 的 rank 是 **相邻的**（0,1,2,3），通常对应同一个节点的 GPU，利用 NVLink 高带宽

#### ep_outside（`ep_outside=True`）

直接 `view(ep_size=4, ep_fsdp_size=4)`：

```
                ep_fsdp_rank
ep_rank     0     1     2     3
  0       [ 0,    1,    2,    3]
  1       [ 4,    5,    6,    7]
  2       [ 8,    9,   10,   11]
  3       [12,   13,   14,   15]
```

**读法**：
- **EP group**（同一行）：GPU {0,1,2,3} 持有相同 EP rank 的专家 → 它们之间做 ep_fsdp
- **EP peers**（同一列）：GPU {0,4,8,12} 各持有不同的专家 → 它们组成一个 EP group
- ep_outside 的特点：同一个 EP group 的 rank 是 **跨节点的**（0,4,8,12），适合 EP all-to-all 走 RDMA 的场景

### 3.4 Mesh 拓扑可视化

以 16 GPU / 2 节点（每节点 8 GPU）、`ep_size=4`、`ep_inside` 为例：

```
节点 0: GPU 0  1  2  3  4  5  6  7
节点 1: GPU 8  9  10 11 12 13 14 15

ep_inside mesh:
         ep_fsdp_rank=0  ep_fsdp_rank=1  ep_fsdp_rank=2  ep_fsdp_rank=3
         (节点0前半)      (节点0后半)      (节点1前半)      (节点1后半)
ep=0  :     GPU 0           GPU 4           GPU 8           GPU 12
ep=1  :     GPU 1           GPU 5           GPU 9           GPU 13
ep=2  :     GPU 2           GPU 6           GPU 10          GPU 14
ep=3  :     GPU 3           GPU 7           GPU 11          GPU 15

EP group（同一列，all-to-all 通信路径）：
  {0,1,2,3}  ← 全在节点 0，NVLink
  {4,5,6,7}  ← 全在节点 0，NVLink
  {8,9,10,11} ← 全在节点 1，NVLink
  {12,13,14,15} ← 全在节点 1，NVLink

ep_fsdp group（同一行，FSDP reduce-scatter/all-gather 路径）：
  {0,4,8,12}  ← 跨节点，需要 RDMA
  {1,5,9,13}  ← 跨节点，需要 RDMA
  ...
```

> **NOTE**：`ep_inside`（默认）将 EP all-to-all 限制在节点内（NVLink），而 ep_fsdp 的通信走跨节点。这与 TorchTitan 的 sparse mesh 中 EP 维度和 efsdp 维度的排布逻辑类似，但 VeOmni 的实现更直接——用一个独立的 2D mesh 矩阵完成，不需要从三套全局 mesh 中推导。

### 3.5 EP 相关属性

`ParallelState` 提供了多个 EP 相关的 cached property（`parallel_state.py:319-357`）：

```python
# parallel_state.py:319-357
@property
def ep_enabled(self) -> bool:
    return self.ep_size > 1

@property
def ep_rank(self) -> int:
    return self.ep_fsdp_device_mesh.get_local_rank("ep")

@property
def ep_fsdp_size(self) -> int:
    assert self.ep_enabled
    return self.fsdp_size // self.ep_size    # fsdp_size = world_size / (pp * tp)

@property
def ep_gradient_divide_factor(self) -> int:
    assert self.tp_size == 1
    assert self.pp_size == 1
    return self.world_size                   # 梯度除法因子 = world_size
```

**`ep_gradient_divide_factor` 的含义**：

对于 EP+FSDP 的专家参数，梯度需要在 **所有** 数据并行 rank 上取平均。由于 VeOmni 当前要求 `tp_size == 1` 和 `pp_size == 1`（当使用 EP 时），所以 `ep_gradient_divide_factor = world_size`。

> **NOTE**：这个断言说明 VeOmni 目前不支持 EP + TP 或 EP + PP 的组合。相比之下，TorchTitan 通过 `fsdp_gradient_divide_factor = dp_replicate * dp_shard * cp` 支持更灵活的组合。

### 3.6 FSDP Mesh 的选择逻辑

`fsdp_mesh` 属性（`parallel_state.py:252-269`）根据配置动态选择合适的 mesh：

```python
# parallel_state.py:252-269
@property
def fsdp_mesh(self) -> "DeviceMesh":
    if self.dp_replicate_enabled:
        # HSDP 模式
        if self.dp_shard_sp_enabled:
            return self.device_mesh["dp_replicate", "dp_shard_sp"]
        elif self.dp_shard_enabled:
            return self.device_mesh["dp_replicate", "dp_shard"]
        else:
            return self.device_mesh["dp_replicate"]
    # FSDP 模式
    elif self.dp_shard_sp_enabled:
        return self.device_mesh["dp_shard_sp"]
    elif self.dp_shard_enabled:
        return self.device_mesh["dp_shard"]
    else:
        return self.device_mesh["dp"]
```

**决策树**：

```
fsdp_mesh 选择:
├── dp_replicate > 1?
│   ├── YES (HSDP):
│   │   ├── SP enabled? → ["dp_replicate", "dp_shard_sp"]  (2D HSDP + SP)
│   │   ├── dp_shard?   → ["dp_replicate", "dp_shard"]     (2D HSDP)
│   │   └── else        → ["dp_replicate"]                  (DDP-like)
│   └── NO (FSDP):
│       ├── SP enabled? → ["dp_shard_sp"]                   (1D FSDP + SP)
│       ├── dp_shard?   → ["dp_shard"]                      (1D FSDP)
│       └── else        → ["dp"]                             (全局 DP)
```

---

## 第 4 节：ParallelPlan -- EP 切分声明与执行

> **一句话总结**：`ParallelPlan` 通过声明式的 `ep_plan` 字典定义哪些参数需要 EP 切分，`apply()` 方法使用 `DTensor.from_local → redistribute → to_local` 三步完成实际的参数切片，同时为每个参数附加 `SpecInfo` 元信息，供后续 checkpoint 保存/加载使用。

### 4.1 ParallelPlan 类设计

```python
# parallel_plan.py:44-48
class ParallelPlan:
    def __init__(self, ep_plan: Dict[str, Shard]):
        self.ep_plan = ep_plan
        # 从 ep_plan keys 提取最后一级名称，如 {"gate_proj", "up_proj", "down_proj"}
        self.ep_param_suffix = {k.split(".")[-1] for k in ep_plan.keys()}
        # 去掉最后一级得到父模块 FQN，如 {"model.layers.*.mlp.experts"}
        self.fsdp_no_shard_module = {".".join(list(ep_plan.keys())[0].split(".")[:-1])}
```

三个属性的关系（以 Qwen3-MoE 为例）：

```
ep_plan:
  "model.layers.*.mlp.experts.gate_proj" → Shard(0)
  "model.layers.*.mlp.experts.up_proj"   → Shard(0)
  "model.layers.*.mlp.experts.down_proj" → Shard(0)

ep_param_suffix:   {"gate_proj", "up_proj", "down_proj"}
fsdp_no_shard_module: {"model.layers.*.mlp.experts"}
```

- `ep_param_suffix`：用于快速判断一个参数名是否可能是专家参数
- `fsdp_no_shard_module`：标识需要从主 FSDP 排除的模块 FQN 模式

### 4.2 SpecInfo 数据类

```python
# parallel_plan.py:30-41
@dataclass
class SpecInfo:
    ep_fsdp_mesh: DeviceMesh      # 完整的 (ep, ep_fsdp) 2D mesh
    placement: Union[Shard, Replicate]  # 该参数的 EP placement
    fqn: str                       # 参数的完整 FQN

    @property
    def ep_mesh(self):
        if self.ep_fsdp_mesh is not None:
            return self.ep_fsdp_mesh["ep"]
        else:
            return None
```

`SpecInfo` 的作用是在参数上附加 EP 元信息，供 checkpoint 保存时正确重建 DTensor 分布。

### 4.3 `apply()` 方法逐行分析

`apply()`（`parallel_plan.py:50-83`）是 EP 切分的核心执行函数：

```python
# parallel_plan.py:50-83
def apply(self, model: nn.Module, ep_fsdp_mesh: DeviceMesh):
    ep_mesh = ep_fsdp_mesh["ep"]                          # 提取 EP 子 mesh
    fqn2spec_info = {}

    if self.ep_plan:
        ep_size = ep_mesh.size(-1)                         # EP 并行度
        ep_replicate = [Replicate() for _ in range(ep_mesh.ndim)]  # 全 Replicate placement

        for fqn, param in model.named_parameters():       # 遍历所有参数
            for fqn_pattern, shard in self.ep_plan.items():
                if check_fqn_match(fqn_pattern, fqn):     # FQN 模式匹配
                    assert param.size(shard.dim) % ep_size == 0   # 可整除检查

                    # Step 1: 构造 EP placement
                    ep_placement = ep_replicate[:-1] + [shard]    # [..., Shard(0)]

                    # Step 2: DTensor.from_local → 标记为 Replicate
                    dtensor = DTensor.from_local(
                        local_tensor=param.data,
                        device_mesh=ep_mesh,
                        placements=ep_replicate           # 声称当前数据是 Replicate 的
                    )

                    # Step 3: redistribute → 执行实际切分
                    dtensor = dtensor.redistribute(
                        device_mesh=ep_mesh,
                        placements=ep_placement           # 目标：Shard(0)
                    )

                    # Step 4: to_local → 取出本地切片
                    local_chunk = torch.nn.Parameter(
                        dtensor.to_local(),
                        requires_grad=param.requires_grad
                    )

                    # Step 5: 附加 SpecInfo 元信息
                    local_chunk.spec_info = SpecInfo(
                        ep_fsdp_mesh=ep_fsdp_mesh,
                        placement=shard,
                        fqn=fqn
                    )

                    # Step 6: 替换模型中的参数
                    set_module_from_path(model, fqn, local_chunk)
                    fqn2spec_info[fqn] = SpecInfo(...)
                    break                                  # 匹配成功，跳出内循环

            if fqn not in fqn2spec_info:                   # 非 EP 参数
                param.spec_info = SpecInfo(
                    ep_fsdp_mesh=ep_fsdp_mesh,
                    placement=Replicate(),                 # 标记为 Replicate
                    fqn=fqn
                )
                fqn2spec_info[fqn] = SpecInfo(...)

    # 完整性检查
    for param in model.parameters():
        assert hasattr(param, "spec_info"), f"Internal Error: {param} is omitted"

    return fqn2spec_info
```

### 4.4 DTensor 生命周期图解

`apply()` 中的 DTensor 转换是一个 `Replicate → Shard → local` 的三步过程：

```
原始参数（每个 rank 持有完整的 [128, 4096, 2048] 张量）
    │
    ▼  DTensor.from_local(placements=[Replicate()])
DTensor: 逻辑上 [128, 4096, 2048]，每个 EP rank 物理上也是 [128, 4096, 2048]
    │         标记为 Replicate，即"所有 rank 数据相同"
    │
    ▼  redistribute(placements=[Shard(0)])
DTensor: 逻辑上 [128, 4096, 2048]，但每个 EP rank 物理上是 [32, 4096, 2048]
    │         Shard(0) 触发底层的 scatter/split 操作
    │         4 个 EP rank 各取 128/4 = 32 个专家
    │
    ▼  to_local()
普通 Tensor: [32, 4096, 2048]
    │
    ▼  nn.Parameter(local_chunk)
参数替换回模型，附带 spec_info 元信息
```

**数值示例（ep_size=4, num_experts=128）**：

```
EP rank 0: gate_proj [128, 4096, 2048] → [32, 4096, 2048]  (专家 0-31)
EP rank 1: gate_proj [128, 4096, 2048] → [32, 4096, 2048]  (专家 32-63)
EP rank 2: gate_proj [128, 4096, 2048] → [32, 4096, 2048]  (专家 64-95)
EP rank 3: gate_proj [128, 4096, 2048] → [32, 4096, 2048]  (专家 96-127)
```

> **NOTE**：`DTensor.from_local(placements=[Replicate()])` 并不执行任何通信。它只是 **声称** 当前的本地数据在所有 rank 上是相同的。随后的 `redistribute(placements=[Shard(0)])` 才执行实际的切分。由于数据确实是 Replicate 的（每个 rank 初始化了相同的完整参数），所以 redistribute 只需要做本地 slice，无需网络通信。

### 4.5 `get_fsdp_no_shard_info()` -- 查找专家模块

```python
# parallel_plan.py:85-96
def get_fsdp_no_shard_info(self, model: nn.Module):
    if self.fsdp_no_shard_module is None:
        return None

    fsdp_no_shard_states_fqn_to_module = {}
    for fqn, param in model.named_modules():          # 注意：named_modules，不是 named_parameters
        for no_shard_pattern in self.fsdp_no_shard_module:
            if check_fqn_match(no_shard_pattern, fqn):
                fsdp_no_shard_states_fqn_to_module[fqn] = get_module_from_path(model, fqn)

    assert len(fsdp_no_shard_states_fqn_to_module) > 0
    return fsdp_no_shard_states_fqn_to_module
```

返回一个 `{FQN: Module}` 字典，例如：

```python
{
    "model.layers.0.mlp.experts": <PatchQwen3MoeExperts object>,
    "model.layers.1.mlp.experts": <PatchQwen3MoeExperts object>,
    ...
    "model.layers.N.mlp.experts": <PatchQwen3MoeExperts object>,
}
```

这些模块会被传递给主 FSDP 的 `ignored_states` 参数，使主 FSDP 跳过这些模块的参数。

### 4.6 `shard_tensor()` -- EP-aware Checkpoint 加载

`shard_tensor()`（`parallel_plan.py:106-170`）在加载 checkpoint 时使用，根据 EP rank 从完整的专家权重中切出本地部分：

```python
# parallel_plan.py:106-122
def shard_tensor(self, tensor, full_param_name, target_shape):
    if not self._is_expert_parameter(full_param_name):
        return tensor                    # 非专家参数，直接返回
    return self._slice_expert_tensor_for_ep(tensor, full_param_name, target_shape)
```

```python
# parallel_plan.py:134-170
def _slice_expert_tensor_for_ep(self, tensor, parameter_name, target_shape):
    parallel_state = get_parallel_state()

    if len(tensor.shape) >= 1 and len(target_shape) >= 1:
        tensor_experts = tensor.shape[0]       # checkpoint 中的专家数（如 128）
        target_experts = target_shape[0]       # 本地期望的专家数（如 32）

        if tensor_experts > target_experts and tensor_experts % target_experts == 0:
            ep_size = tensor_experts // target_experts
            ep_rank = parallel_state.ep_rank
            start_idx = ep_rank * target_experts
            end_idx = start_idx + target_experts

            sliced_tensor = tensor[start_idx:end_idx]    # 按 EP rank 切片
            return sliced_tensor

    return tensor   # 无需切片
```

**切片逻辑**：

```
checkpoint 权重: gate_proj [128, 4096, 2048]  (完整的 128 个专家)
target_shape:    gate_proj [32, 4096, 2048]   (模型期望本地 32 个专家)

ep_size = 128 / 32 = 4
EP rank 0: tensor[0:32]    → [32, 4096, 2048]
EP rank 1: tensor[32:64]   → [32, 4096, 2048]
EP rank 2: tensor[64:96]   → [32, 4096, 2048]
EP rank 3: tensor[96:128]  → [32, 4096, 2048]
```

### 4.7 `check_fqn_match()` -- 通配符匹配

`check_fqn_match()`（`utils.py:100-110`）将 FQN 模式中的 `*` 转换为正则表达式的 `.*` 进行匹配：

```python
# utils.py:100-110
def check_fqn_match(fqn_pattern: str, fqn: str, prefix: str = None):
    regex_str = re.escape(fqn_pattern).replace(r"\*", r".*")
    regex_str = f"^{regex_str}$"
    regex = re.compile(regex_str)
    return regex.match(fqn) is not None
```

匹配示例：

```
模式: "model.layers.*.mlp.experts.gate_proj"
匹配: "model.layers.0.mlp.experts.gate_proj"    ✓
匹配: "model.layers.15.mlp.experts.gate_proj"   ✓
不匹配: "model.layers.0.mlp.gate.weight"         ✗
不匹配: "model.layers.0.self_attn.q_proj.weight"  ✗
```

### 4.8 `apply()` 完整数据流总结

```
apply(model, ep_fsdp_mesh)
│
├── for fqn, param in model.named_parameters():
│   │
│   ├── [匹配 EP 模式] "model.layers.3.mlp.experts.gate_proj"
│   │   │
│   │   ├── DTensor.from_local(param, ep_mesh, [Replicate()])
│   │   │   └── 不做通信，只是标记
│   │   │
│   │   ├── dtensor.redistribute(ep_mesh, [Shard(0)])
│   │   │   └── 本地 slice: param[ep_rank*chunk : (ep_rank+1)*chunk]
│   │   │
│   │   ├── dtensor.to_local() → local_chunk
│   │   │
│   │   ├── local_chunk.spec_info = SpecInfo(ep_fsdp_mesh, Shard(0), fqn)
│   │   │
│   │   └── set_module_from_path(model, fqn, local_chunk)
│   │       └── 替换模型中的原参数
│   │
│   └── [不匹配 EP 模式] "model.layers.3.self_attn.q_proj.weight"
│       │
│       └── param.spec_info = SpecInfo(ep_fsdp_mesh, Replicate(), fqn)
│           └── 仅标记，不修改参数
│
└── return fqn2spec_info
    └── 所有参数的 SpecInfo 映射，供 checkpoint extension 使用
```

---

## 第 5 节：FSDP1 路径 -- EP + FSDP1 集成

> **一句话总结**：`parallelize_model_fsdp1()` 通过"先排除再单包"的策略实现 EP+FSDP 协同——主 FSDP 包装整个模型时排除专家模块，然后用独立的 `ep_fsdp` mesh 单独包装每个专家模块，并手动修正 `_gradient_postdivide_factor` 确保梯度平均正确。

### 5.1 函数签名与入口

```python
# torch_parallelize.py:84-100
def parallelize_model_fsdp1(
    model: "nn.Module",
    weights_path: Optional[str] = None,
    enable_full_shard: bool = True,
    enable_shard_grad_op: bool = False,
    enable_mixed_precision: bool = True,
    use_orig_params: bool = True,
    basic_modules: Optional[List[str]] = None,
    fsdp_no_shard_states=None,
    fsdp_no_shard_states_fqn=None,
    ep_param_suffix=None,
    fqn2spec_info=None,
    **kwargs,
) -> "nn.Module":
```

### 5.2 Step 1：EP 切分与模块发现

```python
# torch_parallelize.py:104-117
parallel_state = get_parallel_state()

if parallel_state.ep_enabled:
    # 1a. 获取模型定义的 EP plan
    parallel_plan = model.get_parallel_plan()

    # 1b. 执行 EP 切分（DTensor from_local → redistribute → to_local）
    fqn2spec_info = parallel_plan.apply(model, parallel_state.ep_fsdp_device_mesh)

    # 1c. 获取需要从主 FSDP 排除的专家模块
    fsdp_no_shard_states_fqn_to_module = parallel_plan.get_fsdp_no_shard_info(model)
    fsdp_no_shard_states = list(fsdp_no_shard_states_fqn_to_module.values())
    fsdp_no_shard_states_fqn = list(fsdp_no_shard_states_fqn_to_module.keys())
else:
    fqn2spec_info = None
    fsdp_no_shard_states = None
    fsdp_no_shard_states_fqn = None
```

执行后的状态：

```
fsdp_no_shard_states:      [<experts layer 0>, <experts layer 1>, ..., <experts layer N>]
fsdp_no_shard_states_fqn:  ["model.layers.0.mlp.experts", "model.layers.1.mlp.experts", ...]
fqn2spec_info:             {fqn → SpecInfo} 的完整映射
```

### 5.3 Step 2：主 FSDP 包装配置

```python
# torch_parallelize.py:119-134
# 自动包装策略：按 basic_modules 的类名判断
wrap_policy = partial(
    lambda_auto_wrap_policy,
    lambda_fn=lambda module: module.__class__.__name__ in basic_modules
)

# 选择 sharding strategy
if parallel_state.fsdp_mesh.ndim > 1 and parallel_state.fsdp_mesh.size() > 1:
    # HSDP 模式：2D mesh → HYBRID_SHARD
    strategy = ShardingStrategy.HYBRID_SHARD if enable_full_shard \
               else ShardingStrategy._HYBRID_SHARD_ZERO2
else:
    # 标准 FSDP：1D mesh → FULL_SHARD
    strategy = ShardingStrategy.FULL_SHARD if enable_full_shard \
               else ShardingStrategy.SHARD_GRAD_OP

fsdp_kwargs = {
    "auto_wrap_policy": wrap_policy,
    "ignored_states": fsdp_no_shard_states,   # ← 关键：排除专家模块
    "device_id": get_device_id(),
    "sharding_strategy": strategy,
    "use_orig_params": use_orig_params,
    "device_mesh": parallel_state.fsdp_mesh,   # ← 使用主 FSDP mesh
}
```

**`ignored_states` 的作用**：告诉 FSDP "这些模块的参数不要做 all-gather / reduce-scatter"。专家模块被排除后，它们的参数在主 FSDP 看来是"不存在的"。

### 5.4 Step 3：Meta Init 路径（可选）

```python
# torch_parallelize.py:155-169
elif kwargs.get("init_device") == "meta":
    # Meta 初始化路径：参数在 meta device 上，需要通过 param_init_fn 实际分配
    shard_states = kwargs.get("shard_states", {})
    if not shard_states and weights_path:
        shard_states = parallel_load_safetensors(weights_path)
    shard_states = _convert_state_dict_keys(shard_states, model)

    fsdp_kwargs["param_init_fn"] = parallel_init_fsdp_fn(
        model,
        shard_states.copy(),
        ignore_states=fsdp_no_shard_states,   # 同样排除专家模块
        strict=kwargs.pop("strict", False),
    )
```

### 5.5 Step 4：执行主 FSDP 包装

```python
# torch_parallelize.py:182
model = FullyShardedDataParallel(model, **fsdp_kwargs)
```

此时模型的结构为：

```
FSDP(model)                               ← 主 FSDP，fsdp_mesh
├── FSDP(model.layers.0)                  ← auto_wrap_policy 按 basic_modules 包装
│   ├── attention, attention_norm, ...    ← 这些参数被 FSDP 管理
│   └── mlp.experts                       ← ignored！参数不受 FSDP 管理
├── FSDP(model.layers.1)
│   ├── ...
│   └── mlp.experts                       ← ignored！
└── ...
```

### 5.6 Step 5：专家模块单独包装

这是整个 FSDP1 路径中最关键的部分（`torch_parallelize.py:184-214`）：

```python
# torch_parallelize.py:184-214
if fsdp_no_shard_states is not None:
    # 根据 ep_fsdp mesh 大小选择策略
    if parallel_state.ep_fsdp_mesh["ep_fsdp"].size() == 1:
        # ep_fsdp 只有 1 个 rank → 不需要 FSDP 切分
        moe_sharding_strategy = ShardingStrategy.NO_SHARD
        ep_fsdp_device_mesh = parallel_state.fsdp_mesh
    else:
        # ep_fsdp 有多个 rank → 在 ep_fsdp 维度做 FULL_SHARD
        moe_sharding_strategy = ShardingStrategy.FULL_SHARD
        ep_fsdp_device_mesh = parallel_state.ep_fsdp_mesh["ep_fsdp"]

    # 更新 fsdp_kwargs
    fsdp_kwargs.pop("ignored_states", None)
    fsdp_kwargs.pop("auto_wrap_policy", None)
    fsdp_kwargs["sharding_strategy"] = moe_sharding_strategy
    fsdp_kwargs["device_mesh"] = ep_fsdp_device_mesh

    for fqn in fsdp_no_shard_states_fqn:
        no_shard_module = get_module_from_path(model, fqn)

        # Meta init 路径的额外处理
        if kwargs.get("init_device") == "meta":
            key_prefix = fqn + "."
            ep_shard_states = {
                k[len(key_prefix):]: v for k, v in shard_states.items()
                if k.startswith(key_prefix)
            }
            fsdp_kwargs["param_init_fn"] = parallel_init_fsdp_fn(
                no_shard_module, ep_shard_states, strict=False,
            )

        # 用独立的 mesh 包装专家模块
        fsdp_module = FullyShardedDataParallel(no_shard_module, **fsdp_kwargs)

        # *** 关键：修正梯度除法因子 ***
        fsdp_state = _get_module_fsdp_state_if_fully_sharded_module(fsdp_module)
        fsdp_state._gradient_postdivide_factor *= parallel_state.ep_size

        # 替换模型中的模块
        set_module_from_path(model, fqn, fsdp_module)
```

包装后的模型结构：

```
FSDP(model, fsdp_mesh)                           ← 主 FSDP
├── FSDP(model.layers.0, fsdp_mesh)              ← 普通层 FSDP
│   ├── attention, attention_norm, ...
│   └── FSDP(mlp.experts, ep_fsdp_mesh)          ← 专家层 FSDP（独立 mesh！）
│       ├── gate_proj: [32, 4096, 2048]
│       ├── up_proj:   [32, 4096, 2048]
│       └── down_proj: [32, 2048, 4096]
├── FSDP(model.layers.1, fsdp_mesh)
│   ├── ...
│   └── FSDP(mlp.experts, ep_fsdp_mesh)
└── ...
```

### 5.7 `_gradient_postdivide_factor` 修正的原理

这是一个容易出错的关键细节。让我们通过数值示例理解为什么需要这个修正。

**场景**：16 GPU，ep_size=4，ep_fsdp_size=4

FSDP 在 reduce-scatter 时会将梯度除以 FSDP group 的大小。对于专家模块：

```
未修正时：
  ep_fsdp group 大小 = 4
  FSDP 默认 _gradient_postdivide_factor = 4
  reduce-scatter 后梯度被除以 4

但正确的行为应该是：
  梯度应该在所有 16 个 GPU 上取平均（world_size = 16）
  reduce-scatter 在 ep_fsdp group (4 个 rank) 上执行
  所以 FSDP 内部除以 4 后，还需要除以 ep_size = 4
  总除法因子 = 4 * 4 = 16 = world_size  ✓

修正后：
  fsdp_state._gradient_postdivide_factor *= parallel_state.ep_size
  即 4 *= 4 → 16
  梯度被正确地除以 16
```

> **NOTE**：对比 TorchTitan 的做法——TorchTitan 使用 `disable_fsdp_gradient_division()` 完全禁用 FSDP 的自动梯度除法，然后在训练循环中手动用全局 token count 控制。VeOmni 的方式更精确但更脆弱——它依赖于 FSDP 内部 `_gradient_postdivide_factor` 属性的存在和语义。

### 5.8 Sharding Strategy 选择逻辑

```
ep_fsdp_mesh["ep_fsdp"].size() == 1?
├── YES:
│   strategy = NO_SHARD
│   mesh = fsdp_mesh（主 mesh，用于混合精度等元数据管理）
│   含义：专家参数不做 FSDP 切分，只做 EP 切分
│   场景：world_size == ep_size（所有 GPU 各持有不同的专家）
│
└── NO:
    strategy = FULL_SHARD
    mesh = ep_fsdp_mesh["ep_fsdp"]（EP 内的 FSDP 子 mesh）
    含义：专家参数先被 EP 在 dim 0 切分，再被 FSDP 做进一步切分
    场景：world_size > ep_size（FSDP 和 EP 共同分担参数切分）
```

**数值示例（16 GPU, ep_size=4）**：

```
ep_fsdp_mesh["ep_fsdp"].size() = 4 > 1 → FULL_SHARD

专家参数的切分过程：
  原始: gate_proj [128, 4096, 2048]
  EP 切分 (dim 0): → [32, 4096, 2048]  (每个 EP rank 32 个专家)
  FSDP 切分:       → [8, 4096, 2048]   (每个 ep_fsdp rank 进一步切分)
                       或者可能切其他维度，取决于 FSDP 的默认 flatten 行为
```

### 5.9 Step 6：注册 Checkpoint Extension

```python
# torch_parallelize.py:218-225
save_hook_mesh = parallel_state.ep_fsdp_device_mesh if parallel_state.ep_enabled else None
register_checkpoint_extension(
    fsdp_model=model,
    save_hook_mesh=save_hook_mesh,
    fqn2spec_info=fqn2spec_info,
)
```

Checkpoint extension 的作用是在保存/加载 checkpoint 时，正确处理 EP 切分的参数——保存时将本地切片标记为 DTensor shard，加载时根据 EP rank 从完整 checkpoint 中切出正确的部分。

### 5.10 Step 7：替换梯度裁剪函数

```python
# torch_parallelize.py:227-228
if parallel_state.ep_enabled:
    model.clip_grad_norm_ = types.MethodType(clip_grad_norm_, model)
```

EP-aware 的 `clip_grad_norm_`（`fsdp/clip_grad_norm.py:15-30`）需要跨 EP group 聚合梯度范数：

```python
# fsdp/clip_grad_norm.py:15-30
def clip_grad_norm_(fsdp_model: FSDP, max_norm, norm_type=2.0):
    extension = fsdp_model._fsdp_extension
    ep_mesh = extension.ep_mesh
    ep_group = None if ep_mesh is None else ep_mesh.get_group()

    if ep_group is None or dist.get_world_size(ep_group) in (1, dist.get_world_size()):
        return FSDP.clip_grad_norm_(fsdp_model, max_norm, norm_type)

    # EP-aware: 需要在 EP group 上额外 all-reduce 梯度范数
    # ...
```

**为什么需要 EP-aware 的梯度裁剪？** 因为标准 FSDP 的 `clip_grad_norm_` 只在 FSDP group 内聚合梯度范数。但 EP 参数分布在不同的 EP rank 上，每个 rank 只有部分专家的梯度。要计算全局梯度范数，需要在 EP group 上额外做一次 all-reduce。

### 5.11 FSDP1 路径完整流程图

```
parallelize_model_fsdp1(model)
│
├── 1. EP 切分
│   ├── parallel_plan = model.get_parallel_plan()
│   ├── fqn2spec_info = parallel_plan.apply(model, ep_fsdp_mesh)
│   │   └── 每个专家参数: [128,H,D] → [32,H,D] (ep_size=4)
│   └── fsdp_no_shard_info = parallel_plan.get_fsdp_no_shard_info(model)
│       └── {fqn → module} 字典
│
├── 2. 主 FSDP 配置
│   ├── wrap_policy = lambda_auto_wrap_policy(basic_modules)
│   ├── strategy = HYBRID_SHARD / FULL_SHARD
│   └── ignored_states = [experts modules]
│
├── 3. Meta Init（可选）
│   └── param_init_fn = parallel_init_fsdp_fn(shard_states, ignore=experts)
│
├── 4. 主 FSDP 包装
│   └── model = FSDP(model, **fsdp_kwargs)
│       └── experts 被 ignored，不参与主 FSDP
│
├── 5. 专家 FSDP 包装
│   └── for each experts module:
│       ├── fsdp_module = FSDP(experts, mesh=ep_fsdp_mesh["ep_fsdp"])
│       ├── fsdp_state._gradient_postdivide_factor *= ep_size  ← 关键修正
│       └── set_module_from_path(model, fqn, fsdp_module)
│
├── 6. Checkpoint Extension
│   └── register_checkpoint_extension(fqn2spec_info)
│
└── 7. 梯度裁剪
    └── model.clip_grad_norm_ = EP-aware clip_grad_norm_
```

### 5.12 FSDP1 嵌套包装结构

```
┌──────────────────────────────────────────────────────────────────┐
│ FSDP(model, mesh=fsdp_mesh, strategy=FULL_SHARD)                │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │ FSDP(model.layers.0, mesh=fsdp_mesh)                      │   │
│  │  ├── self_attn (q_proj, k_proj, v_proj, o_proj)           │   │
│  │  ├── mlp.gate (路由器，不参与 EP)                          │   │
│  │  │                                                         │   │
│  │  │  ┌──────────────────────────────────────────────────┐   │   │
│  │  │  │ FSDP(mlp.experts, mesh=ep_fsdp_mesh["ep_fsdp"]) │   │   │
│  │  │  │  ├── gate_proj: [32, 4096, 2048]                │   │   │
│  │  │  │  ├── up_proj:   [32, 4096, 2048]                │   │   │
│  │  │  │  └── down_proj: [32, 2048, 4096]                │   │   │
│  │  │  │  _gradient_postdivide_factor *= ep_size          │   │   │
│  │  │  └──────────────────────────────────────────────────┘   │   │
│  │  ├── input_layernorm                                       │   │
│  │  ├── post_attention_layernorm                              │   │
│  └───────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │ FSDP(model.layers.1, mesh=fsdp_mesh)                      │   │
│  │  ├── ...                                                   │   │
│  │  └── FSDP(mlp.experts, mesh=ep_fsdp_mesh["ep_fsdp"])     │   │
│  └───────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ...                                                             │
│                                                                  │
│  ├── model.embed_tokens                                         │
│  ├── model.norm                                                  │
│  └── lm_head                                                     │
└──────────────────────────────────────────────────────────────────┘
```

> **NOTE**：与 TorchTitan 的双层 FSDP 相比，VeOmni 的 FSDP1 路径有几个显著区别：(1) VeOmni 使用 `ignored_states` 排除专家模块，而 TorchTitan 是先包内层再包外层，利用 FSDP2 的自动跳过机制；(2) VeOmni 通过修改 `_gradient_postdivide_factor` 修正梯度，TorchTitan 通过 `disable_fsdp_gradient_division()` + 手动控制；(3) VeOmni 不实现显式 prefetch（FSDP1 的隐式 prefetch 在大多数情况下足够），TorchTitan 有完整的前向/反向 prefetch 链来解决 EP 的 D2H 同步问题。

---

## 第 6 节：FSDP2 路径 -- EP + FSDP2 集成

> **一句话总结**：`parallelize_model_fsdp2()` 通过四步流程——提取目标 decoder block、应用 EP 切分专家权重、构造双套 FSDP kwargs（标准 mesh vs ep_fsdp mesh）、内层-外层双重 `fully_shard` 包装——实现了 EP 与 FSDP2 的正交切分，其中 `Shard(0)` 用于 EP 的专家维度、`Shard(1)` 用于 FSDP2 的 hidden 维度。

### 6.1 函数签名与入口

`torch_parallelize.py:237-253` 定义了 FSDP2 路径的主函数：

```python
# torch_parallelize.py:237-244
def parallelize_model_fsdp2(
    model: "nn.Module",
    weights_path: Optional[str] = None,
    enable_reshard_after_forward: bool = True,
    enable_mixed_precision: bool = True,
    basic_modules: Optional[List[str]] = None,
    **kwargs,
) -> "nn.Module":
```

文档注释（`torch_parallelize.py:245-253`）清晰地描述了整体流程：

```
Flow:
1. Apply EP: Expert tensors [128,H,I] -> [32,H,I] local tensors per EP rank
2. Apply FSDP2 to expert modules: Shard expert tensors along dim-1 (hidden dim)
3. Apply FSDP2 to regular modules: Standard dim-0 sharding
4. Result: Expert params [32,H/fsdp_size,I], regular params use standard FSDP2
```

### 6.2 Step 0：提取目标 Decoder Block

`torch_parallelize.py:256-261`：

```python
# Step 0: Get target classes to shard later
target_classes = set((getattr(model, "_no_split_modules", []) or []) + (basic_modules or []))
# Make a list of tuples that contains layer's name and module
decoder_blocks: List[Tuple[str, nn.Module]] = [
    (fqn, mod) for fqn, mod in model.named_modules() if mod.__class__.__name__ in target_classes
]
```

**设计要点**：

- `_no_split_modules` 是 HuggingFace 模型的标准属性，用于标识不应被 pipeline parallelism 拆分的模块。对于 Qwen3MoE，这通常是 `Qwen3MoeDecoderLayer`
- `basic_modules` 是用户可额外指定的模块类名
- 结果是一个 `(fqn, module)` 元组列表，例如 `[("model.layers.0", <Qwen3MoeDecoderLayer>), ...]`

### 6.3 Step 1：应用 Expert Parallelism

`torch_parallelize.py:264-285`：

```python
# Step 1: Apply expert parallelism (slice expert tensors [128,H,I] -> [16,H,I])
if parallel_state.ep_enabled:
    parallel_plan = model.get_parallel_plan()
    assert parallel_plan is not None, (
        "Expert parallelism needs parallel plan defined in the model!"
    )
    ep_fqn2spec_info = parallel_plan.apply(model, parallel_state.ep_fsdp_device_mesh)
    # Attach spec mapping for checkpoint load-time reconstruction
    setattr(model, "_fqn2spec_info", ep_fqn2spec_info)
    ep_mesh = parallel_state.ep_fsdp_device_mesh["ep"]
    # experts_map is a dict {experts_fqn: experts_mod}
    experts_map = parallel_plan.get_fsdp_no_shard_info(model)
```

这里发生了三件重要的事：

1. **`parallel_plan.apply()`**：遍历模型的所有参数，对匹配 EP plan 中模式（如 `model.layers.*.mlp.experts.gate_proj`）的参数执行 `DTensor.from_local() -> redistribute(Shard(0)) -> to_local()`，将全局专家权重 `[128, H, I]` 切分为本地 `[128/ep_size, H, I]`

2. **`setattr(model, "_fqn2spec_info", ...)`**：将 FQN 到 `SpecInfo` 的映射挂载到模型上，用于 checkpoint 保存/加载时重建 EP 切分

3. **`get_fsdp_no_shard_info()`**：返回需要独立 FSDP 包装的模块映射。对于 Qwen3MoE，key 是 `model.layers.*.mlp.experts`，value 是对应的 `PatchQwen3MoeExperts` 模块

> **NOTE**：`parallel_plan.apply()` 的实现（`parallel_plan.py:50-83`）中有一个关键细节——EP 切分后的参数被转回普通 Tensor（`dtensor.to_local()`），而不是保持 DTensor 状态。这是因为后续 FSDP2 会重新将这些参数包装为 DTensor，但使用的是 `ep_fsdp_mesh` 而非 `ep_mesh`。

### 6.4 Layer Pair 提取

`torch_parallelize.py:287-302`：

```python
# Extract experts module from the layer if any, then pair them
layer_pairs = []
for layer_fqn, layer_mod in decoder_blocks:
    if experts_map is not None:
        # extract experts module from the layer
        experts_mod = next(
            (exp_mod for exp_fqn, exp_mod in experts_map.items()
             if exp_fqn.startswith(layer_fqn + ".")),
            None,
        )
        layer_pairs.append((layer_fqn, layer_mod, experts_mod))
    else:
        # No experts module found in this layer
        # this is often the case for models like deepseek
        # in which some decoder layers are dense instead of MoE
        layer_pairs.append((layer_fqn, layer_mod, None))
```

**设计考虑**：

- 并非每个 decoder layer 都包含 MoE 层。例如 DeepSeek 模型中，部分层是 dense FFN，另一部分是 MoE
- 通过 `exp_fqn.startswith(layer_fqn + ".")` 来判断某个 experts 模块是否属于当前 layer
- `experts_mod` 可能为 `None`（dense 层的情况）

**数据结构示例**（Qwen3MoE，假设所有层都是 MoE）：

```
layer_pairs = [
    ("model.layers.0", <Qwen3MoeDecoderLayer>, <PatchQwen3MoeExperts>),
    ("model.layers.1", <Qwen3MoeDecoderLayer>, <PatchQwen3MoeExperts>),
    ...
    ("model.layers.35", <Qwen3MoeDecoderLayer>, <PatchQwen3MoeExperts>),
]
```

### 6.5 Step 2：构造 FSDP2 Kwargs

#### 6.5.1 基础 FSDP kwargs

`torch_parallelize.py:304-312`：

```python
# Step 2: Update fsdp2 kwargs
fsdp_kwargs = {"mesh": parallel_state.fsdp_mesh, "reshard_after_forward": enable_reshard_after_forward}
# mp_policy kwargs
if enable_mixed_precision:
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )
    fsdp_kwargs["mp_policy"] = mp_policy
```

- `mesh`：标准的 FSDP mesh，用于 dense 层和 MoE 层的非专家部分（attention、gate 等）
- `reshard_after_forward`：前向完成后是否释放 all-gather 的参数（节省内存，但反向需要重新 all-gather）
- `MixedPrecisionPolicy`：参数以 bf16 参与计算，梯度 reduce-scatter 使用 fp32

#### 6.5.2 混合精度豁免处理

`torch_parallelize.py:314-331`：

```python
if hasattr(model, "get_ignore_modules_in_mixed_precision"):
    modules_to_ignore_in_mixed_precision = model.get_ignore_modules_in_mixed_precision()
else:
    modules_to_ignore_in_mixed_precision = None

if modules_to_ignore_in_mixed_precision:
    assert isinstance(modules_to_ignore_in_mixed_precision, tuple)
    mp_ignored_classes = modules_to_ignore_in_mixed_precision
    fsdp_kwargs_without_mp = dict(fsdp_kwargs)
    fsdp_kwargs_without_mp.pop("mp_policy", None)
    # for high-precision modules, we do not reshard them after forward
    fsdp_kwargs_without_mp["reshard_after_forward"] = False
```

**设计含义**：

- 某些模块（例如 MoE 的 gate/router 层）需要在 fp32 精度下运行
- 为这些模块创建一套单独的 `fsdp_kwargs_without_mp`，不包含 `mp_policy`
- 同时设置 `reshard_after_forward=False`——因为高精度模块通常参数量小，保持在 GPU 内存中不会造成显著压力，但可以避免反向时的额外 all-gather 开销

#### 6.5.3 Expert FSDP kwargs

`torch_parallelize.py:333-344`：

```python
# prepare ep_fsdp2 kwargs
if parallel_state.ep_enabled:
    # Use the ep_fsdp dimension as DP mesh for experts
    ep_fsdp_mesh = parallel_state.ep_fsdp_device_mesh["ep_fsdp"]
    expert_fsdp_kwargs = dict(fsdp_kwargs)
    expert_fsdp_kwargs["mesh"] = ep_fsdp_mesh

    # Prefer dim-1 sharding for expert weights when composing with EP shard on dim-0
    def _experts_shard_placement_fn(param):
        return Shard(1)

    expert_fsdp_kwargs["shard_placement_fn"] = _experts_shard_placement_fn
```

**这是 EP+FSDP2 集成的核心设计**。两个关键点：

1. **`mesh = ep_fsdp_mesh`**：专家模块使用 `ep_fsdp` 子 mesh 而非标准 `fsdp_mesh`。这是因为 EP 已经将专家按 dim 0 分配到不同的 EP rank 上，FSDP 只需要在 **同一个 EP rank 内的 FSDP shard group** 上再次切分

2. **`shard_placement_fn = lambda param: Shard(1)`**：FSDP2 默认在 dim 0 上切分参数。但专家权重的 dim 0 是专家维度，已经被 EP 切分过了。因此 FSDP2 必须切换到 dim 1（hidden 维度）来进行正交切分

### 6.6 Shard(0) vs Shard(1) 正交切分

这是 VeOmni EP+FSDP2 最核心的设计思想，值得详细展开。

以 Qwen3MoE 的 `gate_proj` 权重为例：

```
原始权重形状:  [num_experts, intermediate_dim, hidden_dim]
              例如: [128, 4096, 2048]

Step 1 — EP 应用 Shard(0)（专家维度）:
  ep_size = 4
  每个 EP rank 持有: [128/4, 4096, 2048] = [32, 4096, 2048]

Step 2 — FSDP2 应用 Shard(1)（hidden 维度）:
  ep_fsdp_size = 8
  每个 rank 最终持有: [32, 4096/8, 2048] = [32, 512, 2048]
```

两次切分完全正交，互不干扰：

```
原始参数（全局视角）:
┌──────────────────────────────────────────────────┐
│              [128, 4096, 2048]                    │
│                                                    │
│  dim 0: 专家维度 ──── EP Shard(0) 切分            │
│  dim 1: intermediate_dim ── FSDP2 Shard(1) 切分  │
│  dim 2: hidden_dim ──── 不切分                    │
└──────────────────────────────────────────────────┘

切分后（EP rank 0, FSDP shard 0 视角）:
┌──────────────────────────┐
│    [32, 512, 2048]      │
│                          │
│  32 = 128/4 (EP)        │
│  512 = 4096/8 (FSDP)   │
│  2048 = 不变            │
└────────────────────────┘
```

> **NOTE**：为什么不能对 dim 0 再次切分？假设 `ep_size=4`，每个 EP rank 有 32 个专家。如果 `ep_fsdp_size=8`，FSDP 在 dim 0 上切分就意味着 32/8=4 个专家/shard。但 GroupGemm 的 `cumsum` 边界需要按本地专家数量划分，dim 0 的 FSDP 切分会破坏这种对应关系。切换到 dim 1 则完全避免了这个问题。

### 6.7 内层-外层双重包装

`torch_parallelize.py:359-391` 是 FSDP2 包装的核心循环：

```python
for layer_fqn, layer_mod, experts_mod in layer_pairs:
    # register all the FSDPModule inside this decoder layer
    layer_mod._fsdp_modules = []

    # ep enabled and this layer contains the expert module
    if parallel_state.ep_enabled and experts_mod is not None:
        # shard expert — 内层 FSDP（ep_fsdp_mesh 上）
        fully_shard(experts_mod, **expert_fsdp_kwargs)

        # average EP grads across EP ranks
        gradient_divide_factor = parallel_state.ep_gradient_divide_factor
        if IS_NPU_AVAILABLE:
            experts_mod.set_reduce_scatter_divide_factor(gradient_divide_factor)
        else:
            experts_mod.set_gradient_divide_factor(gradient_divide_factor)

        layer_mod._fsdp_modules.append(experts_mod)

    # shard module that needs to ignore mixed precision control
    if mp_ignored_classes:
        for sub_mod in layer_mod.modules():
            if isinstance(sub_mod, mp_ignored_classes) and sub_mod is not layer_mod:
                fully_shard(sub_mod, **fsdp_kwargs_without_mp)
                layer_mod._fsdp_modules.append(sub_mod)

    # shard everything else in the decoder layer — 外层 FSDP（fsdp_mesh 上）
    fully_shard(layer_mod, **fsdp_kwargs)
    layer_mod._fsdp_modules.append(layer_mod)
```

**执行顺序极为关键**：

1. **先包装 experts**（内层）：`fully_shard(experts_mod, **expert_fsdp_kwargs)` — 在 `ep_fsdp_mesh` 上
2. **再包装 mp_ignored 模块**（中间层，如有）：`fully_shard(sub_mod, **fsdp_kwargs_without_mp)` — 在 `fsdp_mesh` 上但不含 mp_policy
3. **最后包装整个 layer**（外层）：`fully_shard(layer_mod, **fsdp_kwargs)` — 在 `fsdp_mesh` 上

`fully_shard` 是递归的——当它遇到已经被 `fully_shard` 过的子模块时，会将其识别为独立的 FSDP 单元，跳过对其参数的重复切分。

**`_fsdp_modules` 列表的用途**：记录每个 decoder layer 内部包含的所有 FSDP 单元，用于后续的显式 prefetch 配置。

最后，包装 root model（`torch_parallelize.py:391`）：

```python
# shard root model
fully_shard(model, **fsdp_kwargs)
```

### 6.8 gradient_divide_factor 的设计

`torch_parallelize.py:370-377`：

```python
gradient_divide_factor = parallel_state.ep_gradient_divide_factor
```

对应 `parallel_state.py:348-357`：

```python
@property
def ep_gradient_divide_factor(self) -> int:
    # We assume the world size is the total dp size by now
    assert self.tp_size == 1
    assert self.pp_size == 1
    # For ep+fsdp2, the grad divide factor should always be world size
    return self.world_size
```

**为什么梯度除法因子是 `world_size`？**

- 在 EP 场景下，每个 EP rank 只持有部分专家的参数
- FSDP2 默认的 reduce-scatter 只在 `ep_fsdp` group 内做梯度平均
- 但我们需要梯度在 **所有数据并行 rank** 上的平均值
- `world_size`（在 TP=1, PP=1 的假设下）等于全部数据并行 rank 数
- 因此需要显式设置 `gradient_divide_factor = world_size` 来覆盖 FSDP2 默认的除法行为

> **NOTE**：代码中有两个 API 分支——`set_gradient_divide_factor`（torch 2.8+）和 `set_reduce_scatter_divide_factor`（torch 2.7，NPU 环境）。功能一致，只是 API 名称随版本变化。

### 6.9 显式 Prefetch 配置

`torch_parallelize.py:393-410`：

```python
# configure manual prefetching when needed
need_manual_prefetch = parallel_state.ep_enabled or mp_ignored_classes is not None
if need_manual_prefetch:
    blocks = [pair[1] for pair in layer_pairs]
    next_blocks = blocks[1:] + [None]
    for current_block, next_block in zip(blocks, next_blocks):
        if next_block is not None:
            prefetch_modules = next_block._fsdp_modules
            # prefetch in order of attn, gate, experts
            current_block.set_modules_to_forward_prefetch(list(reversed(prefetch_modules)))

    # configure backward prefetch
    rev_blocks = list(reversed(blocks))
    prev_blocks = rev_blocks[1:] + [None]
    for current_block, prev_block in zip(rev_blocks, prev_blocks):
        if prev_block is not None:
            prefetch_modules = prev_block._fsdp_modules
            current_block.set_modules_to_backward_prefetch(list(reversed(prefetch_modules)))
```

**Prefetch 链的结构**：

假设一个 MoE layer 有 3 个 FSDP 单元（`_fsdp_modules = [experts_mod, gate_mod, layer_mod]`），那么 `reversed()` 后的 prefetch 顺序是 `[layer_mod, gate_mod, experts_mod]`。

```
Forward Prefetch 链:
Block 0 forward 完成后:
  → prefetch Block 1 的 [layer_mod, gate_mod, experts_mod]
     ↓ all-gather layer_mod 参数（attention, norm 等）
     ↓ all-gather gate_mod 参数（router weight）
     ↓ all-gather experts_mod 参数（gate_proj, up_proj, down_proj）

Block 1 forward 完成后:
  → prefetch Block 2 的 FSDP 单元列表
  ...

Backward Prefetch 链（反向遍历）:
Block N backward 完成后:
  → prefetch Block N-1 的 FSDP 单元列表
  ...
```

> **NOTE**：与 TorchTitan 类似，VeOmni 也需要显式 prefetch 来避免 EP 的 D2H 同步阻塞 FSDP 的隐式 prefetch。VeOmni 的实现更简洁——直接使用 `_fsdp_modules` 列表驱动，而 TorchTitan 针对 MoE 和 dense layer 做了条件分支。

### 6.10 FSDP2 包装结构可视化

```
Qwen3MoE Decoder Layer (MoE 层):
┌──────────────────────────────────────────────────────────────────┐
│ fully_shard(layer_mod, mesh=fsdp_mesh)           ← 外层 FSDP   │
│ ├── self_attn (wq, wk, wv, wo)                                  │
│ ├── input_layernorm                                              │
│ ├── post_attention_layernorm                                     │
│ ├── mlp (PatchQwen3MoeSparseMoeBlock)                           │
│ │   ├── gate (PatchQwen3MoeTopKRouter)                          │
│ │   │   └── weight: [num_experts, hidden_dim]                   │
│ │   └── experts (PatchQwen3MoeExperts)                           │
│ │       ┌──────────────────────────────────────────────────┐    │
│ │       │ fully_shard(experts_mod, mesh=ep_fsdp_mesh,      │    │
│ │       │             shard_placement_fn=Shard(1))    内层 │    │
│ │       │ ├── gate_proj:  [E/ep, I/fsdp, H]               │    │
│ │       │ ├── up_proj:    [E/ep, I/fsdp, H]               │    │
│ │       │ └── down_proj:  [E/ep, H/fsdp, I]               │    │
│ │       └──────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘

参数形状演变（以 gate_proj 为例，E=128, I=4096, H=2048, ep=4, fsdp=8）:
  原始:           [128, 4096, 2048]
  → EP Shard(0):  [ 32, 4096, 2048]     (专家维度 128/4=32)
  → FSDP Shard(1):[ 32,  512, 2048]     (intermediate维度 4096/8=512)
  → 前向 all-gather 恢复: [32, 4096, 2048]  (FSDP2 前向时自动恢复)
```

### 6.11 与 FSDP1 路径的关键差异

VeOmni 同时支持 FSDP1 和 FSDP2 两条路径（`torch_parallelize.py:84-234` vs `237-435`），它们的 EP 集成方式有本质区别：

| 方面 | FSDP1 路径 | FSDP2 路径 |
|------|-----------|-----------|
| 包装方式 | 先整体 FSDP，再对 experts 单独 FSDP | 先 experts 内层，再 layer 外层 |
| 专家参数状态 | `ignored_states` 排除后单独包装 | `shard_placement_fn=Shard(1)` 正交切分 |
| Mesh 来源 | `ep_fsdp_device_mesh["ep_fsdp"]` | 同上 |
| 梯度除法 | `_gradient_postdivide_factor *= ep_size` | `set_gradient_divide_factor(world_size)` |
| API 版本 | `FullyShardedDataParallel` (torch < 2.4) | `fully_shard` (torch >= 2.4) |

---

## 第 7 节：MoE 前向传播 -- EP 通信全流程

> **一句话总结**：EP 模式下的 MoE 前向传播经历五个阶段——`expert_mask` 构造、`preprocess` 计算 splits、`token_pre_all2all` 执行 permute + all-to-all + sort、`EPGroupGemm` 执行三次 group_gemm SwiGLU 计算、`tokens_post_all2all` 执行反向 sort + all-to-all + unpermute，形成完整的 dispatch-compute-combine 流水线。

### 7.1 调用链概览

从 `PatchQwen3MoeExperts.forward()` 开始，当 `_moe_implementation == "fused"` 时调用 `fused_moe_forward()`，后者路由到 `group_gemm_fused_moe_forward()`：

```
PatchQwen3MoeExperts.forward()                    (modeling_qwen3_moe.py:81-124)
  └── fused_moe_forward()                          (fused_moe/__init__.py:29-58)
        └── group_gemm_fused_moe_forward()          (group_gemm.py:269-339)
              ├── preprocess()                      (moe_layer.py:30-69)
              ├── token_pre_all2all()               (moe_layer.py:72-99)
              ├── EPGroupGemm.apply()               (moe_layer.py:140-200)
              └── tokens_post_all2all()             (moe_layer.py:102-137)
```

### 7.2 入口：group_gemm_fused_moe_forward

`group_gemm.py:269-339`：

```python
# group_gemm.py:269-280
def group_gemm_fused_moe_forward(
    module: torch.nn.Module,
    num_experts: int,
    routing_weights: torch.Tensor,      # [num_tokens, top_k]
    selected_experts: torch.Tensor,     # [num_tokens, top_k]
    hidden_states: torch.Tensor,        # [num_tokens, hidden_dim]
    fc1_1_weight: torch.Tensor,         # gate_proj [E, I, H]
    fc1_2_weight: torch.Tensor,         # up_proj   [E, I, H]
    fc2_weight: torch.Tensor,           # down_proj [E, H, I]
):
```

当 EP 启用时（`group_gemm.py:279-328`）：

```python
if get_parallel_state().ep_enabled:
    # Step 1: 构造 expert_mask
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=num_experts).permute(2, 1, 0)
    # expert_mask shape: [num_experts, top_k, num_tokens]
```

`expert_mask` 的构造过程：

```
selected_experts: [num_tokens, top_k]  例如 [1024, 8]
  → one_hot:      [num_tokens, top_k, num_experts]  即 [1024, 8, 128]
  → permute(2,1,0): [num_experts, top_k, num_tokens]  即 [128, 8, 1024]

含义: expert_mask[e, k, t] = 1 表示 token t 的第 k 个选择是 expert e
```

### 7.3 preprocess — 计算 token 分配信息

`moe_layer.py:30-69`：

```python
def preprocess(
    expert_mask: torch.Tensor,        # [num_experts, top_k, num_tokens]
    num_experts: int,
    ep_group: dist.ProcessGroup,
) -> torch.Tensor:
    ep_size = ep_group.size()
    num_local_experts = num_experts // ep_size
    rank = dist.get_rank(ep_group)

    # 每个专家的本地 token 数量（对 top_k 和 num_tokens 维度求和）
    num_local_tokens_per_expert = expert_mask.sum(dim=(1, 2))  # [num_experts]

    # 每个 EP rank 要发送的 token 总数
    input_splits = num_local_tokens_per_expert.reshape(ep_size, num_local_experts).sum(dim=1).tolist()
    # input_splits: [ep_size] 个整数
```

**数值示例**（`ep_size=4, num_experts=128, num_local_experts=32`）：

```
num_local_tokens_per_expert = [10, 5, 8, ..., 4]  # 128 个值
                               ↑ expert 0 有 10 个 token
                                         ... expert 127 有 4 个 token

reshape to [4, 32]:
  rank 0 的专家 (E0-E31):  [10, 5, 8, ...]  → sum = 300
  rank 1 的专家 (E32-E63): [12, 3, 7, ...]  → sum = 280
  rank 2 的专家 (E64-E95): [9, 6, 11, ...]  → sum = 320
  rank 3 的专家 (E96-E127):[8, 4, 5, ...]   → sum = 260

input_splits = [300, 280, 320, 260]
含义：当前 rank 要发送给 rank 0/1/2/3 的 token 数
```

接下来，通过 `all_gather` 收集全局信息（`moe_layer.py:44-58`）：

```python
    # 收集所有 EP rank 的 token 分配信息
    num_global_tokens_per_expert = torch.zeros(
        ep_size, num_local_tokens_per_expert.size(0), ...)
    dist.all_gather_into_tensor(
        num_global_tokens_per_expert, num_local_tokens_per_expert, group=ep_group
    )
    # num_global_tokens_per_expert: [ep_size, num_experts]

    # 提取当前 rank 负责的本地专家的全局 token 信息
    start_idx, end_idx = rank * num_local_experts, (rank + 1) * num_local_experts
    num_global_tokens_per_local_expert = num_global_tokens_per_expert[:, start_idx:end_idx].contiguous()
    # [ep_size, num_local_experts]

    # 当前 rank 从各 EP rank 接收的 token 总数
    output_splits = num_global_tokens_per_local_expert.sum(dim=1).tolist()
    # [ep_size]
```

最后计算每个本地专家的全局总 token 数（`moe_layer.py:60-67`）：

```python
    # 每个本地专家的总 token 数（跨所有 EP rank 求和）
    num_global_sum_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(dim=0).to(
        torch.device("cpu"), non_blocking=True
    )
    # [num_local_experts]

    num_global_tokens_per_local_expert = num_global_tokens_per_local_expert.view(-1, num_local_experts).to(
        torch.device("cpu"), non_blocking=True
    )
```

> **NOTE**：`to("cpu", non_blocking=True)` 将 tensor 异步拷贝到 CPU，用于后续的 `sort_chunks_by_idxs` 索引计算。这是一个 D2H 传输但不会阻塞 CUDA stream。

### 7.4 token_pre_all2all — Permute + All-to-All + Sort

`moe_layer.py:72-99`：

#### Step 1：Permute — 按专家排序 token

```python
# moe_layer.py:81-86
hidden_dim = hidden_states.size(-1)
hidden_states = hidden_states.reshape(-1, hidden_dim)
org_hidden_states_shape = hidden_states.shape
routing_map = expert_mask.sum(dim=1)  # [num_experts, num_tokens]

local_permuted_hidden_states, local_input_permutation_mapping = permute(hidden_states, routing_map)
```

`permute()` 函数（`moe_utils.py:19-41`）的实现：

```python
def permute(tokens: torch.Tensor, routing_map: torch.Tensor):
    num_tokens, _ = tokens.shape
    num_experts = routing_map.shape[0]

    routing_map = routing_map.bool()
    # 创建 token 索引矩阵
    token_indices = torch.arange(num_tokens, device=routing_map.device).unsqueeze(0).expand(num_experts, -1)
    # masked_select: 按专家顺序收集 token 索引
    sorted_indices = token_indices.masked_select(routing_map)
    # index_select: 按排序后的索引重排 token
    permuted_input = tokens.index_select(0, sorted_indices)

    return permuted_input, sorted_indices
```

**数据变换可视化**：

```
routing_map ([num_experts, num_tokens]):
           token_0  token_1  token_2  token_3  token_4
expert_0:    1        0        1        0        0       → 取 token_0, token_2
expert_1:    0        1        0        0        1       → 取 token_1, token_4
expert_2:    0        0        0        1        0       → 取 token_3
...

permute 结果:
local_permuted = [token_0, token_2, token_1, token_4, token_3, ...]
                  ↑--- expert_0 ---↑ ↑--- expert_1 ---↑ ↑--expert_2--↑

sorted_indices = [0, 2, 1, 4, 3, ...]  → 用于后续 unpermute
```

> **NOTE**：`routing_map` 是对 `expert_mask` 在 top_k 维度上求和的结果。由于 top_k > 1，一个 token 可能出现在多个专家的列表中，此时该 token 会被复制多次。`routing_map[e, t]` 的值可以是 0, 1, 2 等，表示 token t 被分配给 expert e 的次数。但 `routing_map.bool()` 之后只看"是否分配"，不区分次数。

#### Step 2：All-to-All — 跨 EP rank 交换 token

```python
# moe_layer.py:88
global_permuted_hidden_states = all_to_all(
    ep_group, local_permuted_hidden_states, output_splits, input_splits
)
```

注意 splits 参数的含义：

- `output_splits`：当前 rank **接收** 的 token 数量列表
- `input_splits`：当前 rank **发送** 的 token 数量列表

```
All-to-All 通信示意（EP rank 0 视角）:

发送:                                接收:
input_splits = [300, 280, 320, 260]  output_splits = [300, 290, 310, 270]

  rank 0 ──发送 300 个──→ rank 0 （本地，无需通信）
  rank 0 ──发送 280 个──→ rank 1
  rank 0 ──发送 320 个──→ rank 2
  rank 0 ──发送 260 个──→ rank 3

  rank 0 ←─接收 300 个── rank 0 （本地）
  rank 0 ←─接收 290 个── rank 1
  rank 0 ←─接收 310 个── rank 2
  rank 0 ←─接收 270 个── rank 3

all-to-all 后 global_permuted shape: [sum(output_splits), hidden_dim]
                                     = [1170, hidden_dim]
```

#### Step 3：Sort Chunks — 从"交错"布局到"分块"布局

```python
# moe_layer.py:91-97
num_local_experts = num_experts // ep_group.size()
permute_order = torch.arange(num_experts).reshape(-1, num_local_experts).T.ravel().tolist()
global_permuted_hidden_states = sort_chunks_by_idxs(
    global_permuted_hidden_states,
    num_global_tokens_per_local_expert.ravel(),
    permute_order,
)
```

All-to-All 后数据的布局是"交错"的——来自不同 rank 的 token 按 rank 顺序排列，但同一专家的 token 分散在多个 chunk 中：

```
交错布局（all-to-all 输出）:
[rank0给E0的tokens, rank0给E1的tokens,   ← 来自 rank 0
 rank1给E0的tokens, rank1给E1的tokens,   ← 来自 rank 1
 rank2给E0的tokens, rank2给E1的tokens,   ← 来自 rank 2
 rank3给E0的tokens, rank3给E1的tokens]   ← 来自 rank 3

假设 num_experts=128, ep_size=4, num_local_experts=32
permute_order 计算:
  arange(128) = [0, 1, 2, ..., 127]
  reshape(-1, 32) → [4, 32] 矩阵:
    [[0,  1,  2,  ..., 31 ],    ← rank 0 的专家
     [32, 33, 34, ..., 63 ],    ← rank 1 的专家
     [64, 65, 66, ..., 95 ],    ← rank 2 的专家
     [96, 97, 98, ..., 127]]    ← rank 3 的专家
  .T → [32, 4] 矩阵:
    [[0,  32, 64, 96],
     [1,  33, 65, 97],
     ...]
  .ravel() → [0, 32, 64, 96, 1, 33, 65, 97, 2, 34, 66, 98, ...]

sort_chunks_by_idxs 效果:
  将 chunk[0](rank0给E0) 放到位置 0
  将 chunk[32](rank1给E0) 紧跟其后
  将 chunk[64](rank2给E0) 紧跟其后
  将 chunk[96](rank3给E0) 紧跟其后
  然后是所有 rank 给 E1 的 token...

分块布局（sort 后）:
[E0的所有tokens (来自 rank0,1,2,3),
 E1的所有tokens (来自 rank0,1,2,3),
 ...,
 E31的所有tokens (来自 rank0,1,2,3)]
```

`sort_chunks_by_idxs` 的实现（`moe_utils.py:95-99`）非常简洁：

```python
def sort_chunks_by_idxs(input: torch.Tensor, split_sizes: torch.Tensor, sorted_idxs: torch.Tensor):
    input = torch.split(input, split_sizes.tolist(), dim=0)
    output = torch.cat([input[i] for i in sorted_idxs], dim=0)
    return output
```

### 7.5 EPGroupGemm — 专家计算

专家计算由 `EPGroupGemm.apply()` 完成，详见第 9 节。这里只描述输入输出接口：

```python
# group_gemm.py:305-313
cumsum = torch.cumsum(num_global_sum_tokens_per_local_expert, dim=0).to(permute_tokens.device)
# cumsum: [num_local_experts]  例如 [110, 230, 340, ...]
# 表示前 k 个专家的累计 token 数量

final_permute_tokens = EPGroupGemm.apply(
    permute_tokens,    # [total_tokens, hidden_dim]
    cumsum,            # [num_local_experts]
    fc1_1_weight,      # gate_proj [num_local_experts, I, H]
    fc1_2_weight,      # up_proj   [num_local_experts, I, H]
    fc2_weight,        # down_proj [num_local_experts, H, I]
)
# final_permute_tokens: [total_tokens, hidden_dim]
```

### 7.6 tokens_post_all2all — Sort + All-to-All + Unpermute

`moe_layer.py:102-137`：

#### Step 1：反向 Sort — 从"分块"恢复到"交错"

```python
# moe_layer.py:116-122
num_local_experts = num_experts // ep_group.size()
unpermute_order = torch.arange(num_experts).reshape(num_local_experts, -1).T.ravel().tolist()
expert_outputs = sort_chunks_by_idxs(
    expert_outputs,
    num_global_tokens_per_local_expert.T.ravel(),
    unpermute_order,
)
```

注意与前向的 `permute_order` 对比：

```
permute_order:   arange.reshape(-1, num_local_experts).T.ravel()
                 → 将 [4, 32] 矩阵转置为 [32, 4] 再展平
                 → "交错 → 分块" 变换

unpermute_order: arange.reshape(num_local_experts, -1).T.ravel()
                 → 将 [32, 4] 矩阵转置为 [4, 32] 再展平
                 → "分块 → 交错" 变换（精确逆操作）
```

#### Step 2：反向 All-to-All

```python
# moe_layer.py:124
unpermute_outputs = all_to_all(ep_group, expert_outputs, input_splits, output_splits)
```

注意 splits 参数被交换了：

- forward: `all_to_all(output_splits, input_splits)` — 发送 `input_splits`，接收 `output_splits`
- backward: `all_to_all(input_splits, output_splits)` — 发送 `output_splits`（现在是已处理的 token），接收 `input_splits`

#### Step 3：Unpermute + 加权求和

```python
# moe_layer.py:126-135
# 生成权重索引
weights_idx = generate_weights_idx(routing_weights, selected_experts, num_experts)
# weights_idx: [num_tokens, num_experts]  — 每个 token 对每个 expert 的路由权重

unpermute_outputs = unpermute(
    unpermute_outputs,
    weights_idx,
    org_hidden_states_shape,
    local_input_permutation_mapping,
    routing_map,
)
```

`generate_weights_idx`（`moe_utils.py:75-92`）将稀疏的 `[num_tokens, top_k]` 路由权重展开为稠密的 `[num_tokens, num_experts]` 矩阵：

```python
def generate_weights_idx(routing_weights, selected_experts, num_experts):
    num_tokens, topk = routing_weights.shape
    weights_idx = torch.zeros((num_tokens, num_experts), ...)
    weights_idx.scatter_add_(1, selected_experts, routing_weights)
    return weights_idx
```

`unpermute`（`moe_utils.py:44-72`）执行加权反排列：

```python
def unpermute(tokens, routing_weights, hidden_states_shape, permutation_mapping, routing_map):
    # 从权重矩阵中提取有效权重（只取 routing_map 中为 True 的位置）
    tokens_weight = routing_weights.T.contiguous().masked_select(routing_map.bool())
    # 加权
    tokens = tokens * tokens_weight.unsqueeze(-1)
    # scatter_add: 将加权后的 token 累加回原始位置
    unpermuted_tokens = torch.zeros(hidden_states_shape, ...)
    unpermuted_tokens.scatter_add_(0, permutation_mapping.unsqueeze(1).expand(-1, hidden_dim), tokens)
    return unpermuted_tokens
```

### 7.7 EP 前向完整数据流

```
输入: hidden_states [num_tokens, H], selected_experts [num_tokens, top_k]
  │
  ├── one_hot + permute → expert_mask [num_experts, top_k, num_tokens]
  │
  ├── preprocess()
  │   ├── sum(dim=(1,2)) → num_local_tokens_per_expert [num_experts]
  │   ├── reshape + sum → input_splits [ep_size]
  │   ├── all_gather → num_global_tokens_per_expert [ep_size, num_experts]
  │   ├── slice → num_global_tokens_per_local_expert [ep_size, num_local_experts]
  │   └── sum → output_splits [ep_size]
  │
  ├── token_pre_all2all()
  │   ├── permute(hidden_states, routing_map) → sorted tokens [T_sorted, H]
  │   ├── all_to_all(ep_group, ...) → global tokens [T_global, H]
  │   └── sort_chunks_by_idxs() → blocked tokens [T_global, H]
  │                                 (按专家分组，E0的所有token | E1的所有token | ...)
  │
  ├── EPGroupGemm.apply()
  │   ├── group_gemm_same_nk: gate_proj → [T_global, I]
  │   ├── group_gemm_same_nk: up_proj → [T_global, I]
  │   ├── silu(gate) * up → [T_global, I]
  │   └── group_gemm_same_nk: down_proj → [T_global, H]
  │
  └── tokens_post_all2all()
      ├── sort_chunks_by_idxs(reverse) → interleaved [T_global, H]
      ├── all_to_all(ep_group, swapped splits) → local tokens [T_sorted, H]
      ├── generate_weights_idx() → [num_tokens, num_experts]
      └── unpermute(weighted scatter_add) → final [num_tokens, H]

输出: final_hidden_states [num_tokens, H]
```

---

## 第 8 节：All-to-All 通信原语

> **一句话总结**：VeOmni 通过自定义 `torch.autograd.Function` 封装 `dist.all_to_all_single`，反向传播时自动执行 splits 交换的逆向 All-to-All，同时提供同步和异步两种实现。

### 8.1 _AllToAll 同步实现

`comm.py:20-54`：

```python
class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input, output_split_sizes, input_split_sizes):
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes

        world_size = dist.get_world_size(group=group)
        if world_size == 1:
            return input

        input = input.contiguous()

        if output_split_sizes is None:
            output = torch.empty_like(input)
        else:
            output = torch.empty(
                size=(sum(output_split_sizes), input.size(1)),
                dtype=input.dtype, device=input.device
            )
        dist.all_to_all_single(
            output, input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )
        return output
```

**关键设计点**：

1. **`world_size == 1` 短路**：单 GPU 时直接返回输入，无需通信
2. **`input.contiguous()`**：确保输入内存连续，这是 NCCL all-to-all 的要求
3. **输出 buffer 预分配**：如果 `output_split_sizes` 已知，按实际大小 `sum(output_split_sizes)` 分配；否则分配与输入相同大小的 buffer（等分模式）

### 8.2 反向传播的对称性

```python
# comm.py:47-54
@staticmethod
def backward(ctx, *grad_output):
    return (
        None,
        _AllToAll.apply(ctx.group, *grad_output, ctx.input_split_sizes, ctx.output_split_sizes),
        None,
        None,
    )
```

**数学原理**：All-to-All 的反向传播就是 **splits 交换后的正向 All-to-All**。

```
前向：rank i 发送 input_splits[j] 个元素给 rank j
          rank i 接收 output_splits[j] 个元素从 rank j

反向：梯度流向反转
      rank i 需要将 grad_output 中来自 rank j 的部分（output_splits[j] 个）发回 rank j
      rank i 需要从 rank j 接收 input_splits[j] 个梯度

= _AllToAll.apply(grad_output, input_split_sizes, output_split_sizes)
                               ↑ 发送 sizes       ↑ 接收 sizes
                               （前向的 output 现在变成发送）
                                                   （前向的 input 现在变成接收）
```

这种对称性非常优雅——不需要额外的通信原语，反向 All-to-All 就是参数交换后的正向 All-to-All。

### 8.3 _AllToAll_Async 异步实现

`comm.py:57-92`：

```python
class _AllToAll_Async(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input, output_split_sizes, input_split_sizes):
        # ... （前半部分与同步版本相同）

        async_handle = dist.all_to_all_single(
            output, input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
            async_op=True,   # ← 关键区别：异步执行
        )
        return output, async_handle   # ← 返回 handle，需要调用方手动 wait
```

**同步 vs 异步对比**：

| 方面 | `_AllToAll` | `_AllToAll_Async` |
|------|-----------|-------------------|
| `async_op` | `False`（默认） | `True` |
| 返回值 | `output` tensor | `(output, async_handle)` 元组 |
| 阻塞行为 | 通信完成后才返回 | 立即返回，需手动 `handle.wait()` |
| 使用场景 | 标准 EP 路径 | 可用于通信-计算重叠 |

### 8.4 包装函数

`comm.py:95-101`：

```python
def all_to_all(group, input, output_split_size=None, input_split_size=None):
    return _AllToAll.apply(group, input, output_split_size, input_split_size)

def all_to_all_async(group, input, output_split_size, input_split_size):
    return _AllToAll_Async.apply(group, input, output_split_size, input_split_size)
```

### 8.5 与 TorchTitan 的对比

VeOmni 和 TorchTitan 在 All-to-All 通信上的设计差异：

| 方面 | VeOmni | TorchTitan |
|------|--------|------------|
| 底层原语 | `dist.all_to_all_single` (NCCL) | `dist.all_to_all_single` 或 DeepEP |
| Autograd 封装 | 自定义 `torch.autograd.Function` | `all_to_all_single_autograd` |
| Token 重排 | `permute()` + `sort_chunks_by_idxs()` (PyTorch 原生) | `_permute()` + `generate_permute_indices` (Triton kernel) |
| 对齐填充 | 不做对齐填充 | `TOKEN_GROUP_ALIGN_SIZE_M` 对齐 |
| DeepEP 支持 | 不支持 | 作为替代后端支持 |

**VeOmni 不做对齐填充**意味着什么？

TorchTitan 使用 `generate_permute_indices` Triton kernel 生成对齐到 8/16/32 的索引，确保每个专家的 token 数是对齐值的倍数——这对 `grouped_mm` kernel 的性能至关重要。VeOmni 使用自研的 `group_gemm_same_nk` kernel，其内部自行处理不对齐的 token 数量，因此不需要在上层做显式对齐。

---

## 第 9 节：GroupGemm 融合计算

> **一句话总结**：`EPGroupGemm` 通过自定义 `torch.autograd.Function` 将 SwiGLU MoE 的前向（3 次 `group_gemm_same_nk`）和反向（交替使用 `group_gemm_same_nk` 做 dgrad 和 `group_gemm_same_mn` 做 wgrad，加上 SiLU 激活的重计算）融合为两个高效的 kernel 调用序列。

### 9.1 EPGroupGemm 前向

`moe_layer.py:140-200`：

```python
class EPGroupGemm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        permute_tokens,    # [tokens, hidden_dim]
        cumsum,            # [local_experts]
        fc1_1_weight,      # gate_proj [local_experts, intermediate_dim, hidden_dim]
        fc1_2_weight,      # up_proj   [local_experts, intermediate_dim, hidden_dim]
        fc2_weight,        # down_proj [local_experts, hidden_dim, intermediate_dim]
    ):
```

#### 三次 group_gemm + SwiGLU 激活

```python
        # 第 1 次 group_gemm: gate_proj (fc1_1)
        # permute_tokens @ fc1_1_weight.T → [tokens, intermediate_dim]
        fc1_1_output = group_gemm_same_nk(
            a=permute_tokens, b=fc1_1_weight,
            cumsum_M=cumsum, max_M=permute_tokens.shape[0],
            transpose_a=False, transpose_b=True,
        )

        # 第 2 次 group_gemm: up_proj (fc1_2)
        fc1_2_output = group_gemm_same_nk(
            a=permute_tokens, b=fc1_2_weight,
            cumsum_M=cumsum, max_M=permute_tokens.shape[0],
            transpose_a=False, transpose_b=True,
        )

        # SwiGLU 激活
        fc1_1_activation = torch.ops.aten.silu(fc1_1_output)
        fc1_output = fc1_1_activation * fc1_2_output

        # 第 3 次 group_gemm: down_proj (fc2)
        fc2_output = group_gemm_same_nk(
            a=fc1_output, b=fc2_weight,
            cumsum_M=cumsum, max_M=permute_tokens.shape[0],
            transpose_a=False, transpose_b=True,
        )
```

**数据流**：

```
permute_tokens [T, H]
    │
    ├──── group_gemm × gate_proj.T ──→ fc1_1_output [T, I]
    │                                        │
    │                                    silu(·)
    │                                        │
    │                                 fc1_1_activation [T, I]
    │                                        │
    ├──── group_gemm × up_proj.T ───→ fc1_2_output [T, I]
    │                                        │
    │                                        × (逐元素乘)
    │                                        │
    │                                  fc1_output [T, I]
    │                                        │
    │                                 group_gemm × down_proj.T
    │                                        │
    └────────────────────────────────→ fc2_output [T, H]
```

#### cumsum 的作用

`cumsum` 是每个专家的累计 token 边界：

```
num_global_sum_tokens_per_local_expert = [110, 120, 100, 130, ...]
cumsum = [110, 230, 330, 460, ...]

含义:
  expert 0 的 token: 位置 [0, 110)
  expert 1 的 token: 位置 [110, 230)
  expert 2 的 token: 位置 [230, 330)
  ...

group_gemm_same_nk 使用 cumsum_M 来划分 batch 维度的子矩阵:
  子矩阵 0: permute_tokens[0:110, :]     × fc1_1_weight[0]
  子矩阵 1: permute_tokens[110:230, :]   × fc1_1_weight[1]
  子矩阵 2: permute_tokens[230:330, :]   × fc1_1_weight[2]
  ...
```

#### 保存用于反向的张量

```python
        ctx.save_for_backward(
            permute_tokens,    # 输入 token
            cumsum,            # 专家边界
            fc1_1_weight,      # gate_proj 权重
            fc1_2_weight,      # up_proj 权重
            fc2_weight,        # down_proj 权重
            fc1_1_output,      # gate_proj 输出（用于 silu 反向重计算）
            fc1_2_output,      # up_proj 输出（用于 SwiGLU 反向）
        )
```

> **NOTE**：这里保存了 `fc1_1_output`（silu 前的值）而非 `fc1_1_activation`（silu 后的值）。这是因为反向需要 `silu_backward(grad, input)` 而非 `silu_backward(grad, output)`。同时，`fc1_1_activation` 会在反向中重计算（`torch.ops.aten.silu(fc1_1_output)`），以节省一个 `[T, I]` 大小的 tensor 的内存开销。

### 9.2 EPGroupGemm 反向

`moe_layer.py:202-304` 实现了完整的反向传播。反向计算分为两类：

- **dgrad**（data gradient）：计算输入 token 的梯度，使用 `group_gemm_same_nk`
- **wgrad**（weight gradient）：计算权重的梯度，使用 `group_gemm_same_mn`

#### 反向 Step 1：dgrad fc1（通过 fc2 反传）

```python
# moe_layer.py:217-224
# dgrad fc1 = grad_output × fc2_weight（不转置 B，因为前向时转置了）
grad_fc1_output = group_gemm_same_nk(
    a=grad_output, b=fc2_weight,
    cumsum_M=cumsum, max_M=grad_output.shape[0],
    transpose_b=False,    # ← 前向是 transpose_b=True，反向不转置
)
```

#### 反向 Step 2：重计算 SiLU 激活

```python
# moe_layer.py:226-228
fc1_1_activation = torch.ops.aten.silu(fc1_1_output)    # 重计算！
fc1_output = fc1_1_activation * fc1_2_output
```

这里选择重计算而非保存 `fc1_1_activation`，是经典的时间-空间权衡——节省 `[T, I]` 大小的激活内存，代价是一次额外的 `silu` 计算。

#### 反向 Step 3：wgrad fc2

```python
# moe_layer.py:230-242
grad_fc2_weight = None
if fc2_weight.requires_grad:
    grad_fc2_weight = torch.empty_like(fc2_weight)
    group_gemm_same_mn(
        a=grad_output, b=fc1_output, c=grad_fc2_weight,
        cumsum_K=cumsum, max_K=grad_output.shape[0],
        transpose_a=True, transpose_b=False,
    )
```

> **NOTE**：`group_gemm_same_mn` 与 `group_gemm_same_nk` 的区别在下文 9.3 节详述。此处 `cumsum_K` 而非 `cumsum_M`，因为 wgrad 的矩阵乘法中 token 维度变成了 K（缩减维度）而非 M（批量维度）。

#### 反向 Step 4：SwiGLU 反向

SwiGLU 的反向需要处理两条路径：

```python
# moe_layer.py:244-245
# fc1_output = silu(fc1_1_output) * fc1_2_output
# 对 fc1_2_output 的梯度: grad × silu(fc1_1_output)
grad_fc1_2_output = fc1_1_activation * grad_fc1_output
# 对 silu(fc1_1_output) 的梯度: grad × fc1_2_output
grad_fc1_1_activation = grad_fc1_output * fc1_2_output
```

#### 反向 Step 5-8：两条分支的 dgrad 和 wgrad

```python
# fc1_2 分支 (up_proj):
# dgrad: grad_fc1_2_output × fc1_2_weight → grad_scatter_output_2
grad_scatter_output_2 = group_gemm_same_nk(
    a=grad_fc1_2_output, b=fc1_2_weight,
    cumsum_M=cumsum, max_M=grad_output.shape[0], transpose_b=False,
)
# wgrad: grad_fc1_2_output.T × permute_tokens → grad_fc1_2_weight
group_gemm_same_mn(
    a=grad_fc1_2_output, b=permute_tokens, c=grad_fc1_2_weight,
    cumsum_K=cumsum, max_K=grad_output.shape[0],
    transpose_a=True, transpose_b=False,
)

# SiLU 反向
grad_fc1_1_output = torch.ops.aten.silu_backward(grad_fc1_1_activation, fc1_1_output)

# fc1_1 分支 (gate_proj):
# dgrad: grad_fc1_1_output × fc1_1_weight → grad_scatter_output_1
grad_scatter_output_1 = group_gemm_same_nk(
    a=grad_fc1_1_output, b=fc1_1_weight,
    cumsum_M=cumsum, max_M=grad_output.shape[0], transpose_b=False,
)
# wgrad: grad_fc1_1_output.T × permute_tokens → grad_fc1_1_weight
group_gemm_same_mn(
    a=grad_fc1_1_output, b=permute_tokens, c=grad_fc1_1_weight,
    cumsum_K=cumsum, max_K=grad_output.shape[0],
    transpose_a=True, transpose_b=False,
)
```

#### 合并输入梯度

```python
# moe_layer.py:296
grad_permute_tokens = grad_scatter_output_1 + grad_scatter_output_2
```

两条分支的 dgrad 直接相加——因为前向时 `permute_tokens` 同时作为 `fc1_1` 和 `fc1_2` 的输入，反向梯度需要从两条路径累加。

### 9.3 group_gemm_same_nk vs group_gemm_same_mn

这两个 kernel 的命名暗示了矩阵乘法中哪些维度是"相同的"：

```
标准矩阵乘法: C[M, N] = A[M, K] × B[K, N]

group_gemm_same_nk:
  所有专家共享 N 和 K 维度（权重矩阵的维度），
  各专家的 M 不同（token 数量不同）。
  用于 dgrad：grad_output[M_i, N] × weight[N, K] → grad_input[M_i, K]
  其中 M_i = 第 i 个专家的 token 数（由 cumsum_M 划分）

group_gemm_same_mn:
  所有专家共享 M 和 N 维度（权重矩阵的维度），
  各专家的 K 不同（token 数量不同）。
  用于 wgrad：activation.T[N, K_i] × input[K_i, M] → grad_weight[N, M]
  其中 K_i = 第 i 个专家的 token 数（由 cumsum_K 划分）
```

| 属性 | `group_gemm_same_nk` | `group_gemm_same_mn` |
|------|----------------------|----------------------|
| 用途 | 前向 + dgrad | wgrad |
| 共享维度 | N, K（权重形状） | M, N（权重形状） |
| 变化维度 | M（token 数量） | K（token 数量） |
| 边界参数 | `cumsum_M` | `cumsum_K` |
| 输出形式 | 返回结果 tensor | 写入预分配的 `c` 参数 |

### 9.4 反向数据流可视化

```
grad_output [T, H]
    │
    ├──── dgrad fc1 ─────────────────────────────────────────────┐
    │   group_gemm_same_nk(grad_output, fc2_weight)              │
    │   → grad_fc1_output [T, I]                                 │
    │        │                                                    │
    │        ├── × fc1_2_output → grad_fc1_1_activation [T, I]  │
    │        │                                                    │
    │        └── × fc1_1_activation → grad_fc1_2_output [T, I]  │
    │                                                             │
    │   ┌─── silu_backward(grad_fc1_1_activation, fc1_1_output)  │
    │   │    → grad_fc1_1_output [T, I]                          │
    │   │                                                         │
    │   │    dgrad fc1_1:                                         │
    │   │    group_gemm_same_nk(grad_fc1_1_output, fc1_1_weight) │
    │   │    → grad_scatter_output_1 [T, H]                      │
    │   │                                                         │
    │   │    dgrad fc1_2:                                         │
    │   │    group_gemm_same_nk(grad_fc1_2_output, fc1_2_weight) │
    │   │    → grad_scatter_output_2 [T, H]                      │
    │   │                                                         │
    │   └──→ grad_permute_tokens = output_1 + output_2 [T, H]   │
    │                                                             │
    ├──── wgrad fc2 ─────────────────────────────────────────────┤
    │   group_gemm_same_mn(grad_output.T, fc1_output)            │
    │   → grad_fc2_weight [local_experts, H, I]                  │
    │                                                             │
    ├──── wgrad fc1_1 ───────────────────────────────────────────┤
    │   group_gemm_same_mn(grad_fc1_1_output.T, permute_tokens)  │
    │   → grad_fc1_1_weight [local_experts, I, H]                │
    │                                                             │
    └──── wgrad fc1_2 ───────────────────────────────────────────┘
        group_gemm_same_mn(grad_fc1_2_output.T, permute_tokens)
        → grad_fc1_2_weight [local_experts, I, H]
```

### 9.5 与非 EP 路径（FusedMoeExpertFunction）的对比

`group_gemm.py:23-267` 中的 `FusedMoeExpertFunction` 是非 EP 场景下的实现。两者的核心区别：

| 方面 | `EPGroupGemm` | `FusedMoeExpertFunction` |
|------|---------------|--------------------------|
| Token 分发 | 外部处理（`token_pre_all2all`） | 内部处理（`moe_scatter`/`moe_gather`） |
| 路由权重乘法 | 外部处理（`unpermute` 中） | 内部处理（`scattered_gate_weight *`） |
| 保存的张量数 | 7 个 | 13 个 |
| EP 通信 | 外部的 `all_to_all` | 无 |
| 输入格式 | 已按专家排序的 token | 原始 token + expert_index |

**`FusedMoeExpertFunction` 的特殊设计**（`group_gemm.py:82-89`）：

```python
# FusedMoeExpertFunction 在 fc1 输出上直接乘以路由权重
reshaped_gate_weight = gate_weights.reshape(-1, 1)
scattered_gate_weight = torch.empty_like(reshaped_gate_weight)
scattered_gate_weight[scatter_index.flatten()] = reshaped_gate_weight
fc1_weighted_output = fc1_activation * scattered_gate_weight
```

而 `EPGroupGemm` 将权重乘法留给了 `tokens_post_all2all` 中的 `unpermute` 函数（通过 `scatter_add_` 完成加权累加）。这是因为在 EP 场景下，权重乘法需要在 All-to-All 通信 **之后** 执行——token 从远程 rank 返回后才能与本地的路由权重相乘。

### 9.6 性能特征

`EPGroupGemm` 前向包含 3 次 `group_gemm_same_nk` + 1 次 `silu` + 1 次逐元素乘法。

反向包含：
- 4 次 `group_gemm_same_nk`（1 次 dgrad fc1 + 1 次 dgrad fc1_1 + 1 次 dgrad fc1_2 + 无需额外 dgrad fc2 因为它就是 fc1 的 dgrad）
  - 修正：实际是 3 次 dgrad（fc1 via fc2, fc1_1 via fc1_1_weight, fc1_2 via fc1_2_weight）
- 3 次 `group_gemm_same_mn`（wgrad fc2, wgrad fc1_1, wgrad fc1_2）
- 1 次 `silu` 重计算 + 1 次 `silu_backward`
- 3 次逐元素乘法 + 1 次逐元素加法

反向的计算量约为前向的 3 倍（3 次 dgrad + 3 次 wgrad vs 前向 3 次），这符合深度学习训练中"反向约等于 2-3 倍前向"的经验法则。

---


## 第 10 节：反向传播与梯度同步

> **一句话总结**：反向传播时 All-to-All 通信自动执行 splits 交换的逆向操作，FSDP1 通过修改 `_gradient_postdivide_factor` 乘以 `ep_size` 补偿专家参数的梯度均值，FSDP2 通过 `set_gradient_divide_factor(world_size)` 实现等效修正，两条路径各自配备 EP-aware 的梯度裁剪。

### 10.1 All-to-All 反向 — 通信流反转

第 8 节分析了 `_AllToAll` 的反向实现（`comm.py:47-54`），这里从全链路角度梳理反向传播中的通信模式。

**反向数据流**：

```
反向顺序（与前向相反）：

前向: permute → all_to_all → sort → EPGroupGemm → unsort → all_to_all → unpermute
                                                                     ↑
反向: unpermute_bwd ← a2a_bwd ← unsort_bwd ← EPGroupGemm_bwd ← sort_bwd ← a2a_bwd ← permute_bwd
```

具体到 `_AllToAll.backward`（`comm.py:47-54`）：

```python
@staticmethod
def backward(ctx, *grad_output):
    return (
        None,
        _AllToAll.apply(ctx.group, *grad_output, ctx.input_split_sizes, ctx.output_split_sizes),
        None,
        None,
    )
```

**数学原理**：All-to-All 的反向就是 splits 交换后的正向 All-to-All。前向中 rank i 发送 `input_splits[j]` 个元素给 rank j、接收 `output_splits[j]` 个元素；反向时梯度方向反转，rank i 需要将 `output_splits[j]` 个梯度发回 rank j，从 rank j 接收 `input_splits[j]` 个梯度。

**反向中的两次 All-to-All**：

```
反向传播完整通信序列：

  EPGroupGemm.backward 输出: grad_permute_tokens [T_global, H]
      │
      ├── (1) unsort_bwd: sort_chunks_by_idxs(reverse)
      │       → 从"按专家分块"恢复到"按 rank 交错"排列
      │
      ├── (2) all_to_all_bwd(第一次反向 A2A):
      │       → 交换 splits: 用前向的 (input_splits, output_splits) 反转
      │       → 梯度从远程 rank 返回本地
      │
      ├── (3) permute_bwd: unpermute 梯度回原始 token 位置
      │
      └── (4) 加权求和的反向: 路由权重与梯度的逐元素乘法
```

### 10.2 FSDP1 梯度修正 — `_gradient_postdivide_factor`

`torch_parallelize.py:211-213` 是 FSDP1 路径下梯度修正的关键代码：

```python
# torch_parallelize.py:211-213
fsdp_module = FullyShardedDataParallel(no_shard_module, **fsdp_kwargs)
fsdp_state = _get_module_fsdp_state_if_fully_sharded_module(fsdp_module)
fsdp_state._gradient_postdivide_factor *= parallel_state.ep_size
```

**为什么需要这个修正？** 用数值推导说明：

```
假设 world_size=16, ep_size=4, ep_fsdp_size=4

普通参数的梯度归约：
  FSDP reduce-scatter 跨 dp_world_size=16 个 rank
  → 自动除以 16
  → 等效梯度均值 = sum(grads) / 16  ✓

专家参数（不修正）的梯度归约：
  EP 将专家切到 4 个 rank，每个 rank 持有 1/4 的专家
  该专家参数的 FSDP reduce-scatter 只跨 ep_fsdp_size=4 个 rank
  → 自动除以 4
  → 等效梯度均值 = sum(grads_on_ep_fsdp_group) / 4  ✗
  → 但我们需要 sum(all_grads) / 16

修正方法：
  fsdp_state._gradient_postdivide_factor *= ep_size (=4)
  → 新的除法因子 = 4 * 4 = 16 = world_size  ✓
```

**更直观的理解**：

```
world_size = ep_size * ep_fsdp_size

                    ┌──── ep_fsdp_size=4 ────┐
                    │  R0   R1   R2   R3     │  ← 持有相同专家的 rank
                    │  R4   R5   R6   R7     │  ← 持有另一组专家
ep_size=4          │  R8   R9   R10  R11    │
                    │  R12  R13  R14  R15    │
                    └────────────────────────┘

专家 E0-E7 在 R0-R3 上，FSDP 在 R0/R4/R8/R12 之间 reduce-scatter
但 R0/R4/R8/R12 各自的 E0-E7 梯度只来自本地的数据
要得到全局均值，需要除以 world_size=16 而非 ep_fsdp_size=4
所以 postdivide_factor 需要乘以 ep_size=4
```

### 10.3 FSDP2 梯度修正 — `set_gradient_divide_factor`

`torch_parallelize.py:366-377` 是 FSDP2 路径的等效修正：

```python
# torch_parallelize.py:366-377
if parallel_state.ep_enabled and experts_mod is not None:
    fully_shard(experts_mod, **expert_fsdp_kwargs)
    gradient_divide_factor = parallel_state.ep_gradient_divide_factor
    logger.info(f"setting grad divide factor for ep module to {gradient_divide_factor}")
    if IS_NPU_AVAILABLE:
        # NPU is using torch 2.7
        experts_mod.set_reduce_scatter_divide_factor(gradient_divide_factor)
    else:
        # from torch 2.8
        experts_mod.set_gradient_divide_factor(gradient_divide_factor)
```

**`ep_gradient_divide_factor` 的计算**（`parallel_state.py:349-357`）：

```python
# parallel_state.py:348-357
@property
def ep_gradient_divide_factor(self) -> int:
    # We assume the world size is the total dp size by now
    # TP and PP would make this assumption not true
    assert self.tp_size == 1
    assert self.pp_size == 1
    # For ep+fsdp2, the grad divide factor should always be world size
    # SP does not affect this since SP groups still replicate params
    # and their grads are all-reduced which would match grads for the same data without SP.
    return self.world_size
```

**设计决策对比**：

| 方面 | FSDP1 | FSDP2 |
|------|-------|-------|
| 修正方式 | `_gradient_postdivide_factor *= ep_size` | `set_gradient_divide_factor(world_size)` |
| 修正时机 | FSDP 包装后修改内部状态 | FSDP 包装后通过公开 API 设置 |
| 修正值 | 间接：原始值 * ep_size | 直接：world_size |
| TP/PP 约束 | 无显式约束 | `assert self.tp_size == 1 and self.pp_size == 1` |
| NPU 兼容 | 统一逻辑 | `set_reduce_scatter_divide_factor`（torch 2.7）vs `set_gradient_divide_factor`（torch 2.8+） |

> **NOTE**：FSDP2 的实现显式假设 `tp_size == 1` 和 `pp_size == 1`。当 TP 或 PP 启用时，`world_size` 不再等于 `dp_size`，此时 `ep_gradient_divide_factor` 的值需要调整。这是 VeOmni 当前的已知限制。

### 10.4 EP-aware 梯度裁剪 — FSDP1 路径

`fsdp/clip_grad_norm.py:15-133` 实现了 EP-aware 的梯度裁剪。核心思路是将参数分为三组，对每组采用不同的归约策略。

#### 参数分组（`clip_grad_norm.py:28-51`）

```python
# fsdp/clip_grad_norm.py:28-51
sharded_params_for_gnorm = {}           # 普通 FSDP 参数
ep_fsdp_sharded_params_for_gnorm = {}   # EP 专家参数
nonsharded_params_for_gnorm = {}        # 未切分参数
ep_fsdp_process_group = None

for handle in fsdp_model._all_handles:
    for param in handle.flat_param._params:
        spec_info: SpecInfo = param.spec_info
        # ep param
        if isinstance(spec_info.placement, Shard):
            if ep_fsdp_process_group is None:
                ep_fsdp_process_group = handle.process_group
            ep_fsdp_sharded_params_for_gnorm.setdefault(param, None)
        # fsdp param
        else:
            sharded_params_for_gnorm.setdefault(param, None)
```

**判断逻辑**：通过 `spec_info.placement` 区分——EP 参数的 placement 是 `Shard(0)`（专家维度切分），普通参数的 placement 是 `Replicate`。

#### 两阶段归约（`clip_grad_norm.py:82-99`）

```python
# fsdp/clip_grad_norm.py:92-99
total_norm = local_sharded_norm**norm_type
dist.all_reduce(total_norm, group=fsdp_model.process_group)       # (1) FSDP group 内归约

if local_ep_fsdp_sharded_norm is not None:
    total_ep_fsdp_sharded_norm = local_ep_fsdp_sharded_norm**norm_type
    dist.all_reduce(total_ep_fsdp_sharded_norm, group=ep_fsdp_process_group)  # (2a) ep_fsdp group 内归约
    dist.all_reduce(total_ep_fsdp_sharded_norm, group=ep_group)               # (2b) ep group 内归约
    total_norm += total_ep_fsdp_sharded_norm
```

**两阶段归约的原因**：

```
普通参数: FSDP 切分在 fsdp_model.process_group（= 全部 dp rank）上
  → 一次 all-reduce 即可获得全局 norm²

EP 参数: FSDP 切分在 ep_fsdp_process_group（= ep_fsdp_size 个 rank）上
  → 第一次 all-reduce 在 ep_fsdp group 内：合并同一组 EP rank 内的切片梯度
  → 第二次 all-reduce 在 ep group 内：合并不同专家组之间的梯度范数
  → 两步合起来 = 全局 norm²
```

**数值示例**：

```
world_size=16, ep_size=4, ep_fsdp_size=4

ep_fsdp group: {R0, R4, R8, R12}  — 持有相同专家，FSDP 切分在此组
ep group:      {R0, R1, R2, R3}   — 同一 ep_fsdp 位置的不同 EP rank

Step (2a): R0 与 R4/R8/R12 归约 → 得到 E0-E7 在 4 个数据分片上的 norm²
Step (2b): R0 与 R1/R2/R3 归约 → 加上 E8-E15, E16-E23, E24-E31 的 norm²
→ 最终得到所有专家在所有数据上的全局 norm²
```

### 10.5 EP-aware 梯度裁剪 — FSDP2 路径

`fsdp2/clip_grad_norm.py:21-99` 提供了 FSDP2 版本的 EP-aware 梯度裁剪。

#### 入口分发（`fsdp2/clip_grad_norm.py:21-43`）

```python
# fsdp2/clip_grad_norm.py:21-43
def clip_grad_norm(model, max_norm, norm_type=2.0, ...):
    # EP-aware path (FSDP2 + EP)
    if hasattr(model, "_ep_param_groups"):
        return ep_fsdp2_clip_grad_norm(model, max_norm, ...)
    # Standard path
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm, ...)
    if isinstance(grad_norm, DTensor):
        grad_norm = grad_norm.full_tensor()
    return grad_norm
```

**分发条件**：通过 `model._ep_param_groups` 属性判断是否启用了 EP。这个属性在 EP 初始化时被设置到模型上。

#### `ep_fsdp2_clip_grad_norm` 核心逻辑（`fsdp2/clip_grad_norm.py:47-99`）

```python
# fsdp2/clip_grad_norm.py:60-88
ps = get_parallel_state()
fsdp_group = ps.fsdp_group
ep_group = ps.ep_group if ps.ep_enabled else None
ep_fsdp_group = ps.ep_fsdp_device_mesh["ep_fsdp"].get_group() if ps.ep_enabled else None

# 参数分组
ep_params = [p for p in model._ep_param_groups.get("ep", []) if p.grad is not None]
non_ep_params = [p for p in model._ep_param_groups.get("non_ep", []) if p.grad is not None]

# 普通参数: 在 fsdp_group 上归约
non_ep_total = _fsdp2_reduce_group(
    params=non_ep_params, norm_type=norm_type,
    reduce_groups=[("fsdp", fsdp_group)],
)

# EP 参数: 先在 ep_fsdp group 归约，再在 ep group 归约
ep_total = _fsdp2_reduce_group(
    params=ep_params, norm_type=norm_type,
    reduce_groups=[("ep_fsdp", ep_fsdp_group), ("ep", ep_group)],
)
```

#### `_fsdp2_reduce_group` 通用归约函数（`fsdp2/clip_grad_norm.py:146-171`）

```python
# fsdp2/clip_grad_norm.py:146-171
def _fsdp2_reduce_group(params, norm_type, reduce_groups):
    if math.isinf(norm_type):
        val = _local_max(params)
        for _, group in reduce_groups:
            if group is not None:
                dist.all_reduce(val, op=dist.ReduceOp.MAX, group=group)
        return val
    else:
        p = float(norm_type)
        val = _local_pth_sum(params, p)
        for name, group in reduce_groups:
            if group is not None:
                dist.all_reduce(val, op=dist.ReduceOp.SUM, group=group)
        return val
```

**设计亮点**：`_fsdp2_reduce_group` 接受 `reduce_groups` 列表，按顺序执行多次 `all_reduce`。非 EP 参数传入 `[("fsdp", fsdp_group)]`（一次归约），EP 参数传入 `[("ep_fsdp", ep_fsdp_group), ("ep", ep_group)]`（两次顺序归约）。这种设计将两阶段归约的逻辑抽象为统一接口。

#### DTensor 梯度处理（`fsdp2/clip_grad_norm.py:103-121`）

```python
# fsdp2/clip_grad_norm.py:103-121
def _local_pth_sum(params, p):
    grads = [p.grad for p in params if p.grad is not None]
    grads_local = [
        g.to_local().detach().to(torch.float32) if isinstance(g, DTensor)
        else g.detach().to(torch.float32)
        for g in grads
    ]
    # ... foreach norm computation ...
```

**关键细节**：FSDP2 中的梯度可能是 `DTensor`（分布式张量），需要先调用 `to_local()` 获取本地切片再计算范数。同时强制转为 `float32` 避免低精度累加误差。

### 10.6 FSDP1 vs FSDP2 梯度裁剪对比

```
FSDP1 路径 (fsdp/clip_grad_norm.py):
  参数分组方式：遍历 fsdp_model._all_handles → 检查 param.spec_info.placement
  归约方式：
    普通: all_reduce(fsdp_model.process_group)
    EP:   all_reduce(ep_fsdp_process_group) + all_reduce(ep_group)
  裁剪方式：手动 grad.mul_(clip_coef_clamped)

FSDP2 路径 (fsdp2/clip_grad_norm.py):
  参数分组方式：model._ep_param_groups 字典（预先分好组）
  归约方式：
    普通: _fsdp2_reduce_group([("fsdp", fsdp_group)])
    EP:   _fsdp2_reduce_group([("ep_fsdp", ep_fsdp_group), ("ep", ep_group)])
  裁剪方式：torch.nn.utils.clip_grads_with_norm_()（PyTorch 原生 API）
```

---

## 第 11 节：Prefetch 策略

> **一句话总结**：EP 使 MoE 层产生多个独立的 FSDP 单元（experts 单独包装），VeOmni 通过 `_fsdp_modules` 列表收集每层内所有 FSDP 单元，并在相邻 block 之间建立显式 prefetch 链，确保通信-计算重叠不被 EP 的存在所破坏。

### 11.1 为什么 EP 需要手动 Prefetch

标准 FSDP 在 `TransformerBlock` 级别包装时，只有一个 FSDP 单元，CUDA stream 上的隐式 prefetch 自然生效。但 EP 引入了内层 FSDP 包装：

```
无 EP:
  Block N (单个 FSDP 单元) → 隐式 prefetch → Block N+1 (单个 FSDP 单元)
  ✓ 正常工作

有 EP:
  Block N:
    ├── FSDP(block)           ← 外层 FSDP 单元
    └── FSDP(experts)         ← 内层 FSDP 单元（独立的 all-gather）

  FSDP 只能隐式 prefetch 同级别的下一个 FSDP 单元
  → Block N 的外层 FSDP 不知道还要 prefetch Block N+1 的内层 experts FSDP
  → experts 的 all-gather 只能在需要时才发起，错过了与前一层计算重叠的窗口
```

此外，VeOmni 的 FSDP2 路径中还可能存在 `mp_ignored_classes`（如 TopK gate 需要 FP32 精度的模块），这些模块也会被单独 `fully_shard`，进一步增加 FSDP 单元数量。

### 11.2 `_fsdp_modules` 列表构造

`torch_parallelize.py:359-389`：

```python
# torch_parallelize.py:359-389
for layer_fqn, layer_mod, experts_mod in layer_pairs:
    # 收集当前 decoder layer 内的所有 FSDP 单元
    layer_mod._fsdp_modules = []

    # (1) EP 专家模块（内层 FSDP）
    if parallel_state.ep_enabled and experts_mod is not None:
        fully_shard(experts_mod, **expert_fsdp_kwargs)
        experts_mod.set_gradient_divide_factor(...)
        layer_mod._fsdp_modules.append(experts_mod)

    # (2) 高精度模块（如 TopK gate）
    if mp_ignored_classes:
        for sub_mod in layer_mod.modules():
            if isinstance(sub_mod, mp_ignored_classes) and sub_mod is not layer_mod:
                fully_shard(sub_mod, **fsdp_kwargs_without_mp)
                layer_mod._fsdp_modules.append(sub_mod)

    # (3) Decoder layer 本身（外层 FSDP）
    fully_shard(layer_mod, **fsdp_kwargs)
    layer_mod._fsdp_modules.append(layer_mod)
```

**`_fsdp_modules` 列表的顺序**：

```
典型 MoE decoder layer 的 _fsdp_modules:
  [experts_mod, gate_mod(可选), layer_mod]
   ↑ 内层          ↑ 高精度       ↑ 外层

包装顺序（从内到外）决定了列表顺序：
  先包装 experts → 再包装 gate → 最后包装整个 layer
  → _fsdp_modules = [experts, gate, layer]
```

### 11.3 前向 Prefetch 配置

`torch_parallelize.py:393-402`：

```python
# torch_parallelize.py:393-402
need_manual_prefetch = parallel_state.ep_enabled or mp_ignored_classes is not None
if need_manual_prefetch:
    blocks = [pair[1] for pair in layer_pairs]
    next_blocks = blocks[1:] + [None]
    for current_block, next_block in zip(blocks, next_blocks):
        if next_block is not None:
            prefetch_modules = next_block._fsdp_modules
            # prefetch in order of attn, gate, experts
            current_block.set_modules_to_forward_prefetch(list(reversed(prefetch_modules)))
```

**关键细节**：`reversed(prefetch_modules)` 反转了列表顺序。原始列表是 `[experts, gate, layer]`，反转后变为 `[layer, gate, experts]`。这意味着 prefetch 的提交顺序是：

```
当前 Block N forward 完成后，按以下顺序提交 prefetch:
  1. next_block (layer)    ← 外层 FSDP 的 all-gather
  2. next_block.gate       ← gate 的 all-gather (如果有)
  3. next_block.experts    ← 内层 FSDP 的 all-gather
```

**为什么要反转？** `set_modules_to_forward_prefetch` 按传入列表的顺序提交 all-gather 请求。由于 FSDP 的 forward 先执行外层（layer），再进入内层子模块，所以外层的 all-gather 应该先完成。将 `layer` 放在列表最前面（即最先 prefetch），确保外层参数在需要时已经就绪。

### 11.4 反向 Prefetch 配置

`torch_parallelize.py:404-410`：

```python
# torch_parallelize.py:404-410
# configure backward prefetch
rev_blocks = list(reversed(blocks))
prev_blocks = rev_blocks[1:] + [None]
for current_block, prev_block in zip(rev_blocks, prev_blocks):
    if prev_block is not None:
        prefetch_modules = prev_block._fsdp_modules
        current_block.set_modules_to_backward_prefetch(list(reversed(prefetch_modules)))
```

**反向 Prefetch 链**（与前向镜像）：

```
反向执行顺序: Block N-1 → Block N-2 → ... → Block 0

Block N-1 backward 完成后:
  → prefetch Block N-2 的 [layer, gate, experts]

Block N-2 backward 完成后:
  → prefetch Block N-3 的 [layer, gate, experts]

...

Block 1 backward 完成后:
  → prefetch Block 0 的 [layer, gate, experts]

Block 0 backward 完成后:
  → prefetch: None (最后一层)
```

### 11.5 与 TorchTitan Prefetch 策略的对比

| 方面 | VeOmni | TorchTitan |
|------|--------|------------|
| 触发条件 | `ep_enabled or mp_ignored_classes` | `ep_degree > 1` |
| FSDP 单元收集 | 动态 `_fsdp_modules` 列表 | 硬编码 `[block, block.moe.experts]` |
| Prefetch 顺序 | `reversed(_fsdp_modules)` = `[layer, gate, experts]` | `[block, block.moe.experts]` |
| 非 MoE 层处理 | `_fsdp_modules = [layer_mod]`（自然兼容） | 单独处理 `[next_block]` |
| 高精度模块支持 | 自动纳入 `_fsdp_modules` | 不支持（无类似机制） |
| `tok_embeddings` prefetch | 不单独处理 | 显式 `model.tok_embeddings.set_modules_to_forward_prefetch([blocks[0]])` |

**VeOmni 的优势**：通过动态构建 `_fsdp_modules` 列表，VeOmni 能够自动适配不同模型架构。无论一个 decoder layer 内部有多少个独立的 FSDP 单元，prefetch 链都会正确建立。TorchTitan 的硬编码方式更高效但灵活性较低。

### 11.6 Prefetch 时序图

以包含 EP 和高精度 gate 的场景为例：

```
前向时序:

Block N-1: [attention compute] [ffn compute]
                                └── prefetch Block N:
                                    (1) all-gather layer params
                                    (2) all-gather gate params
                                    (3) all-gather experts params
                                             ↓
Block N:   [────── attention ──────] [gate] [all_to_all] [experts] [all_to_all] [combine]
           ↑ params 已就绪            ↑ ok   ↑ EP 通信     ↑ ok

如果没有 prefetch:
Block N:   [wait all-gather layer] [attention] [wait all-gather experts] [experts]...
           ↑ 被阻塞！                           ↑ 再次被阻塞！
```

---

## 第 12 节：Checkpoint 管理

> **一句话总结**：`CheckpointExtensions(FSDPExtensions)` 类通过 `state_dict_post_hook` 在保存时给 EP 参数追加 EP 维度的 DTensor 元信息，通过 `load_state_dict_pre_hook` 在加载时将 DTensor 还原为本地 tensor，并通过 3 个 monkey-patch 覆盖 FSDP 的 optimizer state dict 序列化逻辑以正确处理 EP 参数。

### 12.1 问题背景

EP 参数在训练时是 DTensor（带 `Shard(0)` placement 的分布式张量），但 FSDP 内部的 checkpoint 保存/加载流程不感知 EP 的存在。具体问题：

```
保存时：
  FSDP.state_dict() 输出 DTensor 状态，但只包含 FSDP 的 placement 信息
  → 缺少 EP 维度的 Shard(0) 信息
  → 恢复时无法知道这个参数需要按专家维度切分

加载时：
  FSDP.load_state_dict() 接收的 state dict 是 DTensor
  → 但 EP 参数需要先按 EP 维度切片为本地 tensor
  → 再由 FSDP 做 FSDP 维度的切片
  → 两步切片的顺序和 mesh 必须正确

Optimizer state:
  FSDP.optim_state_dict() 同样不感知 EP
  → optimizer 中的 exp_avg, exp_avg_sq 等也需要 EP-aware 的处理
```

### 12.2 CheckpointExtensions 类

`fsdp/extension.py:124-411` 定义了 `CheckpointExtensions`，继承自 PyTorch 的 `FSDPExtensions`：

```python
# fsdp/extension.py:124-133
class CheckpointExtensions(FSDPExtensions):
    def __init__(
        self,
        ep_fsdp_device_mesh: DeviceMesh,
        fqn2spec_info: Dict[str, SpecInfo],
    ):
        super().__init__()
        self.ep_fsdp_device_mesh = ep_fsdp_device_mesh
        self.ep_mesh = ep_fsdp_device_mesh["ep"] if ep_fsdp_device_mesh is not None else None
        self.fqn2spec_info = fqn2spec_info
```

**`fqn2spec_info`** 是一个字典，将参数的 fully-qualified name（如 `model.layers.0.mlp.experts.gate_proj`）映射到 `SpecInfo`，其中包含该参数的 EP placement（`Shard(0)`）和对应的 EP mesh。这个映射由第 4 节分析的 `ParallelPlan.apply()` 生成。

### 12.3 state_dict_post_hook — 保存时追加 EP 信息

`fsdp/extension.py:192-215`：

```python
# fsdp/extension.py:192-215
@torch.no_grad()
def state_dict_post_hook(self, module, state_dict, prefix, local_metadata, fqn2spec_info):
    if self.ep_mesh is None:
        return
    global_device_mesh = self.ep_fsdp_device_mesh
    assert global_device_mesh.ndim == 2

    keys = list(state_dict.keys())
    for name in sorted(keys):
        if name in fqn2spec_info and isinstance(fqn2spec_info[name].placement, Shard):
            cur_spec_info = fqn2spec_info[name]
            tensor = state_dict[name]
            tensor = _shard_tensor(tensor, cur_spec_info.ep_fsdp_mesh, cur_spec_info.placement)
            state_dict[name] = tensor
```

**`_shard_tensor` 的作用**（`fsdp/extension.py:43-62`）：

```python
# fsdp/extension.py:43-62
def _shard_tensor(orgin_tensor, device_mesh, shard=Shard(0)):
    assert device_mesh.ndim == 2
    ep_mesh = device_mesh["ep"]

    if orgin_tensor.__class__.__name__ == "DTensor":
        # 已经是 DTensor（来自 FSDP），追加 EP 维度的 placement
        placements = (shard,) + orgin_tensor.placements
        dtensor = DTensor.from_local(orgin_tensor._local_tensor, device_mesh=device_mesh, placements=placements)
    elif orgin_tensor.__class__.__name__ == "Tensor":
        # 普通 tensor，包装为带 EP placement 的 DTensor
        dtensor = DTensor.from_local(orgin_tensor, device_mesh=ep_mesh, placements=[shard])

    return dtensor
```

**数据变换过程**：

```
保存前 state_dict 中的 EP 参数:
  name="model.layers.0.mlp.experts.gate_proj"
  tensor = DTensor(
      local_data=[32, I, H],              ← 本地持有 32 个专家的权重
      device_mesh=ep_fsdp_mesh["ep_fsdp"], ← FSDP 的 1D mesh
      placements=[Shard(1)]                ← FSDP 在 dim1 上切分
  )

保存后:
  tensor = DTensor(
      local_data=[32, I, H],
      device_mesh=ep_fsdp_mesh,            ← 完整的 2D mesh (ep, ep_fsdp)
      placements=[Shard(0), Shard(1)]      ← EP 在 dim0 + FSDP 在 dim1
  )
```

### 12.4 load_state_dict_pre_hook — 加载时还原本地 tensor

`fsdp/extension.py:217-252`：

```python
# fsdp/extension.py:217-252
@torch.no_grad()
def load_state_dict_pre_hook(self, state_dict, prefix, ...):
    if self.ep_mesh is None:
        return
    if self.ep_mesh.size() != self.ep_fsdp_device_mesh.size():
        return  # EP mesh 与全局 mesh 不一致时跳过

    keys = list(state_dict.keys())
    for name in sorted(keys):
        tensor = state_dict[name]
        if check_any_unflat_param_names_match(name, fqn2spec_info, "_fsdp_wrapped_module"):
            fqn = name.split("_fsdp_wrapped_module.")[-1]
            cur_spec_info = fqn2spec_info[fqn]
            tensor = _shard_dtensor(tensor, cur_spec_info.ep_fsdp_mesh, cur_spec_info.placement)
            state_dict[name] = tensor
```

**`_shard_dtensor` 的作用**（`fsdp/extension.py:65-81`）：

```python
# fsdp/extension.py:65-81
def _shard_dtensor(orgin_dtensor, device_mesh, shard=Shard(0)):
    assert isinstance(orgin_dtensor, DTensor)
    local_tensor = orgin_dtensor.to_local()
    return local_tensor
```

**加载流程**：checkpoint 中的 DTensor（带 EP+FSDP 双重 placement）通过 `to_local()` 还原为本地 tensor。FSDP 的 `load_state_dict` 会在之后处理 FSDP 维度的重新分配。

### 12.5 三个 Optimizer State Dict Monkey Patches

EP 参数的 optimizer state（如 Adam 的 `exp_avg`、`exp_avg_sq`）同样需要 EP-aware 的保存/加载。VeOmni 通过三个 monkey-patch 覆盖 FSDP 的原生实现：

#### Patch 1：`_convert_state_with_flat_params`（`fsdp/extension.py:254-323`）

```python
# fsdp/extension.py:320-323
# monkey patch
torch.distributed.fsdp._optim_utils._convert_state_with_flat_params = partial(
    _convert_state_with_flat_params_patch, fqn2spec_info=self.fqn2spec_info
)
```

**修改逻辑**：当检测到参数是 EP 参数时（通过 `check_all_unflat_param_names_match`），在 `_unflatten_optim_state` 调用中将 `shard_state=False`。这是因为 EP 参数的 optimizer state 不应被 FSDP 的标准切片逻辑处理——它们的切片已经由 EP 维度决定。

#### Patch 2：`optim_state_dict`（`fsdp/extension.py:325-358`）

```python
# fsdp/extension.py:357-358
FSDP.optim_state_dict = staticmethod(
    partial(fsdp_optim_state_post_patch_fn, fqn2spec_info=self.fqn2spec_info)
)
```

**修改逻辑**：保存 optimizer state 时，先调用原始的 `orig_optim_state_dict`，然后遍历结果中的 EP 参数，给 optimizer state 的每个 tensor 值（排除 `step` 等标量）追加 EP 维度的 `Shard` placement。

```python
# fsdp/extension.py:345-354
for fqn in sorted(optim_state["state"].keys()):
    if fqn in fqn2spec_info and isinstance(fqn2spec_info[fqn].placement, Shard):
        cur_spec_info = fqn2spec_info[fqn]
        fqn_state = {}
        for key, val in optim_state["state"][fqn].items():
            if key not in OPTIM_STATE_NO_SHARD_KEY:    # 跳过 "step" 等标量
                val = _shard_tensor(val, cur_spec_info.ep_fsdp_mesh, cur_spec_info.placement)
            fqn_state[key] = val
        optim_state["state"][fqn] = fqn_state
```

#### Patch 3：`optim_state_dict_to_load`（`fsdp/extension.py:360-411`）

```python
# fsdp/extension.py:408-411
FSDP.optim_state_dict_to_load = staticmethod(
    partial(optim_state_dict_to_load_pre_patch_fn, fqn2spec_info=self.fqn2spec_info)
)
```

**修改逻辑**：加载 optimizer state 时，先将 EP 参数的 optimizer state DTensor 通过 `_shard_dtensor` 还原为本地 tensor，再调用原始的 `orig_optim_state_dict_to_load`。

### 12.6 register_checkpoint_extension — 完整注册流程

`fsdp/extension.py:414-451` 将所有钩子和 patch 注册到 FSDP 模型上：

```python
# fsdp/extension.py:414-451
def register_checkpoint_extension(fsdp_model, save_hook_mesh=None, fqn2spec_info=None):
    extension = CheckpointExtensions(
        ep_fsdp_device_mesh=save_hook_mesh,
        fqn2spec_info=fqn2spec_info,
    )

    # 1. 为所有 FSDP 子模块注册 extension
    for fsdp_module in FSDP.fsdp_modules(fsdp_model):
        fsdp_module._fsdp_extension = extension
        fsdp_module._handle._fsdp_extension = extension
    fsdp_model._fsdp_extension = extension
    fsdp_model._handle._fsdp_extension = extension

    # 2. 注册 state dict hooks
    if fqn2spec_info is not None:
        state_dict_post_hook_fn = partial(extension.state_dict_post_hook, fqn2spec_info=fqn2spec_info)
        fsdp_model._register_state_dict_hook(state_dict_post_hook_fn)

        load_state_dict_pre_hook_fn = partial(extension.load_state_dict_pre_hook, fqn2spec_info=fqn2spec_info)
        fsdp_model._register_load_state_dict_pre_hook(load_state_dict_pre_hook_fn)

        # 3. 应用 3 个 monkey patches
        extension.patch_convert_state_with_flat_params()
        extension.patch_fsdp_optim_state_dict()
        extension.patch_fsdp_optim_state_dict_to_load()
```

> **NOTE**：`register_checkpoint_extension` 是 FSDP1 路径特有的逻辑（在 `parallelize_model_fsdp1` 的末尾调用）。FSDP2 路径使用 PyTorch 原生的 `DCP`（Distributed Checkpoint）机制，不需要这些 monkey-patch。

### 12.7 Checkpoint 数据流可视化

```
保存流程:
  model.state_dict()
    │
    ├── FSDP 内部: 将 flat_param 解包为原始参数，得到 FSDP DTensor
    │
    ├── state_dict_post_hook:
    │   ├── 非 EP 参数: 不处理
    │   └── EP 参数: _shard_tensor() → 追加 EP placement
    │       DTensor(ep_fsdp_mesh["ep_fsdp"], [Shard(1)])
    │       → DTensor(ep_fsdp_mesh, [Shard(0), Shard(1)])
    │
    └── 输出: state_dict (带完整分布式元信息的 DTensor)

加载流程:
  model.load_state_dict(state_dict)
    │
    ├── load_state_dict_pre_hook:
    │   ├── 非 EP 参数: 不处理
    │   └── EP 参数: _shard_dtensor() → 还原为本地 tensor
    │       DTensor(ep_fsdp_mesh, [Shard(0), Shard(1)])
    │       → local Tensor [num_local_experts, I, H]
    │
    └── FSDP 内部: 将本地 tensor 重新切片为 FSDP DTensor

Optimizer 保存/加载:
  FSDP.optim_state_dict() / FSDP.optim_state_dict_to_load()
    │
    ├── Patch 1 (_convert_state_with_flat_params):
    │   EP 参数的 optimizer state 不做 FSDP 标准切片
    │
    ├── Patch 2 (optim_state_dict):
    │   保存后给 EP 参数的 exp_avg/exp_avg_sq 追加 EP placement
    │
    └── Patch 3 (optim_state_dict_to_load):
        加载前将 EP 参数的 optimizer state DTensor 还原为本地 tensor
```

---

## 第 13 节：端到端示例 -- Qwen3-MoE-30B on 16 GPUs

> **一句话总结**：以 Qwen3-MoE-30B-A3B（128 专家、top-8 路由、48 层）在 16 GPU 上的 EP+FSDP 部署为例，计算完整的 mesh 构成、参数切分、通信量和显存占用。

### 13.1 模型配置

Qwen3-MoE-30B-A3B 的核心参数：

| 参数 | 值 |
|------|---|
| `num_hidden_layers` | 48 |
| `hidden_size` (H) | 2048 |
| `intermediate_size` (FFN) | 11008 |
| `moe_intermediate_size` (I) | 1024 |
| `num_experts` (E) | 128 |
| `num_experts_per_tok` (top_k) | 8 |
| `num_attention_heads` | 32 |
| `num_key_value_heads` | 4 |
| `vocab_size` | 151936 |

### 13.2 并行配置

```
world_size = 16
ep_size = 4
ep_fsdp_size = world_size / ep_size = 4
tp_size = 1
pp_size = 1
dp_mode = "fsdp2"
```

### 13.3 Mesh 计算

```python
# parallel_state.py:538-548
ep_fsdp_size = world_size // ep_size = 16 // 4 = 4

# ep_fsdp_device_mesh 的布局（ep_outside=False 时）:
mesh = arange(16).view(ep_fsdp_size=4, ep_size=4).T
     = [[0,1,2,3],      → ep_fsdp group 0
        [4,5,6,7],      → ep_fsdp group 1
        [8,9,10,11],    → ep_fsdp group 2
        [12,13,14,15]]  → ep_fsdp group 3

转置后:
     = [[0,4,8,12],     → ep group 0 的 4 个 ep_fsdp rank
        [1,5,9,13],     → ep group 1
        [2,6,10,14],    → ep group 2
        [3,7,11,15]]    → ep group 3

mesh_dim_names = ("ep", "ep_fsdp")
```

### 13.4 16-GPU 布局图

```
                           ep_fsdp (FSDP within EP group)
                    ┌──── ep_fsdp=4 ─────────────┐
                    │                             │
            GPU 0      GPU 4      GPU 8      GPU 12     ← ep_rank=0, 持有 E0-E31
ep          GPU 1      GPU 5      GPU 9      GPU 13     ← ep_rank=1, 持有 E32-E63
(EP         GPU 2      GPU 6      GPU 10     GPU 14     ← ep_rank=2, 持有 E64-E95
group)      GPU 3      GPU 7      GPU 11     GPU 15     ← ep_rank=3, 持有 E96-E127
            └──── ep=4 ──────────┘

每个 EP rank 持有: 128/4 = 32 个专家

FSDP mesh（普通参数）:
  fsdp_mesh = device_mesh["dp_shard"]  (size=16, 跨全部 GPU)

EP-FSDP mesh（专家参数）:
  ep_fsdp_mesh = ep_fsdp_device_mesh["ep_fsdp"]
  例如 ep_rank=0 的 ep_fsdp group: {GPU 0, GPU 4, GPU 8, GPU 12}
```

### 13.5 EP 切分数值

```
ParallelPlan:
  "model.layers.*.mlp.experts.gate_proj": Shard(0)
  "model.layers.*.mlp.experts.up_proj":   Shard(0)
  "model.layers.*.mlp.experts.down_proj":  Shard(0)

原始权重形状 → 切分后每个 EP rank 持有:
  gate_proj: [128, 1024, 2048] → [32, 1024, 2048]    Shard(0), ep=4
  up_proj:   [128, 1024, 2048] → [32, 1024, 2048]    Shard(0), ep=4
  down_proj: [128, 2048, 1024] → [32, 2048, 1024]    Shard(0), ep=4

FSDP2 进一步切分（Shard(1) on ep_fsdp_mesh, ep_fsdp_size=4）:
  gate_proj: [32, 1024, 2048] → [32, 256, 2048]    Shard(1), ep_fsdp=4
  up_proj:   [32, 1024, 2048] → [32, 256, 2048]    Shard(1), ep_fsdp=4
  down_proj: [32, 2048, 1024] → [32, 512, 1024]    Shard(1), ep_fsdp=4
```

### 13.6 参数量估算

```
每层 MoE 专家参数（全量）:
  gate_proj: 128 * 1024 * 2048 = 268M params
  up_proj:   128 * 1024 * 2048 = 268M params
  down_proj: 128 * 2048 * 1024 = 268M params
  合计: 804M params/layer

每层 attention 参数:
  q_proj: 2048 * 2048 = 4.2M
  k_proj: 2048 * 256  = 0.5M  (4 KV heads * 64 dim)
  v_proj: 2048 * 256  = 0.5M
  o_proj: 2048 * 2048 = 4.2M
  合计: ~9.4M params/layer

每层 router 参数:
  gate.weight: 128 * 2048 = 0.26M params/layer

总参数量:
  48 layers * (804M + 9.4M + 0.26M) ≈ 39B params (含 embedding)
  → 实际 ~30B（因为只有部分层是 MoE，具体取决于 interleave 配置）

每个 GPU 上的专家参数（EP+FSDP 后）:
  gate_proj: 32 * 256 * 2048 * 2 bytes (bf16) = 32 MB/layer
  up_proj:   32 * 256 * 2048 * 2 bytes = 32 MB/layer
  down_proj: 32 * 512 * 1024 * 2 bytes = 32 MB/layer
  合计: 96 MB/layer * 48 layers ≈ 4.5 GB（专家参数，单 GPU）
```

### 13.7 前向通信量估算

```
假设: batch_size=1, seq_len=4096, top_k=8

每层 MoE 的通信:
  (1) All-to-All dispatch:
      每个 token 发送给 top_k=8 个专家，分布在 ep_size=4 个 rank 上
      发送数据: 4096 * 8 / 4 * 2048 * 2 bytes ≈ 32 MB/rank（每个rank发送）
      → 双向总通信量 ≈ 128 MB（4 个 rank 同时发送）

  (2) All-to-All combine:
      通信量与 dispatch 相同 ≈ 128 MB

  (3) FSDP all-gather (专家参数):
      每个 ep_fsdp rank 需要 all-gather 完整的 32 个专家
      数据量: (32*1024*2048 + 32*1024*2048 + 32*2048*1024) * 2 bytes
            = (67M + 67M + 67M) * 2 = 402 MB
      但 ep_fsdp_size=4，所以每个 rank 持有 1/4
      all-gather 通信量 ≈ 402 * 3/4 = 301 MB

  每层总通信: ~557 MB
  48 层总通信: ~26 GB
```

### 13.8 梯度修正验证

```
gradient_divide_factor = world_size = 16

验证:
  专家参数的 FSDP reduce-scatter 在 ep_fsdp_size=4 个 rank 上执行
  set_gradient_divide_factor(16) 使得 FSDP 除以 16
  等效: reduce-scatter 得到 sum / 16 = 全局平均  ✓

  如果不修正:
  reduce-scatter 默认除以 ep_fsdp_size=4
  sum / 4 ≠ 全局平均  ✗
```

---

## 第 14 节：总结与框架对比

> **一句话总结**：VeOmni 通过声明式 ParallelPlan、独立 EP mesh、双层 FSDP 包装、EP-aware 梯度修正和 checkpoint 扩展五大设计点，在复用 HuggingFace 模型的前提下实现了完整的 EP+FSDP 训练支持。

### 14.1 五大核心设计点

```
设计点 1: 声明式 ParallelPlan
  "model.layers.*.mlp.experts.gate_proj": Shard(0)
  → 16 行配置文件即可适配新 MoE 模型

设计点 2: 独立 EP Mesh
  ep_fsdp_device_mesh = DeviceMesh(mesh_dim_names=("ep", "ep_fsdp"))
  → EP 维度和 FSDP 维度正交，互不干扰

设计点 3: 双层 FSDP 包装
  fully_shard(experts_mod, mesh=ep_fsdp_mesh)   ← 内层
  fully_shard(layer_mod, mesh=fsdp_mesh)         ← 外层
  → 专家参数和普通参数在不同 mesh 上独立切分

设计点 4: EP-aware 梯度修正
  FSDP1: fsdp_state._gradient_postdivide_factor *= ep_size
  FSDP2: experts_mod.set_gradient_divide_factor(world_size)
  → 确保专家参数的梯度均值在全局数据上正确计算

设计点 5: Checkpoint 扩展
  state_dict_post_hook / load_state_dict_pre_hook + 3 monkey patches
  → EP 参数的模型权重和 optimizer state 正确保存/加载
```

### 14.2 框架对比表

| 方面 | VeOmni | TorchTitan | Automodel (Megatron-style) |
|------|--------|------------|----------------------------|
| **模型来源** | HuggingFace 模型 + monkey-patch | 自定义模型 (`model.py`) | Megatron 格式模型 |
| **EP 声明** | 声明式 `ParallelPlan` (16 行) | 4 种 `ParallelStyle` 类 | 手动配置 |
| **FSDP 版本** | FSDP1 + FSDP2 双路径 | 仅 FSDP2 | 不使用 FSDP |
| **EP Mesh** | 独立 2D `(ep, ep_fsdp)` mesh | 三套全局 mesh 中的 sparse mesh | 自定义 process group |
| **Token Dispatch** | 模型内部 (permute + a2a + sort) | 框架提供 (a2a + Triton permute) | 框架提供 |
| **对齐填充** | 不需要 (kernel 内部处理) | TOKEN_GROUP_ALIGN_SIZE_M | Token-based padding |
| **Prefetch** | 动态 `_fsdp_modules` 列表 | 硬编码双重 prefetch | 手动 overlap |
| **Checkpoint** | FSDP1: monkey-patch; FSDP2: DCP | DCP 原生支持 | 自定义 saver/loader |
| **DeepEP** | 不支持 | 作为替代后端 | 支持 |
| **torch.compile** | 不涉及（HF 模型不易编译） | 子模块级编译策略 | 不使用 |
| **PP + EP** | 不支持 | DualPipeV 通信重叠 | 支持 |
| **适配新模型** | 编写 parallel_plan.py + patch | 实现完整 MoE 类 | 适配 Megatron 格式 |

### 14.3 进一步阅读

- **VeOmni 源码**：[github.com/ByteDance-Seed/VeOmni](https://github.com/ByteDance-Seed/VeOmni)
- **TorchTitan EP+FSDP 分析**：本系列姊妹篇 `torchtitan-ep-fsdp-deep-dive.md`
- **PyTorch FSDP2 文档**：[pytorch.org/docs/stable/fsdp.html](https://pytorch.org/docs/stable/fsdp.html)
- **DeepEP 论文**：[arxiv.org/abs/2408.15664](https://arxiv.org/abs/2408.15664)
- **Qwen3-MoE 技术报告**：[huggingface.co/Qwen/Qwen3-MoE-30B-A3B](https://huggingface.co/Qwen/Qwen3-MoE-30B-A3B)

---

## 附录：关键文件与行号索引

| 文件路径 | 行号范围 | 内容 |
|---------|---------|------|
| `torch_parallelize.py` | 84-234 | `parallelize_model_fsdp1()` |
| `torch_parallelize.py` | 211-213 | FSDP1 梯度修正 `_gradient_postdivide_factor` |
| `torch_parallelize.py` | 237-435 | `parallelize_model_fsdp2()` |
| `torch_parallelize.py` | 359-391 | `_fsdp_modules` 构造 + 双层包装 |
| `torch_parallelize.py` | 366-377 | FSDP2 梯度修正 `set_gradient_divide_factor` |
| `torch_parallelize.py` | 393-410 | 显式 prefetch 配置 |
| `parallel_state.py` | 78-93 | `ParallelState` 数据类 |
| `parallel_state.py` | 319-357 | EP 属性: `ep_size`, `ep_fsdp_size`, `ep_gradient_divide_factor` |
| `parallel_state.py` | 538-548 | EP mesh 初始化 |
| `comm.py` | 20-54 | `_AllToAll` 同步通信 (前向 + 反向) |
| `comm.py` | 57-92 | `_AllToAll_Async` 异步通信 |
| `fsdp/clip_grad_norm.py` | 15-133 | FSDP1 EP-aware 梯度裁剪 |
| `fsdp/clip_grad_norm.py` | 28-51 | 参数分组: sharded / ep_fsdp_sharded / nonsharded |
| `fsdp/clip_grad_norm.py` | 92-99 | 两阶段归约: ep_fsdp_group + ep_group |
| `fsdp2/clip_grad_norm.py` | 21-99 | FSDP2 EP-aware 梯度裁剪 |
| `fsdp2/clip_grad_norm.py` | 47-99 | `ep_fsdp2_clip_grad_norm()` |
| `fsdp2/clip_grad_norm.py` | 146-171 | `_fsdp2_reduce_group()` 通用归约 |
| `fsdp/extension.py` | 124-133 | `CheckpointExtensions.__init__()` |
| `fsdp/extension.py` | 192-215 | `state_dict_post_hook` (保存时追加 EP) |
| `fsdp/extension.py` | 217-252 | `load_state_dict_pre_hook` (加载时还原) |
| `fsdp/extension.py` | 254-323 | Patch 1: `_convert_state_with_flat_params` |
| `fsdp/extension.py` | 325-358 | Patch 2: `optim_state_dict` |
| `fsdp/extension.py` | 360-411 | Patch 3: `optim_state_dict_to_load` |
| `fsdp/extension.py` | 414-451 | `register_checkpoint_extension()` |
| `qwen3_moe/parallel_plan.py` | 1-16 | Qwen3-MoE EP 切分声明 |
| `qwen3_moe/modeling_qwen3_moe.py` | 67-124 | `PatchQwen3MoeExperts` |
