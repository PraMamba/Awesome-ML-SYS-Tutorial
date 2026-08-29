# TorchTitan `ParallelDims` 源码走读学习计划

## 计划定位

这是一篇 `code-walkthrough` 类型的源码走读文章计划，目标层级为
`understand-reproduce`。文章对象不是某个并行算法的性能评测，而是解释
TorchTitan 如何把同一个 `world_size` 个 rank 重新解释成多个有用途的
`DeviceMesh` 视图，以及这些视图如何被下游并行实现使用。

- 目标文章：`torch/parallel_dims_lab/01-parallel-dims-source-walkthrough.md`
- 需求规格：`torch/parallel_dims_lab/03-parallel-dims-20-hour-study-plan.md`
- 源码事实来源：
  `/workspace/algorithm/torchtitan/.worktrees/source_code_analysis/torchtitan/distributed/parallel_dims.py`
- 源码固定 commit：
  `d6555c4c35a10bebcce652b58374cdeb2ecbe527`
- 文章语言：中文为主，保留源码 API、变量名和英文术语
- 写作顺序：每一章都遵循“概念 → 模型/场景 → 代码”

20 小时学习计划提供实践范围和验收用例；本计划负责把这些材料压缩成一条
可读的源码推导链。它不把学习计划的十个单元机械地改名为十个文章章节。

## 文章驱动问题

主问题是：

> 给定一组固定的 rank，`ParallelDims` 为什么需要同时构造
> `dataloading`、`loss`、`dense` 和 `sparse` 等不同的 `DeviceMesh` 视图？
> 它如何用整数约束保证这些视图都覆盖同一组设备，又如何根据
> `spmd_backend` 和已启用的轴把正确的 mesh 交给下游代码？

文章需要沿着主问题回答四个子问题：

1. `dp_replicate`、`dp_shard`、`cp`、`tp`、`pp`、`ep` 分别描述什么，为什么
   逻辑 `dp` 还要展开成两个具体轴？
2. `batch`、`loss`、`fsdp`、`efsdp` 是如何从基础并行度推导出来的？这些
   名称表示数据组织视角，不等于新增了物理 GPU。
3. `build_mesh()` 如何从一维 world mesh 生成多种视图，两个 backend 的 dense
   视图为什么不同？
4. `ParallelDims` 负责 mesh 的合法性、构造和解析到哪里；FSDP 参数分片、TP
   集合通信、CP 注意力通信、PP 调度和 MoE token 路由又由谁负责？

文章结尾必须明确一个边界：这个文件是并行维度和 mesh 的编排层，不是完整的
通信实现，也不是 GPU 吞吐、显存或训练效果 benchmark。

## 动机与文章位置

现有草稿已经抓住了两个有价值的入口：密集模型需要正交维度，MoE 需要一个
不同于 dense 计算的稀疏视图。但草稿把一个旧的 `etp` 设计假设和一个当前
源码不存在的梯度属性写进了主叙事，因此不能直接扩写。新文章应以固定源码
为准，把“为什么要有多个视图”作为连续主线，再逐段落到当前 API。

已有 20 小时学习计划覆盖了 collectives、坐标、DP/FSDP、TP、CP/PP、验证、
mesh、EP/routing、梯度统计和配置审计。本文章只吸收其中能解释
`ParallelDims` 的概念与复现实验；它不把模拟实验的结果写成真实分布式训练
结果。

建议把文章放在以下已发布知识的交汇处：

- [PyTorch Distributed](../torch-distributed/readme.md)：先建立 rank、
  process group、collective 和分布式抽象的共同语境。
- [RL 系统深思：FSDP 训练后端](../../rlhf/sys-design/readme-2.md)：补充
  DP/FSDP 参数分片、梯度同步和数据并行的角色边界。
- [深入浅出 DeepSeek MoE，EP 与 FSDP 经典二次开发](../../rlhf/sys-design/readme-4.md)：
  只用于理解 EP、专家区域和 token 路由背景，不把其中的设计直接当作本文件
  的实现。
- [NCCL 与 NVIDIA TOPO](../nccl/readme.md)：可选，用于区分“mesh 逻辑轴”
  与真实硬件拓扑；源码没有凭自身证明某个轴一定对应 NVLink 或某种网络层级。

## 前置知识检查

| 层级 | 读者需要会什么 | 文章中的用法 | 未满足时的补充 |
| --- | --- | --- | --- |
| Python | `dataclass`、属性、断言、字典和 tuple | 读 `ParallelDims` 的状态和缓存 | 先通读类定义与 `_validate()` |
| 分布式基础 | rank、world size、ProcessGroup、collective | 理解一个轴如何代表一组 rank | 先读 `PyTorch Distributed` |
| Mesh | 行主序坐标、轴、子 mesh、flatten/unflatten | 理解同一 world mesh 的多种投影 | 先运行 `02_mesh_coordinates.py` |
| 并行语义 | DP/FSDP、TP、CP、PP、EP 的职责 | 解释基础轴和派生轴 | 对照 20 小时计划的单元 3–8 |
| DTensor/SPMD | `DeviceMesh`、`Shard`/`Replicate`/`Partial` 的基本含义 | 理解两个 `spmd_backend` 分支 | 阅读 [DTensor 文档](https://docs.pytorch.org/docs/stable/distributed.tensor.html) |
| 实验边界 | 模拟、CPU Gloo 演示、真实 GPU 运行的区别 | 正确解读实验结果 | 先读文章的“证据边界”章节 |

前置检查不是要读者先掌握 TorchTitan 全部训练代码。文章应先用一个小的
`(2, 4)` rank 网格建立坐标直觉，再进入 `ParallelDims` 的多轴约束。

## 证据和引用规则

源码链接必须固定到已复核的 commit，不能链接 `main`：

- [轴名、DP 展开和类字段（commit `d6555c4c`）](https://github.com/pytorch/torchtitan/blob/d6555c4c35a10bebcce652b58374cdeb2ecbe527/torchtitan/distributed/parallel_dims.py#L29-L128)
- [mesh 构造和 mesh 尺寸验证](https://github.com/pytorch/torchtitan/blob/d6555c4c35a10bebcce652b58374cdeb2ecbe527/torchtitan/distributed/parallel_dims.py#L147-L329)
- [mesh 获取、backend 解析和共享 mesh](https://github.com/pytorch/torchtitan/blob/d6555c4c35a10bebcce652b58374cdeb2ecbe527/torchtitan/distributed/parallel_dims.py#L331-L512)
- [一维 mesh、启用状态和序列长度约束](https://github.com/pytorch/torchtitan/blob/d6555c4c35a10bebcce652b58374cdeb2ecbe527/torchtitan/distributed/parallel_dims.py#L514-L609)

写作时按下面的证据层级措辞：

1. **源码事实**：可以直接指向函数、字段、分支或断言，并给出固定 commit。
2. **源码注释支持的设计意图**：说明是注释或 docstring 表达的意图，不扩写
   成未经验证的性能结论。
3. **数学模型/实验观察**：说明是本文为了学习而建立的模型，或来自本地纯
   Python/CPU 实验。
4. **推断**：明确标成推断，并说明还需要哪一处下游源码或硬件资料才能确认。

不要把 mesh 的形状验证写成参数分片已经发生，也不要把 CPU Gloo 的 All-Reduce
演示写成 GPU 通信性能证据。

## 总体推导链

文章要让读者沿着下面的单向链路前进，每个箭头都应在正文中用“概念 → 模型
→ 代码”完成一次落地：

```text
rank / world_size
  → 轴名与坐标
  → 基础并行度与合法性约束
  → batch / loss / fsdp / efsdp 派生量
  → world mesh 的多种视图
  → spmd_backend 分支
  → get_mesh / resolve_mesh 选择语义
  → dense / sparse 计算边界与复现实验
```

这不是源码的调用栈图，而是文章的阅读路径。真正的主调用链应在代码章节
中写成：

```text
from_config()
  → ParallelDims(...)
  → __post_init__()
  → _validate()
  → build_mesh()
  → _validate_meshes()
  → get_optional_mesh() / resolve_mesh()
```

## 学习路线图

### 第 1 步：先建立一个 rank 网格，而不是先背字段

- **深度**：understand-reproduce
- **从什么出发**：`02_mesh_coordinates.py` 的行主序坐标模型，以及
  `PyTorch Distributed` 中 rank 与 group 的基本概念。
- **目标**：建立“rank 集合沿某个轴组成 group”的坐标直觉，能读懂后续
  `DeviceMesh` 的轴名和切片。
- **方法**：先手算 `(2,4)` 网格的坐标和 axis group，再把结果逐项对应到
  源码的枚举与 DP 展开函数。
- **概念**：world size、rank、mesh axis、coordinate、axis group；区分
  “轴的名字”与“tensor dimension”。源码的 `MeshAxisName` docstring 明确
  使用 `axis` 表示 `DeviceMesh` 轴，用 `dim` 表示 tensor 维度。
- **模型/场景**：对形状 `(2, 4)` 的 rank 网格，按行主序解释 rank 6 为
  `(1, 2)`；axis 0 的组为 `[4, 5, 6, 7]`，axis 1 的组为 `[2, 6]`。
  这个例子只用于建立坐标直觉，不把坐标顺序解释成硬件拓扑。
- **代码落点**：阅读
  `MeshAxisName`（源码 `29–50` 行）以及
  `unfold_dp_axis()`/`unfold_dp_axes()`（`53–65` 行）。重点回答：为什么
  逻辑 `dp` 展开为 `dp_replicate` 和 `dp_shard`，而不是新增一个独立物理轴。
- **输出物**：在文章中放一张小坐标表和一个简短的 axis-group 例子；不要
  引入与当前源码无关的 `etp`。
- **交叉引用**：`02_mesh_coordinates.py`、`PyTorch Distributed`、
  [DeviceMesh 官方教程](https://docs.pytorch.org/tutorials/recipes/distributed_device_mesh.html)。

### 第 2 步：给每个并行轴一个职责，再推导派生轴

- **深度**：understand
- **从什么出发**：已有学习计划的 DP/FSDP、TP、CP、PP、EP 角色说明。
- **目标**：能从六个基础并行度独立推导 `batch/loss/fsdp/efsdp`，并解释
  每个因子代表的 rank 组织关系。
- **方法**：先用固定 32-rank 配置手算公式，再只阅读计算这些量的源码行，
  最后用 mesh 乘积反算覆盖范围。
- **概念**：`dp_replicate`、`dp_shard`、`cp`、`tp`、`pp`、`ep`；逻辑
  `dp`、dense 计算、sparse expert 区域。
- **模型/场景**：固定配置
  `world_size=32, dpr=2, dps=2, cp=2, tp=2, pp=2, ep=4`，逐步得到：

  ```text
  batch = dpr * dps = 4
  loss  = dpr * dps * cp = 8
  fsdp  = dps * cp = 4
  efsdp = dps * cp * tp // ep = 2
  ```

  同时检查 dense 乘积 `2*2*2*2*2 == 32`、EP 区域 `dps*cp*tp=8`
  能被 `ep=4` 整除，以及 sparse mesh 乘积
  `pp*dpr*efsdp*ep=32`。强调这些是 rank 组织关系，不是显存或吞吐结论。
- **代码落点**：对照 `ParallelDims` 字段（`68–82` 行）、`build_mesh()` 中
  派生量（`216–218` 行）和 docstring 中的 mesh 语义（`151–178` 行）。
- **输出物**：正文先给公式，再给固定数字，再给源码字段和计算位置。不要
  先贴大段实现再让读者自己寻找公式。
- **交叉引用**：`03_dp_fsdp_sim.py`、`04_tp_mlp_sim.py`、
  `05_cp_pp_planner.py`、[Tensor Parallel 教程](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html)。

### 第 3 步：把配置合法性和训练语义分开

- **深度**：understand-reproduce
- **从什么出发**：学习计划中的用例 A–F 和当前源码的构造流程。
- **目标**：能区分构造器保证的 invariant、调用方必须补充的约束，以及
  singleton 轴为何可能没有真实 collective group。
- **方法**：按 A–F 逐例预测成功/失败，再沿构造调用链寻找对应断言、推导和
  fake-backend 分支；不把实验器的额外检查倒灌回源码。
- **概念**：构造期 invariant、自动推导、EP 整除、singleton 轴、序列长度
  外部约束；区分“`ParallelDims` 能构造”与“整个模型配置可运行”。
- **模型/场景**：正文用一张表覆盖六个验收用例：

  | 用例 | 场景 | 预期行为 | 责任边界 |
  | --- | --- | --- | --- |
  | A | 固定 32-rank 配置 | 通过 dense 与 EP 检查 | `ParallelDims` |
  | B | dense 乘积不等于 world size | 构造失败 | `_validate()` |
  | C | `dps*cp*tp` 不能被 `ep` 整除 | `ValueError` | `_validate()` |
  | D | `dp_shard=-1` | 按 `world_size // (dpr*cp*tp*pp)` 推导 | `_validate()` |
  | E | `seq_len` 不能被 `tp*(2*cp)` 整除 | 审计/调用方拒绝；不是构造器检查 | `seq_len_divisor` 的使用方 |
  | F | 某个轴为 1 | 可能使用 fake backend，不提供真实 collective group | `_mesh_exist()` / `build_mesh()` |

- **代码落点**：依次读 `from_config()`（`84–97` 行）、`__post_init__()`
  （`99–100` 行）、`_validate()`（`102–128` 行）和 `_mesh_exist()`
  （`130–145` 行）。明确记录：源码没有在 `ParallelDims` 构造期接收或检查
  `seq_len`，因此用例 E 不能写成“构造器抛出非法序列长度”。
- **输出物**：一条完整的构造调用链和六个用例的事实/责任边界表。
- **交叉引用**：`06_parallel_dims_lite.py`、`10_parallel_config_advisor.py`、
  `tests/test_validation.py`。

### 第 4 步：从一个 world mesh 展开多个 dense 视图

- **深度**：understand-reproduce
- **从什么出发**：第 2 步的派生量和第 3 步的合法配置。
- **目标**：能从同一个一维 world mesh 复述四类 global mesh 的形状、来源和
  用途，并用乘积检查它们没有凭空增加设备。
- **方法**：先填写固定配置的形状表，再顺读 `build_mesh()` 的构造顺序，最后
  用 `_validate_meshes()` 的 `expected_sizes` 逐项核对。
- **概念**：一维 world mesh、unflatten、flatten、视图复用、数据加载轴、
  loss 归约轴；解释“多个 mesh”是同一批 rank 的不同组织视角。
- **模型/场景**：用固定 32-rank 配置填充下表。需要特别区分
  `spmd_types` 中“完整 dense storage mesh”和“前向/反向逻辑 mesh”：

  | 视图 | 形状/来源 | 作用 |
  | --- | --- | --- |
  | `dataloading` | `(pp, batch, cp, tp) = (2, 4, 2, 2)` | 数据分片与 global batch 视角 |
  | `loss` | 从 `dataloading["batch", "cp"]` flatten，因子为 `(4, 2)`，结果大小为 8 | loss 归约视角 |
  | `spmd_types` storage | `(pp, dpr, dps, cp, tp) = (2, 2, 2, 2, 2)` | 给 `fully_shard()` 的完整 dense 视角 |
  | `spmd_types` fwd/bwd | `(pp, dp, cp, tp) = (2, 4, 2, 2)`，公共 SPMD mesh 再取 `(dp, cp, tp)` | 前向/反向 type-checking |
  | `partial_dtensor` dense | `(pp, dpr, fsdp, tp) = (2, 2, 4, 2)` | 将 `dps` 与 `cp` 折叠为 `fsdp` |
  | `sparse` | `(pp, dpr, efsdp, ep) = (2, 2, 2, 4)` | expert region 的 mesh 视角 |

- **代码落点**：完整阅读 `build_mesh()`（`147–304` 行），尤其是本地
  `unflatten_mesh()`、`dataloading_mesh`、`loss_mesh`、两个 backend 分支和
  `full_sparse_mesh`。随后阅读 `_validate_meshes()`（`306–329` 行），将表中
  每个尺寸和源码的 `expected_sizes` 对上。
- **输出物**：一张 mesh 形状表和一幅 Mermaid 流程图，图中只表达
  world mesh 到各视图的关系，不绘制未经源码证实的网络拓扑。
- **交叉引用**：`07_mesh_inspector.py`、`tests/test_mesh_shapes.py`、
  [DeviceMesh 官方文档](https://docs.pytorch.org/docs/stable/distributed)。

### 第 5 步：解释两个 `spmd_backend` 分支，而不是把 `fsdp` 当成固定轴

- **深度**：understand
- **从什么出发**：第 4 步的同一配置、不同视图。
- **目标**：能解释同一并行配置在两个 backend 下为何暴露不同的 dense 轴，
  并避免把 `fsdp` 当成所有 backend 都存在的固定名称。
- **方法**：保持配置不变，只切换 `spmd_backend`，对照两个分支创建的 mesh、
  单轴字典和尺寸验证；再核对 docstring 与运行时字典的差异。
- **概念**：`spmd_types` 与 `partial_dtensor` 的 dense 表达差异；逻辑
  `dp`、物理 `dp_shard`、折叠后的 `fsdp`；fake backend 的用途和限制。
- **模型/场景**：用同一个 `world_size=32` 配置比较：

  | backend | dense 构造 | 单轴访问中的数据轴 | 文章必须说明的差异 |
  | --- | --- | --- | --- |
  | `spmd_types` | `(pp, dpr, dps, cp, tp)` 及 `(pp, dp, cp, tp)` | `dp`、`dp_shard` | SPMD 类型系统看到逻辑 `dp`；完整 storage 视图仍保留 `dps` |
  | `partial_dtensor` | `(pp, dpr, fsdp, tp)` | `fsdp` | `dps` 与 `cp` 在 dense 视图中先折叠 |

  `get_optional_mesh()` 的 docstring 列出 `fsdp`，但在当前
  `spmd_types` 分支 `_single_axis_meshes` 实际加入的是 `dp` 和 `dp_shard`；
  文章应按运行时字典和分支行为解释这一点，而不是把 docstring 的列表当成
  所有 backend 都可用的保证。
- **代码落点**：阅读 `build_mesh()` 的 backend 分支（`230–295` 行）、
  `_validate_meshes()` 的 backend 条件（`318–323` 行）、
  `spmd_dense_mesh()`/`spmd_sparse_mesh()`（`425–435` 行）和
  `get_dense_tp_mesh()`（`437–441` 行）。
- **输出物**：一张“backend → dense mesh → 可访问单轴”的对比表，并明确
  `ParallelDims` 只是准备下游使用的 mesh。
- **交叉引用**：`06_parallel_dims_lite.py`、`07_mesh_inspector.py`、
  [DTensor placements 文档](https://docs.pytorch.org/docs/stable/distributed.tensor.html)。

### 第 6 步：理解 mesh 获取 API 的“可用、启用、共享”三种语义

- **深度**：understand
- **从什么出发**：已经构造完成的 `_single_axis_meshes` 和 `_global_meshes`。
- **目标**：能预测 optional/required/activated/shared 四类 API 在未启用轴、
  singleton 轴和多轴请求下的返回值或异常。
- **方法**：为每个 API 画一条“输入轴集合 → 过滤 → 子 mesh/cache”的短路径，
  再用一个 backend 对比例子验证轴集合而非 placement 值是共享条件。
- **概念**：`get_optional_mesh()` 与 `get_mesh()` 的失败语义、singleton 轴、
  多轴子 mesh、对象 identity cache、`get_activated_mesh()` 的过滤规则、
  backend-specific `resolve_mesh()`。
- **模型/场景**：设计三个对比：

  1. 请求一个未启用的轴：`get_optional_mesh()` 返回 `None`，`get_mesh()` 抛
     `ValueError`。
  2. 请求 `["dp", "cp", "tp", "ep"]`：`spmd_types` 保留
     `dp/cp/tp/ep`，`partial_dtensor` 只保留 `tp/ep`；如果没有任何保留轴，
     返回 `None`。
  3. 请求多个轴：从能够包含这些轴的 global mesh 切子 mesh，并复用缓存对象；
     这解释了为什么不能用“每次都重新切片”代替现有代码。

- **代码落点**：阅读 `get_optional_mesh()`/`get_mesh()`（`331–423` 行）、
  `get_activated_mesh()`/`resolve_mesh()`（`443–482` 行）和
  `resolve_shared_mesh()`（`484–512` 行）。重点解释 `resolve_shared_mesh()`
  检查的是轴集合一致，而不是 placement 值相同。
- **输出物**：画出“输入轴集合 → backend 过滤 → enabled 过滤 → submesh/cache”
  的 Mermaid 流程图，并给出一个成功和一个失败例子。
- **交叉引用**：`10_parallel_config_advisor.py`、`tests/test_mesh_shapes.py`。

### 第 7 步：把 sparse mesh 放回 MoE 边界内

- **深度**：understand-reproduce
- **从什么出发**：第 2 步的 `efsdp` 公式、第 4 步的 sparse 形状，以及
  `08_moe_router_sim.py` 的 16-token/8-expert/`ep=4` 纯 Python 模型。
- **目标**：能说清 sparse mesh 为 expert dispatch 提供什么，以及它没有在
  `parallel_dims.py` 中完成什么，尤其不把 `etp` 或 All-to-All 归入当前 API。
- **方法**：先运行记录级路由模型，再回到 sparse mesh 的构造和 resolver，
  将“路由记录”“mesh 选择”“真实通信”拆成三层证据。
- **概念**：dense region 与 expert region 的 mesh 重组、EP 分桶、计算、还原；
  区分“为 expert dispatch 提供 mesh”与“实现 token all-to-all”。
- **模型/场景**：用 16 个 token、8 个 expert、4 个 EP rank 说明：token 先按
  expert 归属分桶，再在目标 expert 侧计算，最后按原位置还原。这个模型只验证
  路由记录和负载统计；它不声称复现 TorchTitan 的真实通信性能。
- **代码落点**：阅读 `build_mesh()` 的 sparse 构造（`262–279` 行）、
  `spmd_sparse_mesh()`（`431–435` 行）和 `resolve_mesh()` 中
  `spmd_types`/`partial_dtensor` 的 in-band 轴选择（`461–482` 行）。
  正文要明确：当前文件没有 `etp` 轴，也没有在这里执行 token router 或
  all-to-all；实际路由行为需要看下游 expert dispatch 源码。
- **输出物**：一张 dense/sparse 视图对照表，外加一段“本文件做什么/不做什么”。
- **交叉引用**：`08_moe_router_sim.py`、`tests/test_router.py`、
  [MoE/EP 已发布文章](../../rlhf/sys-design/readme-4.md)。

### 第 8 步：收束到序列长度、启用状态和可审计配置

- **深度**：reproduce
- **从什么出发**：前面已经得到 `tp`、`cp` 和 mesh enabled 状态。
- **目标**：能把 `seq_len_divisor` 和 enabled properties 解释成审计输入，
  并指出哪些错误必须由调用方或审计器报告。
- **方法**：用 `tp=2, cp=2` 计算合法/非法序列长度，再把属性输出映射到配置
  审计等级；不伪造构造器的序列长度异常。
- **概念**：`seq_len_divisor`、`tp_enabled`/`cp_enabled`/`fsdp_enabled` 等
  属性、`non_data_parallel_size`；区分“源码给出约束因子”和“调用方执行
  约束检查”。
- **模型/场景**：`tp=2, cp=2` 时 `seq_len_divisor=2*(2*2)=8`；序列长度
  16 可被整除，10 不可被整除。文章须写明当前源码只返回 8，并不接收
  `seq_len` 参数或主动抛出序列长度错误。
- **代码落点**：阅读属性区（`555–599` 行）和 `seq_len_divisor`（`601–609` 行）。
  将 `cp*2` 的 load-balancing 注释作为源码依据，不扩展成未经验证的性能结论。
- **输出物**：以配置审计器为落点，说明哪些是 `ERROR/WARN/CHECK/INFO` 的
  配置提示，哪些只是源码事实；不要把配置通过写成训练有效。
- **交叉引用**：`05_cp_pp_planner.py`、`09_gradient_accounting.py`、
  `10_parallel_config_advisor.py`、`tests/test_validation.py`。

### 第 9 步：用最小实验闭合“理解—复现”

- **深度**：reproduce
- **从什么出发**：文章正文中的公式、mesh 形状表和六个验证用例。
- **目标**：形成一条可重复的最小证据链，能分别说明源码行为、模拟逻辑和
  CPU Gloo collective 演示各自证明了什么。
- **方法**：按实验组回指正文 invariant 和测试；只运行项目已有目录内入口，
  记录失败用例和证据边界，不把模拟输出外推成 benchmark。
- **概念**：证据层级、单元实验、源码行为复现、纯模拟的边界。
- **模型/场景**：按学习链将实验分为四组，而不是在 README 中堆一长串脚本
  命令：

  | 实验组 | 覆盖内容 | 允许得出的结论 |
  | --- | --- | --- |
  | rank/mesh | 坐标、axis group、mesh 形状 | 组织关系和公式实现正确 |
  | parallelism simulation | DP/FSDP、TP、CP/PP | 数学模型与 CPU 模拟路径一致 |
  | sparse/gradient | EP 分桶、还原、全局均值与均值之均值 | 记录级别的路由/统计逻辑正确 |
  | config audit | A–F 验收用例 | 审计器与当前约束的对应关系正确 |

  `01_collectives.py` 的 CPU Gloo 运行可作为唯一真实 collective 演示；其余
  脚本保持纯 Python/CPU 模拟。所有结果都只能证明演示逻辑或公式成立，不能
  写成真实 GPU 通信、吞吐、显存或训练质量 benchmark。
- **代码落点**：将每组实验回指到正文相应函数和测试；测试只验证明确的
  invariant，不通过隐藏实现细节“猜”源码行为。
- **输出物**：文章结尾的“如何复现与如何解读”小节，以及失败用例的预期错误
  说明。当前计划本身不执行或修改这些实验文件。
- **交叉引用**：`tests/` 下的三个 pytest 文件、`notes/README.md` 的运行与
  事实记录；不更新 README 或 `knowledge-graph.json`。

## 设计演进（文章解释路径，不声称是 Git 历史）

文章需要给读者一个逐步收敛的设计模型：

| 阶段 | 读者先看到的模型 | 解决的问题 | 对应源码 |
| --- | --- | --- | --- |
| 基线 | 一个一维 world mesh，所有计算都直接使用它 | 无法表达数据加载、loss、FSDP、TP、专家区域的不同 rank 关系 | `init_device_mesh()` |
| 中间态 | 用 unflatten/flatten 从同一 world mesh 得到 dense 多视图 | 同一批设备可按用途重新分组，且用乘积检查覆盖完整 | `build_mesh()`、`_validate_meshes()` |
| 完整模型 | backend-specific dense 视图 + sparse expert 视图 + resolver | 下游需要不同的轴集合和不同的 SPMD 解释 | `spmd_backend` 分支、`resolve_mesh()` |

每个阶段都要回答“为什么上一种视图不够”，再进入下一段代码。这里的
“演进”是帮助读者理解约束的教学顺序，不得写成源码提交历史或性能优化历史。

## `ParallelDims` 与下游实现的边界

| 能力 | `ParallelDims` 当前负责 | 不应归因于本文件 |
| --- | --- | --- |
| 配置入口 | `from_config()` 映射并行度和 backend | 完整训练配置的所有语义检查 |
| 合法性 | 正数、`dp_shard=-1` 推导、dense 乘积、EP 整除 | sequence length、hidden size、head 数等调用方约束 |
| mesh | world mesh、dataloading/loss/dense/sparse 视图 | 真实模型参数、token 或梯度数据 |
| mesh 选择 | optional/required/activated/shared resolver | 选择之后如何执行 collective |
| FSDP/TP/CP/PP/EP | 准备下游可使用的轴和 mesh | 参数 all-gather、reduce-scatter、TP/CP 通信、PP schedule、MoE all-to-all |
| 实验 | 可被 CPU 模拟或 Gloo 演示验证 | GPU 吞吐、显存、通信带宽、训练收敛效果 |

这张表是文章的防越界检查。尤其不能把 `spmd_sparse_mesh()` 的 docstring
写成“该函数完成了专家路由”；它只返回给 expert dispatch 使用的 mesh。

## 草稿与当前源码的差异清单

以下差异必须在文章修改时显式处理。前五项是已确认的源码冲突，后几项是
草稿缺失或证据不足，不能混写成同一种错误。

### 已确认的冲突

1. **源码链接未固定版本**：草稿第 3 行链接 `torchtitan` 的 `main` 分支；
   文章必须改为 commit
   `d6555c4c35a10bebcce652b58374cdeb2ecbe527`，并在引用附近保留复核信息。
2. **草稿引入了不存在的 `etp` 字段/轴**：草稿第 101–112、121、128、133、
   141、155–186 行把 `etp` 当作当前 `ParallelDims` 字段和 mesh 轴。当前
   `MeshAxisName` 只有 `DP/DP_REPLICATE/DP_SHARD/FSDP/TP/CP/PP/EP/EFSDP`
   （源码第 42–50 行），`ParallelDims` 字段在第 70–77 行结束于 `ep` 和
   `spmd_backend`。
3. **sparse mesh 形状错误**：草稿第 101–102 行写成
   `(pp, dp_replicate, efsdp, ep, etp)`；当前源码第 262–265 行明确构造
   `(pp, dp_replicate, efsdp, ep)`，没有第五个专家内 TP 轴。
4. **`efsdp` 资源公式错误**：草稿第 121、128 行使用
   `efsdp * ep * etp = fsdp * tp` 和除以 `etp * ep`；当前代码第 216–218
   行及第 316 行使用
   `efsdp = dp_shard * cp * tp // ep`，EP 只需整除
   `dp_shard * cp * tp`。
5. **`fsdp_gradient_divide_factor` 不是当前 API**：草稿第 143–186 行围绕
   该属性推导梯度对齐；当前类没有此字段或 property，属性区第 555–609 行
   也没有它。该段不能作为当前源码行为；若保留历史背景，必须明确版本、另找
   事实来源并与本文件实现分开，默认应从主线删除。

### 草稿没有覆盖，但当前源码必须补上的部分

6. `spmd_backend` 是当前 dataclass 字段，默认值为 `"spmd_types"`；两种
   backend 的 dense mesh 形状和可访问单轴不同。
7. `dataloading` 与 `loss` mesh 是源码的显式视图，不能只讲 dense/sparse；
   `loss` 是 `batch` 与 `cp` 轴 flatten 后的 mesh。
8. `_validate()` 的 `dp_shard=-1` 自动推导、dense 乘积检查和 EP 整除检查
   是主流程，不应只作为脚本附录。
9. `_mesh_exist()`、fake backend、singleton 轴过滤、mesh identity cache、
   `get_activated_mesh()` 和 `resolve_shared_mesh()` 是当前 API 的重要行为，
   草稿没有形成调用链。

### 需要重新取证而不是直接判错的表述

10. 草稿第 33–76、88–141 行把并行轴与“最高速网络拓扑”“NVLink”“物理
    节点”直接绑定。当前 `parallel_dims.py` 负责逻辑 mesh 轴和 process-group
    视图，没有单独证明这些硬件映射；文章应改成条件性推断，或用硬件/下游实现
    的独立来源补证。
11. 草稿第 137–141 行把当前文件写成直接触发 All-to-All 并完成 token 还原。
    `ParallelDims` 只构造并返回 sparse mesh；All-to-All、分桶、专家计算和
    还原必须由路由/dispatch 实现负责，文章要把这两层分开。
12. 草稿从概念直接跳到大段代码，没有固定 commit、完整入口链和失败路径；
    这属于可追溯性缺口，不是某个公式冲突，但必须在改稿时修复。

## 推荐文章结构

文章正文仍写入现有草稿路径，但按下面的章节顺序重组；本计划不直接修改
草稿：

1. **前言：同一批 rank 为什么需要多个 mesh 视角**
   - 给出驱动问题、源码 commit 和文章边界。
   - 用固定 32-rank 配置预告最终形状，但暂不解释所有字段。
2. **从 rank 坐标到并行轴**
   - 概念：轴、坐标、逻辑 DP 与 concrete axes。
   - 模型：`(2,4)` 行主序。
   - 代码：`MeshAxisName` 与 `unfold_dp_axis()`。
3. **从六个基础度数到四个派生量**
   - 概念：DP/FSDP/CP/TP/PP/EP 的职责。
   - 模型：`batch/loss/fsdp/efsdp` 公式和 32-rank 数值。
   - 代码：类字段与 `build_mesh()` 的派生量。
4. **构造器如何拒绝非法配置**
   - 概念：invariant 与调用方约束。
   - 模型：A–F 用例。
   - 代码：`from_config → __post_init__ → _validate`，再到 `_mesh_exist`。
5. **`build_mesh()`：unflatten、flatten 与多视图**
   - 概念：一个 world mesh 的多种用途。
   - 模型：完整 mesh 形状表。
   - 代码：dataloading、loss、dense、sparse 和 `_validate_meshes()`。
6. **两个 backend 的 dense 解释**
   - 概念：`spmd_types` 与 `partial_dtensor`。
   - 模型：相同配置下的两张 dense 视图。
   - 代码：backend 分支、`dp`/`dp_shard`/`fsdp` 的访问差异。
7. **从“有 mesh”到“取对 mesh”**
   - 概念：optional、required、activated、shared。
   - 模型：启用/未启用轴和 resolver 过滤。
   - 代码：`get_optional_mesh()`、`get_mesh()`、`resolve_mesh()` 等。
8. **sparse mesh 与 MoE 路由的边界**
   - 概念：expert region 的重组。
   - 模型：16 token、8 expert、EP=4 的桶化/还原模拟。
   - 代码：sparse mesh 构造和下游 router 的边界。
9. **序列长度与最终配置审计**
   - 概念：`seq_len_divisor` 和 enabled properties。
   - 模型：合法与非法序列长度。
   - 代码：属性区、配置审计器和测试映射。
10. **复现、证据边界与常见误解**
    - 回顾固定 commit、A–F 测试、CPU/Gloo 与纯模拟的差异。
    - 明确 `etp`、梯度属性、All-to-All 和硬件拓扑的误读。

每章的最小写作单元都是“先解释一个概念，再给一个可算的场景，最后只贴
支撑该场景的短代码片段”。超过一个屏幕的代码应改为链接到固定 commit，避免
文章退化成源码转录。

## 草稿完成度分析

按“主题覆盖率”而不是“正确性或文笔质量”估计，草稿约完成 **35%**：

| 已有内容 | 当前状态 |
| --- | --- |
| dense 并行维度的动机 | 有概念入口，但缺少当前字段和验证链 |
| dense/sparse 多视角 | 有叙事雏形，但 sparse 形状包含错误的 `etp` |
| 资源守恒 | 有公式意图，但当前 `efsdp` 公式错误 |
| EP 路由 | 有下游行为描述，但归因越过了 `ParallelDims` 文件边界 |
| 构造入口与 `_validate()` | 基本缺失 |
| dataloading/loss/backend 分支 | 基本缺失 |
| mesh 获取和 resolver | 基本缺失 |
| 固定 commit、失败用例、实验证据 | 缺失 |

因此下一次写作不是局部润色，而是保留草稿的动机材料，重建主体代码路径并
删除/改写已确认冲突的段落。35% 不代表已有内容可以直接作为 35% 的最终正文。

## 写作前的硬性检查点

- [ ] 再次执行 `git -C /workspace/algorithm/torchtitan/.worktrees/source_code_analysis rev-parse HEAD`，确认仍为目标 commit。
- [ ] 所有源码链接固定到 `d6555c4c35a10bebcce652b58374cdeb2ecbe527`，不使用 `main`。
- [ ] 正文不把 `etp` 写成当前字段、轴、公式因子或 mesh 维度。
- [ ] 正文不把 `fsdp_gradient_divide_factor` 写成当前属性。
- [ ] 正文同时解释 `dataloading`、`loss`、dense、sparse 四类视图。
- [ ] 逐一给出 `dp_shard=-1`、dense 乘积和 EP 整除的代码依据。
- [ ] 明确 `seq_len_divisor` 是返回约束因子，当前构造器不检查 `seq_len`。
- [ ] 解释两个 backend 的分支和 `fsdp`/`dp_shard` 访问差异。
- [ ] 至少给出三个“源码事实 / 推断 / 实验观察”容易混淆的反例。
- [ ] 不把纯 Python 模拟结果写成 GPU 通信、吞吐、显存或训练质量 benchmark。
- [ ] 文章引用的本地知识只链接已发布文章；不更新 README 和 `knowledge-graph.json`。
- [ ] 文章写作完成后，再按项目目录内既有验证入口运行测试；本计划阶段不修改或运行目标草稿。

## 本计划的非目标

- 不修改 `01-parallel-dims-source-walkthrough.md`。
- 不修改 `03-parallel-dims-20-hour-study-plan.md`。
- 不更新任何 README 或 `.learn/index/knowledge-graph.json`。
- 不把旧版本的 `etp` 或梯度归一化设计恢复为当前 API。
- 不扩展为 TorchTitan 全训练流程、真实多 GPU benchmark 或硬件拓扑教程。
