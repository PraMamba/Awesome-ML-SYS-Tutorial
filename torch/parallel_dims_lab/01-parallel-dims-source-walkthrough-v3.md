# ParallelDims 源码走读：把并行配置变成可组合的 DeviceMesh

我第一次看到 ParallelDims 时，直觉上会把它理解成一组并行度参数：数据并行是几份，Tensor Parallel 是几份，Pipeline Parallel 又是几份。但真正让训练系统变得可靠的，不是把这些整数放进一个配置类，而是回答一个更具体的问题：

> 对同一组 rank，数据加载、loss 统计、dense 参数、MoE expert 参数和运行时 placement，分别应该看到哪个 mesh？这些 mesh 的大小为什么一定能对上？

这篇 code-walkthrough 按 understand-reproduce 深度沿着这个问题阅读 Torchtitan 的 ParallelDims。源码引用固定在 [d6555c4c35a10bebcce652b58374cdeb2ecbe527](https://github.com/pytorch/torchtitan/tree/d6555c4c35a10bebcce652b58374cdeb2ecbe527)，不会把旧文档或漂移的分支链接当作实现依据。

## 阅读路线

1. 先把逻辑并行轴、mesh view 和约束模型分开。
2. 用一个固定的 32-rank 配置推导所有代表性 mesh。
3. 按源码执行顺序走读校验、构造、解析和下游边界。
4. 最后用本地学习实验复现形状，并区分结构证据和真实性能证据。

如果你一路读到复现部分，感谢你愿意把一个看似简单的配置对象读到它真正的边界。

## 一、概念：ParallelDims 管的不是一次通信，而是一份并行坐标契约

### 1.1 同一组 rank，需要多种坐标系

训练系统里至少有三层对象：

| 层次 | 它回答的问题 | 典型消费者 |
| --- | --- | --- |
| 全局 rank 空间 | 32 个进程如何被分解 | `world_size` 和维度乘积 |
| DeviceMesh view | 某个操作应该沿哪些轴组织 rank | dataloader、loss、FSDP、PP、MoE |
| placement / state 语义 | 参数或激活在这些轴上如何复制、切分或参与通信 | `fully_shard`、SPMD state、token dispatcher |

因此，`dp_replicate=2` 并不等于系统只会创建一个大小为 2 的 `dp_replicate` mesh。它还可能和 `dp_shard`、`cp`、`tp` 组合成 dense storage mesh，或者和 `efsdp`、`ep` 组合成 sparse expert mesh。ParallelDims 的职责是把一份整数配置投影为这些有名字的 view，并让不同消费者使用同一份命名结果。

### 1.2 逻辑轴和物理轴不是同一个概念

源码把 mesh 轴命名为 `MeshAxisName`。`axis` 是 DeviceMesh 的命名坐标，`dim` 则是 tensor placement 所使用的维度；二者不能混用。完整枚举见 [parallel_dims.py#L29-L50](https://github.com/pytorch/torchtitan/blob/d6555c4c35a10bebcce652b58374cdeb2ecbe527/torchtitan/distributed/parallel_dims.py#L29-L50)。

其中最容易误读的是 `dp`：

| 名称 | 语义 |
| --- | --- |
| `dp_replicate` | batch 方向的复制份数 |
| `dp_shard` | data-parallel 方向的参数切分份数 |
| `dp` | SPMD 运行时使用的逻辑轴，大小是 `dp_replicate * dp_shard` |
| `fsdp` | 在 `partial_dtensor` 后端下，dense 参数切分的 mesh view |
| `efsdp` | sparse expert 参数切分的 view |

也就是说，`dp` 是一个逻辑折叠结果，不能反推为一个永远独立存在的物理轴。源码中的 `unfold_dp_axis` 正是用来把逻辑 `dp` 展开成 `(dp_replicate, dp_shard)` 的地方：[parallel_dims.py#L53-L65](https://github.com/pytorch/torchtitan/blob/d6555c4c35a10bebcce652b58374cdeb2ecbe527/torchtitan/distributed/parallel_dims.py#L53-L65)。

### 1.3 一个 mesh 的形状必须同时满足两类约束

对全局 mesh，约束是：

<code>所有轴的 degree 相乘 = world_size</code>

对 sparse expert mesh，还要有局部可分性约束：

<code>dp_shard * cp * tp 能被 ep 整除</code>

第一条保证每个 rank 有唯一坐标；第二条保证 expert 侧的有效 data-parallel 规模可以均匀分给 expert-parallel rank。前者是全局拓扑约束，后者是 MoE 路由和 expert 参数切分能够成立的边界。

## 二、模型与场景：用一个固定配置推导所有 mesh

### 2.1 代表性配置

为了让形状推导能被逐项检查，下面一直使用这组配置：

| 字段 | 值 | 解释 |
| --- | ---: | --- |
| `world_size` | 32 | 全局 rank 数 |
| `dp_replicate` | 2 | batch 复制份数 |
| `dp_shard` | 2 | data-parallel 参数切分份数 |
| `cp` | 2 | Context Parallel 度数 |
| `tp` | 2 | Tensor Parallel 度数 |
| `pp` | 2 | Pipeline Parallel stage 数 |
| `ep` | 4 | Expert Parallel 度数 |

从这些输入得到：

<code>batch = dp_replicate * dp_shard = 4</code>

<code>loss = batch * cp = 8</code>

<code>fsdp = dp_shard * cp = 4</code>

<code>efsdp = dp_shard * cp * tp / ep = 2</code>

全局乘积检查为：

<code>dp_replicate * dp_shard * cp * tp * pp = 2 * 2 * 2 * 2 * 2 = 32</code>

这里的 `loss=8` 不是全局 world mesh 的大小。它是 dataloading mesh 在 `batch` 和 `cp` 两轴上的子 mesh；这一点在源码的 `build_mesh` 中很重要。

### 2.2 四个主要场景

同一份配置至少要支持以下场景：

| 场景 | 使用者 | 需要的视图 |
| --- | --- | --- |
| 数据加载 | 根据 batch 维度切分或统计 token | `pp`, `batch`, `cp`, `tp` |
| loss 统计 | 聚合相同数据并行语义下的 loss | `loss` |
| dense 参数 | 在不同后端中完成 FSDP 和 TP 组合 | `dp_replicate`, `dp_shard`, `cp`, `tp` 或 `dp_replicate`, `fsdp`, `tp` |
| sparse expert | 让 expert 参数和 token 路由使用兼容坐标 | `dp_replicate`, `efsdp`, `ep` |

这解释了为什么源码先构造多个全局 mesh，再从它们切出单轴和多轴 view。把所有消费者强行塞进一个 mesh，会让一个消费者的轴语义污染另一个消费者。

### 2.3 两种后端，同一份配置，不同的 dense 坐标

源码支持两个 `spmd_backend`：

| 后端 | dense storage 形状 | dense forward/backward 形状 | 关键语义 |
| --- | --- | --- | --- |
| `spmd_types` | `(pp, dp_replicate, dp_shard, cp, tp)` | `(pp, dp, cp, tp)` | storage 轴保留 `dp_replicate` 和 `dp_shard`，SPMD 运行时把它们折叠为 `dp` |
| `partial_dtensor` | `(pp, dp_replicate, fsdp, tp)` | 使用单轴 `fsdp` view | dense 参数侧把 `dp_shard` 和 `cp` 合成 `fsdp` |

在固定配置下，具体形状是：

| mesh | degree tuple | product |
| --- | --- | ---: |
| dataloading | `(pp=2, batch=4, cp=2, tp=2)` | 32 |
| loss | `(loss=8)` | 8 |
| `spmd_types` storage | `(2, 2, 2, 2, 2)` | 32 |
| `spmd_types` forward/backward | `(2, dp=4, 2, 2)` | 32 |
| `partial_dtensor` dense | `(2, 2, fsdp=4, 2)` | 32 |
| sparse | `(2, 2, efsdp=2, ep=4)` | 32 |

注意 sparse mesh 只有四个轴：`pp`、`dp_replicate`、`efsdp`、`ep`。当前实现没有 `etp` 轴；`tp` 已经进入 `efsdp` 的计算。对当前配置，`efsdp=2*2*2/4=2`。

### 2.4 设计演进：从单一 dense view 到按消费者拆分

可以把这段设计理解成三步：

1. 只看 dense 参数时，可以用一个 FSDP 视图描述切分。
2. 当 `cp` 也参与 dense sharding 时，需要把 `dp_shard * cp` 合成 `fsdp`，否则消费者要自己重复组合。
3. 当 SPMD state 类型系统和 sparse expert 出现后，storage、dense runtime、sparse runtime 的坐标职责不同，于是 ParallelDims 分别 materialize 这些 view。

这个演进不是为了增加名字，而是为了把不同层次的语义隔离开：storage state 不必等同于 forward/backward mesh，MoE expert state 也不必复用 dense FSDP 的轴名。替代方案是让每个下游模块自己做轴拼接，但那会重复计算、重复校验，并且容易出现同名轴在不同模块中 degree 不一致。

## 三、代码：从配置对象到 mesh 缓存

### 3.1 先看数据流，而不是先看某一个字段

ParallelDims 的数据流可以压缩成四步：

1. <code>from_config</code> 把训练配置翻译成 ParallelDims 字段。
2. <code>__post_init__</code> 立即调用 <code>_validate</code>，先锁住整数约束。
3. 第一次访问 mesh 时，<code>build_mesh</code> 一次性物化全局 view 和单轴缓存。
4. 下游通过 <code>get_mesh</code>、<code>get_optional_mesh</code> 或 resolver 取到与语义匹配的子 mesh。

这四步的边界很清楚：构造阶段验证“能不能分解”，构造 mesh 阶段验证“各个消费者看什么”，解析阶段验证“当前参数或激活应该激活哪些轴”。不要把最后一步的 placement 逻辑倒推回配置字段。

### 3.2 配置入口和延迟缓存

<code>from_config</code> 只做字段映射，并没有重新发明一套配置语义。<code>__post_init__</code> 则让每个对象在离开构造函数前完成校验；mesh 本身仍然是延迟创建的，因为 <code>world_mesh</code> 只有在第一次访问时才调用 <code>init_device_mesh</code>。

相关入口见 [parallel_dims.py#L68-L100](https://github.com/pytorch/torchtitan/blob/d6555c4c35a10bebcce652b58374cdeb2ecbe527/torchtitan/distributed/parallel_dims.py#L68-L100)。

核心结构可以读成下面这样：

~~~python
@dataclass
class ParallelDims:
    dp_replicate: int
    dp_shard: int
    cp: int
    tp: int
    pp: int
    ep: int
    world_size: int
    spmd_backend: Literal["partial_dtensor", "spmd_types"] = "spmd_types"

    _single_axis_meshes: dict[str, DeviceMesh] = field(default_factory=dict)
    _multi_axis_meshes: dict[tuple[str, ...], DeviceMesh] = field(
        default_factory=dict
    )
    _world_mesh: DeviceMesh | None = None

    def __post_init__(self):
        self._validate()
~~~

这里的三个缓存分别对应：

| 缓存 | 作用 |
| --- | --- |
| <code>_world_mesh</code> | 一维 <code>world</code> mesh，作为所有 view 的坐标底座 |
| <code>_single_axis_meshes</code> | 常用单轴 view，例如 <code>pp</code>、<code>batch</code>、<code>tp</code> |
| <code>_multi_axis_meshes</code> | 按调用者请求顺序缓存的多轴 view |

### 3.3 _validate 先解决可推导值，再检查乘积

校验的第一层是 degree 的下界：<code>dp_replicate</code>、<code>cp</code>、<code>tp</code>、<code>pp</code>、<code>ep</code> 必须至少为 1；<code>dp_shard</code> 允许使用 <code>-1</code> 表示由 world size 推导，但不能是其他负数。

如果 <code>dp_shard == -1</code>，源码计算：

<code>dp_shard = world_size // (dp_replicate * cp * tp * pp)</code>

随后检查：

<code>dp_replicate * dp_shard * cp * tp * pp == world_size</code>

所以 <code>-1</code> 不是“随便找一个最接近的 degree”。整除失败会在后面的全局乘积断言暴露出来。

第二层是 sparse 约束。源码把：

<code>sparse_region = dp_shard * cp * tp</code>

要求 <code>sparse_region % ep == 0</code>，否则抛出 <code>ValueError</code>。这也是为什么 <code>ep=4</code> 时，固定配置的 <code>dp_shard * cp * tp = 8</code> 可以得到 <code>efsdp=2</code>，而不能只用 <code>dp_shard * cp = 4</code> 去算 sparse 侧切分。

校验实现见 [parallel_dims.py#L102-L128](https://github.com/pytorch/torchtitan/blob/d6555c4c35a10bebcce652b58374cdeb2ecbe527/torchtitan/distributed/parallel_dims.py#L102-L128)。

这里有一个工程上的取舍：普通 dense 不满足乘积时使用 <code>assert</code>，sparse 不满足可分性时使用显式 <code>ValueError</code>。前者更像内部不变量，后者则是需要向配置用户解释的 MoE 边界。无论哪一种，文章或实验都不能只打印一个派生值就跳过约束检查。

### 3.4 _mesh_exist 决定是真实 process group 还是 fake axis

<code>build_mesh</code> 不会为每个配置轴都无条件创建真实 mesh。<code>_mesh_exist</code> 对轴是否有效做了一次后端相关判断：

- <code>fsdp</code> 是兼容 dense FSDP view 的特殊名字。
- <code>spmd_types</code> 下，<code>dp_shard</code> 是可见的 storage / state 轴。
- <code>efsdp</code> 只有在 <code>ep &gt; 1</code> 时才有实际 sparse expert 语义。
- 其他轴通常由对应 degree 是否大于 1 决定。

不存在的轴会在 <code>unflatten_mesh</code> 中传给 <code>backend_override</code>，被标记成 <code>fake</code>。这保留了统一的轴名和形状，但 fake process group 不能承载 collective。源码实现见 [parallel_dims.py#L130-L208](https://github.com/pytorch/torchtitan/blob/d6555c4c35a10bebcce652b58374cdeb2ecbe527/torchtitan/distributed/parallel_dims.py#L130-L208)。

因此，“mesh shape 里出现一个 degree 为 1 的轴”和“存在一个可通信的 process group”不是同一件事。后面 <code>get_all_one_dimensional_meshes</code> 还会主动排除 fake mesh。

## 四、代码：build_mesh 如何 materialize 多套 view

### 4.1 先算派生 degree

<code>build_mesh</code> 直接计算三个 degree；<code>loss</code> 的 degree 在后面的 <code>batch</code>、<code>cp</code> flatten 中体现：

~~~python
batch = self.dp_replicate * self.dp_shard
fsdp = self.dp_shard * self.cp
efsdp = fsdp * self.tp // self.ep
~~~

这几行和 loss mesh 的 flatten 共同构成整段实现的代数骨架：

- <code>batch</code> 把复制和切分都纳入 data-loading 轴。
- <code>loss</code> 在 <code>dataloading_mesh["batch", "cp"]._flatten("loss_mesh")</code> 中把 <code>cp</code> 纳入需要聚合的范围。
- <code>fsdp</code> 只描述 partial DTensor dense 侧的参数切分范围。
- <code>efsdp</code> 在 dense sharding 范围上再纳入 <code>tp</code>，最后除以 <code>ep</code>。

<code>efsdp</code> 的公式对应源码 [parallel_dims.py#L210-L218](https://github.com/pytorch/torchtitan/blob/d6555c4c35a10bebcce652b58374cdeb2ecbe527/torchtitan/distributed/parallel_dims.py#L210-L218)。把 <code>tp</code> 从这个公式拿掉，或者另造一个 <code>etp</code> 轴，都会改变当前实现的坐标契约。

### 4.2 dataloading mesh 和 loss mesh

源码首先创建一维 world mesh：

~~~python
world_mesh = init_device_mesh(
    device_type,
    (self.world_size,),
    mesh_dim_names=("world",),
)
~~~

随后把它展开成：

~~~python
dataloading_mesh = world_mesh._unflatten(
    0,
    (self.pp, batch, self.cp, self.tp),
    ("pp", "batch", "cp", "tp"),
)
loss_mesh = dataloading_mesh["batch", "cp"]._flatten("loss_mesh")
~~~

固定配置得到 (2, 4, 2, 2) 的 dataloading mesh；<code>loss_mesh</code> 从其中取 <code>batch</code> 和 <code>cp</code>，因此大小是 <code>4 * 2 = 8</code>。它不是把完整 32-rank mesh 改名，也不是一个隐含的全局 loss collective。

这两处构造见 [parallel_dims.py#L220-L228](https://github.com/pytorch/torchtitan/blob/d6555c4c35a10bebcce652b58374cdeb2ecbe527/torchtitan/distributed/parallel_dims.py#L220-L228)。<code>_unflatten</code> 和 <code>_flatten</code> 都是当前源码直接使用的 DeviceMesh 操作，阅读时要把轴名和 degree 一起看，不能只看变量名。

### 4.3 spmd_types：storage 轴和 runtime 轴分开

在 <code>spmd_types</code> 分支，源码建立两套 dense 相关 view。

第一套用于 storage：

<code>(pp, dp_replicate, dp_shard, cp, tp)</code>

第二套用于 forward/backward：

<code>(pp, dp, cp, tp)</code>

其中 <code>dp</code> 的 degree 是 <code>batch = dp_replicate * dp_shard</code>。随后源码保存：

<code>full_dense_mesh_for_fwdbwd["dp", "cp", "tp"]</code>

作为 <code>spmd_dense_for_fwdbwd</code>。这一步去掉了 <code>pp</code>，因为调用者通常已经处在某个 pipeline stage 内；同一 stage 内的 SPMD runtime 只需要 (dp, cp, tp) 坐标。

完整分支见 [parallel_dims.py#L230-L253](https://github.com/pytorch/torchtitan/blob/d6555c4c35a10bebcce652b58374cdeb2ecbe527/torchtitan/distributed/parallel_dims.py#L230-L253)。这里的关键不是“创建更多 mesh”，而是明确区分：

- <code>fully_shard</code> 看到的 storage axes；
- SPMD type system 看到的 logical <code>dp</code>、<code>cp</code>、<code>tp</code>；
- pipeline 外层看到的 <code>pp</code>。

### 4.4 partial_dtensor：把 dense sharding 压成 fsdp

另一个分支构造：

<code>(pp, dp_replicate, fsdp, tp)</code>

其中：

<code>fsdp = dp_shard * cp</code>

这个分支不建立 <code>spmd_types</code> 专用的 storage/runtime 折叠关系，而是让 dense 参数使用 <code>fsdp</code> view 表达 <code>dp_shard</code> 与 <code>cp</code> 的组合。对应源码 [parallel_dims.py#L254-L260](https://github.com/pytorch/torchtitan/blob/d6555c4c35a10bebcce652b58374cdeb2ecbe527/torchtitan/distributed/parallel_dims.py#L254-L260)。

固定配置下，两个后端的 dense 形状都乘到 32，但轴名不同。形状乘积相同不代表 placement 语义相同；比较后端时必须同时比较 degree 和 axis name。

### 4.5 sparse mesh：四个轴，不再增加一个 etp

无论 dense 后端是哪一种，sparse 侧都构造：

~~~python
full_sparse_mesh = unflatten_mesh(
    (self.pp, self.dp_replicate, efsdp, self.ep),
    ("pp", "dp_replicate", "efsdp", "ep"),
)
~~~

它的四个轴是：

<code>pp -> dp_replicate -> efsdp -> ep</code>

<code>tp</code> 的影响已经通过 <code>efsdp = dp_shard * cp * tp // ep</code> 进入 degree；源码没有 <code>etp</code>。这个边界同时影响 expert 参数 sharding 和 token exchange 的 rank 组织，不能从 dense mesh 的 <code>tp</code> 轴名直接推导出一个 sparse <code>etp</code>。

对应源码 [parallel_dims.py#L262-L265](https://github.com/pytorch/torchtitan/blob/d6555c4c35a10bebcce652b58374cdeb2ecbe527/torchtitan/distributed/parallel_dims.py#L262-L265)。

### 4.6 全局缓存和单轴缓存

<code>_global_meshes</code> 至少包含 dataloading、loss、dense 和 sparse 四类全局 view；启用 <code>spmd_types</code> 时，还会加入 dense forward/backward runtime view；当 <code>ep &gt; 1</code> 时，会加入 sparse forward/backward view。

然后 <code>_single_axis_meshes</code> 保存常用的：

<code>pp, batch, loss, dp_replicate, cp, tp, ep, efsdp</code>

<code>spmd_types</code> 还会加入 <code>dp</code> 和 <code>dp_shard</code>；<code>partial_dtensor</code> 则加入 <code>fsdp</code>。这使得大多数下游模块只需要请求语义明确的单轴 mesh，不必知道完整全局 mesh 的排列。

构造和缓存逻辑见 [parallel_dims.py#L268-L304](https://github.com/pytorch/torchtitan/blob/d6555c4c35a10bebcce652b58374cdeb2ecbe527/torchtitan/distributed/parallel_dims.py#L268-L304)。

### 4.7 _validate_meshes 是第二道证据

配置校验只证明整数能相乘；<code>_validate_meshes</code> 则检查实际 materialized mesh 的 <code>.size()</code>：

| 单轴 | 期望大小 |
| --- | ---: |
| <code>pp</code> | <code>pp</code> |
| <code>batch</code> | <code>dp_replicate * dp_shard</code> |
| <code>loss</code> | <code>dp_replicate * dp_shard * cp</code> |
| <code>dp_replicate</code> | <code>dp_replicate</code> |
| <code>cp</code> | <code>cp</code> |
| <code>tp</code> | <code>tp</code> |
| <code>ep</code> | <code>ep</code> |
| <code>efsdp</code> | <code>dp_shard * cp * tp // ep</code> |

<code>spmd_types</code> 额外检查 <code>dp = dp_replicate * dp_shard</code> 和 <code>dp_shard = dp_shard</code>；<code>partial_dtensor</code> 额外检查 <code>fsdp = dp_shard * cp</code>。实现见 [parallel_dims.py#L306-L329](https://github.com/pytorch/torchtitan/blob/d6555c4c35a10bebcce652b58374cdeb2ecbe527/torchtitan/distributed/parallel_dims.py#L306-L329)。

到这里，固定配置的结构性结论才完整：不是“我算出了几个数字”，而是“源码构造出的 mesh 的真实 size 与每个语义轴的期望一致”。

## 五、代码：从 mesh 获取到 placement resolver

### 5.1 get_optional_mesh 的两条路径

mesh 获取接口分为“必须存在”和“可以不存在”两种语义。

<code>get_optional_mesh</code> 接受一个轴名或轴名列表。它会先确保单轴缓存已经建立，然后：

1. 轴名无效时抛出 <code>ValueError</code>。
2. 请求单轴时返回对应缓存；当前配置没有实际语义的轴可以返回 <code>None</code>。
3. 请求多轴时，在全局 mesh 中寻找包含这些轴的候选，再按请求顺序切出子 mesh。
4. 多轴结果按轴名 tuple 缓存，之后相同顺序的请求复用同一对象。

默认参数 <code>include_singleton_axes=False</code> 会把被当前 <code>_mesh_exist</code> 判为未启用的 size-1 轴过滤掉；在需要给 <code>spmd_types</code> 注册 state 的场景，调用方可以显式保留这些 singleton axes。这个开关只影响返回 view 的筛选，不会把 size-1 轴变成可通信的 collective group。

实现见 [parallel_dims.py#L331-L397](https://github.com/pytorch/torchtitan/blob/d6555c4c35a10bebcce652b58374cdeb2ecbe527/torchtitan/distributed/parallel_dims.py#L331-L397)。

这里有两个容易忽略的细节。第一，列表顺序属于缓存 key，也会影响切片后的轴顺序；<code>["cp", "tp"]</code> 和 <code>["tp", "cp"]</code> 不是同一个请求。第二，多轴请求必须被某一个已经 materialize 的全局 mesh 完整包含。它不会把来自 dense 和 sparse 的轴临时拼成一个新拓扑。

如果调用方要求 mesh 必须存在，则使用：

~~~python
mesh = parallel_dims.get_mesh("loss")
~~~

<code>get_mesh</code> 在 optional 结果为 <code>None</code> 时显式抛出 <code>ValueError</code>，因此缺配置不会静默退化。包装实现见 [parallel_dims.py#L399-L423](https://github.com/pytorch/torchtitan/blob/d6555c4c35a10bebcce652b58374cdeb2ecbe527/torchtitan/distributed/parallel_dims.py#L399-L423)。

### 5.2 dense TP、SPMD dense 和 sparse runtime view

三个辅助入口表达了三个不同边界：

| 接口 | 意义 |
| --- | --- |
| <code>spmd_dense_mesh</code> | 仅在 <code>spmd_types</code> 下取得 dense forward/backward runtime mesh |
| <code>spmd_sparse_mesh</code> | 取得 sparse forward/backward runtime mesh；没有 <code>ep &gt; 1</code> 的语义时可以为空 |
| <code>get_dense_tp_mesh</code> | 后端无关地取得 dense TP 所在的 mesh |

<code>get_dense_tp_mesh</code> 在 <code>spmd_types</code> 下从 <code>spmd_dense_mesh()["tp"]</code> 取 TP 子 mesh，在 <code>partial_dtensor</code> 下直接取单轴 <code>tp</code>。所以它是给 dense TP 相关消费者的统一入口，而不是宣称两个后端拥有同一套 dense state axes。相关实现见 [parallel_dims.py#L425-L441](https://github.com/pytorch/torchtitan/blob/d6555c4c35a10bebcce652b58374cdeb2ecbe527/torchtitan/distributed/parallel_dims.py#L425-L441)。

### 5.3 activated mesh：先按配置过滤，再按语义取子集

<code>get_activated_mesh</code> 接收一组轴，逐个保留当前配置中实际启用的轴，再把剩余轴交给 <code>get_optional_mesh</code>。如果所有轴都被过滤掉，就返回 <code>None</code>。

~~~python
axes = [
    axis
    for axis in axes
    if axis in self._single_axis_meshes
    and self.get_optional_mesh(axis) is not None
]
return self.get_optional_mesh(axes) if axes else None
~~~

这解释了为什么下游可以用一个候选轴列表描述 placement，而不必在每个模块里重复写 <code>if tp &gt; 1</code>、<code>if ep &gt; 1</code>。但它也有一个边界：过滤只保证每个轴单独允许出现，不保证任意组合已经存在于某个全局 mesh。如果请求 <code>spmd_types</code> 下的 <code>dp, cp, tp, ep</code> 全部组合，固定实现中没有一个全局 mesh 同时包含这四个轴，最终仍会在多轴查找阶段失败。

实现见 [parallel_dims.py#L443-L459](https://github.com/pytorch/torchtitan/blob/d6555c4c35a10bebcce652b58374cdeb2ecbe527/torchtitan/distributed/parallel_dims.py#L443-L459)。

### 5.4 resolve_mesh 的后端白名单

<code>resolve_mesh</code> 不是任意轴的别名器。它先根据后端选择允许进入 SPMD placement 的轴：

| 后端 | in-band axes |
| --- | --- |
| <code>spmd_types</code> | <code>dp</code>、<code>cp</code>、<code>tp</code>、<code>ep</code> |
| <code>partial_dtensor</code> | <code>tp</code>、<code>ep</code> |

然后把枚举值转成字符串，只对允许的轴调用 <code>get_activated_mesh</code>。<code>partial_dtensor</code> 不把 <code>dp</code>、<code>cp</code> 作为这条 resolver 的 in-band placement 轴；dense 参数侧的 <code>fsdp</code> 组合已经在它自己的 mesh 构造和 FSDP 入口中表达。

源码见 [parallel_dims.py#L461-L482](https://github.com/pytorch/torchtitan/blob/d6555c4c35a10bebcce652b58374cdeb2ecbe527/torchtitan/distributed/parallel_dims.py#L461-L482)。

### 5.5 resolve_shared_mesh 的匹配是精确的

当多个参数或激活希望共享一个 mesh 时，<code>resolve_shared_mesh</code> 会跳过空 placement，并比较候选的 axis tuple。这里是 tuple equality，因此不仅轴集合要相同，轴顺序也要相同；它不会把一个包含更多轴的 mesh 自动当成共享 mesh。

匹配成功后才对对应的 ParallelDims 调用 <code>resolve_mesh</code>；全部候选都为空时返回 <code>None</code>。这个行为把“能否共享”从隐式猜测变成显式匹配，见 [parallel_dims.py#L484-L512](https://github.com/pytorch/torchtitan/blob/d6555c4c35a10bebcce652b58374cdeb2ecbe527/torchtitan/distributed/parallel_dims.py#L484-L512)。

### 5.6 all-one-dimensional meshes 只暴露可通信的轴

<code>get_all_one_dimensional_meshes</code> 用于收集所有可用的单轴 mesh，但会排除：

- size 为 1 的轴；
- fake process group；
- 不能承载 collective 的占位 view。

因此它更接近“可用于通信的轴清单”，不是“所有配置字段的打印结果”。源码和说明见 [parallel_dims.py#L514-L553](https://github.com/pytorch/torchtitan/blob/d6555c4c35a10bebcce652b58374cdeb2ecbe527/torchtitan/distributed/parallel_dims.py#L514-L553)。

## 六、下游边界：ParallelDims 提供坐标，其他模块执行语义

### 6.1 FSDP 读取 storage axes

FSDP 模块定义了 dense 和 sparse 两组 storage axes：

~~~python
_DENSE_STORAGE_AXES = ["dp_replicate", "dp_shard", "cp", "tp"]
_SPARSE_STORAGE_AXES = ["dp_replicate", "efsdp", "ep"]
~~~

它通过 <code>get_activated_mesh</code> 得到实际 mesh，并把 <code>dp_shard</code> 或 <code>cp</code> 映射成 dense data-parallel sharding 语义；sparse expert 参数则使用 <code>efsdp</code> 和 <code>ep</code>。因此 ParallelDims 中的轴名不是注释性标签，而是 FSDP placement 的输入。参见固定版本的 [fsdp.py#L28-L82](https://github.com/pytorch/torchtitan/blob/d6555c4c35a10bebcce652b58374cdeb2ecbe527/torchtitan/distributed/fsdp.py#L28-L82)。

在 decoder 应用阶段，expert 参数和普通参数还会使用不同的 data-parallel mesh，然后交给 <code>fully_shard</code>。这说明 sparse mesh 不能简单当作 dense mesh 的别名，见 [fsdp.py#L267-L360](https://github.com/pytorch/torchtitan/blob/d6555c4c35a10bebcce652b58374cdeb2ecbe527/torchtitan/distributed/fsdp.py#L267-L360)。

### 6.2 SPMD state 使用 dp 展开规则

SPMD type system 会把逻辑 placement 中的 <code>dp</code> 展开为 <code>dp_replicate</code> 和 <code>dp_shard</code>，再通过 activated mesh 生成 state 对应的 storage view。这个边界正好对应前面所说的“逻辑轴和物理轴分离”，不是把两个名字当作同义词。相关实现见 [spmd_types.py#L54-L127](https://github.com/pytorch/torchtitan/blob/d6555c4c35a10bebcce652b58374cdeb2ecbe527/torchtitan/distributed/spmd_types.py#L54-L127)。

当 sparse state 注册或更新时，runtime 还会设置 sparse mesh；它使用的是 <code>dp_replicate</code>、<code>efsdp</code>、<code>ep</code> 这套边界，而不是在当前 <code>parallel_dims.py</code> 中寻找 <code>etp</code>。参见 [spmd_types.py#L184-L192](https://github.com/pytorch/torchtitan/blob/d6555c4c35a10bebcce652b58374cdeb2ecbe527/torchtitan/distributed/spmd_types.py#L184-L192)。

### 6.3 pipeline、训练器和 compile 只取自己需要的 view

下游模块按职责请求局部 view：

| 下游 | 请求 | 结果 |
| --- | --- | --- |
| trainer 的 token 统计 | <code>get_mesh("batch")</code> | 使用 batch 轴的 global token 计数 |
| trainer 的 loss / pipeline 条件 | <code>get_optional_mesh("loss")</code>、<code>get_optional_mesh("pp")</code> | 按配置启用对应聚合或 pipeline 逻辑 |
| pipeline runtime | <code>get_mesh("pp")</code> | stage 间的 pipeline mesh |
| compile 辅助逻辑 | <code>get_dense_tp_mesh()</code> | dense TP 相关的 mesh |

初始化时，trainer 先创建 ParallelDims；真正的分布式初始化与 ParallelDims 构造边界在 [trainer.py#L628-L637](https://github.com/pytorch/torchtitan/blob/d6555c4c35a10bebcce652b58374cdeb2ecbe527/torchtitan/trainer.py#L628-L637)。使用 batch 和 loss view 的位置见 [trainer.py#L798-L876](https://github.com/pytorch/torchtitan/blob/d6555c4c35a10bebcce652b58374cdeb2ecbe527/torchtitan/trainer.py#L798-L876)；pipeline 使用 <code>pp</code> view 的位置见 [pipeline_parallel.py#L41-L79](https://github.com/pytorch/torchtitan/blob/d6555c4c35a10bebcce652b58374cdeb2ecbe527/torchtitan/distributed/pipeline_parallel.py#L41-L79)。

compile 代码拿到 dense TP 子 mesh，用于相关 async TP 配置，但它不是 TP 算子实现本身，见 [compile.py#L39-L56](https://github.com/pytorch/torchtitan/blob/d6555c4c35a10bebcce652b58374cdeb2ecbe527/torchtitan/distributed/compile.py#L39-L56)。

### 6.4 MoE 的 all-to-all 不在 ParallelDims 中

ParallelDims 只构造并提供 <code>ep</code> 轴和 sparse runtime view。真正的 token dispatch / combine 在 MoE token dispatcher 中执行：路由后交换 token，expert 计算完成后再按原顺序合并。固定版本的 <code>AllToAllTokenDispatcher</code> 使用 <code>all_to_all_single</code> 和 <code>ep_mesh</code>，见 [token_dispatcher.py#L228-L370](https://github.com/pytorch/torchtitan/blob/d6555c4c35a10bebcce652b58374cdeb2ecbe527/torchtitan/models/common/token_dispatcher.py#L228-L370) 以及 combine 路径 [token_dispatcher.py#L556-L610](https://github.com/pytorch/torchtitan/blob/d6555c4c35a10bebcce652b58374cdeb2ecbe527/torchtitan/models/common/token_dispatcher.py#L556-L610)。

MoE 模块把 <code>get_optional_mesh("ep")</code> 传给路由和 expert 相关逻辑；router、dispatch、expert compute、combine 的职责边界见 [moe.py#L171-L180](https://github.com/pytorch/torchtitan/blob/d6555c4c35a10bebcce652b58374cdeb2ecbe527/torchtitan/models/common/moe.py#L171-L180) 和 [moe.py#L338-L355](https://github.com/pytorch/torchtitan/blob/d6555c4c35a10bebcce652b58374cdeb2ecbe527/torchtitan/models/common/moe.py#L338-L355)。

这也是一个重要的证据边界：从 <code>parallel_dims.py</code> 可以确认 EP mesh 的坐标和 degree，不能仅凭它确认 all-to-all 的吞吐、通信代价或 MoE 科学效果。

## 七、代码：序列长度、启用状态和属性不是同一层校验

### 7.1 seq_len_divisor 是给调用者的合同

源码提供：

<code>seq_len_divisor = tp * (cp * 2)</code>

注释说明，TP 需要按 TP degree 做负载平衡，CP 还需要额外的两倍因子以支持它的序列切分方式。这个 property 给出的是调用者需要遵守的除数；它不在 ParallelDims 构造期间读取具体 sequence length，也不负责执行完整的数据形状检查。

相关 property 见 [parallel_dims.py#L555-L609](https://github.com/pytorch/torchtitan/blob/d6555c4c35a10bebcce652b58374cdeb2ecbe527/torchtitan/distributed/parallel_dims.py#L555-L609)。trainer 在更靠近实际 batch / microbatch 参数的位置执行相应检查，而且会根据 sequence parallel 开关和 <code>cp &gt; 1</code> 条件构造实际除数，见 [trainer.py#L292-L305](https://github.com/pytorch/torchtitan/blob/d6555c4c35a10bebcce652b58374cdeb2ecbe527/torchtitan/trainer.py#L292-L305)。

所以不能把“ParallelDims 有 seq_len_divisor”写成“ParallelDims 已经验证了本次输入的 sequence length”。

### 7.2 enabled properties 是派生观察，不是新 mesh

源码还提供：

| property | 当前含义 |
| --- | --- |
| <code>dp_enabled</code> | <code>dp_replicate &gt; 1</code> 或 <code>dp_shard &gt; 1</code> |
| <code>dp_replicate_enabled</code> | <code>dp_replicate &gt; 1</code> |
| <code>dp_shard_enabled</code> | <code>dp_shard &gt; 1</code> |
| <code>cp_enabled</code>、<code>tp_enabled</code>、<code>pp_enabled</code>、<code>ep_enabled</code> | 对应 degree 大于 1 |
| <code>dp_cp_enabled</code> | data parallel 或 context parallel 任一启用 |
| <code>fsdp_enabled</code> | <code>dp_shard &gt; 1</code> 或 <code>cp &gt; 1</code> |
| <code>non_data_parallel_size</code> | <code>cp * tp * pp</code> |

这些属性方便 caller 做分支判断，但不会额外 materialize 一个 mesh，也不能替代 <code>_validate_meshes</code> 的 <code>.size()</code> 检查。

当前这份源码没有 <code>fsdp_gradient_divide_factor</code> property。把其他版本、其他分支或草稿中的同名概念带进这里，会把“并行坐标描述”和“梯度归约缩放策略”混在一起。后者若存在，应以对应训练或优化器源码为事实来源。

## 八、复现：本地实验能证明什么，不能证明什么

### 8.1 推荐的本地入口

本仓库的学习实验和源码基线说明在 [torch/parallel_dims_lab/notes/README.md](notes/README.md)。进入实验目录后，可以执行：

~~~bash
cd torch/parallel_dims_lab
python -m pytest tests/ -q
python 02_mesh_coordinates.py
python 06_parallel_dims_lite.py
python 07_mesh_inspector.py
python 08_moe_router_sim.py
python 10_parallel_config_advisor.py --world-size 32 --dp-replicate 2 --dp-shard 2 --cp 2 --tp 2 --pp 2 --ep 4
~~~

如果需要验证通信演示，再执行：

~~~bash
torchrun --standalone --nproc-per-node=4 01_collectives.py
~~~

最后一个命令只验证 CPU/Gloo 演示的通信逻辑；不能据此推出 GPU 通信性能。其余脚本主要是 CPU tensor 或本地列表模拟，也不测量真实通信吞吐和显存。

### 8.2 固定配置的实际观察

我在当前工作区对上述实验做了最小验证，观察到：

| 检查 | 输出或结果 | 证据等级 |
| --- | --- | --- |
| pytest | <code>9 passed</code> | 本地测试通过 |
| ParallelDims lite | <code>batch=4</code>、<code>loss=8</code>、<code>fsdp=4</code>、<code>efsdp=2</code>、<code>seq_len_divisor=8</code> | 本地结构复现 |
| mesh inspector | dataloading、loss、两种 dense、sparse 的 product 分别为 32、8、32、32、32、32 | 本地形状复现 |
| router simulation | 16 tokens、8 experts、<code>ep=4</code>，四组负载为 <code>[4,4,4,4]</code> | 本地路由模拟 |
| config advisor | dense、EP、sequence checks 均通过 | 本地配置检查 |

这些输出与源码的代数推导相互印证，但实验脚本使用的是学习用简化实现，并没有替代在完整 Torchtitan 依赖和多进程 runtime 中导入真实 <code>parallel_dims.py</code>。因此它们是 structural / proxy evidence，不是官方 benchmark、GPU 性能结果或科学有效性结论。

### 8.3 失败案例也应成为复现的一部分

可以把以下情况作为边界测试：

- 让 <code>world_size</code> 不能被其他 dense degree 整除，确认全局乘积约束失败。
- 让 <code>dp_shard * cp * tp</code> 不能被 <code>ep</code> 整除，确认 sparse 约束失败。
- 在 <code>ep=1</code> 时请求 sparse runtime view，确认 optional 和 required 两种接口的差别。
- 请求一个没有被任何全局 mesh 完整包含的多轴组合，确认它不会被临时拼接。
- 检查 degree 为 1 的 fake axis 不会被当成可通信的 collective group。

这些是实现边界测试，不是性能或收敛性实验。真正的全量运行还需要对应设备、分布式 backend、模型配置和数据依赖，当前没有把这些外部依赖伪装成已验证。

## 九、回看：这段实现真正解决了什么

把整段代码重新压缩成一句话：

> ParallelDims 把一组整数并行配置校验成可分解的全局坐标，再按 dataloading、loss、dense storage、dense runtime 和 sparse expert 的不同语义 materialize 多套命名 mesh，最后通过 optional / required / resolver 接口把恰当的局部坐标交给下游模块。

固定配置的关键链路是：

<code>(2, 2, 2, 2, 2, ep=4) -> batch=4, loss=8, fsdp=4, efsdp=2</code>

然后：

<code>world(32) -> dataloading(pp,batch,cp,tp) -> loss(batch,cp)</code>

以及：

<code>spmd_types: storage(dp_replicate,dp_shard,cp,tp) / runtime(dp,cp,tp)</code>

<code>partial_dtensor: dense(dp_replicate,fsdp,tp)</code>

<code>sparse(dp_replicate,efsdp,ep)</code>

最值得保留的阅读方法不是记住这些名字，而是每次都同时问三件事：

1. 这个轴的 degree 从哪个配置字段或公式来？
2. 这个轴属于哪一个 global mesh，谁会消费它？
3. 当前证据只是源码结构、局部复现，还是已经覆盖真实运行和性能？

在这三问都能回答之前，不应把一个能构造出 mesh 的配置包装成“并行训练已经有效”。

<!--
AUTO-CHECK:
- [x] 使用 code-walkthrough 结构，按 概念 -> 模型/场景 -> 代码 -> 复现 展开
- [x] 源码行为先于解释，源码链接固定到 d6555c4c35a10bebcce652b58374cdeb2ecbe527
- [x] 明确 spmd_types、partial_dtensor、sparse 四轴和 efsdp 公式
- [x] 明确 source / local proxy / runtime / performance 的证据边界
- [x] 未更新 README、README-cn.md 或 knowledge-graph.json
-->
