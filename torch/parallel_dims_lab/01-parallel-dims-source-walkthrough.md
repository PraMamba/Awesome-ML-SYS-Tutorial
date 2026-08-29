本文基于 torchtitan 官方仓库的 ParallelDims 实现。源文件地址：

[torchtitan/distributed/parallel_dims.py](https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/parallel_dims.py)

在分布式大模型训练中，将一维的物理 GPU 集群映射为多维的逻辑通信网格（DeviceMesh）是所有并行策略的基础。torchtitan 的 ParallelDims 类提供了这一层抽象。本文将从源码出发，剥离顶层算法，深入解析多维并行拓扑的构建逻辑，分为“正交维度”与“非正交的专家维度”两部分进行探讨。

# 完全正交的并行维度构建 (PP, DP, TP, CP)

## 1. Dense 模型的五个正交维度

在标准的密集型（Dense）Transformer 模型中，流水线并行（PP）、数据并行（DP_Replicate / DP_Shard）、张量并行（TP）和上下文并行（CP）是完全正交的。

在未进行任何并行切分时，模型的前向过程可抽象为 $y = model(x)$。其中，输入数据 $x$ 具有 Batch 和 Sequence 两个核心维度；模型结构具有宽度（Hidden Size）和深度（Layer Number）两个核心维度。**多维并行的本质是对这四个维度进行正交切分**：

- 数据维度切分 
- 切分 Batch 维度：构成传统的数据并行 DP（包含 DP_Replicate 和 DP_Shard）。
- 切分 Sequence 维度：构成上下文并行 CP。
- 模型维度切分 
- 切分模型宽度（Hidden Size）：构成张量并行 TP。
- 切分模型深度（Layer Number）：构成流水线并行 PP。

代码中的 `_validate` 方法强制校验了这五个正交维度的乘积必须严格等于全局可用物理显卡的数量 `world_size`，任何维度的调整在数学上均相互独立：

```python
def _validate(self):
        # ... 
        assert dp_replicate * dp_shard * cp * tp * pp == self.world_size, (
            f"Invalid parallel dims: dp_replicate({dp_replicate}) * dp_shard({dp_shard}) * "
            f"cp({cp}) * tp({tp}) * pp({pp}) != WORLD_SIZE({self.world_size})"
        )
```

## 2. 算力与通信带宽驱动的 Mesh 嵌套层级

`DeviceMesh` 维度的内外嵌套顺序并非随意设定，而是直接映射底层的物理硬件拓扑。排序原则严格遵循通信频率与单次通信数据量，将通信最密集的维度放置在最内层（享有最高带宽，如 NVLink），通信频率最低的放置在最外层（容忍高延迟，如以太网）。从内向外（通信频率由高到低）的抽象顺序为：`"tp" -> "cp" -> "dp_shard" -> "dp_replicate" -> "pp"`。

在代码中，这体现在 `dense_mesh` 的重塑过程。该网格专门用于前向参数计算与反向梯度同步：

```python
        # 用于参数计算与梯度同步的网格 
        # "fsdp" = "dp_shard" * "cp"
        dense_mesh = unflatten_mesh(
            self._world_mesh,
            ("pp", "dp_replicate", "fsdp", "tp"),
            (self.pp, self.dp_replicate, fsdp, self.tp),
        )
```

从上述元组 ("pp", "dp_replicate", "fsdp", "tp") 由右至左（即物理拓扑由内向外）看：

- **TP（张量并行，最内层）**： 每次矩阵乘法都需要 All-Gather和Reduce-Scatter（或 All-Reduce），通信极度密集，延迟容忍度几乎为零。必须映射在节点内（如单机 8 卡的 NVLink 网络）。
- **FSDP（参数切分与上下文域，中内层）**： 这里将 dp_shard 和 cp 折叠为了统一的 fsdp 维度（$fsdp = dp\_shard \times cp$）。无论是切分数据的 Sequence 维度（CP），还是切分优化器状态/参数（DP_Shard），都需要在每层计算时进行通信。若节点内带宽有冗余优先分配给此层，否则跨节点依赖高带宽的 RDMA 网络。
- **DP_Replicate（数据复制，中外层）**： 纯粹的数据并行复制维度。仅在反向传播计算结束时执行一次梯度的 All-Reduce，对网络延迟不敏感。
- **PP（流水线并行，最外层）**： 仅在层级边界进行 Point-to-Point 的激活值与梯度传递，通信频率极低（每微批次两次），带宽要求最低。

## 3. 拓扑隔离：Dataloading 与 Dense Mesh 的双视角映射

模型训练的首要步骤是数据分发。数据加载器（Dataloader）并不关心底层的参数是如何切分（FSDP）或复制（DDP）的，它只需要知道全局的 Batch 和 Sequence 应该如何路由给各个 GPU。

为此，`ParallelDims` 针对相同的一维物理设备资源，构建了另一个独立视角的逻辑网格：`dataloading_mesh`。

```text
        # 用于数据分发的网格 
        # "batch" = "dp_replicate" * "dp_shard"
        dataloading_mesh = unflatten_mesh(
            self._world_mesh,
            ("pp", "batch", "cp", "tp"),
            (self.pp, batch, self.cp, self.tp),
        )
```

`dataloading_mesh` 的作用解析：

- 在该网格中，dp_replicate 和 dp_shard 被合并为统一的 batch 维度。在该维度上的各个 Rank 会接收到完全独立的不同样本。
- cp 维度决定了单个样本内的 Sequence 将被继续切分给多少个 Rank。
- 位于同一个 tp 和 pp 组内的 Rank，实际上必须读取完全相同的数据，因为它们处理的是同一份样本的不同模型切片或不同网络层。dataloading_mesh 提供了一个清晰的坐标系，使得数据并行采样器可以精确计算每个设备的全局数据索引。

## 其它注意点

**CP 的实现无感知：** 拓扑层中的 cp 维度不区分 Ulysses CP、Ring CP 或两者的混合部署（USP）。具体的 CP 计算模式是在模型推理算子层定义的。例如，Ulysses CP 受限于算法物理瓶颈，要求 $TP \times CP$ 必须能被注意力头数整除。这种基于模型架构超参数（如 Head Number）的校验，不会在底层的 ParallelDims 拓扑初始化中进行拦截，从而保证了通信组的通用性。

**这里的TP就是Megatron中的TPSP：** TPSP 主要是 Megatron-LM 论文中提出的概念（即 Tensor Parallelism + Sequence Parallelism）。需要注意的是，这里的SP和CP（在sequence维度切分数据）是没关系的，其实是TP的一种改进形式，使得TP的通信从 All-Reduce变为 All-Gather + Reduce-Scatter。 TPSP没有单独的SP通信组，TP和SP公用通信组。Megatron提出的TPSP概念跟这里的TP是等价的。

# 引入专家并行 (EP) 的非正交重组

当大模型引入混合专家架构（Mixture of Experts, MoE）时，严格的维度正交性被打破。专家并行（EP）并未创造新的物理维度，而是对现有Mesh的“时分复用”。

## 1. 物理拓扑的动态切换与 Sparse Mesh

在 MoE 模型的前向传播中，数据的物理流向会发生拓扑级别的动态切换：

- 当计算流经密集层（Dense Layer，如 Attention 模块）时，硬件拓扑严格遵循上一部分定义的 `dense_mesh`，即 `("pp", "dp_replicate", "fsdp", "tp")`。
- 当计算流经专家层（MoE Layer）时，部分维度必须被重构以支持专家的分布式路由。系统将硬件拓扑重定义为 `sparse_mesh`。

在 ParallelDims 的代码实现中，`sparse_mesh` 的构建逻辑如下：

```text
        # 针对 MoE 层的 Sparse Mesh
        sparse_mesh = unflatten_mesh(
            self._world_mesh,
            ("pp", "dp_replicate", "efsdp", "ep", "etp"),
            (self.pp, self.dp_replicate, efsdp, self.ep, self.etp),
        )
```

在这个新的网格中，原有的 fsdp 和 tp 维度被拆解并重组为 efsdp、ep 和 etp。 这三个维度相互正交，且嵌套层级由内向外依次为：

- etp（专家内张量并行，最内层）： 进一步切分单个专家的宽度，置于最高速的网络拓扑中以支撑高频的矩阵切分通信。
- ep（专家并行，中层）： 将不同专家分布到不同的物理节点上，承担 Token 的 All-to-All 路由。
- efsdp（专家特有 FSDP 域，相对外层）： 负责在 MoE 层内部应用 ZeRO 技术，进行参数和状态分片。

在这个新的网格中，原有的 fsdp 和 tp 维度被拆解并重组为 `efsdp`（专家层特有 FSDP 域）、`ep`（专家并行度）和 `etp`（专家内张量并行度），这里`ep`对让不同专家分部到不同的rank上，`etp`进一步让同一个专家分到不同的`rank`上，都是MOE层模型宽度切分。`efsdp`是在MOE层用ZERO技术。

## 2. 资源守恒与 MoE 组件的异构切分

维度的重组并非随意设定，而是严格遵循底层算力资源守恒定律。物理集群的 GPU 总数固定，MoE 层必须复用密集层所占用的硬件资源，以保持整体计算吞吐量的对等。

在密集层中，参与参数分片和张量计算的总设备度数为 $fsdp \times tp$。进入专家层后，为了承载多个专家的海量参数，总设备数被重分配给 $ep$、$etp$ 以及 $efsdp$。由此得出资源守恒等式：

$$
fsdp \times tp = efsdp \times ep \times etp \\
$$

代码中有：

```text
        fsdp = self.dp_shard * self.cp
        efsdp = fsdp * self.tp // (self.etp * self.ep)
```

需注意，MOE层内部的不同组件的并行方式并不完全一样：

- **Routed Experts（路由专家）**： 使用 `sparse_mesh`。通过 ep 将不同专家分布到不同物理节点，通过 etp 切分单个专家的张量宽度，通过 efsdp 切分优化器状态。
- **Shared Experts（共享专家）**： 处理所有无差别 Token，无需路由。其物理行为完全等同于普通的 Dense MLP 层，因此直接回退使用 dense_mesh（即 tp 和 fsdp）。
- **Router（路由门控）**： 负责为所有 Token 计算分配概率。由于其权重矩阵极小（通常为 $Hidden\_Size \times Num\_Experts$），为避免额外的集合通信开销，Router 不参与张量并行（TP）切分，在 tp 网格内全量复制；在 fsdp 网格中，其静态存储通常跟随 Dense 层进行 ZeRO 切片，并在前向计算时 All-Gather 为全量参数。这意味着在执行路由计算时，整个Dense计算域的所有设备都持有完整的 Router 副本。

## 3. 动态通信代价：基于 ep 维度的 All-to-All 路由

维度的非正交重组带来了显式的网络通信代价。当数据流从 Dense Mesh 切换到 Sparse Mesh 时，各个 GPU 上的 Token 必须根据 Router 网络的门控得分（Gating Score）进行重新分发。

这一过程精准地触发了跨 `ep` 通信组的 All-to-All 集合通信，而非在整个混合网格中穿插。 当前节点将打乱的 Token 沿 ep 维度发送给负责特定专家的目标节点；目标节点上的专家网络（由 `etp` 组内的多张卡共同完成局部张量计算）处理完毕后，再通过第二次 All-to-All 通信将输出特征收回原位，以恢复 Dense Mesh 的数据排布形态并进入下一层。这种剧烈的拓扑切换构成了 MoE 模型横向扩展时的核心网络瓶颈。

## 4. 梯度对齐与归一化：为什么需要fsdp_gradient_divide_factor？

在分布式训练中，梯度的全局归一化（取平均）由“本地损失缩放”和“分布式求和求平均”两步完成。由于 MoE 架构在 Dense 层与专家层之间引入了跨层的异构拓扑重组，底层通信组大小发生突变，导致框架默认的梯度平均逻辑在专家层会产生严重的数值错位。

**Dense 层的默认梯度归一化逻辑**

设定单卡本地 Batch Size 为 $B$。通常损失函数默认按本地 Batch 求平均（reduction='mean'），这意味着反向传播算出的本地梯度天然带有一个 $\frac{1}{B}$ 的缩放因子。 在 Dense 层中，进行梯度同步的通信组大小为 fsdp（即 $DP\_Shard \times CP$）。底层的 Reduce-Scatter 会将组内各卡的梯度求和，并默认除以组大小 $N_{fsdp}$。 最终得到的全局等效梯度为 $\frac{1}{N_{fsdp} \times B} \sum \nabla loss$，完美等价于对全局所有 Token 求严格的算术平均。Dense 层的原生归一化逻辑是自洽的。

**MoE 专家层的梯度放大问题**

进入 MoE 层后，硬件拓扑切换为 Sparse Mesh。单个专家的参数分片与梯度同步不再发生于全局大网格，而是退化到局部的 efsdp 通信组内。

根据守恒等式 $efsdp \times ep \times etp = fsdp \times tp$，随着专家数量（$ep$）的增加，单个专家的通信组大小 efsdp 会急剧缩小，必然远小于全局的数据并行度大小。

假设全局有 8 张卡在同步提供训练数据（即全局数据并行总度数为 8），设定每张卡读取的本地 Batch Size 为 $B$。由于切分策略，某个特定专家仅被部署在其中 2 张卡上（即 $efsdp=2$）。

1. 数据与梯度的物理来源： 前向传播时，全局 8 张卡上总计 $8 \times B$ 个 Token 中，有一部分子集通过 All-to-All 路由到了这 2 张卡上的专家网络。反向传播时，最终的 Loss 是基于各卡本地 Batch $B$ 求平均得出的，因此沿着网络反传回专家层的梯度，天然自带了初始的 $\frac{1}{B}$ 缩放因子。**这 2 张卡分别算出的本地梯度，加起来已经包含了全局完整数据对该专家的全部梯度贡献**。
2. 底层同步的错误除法： 如果不加干预，底层 FSDP 框架会在该专家的 efsdp 组内触发 Reduce-Scatter 对本地梯度求和，并默认除以当前的组大小 2。
3. 最终数学错位： 专家层最终得到的等效梯度为 $\frac{1}{2 \times B} \sum \nabla loss$。而真实的全局归一化期望值应当是 $\frac{1}{8 \times B} \sum \nabla loss$。

这意味着，在未改变全局真实 Token 样本基数（$8 \times B$）的前提下，专家层的参数更新梯度在绝对数值上被异常放大了 4 倍（$8 / 2 = 4$）。这种错误的除法逻辑，导致开启专家并行（EP）后的梯度计算结果，与不开启 EP（即完全使用数据并行）时的真实数学期望完全不一致。这种数值错误会直接放大该层的实际学习率，最终引发模型训练发散。

**`fsdp_gradient_divide_factor` 的强制对齐**

为了消除这种由物理拓扑引起的数值错位，必须在代码中显式覆盖底层默认的除法逻辑。

```text
    @property
    def fsdp_gradient_divide_factor(self) -> int:
        # This is needed for FSDP-sharded experts when Expert Parallel is enabled.
        # Although the FSDP sharding of experts is done on a mesh of a different size than
        # other parameters, the gradient division factor should be consistent with data.
        return self.dp_replicate * self.dp_shard * self.cp
```

该参数的物理意义是：**无视局部网格的大小，强制将梯度归一化的分母锁定为全局真实的数据并行总度数**。

无论 Token 被路由到哪个专家，处理全局 Batch 的基数是不变的。在专家层的梯度 All-Reduce 或 Reduce-Scatter 完成求和后，框架将丢弃默认的 efsdp 组大小，强行除以 dp_replicate * dp_shard * cp。这保证了 MoE 专家层与 Dense 层的梯度在分布式前后是一致的。

**TP 与 ETP 对梯度归一化的正交性（无关性）**

梯度的归一化除数，只与“数据被切分了多少份”有关，与“模型被切分了多少份（如 TP/ETP）”完全无关。

张量并行（无论是 Dense 层的 tp 还是专家层的 etp）是对权重矩阵的纵向或横向切分。在同一个 etp 组内的不同显卡，持有的是单个专家不同部分的权重切片。反向传播时，它们分别计算自身对应参数的本地梯度片段，在物理上完全独立，绝对不会在 etp 组内进行梯度的聚合或平均。