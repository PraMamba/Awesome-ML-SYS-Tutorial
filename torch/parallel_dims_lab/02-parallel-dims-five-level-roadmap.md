# TorchTitan `ParallelDims` 五级学习路线

## 开始前先校准学习对象

你给出的文章适合帮助建立直觉，但**不能直接当作当前 `main` 分支的逐行源码说明**。截至 **2026 年 8 月 28 日**，需要先修正四点：

1. 当前 `ParallelDims` 的主要输入字段是：

$$
dp\_replicate,\ dp\_shard,\ cp,\ tp,\ pp,\ ep,\ world\_size
$$

当前类中没有独立的 `etp` 字段。([GitHub][1])

2. Dense 拓扑满足：

$$
dp\_replicate\times dp\_shard\times cp\times tp\times pp
=
world\_size
$$

但 `ep` 不再乘进这个等式，而是要求：

$$
ep \mid dp\_shard\times cp\times tp
$$

也就是 `ep` 必须整除可供专家层重新分组的区域。([GitHub][1])

3. 当前代码中：

$$
efsdp=\frac{dp\_shard\times cp\times tp}{ep}
$$

Sparse Mesh 是：

```text
("pp", "dp_replicate", "efsdp", "ep")
```

不是文中的：

```text
("pp", "dp_replicate", "efsdp", "ep", "etp")
```

([GitHub][1])

4. 你文中的 `fsdp_gradient_divide_factor` 属性没有出现在当前 `main` 的 `ParallelDims` 中。因此应把它当成**特定历史版本或其他实现中的专题**，而不是当前源码的核心主线。学习时最好固定一个 commit，而不是一直跟随变化中的 `main`。

此外，“正交”只能理解为**逻辑坐标轴可以组合**，不能理解为各并行算法在性能、通信和约束上完全互不影响。当前 TorchTitan 甚至明确把 CP 与 FSDP 的使用联系在一起。([GitHub][1])

---

## 贯穿五级的固定例子

后面统一使用下面的 32-GPU 配置：

```text
world_size   = 32
dp_replicate = 2
dp_shard     = 2
cp           = 2
tp           = 2
pp           = 2
ep           = 4
```

Dense 维度检查：

$$
2\times2\times2\times2\times2=32
$$

专家区域：

$$
dp\_shard\times cp\times tp=2\times2\times2=8
$$

因为：

$$
8 \bmod 4=0
$$

所以 `ep=4` 合法。

派生维度为：

$$
batch=dp\_replicate\times dp\_shard=4
$$

$$
loss=dp\_replicate\times dp\_shard\times cp=8
$$

$$
fsdp=dp\_shard\times cp=4
$$

$$
efsdp=\frac{fsdp\times tp}{ep}
=\frac{4\times2}{4}=2
$$

因此不同 Mesh 视角可以写成：

```text
dataloading: (pp=2, batch=4, cp=2, tp=2)
dense:       (pp=2, dp_replicate=2, fsdp=4, tp=2)
sparse:      (pp=2, dp_replicate=2, efsdp=2, ep=4)
```

这些不是三批 GPU，而是**同一批 32 张 GPU 的三种分组视角**。([GitHub][1])

---

# 第 1 级：完全新手

## 1. 等级名称

**看懂 GPU 的编号、分组和座位表**

## 2. 这个阶段应该理解什么

先不要学习 TP、FSDP 或 MoE。只理解五个最基础的对象：

* **GPU / Process**：真正执行程序的工作者。
* **Rank**：每个工作者在整个集群中的编号。
* **World Size**：工作者总数。
* **Process Group**：需要互相通信的一组 Rank。
* **DeviceMesh**：用多维坐标管理这些通信组的座位表。

`DeviceMesh` 本质上是管理 `ProcessGroup` 的高层抽象，让程序不必手工计算每一个通信组由哪些 Rank 组成。([PyTorch Docs][2])

例如，将 8 个 Rank 排成：

$$
2\times4
$$

可以画成：

```text
[0, 1, 2, 3]
[4, 5, 6, 7]
```

同一行可以是一组，同一列也可以是另一组。

## 3. 达到这个等级的精通状态

看到：

```python
mesh = init_device_mesh(
    "cuda",
    (2, 4),
    mesh_dim_names=("replicate", "shard"),
)
```

你能够立即说出：

* 总共有 8 个 Rank；
* Mesh 有两个轴；
* 第一个轴大小为 2；
* 第二个轴大小为 4；
* 可以分别取得两个轴对应的通信组。

PyTorch 官方示例也展示了通过 Mesh 轴取得底层 Process Group，以及从大 Mesh 中切出子 Mesh。([PyTorch Docs][2])

## 4. 最需要专注的核心概念或技能

只专注这条转换：

```text
一维 Rank 列表
    ↓
多维坐标
    ↓
沿某个轴形成通信组
```

还要区分：

* **物理拓扑**：GPU 实际在哪台机器、通过什么网络连接。
* **逻辑拓扑**：程序把这些 GPU 命名为哪个并行轴。

DeviceMesh 是逻辑分组工具，它不会自动替你完成 TP、FSDP 或 MoE 算法。

## 5. 准备继续前进的里程碑

给你一个形状为 `(2, 4)` 的 Mesh，你能手工写出：

* 每个 Rank 的二维坐标；
* 每一行有哪些 Rank；
* 每一列有哪些 Rank；
* 某个 Rank 属于哪些组。

## 6. 动手练习

不用 GPU，先写一个纯 Python 小程序：

```python
def rank_to_coord(rank: int, shape: tuple[int, int]) -> tuple[int, int]:
    rows, cols = shape
    assert 0 <= rank < rows * cols
    return rank // cols, rank % cols
```

然后输出：

```text
rank 0 -> (0, 0)
rank 1 -> (0, 1)
...
rank 7 -> (1, 3)
```

再写两个函数：

```text
get_row_group(rank)
get_column_group(rank)
```

## 7. 常犯的错误

* 把 Rank 当作固定的物理 GPU 型号或机器编号。
* 认为 Mesh 的轴名称会自动执行并行算法。
* 混淆“Mesh 轴”和张量自身的 Batch、Sequence、Hidden 维度。
* 认为一张 GPU 只能属于一个通信组。

## 8. 进入下一级前的自测问题

对于：

```text
[0, 1, 2, 3]
[4, 5, 6, 7]
```

Rank 6 的坐标是什么？它所在的行组和列组分别有哪些 Rank？

---

# 第 2 级：基本理解

## 1. 等级名称

**会计算 Dense 模型的五个并行维度**

## 2. 这个阶段应该理解什么

理解每个并行轴解决的主要问题：

### DP Replicate

保存模型副本或参数分片副本，不同组读取不同数据。

### DP Shard

不同 Rank 仍然处理不同数据，但模型参数、梯度和优化器状态会被分片保存。

FSDP2 在计算外保持参数分片；计算前执行参数 All-Gather；反向时用 Reduce-Scatter 得到分片梯度。([PyTorch Docs][3])

### CP

把一条很长的序列分给多个 Rank。

它不是简单地创造更多独立样本，而是把同一条序列中的 Token 分散出去。

### TP

把同一层中的矩阵或张量计算切开。运行时具体使用 All-Reduce、All-Gather 还是 Reduce-Scatter，取决于张量布局和并行方案，并不是每个矩阵乘法都固定执行同一种通信。([PyTorch Docs][4])

### PP

把不同模型层分到不同流水线阶段。

## 3. 达到这个等级的精通状态

给出：

```text
world_size = 32
dp_replicate = 2
dp_shard = 2
cp = 2
tp = 2
pp = 2
```

你能迅速验证：

$$
2\times2\times2\times2\times2=32
$$

并解释每张 GPU 同时拥有五个坐标：

```text
(pp坐标, dp_replicate坐标, dp_shard坐标, cp坐标, tp坐标)
```

## 4. 最需要专注的核心概念或技能

重点掌握三个区别。

### 数据分布和模型分布不同

* DP、CP 主要改变数据或 Token 如何分布。
* TP、PP 主要改变模型计算如何分布。
* FSDP 主要改变参数、梯度和优化器状态如何存储与通信。

### “正交”不等于“完全独立”

逻辑上可以使用不同坐标轴描述，但配置之间仍有约束：

* 总乘积必须等于 `world_size`；
* Sequence Length 需要满足 TP、CP 的切分要求；
* TP 可能受注意力头数约束；
* PP 可能受模型层数约束；
* 不同轴会争夺同一套网络带宽。

### TP 不应直接等同于 TP+SP

PyTorch 官方把 Sequence Parallel 说明为 TP 相关的一种并行方式，主要用于 LayerNorm、RMSNorm 等模块的激活分片。因此，“TP”和“TP 加 SP”关系紧密，但不应直接声明为完全相同的概念。([PyTorch Docs][4])

## 5. 准备继续前进的里程碑

你能够完成一张表：

| 维度           | 切什么      | 主要省什么       | 主要通信什么   |
| ------------ | -------- | ----------- | -------- |
| DP Replicate | 数据       | 不一定省模型显存    | 梯度       |
| DP Shard     | 数据与参数存储域 | 参数、梯度、优化器显存 | 参数和梯度    |
| CP           | Sequence | 激活显存        | 序列相关中间结果 |
| TP           | 层内张量     | 参数和激活显存     | 层内张量     |
| PP           | 模型层      | 单卡模型显存      | 阶段间激活与梯度 |

## 6. 动手练习

写一个配置检查器：

```python
def validate_dense(
    world_size: int,
    dp_replicate: int,
    dp_shard: int,
    cp: int,
    tp: int,
    pp: int,
) -> None:
    product = dp_replicate * dp_shard * cp * tp * pp
    if product != world_size:
        raise ValueError(f"{product=} != {world_size=}")
```

再增加：

```text
batch = dp_replicate × dp_shard
fsdp  = dp_shard × cp
loss  = dp_replicate × dp_shard × cp
```

并让程序打印这些派生维度。

## 7. 常犯的错误

* 认为 DP Shard 只是“切 Batch”。
* 认为 CP 和 TP 中的 Sequence Parallel 是一回事。
* 把 CP 度数直接称为独立样本数。
* 认为乘积合法就表示配置一定能运行。
* 认为 Mesh 轴从右到左必然严格对应 NVLink、RDMA、以太网。
* 认为 PP 的通信一定比所有其他并行方式都小。

## 8. 进入下一级前的自测问题

已知：

```text
world_size = 64
dp_replicate = 2
dp_shard = 4
cp = 2
tp = 2
```

`pp` 应该是多少？`batch`、`fsdp` 和 `loss` 的大小分别是多少？

---

# 第 3 级：实用使用者

## 1. 等级名称

**能够读懂并运行当前 `ParallelDims`**

## 2. 这个阶段应该理解什么

开始按照真实调用顺序阅读源码：

```text
from_config()
    ↓
__post_init__()
    ↓
_validate()
    ↓
build_mesh()
    ↓
get_mesh() / get_optional_mesh()
```

重点不再只是“每个并行方式是什么”，而是：

> 同一批 Rank 为什么需要被构造成多个不同的 Mesh 视角？

当前 `build_mesh()` 会创建数据加载、Loss 汇总、Dense 计算和 Sparse 专家区域等视角。([GitHub][1])

## 3. 达到这个等级的精通状态

你能够解释下面四个名字：

### `batch`

$$
batch=dp\_replicate\times dp\_shard
$$

用于确定哪些 Rank 读取不同的数据。

### `loss`

$$
loss=dp\_replicate\times dp\_shard\times cp
$$

用于汇总分散在不同数据分片和 CP 分片上的 Loss。

### `dense`

Dense 参数和前向、反向计算所使用的 Mesh 视角。

### `sparse`

MoE 专家参数和专家计算所使用的 Mesh 视角。

当前代码还区分：

* `spmd_types`
* `partial_dtensor`

在 `partial_dtensor` 下，`dp_shard` 和 `cp` 被折叠成 `fsdp`；在当前默认的 `spmd_types` 路径下，存储 Mesh 与前向、反向类型检查 Mesh 又被分开构建。([GitHub][1])

## 4. 最需要专注的核心概念或技能

### Flatten 与 Unflatten

把：

```text
world = [0, 1, 2, ..., N-1]
```

重新解释为：

```text
(pp, batch, cp, tp)
```

叫作 Unflatten。

把：

```text
(batch, cp)
```

重新合成：

```text
loss
```

叫作 Flatten。

### 多视角，而不是多份设备

```text
dataloading_mesh
dense_mesh
sparse_mesh
loss_mesh
```

都引用同一套 Rank，只是通信组划分不同。

### 单轴 Mesh 和多轴 Mesh

当前类会缓存单轴和多轴子 Mesh，并通过：

```python
get_mesh(...)
get_optional_mesh(...)
get_all_one_dimensional_meshes()
```

提供给下游算法。尺寸为 1 或使用 Fake backend 的轴不一定拥有可执行集合通信的真实 Process Group。([GitHub][1])

## 5. 准备继续前进的里程碑

你能给 `parallel_dims.py` 添加逐段注释，并准确解释：

* 为什么先创建一维 `world_mesh`；
* 为什么随后反复 `_unflatten()`；
* 为什么 `batch` 和 `loss` 不等于普通模型参数轴；
* 为什么 `get_optional_mesh()` 可能返回 `None`；
* 为什么同一轴大小为 1 时，仍可能因混合精度等原因保留特殊 Mesh。

## 6. 动手练习

编写 `mesh_inspector.py`，输入固定示例：

```text
world_size=32
dp_replicate=2
dp_shard=2
cp=2
tp=2
pp=2
ep=4
```

输出：

```text
batch size
loss size
fsdp size
efsdp size
dataloading mesh shape
dense mesh shape
sparse mesh shape
```

没有 32 张 GPU 时，先做纯 Python 坐标模拟。

之后可以在本机使用 4 个进程测试较小配置：

```text
world_size=4
dp_replicate=1
dp_shard=2
cp=1
tp=2
pp=1
ep=1
```

## 7. 常犯的错误

* 只读文章，不按照真实函数调用顺序读源码。
* 只看 `partial_dtensor` 分支，忽视当前默认 `spmd_types` 分支。
* 认为 `batch` 一定对应真实 NCCL 通信组；当前实现可能用 Fake backend 避免无意义的通信组创建。
* 认为维度大小为 1 就一定不会出现在任何内部结构里。
* 把 `loss` Mesh 误解成“模型参数同步 Mesh”。
* 认为 `DeviceMesh` 本身执行了 FSDP、TP 或 CP。

## 8. 进入下一级前的自测问题

为什么同一组 GPU 需要同时拥有 `dataloading` 和 `dense` 两个 Mesh 视角，而不能只保留一个 Mesh？

---

# 第 4 级：问题解决者

## 1. 等级名称

**能够推导 MoE 的 Sparse Mesh 与 Token 路由**

## 2. 这个阶段应该理解什么

MoE 的关键不是“又增加一个 GPU 维度”，而是：

> Dense 层和专家层，对同一批 GPU 使用不同的分组方式。

当前源码先定义专家可使用的区域：

$$
sparse\_region=dp\_shard\times cp\times tp
$$

要求：

$$
sparse\_region\bmod ep=0
$$

然后计算：

$$
efsdp=\frac{sparse\_region}{ep}
$$

最后构造：

```text
(pp, dp_replicate, efsdp, ep)
```

([GitHub][1])

## 3. 达到这个等级的精通状态

对于固定例子，你能够完整推导：

$$
sparse\_region=2\times2\times2=8
$$

$$
efsdp=8/4=2
$$

所以：

```text
sparse mesh = (pp=2, dp_replicate=2, efsdp=2, ep=4)
```

总 Rank 数仍然是：

$$
2\times2\times2\times4=32
$$

没有增加任何 GPU。

## 4. 最需要专注的核心概念或技能

### EP 是专家分组轴

不同 EP Rank 负责不同的专家分区。

### EFSDP 是专家参数的分片或复制范围

对于同一个专家分区，EFSDP 轴上的 Rank 共同参与其参数管理。

### Token 路由是下游算法，不是 Mesh 构建本身

典型 MoE 会经历：

```text
本地 Token
    ↓
Router 选择专家
    ↓
按 EP 轴重新发送
    ↓
专家计算
    ↓
按原映射发送回来
```

但 `ParallelDims` 只提供 EP 组；真正的 dispatch、combine 和 All-to-All 由 MoE 实现调用。

### 当前 `ParallelDims` 没有独立 ETP 轴

当前 TorchTitan 的模型接入说明仍提到额外的 `apply_moe_ep_tp`，说明专家 TP 可以在模型并行化阶段继续处理，但它不是当前 `ParallelDims` 中独立声明的 Mesh 轴。([GitHub][5])

## 5. 准备继续前进的里程碑

给出任意配置，你能回答：

1. EP 是否合法？
2. EFSDP 是多少？
3. Sparse Mesh 的形状是什么？
4. 哪些 Rank 属于同一个 EP 组？
5. Token 第一次交换和返回交换分别为什么存在？
6. 哪些行为属于 `ParallelDims`，哪些属于 MoE 模型实现？

## 6. 动手练习

写一个 `moe_router_sim.py`。

输入 16 个 Token：

```text
token 0 -> expert 2
token 1 -> expert 0
token 2 -> expert 2
...
```

设置：

```text
ep = 4
num_experts = 8
```

让程序完成：

1. 计算每个专家属于哪个 EP Rank；
2. 将 Token 按目标 Rank 分桶；
3. 输出每个 Rank 需要发送多少 Token；
4. 模拟专家计算；
5. 按原始 Token 顺序还原结果；
6. 统计最忙 Rank 与最闲 Rank 的 Token 数量。

这个练习会让你真正理解：

> EP 的主要困难不仅是专家参数放在哪里，还包括 Token 是否均匀路由。

## 7. 常犯的错误

* 把 `ep` 乘进 Dense 的 `world_size` 等式。
* 认为 EP 创建了新的 GPU。
* 认为 `ep` 必须等于专家总数。
* 认为每个 EP Rank 一定只保存一个专家。
* 把 `etp` 当作当前 `ParallelDims` 的真实字段。
* 认为 Router、Shared Expert 的放置方式由 `ParallelDims` 统一规定。
* 认为创建 Sparse Mesh 就已经完成了 All-to-All。
* 未固定源码版本，就直接套用历史梯度缩放公式。

## 8. 进入下一级前的自测问题

在固定例子中：

$$
dp\_shard\times cp\times tp=8
$$

那么：

* `ep=3` 是否合法？
* `ep=8` 是否合法？
* 当 `ep=8` 时，`efsdp` 是多少？

---

# 第 5 级：自信的实践者

## 1. 等级名称

**能够设计、验证和诊断真实并行配置**

## 2. 这个阶段应该理解什么

这一阶段不再满足于“公式算对”，而要同时处理四类问题：

### 配置正确性

* 世界大小乘积是否合法；
* EP 是否整除专家区域；
* Sequence Length 是否可切分；
* 注意力头数是否满足相关算子要求；
* PP 阶段能否合理分配模型层；
* 专家数量和 EP 度数是否兼容。

当前 TorchTitan 的 `seq_len_divisor` 还将 TP 和默认带负载均衡的 CP 约束组合起来，说明通过 Mesh 验证不代表所有运行约束都已满足。([GitHub][1])

### 数值正确性

不同并行配置应尽量与单卡或较低并行度基线保持：

* 相同前向输出；
* 相同或可解释的 Loss；
* 相同梯度尺度；
* 相近训练曲线。

### 性能正确性

必须观察：

* All-Gather 时间；
* Reduce-Scatter 时间；
* All-Reduce 时间；
* All-to-All 时间；
* PP 空泡；
* GPU 利用率；
* 显存峰值；
* 专家负载是否均衡。

TP 通常更适合高速节点内通信，而 FSDP 常被放到节点间；但这只是常见部署模式，不是只看 Mesh 元组就能保证的硬规则。实际效果取决于 Rank 到物理 GPU 的映射和网络拓扑。([PyTorch Docs][4])

### 源码版本正确性

每次分析都要记录：

```text
repository
branch
commit SHA
PyTorch version
TorchTitan version
configuration
```

不能把不同版本中的：

```text
ParallelDims
Mesh 名称
ETP 支持方式
梯度缩放接口
SPMD backend
```

混在同一套解释中。

## 3. 达到这个等级的精通状态

面对一个真实任务，例如：

```text
2 个节点
每节点 8 张 GPU
共 16 张 GPU

模型：
32 层
32 个注意力头
序列长度 8192
8 个 Routed Experts
```

你能够提出两到三个候选配置，并分别说明：

* 为什么这样分配 DP、TP、CP、PP、EP；
* 哪些通信发生在节点内；
* 哪些通信可能跨节点；
* 显存瓶颈在哪里；
* All-to-All 是否可能成为瓶颈；
* 有哪些整除要求；
* 应该怎样验证数值等价性；
* 应该采集哪些性能指标。

## 4. 最需要专注的核心概念或技能

建立一张完整的约束矩阵：

| 类别   | 需要检查的内容                        |
| ---- | ------------------------------ |
| 集群   | GPU 数量、每节点 GPU 数、NVLink、RDMA   |
| 模型   | 层数、Hidden Size、头数、专家数          |
| 数据   | Batch、Sequence Length、Token 数量 |
| Mesh | DP、CP、TP、PP、EP、EFSDP           |
| 数值   | Loss、梯度尺度、参数更新                 |
| 性能   | 通信量、空泡、显存、负载均衡                 |
| 版本   | Commit、配置格式、后端实现               |

同时建立一个重要习惯：

> 不要只从 Mesh 公式推断训练行为，要继续追踪下游是谁取得这个 Mesh、调用了什么集合通信、如何缩放 Loss 和梯度。

## 5. 达到最终等级的里程碑

完成一份“并行配置审计报告”，至少包含：

1. Dense 乘积校验；
2. EP 整除校验；
3. 所有派生 Mesh 的形状；
4. 每个 Mesh 的用途；
5. Rank 到物理节点的映射；
6. 预期集合通信；
7. Sequence、Head、Layer、Expert 约束；
8. 单卡或低并行度数值基线；
9. 性能分析结果；
10. 可能的失败模式；
11. 固定的源码 commit。

## 6. 动手练习或迷你项目

编写一个 `parallel_config_advisor.py`。

输入：

```text
world_size
gpus_per_node
num_layers
num_heads
hidden_size
seq_len
num_experts
candidate dp/cp/tp/pp/ep
```

输出：

```text
是否合法
失败原因
batch / loss / fsdp / efsdp 大小
dataloading / dense / sparse mesh 形状
可能的节点内轴
可能的跨节点轴
可能的主要通信瓶颈
还需要下游检查的模型约束
```

然后为一个小型模型做两次运行：

```text
配置 A：无 EP
配置 B：启用 EP
```

比较：

* 初始权重相同时的前向输出；
* Loss；
* 每层梯度范数；
* 参数更新差异；
* 通信时间。

对于你文中的梯度归一化问题，正确的学习方法不是先假定“必然放大多少倍”，而是：

1. 固定确切 commit；
2. 找到 Loss reduction 的位置；
3. 找到 FSDP Reduce-Scatter 的缩放位置；
4. 找到专家 Token dispatch 后每个参数梯度的来源；
5. 用单卡或无 EP 基线做梯度对比；
6. 再推导实际除数。

## 7. 常犯的错误

* 认为公式合法就等于配置优秀。
* 只检查显存，不检查网络通信。
* 只看平均 Token 数，不看最忙专家。
* 只看训练是否运行，不检查数值等价性。
* 认为所有 All-to-All 都只沿一个简单的二维轴发生。
* 用“TP 最内、PP 最外”一句话代替真实硬件分析。
* 将历史版本属性误写成当前 API。
* 把 Router、Shared Expert、Loss 缩放等下游行为归因于 `ParallelDims`。
* 直接引用 `main`，但不记录 commit SHA。

## 8. 最终自测问题

下面两个配置都满足 GPU 总数乘积：

```text
配置 A：
dp=8, tp=2, cp=1, pp=1

配置 B：
dp=2, tp=4, cp=2, pp=1
```

为什么不能只根据乘积判断哪个配置更好？你还必须知道哪些模型、数据、硬件和通信信息？

---

# 五个自测题参考答案

### 第 1 级

Rank 6 的坐标是：

$$
(1,2)
$$

所在行组：

```text
[4, 5, 6, 7]
```

所在列组：

```text
[2, 6]
```

### 第 2 级

$$
pp=\frac{64}{2\times4\times2\times2}=2
$$

$$
batch=2\times4=8
$$

$$
fsdp=4\times2=8
$$

$$
loss=2\times4\times2=16
$$

### 第 3 级

因为数据加载关心“谁读取不同样本、谁读取同一序列的不同部分”，Dense 计算则关心“参数和层内计算如何分片”。两者使用同一批 GPU，但需要不同的分组语义。

### 第 4 级

* `ep=3`：不合法，因为 \(8\bmod3\neq0\)。
* `ep=8`：合法。
* 此时：

$$
efsdp=8/8=1
$$

### 第 5 级

还必须比较：

* 单卡显存需求；
* Sequence Length；
* 注意力头数；
* 模型层数；
* 每节点 GPU 数量；
* NVLink 与节点间网络；
* TP、CP、FSDP 的通信代价；
* Local Batch Size；
* 是否为 MoE；
* 数值一致性和实际 profiling 结果。

---

# 最推荐的学习顺序

不要直接从文章最后的“专家梯度归一化”开始。最稳妥的顺序是：

```text
Rank 与 Process Group
    ↓
DeviceMesh 坐标和子 Mesh
    ↓
Dense 的 DP / CP / TP / PP
    ↓
ParallelDims 的 validate 与 build_mesh
    ↓
dataloading / loss / dense 多视角
    ↓
EP 与 sparse mesh 重组
    ↓
真实 Token dispatch
    ↓
下游 FSDP、Loss 和梯度缩放
    ↓
性能与数值诊断
```

完成五个对应成果即可视为真正掌握：

```text
1. Rank 坐标模拟器
2. Dense 配置检查器
3. ParallelDims Mesh Inspector
4. MoE Token Router 模拟器
5. Parallel Config Advisor 与审计报告
```

[1]: https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/parallel_dims.py "torchtitan/torchtitan/distributed/parallel_dims.py at main · pytorch/torchtitan · GitHub"
[2]: https://docs.pytorch.org/tutorials/recipes/distributed_device_mesh.html "Getting Started with DeviceMesh — PyTorch Tutorials 2.13.0+cu130 documentation"
[3]: https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html "Getting Started with Fully Sharded Data Parallel (FSDP2) — PyTorch Tutorials 2.13.0+cu130 documentation"
[4]: https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html "Large Scale Transformer model training with Tensor Parallel (TP) — PyTorch Tutorials 2.13.0+cu130 documentation"
[5]: https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/README.md?utm_source=chatgpt.com "torchtitan/torchtitan/models/README.md at main · pytorch ..."
