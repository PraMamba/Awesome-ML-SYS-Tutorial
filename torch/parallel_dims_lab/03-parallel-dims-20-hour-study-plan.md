# TorchTitan `ParallelDims`：20 小时高效学习计划

## 一、20 小时后的目标

这 20 小时不是让你学完整个分布式训练领域，而是优先获得五项真实能力：

1. 看到 `dp_replicate / dp_shard / cp / tp / pp / ep` 配置后，能解释每张 GPU 在做什么。
2. 能手工或用程序推导 `batch`、`loss`、`fsdp`、`efsdp` 以及各个 Mesh 的形状。
3. 能顺着源码读懂 `_validate()`、`build_mesh()` 和 `get_mesh()`。
4. 能区分“Mesh 只负责分组”和“下游算法真正执行通信”。
5. 能在启动昂贵任务前，发现配置错误、数值风险和潜在通信瓶颈。

本计划以 **2026 年 8 月 28 日的 TorchTitan `main` 分支**为学习对象。当前 `ParallelDims` 包含 `dp_replicate、dp_shard、cp、tp、pp、ep、world_size`，没有独立 `etp` 字段；Dense 维度乘积必须等于 `world_size`，而 `ep` 要整除 `dp_shard × cp × tp`。当前文件也没有 `fsdp_gradient_divide_factor` 属性。([GitHub][1])

开始学习时先固定源码版本：

```bash
git clone https://github.com/pytorch/torchtitan.git
cd torchtitan
git rev-parse HEAD | tee LEARNING_COMMIT.txt
```

这样未来 `main` 分支改变时，你仍然知道自己学的是哪个版本。

---

# 二、最有价值的 20%

| 高杠杆内容                        | 为什么重要                        | 现实用途                                  |
| ---------------------------- | ---------------------------- | ------------------------------------- |
| Rank、ProcessGroup、Collective | 所有分布式并行最终都落到“哪些 Rank 进行什么通信” | 排查通信组错误、Hang、结果不同步                    |
| DeviceMesh 坐标系统              | 把复杂通信组变成可组合的逻辑坐标轴            | 看懂多维并行配置和源码                           |
| 每个并行轴到底切什么                   | 防止把 DP、FSDP、TP、SP、CP、PP 混在一起 | 配置并行方案、解释显存和通信                        |
| Mesh 乘积与派生维度                 | 大量配置错误可以在运行前直接发现             | 编写配置检查器                               |
| 同一批 GPU 的多个 Mesh 视角          | 这是 `ParallelDims` 最核心的设计     | 看懂 dataloading、loss、dense、sparse Mesh |
| Dense 与 Sparse Mesh 重组       | 这是理解 EP 的关键                  | 看懂 MoE 专家分布与 Token 路由                 |
| 数值与性能审计                      | “能跑”不等于“算对”或“跑得快”            | 比较并行配置、排查梯度与通信问题                      |

`DeviceMesh` 的核心价值，就是替用户管理底层 ProcessGroup，并为跨节点和节点内的多维并行提供统一坐标系统。([PyTorch Docs][2])

## 20 小时内暂时不深挖

以下内容先达到“知道它存在”即可：

* NCCL Ring/Tree 算法的底层实现；
* 所有 Ulysses、Ring Attention、USP 变体；
* 所有 Pipeline Schedule；
* SPMD 类型系统的内部推导算法；
* DeepEP、Grouped GEMM、专家负载均衡算法的完整实现；
* `torch.compile`、Float8、Checkpoint、Slurm；
* 未固定版本时对历史 `fsdp_gradient_divide_factor` 的详细推导。

---

# 三、统一学习方法

每个单元固定为 2 小时：

```text
前 30 分钟：阅读官方资料，整理概念
中间 65 分钟：完成实践练习
接着 15 分钟：用自己的语言写解释
最后 10 分钟：回答 5 个复习问题
```

建议建立下面的练习目录：

```text
parallel_dims_lab/
├── 01_collectives.py
├── 02_mesh_coordinates.py
├── 03_dp_fsdp_sim.py
├── 04_tp_mlp_sim.py
├── 05_cp_pp_planner.py
├── 06_parallel_dims_lite.py
├── 07_mesh_inspector.py
├── 08_moe_router_sim.py
├── 09_gradient_accounting.py
├── 10_parallel_config_advisor.py
├── tests/
└── notes/
```

贯穿全程使用固定例子：

```text
world_size   = 32
dp_replicate = 2
dp_shard     = 2
cp           = 2
tp           = 2
pp           = 2
ep           = 4
```

对应：

$$
batch=2\times2=4
$$

$$
loss=2\times2\times2=8
$$

$$
fsdp=2\times2=4
$$

$$
efsdp=\frac{2\times2\times2}{4}=2
$$

---

# 单元 1：Rank、ProcessGroup 与集合通信

## 主要学习目标

理解分布式程序最底层到底是什么：

> 多个独立进程，各自拥有 Rank，通过 ProcessGroup 进行通信。

## 核心概念

* Process：一个独立运行的训练进程。
* Global Rank：进程在整个任务中的编号。
* Local Rank：进程在当前节点内的编号。
* World Size：总进程数。
* ProcessGroup：允许一组 Rank 相互通信的组。
* Collective：

  * All-Reduce；
  * All-Gather；
  * Reduce-Scatter；
  * All-to-All；
  * Broadcast；
  * Point-to-Point Send/Recv。

PyTorch Distributed 使用消息传递方式让不同进程通信；ProcessGroup 和集合通信是其底层基础。([PyTorch Docs][3])

## 实践练习

编写 `01_collectives.py`：

1. 用 CPU 和 Gloo 启动 4 个进程；
2. 每个 Rank 创建：

```python
x = torch.tensor(float(rank))
```

3. 执行 All-Reduce Sum；
4. 验证所有 Rank 最终都得到：

$$
0+1+2+3=6
$$

运行：

```bash
torchrun --standalone --nproc-per-node=4 01_collectives.py
```

进阶一步：创建只包含 `[0, 2]` 的子 ProcessGroup，观察 Rank 1 和 Rank 3 为什么不能参与该组通信。

## 推荐资源

PyTorch 官方教程 **Writing Distributed Applications with PyTorch**，重点阅读 Setup 和 Collective Communication。([PyTorch Docs][3])

## 完成后的预期成果

你能够准确解释：

```text
谁在计算？
谁和谁通信？
通信结果放在哪里？
```

而不是只说“用了多卡”。

## 复习问题

1. `rank` 和 `local_rank` 有什么区别？
2. `world_size=8` 是否一定表示一台机器有 8 张 GPU？
3. ProcessGroup 解决了什么问题？
4. All-Reduce 与 All-Gather 的输出有什么根本区别？
5. 为什么同一 ProcessGroup 中的 Rank 必须以兼容的顺序调用集合通信？

---

# 单元 2：DeviceMesh——给 Rank 建立坐标系

## 主要学习目标

理解 DeviceMesh 不是并行算法，而是：

> 对 Rank 进行多维编号，并自动生成对应通信组。

## 核心概念

将 8 个 Rank 排成：

```text
[0, 1, 2, 3]
[4, 5, 6, 7]
```

Mesh 形状是：

$$
(2,4)
$$

Rank 6 的坐标为：

$$
(1,2)
$$

它同时属于：

```text
行组：[4, 5, 6, 7]
列组：[2, 6]
```

每张 GPU 可以同时属于多个通信组，因为它在多个逻辑轴上都有坐标。

DeviceMesh 是管理 ProcessGroup 的高层抽象，可以构造节点内和节点间子组，并允许从多维 Mesh 中取得子 Mesh。([PyTorch Docs][2])

## 实践练习

编写 `02_mesh_coordinates.py`：

```python
def rank_to_coord(rank, shape):
    ...

def coord_to_rank(coord, shape):
    ...

def axis_group(rank, shape, axis):
    ...
```

程序应支持：

```text
shape = (2, 4)
rank = 6
```

并输出：

```text
coordinate = (1, 2)
axis 0 group = [2, 6]
axis 1 group = [4, 5, 6, 7]
```

进阶：支持任意形状，例如：

```text
(2, 2, 2)
(2, 4, 2)
```

## 推荐资源

PyTorch 官方 **Getting Started with DeviceMesh**。([PyTorch Docs][2])

## 完成后的预期成果

你能够看见：

```python
mesh_dim_names=("dp", "tp")
```

就想到：

> 这是在同一组 Rank 上定义两个通信方向，不是创建了两批 GPU。

## 复习问题

1. DeviceMesh 与 ProcessGroup 是什么关系？
2. Mesh 形状 `(2,4)` 为什么需要 8 个 Rank？
3. 一个 Rank 为什么可以属于多个通信组？
4. Mesh 轴名称本身会不会自动执行 TP 或 DP？
5. 物理 GPU 拓扑和逻辑 DeviceMesh 有什么区别？

---

# 单元 3：DP Replicate 与 DP Shard/FSDP

## 主要学习目标

真正区分：

* 不同 GPU 处理不同数据；
* 不同 GPU 是否保存完整模型；
* 参数和梯度如何同步。

## 核心概念

### DDP / DP Replicate

每个 Rank：

* 保存完整模型；
* 读取不同数据；
* 独立计算梯度；
* 使用 All-Reduce 同步梯度。

### FSDP / DP Shard

模型参数、梯度和优化器状态被切分保存：

```text
计算前：All-Gather 参数
计算中：使用临时完整参数
反向时：Reduce-Scatter 梯度
计算后：继续只保存参数分片
```

PyTorch FSDP2 在计算外保持参数分片，在前向和反向前 All-Gather 参数，并在反向中把梯度 Reduce-Scatter 回各 Rank。([PyTorch Docs][4])

## 实践练习

编写 `03_dp_fsdp_sim.py`，不需要真正实现 FSDP：

1. 创建长度为 16 的“参数向量”；
2. 将它切成 4 个 shard；
3. 模拟 All-Gather，还原完整参数；
4. 创建 4 份本地梯度；
5. 模拟 Reduce-Scatter；
6. 打印每个 Rank 最后保留的参数和梯度范围。

同时完成下面的 Batch 计算：

```text
dp_replicate = 2
dp_shard = 4
local_batch = 3
```

得到：

$$
data\_parallel\_size=2\times4=8
$$

$$
global\_batch=8\times3=24
$$

暂不考虑 Gradient Accumulation。

## 推荐资源

PyTorch 官方 **Getting Started with Fully Sharded Data Parallel (FSDP2)**。([PyTorch Docs][4])

## 完成后的预期成果

你可以解释：

> DP Shard 仍然是数据并行，但它还改变了参数、梯度和优化器状态的存储方式。

## 复习问题

1. DDP 中每个 Rank 是否保存完整参数？
2. FSDP 为什么能减少参数显存？
3. FSDP 前向计算前为什么需要 All-Gather？
4. Reduce-Scatter 与 All-Reduce 的结果布局有什么不同？
5. 为什么 `dp_shard` 不能简单理解成“只切模型，不切数据”？

---

# 单元 4：TP 与 Sequence Parallel

## 主要学习目标

通过一个真实矩阵乘法理解 TP，而不是只背“切 Hidden Size”。

## 核心概念

考虑两层 MLP：

$$
H=XW_1
$$

$$
Y=HW_2
$$

常见 TP 方案：

1. 沿 \(W_1\) 的输出维切分；
2. 每个 TP Rank 得到一部分 \(H\)；
3. 沿 \(W_2\) 的输入维进行对应切分；
4. 每个 Rank 计算部分 \(Y\)；
5. 将部分结果相加或重新分布。

Sequence Parallel 通常与 TP 配合，用于让 LayerNorm、RMSNorm 等模块的激活在 Sequence 方向保持分片，从而减少重复激活存储；它不是 CP 的同义词。([PyTorch Docs][5])

## 实践练习

编写 `04_tp_mlp_sim.py`：

1. 随机生成：

```python
X:  [batch, hidden]
W1: [hidden, 4 * hidden]
W2: [4 * hidden, hidden]
```

2. 先计算完整结果：

```python
Y_dense = X @ W1 @ W2
```

3. 模拟 `tp=2`：

   * 把 `W1` 按列切成两份；
   * 得到两个 `H_shard`；
   * 把 `W2` 按行切成两份；
   * 分别计算两个局部输出；
   * 将局部输出相加。

4. 验证：

```python
torch.testing.assert_close(Y_tp, Y_dense)
```

## 推荐资源

PyTorch 官方 **Large Scale Transformer Model Training with Tensor Parallel**。([PyTorch Docs][5])

## 完成后的预期成果

你能从张量形状出发解释：

* 哪个权重维度被切分；
* 每张 GPU 计算什么；
* 为什么需要集合通信；
* TP 为什么没有增加独立训练样本。

## 复习问题

1. TP 切分的是数据样本还是模型内部计算？
2. 为什么同一个 TP 组必须处理同一批输入？
3. Column Parallel 和 Row Parallel 通常如何配合？
4. Sequence Parallel 主要用于哪些类型的模块？
5. 为什么 TP/SP 和 CP 不能简单视为同一种并行？

---

# 单元 5：CP、PP 与多维并行组合

## 主要学习目标

补齐 Dense 模型中的另外两个方向：

* CP：切长序列；
* PP：切模型层。

## 核心概念

### Context Parallel

将同一条长序列分到多个 Rank：

```text
原始 Sequence：8192 Token
CP = 4
每个 Rank 大致处理：2048 Token
```

注意力仍然需要跨 CP Rank 获得其他序列片段的信息，因此需要额外通信。PyTorch CP 的目标是通过跨设备切分长输入序列，降低激活显存峰值。([PyTorch Docs][6])

当前 `ParallelDims.seq_len_divisor` 返回：

$$
tp\times(2cp)
$$

因为当前实现假设 Sequence Parallel 要求 Sequence Length 可被 TP 切分，而默认带负载均衡的 CP 要求额外的 \(2\times CP\) 可整除性。([GitHub][1])

### Pipeline Parallel

把模型层分成多个阶段：

```text
32 层
PP = 4
每个阶段大致负责 8 层
```

再把 Batch 切成多个 Microbatch，让不同阶段同时工作。PyTorch Pipeline API 负责模型阶段、Microbatch 和执行 Schedule。([PyTorch Docs][7])

## 实践练习

编写 `05_cp_pp_planner.py`：

输入：

```text
seq_len = 8192
num_layers = 32
tp = 2
cp = 4
pp = 4
num_microbatches = 8
```

输出：

1. 每个 CP Rank 的大致 Token 数；
2. 当前 `seq_len_divisor`；
3. Sequence Length 是否满足整除；
4. 每个 PP Stage 的层范围；
5. 一个简化 Pipeline 时间表：

```text
time 0: stage0 -> microbatch0
time 1: stage0 -> microbatch1, stage1 -> microbatch0
...
```

## 推荐资源

主资源：PyTorch **Introduction to Context Parallel**。([PyTorch Docs][6])
补充阅读：PyTorch **Introduction to Distributed Pipeline Parallelism** 中关于 Stage 和 Microbatch 的部分。([PyTorch Docs][7])

## 完成后的预期成果

你能够从四个方向描述 Dense 并行：

```text
DP：不同数据
CP：同一样本的不同序列部分
TP：同一层的不同张量部分
PP：模型的不同层
```

## 复习问题

1. CP 是否增加了独立训练样本数？
2. CP 为什么仍需要跨 Rank 通信？
3. PP Stage 和 Microbatch 分别是什么？
4. 为什么 `num_layers % pp != 0` 需要进一步规划？
5. 配置乘积等于 `world_size` 后，为什么还要检查 Sequence Length？

---

# 单元 6：逐行读懂 `_validate()`

## 主要学习目标

第一次真正进入 `ParallelDims` 源码，理解：

```text
配置进入类后，首先会进行哪些检查？
```

## 核心概念

当前调用主线：

```text
from_config()
    ↓
__post_init__()
    ↓
_validate()
```

当前关键检查包括：

### Dense 资源守恒

$$
dp\_replicate\times dp\_shard\times cp\times tp\times pp
=
world\_size
$$

### 自动推导 `dp_shard`

当：

```text
dp_shard = -1
```

代码计算：

$$
dp\_shard=
\frac{world\_size}
{dp\_replicate\times cp\times tp\times pp}
$$

### EP 合法性

$$
sparse\_region=dp\_shard\times cp\times tp
$$

要求：

$$
sparse\_region\bmod ep=0
$$

这些都是当前 `_validate()` 明确执行的检查。([GitHub][1])

## 实践练习

编写简化版 `06_parallel_dims_lite.py`：

```python
@dataclass
class ParallelDimsLite:
    dp_replicate: int
    dp_shard: int
    cp: int
    tp: int
    pp: int
    ep: int
    world_size: int

    def validate(self) -> None:
        ...
```

至少编写五个测试：

```text
1. 固定 32-GPU 配置合法
2. Dense 乘积不等于 world_size
3. dp_shard=-1 自动推导
4. sparse_region=8、ep=3 不合法
5. sparse_region=8、ep=8 合法
```

## 推荐资源

当前 TorchTitan `parallel_dims.py`，重点阅读 `from_config()`、`__post_init__()` 和 `_validate()`。([GitHub][1])

## 完成后的预期成果

你能够在不启动分布式任务的情况下，判断大部分基础配置是否合法。

## 复习问题

1. 为什么 `ep` 不乘进 Dense 的 `world_size` 等式？
2. `dp_shard=-1` 表示什么？
3. `sparse_region` 为什么包含 `tp`？
4. `sparse_region=12` 时，哪些 `ep` 值合法？
5. `_validate()` 通过后，是否意味着模型一定能够正常运行？

---

# 单元 7：读懂 `build_mesh()` 的多视角设计

## 主要学习目标

掌握 `ParallelDims` 最核心的思想：

> 同一批 Rank，被重新解释成多个服务于不同任务的 Mesh。

## 核心概念

当前 `build_mesh()` 首先计算：

$$
batch=dp\_replicate\times dp\_shard
$$

$$
fsdp=dp\_shard\times cp
$$

$$
efsdp=\frac{fsdp\times tp}{ep}
$$

随后创建多个视角。([GitHub][1])

### Dataloading Mesh

```text
(pp, batch, cp, tp)
```

负责回答：

* 哪些 Rank 读取不同数据？
* 哪些 Rank 使用同一份数据？
* 哪些 Rank 接收同一序列的不同 CP 片段？

### Loss Mesh

将：

```text
(batch, cp)
```

Flatten 为：

```text
loss
```

因为 Loss 需要覆盖分散在 Batch 与 CP 方向上的数据贡献。

### `spmd_types` Dense 视角

参数存储视角：

```text
(pp, dp_replicate, dp_shard, cp, tp)
```

前向和反向类型检查视角：

```text
(pp, dp, cp, tp)
```

其中：

$$
dp=dp\_replicate\times dp\_shard
$$

### `partial_dtensor` Dense 视角

```text
(pp, dp_replicate, fsdp, tp)
```

其中：

$$
fsdp=dp\_shard\times cp
$$

### Sparse Mesh

```text
(pp, dp_replicate, efsdp, ep)
```

当前源码明确创建了上述多个 Mesh，并缓存单轴和多轴子 Mesh。([GitHub][1])

## 实践练习

编写 `07_mesh_inspector.py`。

对于固定配置，输出：

```text
batch = 4
loss = 8
fsdp = 4
efsdp = 2
```

并分别输出：

```text
dataloading:
(pp=2, batch=4, cp=2, tp=2)

spmd_types storage:
(pp=2, dp_replicate=2, dp_shard=2, cp=2, tp=2)

spmd_types fwd/bwd:
(pp=2, dp=4, cp=2, tp=2)

partial_dtensor dense:
(pp=2, dp_replicate=2, fsdp=4, tp=2)

sparse:
(pp=2, dp_replicate=2, efsdp=2, ep=4)
```

程序还应验证每个形状的乘积等于 32。

进阶：打印指定 Rank 在每个 Mesh 中的坐标和所在轴组。

## 推荐资源

当前 `parallel_dims.py` 的 `build_mesh()`、`get_optional_mesh()` 和 `_validate_meshes()`。([GitHub][1])

## 完成后的预期成果

你可以解释：

> `dataloading_mesh`、`dense_mesh` 和 `sparse_mesh` 不是三套 GPU，而是同一批 GPU 的三种分组方案。

## 复习问题

1. 为什么 `batch` 不包含 CP？
2. 为什么 `loss` 包含 CP？
3. Flatten 与 Unflatten 分别改变了什么？
4. `spmd_types` 为什么需要存储和前向/反向两个 Dense 视角？
5. 为什么尺寸为 1 的 Mesh 轴有时仍会被内部保留？

---

# 单元 8：MoE、EP、EFSDP 与 Token 路由

## 主要学习目标

理解 MoE 的“非正交重组”到底是什么意思。

## 核心概念

Dense 层使用的区域可写为：

$$
dp\_shard\times cp\times tp
$$

专家层将同一片 GPU 区域重新组织成：

$$
efsdp\times ep
$$

满足：

$$
dp\_shard\times cp\times tp
=
efsdp\times ep
$$

因此：

$$
efsdp=
\frac{dp\_shard\times cp\times tp}{ep}
$$

当前 `ParallelDims` 的 Sparse Mesh 是：

```text
(pp, dp_replicate, efsdp, ep)
```

它没有独立 `etp` 轴。TorchTitan 的模型接入说明另外提到 `apply_moe_ep_tp`，说明专家 TP 可以在模型并行化代码中进一步处理，但不是当前 `ParallelDims` 的独立字段。([GitHub][1])

### ParallelDims 的职责

```text
定义 EP 组
定义 EFSDP 组
提供 Sparse Mesh
```

### MoE 模型实现的职责

```text
Router 计算专家选择
Token 打包
All-to-All Dispatch
专家计算
All-to-All Combine
恢复原 Token 顺序
```

从源码边界可以判断，创建 Sparse Mesh 并不等于已经完成 Token 路由；真正的专家并行逻辑由模型和通信实现调用这些组。官方 TorchTitan/PyTorch 的 EP 实践也将标准 All-to-All 视为专家通信的关键部分。([GitHub][8])

## 实践练习

编写 `08_moe_router_sim.py`：

输入：

```text
num_tokens = 16
num_experts = 8
ep = 4
```

自行定义一个透明映射：

```text
expert 0,1 -> EP rank 0
expert 2,3 -> EP rank 1
expert 4,5 -> EP rank 2
expert 6,7 -> EP rank 3
```

程序完成：

1. 为每个 Token 指定专家；
2. 找到目标 EP Rank；
3. 按目标 Rank 分桶；
4. 模拟专家计算；
5. 按原始 Token 编号恢复顺序；
6. 输出每个 EP Rank 收到的 Token 数；
7. 计算最忙与最闲 Rank 的差距。

## 推荐资源

TorchTitan 官方模型接入说明中关于 Dense/Sparse SPMD 轴和 `apply_moe_ep_tp` 的部分。([GitHub][8])

## 完成后的预期成果

你能够区分三个问题：

```text
专家参数放在哪里？
Token 被发送到哪里？
哪个 Mesh 负责定义通信范围？
```

## 复习问题

1. EP 为什么不意味着增加新的 GPU？
2. `ep` 是否必须等于专家总数？
3. `efsdp` 表示什么？
4. 为什么专家计算前后通常各需要一次 Token 交换？
5. 为什么 `ParallelDims.build_mesh()` 本身没有完成 Router 和 All-to-All？

---

# 单元 9：数值正确性与梯度归一化审计

## 主要学习目标

建立一个重要习惯：

> 不根据 Mesh 大小猜梯度缩放，而是追踪完整的 Loss 和通信路径。

## 核心概念

梯度最终缩放取决于：

1. Loss 是 `sum` 还是 `mean`；
2. 每个 Rank 包含多少有效 Token；
3. 是否进行 Gradient Accumulation；
4. 集合通信执行 Sum 还是 Average；
5. FSDP/DTensor 对结果进行了什么缩放；
6. PP 是否对 Microbatch Loss 做了额外平均；
7. MoE Router 是否让不同 Rank 获得不同数量的 Token。

因此，看到：

```text
efsdp size = 2
```

不能仅凭这一点断言梯度一定“除以 2”。

当前 `ParallelDims` 文件中没有 `fsdp_gradient_divide_factor`。学习文中那段历史逻辑时，必须先固定对应 commit，再追踪当时的 Loss、FSDP 和专家通信实现。([GitHub][1])

TorchTitan 官方调试文档建议使用相同 Seed Checkpoint 比较不同并行配置的 Loss；同时明确指出，即使初始权重相同，不同张量形状、精度或随机路径仍可能导致运行不完全相同。([GitHub][9])

## 实践练习

编写 `09_gradient_accounting.py`。

### 实验 A：等量数据

两个 Rank，各有两个样本：

```text
rank 0 loss: [1, 3]
rank 1 loss: [5, 7]
```

验证：

$$
global\ mean=\frac{1+3+5+7}{4}=4
$$

比较：

* 先求本地 mean，再对两个 mean 求平均；
* 直接计算全局 mean。

两者此时相同。

### 实验 B：不等量 Token

```text
rank 0: [1]
rank 1: [3, 5, 7]
```

全局 mean：

$$
\frac{1+3+5+7}{4}=4
$$

但本地 mean 再平均：

$$
\frac{1+(3+5+7)/3}{2}=3
$$

这说明：

> 专家路由不均衡时，“平均各 Rank 的本地平均值”未必等于“所有 Token 的全局平均值”。

最后制作一份梯度审计表：

```text
全局训练目标是什么？
本地 Loss 如何归约？
每个 Rank 有多少有效 Token？
发生了哪些 Collective？
Collective 是 Sum 还是 Average？
最终应该除以什么？
参考基线是什么？
```

## 推荐资源

TorchTitan 官方 **Debugging** 文档，重点阅读 Seed Checkpoint、Loss Curve Comparison 和 Fake Backend。Fake Backend 适合验证配置和不依赖真实通信的数据流，但不能用于真实性能测试，也不应验证依赖真实数据通信的逻辑。([GitHub][9])

## 完成后的预期成果

你能够设计一次严谨的并行数值对比：

```text
固定权重
固定输入
比较前向输出
比较 Loss
比较每层梯度范数
比较一步参数更新
```

## 复习问题

1. 为什么不能根据 ProcessGroup 大小直接推断最终梯度除数？
2. 本地 Mean 的平均为什么可能不等于全局 Mean？
3. 比较两个并行配置时为什么要固定初始权重？
4. 哪些数值应该在一步训练后进行对比？
5. Fake Backend 能否用于判断真实 All-to-All 性能？

---

# 单元 10：物理拓扑、性能判断与配置审计器

## 主要学习目标

把逻辑 Mesh 连接到现实硬件：

```text
逻辑上合法
≠
数值一定正确
≠
性能一定优秀
```

## 核心概念

### 通信频率与通信量

需要区分：

* 高频小通信；
* 低频大通信；
* All-Reduce；
* All-Gather；
* Reduce-Scatter；
* All-to-All；
* PP Point-to-Point。

### 节点内与节点间网络

常见经验是：

* TP 尽量利用节点内高速互联；
* FSDP/DP 可扩展到节点间；
* EP 跨节点时 All-to-All 可能成为瓶颈；
* PP 的优劣还取决于 Microbatch 数、阶段平衡和激活大小。

但这些是配置候选的设计原则，不是仅凭 Mesh 元组就能证明的结论。PyTorch TP 教程展示了常见的“节点内 TP、节点间 FSDP”组合；TorchTitan 官方基准也记录了具体硬件拓扑、并行配置、吞吐和显存，说明实际选择必须依靠测量。([PyTorch Docs][5])

### 配置错误与性能问题要分开

硬错误：

```text
Dense 乘积错误
EP 不能整除 sparse_region
Sequence Length 整除失败
```

模型相关约束：

```text
注意力头数与 TP 的兼容性
专家数量与 EP 的映射
模型层数与 PP Stage 切分
```

性能问题：

```text
TP 跨低速网络
EP All-to-All 过重
PP 空泡过大
专家严重负载不均衡
显存峰值不平衡
```

## 实践练习

编写 `10_parallel_config_advisor.py`。

输入：

```text
world_size
gpus_per_node
dp_replicate
dp_shard
cp
tp
pp
ep
seq_len
num_layers
num_heads
num_experts
spmd_backend
```

输出四类结果。

### 1. 硬校验

```text
Dense 乘积是否合法
dp_shard=-1 能否自动推导
ep 是否整除 sparse_region
seq_len 是否满足 seq_len_divisor
```

### 2. 派生信息

```text
batch
loss
fsdp
efsdp
各 Mesh 的形状
```

### 3. 模型相关提醒

例如：

```text
[CHECK] 请确认 num_heads 与 TP/CP 算子兼容
[WARN] num_layers 不能平均分到所有 PP stage
[CHECK] 请确认专家到 EP rank 的映射
```

这些应标成 `CHECK/WARN`，不要在不了解具体模型实现时全部写成硬错误。

### 4. 物理拓扑提示

例如：

```text
TP group 是否可能跨节点
EP group 是否可能跨节点
每个 PP stage 使用多少 GPU
可能出现的主要集合通信
```

## 推荐资源

TorchTitan 官方 H100 基准报告。重点不是记住数字，而是观察报告如何同时记录：

```text
硬件
Commit
并行配置
Sequence Length
显存
吞吐
```

([GitHub][10])

## 完成后的预期成果

你拥有一个可以在真实任务启动前使用的配置审计器 MVP。

## 复习问题

1. 为什么两个都满足 GPU 乘积的配置，性能可能相差很大？
2. 为什么 TP 通常优先考虑节点内高速互联？
3. EP 的主要网络风险是什么？
4. PP 性能为什么依赖 Microbatch 数和阶段平衡？
5. 哪些问题可以静态检查，哪些问题必须通过 Profiling 才能判断？

---

# 四、最终项目：`ParallelDims Auditor & Visualizer`

## 项目目标

实现一个独立的小工具：

> 输入集群、模型和并行配置，输出合法性检查、Mesh 推导、Rank 分组、风险提示和可读审计报告。

核心部分完全可以用纯 Python 完成，不依赖 32 张真实 GPU。

---

## 1. 命令行接口

示例：

```bash
python parallel_config_advisor.py \
  --world-size 32 \
  --gpus-per-node 8 \
  --dp-replicate 2 \
  --dp-shard 2 \
  --cp 2 \
  --tp 2 \
  --pp 2 \
  --ep 4 \
  --seq-len 8192 \
  --num-layers 32 \
  --num-heads 32 \
  --num-experts 8 \
  --spmd-backend spmd_types
```

---

## 2. 预期输出

```text
[VALID] Dense world-size equation:
2 × 2 × 2 × 2 × 2 = 32

[VALID] Expert region:
dp_shard × cp × tp = 2 × 2 × 2 = 8
8 % ep(4) = 0

Derived dimensions:
batch = 4
loss  = 8
fsdp  = 4
efsdp = 2

Mesh views:
dataloading          = (pp=2, batch=4, cp=2, tp=2)
dense_storage        = (pp=2, dpr=2, dps=2, cp=2, tp=2)
dense_fwd_bwd        = (pp=2, dp=4, cp=2, tp=2)
partial_dtensor      = (pp=2, dpr=2, fsdp=4, tp=2)
sparse               = (pp=2, dpr=2, efsdp=2, ep=4)

Sequence check:
seq_len_divisor = tp × 2 × cp = 8
8192 % 8 = 0

Topology notes:
TP degree 2 can fit inside an 8-GPU node.
EP group placement must be checked against rank-to-node mapping.
```

---

## 3. 必须实现的功能

### 配置检查

* Dense 乘积；
* `dp_shard=-1`；
* EP 整除；
* Sequence Length 整除；
* 所有维度必须为正数。

### Mesh 推导

* Dataloading；
* Loss；
* Dense Storage；
* Dense Forward/Backward；
* Partial DTensor Dense；
* Sparse。

### Rank 可视化

对于任意 Rank，输出：

```text
global rank
node id
local rank
各 Mesh 中的坐标
每个轴的通信组成员
```

### MoE 路由模拟

复用 `moe_router_sim.py`：

* Token 到 Expert；
* Expert 到 EP Rank；
* 每个 Rank 的 Token 数；
* 负载最大值、最小值和平均值。

### 风险分级

```text
ERROR：一定不能运行
WARN：可能运行，但配置可疑
CHECK：依赖具体模型或实现
INFO：普通说明
```

---

## 4. 测试用例

至少覆盖：

### Case A：合法配置

```text
world=32
dpr=2, dps=2, cp=2, tp=2, pp=2, ep=4
```

### Case B：Dense 乘积错误

```text
world=32
dpr=2, dps=2, cp=2, tp=2, pp=1
```

乘积只有 16。

### Case C：EP 非法

```text
sparse_region=8
ep=3
```

### Case D：自动 FSDP

```text
dp_shard=-1
```

### Case E：Sequence Length 非法

选择一个不能被：

$$
tp\times2cp
$$

整除的长度。

### Case F：Singleton 轴

```text
cp=1
tp=1
pp=1
ep=1
```

检查工具是否仍能正确输出 Mesh。

---

## 5. 最终交付物

```text
parallel_dims_auditor/
├── parallel_config_advisor.py
├── mesh_coordinates.py
├── moe_router_sim.py
├── tests/
│   ├── test_validation.py
│   ├── test_mesh_shapes.py
│   └── test_router.py
├── examples/
│   ├── valid_32gpu.json
│   ├── invalid_ep.json
│   └── comparison_report.md
├── LEARNING_COMMIT.txt
└── README.md
```

README 需要回答：

1. `ParallelDims` 解决什么问题？
2. 为什么同一批 Rank 需要多个 Mesh？
3. Dense Mesh 和 Sparse Mesh 如何守恒？
4. 哪些检查属于 `ParallelDims`？
5. 哪些约束必须继续追踪模型和通信代码？
6. 为什么配置合法不代表性能优秀？
7. 为什么不能脱离具体版本讨论梯度归一化？

---

# 五、判断自己是否真正学会

满足下面四项，就说明 20 小时取得了实际成果：

### 源码理解

不看文章，也能从源码解释：

```text
_validate()
build_mesh()
get_optional_mesh()
seq_len_divisor
```

### 配置推导

能在 5 分钟内手算固定 32-GPU 配置的：

```text
batch
loss
fsdp
efsdp
Dense Mesh
Sparse Mesh
```

### 边界意识

能明确说出：

```text
ParallelDims 负责创建和提供通信组
下游 FSDP/TP/CP/PP/MoE 实现负责真正的张量切分与通信
```

### 实际诊断

面对一个失败或很慢的配置，能够依次检查：

```text
1. 维度是否合法
2. 模型形状是否兼容
3. Loss 和梯度是否等价
4. Rank 是否映射到合适的物理网络
5. 哪个 Collective 是主要瓶颈
```

这比单纯记住“TP 最内层、PP 最外层”更接近真实的分布式训练工程能力。

[1]: https://github.com/pytorch/torchtitan/blob/main/torchtitan/distributed/parallel_dims.py "torchtitan/torchtitan/distributed/parallel_dims.py at main · pytorch/torchtitan · GitHub"
[2]: https://docs.pytorch.org/tutorials/recipes/distributed_device_mesh.html "Getting Started with DeviceMesh — PyTorch Tutorials 2.13.0+cu130 documentation"
[3]: https://docs.pytorch.org/tutorials/intermediate/dist_tuto.html?utm_source=chatgpt.com "Writing Distributed Applications with PyTorch"
[4]: https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html "Getting Started with Fully Sharded Data Parallel (FSDP2) — PyTorch Tutorials 2.13.0+cu130 documentation"
[5]: https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html "Large Scale Transformer model training with Tensor Parallel (TP) — PyTorch Tutorials 2.13.0+cu130 documentation"
[6]: https://docs.pytorch.org/tutorials/unstable/context_parallel.html "Introduction to Context Parallel — PyTorch Tutorials 2.13.0+cu130 documentation"
[7]: https://docs.pytorch.org/tutorials/intermediate/pipelining_tutorial.html "Introduction to Distributed Pipeline Parallelism — PyTorch Tutorials 2.13.0+cu130 documentation"
[8]: https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/README.md "torchtitan/torchtitan/models/README.md at main · pytorch/torchtitan · GitHub"
[9]: https://github.com/pytorch/torchtitan/blob/main/docs/debugging.md?utm_source=chatgpt.com "torchtitan/docs/debugging.md at main"
[10]: https://github.com/pytorch/torchtitan/blob/main/benchmarks/llama3_h100_202412_torchtitan.md?utm_source=chatgpt.com "torchtitan/benchmarks/llama3_h100_202412_torchtitan.md ..."
