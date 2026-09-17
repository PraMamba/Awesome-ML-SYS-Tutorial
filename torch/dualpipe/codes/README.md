# torch/dualpipe/codes：配套教学实验

本目录是 `torch/dualpipe/deep-dive.md` 正文引用的两个可运行实验。它们各自只验证一件事，**都不是 DualPipe 实现，也不测量任何 GPU 性能**。

| 脚本 | 验证的问题 | 依赖 |
| --- | --- | --- |
| `01_split_backward_minimal.py` | 把一次反向拆成「先返回输入梯度、稍后再补参数梯度」，数学上是否与常规 autograd 等价 | 仅 PyTorch，CPU |
| `02_schedule_sim.py` | DualPipe / DualPipeV 的八阶段调度按官方**提交语义**需要多长关键路径、气泡占比多少 | 仅 Python 标准库，无 PyTorch、无 GPU |

## 环境与运行

```bash
# 本仓库验证时使用
python -V            # Python 3.12
python -c "import torch; print(torch.__version__)"   # 2.10.0+cu129

python torch/dualpipe/codes/01_split_backward_minimal.py
python torch/dualpipe/codes/02_schedule_sim.py                 # 默认 p=4, m=10
python torch/dualpipe/codes/02_schedule_sim.py --p 8 --m 20
python torch/dualpipe/codes/02_schedule_sim.py --sweep         # 多组 (p, m) 一致性扫描
```

两个脚本都只用手写张量与纯 Python 逻辑，不需要分布式初始化、不需要 CUDA，也不写任何文件。

## `01_split_backward_minimal.py`

前向照常执行；反向只返回 `dX`，把 `dTheta = GᵀX` 记入一个 `pending` 队列，最后统一补算。脚本同时检查：

1. `loss.backward()` 返回后 `theta.grad` 仍然为 `None`——即「反向传播结束」不等于「参数梯度已写完」；
2. 拆分版与常规 autograd 的 `dX`、`dTheta` 在 `float64` 下逐元素一致，并打印最大绝对误差。

本机实测（PyTorch 2.10.0+cu129，CPU，float64，`x[8,8]`、`theta[16,8]`、2 个 microbatch）：

```
backward 返回后 dTheta is None : True
max |dX_split - dX_ref|        : 0.000e+00
max |dTheta_split - dTheta_ref|: 1.110e-16
PASS
```

**结论边界**：这只证明「I 与 W 在数学上可以分开产出，且推迟 W 不改变梯度数值」。它不涉及分布式梯度归约、混合精度、`checkpoint`、高阶梯度，也不证明任何调度一定更快。手工写 `.grad` 不能当成 DDP/FSDP 的通用接入方式。

## `02_schedule_sim.py`

脚本逐行镜像固定 revision 的官方 `step()` 循环结构（`deepseek-ai/DualPipe@030ce4325f4ebeb437da4ebc6d00a70469dd58ae`：`dualpipe/dualpipe.py` L294–440、`dualpipe/dualpipev.py` L288–411），生成每个 rank 上的 `(F, I, W)` 任务序列，然后：

- 为每个 rank 加入同设备程序顺序约束；
- 为跨 stage 的前向加入「上游第 k 个 F → 下游第 k 个 F」依赖；
- 为跨 stage 的反向加入「上游第 k 个 I → 下游第 k 个 I」依赖——**只依赖 I，不依赖 W**，这正是 Zero Bubble 拆分掉的那条边；
- 用 Kahn 拓扑排序求关键路径，得到调度总时长。

时间模型与 Zero Bubble 论文 §2 的手工调度分析一致：一个逻辑 stage chunk 的 `F`、`I`、`W` 各记 1 个时间单位；`enable_zb=True` 的反向只做 `I`，其 `W` 进入 `WeightGradStore` 队列，在之后 `_weight_chunk()` 的位置按 FIFO 执行。

### 运行结果

`p=4, m=10`（DualPipeV 即 8 个逻辑 stage）：

| 方法 | busy | 关键路径 | 气泡 | 气泡率 |
| --- | ---: | ---: | ---: | ---: |
| DualPipe | 30 | 32 | 2 | 6.25% |
| DualPipeV | 60 | 66 | 6 | 9.09% |

`p=8, m=20`：

| 方法 | busy | 关键路径 | 气泡 | 气泡率 |
| --- | ---: | ---: | ---: | ---: |
| DualPipe | 60 | **66** | **6** | 9.09% |
| DualPipeV | 120 | 134 | 14 | 10.45% |

**DualPipe `p=8, m=20` 的这组数与官方 `dualpipe.png` 逐格取色的结果完全一致**：那张图共 8 行、每行 66 个单位，其中前向 20 个单位、其余 40 个单位是反向的三类、**空闲 6 个单位**；`dualpipev.png` 同样是 4 行、每行 66 个单位、空闲 6 个单位。也就是说，这个脚本算出的关键路径与官方调度图的绘制口径是同一套。

`--sweep` 覆盖 `p ∈ {2,4,6,8}`、`m ∈ {8,10,16,20,32}`（跳过 `m < 2p` 的组合后实为 32 组），两个方法各自满足（换算到「标准 stage chunk」单位 `t`，即 DualPipe 的一个 chunk、DualPipeV 的两个 chunk）：

| 量 | 闭式 | 说明 |
| --- | --- | --- |
| 每设备有效工作 | `3mt` | 与 `p` 无关；三种调度在同设备数、同模型下工作量相同 |
| DualPipe 气泡 | `(p-2)t` | 与 `m` 无关 |
| DualPipeV 气泡 | `(p-1)t` | 与 `m` 无关 |
| DualPipe 气泡率 | `(p-2)/(3m+p-2)` | |
| DualPipeV 气泡率 | `(p-1)/(3m+p-1)` | |

这些闭式同时给出一个重要交叉验证：把官方 README 的 `(PP/2-1)(F&B+B-3W)` 在 `F&B = F + B_full = 3t`、`B = B_full = 2t`、`W = t` 下代入，得到 `(p/2-1)·2t`，与上表的 `(p-2)t` 相同；**官方调度图也正是按 `F&B = F + B_full` 绘制的**，所以这里用的是与官方材料一致的那一端。另一端 `F&B = max(F, B_full) = 2t` 对应「两个 chunk 的计算真正并排跑」，会让气泡减半，但**没有来源支持**：DeepSeek-V3 Figure 4 画的是计算行串行、被藏起来的是通信，而本脚本的时间模型本来就忽略通信。

### 建模假设（第一条最重要）

脚本**只**保证两件事：循环结构与任务次序逐行对应官方 `step()`；**发送的释放时刻按官方提交语义建模**。

第二件事有一个容易踩的坑，值得单独写出来：官方代码里的 `_send_forward` / `_send_backward` 只是把 `P2POp` 追加进 `self.comm_ops`，真正发出发生在**下一次 `_commit_and_wait_comm()`**（每个 `_forward_chunk` / `_backward_chunk` / `_forward_backward_chunk` / `_weight_chunk` 开头都会先调它，`step()` 结尾还有一次）。因此脚本给每个提交点插一个时长为 0 的 `REL` 任务，跨 rank 的数据依赖一律挂在 `REL` 上。**如果改成让下游直接依赖 `F` 任务的结束时刻，配对块里的前向输出会被乐观地提前释放，整体气泡会少一半**（`p=4, m=10` 会从 2 变成 1）——早期版本正是犯了这个错，得到的数字与官方图对不上。唯一不走提交点的是 DualPipeV 的 V 底部本地交接：官方在计算结束时直接 `append` 进本地队列，因此那条边挂的是计算任务本身。

其余建模取舍：

- 非 zb 的反向统一记成 `I → W` 两个任务，而官方示例的重叠钩子里 `grad_weight_fn()` 排在 `grad_input = grad_output @ weight` 之前。这**不改变任何一组配置的气泡**（依赖边的结构不变，只是 `I` 的完成时刻后移 1 个单位，没有落在关键路径上），但它确实意味着脚本的 rank 内顺序不是示例的逐字复刻。
- 通信耗时记 0；配对块内部当作不可分的整体（示例钩子顺序执行时 `C = 3`）；中间 rank 第 4 步 `i=0` 的「故意不重叠」特判按源码拆成两段独立计算。

### 额外核对

- 每 rank 的 `F`、`I` 计数恒等于 `2 × (m/2)`（DualPipe）或 `2 × m`（DualPipeV），且全设备 `I` 总数等于 `W` 总数——与源码里 `assert WeightGradStore.funcs_queue.empty()` 的会计一致；
- 第一组稳态工作的 chunk 序号：`p=4, r=1` 时 DualPipe 给出 `(F0_2, B1_0, F1_2, B0_0)`（`x=2`）、DualPipeV 给出 `(F0_6, B1_2, F1_4, B0_0)`（`x=6`）；`p=8, r=1` 时 DualPipe 给出 `(F0_6, B1_2, F1_4, B0_0)`（`x=6`）。脚本只在 step 4 的 `i=0` 打印这一组，`x=8` 那组由主循环每步四个计数器同步 +1 推出，**脚本不会打印它**。

## 验证范围与结论边界

**已验证**：反向拆分的数值等价性；八阶段调度的任务计数与会计一致性；在「F=I=W 等时 + I 独立于 W」这一依赖模型下的关键路径、气泡时间与气泡率。

**未验证、不可从本目录推断**：

- 任何 GPU 吞吐、显存占用、通信带宽或端到端 step time。本目录没有跑过任何 GPU 实验。
- 真实的重叠收益。本模拟器只算依赖与时间，**不建模计算与通信的重叠**；它给出的气泡对应「配对块里的计算与反向按顺序做完、通信被忽略」这一时间模型，与官方调度图的绘制口径一致，但它不是实测值，也不能说明真实 GPU 上重叠掉多少。
- `F = I = W` 等时假设。Zero Bubble 论文自己的测量结论是 `T_W < T_F < T_B` 且 `T_B + T_W = 2T_F`（论文 Table 1）；等时假设是论文 §2 手工调度分析所用的理想化前提。
- 通信时间。Zero Bubble 论文 §3 明确指出手工调度忽略了 `T_comm`，本模拟器同样忽略。
- 张量级的端到端数值正确性（即「按 V 形调度真实前反向一遍，输出与梯度是否与串行参考一致」）。草稿曾引用一个 `dualpipe_cpu_lab.py` 声称验证到这一点并给出误差数字，但该文件不在仓库内、也不在可获取的路径上，**本次未重建**，因此正文不保留相关结论与数字。
