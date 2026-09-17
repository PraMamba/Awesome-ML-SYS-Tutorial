# DeepEP 文章配套教学实现

这两个脚本配合 [DeepEP 深度学习笔记](../deep-dive.md) 的第 2、3 章使用，目的是把
dispatch / combine 的**语义与梯度契约**显式写出来，方便读者在自己的机器上复算。

> **它们是教学实现，不是 DeepEP 源码，也不调用 DeepEP。**
> 它们不测量 RDMA / NVLink 吞吐，不涉及 DeepEP 的 CUDA 内核、warp specialization、
TMA、布局转换或 NCCL Gin。文中所有性能数字都来自官方 README / `docs/legacy.md`，
与这两个脚本无关。

## 环境

- Python 3.12、PyTorch 2.10.0（本仓库验证环境为 `2.10.0+cu129`，但脚本只用 CPU 张量）
- `minimal_moe.py`：单进程，无需 GPU
- `distributed_ep.py`：4 个进程，Gloo + CPU 张量，无需 GPU、无需 NVLink/RDMA

## 运行

```bash
# 1) 单进程：assignment 展开、按专家排序、加权归约，以及与朴素参考实现的前向/梯度对照
python minimal_moe.py

# 2) 四进程：不等长 all_to_all_single 的 dispatch / combine，以及与全局参考实现的前向/梯度对照
torchrun --standalone --nproc-per-node=4 distributed_ep.py
```

## 已验证范围（2026-09-10 实测）

`minimal_moe.py` 的 4 个用例全部通过（float64，前向与 `dX` / `dProbs` / `dWeights` 三项梯度
与逐 token 参考实现的最大绝对误差 ≤ 2.3e-16）：

- 常规路由，且存在全局无人选中的专家
- 含无效路由（`-1`）
- 单个 token 命中同一专家两次
- 所有路由均无效（输出恒为 0，梯度为 0）

它同时打印文章 §1.1 的例子：8 个 token-expert assignment 里，按
`(token, 目的 rank)` 去重后只需要传 6 份；以及 §9 的权重位置算例
（正确 12、漏乘 30、重乘 7.2）。

`distributed_ep.py` 的 4 个用例全部通过（4 rank，float64，前向与三项梯度对全局参考实现的
最大绝对误差 ≤ 4.5e-16）：

- 常规路由，且有一个专家全局无人选中
- 某个 rank 完全没有 token
- 同一 token 命中同一 rank 的两个专家
- 所有 rank 都没有 token

## 这个教学实现刻意没做的事

1. **没有做 rank 去重。** 它按 assignment 逐份发送，所以第 3 个用例的 `rank0` 会发送 8 行，
   而按 `(token, 目的 rank)` 去重后只需要 4 行。DeepEP 的 normal 路径正是在这一步之后
   继续优化，脚本故意停在更朴素的位置，方便对照。
2. **没有处理计算与通信重叠。** 它是严格的顺序执行：交换计数 → 交换数据 → 计算 → 交换回来。
3. **没有 Expert Parallel 之外的东西。** 没有 TP、没有 DP、没有专家参数的分布式存储，
   每个 rank 只持有自己那几个专家的权重。
4. **没有实现 DeepEP 的接口。** 这里没有 `Buffer`、`ElasticBuffer`、`EPHandle`、layout、
   `num_sms`、FP8 或 TMA，只是用 PyTorch 的 `all_to_all_single` 复刻语义。

## 反向交换为什么必须对调分片

`_AllToAll.forward` 用 `(output_split_sizes, input_split_sizes)`；`_AllToAll.backward` 把两者
对调，因为沿着同一条通信边回来的是每个来源 rank 的梯度。这正是
`dispatch` 的反向是 `combine`、`combine` 的反向是 `dispatch` 的来源：路由索引本身离散、
不参与求导，所以反向只需要把「每个目的 rank 拿到哪些行」这件事按相反方向再做一次。
