# DeepEP 深度解析：图片资产清单

本目录保存文章正文实际引用的图片副本。命名沿用 `transformers/kda_linear_attention/pics/` 的惯例：`<论文/项目简称>-fig<N><子图>-<内容 slug>.<ext>`。所有图片都从 `torch/deepep/references/papers/` 的论文抽取图，或从上游 DeepEP 仓库的锁定 revision 复制而来，**不做重绘、不做裁剪**；每张都经人工逐张核对（文件存在、确为正文 Figure、图号与论文一致），核对记录见第 5 节。

参考文献编号沿用仓库既有写法：DeepSeekMoE 为 arXiv `2401.06066v1`，DeepSeek-V3 Technical Report 为 arXiv `2412.19437v2`。

文件名与 `torch/deepep/learn-plan.md` 的「参考图复用策略」一节逐一对齐（该节的采用清单 12 张与本目录一一对应），以免正文按计划写完后链接落空。这 12 张里包含计划早期只列了 7 张核心图之后补入的 5 张（`deepseek-moe-fig5-…`、`deepseek-moe-fig6-…`、`deepseek-v3-fig7a-…`、`deepseek-v3-fig7b-…`、`deepseek-v3-fig9-…`），它们与核心图一样都已经核过出处与内容、都在正文中有引用位置。另需说明，`pics/.superseded/` 目录里另有 4 张图，是同一批素材的早期命名副本，正文**不要**引用它们。

## 0. 正文引用格式

与 `transformers/kda_linear_attention/kda_linear_attention-deep-dive.md` 保持一致：居中 `<div>` + `<img>` + 紧随其后的「图片来源」引用块。正文中的图片一律使用相对路径 `./pics/<文件名>`，alt 文本需包含「图号 + 图中关键读数」，方便无法加载图片时仍然可读。

```
<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/<文件名>" alt="<图号 + 内容 + 关键读数>" style="width: 60%;">
</div>

> **图片来源**：<论文名>（arXiv <编号>）Figure <N>，§<小节>。本地副本：`references/papers/<slug>/<slug>.md` 第 <L> 行引用图。<口径说明>
```

## 1. 已复制到本目录的图片（12 张）

### 1.1 `deepseek-moe-fig2-fine-grained-shared-expert.jpg`

| 项 | 值 |
| --- | --- |
| 尺寸 / 体积 | 1246 × 617 / 87,277 B |
| 来源 | `references/papers/deepseek-moe/images/f77205913b6fd151e304c5a70d7e631629fa19ca95d08f9fb369d1ef4762eb67.jpg` |
| 出处 | DeepSeekMoE（arXiv `2401.06066v1`）Figure 2（md 第 74 行引用，所属小节为 §2 Preliminaries: Mixture-of-Experts for Transformers，在 §3 DeepSeekMoE Architecture 之前） |
| 逐张核对结果 | 三栏对比图，与论文图注一致：(a) Conventional Top-2 Routing；(b) + Fine-grained Expert Segmentation，激活专家数 K = 4；(c) + Shared Expert Isolation，即 DeepSeekMoE，1 个 shared expert + routed experts、K = 3；右上角图例区分 Routed Expert（蓝）与 Shared Expert（绿） |
| 建议章节 | 模型侧特征 → 细粒度专家切分与共享专家隔离（讲「为什么 DeepEP 面对的不是 top-2 稠密通信模式」） |

可直接粘贴的正文块：

```
<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/deepseek-moe-fig2-fine-grained-shared-expert.jpg" alt="DeepSeekMoE Figure 2：(a) 传统 top-2 routing（N 个专家激活 2 个）；(b) 细粒度专家切分后专家数变为 2N、激活 K=4；(c) 共享专家隔离：1 个 shared expert 始终参与，routed experts 激活 K=3" style="width: 80%;">
</div>

> **图片来源**：DeepSeekMoE（arXiv 2401.06066v1）Figure 2，§2 Preliminaries（在 §3 DeepSeekMoE Architecture 之前）。本地副本：`references/papers/deepseek-moe/deepseek-moe.md` 第 74 行引用图。
```

### 1.2 `deepseek-moe-fig5-activated-experts.jpg`

| 项 | 值 |
| --- | --- |
| 尺寸 / 体积 | 720 × 521 / 41,996 B |
| 来源 | `references/papers/deepseek-moe/images/a40c85d13d4d10480ad94a6298bcc81c4905486a9e1030fe423e6332137bbc28.jpg` |
| 出处 | DeepSeekMoE Figure 5（md 第 249 行引用，所属小节为 §4.5 Analysis on Expert Specialization） |
| 逐张核对结果 | 横轴 Activated Routed Experts（3→7），纵轴 Pile Loss；DeepSeekMoE 曲线从 1.956 降到 1.810 附近，虚线为 GShard（full top-2 activated）的 1.867；图中标注 "same activated expert parameters" 的双向箭头对齐 7 个激活专家处 |
| 建议章节 | 模型侧特征 → 细粒度切分的收益（用来说明「专家变多、每 token 触达专家数变多，通信结构随之改变」） |

```
<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/deepseek-moe-fig5-activated-experts.jpg" alt="DeepSeekMoE Figure 5：Pile loss 随激活 routed expert 数（3 到 7）下降，激活 4 个时已低于 GShard full top-2 的 1.867；图中标注在 7 个激活专家处两者激活参数量相同" style="width: 55%;">
</div>

> **图片来源**：DeepSeekMoE Figure 5。本地副本：`references/papers/deepseek-moe/deepseek-moe.md` 第 249 行引用图。
```

### 1.3 `deepseek-moe-fig6-gshard-vs-deepseekmoe.jpg`

| 项 | 值 |
| --- | --- |
| 尺寸 / 体积 | 1226 × 557 / 64,124 B |
| 来源 | `references/papers/deepseek-moe/images/b7f2ad03a5499b898dcdecb0b32663662464722ba5f984ef481382dedbc4cddc.jpg` |
| 出处 | DeepSeekMoE Figure 6（md 第 252 行引用，所属小节为 §4.5 Analysis on Expert Specialization） |
| 逐张核对结果 | 6 个 benchmark 柱状对比：蓝色为 GShard（0 shared + 2 of 16 routed），橙色为 DeepSeekMoE with half activated experts（1 shared + 3 of 63 routed）；在 TriviaQA / NaturalQuestions 上两者差距最明显 |
| 建议章节 | 模型侧特征 → 共享专家隔离的收益（可选图；若正文已用 Figure 2 + Figure 5 说明结构，这张作为「同激活参数量下的对照」） |

```
<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/deepseek-moe-fig6-gshard-vs-deepseekmoe.jpg" alt="DeepSeekMoE Figure 6：GShard（0 shared + 2/16 routed）与 DeepSeekMoE（1 shared + 3/63 routed，激活专家数减半）在 HellaSwag、PIQA、ARC-easy、ARC-challenge、TriviaQA、NaturalQuestions 上的柱状对比" style="width: 70%;">
</div>

> **图片来源**：DeepSeekMoE Figure 6。本地副本：`references/papers/deepseek-moe/deepseek-moe.md` 第 252 行引用图。
```

### 1.4 `deepseek-v3-fig2-mla-deepseekmoe-architecture.jpg`

| 项 | 值 |
| --- | --- |
| 尺寸 / 体积 | 1251 × 1004 / 162,172 B |
| 来源 | `references/papers/deepseek-v3-report/images/d9af045bad6955d9d437d94a3e81725e70e117b0525d70b9ab0a982019177d77.jpg` |
| 出处 | DeepSeek-V3 Technical Report（arXiv `2412.19437v2`）Figure 2（md 第 127 行引用，所属小节为 §2.1 Basic Architecture） |
| 逐张核对结果 | 左侧为 Transformer Block × L（Attention + RMSNorm + Feed-Forward Network + RMSNorm）；右上展开 DeepSeekMoE：Input Hidden u_t → Router → Top-K_r → N_r 个 Routed Expert（蓝）+ N_s 个 Shared Expert（绿）→ 加权求和输出 h'_t；右下展开 MLA：Latent c^Q_t / c^KV_t 低秩投影、RoPE 分支拼接、Cached During Inference 标记 |
| 建议章节 | 模型侧特征 → DeepSeek-V3 全貌（进入计算/通信特征之前的模型定位图） |

```
<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/deepseek-v3-fig2-mla-deepseekmoe-architecture.jpg" alt="DeepSeek-V3 Figure 2：左侧为 L 层 Transformer Block（Attention / RMSNorm / FFN 各带残差）；右上为 DeepSeekMoE 细节：Router 从 Input Hidden 选出 Top-K_r，路由到 N_r 个 Routed Expert 与 N_s 个 Shared Expert 后加权求和；右下为 MLA 细节：低秩 Latent 投影 c^Q、c^KV，部分分支过 RoPE，训练时缓存的 KV 标注为推理期缓存" style="width: 85%;">
</div>

> **图片来源**：DeepSeek-V3 Technical Report（arXiv 2412.19437v2）Figure 2，§2.1 Basic Architecture。本地副本：`references/papers/deepseek-v3-report/deepseek-v3-report.md` 第 127 行引用图。
```

### 1.5 `deepseek-v3-fig4-dualpipe-overlap.jpg`

| 项 | 值 |
| --- | --- |
| 尺寸 / 体积 | 1134 × 129 / 24,049 B |
| 来源 | `references/papers/deepseek-v3-report/images/ea1a52decfb55f603ceea7de1e14bbb72c8e161ed45f792c75afdcfc85a8b59b.jpg` |
| 出处 | DeepSeek-V3 Technical Report Figure 4（md 第 291 行引用，所属小节为 §3.1 Compute Clusters，正文由 §3.2.1 DualPipe 引用） |
| 逐张核对结果 | 上下两行时间轴：上排 Computation（MLP(B) / MLP(W) / MLP(F) / ATTN(B) / ATTN(W) / ATTN(F)），下排 Communication（DISPATCH(F) / DISPATCH(B) / COMBINE(F) / PP / COMBINE(B)）；三角标区分 Forward chunk（空心）与 Backward chunk（实心），每个计算块正下方压着对应的通信块，体现「all-to-all 与 PP 完全被隐藏」 |
| 建议章节 | 计算通信重叠 → 「DeepEP ≠ DualPipe」的边界说明（说明 DeepEP 提供的是 all-to-all 原语，重叠调度由上游框架决定） |

```
<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/deepseek-v3-fig4-dualpipe-overlap.jpg" alt="DeepSeek-V3 Figure 4：一对前向/反向 chunk 的重叠策略。上排 Computation 为 MLP(B)/MLP(W)/MLP(F) 与 ATTN(B)/ATTN(W)/ATTN(F)，下排 Communication 为 DISPATCH(F)/DISPATCH(B)/COMBINE(F)/PP/COMBINE(B)，每个通信段与对应的计算段在时间上对齐，空心三角为 forward chunk、实心三角为 backward chunk" style="width: 95%;">
</div>

> **图片来源**：DeepSeek-V3 Technical Report（arXiv 2412.19437v2）Figure 4，§3.1 Compute Clusters（§3.2.1 DualPipe and Computation-Communication Overlap 引用此图）。本地副本：`references/papers/deepseek-v3-report/deepseek-v3-report.md` 第 291 行引用图。图注原文说明橙色为 forward、绿色为 backward for input、蓝色为 backward for weights、紫色为 PP 通信、红色为 barrier，并声明 all-to-all 与 PP 通信都可以被完全隐藏。
```

### 1.6 `deepseek-v3-fig5-dualpipe-schedule.jpg`

| 项 | 值 |
| --- | --- |
| 尺寸 / 体积 | 1256 × 212 / 68,536 B |
| 来源 | `references/papers/deepseek-v3-report/images/e6f2d6c79a43060cbcca6151b7ab9f65a07022a2b6785819b193e4c6639157a4.jpg` |
| 出处 | DeepSeek-V3 Technical Report Figure 5（md 第 306 行引用，所属小节为 §3.2.1 DualPipe and Computation-Communication Overlap） |
| 逐张核对结果 | 8 行（Device 0–7）× 时间轴的调度网格，格子内是 micro-batch 编号；两个方向同时填充流水线，图例区分 Forward、Backward、Backward for input、Backward for weights、Overlapped forward & Backward；图注说明为 8 PP ranks / 20 micro-batches、双向镜像 |
| 建议章节 | 计算通信重叠 → DualPipe 的调度全貌（讲清「通信被藏起来」依赖的是调度，而不是某个 kernel 的加速） |

```
<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/deepseek-v3-fig5-dualpipe-schedule.jpg" alt="DeepSeek-V3 Figure 5：8 个 PP rank、20 个 micro-batch 的双向 DualPipe 调度网格，Device 0-7 各行按时间排列 micro-batch，橙为 forward、绿为 backward、深绿为 backward for input、蓝为 backward for weights、橙绿相间为重叠的 forward 与 backward，两侧流水线镜像推进" style="width: 95%;">
</div>

> **图片来源**：DeepSeek-V3 Technical Report Figure 5，§3.2.1 DualPipe and Computation-Communication Overlap。本地副本：`references/papers/deepseek-v3-report/deepseek-v3-report.md` 第 306 行引用图。反向方向的 micro-batch 与正向对称，图注省略了它们的 batch ID。
```

### 1.7 `deepseek-v3-fig6-fp8-mixed-precision.jpg`

| 项 | 值 |
| --- | --- |
| 尺寸 / 体积 | 1234 × 346 / 47,119 B |
| 来源 | `references/papers/deepseek-v3-report/images/fb915dd3aeea3d7d5996a4d9cbcad0acab98d9ca4cd190c80373101c459c4f82.jpg` |
| 出处 | DeepSeek-V3 Technical Report Figure 6（md 第 336 行引用，所属小节为 §3.3 FP8 Training） |
| 逐张核对结果 | 一张数据流图：Input（BF16）→ Fprop 的 FP8 GEMM（FP32 累加）→ Output（BF16）；Output Gradient（BF16）→ Dgrad（FP8，输出到 Input Gradient 的 BF16）与 Wgrad（FP8，输出 BF16）；Weight 从 Master Weight 以 FP8 广播给 Fprop 与 Wgrad；Wgrad 产出 FP32 Weight Gradient 再回到 Master Weight，Optimizer States 经 FP32 写回；绿色标注的 "To FP8" 是低精度转换点 |
| 建议章节 | 通信量估算 → 为什么 dispatch 走 FP8 而 combine 回 BF16（通信 payload 的数据类型来源） |

```
<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/deepseek-v3-fig6-fp8-mixed-precision.jpg" alt="DeepSeek-V3 Figure 6：FP8 混合精度框架。Input 为 BF16，Fprop 以 FP8 做 GEMM、FP32 累加并输出 BF16；Output Gradient 为 BF16，Dgrad 输出 FP8、Wgrad 输出 FP8；Weight 由 Master Weight 以 FP8 提供，Wgrad 得到 FP32 Weight Gradient 更新 Master Weight，Optimizer States 以 FP32 参与" style="width: 90%;">
</div>

> **图片来源**：DeepSeek-V3 Technical Report Figure 6，§3.3 FP8 Training（图注说明为简化起见只画 Linear 算子）。本地副本：`references/papers/deepseek-v3-report/deepseek-v3-report.md` 第 336 行引用图。
```

### 1.8 `deepseek-v3-fig7a-fp8-fine-grained-quant.jpg`

| 项 | 值 |
| --- | --- |
| 尺寸 / 体积 | 721 × 601 / 50,330 B |
| 来源 | `references/papers/deepseek-v3-report/images/8afcbbea54504587dfc7fc155c4f5c6cf4263d8cab7c1d77e2be58235bf6f860.jpg` |
| 出处 | DeepSeek-V3 Technical Report Figure 7(a)（md 第 347 行引用，§3.3.1） |
| 逐张核对结果 | 左侧 Input 按 1×N_c 分块、每块一个 Scaling Factor；右侧 Weight 按 N_c×N_c 分块、每块一个 Scaling Factor（图中以粉色方块表示 scale）；下方虚线框给出数据流：Tensor Core 做输入×权重，Output 在 CUDA Core 上用 Scaling Factor 反量化，并标注 N_c 间隔 |
| 建议章节 | 通信量估算 → FP8 dispatch 的 scale 布局（每个 128 元素一个 FP32 scale，是把 231 MiB 这个账算准的关键前提） |

```
<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/deepseek-v3-fig7a-fp8-fine-grained-quant.jpg" alt="DeepSeek-V3 Figure 7(a)：细粒度量化。Input 的激活按 1×N_c 分组、每组一个 Scaling Factor；Weight 按 N_c×N_c 分块、每块一个 Scaling Factor；Tensor Core 计算后，Output 在 CUDA Core 上乘回 Scaling Factor（标注 N_c 间隔）" style="width: 55%;">
</div>

> **图片来源**：DeepSeek-V3 Technical Report Figure 7(a)，§3.3.1。本地副本：`references/papers/deepseek-v3-report/deepseek-v3-report.md` 第 347 行引用图。论文正文明确：激活在 1×128 tile 粒度分组（per token per 128 channels），权重在 128×128 block 粒度分组。
```

### 1.9 `deepseek-v3-fig7b-fp8-accumulation-precision.jpg`

| 项 | 值 |
| --- | --- |
| 尺寸 / 体积 | 431 × 512 / 28,868 B |
| 来源 | `references/papers/deepseek-v3-report/images/f1df2e9f8ff579bf54458b74531b9b78ce5282ae199c170e1ab612451553a31a.jpg` |
| 出处 | DeepSeek-V3 Technical Report Figure 7(b)（md 第 349 行引用，§3.3.1） |
| 逐张核对结果 | 上方为 WGMMA 1 与 WGMMA 4 两段 Tensor Core 运算（低精度累加，粉色块为 Low Prec Acc），下方为 CUDA Core 上的 Output：每隔 N_c 的 Interval 把部分结果搬到 FP32 Register（粉色）并乘 Scaling Factor（青色） |
| 建议章节 | 通信量估算 → FP8 通信的精度代价（可选；只在需要解释「scale 为什么必需」时使用） |

```
<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/deepseek-v3-fig7b-fp8-accumulation-precision.jpg" alt="DeepSeek-V3 Figure 7(b)：Tensor Core 的 WGMMA 内为低精度累加，每经过 N_c 间隔把部分结果提升到 CUDA Core 的 FP32 Register，再与 Scaling Factor 相乘完成高精度累加" style="width: 45%;">
</div>

> **图片来源**：DeepSeek-V3 Technical Report Figure 7(b)，§3.3.1。本地副本：`references/papers/deepseek-v3-report/deepseek-v3-report.md` 第 349 行引用图。
```

### 1.10 `deepseek-v3-fig9-expert-load.jpg`

| 项 | 值 |
| --- | --- |
| 尺寸 / 体积 | 1243 × 598 / 108,435 B |
| 来源 | `references/papers/deepseek-v3-report/images/5ed9b418dd78752d04a362f42e53abe50f3dfd6dea682e641aba9888e2ffab14.jpg` |
| 出处 | DeepSeek-V3 Technical Report Figure 9（md 第 543 行引用，所属小节为 §4.5.3 Batch-Wise Load Balance VS. Sequence-Wise Load Balance） |
| 逐张核对结果 | 四张热力图堆叠：Aux-Loss-Based Layer 9 / Aux-Loss-Free Layer 9 / Aux-Loss-Based Layer 18 / Aux-Loss-Free Layer 18，纵轴为 Wikipedia(en)、Github、DM Mathematics 三个域，横轴为 expert 编号 1–64；Colorbar 为 Relative Expert Load（0–10），aux-loss-free 的色块明显更不均匀 |
| 建议章节 | 概念层或 V2 SM 估算 → 「假设 gate 分布均衡」的前提（论文证据说明真实负载并不均衡，与 `get_theoretical_num_sms` 的 "assumes a balanced gate distribution" 形成对照） |

```
<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/deepseek-v3-fig9-expert-load.jpg" alt="DeepSeek-V3 Figure 9：Layer 9 与 Layer 18 上 aux-loss-based 与 aux-loss-free 模型的专家负载热力图，纵轴为 Wikipedia(en)/Github/DM Mathematics，横轴为 64 个专家的相对负载（0-10）；aux-loss-free 出现明显更集中的深色高负载块" style="width: 85%;">
</div>

> **图片来源**：DeepSeek-V3 Technical Report Figure 9，§4.5.3 Batch-Wise Load Balance VS. Sequence-Wise Load Balance。本地副本：`references/papers/deepseek-v3-report/deepseek-v3-report.md` 第 543 行引用图。图中 "Relative Expert Load" 的定义是「实际专家负载 / 理论均衡负载」的比值，取自 Pile test set 的三个域，论文只展示了两层作为示例。
```

### 1.11 `deepep-v1-normal-overlap.png`

| 项 | 值 |
| --- | --- |
| 尺寸 / 体积 | 2934 × 724 / 518,233 B |
| 来源 | 上游 DeepEP 仓库 `figures/normal.png`，blob `7920c177d869cd691c5bcdc9da59ea4907f9cdf4` |
| 出处 | DeepEP `README.md`（tag `v1.2.1`，§ Example use in training 之后）第 229 行 `![normal](figures/normal.png)`；该 blob 在 `v1.2.1` 与 `01dc3aaa` 下完全一致（见第 4 节） |
| 逐张核对结果 | 上下两条泳道：GPU 侧依次是 Notify → Dispatch（内含 2 个 IB chunk + 2 个 NVL chunk）→ Computation kernels → Combine（内含 2 个 NVL chunk + 2 个 IB chunk）；CPU 侧依次是 Launch notify → Waiting → Tensor allocation → Launch dispatch → Launch computation → Launch combine。箭头标注 "Notify tensor size ASAP" 与 "Reuse layout information"，右上角注明 "Notes: just for demo, real cases may have hundreds of chunks" |
| 建议章节 | V1 normal 模式 → 「为什么 dispatch 需要一次隐式 CPU 等待」（对齐 notify → 分配 → dispatch 的启动序列） |

```
<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/deepep-v1-normal-overlap.png" alt="DeepEP normal 模式时序图：GPU 侧为 Notify、Dispatch（IB chunk 与 NVL chunk 混排）、Computation kernels、Combine（NVL chunk 与 IB chunk 混排）；CPU 侧为 Launch notify、Waiting、Tensor allocation、Launch dispatch、Launch computation、Launch combine；标注 Notify tensor size ASAP 与 Reuse layout information，图注说明真实场景可能有成百上千个 chunk" style="width: 95%;">
</div>

> **图片来源**：DeepEP 仓库 `figures/normal.png`（tag `v1.2.1`，commit `9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee`；该 blob 与 `01dc3aaa` 下一致）。它配的是 README 中这句说明：dispatch 内部并不知道当前 rank 要收多少 token，因此会引入一次针对 GPU 侧接收计数的隐式 CPU 等待。
```

### 1.12 `deepep-v1-low-latency-overlap.png`

| 项 | 值 |
| --- | --- |
| 尺寸 / 体积 | 2928 × 1068 / 688,315 B |
| 来源 | 上游 DeepEP 仓库 `figures/low-latency.png`，blob `777df8de0785ddbaadf020c66fb40d0517ea91ce` |
| 出处 | DeepEP `README.md`（tag `v1.2.1`）第 292 行 `![low-latency](figures/low-latency.png)`；blob 在 `v1.2.1` 与 `01dc3aaa` 下一致 |
| 逐张核对结果 | 上下对比：上半 "Traditional overlapping with communication SMs"，Stream 0 与 Stream 1 各自顺序执行 Attention → Dispatch → MoE → Combine；下半 "Overlapping without communication SMs → Faster computation with more SMs"，Stream 0 把 Attention 0/1、MoE 0/1 与 with background RDMA 的段落串起来，底部标注 Dispatch 0 issue / Dispatch 0 receive + Dispatch 1 issue / Dispatch 1 receive + Combine 0 issue / Combine 0 receive 等时序点 |
| 建议章节 | V1 low-latency 模式 → hook-based 的 0 SM 重叠（并作为 V2「0 SM RDMA low-latency 不再支持」的对照物） |

```
<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/deepep-v1-low-latency-overlap.png" alt="DeepEP low-latency 模式重叠示意图：上图为传统做法，Stream 0/1 依次执行 Attention、Dispatch、MoE、Combine 并占用通信 SM；下图为 hook-based 做法，两条 micro-batch 的 Attention/MoE 与 background RDMA 重叠，不再占用通信 SM，底部标出 Dispatch 与 Combine 的 issue/receive 时序" style="width: 95%;">
</div>

> **图片来源**：DeepEP 仓库 `figures/low-latency.png`（tag `v1.2.1`，commit `9af0e0d0e74f3577af1979c9b9e1ac2cad0104ee`；该 blob 与 `01dc3aaa` 下一致）。README 原文说明：两个 micro-batch 重叠时，RDMA 流量在后台发生、不占用计算侧的 SM；但 attention/dispatch/MoE/combine 四段的耗时不一定相等，需要按 workload 调整 stage 设置。
```

## 2. 参考图 → 文章章节映射（全量账目）

两篇论文的抽取图共 **122** 张（DeepSeekMoE 34 + DeepSeek-V3 88），本节逐张交代去向。判定「正文 Figure」的依据是论文 md 的图片引用行与其后的图注文字；判定「表格渲染图」的依据是实图内容为排版好的表格；其余归入「碎片」（页眉页脚、公式片段、图内小图切块）。

### 2.1 DeepSeekMoE（34 张）

| 图块 | md 引用 | 类型 | 去向与理由 |
| --- | --- | --- | --- |
| `ac61dc9d…` Figure 1 | L19 | 正文图 | **未采用**。Open LLM Leaderboard 上的平均性能 vs 激活参数量散点图，讲的是模型性价比，与通信/专家结构无关 |
| `f7720591…` Figure 2 | L74 | 正文图 | **已采用** → `deepseek-moe-fig2-fine-grained-shared-expert.jpg`（细粒度切分 + 共享专家隔离） |
| `0c75664b…` Figure 3 | L227 | 正文图 | **未采用**。消融实验的归一化柱状图（4 种专家配置 × 6 个 benchmark），用于论证模型设计，不涉及通信形态 |
| `db414b37…` Figure 4 | L240 | 正文图 | **未采用**。禁用 top routed expert 比例下的 Pile loss，讨论的是「routed expert 冗余度」，与本文主题距离较远 |
| `a40c85d1…` Figure 5 | L249 | 正文图 | **已采用** → `deepseek-moe-fig5-activated-experts.jpg`（激活专家数与 Pile loss） |
| `b7f2ad03…` Figure 6 | L252 | 正文图 | **已采用** → `deepseek-moe-fig6-gshard-vs-deepseekmoe.jpg`（同激活参数量对照，选配） |
| `3b942c58…` Figure 7 | L544 | 正文图 | **未采用**。9 宫格训练曲线（DeepSeekMoE 16B vs DeepSeek 7B Dense），属于训练过程料，与文章主线无关 |
| 9 张表格渲染图 | 无 | 表格图 | **未采用**：`0ef10fd8`、`0f4a4ffd`、`11b08751`、`298010ee`、`3f30f593`、`5cd9a900`、`c467f176`、`d316874c`、`df311e60` 以及 `2ee407f8`（超参表）。合计 10 张，全部为论文表格的图片化产物，若需要应用 Markdown 表格重排，不应插图 |
| 17 张公式块 | 无 | 公式块 | **未采用**。体积均小于 20 KB（2.7–10 KB），是论文式 (1)–(17) 的抽取产物；md 正文已用 LaTeX 还原，因此图片冗余 |

小计：34 = 7 张正文图（采用 3 张）+ 10 张表格渲染图 + 17 张公式块。DeepSeekMoE 侧的逐张判读另有 `research/image-map-deepseek-moe.md` 一份更细的记录，其分类与本表一致。

### 2.2 DeepSeek-V3 Technical Report（88 张）

| 图块 | md 引用 | 类型 | 去向与理由 |
| --- | --- | --- | --- |
| `17d0e319…` Figure 1 | L11 | 正文图 | **未采用**。与竞品的 benchmark 柱状图，属于模型能力对比 |
| `d9af045b…` Figure 2 | L127 | 正文图 | **已采用** → `deepseek-v3-fig2-mla-deepseekmoe-architecture.jpg`（模型全貌） |
| `6a5b2be7…` Figure 3 | L238 | 正文图 | **未采用**。MTP 的多 token 预测结构，与 expert-parallel 通信没有直接关系（若正文讨论 PP 与 MTP 的耦合可再考虑） |
| `ea1a52de…` Figure 4 | L291 | 正文图 | **已采用** → `deepseek-v3-fig4-dualpipe-overlap.jpg`（一对前向/反向 chunk 的重叠） |
| `e6f2d6c7…` Figure 5 | L306 | 正文图 | **已采用** → `deepseek-v3-fig5-dualpipe-schedule.jpg`（DualPipe 调度全貌） |
| `fb915dd3…` Figure 6 | L336 | 正文图 | **已采用** → `deepseek-v3-fig6-fp8-mixed-precision.jpg`（FP8 混合精度框架） |
| `8afcbbea…` Figure 7(a) | L347 | 正文图 | **已采用** → `deepseek-v3-fig7a-fp8-fine-grained-quant.jpg`（1×128 / 128×128 分组量化） |
| `f1df2e9f…` Figure 7(b) | L349 | 正文图 | **已采用** → `deepseek-v3-fig7b-fp8-accumulation-precision.jpg`（N_c 间隔提升到 FP32 累加，选配） |
| `b855efd8…` Figure 8 | L466 | 正文图 | **未采用**。128K 上下文的 NIAH 测试热力图，与通信无关 |
| `5ed9b418…` Figure 9 | L543 | 正文图 | **已采用** → `deepseek-v3-fig9-expert-load.jpg`（expert 负载不均衡，与「gate 分布均衡」假设对照） |
| `d94eaa15…` Figure 10 | L1009 | 附录图 | **未采用**。附录 B.1 的 BF16 vs FP8 损失曲线，属于训练数值稳定性证据 |
| 37 张附录 C tile | L1024–L1100 | 附录图阵 | **未采用**。附录 C 逐层铺开的 expert load 热力图（含 `(a) Layers 1-7`、`(b) Layers 7-13`、Aux-Loss-Based Layer 13/19 等分块），信息与正文 Figure 9 重复，只是层数更全 |
| 12 张表格/名单图 | 无 | 表格图 | **未采用**：`d1ed1e21`（V2/V2.5/LLaMA/Qwen 对比表）、`289359f4`（与 GPT-4o/Claude 对比表）、`5b3ce649` 与 `4fb1b038`（作者名单）、`337acd5a`（aux-loss 消融表）、`69e313a9`（MTP 消融表）、`30d70ee1`（Chat/Safety/Reasoning 表）、`d7b43929`（Arena-Hard 表）、`ebf9b32c`（训练成本表）、`22b20fa2`（R1 distill 表）、`131abc35`（1F1B / ZB1P / DualPipe 的 bubble 公式表）、`022638e9`（GRPO 目标函数）。其中 `131abc35` 的内容与计算通信重叠相关，但**应以 Markdown 表格重排后引用**，不使用这张图片 |
| 28 张公式块 | 无 | 公式块 | **未采用**。体积小于 20 KB（2.2–10.5 KB），是正文公式的抽取产物，md 正文已用 LaTeX 还原 |

小计：88 = 48 张 md 引用图（正文 Figure 1–9 共 10 个图文件 + 附录 B.1 Figure 10 + 附录 C 的 37 个 tile）+ 40 张未引用图（12 张表格/名单 + 28 张公式块）。

### 2.3 碎片里的公式（不采用为图片，但应当转写为正文公式）

两篇论文中体积最小的一批抽取块几乎全是**公式的切片**，它们恰好覆盖了本文数学一节需要的原始式子，因此结论是「不插图、但要按原文转写成 LaTeX 公式」。逐条复核后的清单如下（行号均为两篇论文 md 中可直接复核的位置）：

| 来源 | 式子 | 行号 | 用途 |
| --- | --- | --- | --- |
| DeepSeekMoE | $h^l_t=\sum_{i=1}^{N}\left(g_{i,t}\,\mathrm{FFN}_i(u^l_t)\right)+u^l_t$ | `deepseek-moe.md:61`（式 3） | 门控加权求和的原始形式 |
| DeepSeekMoE | $g_{i,t}=\begin{cases}s_{i,t}, & s_{i,t}\in\mathrm{Topk}(\{s_{j,t}\mid 1\le j\le N\},K)\\ 0, & \text{otherwise}\end{cases}$ | `:65`（式 4） | Top-K 指示函数，稀疏化的关键定义 |
| DeepSeekMoE | $s_{i,t}=\mathrm{Softmax}_i\left({u^l_t}^\top e^l_i\right)$ | `:69`（式 5） | routing score 的原始定义 |
| DeepSeekMoE | $\mathcal{L}_{\mathrm{ExpBal}}=\alpha_1\sum_{i=1}^{N'} f_i P_i$，$f_i=\frac{N'}{K'T}\sum_t \mathbb{1}(\cdot)$，$P_i=\frac{1}{T}\sum_t s_{i,t}$ | `:132`、`:136`、`:140`（式 12–14） | 专家级负载均衡损失 |
| DeepSeekMoE | $\mathcal{L}_{\mathrm{DevBal}}=\alpha_2\sum_{i=1}^{D} f'_i P'_i$，$f'_i=\frac{1}{|\mathcal{E}_i|}\sum_{j\in\mathcal{E}_i} f_j$ | `:148`、`:152`（式 15–16） | 设备级（EP 组级）负载均衡 |
| DeepSeek-V3 | $u_t=W^O[o_{t,1};o_{t,2};\dots;o_{t,n_h}]$，$W^O\in R^{d\times d_h n_h}$ | `deepseek-v3-report.md:183`、`:186`（式 11） | MLA 输出投影，讲 attention 侧算力占比时的口径 |
| DeepSeek-V3 | $h'_t=u_t+\sum_{i=1}^{N_s}\mathrm{FFN}^{(s)}_i(u_t)+\sum_{i=1}^{N_r} g_{i,t}\mathrm{FFN}^{(r)}_i(u_t)$ | `:193`（式 12） | 共享专家**不带门控权重**、routed expert 带权重 |
| DeepSeek-V3 | $g_{i,t}=\dfrac{g'_{i,t}}{\sum_{j=1}^{N_r} g'_{j,t}}$ | `:197`（式 13） | **路由权重在所选专家内部归一化**，决定 DeepEP 拿到的 `topk_weights` 是什么口径 |
| DeepSeek-V3 | $g'_{i,t}=\begin{cases}s_{i,t}, & s_{i,t}\in\mathrm{Topk}(\{s_{j,t}\mid 1\le j\le N_r\},K_r)\\ 0, & \text{otherwise}\end{cases}$ | `:201`（式 14） | 与式 4 的区别在于：先在 $N_r$ 个 routed expert 中取 Top-$K_r$，再归一化 |
| DeepSeek-V3 | $s_{i,t}=\mathrm{Sigmoid}\left(u_t^\top e_i\right)$ | `:205`（式 15） | V3 用 sigmoid 取代 softmax 计算 affinity |

写正文时还有两条容易被忽略、但直接影响通信语义的原文事实，建议一并落到数学一节：

其一，`deepseek-v3-report.md:210` 与 `:216` 说明 V3 的 bias 项 $b_i$ **只参与 top-K 的选择**，真正乘到 FFN 输出上的 gating value 仍然来自原始 affinity 分数 $s_{i,t}$。其二，`:241` 说明 node-limited routing 的约束是「每个 token 最多送到 $M$ 个节点」，节点按其上专家的最高 $\frac{K_r}{M}$ 个 affinity 分数之和选出；`:463` 给出训练配置为每个 token 最多 4 个节点、每层 routed experts 均匀部署在 8 个节点共 64 张 GPU 上。这两条决定了 DeepEP 在 V3 里的通信形态是「节点内 NVLink + 节点间 RDMA」的两级结构，而不是任意的 all-to-all 拓扑。

把这些式子写进正文时，符号的 shape 与单位必须在正文里交代清楚（例如 $s_{i,t}$ 是标量 score、$e_i$ 是第 $i$ 个专家的路由向量、$N_r$ 是 routed expert 数），并且要说明它们属于论文的模型侧定义，与 DeepEP API 里 `topk_idx` / `topk_weights` 的命名不是同一套符号系统。

## 3. 未采用图的三条通用理由

第一类是与主题无关的正文图：benchmark 对比、训练曲线、NIAH、MTP 结构等，回答的是「模型好不好」，而本文要回答的是「token 在 rank 之间怎么走」。第二类是表格的图片化产物：MinerU 把论文表格渲染成 jpg，这类内容若确实需要，应当用 Markdown 表格重排后引用，而不是贴图（否则读者无法检索、无法复制数值）。第三类是公式块：它们在 md 里已经被还原成 LaTeX，作为图片重复出现并没有增量信息，正文要引用的是 LaTeX 式子本身。

需要特别提醒一点：**不要把「未引用的图」当成「论文里不存在的图」**。这两篇论文的 md 是 MinerU 转换产物，正文图片全部被引用，但表格被渲染为图片并脱离了引用行，因此 `references/README.md` 里「未被 md 引用的抽取图块」这一统计不能直接当作「无用碎片数」使用。

## 4. 版本分层核对（上游 DeepEP 官方图）

用户材料把 `figures/normal.png` 与 `figures/low-latency.png` 归为「DeepEP V1 示意图」，这一点经核对需要精确化：

| 核对项 | 结果 |
| --- | --- |
| 文件存在性 | `v1.2.1` 与 `01dc3aaa` 下均有 `figures/normal.png`、`figures/low-latency.png` |
| 是否同一份 | 是。`git hash-object` 与 `git rev-parse v1.2.1:<path>` 一致：normal 为 `7920c177…`，low-latency 为 `777df8de…` |
| v1.2.1 是否引用 | 是。README 第 229 行（normal）与第 292 行（low-latency） |
| `01dc3aaa` 是否引用 | 否。V2 重写的 README 里已经没有 `figures/` 引用，但文件仍保留在仓库中 |
| 图的实际内容 | `normal.png` 是 **normal 模式的启动时序**（notify → tensor allocation → dispatch → computation → combine，标注隐式 CPU 等待与 layout 复用），不是 kernel 结构图；`low-latency.png` 是 **low-latency 的两 micro-batch 重叠对比**（占用通信 SM vs hook-based 不占用 SM），不是 low-latency kernel 的内部结构图 |

由此得到的写作约束：这两张图在正文里应当配「V1 README 的原始说明」引用，并注明它们是 v1.2.1 README 的配图、blob 至今未变；**不要**把它们描述成「V1 normal / low-latency 的实现结构图」，也不要在讲 V2 的时候暗示这是 V2 的图。V2 的 hybrid dispatch / warp specialization 结构若需要图，参考资料里没有对应图，只能自绘 Mermaid。

## 5. 复核记录

核对时间：2026-09-10（本机时区 America/New_York）。**122 张抽取图已全部实看**：DeepSeekMoE 的 17 张 ≥20 KB 图块与 17 张碎片、DeepSeek-V3 的 60 张 ≥20 KB 图块与 28 张碎片，其中体积较大的图逐张单独查看，附录 C 的 37 个 tile 与两组碎片以图版（contact sheet）一次性核完。核对方式与命令：

| 步骤 | 命令 |
| --- | --- |
| 图片数量与体积 | `ls -la references/papers/deepseek-moe/images`、`ls -la references/papers/deepseek-v3-report/images` |
| md 引用位置 | 对 md 逐行匹配 `images/[0-9a-f]+\.jpg`，记录引用行号并抓取紧随其后的图注行 |
| 正文图 / 表格图 / 公式块分类 | 对体积 ≥ 20 KB 的图逐张实看；附录 C 的 37 个 tile 与 45 张公式块（DeepSeekMoE 17 + DeepSeek-V3 28）用 Python/PIL 拼成图版后整版实看，逐块确认类型 |
| 上游图版本一致性 | `git hash-object figures/normal.png`、`git rev-parse v1.2.1:figures/normal.png`（在 `/workspace/algorithm/DeepEP`） |
| 复制后一致性 | `sha256sum pics/*`，与源文件逐一同名比对 |

未完成项：本目录 12 张图未做像素级逐一比对（只比对文件大小与头部解析出的尺寸）；论文抽取图本身是 MinerU 的 jpg 转写产物，若后续需要印刷级精度，应回到 `references/raw/` 的原始 PDF 重新导出。
