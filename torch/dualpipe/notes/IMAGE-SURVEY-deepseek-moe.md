# DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models（arXiv 2401.06066v1）图片逐张核对

资料定位（对 `torch/dualpipe/deep-dive.md` 而言）：**EP（专家并行）通信从何而来的模型侧背景**。文章用它解释 DualPipe 要掩盖的那类 All-to-All 通信为什么必须存在——细粒度专家切分把专家数从 `N` 放大到 `mN`（每层 63 → 64 个路由专家、每专家 FFN 中间维缩到 `1/m`），共享专家隔离再扣掉 `K_s` 个恒定激活的专家；这两条设计直接决定了 MoE 层里 token 要被 dispatch / combine 往返多少次。

- 本地副本：`torch/dualpipe/references/papers/deepseek-moe/deepseek-moe.md`（MinerU 产物，545 行）
- 图片目录：`torch/dualpipe/references/papers/deepseek-moe/images/`

## 汇总

- `images/` 实际文件数（`ls torch/dualpipe/references/papers/deepseek-moe/images/ | wc -l`）：**34**
- 本清单覆盖行数：**34**（逐张清单 34 行，与 `ls | wc -l` 一致）
- 真插图：**7**（Figure 1–7）；公式截图：**17**；表格截图：**10**；页眉/装饰/碎片：**0**
- 建议采用：**2 张**（1 张核心：Figure 2；1 张可选：Figure 3）
- md 中带图引用的行：第 19、74、227、240、249、252、544 行（共 7 处，对应 7 张真插图）；**另外 27 张图在 md 中没有任何 `![]()` 引用**——其中 10 张是 Table 1–10 的截图（内容已在 md 中以 HTML `<table>` 另行转录，属重复产物），17 张是第 2–3 节与负载均衡章节的行间公式截图（内容已在 md 中以 `$$…\tag{N}$$` 文本形式存在）
- 全部 7 张真插图都**不使用** `Figure N:` 而是 `Figure N |` 的图注格式，与仓库内其他 MinerU 产物一致

## 逐张清单

（按文件名字典序排列，便于与 `ls` 输出逐行对照）

| 文件名 | md 引用行 | 图号/表号 | 类别 | 内容（读图后的事实描述，含图中可读的关键标注/数值） | 结论 | 建议章节 |
| --- | --- | --- | --- | --- | --- | --- |
| `0546107fcba0642dd37c8d252c610defa078d555aad41cef224030de0f52afe4.jpg` | —（md 未引用） | 公式 (3) | 公式截图 | 单行公式，可读为 `h_t^l = Σ_(i=1)^N ( g_(i,t) FFN_i(u_t^l) ) + u_t^l`（传统 MoE 层，专家总数 `N`；md 第 61 行） | 不采用（公式截图，非插图） | — |
| `08472c3fd0c9a375baacf8c526e5d6bd7e6659982e2b8a64373d06d905903dda.jpg` | —（md 未引用） | 公式 (6) | 公式截图 | 单行公式，可读为 `h_t^l = Σ_(i=1)^(mN) ( g_(i,t) FFN_i(u_t^l) ) + u_t^l`（**细粒度切分后专家总数变成 `mN`**，是双引用的直接来源；md 第 88 行） | 不采用（公式截图，非插图） | — |
| `0c75664b014d1ebcd2304451abb55968b12f7d2c48a59126288d5e9b351bb2df.jpg` | 227（图注 228） | Figure 3 | 真插图 | 分组柱状图。纵轴 `Normalized Performance` 0.5–1.2，横轴 `Metrics` = `HellaSwag / PIQA / ARC-easy / ARC-challenge / TriviaQA / NaturalQuestions`（每个指标的 4 根柱按同一指标内的最好值归一化）。图例四条：`0 shared expert + 2 out of 16 routed experts (GShard)`（蓝）、`1 shared expert + 1 out of 15 routed experts (+ shared expert isolation)`（橙）、`1 shared expert + 3 out of 31 routed experts (+ fine-grained expert segmentation)`（绿）、`1 shared expert + 7 out of 63 routed experts (+ finer expert segmentation)`（棕）。可读趋势：TriviaQA 上蓝柱约 `0.61`、橙 `0.85`、绿 `0.93`、棕 `1.00`；NaturalQuestions 上蓝柱约 `0.55`、橙 `0.79`、绿 `0.88`、棕 `1.00`；PIQA / ARC-easy 上四条柱差别很小 | **采用（可选）** | **§9**（选型：细粒度切分 + 共享专家换来的是可观的任务收益，这正是「愿意为 All-to-All 通信付代价」的动机侧证据）；§4 背景（可选） |
| `0ef10fd835911e7a2a8eee00a291505d0db65942d78790e3080103acb3d7b710.jpg` | —（md 未引用） | Table 5 | 表格截图 | 对齐后的 Chat 模型对比表，列头 `Metric / # Shot / LLaMA2 SFT 7B / DeepSeek Chat 7B / DeepSeekMoE Chat 16B`。表头行可读 `# Total Params` 6.7B / 6.9B / 16.4B，`# Activated Params` 6.7B / 6.9B / 2.8B，`FLOPs per 4K Tokens` 187.9T / 183.5T / 74.4T；22 行指标（HellaSwag 67.9/71.0/72.2、MBPP 27.8/39.0/46.2 等） | 可转写为 Markdown 表格，不直接引图；内容（SFT 后的对话模型评测）与本文系统主线无关，建议也不转写 | — |
| `0f4a4ffddd3e6b0e1f60a729f598c94f6ecfdd370f03fd83753840146fdbdda6.jpg` | —（md 未引用） | Table 1 | 表格截图 | 验证实验主表，列头 `Metric / # Shot / Dense / Hash Layer / Switch / GShard / DeepSeekMoE`。关键行可读：`# Total Params` 全为 2.0B；`# Activated Params` 0.2B / 0.2B / 0.2B / 0.3B / 0.3B；`FLOPs per 2K Tokens` 2.9T / 2.9T / 2.9T / 4.3T / 4.3T；`# Training Tokens` 全为 100B；`Pile (Loss)` 2.060 / 1.932 / 1.881 / 1.867 / 1.808；HellaSwag 38.8 / 46.2 / 49.1 / 50.5 / 54.8 | 可转写为 Markdown 表格，不直接引图；内容（MoE 架构横向打榜）与本文系统主线无关，建议也不转写 | — |
| `11b0875151468e94c0aaac959d20bfd61b15b4ee9a0cd25267aff43ef5bf9d57.jpg` | —（md 未引用） | Table 8 | 表格截图 | 附录 B 对比表，列头 `Metric / # Shot / GShard×1.2 / GShard×1.5 / DeepSeekMoE`。关键行可读：`# Experts` `0 + 16` / `0 + 16` / `1 + 63`；`# Activated Experts` `0 + 2` / `0 + 2` / `1 + 7`；`# Total Expert Params` 2.3B / 2.8B / 1.9B；`# Activated Expert Params` 0.28B / 0.35B / 0.24B；`Pile (Loss)` 1.824 / 1.808 / 1.808 | 可转写为 Markdown 表格，不直接引图；若 §9 需要「同激活参数下专家数更少 / 更多」的对照，可只取 `# Experts` 与 `# Activated Expert Params` 两行 | — |
| `298010ee2dcf95d989aea4916721ed727f6839a332ecf6913b6f470ecf3f7798.jpg` | —（md 未引用） | Table 4 | 表格截图 | 列头 `Metric / # Shot / LLaMA2 7B / DeepSeekMoE 16B`。关键行可读：`# Total Params` 6.7B / 16.4B；`# Activated Params` 6.7B / 2.8B；`FLOPs per 4K Tokens` 187.9T / 74.4T；`# Training Tokens` 2T / 2T；`Pile (BPB)` 0.76 / 0.74 | 可转写为 Markdown 表格，不直接引图；内容（与开源 dense 模型对比）与本文系统主线无关，建议也不转写 | — |
| `2ee407f842d90c2ad2fd0d6fbcb7eb72d615c80e1e530aee6d6191f96a518cd0.jpg` | —（md 未引用） | Table 7 | 表格截图 | 超参数总览表，列头 `# Params / # Layers / Hidden Size / # Attn Heads / # Shared Experts / # Routed Experts / Relative Expert Size / Sequence Length / Batch Size (Sequence) / Learning Rate`。三行全部可读：`2.0B` → 9 / 1280 / 10 / **1** / **63 (7 activated)** / **0.25** / 2048 / 2048 / 1.08e-3；`16.4B` → 28 / 2048 / 16 / **2** / **64 (6 activated)** / **0.25** / 4096 / 4608 / 4.2e-4；`144.6B` → 62 / 4096 / 32 / **4** / **128 (12 activated)** / **0.125** / 4096 / 4608 / 3.0e-4 | 可转写为 Markdown 表格，不直接引图（md 第 518 行已有同内容 HTML 表） | **§4 / §7**（本文需要的模型侧量化锚点：`# Shared Experts` 1/2/4、`# Routed Experts` 63/64/128、`Relative Expert Size` 0.25→0.125。估算 All-to-All 通信量与 EP 组的大小时必须用这组数字——**同一层内专家数最多 128，就是「All-to-All 为什么会成为瓶颈」的最直接解释**） |
| `3553478b6082f91947bd9e88e697d2fb17c50b7818981a4b3de4c1e6791760a2.jpg` | —（md 未引用） | 公式 (11) | 公式截图 | 单行公式，可读为 `s_(i,t) = Softmax_i( u_t^(l^T) e_i^l ) .`（**末尾是句号**，据此对应 (11)；md 第 118 行） | 不采用（公式截图，非插图） | — |
| `3b942c584214c882c9054ea8fcbe57b1cab016fb4a8255bfd037a9b2590e0707.jpg` | 544（图注 545） | Figure 7 | 真插图 | 6 行 × 3 列共 **18 个面板**的训练曲线，横轴统一为 `# Training Tokens (B)` 0–2000，每面板两条曲线：`DeepSeekMoE 16B`（橙）与 `DeepSeek 7B (Dense)`（蓝）。面板标题按行读为：`HellaSwag (Acc.) / PIQA (Acc.) / ARC-easy (Acc.)`、`ARC-challenge (Acc.) / RACE-middle (Acc.) / RACE-high (Acc.)`、`DROP (EM) / GSM8K (EM) / HumanEval (Pass@1)`、`MBPP (Pass@1) / TriviaQA (EM) / NaturalQuestions (EM)`、`MMLU (Acc.) / WinoGrande (Acc.) / CLUEWSC (EM)`、`CEval (Acc.) / CMMLU (Acc.) / CHID (Acc.)`。可读趋势：多数面板两条曲线几乎重合；`HumanEval` / `CMMLU` / `CEval` 上橙线在 500B token 之后略高 | 不采用（内容为 16B 模型的训练过程曲线，与流水线调度及 All-to-All 通信均无关） | — |
| `3f30f593f141eb114ab82f40cd366eeae5e87c1e40ec4bc3802d9bc883c70d1b.jpg` | —（md 未引用） | Table 2 | 表格截图 | 主文 Table 2，列头 `Metric / # Shot / GShard×1.5 / Dense×16 / DeepSeekMoE`。关键行可读：`Relative Expert Size` 1.5 / 1 / **0.25**；`# Experts` `0 + 16` / `16 + 0` / **`1 + 63`**；`# Activated Experts` `0 + 2` / `16 + 0` / **`1 + 7`**；`# Activated Expert Params` 0.35B / 1.89B / **0.24B**；`FLOPs per 2K Tokens` 5.8T / 24.6T / **4.3T**；`# Training Tokens` 100B ×3 | 可转写为 Markdown 表格，不直接引图；若 §9 要论证「细粒度 MoE 用 0.24B 激活专家参数逼近 Dense×16 的 1.89B」，只取 `# Activated Expert Params` 与 `FLOPs per 2K Tokens` 两行即可 | — |
| `403a4e87e0d404f5e6cf9e2df5ca19af624b20866971ab05e8279873a3d36861.jpg` | —（md 未引用） | 公式 (7) | 公式截图 | 分段函数公式，可读为 `g_(i,t) = { s_(i,t), s_(i,t) ∈ Topk({s_(j,t) | 1 ≤ j ≤ mN}, mK); 0, otherwise }`（**细粒度切分后的门控**，取 `mK` 个最高亲和度；md 第 92 行） | 不采用（公式截图，非插图） | — |
| `4c8a29db4be06f9e1cd6eb702318c53773f5901c90c9291803304092e32d9320.jpg` | —（md 未引用） | 公式 (16) | 公式截图 | 单行公式，可读为 `f_i′ = (1/|E_i|) Σ_(j ∈ E_i) f_j`（设备级负载均衡里把一组专家上的 `f` 取均值；md 第 152 行） | 不采用（公式截图，非插图） | — |
| `5336b9fbf9d538299293ff7db31e8ecdc3d8aa6a0405aac57ac5e279d30da6e3.jpg` | —（md 未引用） | 公式 (9) | 公式截图 | 单行长公式，可读为 `h_t^l = Σ_(i=1)^(K_s) FFN_i(u_t^l) + Σ_(i=K_s+1)^(mN) ( g_(i,t) FFN_i(u_t^l) ) + u_t^l`（**完整 DeepSeekMoE 层：前 `K_s` 个共享专家不走门控、恒定激活**；md 第 110 行） | 不采用（公式截图，非插图） | — |
| `5b6a6de32daa0310526ec3a33f8aed6020ad786aec95558ea5944bbc6160cf98.jpg` | —（md 未引用） | 公式 (1) | 公式截图 | 单行公式，可读为 `u_(1:T)^l = Self-Att( h_(1:T)^(l-1) ) + h_(1:T)^(l-1)`（md 第 49 行） | 不采用（公式截图，非插图） | — |
| `5cd9a900c8c373bd90e83ea53d08f796957171d433e4d60249df784b87261e8f.jpg` | —（md 未引用） | Table 10 | 表格截图 | 附录表，列头 `Metric / # Shot / GShard×1.2 / GShard×1.5 / DeepSeekMoE`，但规模放大 5–7 倍。关键行可读：`# Total Expert Params` **15.9B / 19.8B / 13.3B**；`# Activated Expert Params` 2.37B / 2.82B / 2.05B；`# Experts` `0 + 16` / `0 + 16` / `1 + 63`；`# Training Tokens` 100B ×3；HellaSwag 66.6 / 67.7 / **69.1** | 可转写为 Markdown 表格，不直接引图；与 Table 8 是同一对照的不同规模，优先级低 | — |
| `7d517037ab24c4281f12a5a314186a80ec39515a2ab0d3cf750a39721694bb85.jpg` | —（md 未引用） | 公式 (13) | 公式截图 | 单行公式，可读为 `f_i = (N′/(K′T)) Σ_(t=1)^T 1( Token t selects Expert i )`（md 第 136 行） | 不采用（公式截图，非插图） | — |
| `82541b18ac85d67f6a769ea43d54bd6ac2f85f81909c388887e52343e8efd7fd.jpg` | —（md 未引用） | 公式 (12) | 公式截图 | 单行公式，可读为 `L_ExpBal = α_1 Σ_(i=1)^(N′) f_i P_i`（专家级负载均衡损失；md 第 132 行） | 不采用（公式截图，非插图） | — |
| `8496d54827632f35837b32cae9fe3244806223a6f90117c7fc4b3c5294422153.jpg` | —（md 未引用） | 公式 (14) | 公式截图 | 单行公式，可读为 `P_i = (1/T) Σ_(t=1)^T s_(i,t)`（md 第 140 行） | 不采用（公式截图，非插图） | — |
| `93911de63af5c203bb1c89021234c8fe8004e5329ef9be1af3b7614ab5925929.jpg` | —（md 未引用） | 公式 (8)（与 (5) 渲染相同，见存疑 6） | 公式截图 | 单行公式，可读为 `s_(i,t) = Softmax_i( u_t^(l^T) e_i^l ) ,`（**末尾是逗号**；md 第 96 行） | 不采用（公式截图，非插图） | — |
| `a341843d4958ccf935339a7debc07783ee3a9c42a9449a0ad238da3aea5f62d2.jpg` | —（md 未引用） | 公式 (5)（与 (8) 渲染相同，见存疑 6） | 公式截图 | 单行公式，可读为 `s_(i,t) = Softmax_i( u_t^(l^T) e_i^l ) ,`（**末尾是逗号**；md 第 69 行） | 不采用（公式截图，非插图） | — |
| `a40c85d13d4d10480ad94a6298bcc81c4905486a9e1030fe423e6332137bbc28.jpg` | 249（图注 250） | Figure 5 | 真插图 | 折线图。纵轴 `Pile Loss` 1.81–1.96，横轴 `Activated Routed Experts` 3–7。图例两条：`DeepSeekMoE`（橙，圆点连线）与 `GShard (full top-2 activated)`（蓝 `x` 标记）。可读数值：橙色曲线在 3 → `1.955`、4 → `1.869`、5 → `1.832`、6 → `1.816`、7 → `1.812`；蓝色为一条水平虚线 `1.867`，并有一个双向箭头标注 **`same activated expert parameters`** 指向 x=7 处 | 不采用（内容为「DeepSeekMoE 用更少激活专家即达到 GShard 的 Pile loss」，属模型质量论述，与流水线调度无关） | — |
| `aa2268c5670e096bb6d5387ed956315f8a9f036a7ae22814f7e6e524c1aad4b7.jpg` | —（md 未引用） | 公式 (10) | 公式截图 | 分段函数公式，可读为 `g_(i,t) = { s_(i,t), s_(i,t) ∈ Topk({s_(j,t) | K_s + 1 ≤ j ≤ mN}, mK − K_s); 0, otherwise }`（**路由专家的下标从 `K_s + 1` 起、激活数减为 `mK − K_s`**；md 第 114 行） | 不采用（公式截图，非插图） | — |
| `ac61dc9d6ddc91d9dadcb62263a1b1937f2ca66470cb03fd9938ba2d73bbf0bb.jpg` | 19（图注 20） | Figure 1 | 真插图 | 散点图。纵轴 `Average Performance` 36–52，横轴 `Number of Activated Parameters (Billions)` 2–7。可读标注与大致坐标：`DeepSeekMoE 16B`（红色五角星，约 (2.8, 51.2)，上方有灰色水平虚线）、`LLaMA2 7B`（约 (7, 51)）、`LLaMA 7B`（约 (6.6, 45.5)）、`Falcon 7B`（约 (7, 44)）、`Open LLaMA 7B`（约 (6.7, 42.3)）、`RedPajama-INCITE 7B`（约 (6.1, 41.3)）、`GPT-J 6B`（约 (5.8, 40)）、`RedPajama-INCITE 3B`（约 (2.9, 38.4)）、`Open LLaMA 3B`（约 (3.4, 38.3)）、`Pythia 2.8B`（约 (2.7, 37.2)）、`OPT 2.7B`（约 (2.7, 36.6)）、`BLOOM 3B`（约 (3.0, 36.3)）、`GPT-neo 2.7B`（约 (2.6, 36.2)）；一条**红色虚线**穿过除 DeepSeekMoE 外的所有点作线性拟合 | 不采用（内容为 Open LLM Leaderboard 上的质量-规模对比，与流水线调度及 All-to-All 通信均无关；**注意勿把它误当作架构图引用**） | — |
| `b7f2ad03a5499b898dcdecb0b32663662464722ba5f984ef481382dedbc4cddc.jpg` | 252（图注 253） | Figure 6 | 真插图 | 分组柱状图（未归一化）。纵轴 `Performance` 0–80，横轴 `Metrics` = `HellaSwag / PIQA / ARC-easy / ARC-challenge / TriviaQA / NaturalQuestions`。图例两条：`0 shared expert + 2 out of 16 routed experts (GShard)`（蓝）、`1 shared expert + 3 out of 63 routed experts (DeepSeekMoE with half the activated experts)`（橙）。可读数值大致为：HellaSwag 50.6 / 52.0；PIQA 70.5 / 71.2；ARC-easy 44.1 / 48.0；ARC-challenge 31.8 / 33.0；TriviaQA 10.2 / 16.0；NaturalQuestions 2.8 / 5.0（橙柱在所有指标上都不低于蓝柱） | 不采用（内容为「激活专家数减半后仍优于 GShard」，属模型质量论述） | — |
| `c467f1765470b7cb432e8a421c8b3bc8904e119732ffc21794912d2b7f33645b.jpg` | —（md 未引用） | Table 6 | 表格截图 | 145B 规模对比表，列头 `Metric / # Shot / DeepSeek 67B (Dense) / GShard 137B / DeepSeekMoE 145B / DeepSeekMoE 142B (Half Activated)`。关键行可读：`# Activated Params` 67.4B / 21.6B / 22.2B / 12.2B；`Relative Expert Size` N/A / 1 / **0.125** / **0.125**；`# Experts` N/A / `0 + 16` / **`4 + 128`** / **`2 + 128`**；`# Activated Experts` N/A / `0 + 2` / **`4 + 12`** / **`2 + 6`**；`FLOPs per 4K Tokens` 2057.5T / 572.7T / 585.6T / 374.6T | 可转写为 Markdown 表格，不直接引图 | **§4 / §7**（与 Table 7 互补：145B 配置是 `4` 个共享专家 + `128` 个路由专家、激活 `4 + 12`，`Relative Expert Size = 0.125`。这是「专家数越多、EP 组越大、All-to-All 越难掩盖」的极限配置） |
| `d316874c422e9d6f2fab9b9fa9c72c420d34c110e24cf5d08bcc90c4ed34496a.jpg` | —（md 未引用） | Table 3 | 表格截图 | 列头 `Metric / # Shot / DeepSeek 7B (Dense) / DeepSeekMoE 16B`。关键行可读：`# Total Params` 6.9B / 16.4B；`# Activated Params` 6.9B / 2.8B；`FLOPs per 4K Tokens` 183.5T / 74.4T；`# Training Tokens` 2T / 2T；`Pile (BPB)` 0.75 / 0.74 | 可转写为 Markdown 表格，不直接引图；内容（与同门 dense 模型对比）与本文系统主线无关，建议也不转写 | — |
| `db414b37c942820d8466077a07a5cbba03d36a9e48211fa70d6f923dfcc66b9f.jpg` | 240（图注 241） | Figure 4 | 真插图 | 折线图。纵轴 `Pile Loss` 2–9，横轴 `Ratio of Disabled Top Routed Experts` 刻度为 `0 / 1/16 / 2/16 / 3/16 / 4/16`。图例两条：`DeepSeekMoE`（橙，圆点连线）与 `GShard × 1.5`（蓝，`x` 标记）。可读数值：x=0 两线重合于约 `1.81`；`1/16` → 橙 `7.5` vs 蓝 `5.6`；`2/16` → 橙 `8.2` vs 蓝 `7.5`；`3/16` → 橙 `8.65` vs 蓝 `7.6`；`4/16` → 橙 `9.05` vs 蓝 `7.7` | 不采用（内容为「路由专家冗余度更低」的分析，属模型质量论述） | — |
| `dd87c94793da0b9f5c19afed4411f6e9f3f49f5e24f64c37f95a3208cb7b5822.jpg` | —（md 未引用） | 公式 (4) | 公式截图 | 分段函数公式，可读为 `g_(i,t) = { s_(i,t), s_(i,t) ∈ Topk({s_(j,t) | 1 ≤ j ≤ N}, K); 0, otherwise }`（传统 MoE 门控；md 第 65 行） | 不采用（公式截图，非插图） | — |
| `df311e602924be0b9c7e002e1fa071668bde490d0eaaa18bbfcfc21b5535eccd.jpg` | —（md 未引用） | Table 9 | 表格截图 | 附录表，列头 `Metric / # Shot / Dense×4 / Dense×16 / DeepSeekMoE`。关键行可读：`# Experts` `4 + 0` / `16 + 0` / `1 + 63`；`# Total Expert Params` 0.47B / 1.89B / 1.89B；`# Activated Expert Params` 0.47B / 1.89B / **0.24B**；`Pile (Loss)` 1.908 / 1.806 / 1.808 | 可转写为 Markdown 表格，不直接引图；与 Table 2 的 Dense×16 列重合，优先级低 | — |
| `ecfdc6f72c7cc4637f96e099a8a66ba5200048b6007f0dd0d503095572b5e71c.jpg` | —（md 未引用） | 公式 (2) | 公式截图 | 单行公式，可读为 `h_t^l = FFN(u_t^l) + u_t^l`（dense FFN 基线；md 第 53 行） | 不采用（公式截图，非插图） | — |
| `edd2ff38edf9bcd89cb54d6435e8b478f12cbe869b66a6d2acfd587e2aa91d17.jpg` | —（md 未引用） | 公式 (17) | 公式截图 | 单行公式，可读为 `P_i′ = Σ_(j ∈ E_i) P_j`（md 第 156 行） | 不采用（公式截图，非插图） | — |
| `f60a19c29ab4373038763b659575dd93f4228dacc6152dd3c3bd4ac4e344bba5.jpg` | —（md 未引用） | 公式 (15) | 公式截图 | 单行公式，可读为 `L_DevBal = α_2 Σ_(i=1)^D f_i′ P_i′`（设备级负载均衡损失；md 第 148 行） | 不采用（公式截图，非插图） | — |
| `f77205913b6fd151e304c5a70d7e631629fa19ca95d08f9fb369d1ef4762eb67.jpg` | 74（图注 75） | Figure 2 | 真插图 | 三联架构图（三个子图之间用竖直虚线分隔，底部用粗黑箭头串成递进关系 `(a) Conventional Top-2 Routing ➜ (b) + Fine-grained Expert Segmentation ➜ (c) + Shared Expert Isolation (DeepSeekMoE)`）。右上角图例：浅蓝 = `Routed Expert`，绿色 = `Shared Expert`。**每个子图内部结构相同**：底部 `Input Hidden` → `Router`（框内另有一个小的直方图示意门控分布）→ 若干专家框 → 顶部 `⊕` 汇聚 → `Output Hidden`。可读差异：**(a)** 专家框标 `1 2 … N`，`Router` 旁标 **`K = 2`**，两根实线连到被选中的专家、其余用黄色虚线（表示未选中）；**(b)** 专家框标 `1 2 3 4 … 2N-1 2N`（**专家数翻倍**），`Router` 旁标 **`K = 4`**，四条实线；**(c)** 第一个专家框为**绿色**（共享专家）、其余浅蓝，`Router` 旁标 **`K = 3`**，且绿色专家有一根**从 `Input Hidden` 直连的实线**（不经 Router），另有三根实线接路由专家 | **采用** | **§4**（EP / All-to-All 通信的来源：`N → 2N` 的专家数放大 + 共享专家直连，是 DualPipe 要掩盖的 dispatch/combine 通信量的模型侧根据；主用）；**§1**（可选，作为「stage 内通信等待」的背景） |

## 采用图的正文引用块（可直接粘贴）

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/deepseek-moe-fig2-fine-grained-segmentation-shared-expert.jpg" alt="DeepSeekMoE Figure 2：三联架构图。(a) Conventional Top-2 Routing，专家 1…N、Router 旁 K = 2；(b) + Fine-grained Expert Segmentation，专家 1 2 3 4 … 2N-1 2N、K = 4；(c) + Shared Expert Isolation (DeepSeekMoE)，第 1 个专家为绿色 Shared Expert 且由 Input Hidden 直连、不经过 Router，K = 3；右上图例：浅蓝 = Routed Expert，绿色 = Shared Expert" style="width: 100%;">
</div>

> **图片来源**：DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models（arXiv 2401.06066v1）Figure 2，§2 Preliminaries: Mixture-of-Experts for Transformers（图注 md 第 75 行；§3 DeepSeekMoE Architecture 正文复引）。本地副本：`references/papers/deepseek-moe/deepseek-moe.md` 第 74 行引用图（图注在第 75 行）。

建议文件名：`deepseek-moe-fig2-fine-grained-segmentation-shared-expert.jpg`（原文件 `f7720591…jpg`，1246×617，`.jpg` 沿用原扩展名）。

**引用时需要注意（读图结论）**：草稿把它标为「Figure 2（细粒度专家切分 + 共享专家隔离）」——**图号正确**，但描述需要收紧两点：① 它是**三联图**，(a) 是 `Conventional Top-2 Routing`（专家 `1…N`、`K = 2`）作为对照基线，草稿的描述只覆盖了 (b)(c)；② 图注明确写着 `across these three architectures, the number of expert parameters and computational costs remain constant`——**三个架构的专家参数总量与计算量恒定**。第二点恰恰是本文最需要的逻辑：专家数从 `N` 涨到 `2N`（乃至 145B 配置的 `128` 个路由专家）而**计算量不变**，代价全部落在 token 的 dispatch / combine 通信上。这句话必须在正文里点出来，否则读者会以为「专家变多 = 计算变多」，从而误判 All-to-All 为什么值得被掩盖。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/deepseek-moe-fig3-ablation-fine-grained-shared-expert.jpg" alt="DeepSeekMoE Figure 3：消融柱状图。纵轴 Normalized Performance 0.5–1.2，横轴 HellaSwag / PIQA / ARC-easy / ARC-challenge / TriviaQA / NaturalQuestions；四条柱依次为 0 shared + 2/16 routed (GShard)、1 shared + 1/15（共享专家隔离）、1 shared + 3/31（细粒度切分）、1 shared + 7/63（更细切分）；TriviaQA 与 NaturalQuestions 上增益最明显" style="width: 100%;">
</div>

> **图片来源**：DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models（arXiv 2401.06066v1）Figure 3，§4.4 Ablation Studies。本地副本：`references/papers/deepseek-moe/deepseek-moe.md` 第 227 行引用图（图注在第 228 行）。

建议文件名：`deepseek-moe-fig3-ablation-fine-grained-shared-expert.jpg`（原文件 `0c75664b…jpg`，1228×551）。

**引用时需要注意**：这一张是**可选**项。它的作用是回答「为什么要付出 All-to-All 通信的代价」——四条柱对应 `0+2/16` → `1+1/15` → `1+3/31` → `1+7/63`，即从 GShard 逐步走到「1 共享 + 63 路由」；读图可见增益主要出现在 `TriviaQA` 与 `NaturalQuestions`，而 `PIQA` / `ARC-easy` 上四条柱几乎无差别。若 §9 要论证选型，需要照实写成「增益集中在知识密集型任务」，不能概括成「全面更好」。

## 存疑与分歧

1. **本目录没有任何一张图画出 EP 的 device 布局或 All-to-All 通信本身。** 7 张真插图里，Figure 2 只画了「共享专家 + 路由专家的逻辑结构」（Router → 专家 → `⊕` 汇聚），既没有画专家如何被放置到不同 rank，也没有画 dispatch / combine 两条 All-to-All。因此「EP 通信从何而来」这一环，本目录只能提供**模型侧的根据**（专家数 `N → mN`、共享专家隔离），**提供不了通信侧的图示**。若文章 §4 / §7 需要一张「MoE 层 = 两次 All-to-All + 本地专家计算」的图，必须转向同级的 `torch/dualpipe/references/papers/deepseek-v3-report/`（DualPipe 与 EP 的布局图在那里）或仓库里的 `torch/deepep/`，本目录无替代品。

2. **草稿对 `f7720591…jpg` 的图号判断正确（Figure 2），但对内容的概括漏了 (a) 子图与「参数/计算量恒定」这句图注。** 已在「采用图的正文引用块」下写明修正建议。用户任务里提到的「被当作 Figure 2（细粒度专家切分 + 共享专家隔离）」核实无误，只是需要按三联图来引。

3. **Figure 1 是散点图，不是架构图。** `ac61dc9d…jpg`（md 第 19 行）是 Open LLM Leaderboard 上「平均分 vs 激活参数量」的散点图，含一条红色线性拟合虚线与灰色水平虚线。它与流水线调度、与 All-to-All 通信都没有关系，**切勿因为它是 Figure 1 就当作总览架构图放进文章**。

4. **7 张真插图中只有 1 张（Figure 2）与文章主线直接相关，其余 5 张都是模型质量论述。** Figure 1（榜单散点）、Figure 3（消融柱状，可选）、Figure 4（禁用专家的 Pile loss）、Figure 5（激活专家数 vs Pile loss）、Figure 6（半激活专家对比）、Figure 7（18 面板训练曲线）都属于「DeepSeekMoE 比 GShard / dense 更好」的证据链，与「DualPipe 如何掩盖通信」无关。文章引用本目录时建议最多只取 Figure 2（+ 可选 Figure 3），不要铺开。

5. **10 张表格截图里只有 Table 7 与 Table 6 对本文有用。** 其余 8 张（Table 1/2/3/4/5/8/9/10）都是 benchmark 打榜表。Table 7（超参数总览）与 Table 6（145B 配置 `4 + 128` 专家）之所以有用，是因为它们给出了**每层专家数与相对专家大小**——这是估算 All-to-All 通信量和 EP 组大小的唯一模型侧输入。Table 7 对应 md 第 518 行的 HTML 表，Table 6 对应第 369 行，直接转写即可。

6. **三张 Softmax 门控公式截图渲染完全相同，其中两张无法区分。** `a341843d…jpg`(309×62) 与 `93911de6…jpg`(309×60) 的像素内容都是 `s_(i,t) = Softmax_i(u_t^(l^T) e_i^l) ,`（末尾逗号），分别对应 md 第 69 行的公式 (5)（传统 MoE）与第 96 行的公式 (8)（细粒度 MoE）——这两个公式在论文里字面相同，因此**无法从图像判断哪张对应哪一号**，表中已按「(5) / (8) 之一」标注。第三张 `3553478b…jpg`(303×62) 末尾是句号，对应 md 第 118 行的公式 (11)（共享专家隔离后的门控），可以确定。

7. **17 张公式截图与 10 张表格截图都是纯重复产物，不是「MinerU 丢掉又捡回来的插图」。** 公式截图的内容已在 md 第 49、53、61、65、69、88、92、96、110、114、118、132、136、140、148、152、156 行以 `$$…\tag{N}$$` 文本形式存在（编号 (1)–(17) 连续无缺）；表格截图的内容在 md 第 203、215、299、309、333、369、518、526、532、536 行以 HTML `<table>` 形式存在（Table 1–10 与 md 的 `Table N |` 图注逐一对上，其中 8 处图注在表体下一行，2 处——Table 1 与 Table 2——图注在表体上一行）。要引公式或数据，直接从 md 转写。

8. **本目录 34 张里没有页眉 / logo / 装饰 / 图内碎片**，也没有读不清的图。唯一在读取时被降采样的是 `3b942c58…jpg`（原图 1234×1645，harness 降到 692×922），但 18 个面板标题与坐标轴仍全部可读。

9. **`torch/dualpipe/pics/` 目录当前不存在**——`torch/dualpipe/` 下只有 `deep-dive_draft-1.md`、`deep-dive_draft-2.md`、`PROMPT-learn-pipeline.md`、`notes/`、`references/`。粘贴上面的引用块前需要先建 `pics/` 并按建议文件名把原图复制过去；本次任务只产出核对表，未创建或复制任何图片文件。

10. 「建议章节」是按任务下达的 9 节拟定结构给出的，不是从草稿既有引用反推的。两个草稿（`deep-dive_draft-1.md`、`deep-dive_draft-2.md`）里**没有出现 deepseek-moe 的图片哈希**；`PROMPT-learn-pipeline.md` 的待办清单里也没有本目录的条目（它只列了 zero-bubble 的两张，且那两处图号均已在 `IMAGE-SURVEY-zero-bubble.md` 中证伪）。
