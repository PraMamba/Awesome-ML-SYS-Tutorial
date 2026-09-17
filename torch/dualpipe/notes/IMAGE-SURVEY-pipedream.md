# PipeDream（arXiv 1806.03377）图片逐张核对

核对对象：`torch/dualpipe/references/papers/pipedream/images/` 下的全部 22 张 `.jpg`。
配图来源稿：`torch/dualpipe/references/papers/pipedream/pipedream.md`（417 行，MinerU 产物；原始 PDF 在同目录 `references/raw/pipedream/PipeDream.pdf`）。
核对方式：22 张**全部**用 `read_image` 逐张打开读图，「内容」列全部为看图后的事实描述（含图中可读文字/数值），再与 md 引用行、图注、`Table N:` 文本互相印证；分歧记录在文末。
本文定位：PipeDream 在 `torch/dualpipe/deep-dive.md` 里只是「异步流水线 / weight stashing / 参数版本」谱系的一句话级旁证，不是主线，因此下述结论以「不采用」为主，每张都给出具体理由。

## 汇总

- `images/` 实际文件数（`ls | wc -l`）：**22**
- 本清单覆盖行数：**22**（逐张一一对应，无跳过、无合并）
- 真插图：**15**；公式截图：**6**；表格截图：**1**；页眉/装饰/碎片：**0**
- 建议采用：**0** 张
- 交叉核对：md 中 `images/` 引用共 **15** 处（`grep -c 'images/' pipedream.md`），恰好等于真插图数；其余 7 张（6 公式 + 1 表格）在 md 里被 MinerU 以 LaTeX / HTML 形式复现，故未被 `<img>` 引用。`references/README.md` 第 174 行记录的「pipedream 15 处引用」与本目录核对一致。
- 读图数值说明：表中柱高、曲线终点等带「≈」的数字均为**读图估计**（±1～2 单位），不是论文原文数字；要写进正文的数字请以论文正文与 Table 1 为准。

## 逐张清单

| 文件名 | md 引用行 | 图号/表号 | 类别 | 内容（读图后的事实描述，含图中可读的关键标注/数值） | 结论 | 建议章节 |
| --- | --- | --- | --- | --- | --- | --- |
| `0735616b9803acb52b741960c8cb54facc6d91e61e80dd6cbec5c3d0ab01a34d.jpg` | 41 | Figure 1 | 真插图（柱状图） | 纵轴 `Communication overhead (% of total time)` 0–100；横轴 5 组模型 `VGG16 / ResNet50 / Inception v3 / AlexNet / S2VT`，每组 3 档机器数 `4 / 8 / 16`；图例 `K80`（白柱）/`Titan X`（绿柱）/`V100`（斜纹柱）。最高柱：VGG16-16 与 AlexNet-8 的 V100 均 ≈90%，S2VT-16 的 V100 ≈93%；ResNet50 最低（4 台机器时个位数百分比） | **不采用**：讲的是数据并行通信占比这一「动机」（论文 §2），而本文第 1 章的两种等待是「流水线气泡」与「stage 内通信等待」，不需要 DP 通信占比 | — |
| `c265a57d267d9c5ba953f412e6c28a27e7ff6d849f669d8973c58784897bdf5a.jpg` | 50 | Figure 2 | 真插图（拓扑 + 时间轴） | 上半：4 台机器 `Machine 1`–`Machine 4`，每台一个完整模型副本（彩色圆点层图），机器间双向粗箭头互连。下半虚线框展开 Machine 4 的时间轴：`F_n`（斜纹 = `Forward Work`）、`B_n`（点纹 = `Backward Work`）、`C_n`（黑块 = `Communication Stall (Model Parameters)`），三段 `F/B/C` 循环推进；图例三色齐全 | **不采用**：BSP 数据并行的通信 stall 时间轴，讨论的是 DP 的等待；本文第 1 章的等待要用 PP 时间轴说明，GPipe 气泡基线已另采 GPipe 论文原图 | — |
| `1a4f06e7524c8ab67b2fb6b384df81c82f2cf9459b3696ed4a9733f56abeeef8.jpg` | 65 | Figure 3 | 真插图（调度矩阵） | 4 行 `Machines 1`–`4`，横轴 `Time`（右向箭头）；格内数字是 minibatch ID（`1`、`2` 两轮）；图例 `Forward Work`（蓝）/`Backward Work`（绿）/`Idle`（斜纹）。每行只有一个蓝格和一个绿格，其余全部是斜纹 `Idle`；对角线错位排布（3 号机器在第 3 列有 `1`，4 号机器在第 3、4 列有 `1 1`） | **不采用**：这是「纯模型并行 = 任一时刻最多一个 stage 在算」的原始基线图。气泡的成因（F/B 固定依赖）已由 GPipe Figure 2 与 AIInfra 教材调度图承担，重复放会增加版面噪声 | —（若 §1 想给「早期 PP 的空闲」一个出处可备选） |
| `0b02a4358cb4e96418075ecfbf16e34b1249d3672d1ece6c7dd8aa59952a1edb.jpg` | 78 | Figure 4 | 真插图（拓扑 + 时间轴） | 上半：`Mach. 1`–`Mach. 4` 各持一段连续层（彩色圆点串联），Mach. 1 标 `Input stage`、Mach. 4 标 `Output stage`，Mach. 3→4 之间额外画交叉连接（表示 stage 复制）。下半虚线框展开 `Mach. 1` 时间轴：`F_n`（斜纹）、`B_{n-x}`（点纹）、`F_{n+1}`、`B_{n-x+1}`；其下方一排黑块 `C_{n-x-1}`、`C_n`、`C_{n-x}`、`C_{n+1}`，图例 `Background Communication (Activations & Gradients)` | **不采用**（备选）：这张是「通信藏在计算背后」的关键图，与本文第 1 章「stage 内通信等待」话题相通；但图中刻意让 F 与 B 错位（`F_{n+1}` 与 `B_{n-x}` 交叉）正是异步谱系的卖点，本文不展开异步正确性 | §1（备选） |
| `da779d738fd044df72a3b262227192bfee89c1e58ac563941375aa7b07b227b1.jpg` | 89 | Figure 5 | 真插图（柱状图） | 纵轴 `Output size (millions of floats)` 0–150，横轴 `Layer`；VGG16（minibatch 32、ImageNet1K）各层输出大小：前两层 ≈103M，随后 ≈50M → ≈25M → ≈7M 递减到接近 0；一条黑色虚线横贯全图并标注 `Size of parameters`（≈140M 处） | **不采用**：论证「PP 只传一层激活，比 DP 传全部参数省 90%+」的动机图；本文主线是气泡与等待，不需要通信量对比 | — |
| `a6e7edc8d199ca1e46ba2feca8924bc91a09c03ae74e5b4faec9b7183675f5f9.jpg` | 110 | Figure 6 | 真插图（结构示意） | 标题 `Pipelined model parallelism`；四个方框 `Stage 1`–`Stage 4` 依次用双向箭头串接；各 stage 内部是若干竖直小框（框内彩色圆点 = 层）：Stage 1 = 2 个小框、Stage 2 = 3 个小框（下方用大括号标 `Data parallelism`）、Stage 3 = 2 个小框、Stage 4 = 1 个小框 | **不采用**：说明「PP 可在 stage 内叠加数据并行、层如何分组」，属 PipeDream 自动划分（§3.2）的卖点，与本文主线无关 | — |
| `95c289ae693770047d73dd5ecccb379ef19fb8cdeab030fdcb80e1f384773069.jpg` | 113 | Figure 7 | 真插图（流程图） | 左：`Input DNN` →（箭头上写 `profiled by`）→ `PipeDream Profiler` → 右上框 `CDF Profiles of Layer Compute Times & Output Sizes`（框内画 `CDF` vs `layer id` 的阶梯曲线）；再向下 → `PipeDream Optimizer`（另一路输入 `# of machines`）→ 右下另一张 CDF 图 → 左下方框 `PipeDream's distributed runtime executes generated plan`，指向 4 段彩虹色 stage 的模型分块图 | **不采用**：讲 profiling → 规划器 → 运行时这条自动划分流水线，与本文「气泡 / 等待 / 显存口径」主线无关 | — |
| `6d16caf690e85f7906976dcb986c864efc7b1ea66cdfc77d45bbb63eed5bd442.jpg` | 166 | Figure 8 | 真插图（调度矩阵） | 4 行 `Machine 1`–`4`；图例 `Forward Work`（蓝）/`Backward Work`（绿）/`Idle`（斜纹）；底部横轴箭头标 `Time`，并用竖线把矩阵切成 `Startup State` 与 `Steady State` 两段。稳态段每行呈 F/B 交替，Machine 1 为 `1 5 2 6 3 7 4 8 5`（蓝绿相间）；Machine 2 起始列右移 1 格、Machine 3 右移 2 格、Machine 4 右移 3 格且先出现 `1 1 2 2 3 3 4 4` 的连续 F/F、B/B | **不采用**（备选）：这张就是 1F1B 的原始出处（§3.3），且是「同一 minibatch 跨 stage 用了不同权重版本」的例子。但本文第 3 章的 1F1B 时间轴已采用 AIInfra 教材图与 Zero Bubble Figure 2，画面结构几乎相同，再放会重复；正文只需在谱系句里点名 PipeDream | §3（备选：讲「1F1B 与 weight stashing 同源」时） |
| `ac9627ec1a0306f46e68f8b0b76beb49f75fe3177cd1513487ac4bc88b481920.jpg` | 221 | Figure 9 | 真插图（软件结构框图） | 外框标注「与 Caffe 集成」：左上 `F/w Network Receive` → `F/w Copy to GPU`；右上 `F/w Network Send` ← `F/w Copy to CPU`；左下 `B/w Network Receive`、`B/w Copy to GPU`；右下 `B/w Network Send`、`B/w Copy to CPU`；中心黄框 `Intermediate Data Manager`（四角箭头汇聚）；其下 `Caffe Compute Thread`，向上标 `Read Intermediate data`、`Get GPU memory for output`，向下标 `Read Model Parameters`、`Update Model`；最下方绿色框组 `Client Side Cache & Parameter Versioning` 与 `Sharded Parameter Server` | **不采用**：这是「参数版本管理（Client Side Cache & Parameter Versioning）+ 参数服务器」的最早工程形态，但画面主体是 Caffe 时代的数据搬运/缓冲池结构。本文只在谱系句里提「参数版本」一词，放这张对 DualPipe 读者没有信息增量 | —（若 §6 要讲「多版本参数驻留」的最早工程形态，可备选） |
| `da8aee1432c57156e41dfb06f23edddeda3f4e1e3f46656609d3dc2c9c2be22b.jpg` | 272 | Figure 10(a) | 真插图（折线图） | 纵轴 `Top-1 Accuracy` 0–1，横轴 `Time (in hours)` 0–220+；图例 `1 Worker`（蓝圆点）/`8 BSP`（黑三角）/`8 PipeDream`（绿方块）。8 PipeDream ≈40 h 到 ≈0.72 后停止；8 BSP ≈80 h 到 ≈0.70 后停止；1 Worker 220 h 到 ≈0.70 | **不采用**：time-to-accuracy 曲线（VGG16 / Cluster-A），论证 PipeDream 比 BSP 快 2.99×；与本文气泡/显存主线无关 | — |
| `58a6b6b551d4ced9ebc8a74416848961d9794d0abfe76ebc5f75e60c927da1d3.jpg` | 275 | Figure 10(b) | 真插图（折线图） | 同版式：纵轴 `Top-1 Accuracy` 0–1，横轴 `Time (in hours)` 0–50；**只有两条曲线**：`1 Worker`（≈48 h 到 ≈0.57）与 `8 PipeDream`（≈32 h 到 ≈0.65），没有 `8 BSP` 曲线——与正文一致（Inception-v3 在 Cluster-A 上规划器选出的最优配置就是纯数据并行，两条曲线重合故只画一条） | **不采用**：同上，属 time-to-accuracy 评估图 | — |
| `33c1b554c4d0295528254c84f21153ca50e6061e1e3b21e6bb659bf9cee0248b.jpg` | 287 | Figure 11(a) | 真插图（折线图） | 纵轴 `Top-1 Accuracy` 0–1，横轴 `Time (in hours)` 0–100；三条曲线：`1 Worker`（≈95 h 到 ≈0.68）、`8 BSP`（≈75 h 到 ≈0.65）、`8 PipeDream`（≈15 h 到 ≈0.68）。与正文「VGG16 从 Cluster-A 的 220 h 降到 Cluster-B 的略少于 100 h」对应 | **不采用**：同上（Cluster-B 的 V100/10Gbps 版本），评估图 | — |
| `8e15e4ad7a774df01c47f87bbcac539b77bfdae6abeb5fabf9e38a0308eefa62.jpg` | 290 | Figure 11(b) | 真插图（折线图） | 纵轴 `Top-1 Accuracy`，横轴 `Time (in hours)` 0–40；`8 BSP`（黑三角）与 `8 PipeDream`（绿方块）两条曲线几乎完全重合（≈28 h 到 ≈0.65），`1 Worker`（蓝）≈27 h 到 ≈0.57。对应正文「即使 Inception-v3，PipeDream 也比 BSP 快 45%」 | **不采用**：同上，评估图 | — |
| `c1cc88be12f813743227b08d68e47876b8c13ab3b2bf05feecd6d3ffd17ace66.jpg` | 294 | Figure 12 | 真插图（折线图） | 纵轴 `Top-1 Accuracy` 0–1，横轴 `Time (in hours)` 0–220+；图例六项齐全：`1 Worker`（蓝圆）、`4 ASP`（黄圆）、`4 BSP`（黑三角）、`4 PipeDream`（绿方）、`16 BSP`（红菱）、`16 PipeDream`（橙十字）。16 PipeDream 最快（≈15 h 到 ≈0.72）；4 ASP 最慢且终点只到 ≈0.50（正文：PipeDream 比 4 机 ASP 快 7.4× 达到 48% 准确率） | **不采用**：这张是「ASP 统计效率差 + 机器数扩展」的综合评估图；本文不讨论 ASP 的收敛性，画面对 DualPipe 读者只会增加负担 | — |
| `6ea90a90b548395d03dcb1b48bd4fc2d74b7144826b6f647bd8a2e582ac65816.jpg` | 307 | Figure 13 | 真插图（柱状图） | 纵轴 `Speedup over 1 machine` 0–8；横轴 6 组柱：`4 Model Parallel`（≈1.0）、`4 Straight Pipeline`（≈2.6）、`4 PipeDream`（≈3.1）、`8 Model Parallel`（≈0.7）、`8 Straight Pipeline`（≈3.5）、`8 PipeDream`（≈7.0） | **不采用**：论证「纯 MP 无收益、pipelining 才带来加速」；这个结论本文已用气泡口径在第 1/9 章说明，无需再引 2018 年的加速比柱状图 | — |
| `5f7437043c5defb582406afe2e1103280a1ab481060adaa2fd6792a03c4de7d4.jpg` | 未引用（md 第 130–132 行为同一式的 LaTeX） | 公式（§3.2 划分算法中单 stage 耗时） | 公式截图 | `T(i → j, m) = (1/m) · max( Σ_{l=i}^{j} T_l , Σ_{l=i}^{j} W_l^m )` | **不采用**：公式截图，非插图 | — |
| `8afa6f31cbec7435d74e03fb43dabd41fd546607ac04a3088e084542ec37a164.jpg` | 未引用（md 第 140–142 行） | 公式（`A(j,m)` Case 1：单 stage 复制 m 次） | 公式截图 | `A(j, m) = T(1 → j, m)` | **不采用**：公式截图，非插图 | — |
| `6915aa255519d21c85a088c852ddb214c0109ee652b45758dd13cf11cb46e4e8.jpg` | 未引用（md 第 146–148 行） | 公式（`A(j,m)` Case 2：DP 递推） | 公式截图 | `A(j, m) = min_{1≤i<j} min_{1≤m'<m} max{ A(i, m−m'),  2·C_i,  T(i+1 → j, m') }`（大括号三行） | **不采用**：公式截图，非插图 | — |
| `c8802aabd47bc9d85d612cb7b0ca2f7dfe86fd1c9ff79ddb5de2477df31c8514.jpg` | 未引用（md 第 199–201 行） | 公式（weight stashing 的更新式） | 公式截图 | `w^{(t+1)} = w^{(t)} − ν · ∇f( w_1^{(t−n+1)}, w_2^{(t−n+2)}, …, w_n^{(t)} )`——各 stage 的滞后量递变（stage n 用最新权重） | **不采用**：公式截图，非插图。**注**：这正是本文讲「参数版本滞后」谱系时要引的式子；请用 LaTeX 在正文重写，而不要引截图 | §3/§6（用 LaTeX 复现） |
| `a4f4ea17c261a44f9c61e4975cdc5bc403c75a8710f5c03c515357a3acae12dd.jpg` | 未引用（md 第 207–209 行） | 公式（vertical sync 的更新式） | 公式截图 | `w^{(t+1)} = w^{(t)} − ν · ∇f( w_1^{(t−n+1)}, w_2^{(t−n+1)}, …, w_n^{(t−n+1)} )`——所有 stage 统一滞后 n−1 步，论文指出与 n 机 BSP 语义等价 | **不采用**：公式截图，非插图。与上一条构成「逐 stage 递变 vs 全局统一滞后」的对照，正文可用 LaTeX 并列两式 | §3（用 LaTeX 复现） |
| `55826f581f91174a75d832ff2a032c0c82d70029664f58cb050280c524b7aaab.jpg` | 未引用（md 第 193–195 行） | 公式（vanilla minibatch SGD） | 公式截图 | `w^{(t+1)} = w^{(t)} − ν · ∇f( w_1^{(t)}, w_2^{(t)}, …, w_n^{(t)} )` | **不采用**：公式截图，非插图 | — |
| `9d37313fe0e531e415ba03c84cce1835cb1274b9ec2936b55b89deab8a8ff513.jpg` | 未引用（md 第 268 行为同内容的 HTML 表） | Table 1 | 表格截图 | 7 列表头 `DNN Model / # Machines (Cluster) / BSP speedup over 1 machine / PipeDream Config / PipeDream speedup over 1 machine / PipeDream speedup over BSP / PipeDream communication reduction over BSP`。逐行数值：VGG16 4(A) `1.47× / 2-1-1 / 3.14× / 2.13× / 90%`；8(A) `2.35× / 7-1 / 7.04× / 2.99× / 95%`；16(A) `3.28× / 9-5-1-1 / 9.86× / 3.00× / 91%`；8(B) `1.36× / 7-1 / 6.98× / 5.12× / 95%`；Inception-v3 8(A) `7.66× / 8 / 7.66× / 1.00× / 0%`；8(B) `4.74× / 7-1 / 6.88× / 1.45× / 47%`；S2VT 4(A) `1.10× / 2-1-1 / 3.34× / 3.01× / 95%`。与 md 第 268 行 HTML 表**逐个数字一致** | **不采用**：表格截图。内容是 PipeDream 相对 BSP 的加速比与通信削减，与本文主线无关，也不需要转写 | — |

## 采用图的正文引用块（可直接粘贴）

**本目录无建议采用图。**

整体理由：`torch/dualpipe/deep-dive.md` 只在讲「异步流水线与 weight stashing / 参数版本」的谱系时点名 PipeDream（1～2 句），本目录 15 张真插图分三类，没有一类真正服务于该文主线：

1. **动机与评估类**（Figure 1、5、10、11、12、13）：论证「DP 通信占比高」「PP 传得更少」「time-to-accuracy 快 5×」。这些是 2018 年 PipeDream 对 DP 的对比结论，本文既不复现这些数字，也不用它们支撑「两种等待」。
2. **自动划分与系统结构类**（Figure 6、7、9）：讲 profiling → 规划器 → 运行时的工程实现，与气泡/显存公式无关。
3. **调度与等待类**（Figure 2、3、4、8）：其中只有 Figure 8 与本文第 3 章（1F1B 起源）直接相关，但 **AIInfra 教材 `10pipeline02.png`（1F1B 标准时间轴）已被第 3 章采用**，两张画面结构几乎相同（4 行 device、蓝 F / 绿 B、斜纹 idle、微妙错位），并列会造成读者混淆；其余三张分别讲 DP 通信 stall（Figure 2）、纯 MP 空闲（Figure 3）、异步下的通信隐藏与权重错位（Figure 4），都不是本文要立的正面图。

因此本目录结论是「不采用」，但如果第 3 章最终决定给「1F1B 起源」配一张历史图，**优先候选是 Figure 8**（`6d16caf690e85f7906976dcb986c864efc7b1ea66cdfc77d45bbb63eed5bd442.jpg`，第 166 行，§3.3 `Work Scheduling`），并须与 AIInfra `10pipeline02` 二选一。

## 存疑与分歧

1. **7 张未被 md 引用的图不是「漏图」，而是 MinerU 的双重产物。** 6 张公式截图与 1 张表格截图的内容，在 md 里分别以 `$$…$$`（第 130–132、140–142、146–148、193–195、199–201、207–209 行）和 HTML `<table>`（第 268 行）复现，数字与截图逐项一致（Table 1 已逐个核对）。所以 `images/` 22 张 > md 引用 15 处是正常的，不表示文档缺图。
2. **MinerU v2 `content_list` 把 Figure 6 的图注挂错了图。** `references/raw/pipedream/2e03df6f-…_content_list_v2.json` 中第 5 页把 `Figure 6: Pipeline Parallel training in PipeDream combines pipelining, model- and data-parallel training.` 挂在 `95c289ae…`（阶梯流程图）上，而 `a6e7edc8…`（Stage 1–4 结构图）没有图注。以**读图内容**为准：`a6e7edc8…` 画的是 stage 分组 + `Data parallelism` 大括号，正是 Figure 6；`95c289ae…` 画的是 profiler/optimizer 流程，正是 Figure 7。md 第 110/113 行的放置与读图一致。另用 PyMuPDF 取 PDF 第 6 页坐标复核：`Figure 6:` 图注在 y=224–244、`Figure 7:` 在 y=416–469，两张图的纵向位置与「图注在所注图之下」相符。
3. **Figure 10/11 的 (a)/(b) 归属靠外部证据判定，图内没有 (a)/(b) 标注。** 我读到的是：`da8aee14…` 横轴到 220 h（对应 VGG16 在 Cluster-A 的 220 小时）、`58a6b6b5…` 横轴到 50 h（Inception-v3）、`33c1b554…` 到 100 h（Cluster-B 的 VGG16「略少于 100 小时」）、`8e15e4ad…` 到 40 h。这与 md 引用行顺序、论文正文数字三者自洽，故按 md 归属登记；如果只按文件名/顺序猜，很容易把 Cluster-A 与 Cluster-B 的 (a) 混成一对。
4. **Figure 10(b) 只有两条曲线是真实情况，不是截图丢失。** `58a6b6b5…` 里确实没有 `8 BSP` 曲线；论文 §5.2 说明 Inception-v3 在 Cluster-A 上 PipeDream 规划器选出的最优配置就是纯数据并行，因此 PipeDream 与 BSP 曲线重合、只画一条。转录时不要「补」第三条曲线。
5. **weight stashing 与 vertical sync 两式只差下标，容易被 OCR/转写混淆。** `c8802aab…`（stashing）为 `(t−n+1), (t−n+2), …, (t)`，`a4f4ea17…`（vertical sync）为全 `(t−n+1)`。md 第 197–211 行文字与两图各自一致；若要引这两个式子，建议直接在正文写 LaTeX（`g^{(t)}` 口径统一后再写），不要引截图。
6. **本目录没有任何页眉/logo/装饰图，也没有「图内碎片」。** 22 张全部是可判定的内容图（15 插图 / 6 公式 / 1 表格），不存在需要归入「页眉、装饰、碎片」的残余项。
