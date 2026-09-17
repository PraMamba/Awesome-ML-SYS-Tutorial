# Zero Bubble Pipeline Parallelism（arXiv 2401.10241v1）图片逐张核对

资料定位（对 `torch/dualpipe/deep-dive.md` 而言）：**全文数学推导与调度演进的第一依据**。文章用它的

- Figure 1（MLP 计算图）建立 F / B / W 三记号与「反向可拆两份」的数学来源；
- Figure 2 / Figure 3（1F1B 与手工 ZB-H1 / ZB-H2）支撑「I 优先、W 可延迟是调度策略而非数学必然」；
- Figure 8（ZB-V）作为 DualPipeV cut-in-half / V 形布局的直系前序；
- Table 1 / Table 2 / Table 5 / Table 8 / Table 9 作为 §6「气泡 / 显存公式重算与比较口径统一」的数值口径。

- 本地副本：`torch/dualpipe/references/papers/zero-bubble/zero-bubble.md`（MinerU 产物，350 行）
- 图片目录：`torch/dualpipe/references/papers/zero-bubble/images/`

## 汇总

- `images/` 实际文件数（`ls torch/dualpipe/references/papers/zero-bubble/images/ | wc -l`）：**31**
- 本清单覆盖行数：**31**（逐张清单 31 行，与 `ls | wc -l` 一致）
- 真插图：**12**（Figure 1、2、3、4、5、6 上栏、6 下栏、7、8、9、10、11）；公式截图：**7**；表格截图：**12**；页眉/装饰/碎片：**0**
- 建议采用：**12 张**（11 张核心，另 1 张列为可选：Figure 11 与 Figure 6 上栏高度冗余）
- md 中带图引用的行：第 23、38、41、94、119、144、146、149、158、191、248、265 行（共 12 处，对应 12 张真插图）；**另外 19 张图在 md 中没有任何 `![]()` 引用**——其中 12 张是 Table 1–12 的截图（内容已在 md 中以 HTML `<table>` 另行转录，属重复产物），7 张是 ILP 与 memory-limit 章节的行间公式截图（内容已在 md 中以 `$$…\tag{N}$$` 文本形式存在）


> **〔第 3 轮订正〕** 本文件原先在下栏（ZB-H2）的 warm-up F 数上记的是 `7/6/5/1`。第 3 轮审核用像素颜色分类复核（格宽 24.5 px、首条竖线 x=72，取每格边缘附近的背景像素以避开数字字形），实测为 **`7/5/3/1`**，且只有 7/5/3/1 结构自洽：H2 每行的前导 F 数等于 H1 同一行的「F 数 + 白格数」（4+3=7、3+2=5、2+1=3、1+0=1）。文件内相关处已改为 7/5/3/1，逐张核对记录见 `IMAGE-VERIFY.md` 与 `REVIEW-deep-r3b.md`。
>
> 另外一处需要连带说明：本文件把 Figure 3 **上栏**（ZB-H1）warm-up 之后的 3/2/1/0 个白格写成「由行末的 W 填掉一部分」，第 3 轮复核认为这些白格**一直保留**、合计构成 ZB-H1 剩余的 `(p-1)t` 气泡；论文说被 W 填掉的是**尾部**气泡。涉及该说法的句子请以 `IMAGE-VERIFY.md` 的结论为准。


## 逐张清单

（按文件名字典序排列，便于与 `ls` 输出逐行对照）

| 文件名 | md 引用行 | 图号/表号 | 类别 | 内容（读图后的事实描述，含图中可读的关键标注/数值） | 结论 | 建议章节 |
| --- | --- | --- | --- | --- | --- | --- |
| `0de5157ff2ed77a68d0f98de960c3b06cf3d522e1d56c22fb265a7f6c6760630.jpg` | —（md 未引用） | 公式 (2) | 公式截图 | 单行公式，可读为 `β ≥ p(T_F + T_B) + 2(p−1)T_comm − kT_F − T_B = (p−1)(T_B + 2T_comm) + (p−k)T_F`（md 第 260 行有同一公式的 LaTeX） | 不采用（公式截图，非插图） | — |
| `0f7b9870ad961c2f948d97b8d9e28316c005677fca0c7fa8704578ecdb8b5621.jpg` | —（md 未引用） | 公式 (6) | 公式截图 | 单行公式，可读为 `E_(i,j,c) ≥ E_(i,j′,c′) + T_(i,j,c) − O_(i,j,c)→(i,j′,c′) ∞`（ILP 的「调度顺序」约束；md 第 333 行） | 不采用（公式截图，非插图） | — |
| `175564bd0fc0e300633c103d7404130c8dfea01ba17d4e6cff0df6eda1cc6b2f.jpg` | —（md 未引用） | 公式 (4) | 公式截图 | 单行公式，可读为 `s.t. E_(i,j,F) ≥ E_(i−1,j,F) + T_comm + T_(i,j,F)`（F 在相邻 stage 的依赖；md 第 325 行） | 不采用（公式截图，非插图） | — |
| `1cfd9a39e33e68f4174b3a672afbaaf75fdde78c7d720a6203a2014515c2c767.jpg` | 94（图注 95） | Figure 4 | 真插图 | 4 行（4 个 stage）× 约 12 列的时间网格。图例：浅蓝 `Reduce local values by propagating from 1 to 4`、米色 `Optimizer step`、橙色 `5 - 8 Propagate globally reduced value to each stage`、深红 `Rollback if validation fails`。可读标注：绿色块（F）在对角线阶梯上递减，米色 optimizer 块在中间呈错位的阶梯，橙色块 `1 2 3 4` 沿对角线向下传播、`5 6 7 8` 在上方回传，红色块紧贴橙色块标注 rollback 位置 | **采用** | **§3**（ZB-H2 能成立的前提：优化器同步必须被绕过，属 `§4`）；**§9**（误区：「零气泡调度可以直接套在普通 1F1B 框架的优化器同步上」） |
| `2483da0e0dd10036e50c45696ac9f3ea765f2dedf8c65e2f6e71c29ff305117d.jpg` | —（md 未引用） | Table 4 | 表格截图 | 12 列吞吐/显存明细表。可读数值：1.5B/8GPU/24MB → ZB-2p `14.5`、ZB-1p `12.9`、1F1B `11.8`、1F1B-I `13.1`（samples per GPU per second）；28.3B/32GPU/256MB → ZB-2p `1.00`、1F1B `0.88`；Memory(GB) 行 ZB-2p 59/70/51/74 vs 1F1B 30/39/32/43 | 可转写为 Markdown 表格，不直接引图（md 第 124 行已有同内容 HTML 表） | §6（吞吐与显存同时给出的唯一明细口径，可作「同一显存下不可比」的原始依据） |
| `269fe44bfe7fe26f278d630cf5dd59e2840b17163636a7a48110ded58c832360.jpg` | 265（图注 266） | Figure 11 | 真插图 | 8 行（`Device 0`–`Device 7`）× 约 40 列时间网格，图例只有 `F`（蓝）/`B`（青）/`W`（绿）三色，**没有 Optimizer step 图例**。左端 F（蓝）呈下降阶梯（Device 0 最长、Device 7 最短），右半部分由大块绿色 `W` 构成一个上宽下窄的三角形，行内看不到明显白色空隙 | **采用（可选）** | §6（附录 B 的 1.5B / 32 microbatches 零气泡实例；与 Figure 6 上栏展示的是同一现象的不同规模，若篇幅紧张可只留 Figure 6） |
| `27cf2bf2076c055377423d6efdb9874c339fac60529b69ea561bd0153a65afbd.jpg` | 119（图注 120） | Figure 5 | 真插图 | 2×2 四联柱状图：`1.5B Model, 8GPUs`（纵轴 11–17 samples per second per gpu，横轴 24/32/64 microbatches）、`6.2B Model, 8GPUs`（3.25–5.00）、`14.6B Model, 16GPUs`（1.4–2.0）、`28.3B Model, 32GPUs`（0.7–1.1）。每面板图例固定为 `1F1B`（蓝）/`1F1B-I`（橙）/`ZB-1p`（绿）/`ZB-2p`（红），另有一个**空心黑框 `upper bound`** 只出现在 1F1B 柱上方；ZB-2p（红）在每个面板都最高且贴近 upper bound | **采用** | §6（「零气泡的吞吐上界」与「随 micro-batch 数增长的收敛趋势」的唯一原始图） |
| `2a378d26a2d9930745e8cb6a19ba615f176839e3a2428ed2ec56b880e7d9b00c.jpg` | —（md 未引用） | Table 6 | 表格截图 | 列头 `Model / #GPU / b / m`，三组（6.2B/16/6、14.6B/24/2、28.3B/32/2）。可读数值：ZB-V `4.15 / 4.21 / 4.35`（6.2B），ZB-2p\* `4.36 / 4.37 / 4.45`，ZB-1p `3.87 / 4.00 / 4.29`，1F1B `3.38 / 3.57 / 3.91`；Memory(GB) 行 ZB-V 64/64/64 vs 1F1B 61/61/61 | 可转写为 Markdown 表格，不直接引图（md 第 177 行已有同内容 HTML 表） | §5 / §6（ZB-V 与「同显存」的 ZB-2p\* 对比，是 DualPipeV 选型的直接前序证据） |
| `2d9fe9b1db1c7aa757185f0e073dcff130a8f728a5ffb5e46e225bba7465ab62.jpg` | —（md 未引用） | Table 7 | 表格截图 | 列头 `Model / #GPU / m / b / Δ`。可读数值：6.2B/16/m=64 → ZB-V `4.13`→`4.21`（Δ `1.94%`）、ZB-1p `3.91`→`4.00`（`2.30%`）、1F1B `3.48`→`3.57`（`2.59%`）；14.6B/24/m=96 → `1.75`→`1.88`（`7.43%`）等 | 可转写为 Markdown 表格，不直接引图（md 第 181 行已有同内容 HTML 表） | §6（micro-batch 大小与气泡的 trade-off 量化，可选） |
| `3955441ba7fa9c5a066548e680cb97da181f5cb01b31e187b673517bfa0053d2.jpg` | 149（图注 150） | **Figure 7**（**不是 Figure 8**） | 真插图 | 1×4 四联折线图，面板标题 `1.5B Model, p = 8` / `6.2B Model, p = 8` / `14.6B Model, p = 16` / `28.3B Model, p = 32`；纵轴 `Bubble rate`（0.00–0.175），横轴 `M_limit`，刻度为 `1.0pM_B / 2.0pM_B / 3.0pM_B`。每面板三条曲线（图例 `#Microbatches = 24 / 32 / 64` 等，按面板变化）。可读趋势：曲线从 `1.0pM_B` 处约 0.10–0.16 下降，在 `2.0pM_B` 处全部压到 0.00 附近；`2.0pM_B` 之后曲线**变为水平**（不再改善） | **采用** | §6（「2pM_B 是经验阈值」的直接图形证据；也是把「气泡率」与「显存上限」两个口径绑定的唯一图） |
| `450daae1fc3aed64a343d89a1caf2db2a0f4691a4b23f5e784f26fa79454c6a2.jpg` | 248（图注 249） | Figure 10 | 真插图 | 上下两条横排时间轴。上排标 `(a) The schedule grouped by W`：先是一段 16 个同色块（黄/蓝/红/绿各 4 个、块内数字 `1 1 1 1 2 2 2 2 …`），右侧接 4 个 `AR`（All-Reduce）块（蓝黄绿红），**计算段与 AR 段在时间上不重叠**。下排标 `(b) The schedule grouped by parameter`：块内数字变成 `1 2 3 4 / 1 2 3 4 / …`，4 个 `AR` 块错位插入对应的计算块下方，**每个 AR 与相邻计算块并行** | **采用** | **§1**（「两种等待」中的第二类等待：stage 内的通信等待，与流水线气泡是两件事）；**§8**（DP all-reduce 与 W 的重排顺序，是「计算-通信重叠」在本文里最早的可引用出处） |
| `48cf37b82db9d0f54f5408d4a1f21d286a1f7f8880d5f809c63559e5989c0e41.jpg` | —（md 未引用） | Table 12 | 表格截图 | 长表，列头 `Model / p / m / b / Samples per GPU per second / Memory(GB) / Schedule`，四组模型。可读数值：1.5B/8/m=2 → 1F1B `3.56`（11GB）vs ZB-2p `4.25`（12GB）；1.5B/8/m=8 → `8.26` vs `9.90`；14.6B/16/m=16 → `0.95` vs `1.25` | 可转写为 Markdown 表格，不直接引图；但内容（`m ≤ p` 的极端小 batch 场景）与本文主线无关，建议也不转写 | — |
| `4bce0021333fdfdcf46b51abfe81a58b2ada751438e9f24cc5d4acdd2551b19c.jpg` | 191（图注 192） | Figure 9 | 真插图 | 3×3 九联折线图，列标题 `6.2B, p = 16, m = 48 / 64 / 128`、`14.6B, p = 24, m = 72 / 96 / 192`、`28.3B, p = 32, m = 96 / 128 / 256`；纵轴 `Bubble rate`（0.00–0.15+），横轴刻度 `1.0pM_B / 2.0pM_B / 3.0pM_B`。每面板两条曲线：`ZB`（蓝）与 `ZB-V`（橙）。可读趋势：ZB-V 曲线整体位于 ZB 下方，并在**明显小于 2.0pM_B** 处就压到 0.00 | **采用** | **§5**（ZB-V 在同等显存下更早达到零气泡，是 DualPipeV「cut-in-half 换显存」的直接前序证据）；§6 |
| `536478895e0abbcea17d6178866439923d5a02d53f94cfd02f19ea5706b189e0.jpg` | 158（图注 159） | **Figure 8**（ZB-V，**不是 `3955441b…`**） | 真插图 | 4 行（`Device 1`–`Device 4`）× 约 52 列时间网格，图例 `F`（蓝）/`B`（青）/`W`（绿）/`Optimizer step`（米色）。每个 device 的行内数字用**两种颜色**表示两个 chunk（图注：白色文字 = 第一个 chunk，黑色文字 = 第二个 chunk）——例如 `Device 1` 行起始 `1 2 3 4 5 6 7` 为白字，随后 `1 1 5 5 5 2 2 6 6 6 3 3 7 7` 为黑字。左端 F 呈下降阶梯（`Device 1` 从第 1 列起、`Device 2` 空 1 列、`Device 3` 空 2 列、`Device 4` 空 3 列）；稳态区是密集的 1F-1B-1W 交替、**看不到成片空白**；右端 4 个米色 optimizer 块呈向右下方错位的阶梯（不在同一列），随后立刻接下一 iteration 的 `1 2 3 4 5 6 7` | **采用** | **§5**（DualPipeV cut-in-half 与 V 形布局的直系前序，主用） |
| `56ccdcb4dc1389e13b91b26a2aad2ac330a6554bbd215840c906903971022f9a.jpg` | —（md 未引用） | Table 11 | 表格截图 | 列头 `Model / p / m / b / Samples per GPU per second / Memory(GB) / Schedule`。可读数值：1.5B/8/m=24 三行 `b=12` → 1F1B `12.0`（57GB）、`b=12` → ZB-1p `13.0`（61GB）、`b=6` → ZB-2p `14.5`（59GB）；6.2B/8/m=24 → `3.56 / 3.95 / 4.32` | 可转写为 Markdown 表格，不直接引图；内容与 Table 4 高度重合，优先级低 | — |
| `5815006dc2d1adf4c643797cdb8be66f16906b4a270f00704aa5a28b8c6413ec.jpg` | —（md 未引用） | Table 10 | 表格截图 | 4 行对比表，列头 `Model / #Stage (p) / #Microbatch (m) / Post-validation / All-reduce synchronization`。全部数值可读：1.5B/8/24 → `14.5` vs `13.11`；6.2B/8/24 → `4.32` vs `4.00`；14.6B/16/48 → `1.81` vs `1.68`；28.3B/32/96 → `0.99` vs `0.91` | 可转写为 Markdown 表格，不直接引图（md 第 310 行已有同内容 HTML 表） | §3 / §9（「绕过优化器同步」的代价是约 8% 吞吐，是误区一节的反例数据） |
| `76572ce804d44e1d52111124d823efac2e041da3a851ddf120348ac14450f173.jpg` | —（md 未引用） | 公式 (3) | 公式截图 | 单行公式，可读为 `min_(O,E) max_i E_(i,m,W) − E_(i,1,F) + T_(i,1,F)`（ILP 目标：最小化最长 stage 的耗时；md 第 321 行） | 不采用（公式截图，非插图） | — |
| `7828028ce7aed2d6d6439d6f62df284d8048a5ead32e6574fb00299cb51faaa1.jpg` | 38（图注 39） | **Figure 2**（1F1B，**不是 Figure 3**） | 真插图 | 4 行（`Device 1`–`Device 4`）× 约 30 列时间网格，图例 `Forward`（蓝）/`Backward`（橙）/`Optimizer step`（米色）。warm-up F 数呈阶梯：`Device 1` 起始 4 个蓝格（数字 `1 2 3 4`）、`Device 2` 3 个、`Device 3` 2 个、`Device 4` 1 个。稳态为蓝橙交替（`Device 1` 行可读 `1 5 2 6 3 7 4 8 5 … 8`），末尾约 2 个米色 optimizer 格，然后接下一 iteration 的 `1 2 3 4`。**整张图没有任何 W（绿色）格——Backward 是单块橙色**；右端各行尾部留有白色空隙（气泡） | **采用** | **§3**（1F1B 基线；也是「Backward 未被拆开」这一对照的原始形态） |
| `83bb1b328b1293882b60db285c9c212bc8c7250e5193f835d241c30e36c6b336.jpg` | —（md 未引用） | 公式 (1) | 公式截图 | 单行短公式，可读为 `M_limit ≥ k M_B`（md 第 256 行） | 不采用（公式截图，非插图） | — |
| `8a90460c41c6203eb9cae97ed05b17aae1a8a0556193c75b572be66e41a52215.jpg` | —（md 未引用） | Table 3 | 表格截图 | 列头 `Model / Layers / Attention Heads / Hidden Size / Sequence Length / Pipelines (GPUs) / Microbatch Size / Number of Microbatches`。四行可读：1.5B → 22 / 24 / 2304 / 1024 / 8 / 6 / 24·32·64；6.2B → 30 / 32 / 4096 / 1024 / 8 / 3 / 24·32·64；14.6B → 46 / 40 / 5120 / 1024 / 16 / 1 / 48·64·128；28.3B → 62 / 48 / 6144 / 1024 / 32 / 1 / 96·128·256 | 可转写为 Markdown 表格，不直接引图（md 第 105 行已有同内容 HTML 表） | §6（复算气泡 / 显存公式时必须有的一组实验配置：`p`、`m`、`b` 与层数一一对应） |
| `8d84c84746b69917d43a5f30cff45bab5fd8720b7ea9e9e528cd46245f34de0b.jpg` | —（md 未引用） | Table 1 | 表格截图 | 3 行 × 3 列表：`Pass / FLOPs / Activations Memory Required`。全部数值可读：`F` → `sbh(24h + 4s)`，`0`；`B` → `sbh(24h + 8s)`，`sb(34h + 5as)`；`W` → `sbh(24h)`，`32sbh` | 可转写为 Markdown 表格，不直接引图（md 第 60 行已有同内容 HTML 表） | **§6**（`T_W < T_F < T_B`、`T_B + T_W = 2T_F`、`M_W < M_B` 三个口径的原始出处，是复算气泡与显存的起点） |
| `9b05387f3fb57d9004b4a19dcf6f60452e45bec2e0e1e95e5ab2a3fdb9f6b8ed.jpg` | 146（图注 148） | Figure 6 下栏 | 真插图 | 16 行（`Device 0`–`Device 15`）× 约 100 列时间网格，图例 `F`（蓝）/`B`（青）/`W`（绿）/`Optimizer Step`（黄）。与上栏同规模，但**能看到散落的白色空隙**，左端 `Device 0`–`Device 5` 的 warm-up 阶梯因实际测量而不再是理想直线；黄色 optimizer 块错位分布在右端并延伸到下几行 | **采用** | **§6**（理想调度 vs 实测执行的偏差，是「零气泡是渐近性质、不是硬保证」的一手证据）；§9（失败模式：实测通信抖动使气泡无法完全消掉） |
| `9b7189ee7d861471ce1bcd549d0235fdfce700cfb890aa0f981e2f9dc5752a37.jpg` | 144（图注 148） | Figure 6 上栏 | 真插图 | 16 行（`Device 0`–`Device 15`）× 约 100 列时间网格，仅见 `F`（蓝）/`B`（青）/`W`（绿）三色。整体轮廓是**左端蓝色阶梯下降、右端绿色（W）大三角填充**的平行四边形：稳态区密集无空白，`W` 集中在每个 device 行的尾部形成斜边 | **采用** | **§3**（自动调度生成的 ZB-2p 形态：平行四边形 = 零气泡的几何表征）；§6 |
| `a3f9d2a02a9d8cacb8bee2bd14c6ba0504b65af0704d3d4dec44581d27545d12.jpg` | 23（图注 24） | Figure 1 | 真插图 | 上下分栏（`Forward` / `Backward`），`Backward` 栏又被一条竖虚线切成 `B` 与 `W` 两个子栏；底部三个粗体字母 `F`、`B`、`W` 分别对齐三栏。橙色框 = 矩阵乘（`Wx`、`Wᵀ∇_zL`、`∇_zLxᵀ`），白色框 = 逐元素（`σ(z)`、`dσ(z)/dz ∇_yL`）。可读标注：`x → Wx → z → σ(z) → y`；反向 `∇_yL → [dσ(z)/dz ∇_yL] → ∇_zL → Wᵀ∇_zL → ∇_xL`，以及从 `∇_zL` 分出的 `∇_zLxᵀ → ∇_WL`；左侧有 `N ×`、右下有 `× N` 表示 N 层堆叠 | **采用** | **§2**（最小概念模型：F / B / W 三记号与「反向 = ∇_x 一路 + ∇_W 一路」的数学来源，是全文最重要的一张图） |
| `c9b29705469eb8da4e9aa1ad93f81340719df6c60e96ca023ee0cffe2de8111a.jpg` | —（md 未引用） | Table 2 | 表格截图 | 3 行 × 3 列表：`Schedule / Bubble size / Peak activations memory`。全部数值可读：`1F1B` → `(p−1)(T_F + T_B + T_W)`，`pM_B`；`ZB-H1` → `(p−1)(T_F + T_B − T_W)`，`pM_B`；`ZB-H2` → `(p−1)(T_F + T_B − 2T_W)`，`(2p−1)M_B` | 可转写为 Markdown 表格，不直接引图（md 第 70 行已有同内容 HTML 表） | **§6**（气泡与峰值激活显存的成对公式，是「比较口径统一」一节的公式骨架） |
| `d7bc73214438e8cb1fd20f57314cca2350953e8dffdaa0ea6eec235d4ebbcadd.jpg` | —（md 未引用） | Table 9 | 表格截图 | 列头 `Model / #Stage (p) / #Microbatch (m) / T_F / T_B / T_W / T_comm`，12 行数据。全部数值可读，例如 1.5B/p=8/m=24 → `18.522 / 18.086 / 9.337 / 0.601`；28.3B/p=32/m=96 → `10.419 / 10.207 / 7.715 / 0.408`。可验证 `T_W < T_F < T_B` 且 `T_B + T_W ≈ 2T_F`（如 18.086 + 9.337 = 27.423 vs 2×18.522 = 37.04 —— **注意此处并不严格成立**） | 可转写为 Markdown 表格，不直接引图（md 第 302 行已有同内容 HTML 表） | **§6**（气泡率与显存公式的最关键实测输入；同时是「`T_B + T_W = 2T_F` 只是近似」这一口径警告的来源） |
| `d9db5752824976d0d88d27880988b2d36360f4c534d37e998a473d919b4e3299.jpg` | —（md 未引用） | 公式 (5) | 公式截图 | 单行公式，可读为 `E_(i,j,B) ≥ E_(i+1,j,B) + T_comm + T_(i,j,B)`（B 在相邻 stage 的反向依赖；md 第 329 行） | 不采用（公式截图，非插图） | — |
| `e2c1ba8e49b0c79994431d4c405667b4dfb08dcbaa0d535cfc2dd803e8622016.jpg` | 41（图注 42） | **Figure 3**（手工 ZB-H1 / ZB-H2，**不是 `7828028c…`**） | 真插图 | 上下两栏、各 4 行（`Device 1`–`Device 4`）× 约 30 列，共用图例 `F`（蓝）/`B`（青）/`W`（绿）/`Optimizer step`（米色）。**上栏（ZB-H1）**：warm-up F 数 = 4 / 3 / 2 / 1（与 Figure 2 一致），但 warm-up 之后每行仍留有 **3 / 2 / 1 / 0 个白色空格**，稳态区为 B-W 交替（`Device 1` 行可读 `1 1 5 2 2 6 3 3 7 4 4 8 5 5 6 6 7 7 8 8`），行末的 W（绿）填掉一部分尾部气泡，右端米色 optimizer 块呈阶梯且之后接下一 iteration 的 warm-up。**下栏（ZB-H2）**：warm-up F 数增至 **7 / 6 / 5 / 1**，`Device 1` 行在 warm-up 后**不再有白色空格**（直接接 `1 1 8 2 2 3 3 4 4 …`），稳态区密集无空隙，右端 4 个米色 optimizer 块向右下方逐行错位、下一 iteration 的 `1 2 3 4 5 6 7` 在上一 iteration 尚未结束时就开始——整体轮廓是**平行四边形**（正文 §2.2 原文：`changes the layout from trapezoid into a parallelogram`） | **采用** | **§3**（1F1B → Zero Bubble 的调度演进主图，必须采用；同时支撑「W 可延迟」这一结论） |
| `e5c60958dd4e8b1587324e051ce5071a6ba3090d4c6f2a358bae2470ae49d311.jpg` | —（md 未引用） | 公式 (7) | 公式截图 | 单行公式，可读为 `M_limit ≥ ΔM_(i,j′,c′) + Σ_(j,c) ΔM_(i,j,c) O_(i,j,c)→(i,j′,c′)`（ILP 的显存上限约束；md 第 337 行） | 不采用（公式截图，非插图） | — |
| `eae5172efda0beb396e2e577344cd407d5534759f3ef2034c74b54f78006a05c.jpg` | —（md 未引用） | Table 8 | 表格截图 | 列头 `Model / #Stage (p) / #Microbatch (m) / 1F1B / 1F1B-I / ZB-H1 / ZB-H2 / ZB-V`，9 行数据。全部数值可读，例如 6.2B/p=16/m=48 → `0.2668 / 0.1499 / 0.1536 / 0.0823 / 0.0697`；28.3B/p=32/m=256 → `0.1362 / 0.0626 / 0.0593 / 0.0251 / 0.0236`。可验证 ZB-V 全面低于 1F1B / 1F1B-I / ZB-H1，且在中等规模与 ZB-H2 互有胜负 | 可转写为 Markdown 表格，不直接引图（md 第 187 行已有同内容 HTML 表） | **§5 / §6**（ZB-V 与 ZB-H1 / ZB-H2 / 1F1B 的**同口径**气泡率对比，是「V 形布局用一半显存换到接近 ZB-H2 的气泡率」这一结论的唯一数据） |
| `fe536c8d3e686208e719c4b951641ef0d13eee0d53b0a52273beaeb4e093532c.jpg` | —（md 未引用） | Table 5 | 表格截图 | 列头 `Model / #Stage (p) / #Microbatch (m) / 1F1B / 1F1B-I / ZB-H1 / ZB-H2 / ZB-1p / ZB-2p`，12 行数据。全部数值可读，例如 1.5B/p=8/m=24 → `0.2431 / 0.1055 / 0.1585 / 0.1083 / 0.1585 / 0.0433`；6.2B/p=8/m=64 → `0.1091 / 0.0320 / 0.0554 / 0.0294 / 0.0554 / 0.0010`。可验证 ZB-2p 在多数配置下降到 <1%，且 **ZB-H1 与 ZB-1p 数值逐格相同**（说明 ZB-1p 受显存上限支配而退化为 ZB-H1） | 可转写为 Markdown 表格，不直接引图（md 第 136 行已有同内容 HTML 表） | **§6**（各调度气泡率的统一口径表；`ZB-H1 ≡ ZB-1p` 这个巧合本身就是「显存上限成为主导因素」的证据） |

## 采用图的正文引用块（可直接粘贴）

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/zero-bubble-fig1-mlp-computation-graph-f-b-w.jpg" alt="Zero Bubble Figure 1：MLP 计算图。底部分 F / B / W 三栏，Backward 栏内部再用竖虚线分成 B 与 W；橙色框为矩阵乘 Wx、Wᵀ∇_zL、∇_zLxᵀ，白色框为逐元素 σ(z)、dσ(z)/dz·∇_yL；左标 N×、右标 ×N 表示 N 层堆叠" style="width: 72%;">
</div>

> **图片来源**：Zero Bubble Pipeline Parallelism（arXiv 2401.10241v1）Figure 1，§1 Introduction。本地副本：`references/papers/zero-bubble/zero-bubble.md` 第 23 行引用图（图注在第 24 行）。

建议文件名：`zero-bubble-fig1-mlp-computation-graph-f-b-w.jpg`（原文件 `a3f9d2a0…jpg`，762×457，`.jpg` 沿用原扩展名）。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/zero-bubble-fig2-1f1b-schedule.jpg" alt="Zero Bubble Figure 2：1F1B 调度。Device 1–4 四行时间网格，蓝 = Forward、橙 = Backward、米色 = Optimizer step；warm-up 的 F 数依次 4/3/2/1，稳态蓝橙交替，右端留有白色气泡；整图没有任何 W 格" style="width: 100%;">
</div>

> **图片来源**：Zero Bubble Pipeline Parallelism（arXiv 2401.10241v1）Figure 2，§2 Handcrafted Pipeline Schedules。本地副本：`references/papers/zero-bubble/zero-bubble.md` 第 38 行引用图（图注在第 39 行）。

建议文件名：`zero-bubble-fig2-1f1b-schedule.jpg`（原文件 `7828028c…jpg`，1026×159）。

**引用时需要注意（纠正草稿的两处说法）**：① 该文件是 **Figure 2（1F1B）**，草稿 `PROMPT-learn-pipeline.md` 第 72 行把它当作「Figure 3 手工调度 ZB-H1 / ZB-H2」，判别错误。② 草稿同处怀疑「Figure 2 的 1F1B 可能与 Figure 3 在同一图块」——读图结论是**两张完全独立的文件**：本文件只有 1F1B，往后翻一行的 `e2c1ba8e…jpg` 才是 Figure 3（上下两栏的 ZB-H1 / ZB-H2）。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/zero-bubble-fig3-handcrafted-zb-h1-h2.jpg" alt="Zero Bubble Figure 3：手工调度。上栏 ZB-H1（warm-up F 数 4/3/2/1，warm-up 后仍留 3/2/1/0 个白色气泡，W 填尾部）；下栏 ZB-H2（warm-up F 数增至 7/5/3/1，行内无白色空隙，米色 Optimizer step 逐行错位，整体为平行四边形）" style="width: 100%;">
</div>

> **图片来源**：Zero Bubble Pipeline Parallelism（arXiv 2401.10241v1）Figure 3，§2 Handcrafted Pipeline Schedules（§2.1 Memory Efficient Schedule / §2.2 Zero Bubble Schedule）。本地副本：`references/papers/zero-bubble/zero-bubble.md` 第 41 行引用图（图注在第 42 行）。

建议文件名：`zero-bubble-fig3-handcrafted-zb-h1-h2.jpg`（原文件 `e2c1ba8e…jpg`，1032×310）。

**引用时需要注意（这是全文最关键的一张，务必按读图结果写）**：草稿把「手工调度」的哈希指向了 Figure 2 的文件，因此把它换成本文件即可。写正文时可用两个**读图可验证**的对比来锚定 ZB-H1 / ZB-H2 的差别：warm-up 的 F 数从 `4/3/2/1` 增到 `7/5/3/1`；且 ZB-H1 上栏在 warm-up 后每行仍留 `3/2/1/0` 个白色空格，ZB-H2 下栏 `Device 1` 行 warm-up 后**没有空格**，米色 optimizer 块逐行错位（这正是 §2.2 说「移除优化器同步」、§4 专门讨论绕过的原因）。据此可支撑「I（输入梯度）优先、W（权重梯度）可延迟是调度策略而非数学必然」：图里 W 只是被搬到尾部填气泡。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/zero-bubble-fig4-optimizer-post-validation.jpg" alt="Zero Bubble Figure 4：优化器 post-validation。图例：橙块 1–4 = 沿对角线传播局部归约值、橙块 5–8 = 把全局归约值回传各 stage、米色 = Optimizer step、深红 = validation 失败时 rollback" style="width: 100%;">
</div>

> **图片来源**：Zero Bubble Pipeline Parallelism（arXiv 2401.10241v1）Figure 4，§4 Bypassing Optimizer Synchronizations。本地副本：`references/papers/zero-bubble/zero-bubble.md` 第 94 行引用图（图注在第 95 行）。

建议文件名：`zero-bubble-fig4-optimizer-post-validation.jpg`（原文件 `1cfd9a39…jpg`，865×198）。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/zero-bubble-fig5-throughput-comparison.jpg" alt="Zero Bubble Figure 5：四个面板（1.5B/8GPU、6.2B/8GPU、14.6B/16GPU、28.3B/32GPU）的吞吐柱状图，蓝 1F1B、橙 1F1B-I、绿 ZB-1p、红 ZB-2p，另有一个空心黑框 upper bound 只画在 1F1B 柱上" style="width: 100%;">
</div>

> **图片来源**：Zero Bubble Pipeline Parallelism（arXiv 2401.10241v1）Figure 5，§5.2 Main Results。本地副本：`references/papers/zero-bubble/zero-bubble.md` 第 119 行引用图（图注在第 120 行）。

建议文件名：`zero-bubble-fig5-throughput-comparison.jpg`（原文件 `27cf2bf2…jpg`，1032×512）。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/zero-bubble-fig6top-zb2p-searched-schedule.jpg" alt="Zero Bubble Figure 6 上栏：自动搜索出的 ZB-2p 调度。Device 0–15 共 16 行时间网格，蓝 F、青 B、绿 W；左端蓝色阶梯下降、右端绿色 W 构成大三角，稳态区密集无空白，轮廓为平行四边形" style="width: 100%;">
</div>

> **图片来源**：Zero Bubble Pipeline Parallelism（arXiv 2401.10241v1）Figure 6（上栏），§5.3 Efficiency of Automatic Scheduling。本地副本：`references/papers/zero-bubble/zero-bubble.md` 第 144 行引用图（图注在第 148 行）。

建议文件名：`zero-bubble-fig6top-zb2p-searched-schedule.jpg`（原文件 `9b7189ee…jpg`，973×301）。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/zero-bubble-fig6bottom-zb2p-profiled-execution.jpg" alt="Zero Bubble Figure 6 下栏：ZB-2p 在 16 GPU 上的实测 profile。Device 0–15 共 16 行，蓝 F、青 B、绿 W、黄 Optimizer Step；与上栏同规模但可见散落白色空隙，调度轮廓与理想解总体对齐" style="width: 100%;">
</div>

> **图片来源**：Zero Bubble Pipeline Parallelism（arXiv 2401.10241v1）Figure 6（下栏），§5.3 Efficiency of Automatic Scheduling。本地副本：`references/papers/zero-bubble/zero-bubble.md` 第 146 行引用图（图注在第 148 行）。

建议文件名：`zero-bubble-fig6bottom-zb2p-profiled-execution.jpg`（原文件 `9b05387f…jpg`，1043×359）。

**引用时需要注意**：MinerU 把 Figure 6 切成两个文件（第 144、146 行各一个 `![]()`，共用一个第 148 行的图注）。正文若要还原论文的「理想 vs 实测」对照，需要**同时**引这两个文件，并明确上/下栏顺序，否则只引任意一张都会丢掉这层对照。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/zero-bubble-fig7-memory-limit-vs-bubble-rate.jpg" alt="Zero Bubble Figure 7：四个面板（1.5B p=8 / 6.2B p=8 / 14.6B p=16 / 28.3B p=32）的 bubble rate–M_limit 曲线，横轴刻度 1.0pM_B / 2.0pM_B / 3.0pM_B，每面板三条曲线对应不同 #Microbatches；曲线在 2.0pM_B 附近压到 0 并转为水平" style="width: 100%;">
</div>

> **图片来源**：Zero Bubble Pipeline Parallelism（arXiv 2401.10241v1）Figure 7，§5.4 Memory Limit。本地副本：`references/papers/zero-bubble/zero-bubble.md` 第 149 行引用图（图注在第 150 行）。

建议文件名：`zero-bubble-fig7-memory-limit-vs-bubble-rate.jpg`（原文件 `3955441b…jpg`，1035×229）。

**引用时需要注意（纠正草稿的图号）**：草稿 `PROMPT-learn-pipeline.md` 第 73 行把该文件当作「**Figure 8** ZB-V 调度」。读图结论：它是 **Figure 7**，内容是把气泡率画成 `M_limit`（以 `pM_B` 为单位）的函数，根本不涉及 ZB-V 的调度布局。ZB-V 调度图是 `53647889…jpg`（见下一条）。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/zero-bubble-fig8-zbv-schedule.jpg" alt="Zero Bubble Figure 8：ZB-V 调度。Device 1–4 四行、每行两个 chunk（白色数字 = 第一 chunk，黑色数字 = 第二 chunk）；蓝 F、青 B、绿 W、米色 Optimizer step；左端 F 呈下降阶梯，稳态为密集 1F-1B-1W，右端米色 optimizer 块逐行错位" style="width: 100%;">
</div>

> **图片来源**：Zero Bubble Pipeline Parallelism（arXiv 2401.10241v1）Figure 8，§6 Memory Efficient Zero Bubble Schedule。本地副本：`references/papers/zero-bubble/zero-bubble.md` 第 158 行引用图（图注在第 159 行）。

建议文件名：`zero-bubble-fig8-zbv-schedule.jpg`（原文件 `53647889…jpg`，1035×159）。

**引用时需要注意（这是与 DualPipeV 关系最近的一张，图号必须用对）**：该文件是 **Figure 8**，也是草稿真正想要的「ZB-V 调度（DualPipeV 的直系前序）」；草稿把它误记成了 `3955441b…`（实为 Figure 7）。写 §5 时可直接用图注里的一句话锚定读图事实：每个 device 恰好分到 **2 个 chunk**，`V` 形体现在第一个 microbatch 的 forward 上（论文 §6 的 16 层 / 4 stage 例子：worker 1 拿 layers 1-2 与 15-16，worker 2 拿 3-4 与 13-14）。与 DualPipeV 的差别（DualPipeV 是 `2p` 个 chunk 的更细切分 + 双向）需要在 §5 里显式说明，不能把两篇的切分粒度混为一谈。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/zero-bubble-fig9-memory-limit-vs-bubble-rate-zbv.jpg" alt="Zero Bubble Figure 9：3×3 九联曲线，列标题给出 6.2B p=16 m=48/64/128、14.6B p=24 m=72/96/192、28.3B p=32 m=96/128/256；每个面板两条曲线 ZB（蓝）与 ZB-V（橙），横轴 1.0pM_B/2.0pM_B/3.0pM_B，ZB-V 整体更低且更早归零" style="width: 92%;">
</div>

> **图片来源**：Zero Bubble Pipeline Parallelism（arXiv 2401.10241v1）Figure 9，§6.2 Schedule Efficiency。本地副本：`references/papers/zero-bubble/zero-bubble.md` 第 191 行引用图（图注在第 192 行）。

建议文件名：`zero-bubble-fig9-memory-limit-vs-bubble-rate-zbv.jpg`（原文件 `4bce0021…jpg`，1003×776）。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/zero-bubble-fig10-w-grouping-dp-allreduce-overlap.jpg" alt="Zero Bubble Figure 10：(a) The schedule grouped by W —— 同参数的计算聚成 4 个大块，右侧 4 个 AR（All-Reduce）块与计算段完全不重叠；(b) The schedule grouped by parameter —— 块内改为 1 2 3 4 交织，4 个 AR 块错位插入并与相邻计算块并行" style="width: 62%;">
</div>

> **图片来源**：Zero Bubble Pipeline Parallelism（arXiv 2401.10241v1）Figure 10，Appendix A Overlap Communication in Data Parallelism。本地副本：`references/papers/zero-bubble/zero-bubble.md` 第 248 行引用图（图注在第 249 行）。

建议文件名：`zero-bubble-fig10-w-grouping-dp-allreduce-overlap.jpg`（原文件 `450daae1…jpg`，525×179）。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/zero-bubble-fig11-zero-bubble-15b-32mb.jpg" alt="Zero Bubble Figure 11：1.5B 模型、32 microbatches 的零气泡调度。Device 0–7 共 8 行，仅蓝 F、青 B、绿 W 三色图例（无 Optimizer step 图例）；左端蓝色阶梯下降，右半为绿色 W 构成的三角形，行内无明显空白" style="width: 100%;">
</div>

> **图片来源**：Zero Bubble Pipeline Parallelism（arXiv 2401.10241v1）Figure 11，Appendix B The Memory Limit for Automatic Scheduling Algorithm。本地副本：`references/papers/zero-bubble/zero-bubble.md` 第 265 行引用图（图注在第 266 行）。

建议文件名：`zero-bubble-fig11-zero-bubble-15b-32mb.jpg`（原文件 `269fe44b…jpg`，1042×167）。

**引用时需要注意**：这一张是**可选**项。它与 Figure 6 上栏展示的是同一几何形态（左端蓝色阶梯 + 右端绿色 W 三角），只是规模换成 8 个 stage / 32 microbatches。若 §6 篇幅有限，只保留 Figure 6 上栏即可；本图的边际价值在于它的图例**没有 Optimizer step**，可作为「§4 的同步绕过在附录实验里确实被去掉」的旁证。

## 存疑与分歧

1. **草稿的两条关键图号判断都是错的，已按读图结果纠正。** `PROMPT-learn-pipeline.md` 第 72 行称 `7828028ce7aed2d6d6439d6f62df284d8048a5ead32e6574fb00299cb51faaa1.jpg` 是「Figure 3 手工调度 ZB-H1 / ZB-H2」——实际它是 **Figure 2（1F1B）**（md 第 39 行图注 `Figure 2: 1F1B pipeline schedule.`），读图也证实整图只有 Forward / Backward / Optimizer step 三色，**没有任何 W 格**。真正的 Figure 3 是 `e2c1ba8e49b0c79994431d4c405667b4dfb08dcbaa0d535cfc2dd803e8622016.jpg`（md 第 42 行图注 `Figure 3: Handcrafted pipeline schedules, top: ZB-H1; bottom: ZB-H2`），读图证实的上下两栏与图注完全一致。第 73 行称 `3955441ba7fa9c5a066548e680cb97da181f5cb01b31e187b673517bfa0053d2.jpg` 是「Figure 8 ZB-V 调度」——实际它是 **Figure 7**（bubble rate ↔ M_limit 的四联折线图）；真正的 Figure 8 是 `536478895e0abbcea17d6178866439923d5a02d53f94cfd02f19ea5706b189e0.jpg`。两处均属草稿的哈希错位，不影响「要引 Figure 3 / Figure 8」的意图。

2. **Figure 2 的 1F1B 与 Figure 3 的 ZB-H1/H2 不在同一图块。** 草稿的第 72 行括号注记「Figure 2 的 1F1B 可能在同图块，需分辨」经核实不成立：两者是 md 第 38 行与第 41 行的两个独立 `![]()`，各有独立图注，图片几何尺寸也不同（1026×159 vs 1032×310）。

3. **Figure 1 不是「1F1B 与拆分 B/W 的概念对比」，而是 MLP 计算图。** 草稿期望 Figure 1 用来对比 1F1B 与 B/W 拆分；读图结论是它画的是单层 MLP 的前向与反向计算图（`x → Wx → σ(z) → y`，反向分成 `∇_x` 与 `∇_W` 两路），底部用 `F` / `B` / `W` 三个字母标注三栏。它的作用是**定义记号**（对应文章第 2 章的 `I` / `W` 拆分数学），不是调度对比；调度对比在 Figure 2 与 Figure 3。

4. **本目录没有任何「B = B1 + B2」的图，两篇论文的记号需要显式对齐。** 用户任务里提到的「B = B1 + B2 拆分图」在 zero-bubble 里并不存在：这篇论文的拆分记法是 **B 与 W**（`B = ∇_x f(x,W)ᵀ dℓ/dy`，`W = ∇_W f(x,W)ᵀ dℓ/dy`，见 Figure 1 与 md 第 26 行）。DualPipe / DeepSeek-V3 侧常用的记法是 `B = B1 + B2`，其中 `B1`（输入梯度）对应本文的 `B`、`B2`（权重梯度）对应本文的 `W`。**这一层记号映射必须写进文章的 §6「比较口径统一」**，否则把 zero-bubble 的 `T_B` 直接当成 DualPipe 的 `B_full` 会重复计数。

5. **`T_B + T_W = 2T_F` 与 Table 9 的实测值并不严格自洽。** md 第 56 行断言 `T_W < T_F < T_B` 且 `T_B + T_W = 2T_F`；但 Table 9 的实测（图片 `d7bc7321…jpg`）例如 1.5B/p=8/m=24 为 `T_F = 18.522`、`T_B = 18.086`、`T_W = 9.337`，`T_B + T_W = 27.423` 而 `2T_F = 37.044`，相差约 26%，并且 `T_B < T_F` 也成立（与 `T_F < T_B` 相反）。文章 §6 若要复算，必须说明这是「按 matmul FLOPs 的解析近似」与「profiled 实测」两种口径，不能混用；这一点在草稿里尚未被提及。

6. **19 张未引用图全部是重复产物，不是「丢失的插图」。** 12 张 Table 截图（`2483da0e`=T4、`2a378d26`=T6、`2d9fe9b1`=T7、`48cf37b8`=T12、`56ccdcb4`=T11、`5815006d`=T10、`8a90460c`=T3、`8d84c847`=T1、`c9b29705`=T2、`d7bc7321`=T9、`eae5172e`=T8、`fe536c8d`=T5）在 md 中均无 `![]()` 引用，其内容已在 md 第 60/70/105/124/136/177/181/187/302/310/344/352 行以 HTML `<table>` 形式存在；7 张公式截图（`0de5157f`=(2)、`0f7b9870`=(6)、`175564bd`=(4)、`76572ce8`=(3)、`83bb1b32`=(1)、`d9db5752`=(5)、`e5c60958`=(7)）对应 md 第 260/333/325/321/256/329/337 行的 `$$…\tag{N}$$`。要引这些数据或公式，直接从 md 的 HTML 表 / LaTeX 转写，不必引图。

7. **Table 6 / Table 7 / Table 11 / Table 12 属于同一批吞吐实验的重复切面。** 四张表的 `Model / p / m / b / Samples per GPU per second / Memory(GB) / Schedule` 列结构相同，只是筛选条件不同（Table 11 = 同显存下的 1F1B/ZB-1p/ZB-2p；Table 6 = ZB-V 与 ZB-2p\*；Table 7 = 翻倍 micro-batch；Table 12 = `m ≤ p` 极端场景）。文章 §6 如果要把口径讲清楚，建议只转写 Table 9（profiled 时间）+ Table 5 / Table 8（气泡率）+ Table 1 / Table 2（公式），其余按需引用而不铺开。

8. 本目录 31 张里**没有页眉 / logo / 装饰 / 图内碎片**，也没有需要单独标注的「低分辨率不可读」图；所有 31 张都读得清。

9. **`torch/dualpipe/pics/` 目录当前不存在**——`torch/dualpipe/` 下只有 `deep-dive_draft-1.md`、`deep-dive_draft-2.md`、`PROMPT-learn-pipeline.md`（及 `PROMPT-learn-pipeline.md` 之外的同级文件）、`notes/`、`references/`。粘贴上面的引用块前需要先建 `pics/` 并按建议文件名把原图复制过去；本次任务只产出核对表，未创建或复制任何图片文件。

10. 「建议章节」是按任务下达的 9 节拟定结构给出的，不是从草稿既有引用反推的。两个草稿里**没有出现 zero-bubble 的图片哈希**（只有 `PROMPT-learn-pipeline.md` 第 72–73 行两处待办清单式的哈希记录，且两处图号均已证伪）。
