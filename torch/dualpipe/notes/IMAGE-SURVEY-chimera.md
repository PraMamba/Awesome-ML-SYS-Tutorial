# Chimera（arXiv 2107.06925）图片逐张核对

核对对象：`torch/dualpipe/references/papers/chimera/images/` 下的全部 26 张 `.jpg`。
MinerU 产物：`torch/dualpipe/references/papers/chimera/chimera.md`（434 行，18 处图片引用、4 处 HTML 表格）。
核对方式：26 张全部用 `read_image` 逐张打开读图，下表「内容」列全部为**看图后**的事实描述（含图中可读文字与数值），未依据文件名或 md 顺序推断。

## 汇总

- `images/` 实际文件数（`ls | wc -l`）：26
- 本清单覆盖行数：26
- 真插图：19；公式截图：3；表格截图：4；页眉/装饰/碎片：0
- 建议采用：8 张
- 分类明细：
  - 真插图 19 张 = Figure 1–19（每张图各自一个文件，编号与 paper 一致，无缺号）
  - 表格截图 4 张 = Table 1（`8a27caa4`）、Table 2（`e4347bf9`）、Table 3（`8c4e7ed6`）、Table 4（`31651caef`）
  - 公式截图 3 张 = 式 (1) 运行时间模型（`95e8d4b9`）、allreduce 代价（`b0fc08dd`）、mini-batch SGD 梯度（`d706e16f`）
- 关键事实 1（回答草稿的疑问）：`aee6624ecf6b7a7091069e7856d11eadb156d52b3be993e54e78783904dcf06b.jpg` 是 **Figure 2**，不是 Figure 3。读图可见它是六种调度（PipeDream / PipeDream-2BW / GPipe / GEMS / DAPPLE / Chimera）的统一时间轴对照图，与 md 第 75 行图注 `Figure 2: Pipeline parallelism schemes, with four pipeline stages (D=4) and four micro-batches (N=4)…` 完全对应。**Figure 3** 是另一张 `270ff454…jpg`：`Figure 3: Model replicas and bidirectional pipelines scheduling of Chimera.`，画面是「down pipeline + up pipeline 合并成 Chimera」的构造过程（详见「存疑与分歧」第 1 条）。
- 关键事实 2：Chimera 的 Figure 4 在 md 中**没有图片引用** —— MinerU 把 Figure 4 画面里的表格部分转成了第 114 行的 HTML `<table>`、把子图注留在第 116 行，图片本体 `de1d0f4b…jpg` 只落在 `images/` 里。因此 26 张文件里 md 只引用 18 张，未引用的 8 张 = Figure 4（1）+ Table 1–4（4）+ 公式（3）。

## 逐张清单

| 文件名 | md 引用行 | 图号/表号 | 类别 | 内容（读图后的事实描述，含图中可读的关键标注/数值） | 结论 | 建议章节 |
| --- | --- | --- | --- | --- | --- | --- |
| `c77254c3940819fcd0b801c19bd17deb1843c345ba140047eb30d43885fe4633.jpg` | 37 | Figure 1 | 真插图 | 三联柱状图（GPT-2，2048 GPU 节点）。左 `Bubble Ratio (%)`：PipeDream ≈1、PipeDream-2BW ≈1（旁注 `≈0 but with stale weights`）、GPipe ≈45、GEMS ≈83、DAPPLE ≈49、Chimera ≈32。中 `Peak Memory Cost (GB)`：PipeDream 顶到 16、GPipe / PipeDream-2BW / DAPPLE 三根柱顶标 `OOM`、GEMS ≈13、Chimera ≈15.2。右 `Throughput (sequences/s)`：PipeDream ≈255、PipeDream-2BW ≈430、GPipe ≈360、GEMS ≈220、DAPPLE ≈370、Chimera ≈505，柱间用双箭头标 `2.01x`、`1.16x`、`1.42x`、`2.34x`、`1.38x`。图例含配置：`PipeDream (D=8, R)`、`PipeDream-2BW (D=16, R)`、`GPipe (D=8, R)`、`GEMS (D=8)`、`DAPPLE (D=16, R)`、`Chimera (D=32)` | **建议采用** | §6（气泡率 / 峰值显存 / 吞吐三个口径并排：同一张图里 Chimera 气泡率最低但峰值显存并不是最低，是「比较口径统一」最好的引子）；§9（选型权衡） |
| `aee6624ecf6b7a7091069e7856d11eadb156d52b3be993e54e78783904dcf06b.jpg` | 74 | **Figure 2**（草稿疑为 Figure 3，实为 Figure 2） | 真插图 | 六种调度的统一时间轴对照，顶部只有一条公共 `Time` 轴，行标签 P0–P3。自上而下：`PipeDream`、`PipeDream-2BW`（图内标 `PipeDream apply gradients` / `PipeDream-2BW apply gradients`，右侧省略号 `…`）、`GPipe`、`GEMS`、`DAPPLE`、`Chimera`；左侧竖排分组标签为 `asynchronous with stale weights`（前两组）与 `synchronous, convergence-friendly`（后四组）；每行右端标 `flush` 竖线。Chimera 行编号双向交错（如 P0 行 `0 1 2 2 3 3 0 1`），底注 `Will be discussed in detail in Section 3.1.`。各行右侧另附权重/激活显存小柱：`M_θ for PipeDream`、`M_θ for PipeDream-2BW`、`M_a for both`，GPipe 面板标 `proportional to N`。图例框原文：`Bubble`（白格）、`x / x` = Forward and backward passes of *model replica0* for micro-batch x、`y / y` = ……*model replica1* for micro-batch y、`Note that a backward pass is about 2 times workload of a forward pass.` | **建议采用** | §4（**双向流水线前史主图**：一张图同时排出 6 种调度，直接支撑「双向布局并非 DualPipe 首创」，并显示 Chimera 的 replica0/replica1 两向交错长什么样）；§3（1F1B 家族谱系） |
| `270ff45471492c201af79e28da9e509b7beb940b080dd19a299f9cf3930b8471.jpg` | 81 | Figure 3 | 真插图 | 双向流水线的构造过程图。左上 `model replica0`：`P0(stage0)`–`P3(stage3)` 顺序映射，两组微批 `0/1` 相加得到 `down pipeline`，下注 `N/2 = 2 micro-batches, where N = D = 4`。左下 `model replica1`：`P0(stage3)`–`P3(stage0)` 反序映射，微批 `2/3` 相加得到 `up pipeline`。中段 `=` 合并，右上给出 `model replica0 / model replica1` 双列映射（P0 持 stage0+stage3、P1 持 stage1+stage2、P2 持 stage2+stage1、P3 持 stage3+stage0）。右下合并后的 `Chimera` 时间轴 4 行 P0–P3，编号交错（P0 行 `0 1 2 2 3 3 0 1`），右端 `flush`，下注 `Chimera (backward is 2x workload of forward)`。顶部注 `Here we assume equal workload between forward and backward passes for simplicity.` | **建议采用**（chimera 侧**最高优先级**的一张） | §4（**双向流水线整体时间轴主图**：两个方向的数据流如何各自填对方的空档、为什么每个 worker 必须持两个 stage、backward 2× 时形态如何变；也是 §5 DualPipeV「V 形 + 切成两半」最直接的对照基线） |
| `de1d0f4b487616c17dca4bd60c8036b284110c73c93d602f7a1132433644911d.jpg` | 114（md 未引用本图片文件；MinerU 把画面转成 HTML `<table>`，子图注在第 116 行） | Figure 4 | 真插图 | 上下面板。(a) 子图注 `(a) Gradient synchronization after all local computation is finished`：左列 `model replica0 / model replica1` 映射（P0: stage0/stage3、P1: stage1/stage2、P2: stage2/stage1、P3: stage3/stage0），右侧 4 行时间轴编号 `0 1 2 2 3 3 0 1 2` 等，末端红底格依次为 `S3 S0`、`S2 S1`、`S2 S1`、`S3 S0`。(b) 子图注 `(b) Eager gradient synchronization for deeper overlapping`：同样 4 行，但红底 `S3` / `S0` 格被**提前**插进中段编号之间（P0 行在 `3` 之后立刻出现 `S3`），表示用气泡期提前发起 allreduce。图内图例：`S_i` = Gradients synchronization for *stage_i* | **建议采用** | §4（「细粒度通信-计算交错」的前史：Chimera 已把权重梯度同步塞进空闲区间）；§7（源码走读对照：DualPipe 的 `WeightGradStore` 推迟权重梯度计算以对齐 P2P，与这里「梯度一算完就 eager allreduce」是同类思路的两种实现） |
| `b6584fa2a3b009ccf5df3e0dc3fd8c62bb667e8489961bca4a2a6569f6f9a30f.jpg` | 125 | Figure 5 | 真插图 | 双列映射表 + 8 行时间轴。左列 `model replica0`（`stage0`–`stage3`）、右列 `model replica1`（`stage3`–`stage0`），行 P0–P3 与 P4–P7 两段映射完全相同（即 W=2 的两个 stage 副本）；右侧 8 行时间轴编号（P0 行 `0 1 2 2 3 3 0 1`，P4 行 `4 5 6 6 7 7 4 5`），红底格 `S3 S0` / `S2 S1` 表示 `stage_i` 的梯度同步。对应正文 `W = 2, D = 4` | 不采用（讲的是「双向流水线 × 数据并行副本」的组合形态，属 Chimera 的扩展议题；本文章 §4/§8 用 Figure 3 + Figure 8 已覆盖，DP 组合不是主线） | —（备选：若 §9 要讨论「双向流水线如何与 DP/EP 组合」，此图是 Chimera 侧唯一的口径图） |
| `1eb7499859cd9765bf35fb579f66f9b8253ea7b1d46ebad219497f4b7a86c386.jpg` | 140 | Figure 6 | 真插图 | 6 行 `P0`–`P5` 的 Chimera 时间轴（对应 `N = D = 6`），编号 0–5 在蓝/橙格与白格间交错；图上用橙色空心折线标出 `one critical path`（一条斜向贯穿的格子链），右上橙框标 `region (a)`、`region (b)`，中部紫框标 `region (c)`；横轴右端被竖线切成 `training iteration0` / `training iteration1` | 不采用（性能模型的「可重叠自由区」示意，服务 §3.4 的建模细节，本文章不需要） | —（备选：若 §6 要论证「气泡区间可以被通信填满」，这是 Chimera 侧的定量依据） |
| `806f053dc964a454f6fad23f76b19f9a1520884ced549078dd971af07911efc6.jpg` | 163 | Figure 7 | 真插图 | 四个面板。(a) `(a) N=2D micro-batches, where D=4`：左右两块 4 行时间轴中间一个黑 `+`，右块编号 `4 5 6 6 7 7 4 5`。(b) `(b) Direct concatenation (intermediate bubbles)`：4 行 P0–P3，编号 `0 4 1 5 6 6 7 7 …`，中段出现连续空白（`intermediate bubbles`）。(c) `(c) Concatenation with forward doubling (no intermediate bubbles)`：每格塞入两个编号（`0 1`、`4 5`、`6 7`…）。(d) `(d) The final schedule of forward doubling after removing half bubbles at the beginning`：形状同 (c) 但起始段被压缩，一条橙色箭头标注 `→ p2p` 指向相邻两格之间；四行右端均有 `flush` | **建议采用** | §4 / §6（「前向加倍 / 反向减半」是通过**改变前反向粒度**来配平工作量、消除中段气泡；与 DualPipe「把反向拆成 I=GΘ 与 W=GᵀX 再重新配对」属同类思路的不同解法 —— 用来证明「拆反向」不是唯一手段，也说明 DualPipe 的贡献在组合方式而非单向技巧） |
| `5242db9131a634574f7d346b98c5002220dea4489606f4fd0002c5485c850902.jpg` | 170 | Figure 8 | 真插图 | 左侧四个 8 行（P0–P7）子图：`model replica0 (down pipeline0)`、`model replica1 (down pipeline1)`、`model replica2 (up pipeline0)`、`model replica3 (up pipeline1)`，各自标出 `stage0`–`stage7` 的映射（down pipeline1 为 `stage4,5,6,7,0,1,2,3`，即整体平移 4）；中间黑箭头合并，右侧为 `model replicas 0-3` 双列映射表 + 合并后的 8 行时间轴（P0 行 `0 1 6 2 7 3 4 4 5 5 2 6 3 7 0 1`），右下标注 `Chimera (a combination of four pipelines)`，右端 `flush`。图例：`Bubble` / `x` = model replicas 0-1 / `y` = model replicas 2-3 | **建议采用** | §4 / §5（双向布局的推广形态 f=2：4 条管线、每 worker 持 4 个 stage。与 DualPipeV「V 形把一条 device 切成多段持有」正好是一组对照：**「一个 device 持多段」并非 DualPipe 独有**） |
| `9b67cf40eb07954f43df56e69320656aa321a7299c8af0c4e48ea56d12975fbe.jpg` | 195 | Figure 9 | 真插图 | 6 行配置、每行一个箱线/散点图，横轴为方案（`Chimera`、`DAPPLE`、`GEMS`、`GPipe`、`PipeDream`、`PipeDream-2BW`），纵轴 `Memory consumption (GB)` 0–16。行标题：`Bert-48 (W=2, D=16, B=8, B̂=512)`、`Bert-48 (W=4, D=8, B=8, B̂=512)`、`Bert-48 (W=4, D=8, B=16, B̂=512)`、`GPT-2 with 32 layers (W=1, D=32, B=1, B̂=512)`、`GPT-2 with 32 layers (W=2, D=16, B=1, B̂=512)`、`GPT-2 with 32 layers (W=2, D=16, B=2, B̂=512)`；多处标 `OOM`，每格内红圈=最大值、蓝圈=最小值。可读事实：Chimera 与 GEMS 的箱体最矮最稳（如 GPT-2 W=1,D=32 行 ≈3–5 GB 对 DAPPLE ≈5–13）；PipeDream-2BW 随 stage 变粗出现 OOM；DAPPLE 在首 worker 出现高红圈 | **建议采用** | §6（显存口径重算：同一张图同时给出「峰值」与「跨 worker 分布」，正是 DualPipe 论文强调 activation memory 均衡时必须对齐的口径；也提醒「峰值最低」与「分布最均衡」是两件事） |
| `e1ed6bf933213655befc3d5d90a956c7ae5539b63888f001e0b7a3e0eee80666.jpg` | 212 | Figure 10 | 真插图 | 4×5 网格柱状图：行 = `W=2, D=16` / `W=4, D=8` / `W=8, D=4` / `W=16, D=2`；列 = `GPipe` / `GEMS` / `DAPPLE` / `PipeDream-2BW` / `PipeDream`；纵轴 `Throughput (sequences/s)` 0–160；横轴为 `B=1/2/4/8/16`（PipeDream-2BW 为 `B=8/16`，PipeDream 列为 `B̂=24/48/80/128`）。柱上标 `R`（激活重算）、`★`（该方案最佳）、`N/A`、`OOM`。可读最佳点：`(W=8, D=4)` 行内 GPipe 与 DAPPLE 的 `B=4` 柱带 `★` | 不采用（baseline 超参搜索空间图，属 Chimera 特有的调参流程，与本文主线无关） | — |
| `e6c55decc0e4ca3ee5012523a337b8d6764d51280f518645b6140bef4a20f640.jpg` | 223 | Figure 11 | 真插图 | 5 个面板柱状图：`GPipe`、`DAPPLE`、`PipeDream-2BW`、`PipeDream`、`GEMS`；纵轴 `Throughput (sequences/s)` 0–150；横轴为 `D=32,B=1` / `D=16,B=1` / `D=8,B=1` / `D=4,B=1`（PipeDream 为 `B̂=16/96/128`，GEMS 为多组 `D,B`）。标注 `R`、`★`、`OOM`。可读最佳：GPipe 在 `D=8,B=1` 带 `★`（≈100），DAPPLE 在 `D=16,B=1`，PipeDream-2BW 在 `D=16,B=1`，PipeDream 在 `D=8,B̂=128`，GEMS 在 `D=8,B=2` | 不采用（同上，GPT-2 侧的 baseline 调参图） | — |
| `3ff12e346ebc30298f27907d4e9a493e94ac40cfa876e985bc48844950a3723e.jpg` | 228 | Figure 12 | 真插图 | 分组柱状图。横轴 `16 nodes` / `32 nodes` / `64 nodes`，纵轴 `Throughput normalized to Chimera (eager-sync)`（0.0–1.5）。两组：`Chimera (eager-sync)`（红，恒为 1.0，作为基准）与 `Chimera (eager-sync-opt)`（蓝，≈1.03 / ≈1.09 / ≈1.12） | 不采用（Chimera 内部两种梯度同步策略的消融，粒度太细） | — |
| `aa82c408ca2a67ab033ecd539ab0e9e73ab66c4a14677af3bbed7bbd853ff149.jpg` | 231 | Figure 13 | 真插图 | 两段横向条形图。上：`Bert-48 on 32 GPU nodes, B̂ = 512`，纵轴配置 `W=2,D=16,B=16` / `W=4,D=8,B=16` / `W=8,D=4,B=8` / `W=16,D=2,B=4`，横轴 `Throughput (sequences/s)` 60–180，红色折线标 `Performance Model` 的预测落点（选中 `W=8,D=4,B=8`，条形 ≈166）。下：`GPT-2 on 512 GPU nodes, B̂ = 512`，配置 `W=8,D=64,B=1` / `W=16,D=32,B=1` / `W=32,D=16,B=1,R` / `W=64,D=8,B=1,R`，横轴 40–160；模型选中 `W=16,D=32`（≈130），实测最优是 `W=64,D=8`（≈145） | 不采用（性能模型校准图，服务 Chimera 的自动配参；正文结论「模型误差 <10%」用文字即可） | — |
| `186283b449ededab330070695a917e9a3772736561731e170d6a0241c12c1b5e.jpg` | 238 | Figure 14 | 真插图 | 分组柱状图。横轴 `16 nodes` / `32 nodes` / `64 nodes`，纵轴 `Throughput (sequences/s)` 0–400。六个方案（图例带配置）：`PipeDream (D=8, B̂=[24, 96])`、`PipeDream-2BW (D=4, B=16, R)`、`GPipe (D=4, B=4, R)`、`GEMS (D=4, B=32)`、`DAPPLE (D=4, B=4)`、`Chimera (D=4, B=8)`。可读：64 nodes 上 Chimera ≈340 最高、PipeDream-2BW ≈290、GPipe ≈205、DAPPLE ≈275、GEMS ≈140、PipeDream ≈170 | 不采用（弱扩展结果图，与 Figure 15 信息重叠；若只保留一张弱扩展图，建议用 Figure 15） | —（备选） |
| `6d11554c6028e208cce6703f58405def9ed48b762925dafc56a4dcc6f43c220f.jpg` | 241 | Figure 15 | 真插图 | 分组柱状图。横轴 `512 nodes` / `1024 nodes` / `2048 nodes`，纵轴 `Throughput (sequences/s)` 0–500。图例：`PipeDream (D=8, B̂=[128, 512], R)`、`PipeDream-2BW (D=16, B=1, R)`、`GPipe (D=[8,16], B=1, R)`、`GEMS (D=8, B=2)`、`DAPPLE (D=16, B=1, R)`、`Chimera (D=32, B=1)`。可读：2048 nodes 上 Chimera ≈510 最高，PipeDream-2BW ≈440、DAPPLE ≈370、GPipe ≈360、PipeDream ≈255、GEMS ≈215 | **建议采用** | §4 / §9（Chimera 的招牌结果：GPT-2 1.3B 在 2048 节点上比同步方案快 1.38x–2.34x、比异步方案快 1.16x–2.01x。用来锚定「双向流水线在 DualPipe 之前已被大规模验证」并给出量级，避免把双向布局写成 DualPipe 的发明） |
| `f1eaf2cedca3c0ef8b28d9dbdc62624ddee9d391b49678181ee906bf522de2b9.jpg` | 252 | Figure 16 | 真插图 | 分组柱状图。横轴 `2x8 = 16 GPUs` / `4x8 = 32 GPUs`，纵轴 `Throughput (sequences/s)` 0–100。图例：`PipeDream (D=4, B̂=[16, 32])`、`PipeDream-2BW (D=4, B=4)`、`GPipe (D=4, B=2, R)`、`GEMS (D=[4, 8], B=8)`、`DAPPLE (D=4, B=2)`、`Chimera (D=[4, 8], B=4)`。可读：32 GPUs 上 Chimera ≈74 最高、PipeDream-2BW ≈71、DAPPLE ≈69、GPipe ≈60、PipeDream ≈40、GEMS ≈31 | 不采用（V100 小集群复核，结论与 Figure 14/15 同向且更弱） | — |
| `650f7f59e0fe3a03b740ed8d0eea463c0bf968be42eca8fe68d9d39773b0c20a.jpg` | 261 | Figure 17 | 真插图 | 折线图。横轴 `Mini-batch size (B̂)` 0–4096，纵轴 `Throughput (sequences/s)` 0–250。8 条曲线：`PipeDream (B̂=48)`（单点 ≈92）、`PipeDream-2BW (B=[16,32], R)`、`GPipe (B=[4,8], R)`、`GEMS (B=32)`、`DAPPLE (B=[4,8])`、`Chimera (direct, B=8)`、`Chimera (doubling, B=8, R)`、`Chimera (halving, B=4)`；右上文字 `async using stale weights`。可读：B̂=4096 时 PipeDream-2BW ≈240、`Chimera (direct)` ≈235、`Chimera (halving)` ≈225、DAPPLE ≈220、GPipe ≈200、GEMS ≈115 | 不采用（大 mini-batch 场景；若 §6 要论证「气泡被摊薄后同步方案追平异步方案」，此图有用，但 Chimera 不是本文主角） | —（备选） |
| `024c6eec938373c13b1223d82f4dca457440447117a4b2e42d582bdb6c733577.jpg` | 264 | Figure 18 | 真插图 | 折线图。横轴 `Mini-batch size (B̂)` 0–2048，纵轴 `Throughput (sequences/s)` 0–250。曲线：`PipeDream (B̂=128, R)`（单点 ≈77）、`PipeDream-2BW (B=1, R)`、`GPipe (B=1, R)`、`GEMS (B=2)`、`DAPPLE (B=1, R)`、`Chimera (direct, B=1, R)`、`Chimera (doubling, B=1, R)`、`Chimera (halving, B=1)`。可读：B̂=2048 时 `Chimera (doubling)` ≈223 > `Chimera (direct)` ≈198 > PipeDream-2BW ≈190 > GPipe ≈186 > DAPPLE ≈172 > GEMS ≈93 | 不采用（同上，与 Figure 17 成对） | —（备选） |
| `7061f0ac64401b2a757e20e22ea88bc9dae761ca0ff766cf26d7e3b33aad4995.jpg` | 273 | Figure 19 | 真插图 | 两组分组柱状图：左 `W=2, D=32`、右 `W=4, D=16`；纵轴 `Throughput (sequences/s)` 0–40；每组 5 根柱，柱顶文字 `1 pipe` / `2 pipes` / `4 pipes` / `8 pipes` / `16 pipes`（颜色由浅到深）。可读：`W=2, D=32` 下 `4 pipes` ≈29 最高（`1 pipe` ≈22、`16 pipes` ≈23）；`W=4, D=16` 下 `2 pipes` ≈34 最高（`1 pipe` ≈28、`8 pipes` ≈27） | 不采用（f>1 的收益评估，结论是「默认 2 条管线最好」；与本文 §4/§5 讨论的布局维度关系不大） | — |
| `8a27caa488bfaad8f14acc4c549da602c94a8bd3aa04bfd9dae25f15eb598090.jpg` | 62（md 以 HTML `<table>` 呈现，正文第 58 行引用） | Table 1 | 表格截图 | 两列符号表。左列：`D`、`W`、`P`、`B`、`N`、`B̂`、`M_θ`、`M_a`；右列：The number of pipeline stages (depth) / The number of replicated pipelines (width) for data parallelism / The number of workers (= W * D) / Micro-batch size / The number of micro-batches executed by each worker within a training iteration / Mini-batch size (= B * N * W) / Memory consumption for the weights of one stage / Memory consumption for the activations of one stage | 可转写为 Markdown 表格，不直接引图 | §2（最小概念模型：Chimera 的 D/W/B/N/B̂ 与本文 F/I/W/B_full/C 记号必须显式对齐，否则气泡率对比会串口径） |
| `e4347bf9fa8f48ab740eb92e5198bf8e0a645d459950bf841f9f426928a1f9f8.jpg` | 70（md 以 HTML `<table>` 呈现，正文第 58 行引用） | Table 2 | 表格截图 | 五列对照表，**带 👍/👎 手势图标**（md 的 HTML 版把图标丢了）。列：`Pipeline Schemes`、`Bubble Ratio`、`Weights Memory`、`Activations Memory`、`Convergence Friendly`。可读内容：PipeDream `≈0` / `[M_θ, D*M_θ]` / `[M_a, D*M_a]` / Asynchronous；PipeDream-2BW `≈0` / `2M_θ` / `[M_a, D*M_a]`；GPipe `(D−1)/(N+D−1)` / `M_θ` / `N*M_a` / Synchronous；GEMS `≈(D−1)/(D+½)` / `2M_θ` / `M_a`；DAPPLE `(D−1)/(N+D−1)` / `M_θ` / `[M_a, D*M_a]`；Chimera `(D−2)/(2N+D−2)` / `2M_θ` / `[(D/2+1)M_a, D*M_a]` | 可转写为 Markdown 表格，不直接引图 | §6（气泡/权重显存/激活显存三个公式的横向口径表。Chimera 的 `(D−2)/(2N+D−2)` 与 `2M_θ` 正是 DualPipe 气泡/显存公式的直接前身，必须在同一口径下对比；`Weights Memory` 列也是「参数版本」问题的原始记录） |
| `8c4e7ed69cea338918fd9a94b06ee723f90e126de70741065c366ddf24ece653.jpg` | 183（md 以 HTML `<table>` 呈现，正文第 181 行引用） | Table 3 | 表格截图 | 两列表格：`Model Replicas` = `2f`；`Bubble Ratio` = `(D − 2f)/(2fN + D − 2f)`；`Weights Memory` = `2f * M_θ`；`Activations Memory` = `[(D − D/2f + 1)M_a, D * M_a]` | 可转写为 Markdown 表格，不直接引图 | §6（Table 2 从 f=1 推广到 2f 条管线的闭式关系：气泡随管线数下降、权重显存随管线数线性上升 —— 「多管线换气泡」这一权衡在 DualPipe 之前就有公式） |
| `31651caef618a311c91f684124fa502fc920fdf5c8b55229d518da73d8efe6c1.jpg` | 202（md 以 HTML `<table>` 呈现，正文第 198 行引用） | Table 4 | 表格截图 | 四列表格：`Networks` / `Layers` / `Parameters` / `Mini-batch size`，两行数据：`Bert-48 / 48 / 669,790,012 / >=256`、`GPT-2 / 64 / 1,389,327,360 / >=512` | 可转写为 Markdown 表格，不直接引图 | §9（若正文要引 Chimera 的实验设置，转写这 2 行即可；截图本身无引图价值） |
| `95e8d4b9d1d0b589b1fe027afeaff3f979b77854c0d0f6bd5a92febc9352c546.jpg` | 158（md 第 157–159 行同一公式，式 (1)） | 式 (1)：单迭代运行时间模型 | 公式截图 | 两行公式截图：`T = (F_t + Comm_p2p)C_f + (B_t + Comm_p2p)C_b + max{Comm_unoverlapped(i) : i ∈ [0, D−1]}.` | 不采用（公式截图，非插图） | — |
| `b0fc08dd49bb017c8113eed0672252db8462fba697bfb5c55a7f5ba8dcf6b03e.jpg` | 150（md 第 149–151 行同一公式） | 式：allreduce 代价 | 公式截图 | 单行公式截图：`Comm_allreduce = 2(log₂ r)α + 2(r − 1)βL/r` | 不采用（公式截图，非插图） | — |
| `d706e16f470b7de75812bdb52bf54fdc1571f65798acedaddad2e1793276568e.jpg` | 47（md 第 46–48 行同一公式） | 式：mini-batch SGD 梯度 | 公式截图 | 单行公式截图：`g_t = (1/b) Σ_{i=0}^{b} ∇ℓ(w_t, x_i, y_i).` | 不采用（公式截图，非插图） | — |

## 采用图的正文引用块（可直接粘贴）

以下 8 张为建议采用。落地时把源图从 `references/papers/chimera/images/<原名>` 复制到 `torch/dualpipe/pics/<建议文件名>`（`pics/` 目录当前不存在，需新建；相对路径按 `deep-dive.md` 所在目录 `torch/dualpipe/` 计算）。所有文件扩展名沿用原文件的 `.jpg`。

### 1. `chimera-fig3-bidirectional-pipeline-schedule.jpg`（源 `270ff45471492c201af79e28da9e509b7beb940b080dd19a299f9cf3930b8471.jpg`，第 81 行）

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/chimera-fig3-bidirectional-pipeline-schedule.jpg" alt="Chimera Figure 3：model replica0 的 down pipeline（P0-stage0 … P3-stage3，微批 0/1）与 model replica1 的 up pipeline（P0-stage3 … P3-stage0，微批 2/3）分别用 1F1B 调度后合并为 Chimera，右下为合并后的 4 行整体时间轴（编号 0 1 2 2 3 3 0 1，右端 flush），下注 Chimera (backward is 2x workload of forward)" style="width: 100%;">
</div>

> **图片来源**：Chimera: Efficiently Training Large-Scale Neural Networks with Bidirectional Pipelines（arXiv 2107.06925）Figure 3，§3.1 Bidirectional Pipelines。本地副本：`references/papers/chimera/chimera.md` 第 81 行引用图。

### 2. `chimera-fig2-pipeline-schemes-comparison.jpg`（源 `aee6624ecf6b7a7091069e7856d11eadb156d52b3be993e54e78783904dcf06b.jpg`，第 74 行）

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/chimera-fig2-pipeline-schemes-comparison.jpg" alt="Chimera Figure 2：六种流水线调度（PipeDream、PipeDream-2BW、GPipe、GEMS、DAPPLE、Chimera）在同一 Time 轴上对照，P0-P3 四行，左侧分组标签 asynchronous with stale weights 与 synchronous, convergence-friendly，每行右端 flush 竖线；右侧小柱标 M_θ for PipeDream、M_θ for PipeDream-2BW、M_a for both；图例区分 Bubble 与 model replica0 x / model replica1 y" style="width: 100%;">
</div>

> **图片来源**：Chimera: Efficiently Training Large-Scale Neural Networks with Bidirectional Pipelines（arXiv 2107.06925）Figure 2，§2 BACKGROUND AND RELATED WORK。本地副本：`references/papers/chimera/chimera.md` 第 74 行引用图。

### 3. `chimera-fig4-gradient-sync-overlap.jpg`（源 `de1d0f4b487616c17dca4bd60c8036b284110c73c93d602f7a1132433644911d.jpg`；md 未引用图片本体，画面以 HTML 表格落在第 114 行、子图注在第 116 行）

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/chimera-fig4-gradient-sync-overlap.jpg" alt="Chimera Figure 4：(a) 所有本地计算结束后再同步梯度（S3/S0 等红格集中排在时间轴末端）；(b) eager 梯度同步（S3/S0 提前插在中段编号之间，利用气泡发起 allreduce）；图例 S_i = Gradients synchronization for stage_i" style="width: 100%;">
</div>

> **图片来源**：Chimera: Efficiently Training Large-Scale Neural Networks with Bidirectional Pipelines（arXiv 2107.06925）Figure 4，§3.2 Communication Scheme。本地副本：`references/papers/chimera/chimera.md` 第 114、116 行残留（MinerU 未落图片引用，图源见 `references/papers/chimera/images/de1d0f4b487616c17dca4bd60c8036b284110c73c93d602f7a1132433644911d.jpg`）。

### 4. `chimera-fig8-four-pipelines-combination.jpg`（源 `5242db9131a634574f7d346b98c5002220dea4489606f4fd0002c5485c850902.jpg`，第 170 行）

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/chimera-fig8-four-pipelines-combination.jpg" alt="Chimera Figure 8：f=2 时四条 8-stage 管线（model replica0/1 为 down pipeline0/1，replica2/3 为 up pipeline0/1，stage 映射依次平移 4）合并后的 8 行时间轴，右下标注 Chimera (a combination of four pipelines)，右端 flush" style="width: 100%;">
</div>

> **图片来源**：Chimera: Efficiently Training Large-Scale Neural Networks with Bidirectional Pipelines（arXiv 2107.06925）Figure 8，§3.6 Generalize to More than Two Pipelines。本地副本：`references/papers/chimera/chimera.md` 第 170 行引用图。

### 5. `chimera-fig7-forward-doubling-concatenation.jpg`（源 `806f053dc964a454f6fad23f76b19f9a1520884ced549078dd971af07911efc6.jpg`，第 163 行）

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/chimera-fig7-forward-doubling-concatenation.jpg" alt="Chimera Figure 7：N>D 时拼接多个调度单元的四种形态 ——(a) N=2D 的两个基本单元相加；(b) Direct concatenation 存在 intermediate bubbles；(c) forward doubling 每格含两个微批、无中间气泡；(d) 去掉前半气泡后的最终调度，橙色箭头标 p2p 位置" style="width: 100%;">
</div>

> **图片来源**：Chimera: Efficiently Training Large-Scale Neural Networks with Bidirectional Pipelines（arXiv 2107.06925）Figure 7，§3.5 Scale to More Micro-Batches。本地副本：`references/papers/chimera/chimera.md` 第 163 行引用图。

### 6. `chimera-fig1-bubble-memory-throughput-summary.jpg`（源 `c77254c3940819fcd0b801c19bd17deb1843c345ba140047eb30d43885fe4633.jpg`，第 37 行）

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/chimera-fig1-bubble-memory-throughput-summary.jpg" alt="Chimera Figure 1：GPT-2 在 2048 GPU 节点上的三联柱状图 —— Bubble Ratio (%)：Chimera ≈32、GPipe ≈45、DAPPLE ≈49、GEMS ≈83；Peak Memory Cost (GB)：GPipe/PipeDream-2BW/DAPPLE 标 OOM，Chimera ≈15.2；Throughput (sequences/s)：Chimera ≈505 最高，并标 2.01x/1.16x/1.42x/2.34x/1.38x 的加速比" style="width: 100%;">
</div>

> **图片来源**：Chimera: Efficiently Training Large-Scale Neural Networks with Bidirectional Pipelines（arXiv 2107.06925）Figure 1，§1 INTRODUCTION。本地副本：`references/papers/chimera/chimera.md` 第 37 行引用图。

### 7. `chimera-fig9-memory-consumption-distribution.jpg`（源 `9b67cf40eb07954f43df56e69320656aa321a7299c8af0c4e48ea56d12975fbe.jpg`，第 195 行）

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/chimera-fig9-memory-consumption-distribution.jpg" alt="Chimera Figure 9：Bert-48 与 GPT-2 在 6 种配置下、32 个 GPU 节点上的 Memory consumption (GB) 分布（箱线+散点，红圈=最大值、蓝圈=最小值，多处标 OOM）；Chimera 与 GEMS 的箱体最矮最稳，PipeDream-2BW 随 stage 变粗 OOM，DAPPLE 首 worker 峰值高" style="width: 75%;">
</div>

> **图片来源**：Chimera: Efficiently Training Large-Scale Neural Networks with Bidirectional Pipelines（arXiv 2107.06925）Figure 9，§4.1 Memory Consumption。本地副本：`references/papers/chimera/chimera.md` 第 195 行引用图。

### 8. `chimera-fig15-weak-scaling-gpt2.jpg`（源 `6d11554c6028e208cce6703f58405def9ed48b762925dafc56a4dcc6f43c220f.jpg`，第 241 行）

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/chimera-fig15-weak-scaling-gpt2.jpg" alt="Chimera Figure 15：GPT-2 在 Piz Daint 上 512/1024/2048 节点的弱扩展吞吐（sequences/s）；2048 节点上 Chimera (D=32, B=1) ≈510 最高，PipeDream-2BW ≈440、DAPPLE ≈370、GPipe ≈360、PipeDream ≈255、GEMS ≈215" style="width: 75%;">
</div>

> **图片来源**：Chimera: Efficiently Training Large-Scale Neural Networks with Bidirectional Pipelines（arXiv 2107.06925）Figure 15，§4.2.3 Comparison with the Best Performance。本地副本：`references/papers/chimera/chimera.md` 第 241 行引用图。

## 存疑与分歧

1. **Figure 2 / Figure 3 的归属已确认，草稿的疑问可以关闭。** `aee6624e…jpg` = **Figure 2**（六种调度的统一时间轴对照，图内自带 `Bubble`/`model replica0 x`/`model replica1 y` 图例框与 `Will be discussed in detail in Section 3.1.` 底注）；`270ff454…jpg` = **Figure 3**（`Model replicas and bidirectional pipelines scheduling of Chimera`，即 down/up 两条管线如何合并成双向调度）。判定依据是读图内容与 md 第 75、84 行两条图注逐字对应，而非文件顺序。
2. **md 第 82 行的图例文字错位到了 Figure 3 之后。** 第 82 行内容 `x x Forward and backward passes of model replica0 for micro-batch x / y y … model replica1 for micro-batch y / Bubble` 实际是 **Figure 2 内部图例框**的 OCR 文本，MinerU 把它排在了第 81 行（Figure 3 图片）之后；Figure 3 的真正图注是第 84 行。引用 Figure 2 时不要用第 82 行当图注。
3. **Figure 4 在 md 中“消失”了：图片本体存在但没人引用。** `de1d0f4b…jpg` 是完整的 Figure 4（含 (a)(b) 两面板），而 md 第 114 行把它画面里的表格部分转成了 HTML `<table>`（只有 (b) 面板的一部分编号与 `S3`/`S0`），第 116 行留下子图注 `(b) Eager gradient synchronization for deeper overlapping`。**只看 md 的图片引用会漏掉这张图**，核对时必须回到 `images/` 目录。这也是「建议采用」里唯一一张没有正常 md 引用行的图。
4. **任务里问的「权重暂存 / 参数版本」在本资料中没有独立插图。** 26 张逐一读完确认：没有任何一张专门画权重版本暂存的时间轴。相关信息只有三处——(i) Figure 2 右侧小柱 `M_θ for PipeDream` / `M_θ for PipeDream-2BW` / `M_a for both`；(ii) Table 2 的 `Weights Memory` 列（`[M_θ, D*M_θ]`、`2M_θ`）；(iii) Table 3 的 `2f * M_θ`。若要讲「参数版本」，只能引 Figure 2 或转写 Table 2/3；真正的「权重版本双缓冲」时序图在本仓库的 `references/papers/pipedream-2bw/` 里更可能找到。
5. **「双向流水线整体时间轴」以 Figure 3 为准，Figure 2 是横向对照。** Figure 3 单独画出了 down/up 两个方向各自的 1F1B 以及合并后的双向时间轴（含 `backward is 2x workload of forward` 的提示），是回答「两个方向的数据流如何交错填空」最直接的图；Figure 2 里 Chimera 只占 6 行中的 1 行，虽然信息量更大但不适合当唯一主图。建议 §4 先用 Figure 2 立「前史」，再用 Figure 3 讲机制。
6. **图片总数与引用数不对称，已逐项对账。** `ls | wc -l` = 26，md 图片引用 18 处（18 个唯一文件），未引用的 8 个 = Figure 4（`de1d0f4b`）+ Table 1–4（`8a27caa4`、`e4347bf9`、`8c4e7ed6`、`31651caef`）+ 公式 3 张（`95e8d4b9`、`b0fc08dd`、`d706e16f`）。图号 1–19 连续无缺号；表格的「引用行」列是按 md 中同一内容的 HTML `<table>` 所在行反推（例如 Table 1 ↔ 第 62 行），因为 MinerU 没有为它们生成图片引用。
7. **Table 2 的截图比 md 里的 HTML 版信息更多。** 截图保留了 👍/👎 手势图标（用于标记各方案在 Bubble Ratio / Weights Memory / Activations Memory / Convergence Friendly 四列上的优劣），而 md 第 70 行的 HTML 表格把这些图标丢掉了。若要转写这张表，只能得到不带图标的数据，需在正文另加说明。
8. **边界候选（本次判为不采用，若章节需要可回取）**：`b6584fa2`（Figure 5，双向流水线 × 数据并行副本，若 §9 要谈双向布局与 DP/EP 的组合）；`1eb74998`（Figure 6，可重叠自由区 region (a)(b)(c)，若 §6 要论证「气泡区间可被通信填充」）；`650f7f59` / `024c6eec`（Figure 17/18，大 mini-batch 下同步方案追平异步方案，若 §6 要讨论「气泡被摊薄后的比较口径」）。
