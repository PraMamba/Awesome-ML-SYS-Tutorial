# Megatron-LM GPU Clusters（arXiv 2104.04473v5）图片逐张核对

核对对象：`torch/dualpipe/references/papers/megatron-lm-gpu-clusters/images/` 下的全部 32 张 `.jpg`。
MinerU 产物：`torch/dualpipe/references/papers/megatron-lm-gpu-clusters/megatron-lm-gpu-clusters.md`（494 行，19 处图片引用、2 处 HTML 表格）。
核对方式：32 张全部用 `read_image` 逐张打开读图，下表「内容」列全部为**看图后**的事实描述（含图中可读文字与数值），未依据文件名或 md 顺序推断。

## 汇总

- `images/` 实际文件数（`ls | wc -l`）：32
- 本清单覆盖行数：32
- 真插图：19；公式截图：11；表格截图：2；页眉/装饰/碎片：0
- 建议采用：11 张
- 分类明细：
  - 真插图 19 张 = Figure 1、2、3、4、5(a)、5(b)、6、7、8、9、10、11、12、13、14、15、16、17、18（Figure 5 与 Figure 9 各由一个文件承载全部子图）
  - 公式截图 11 张 = `7736c2dd`、`d0a83157`、`7be6bf7c`、`5f5f7cd9`、`c7246376`、`50a7f1f7`、`5a4b5615`、`28651782`、`fc355189`、`36265c3c`、`d5abda27`
  - 表格截图 2 张 = Table 1（`492129ee`）、Table 2（`4e6647c7`）
- 关键事实 1：19 张真插图**全部**在 md 中被引用；13 张未被 md 引用的文件 = 2 张表格截图 + 11 张公式截图，即 MinerU 没有把这些公式/表格塞回正文（正文里公式以 LaTeX 呈现、表格以 HTML `<table>` 呈现），只把图抠了出来。因此本目录**不存在**「公式截图被误当插图引用」的风险，唯一风险来自文件名哈希不可读。
- 关键事实 2：草稿把 `648f402f…jpg` 当成 Figure 4 是**错的**。读图结果：该图是「Transformer layer #1/#2 + Tensor MP partition #1/#2 + Pipeline MP partition #1/#2」，即 **Figure 2**；真正的 Figure 4（default vs interleaved 1F1B）是 `2a3ac338…jpg`（详见「存疑与分歧」第 1 条）。

## 逐张清单

| 文件名 | md 引用行 | 图号/表号 | 类别 | 内容（读图后的事实描述，含图中可读的关键标注/数值） | 结论 | 建议章节 |
| --- | --- | --- | --- | --- | --- | --- |
| `df104f4867c0f61455db883b3e686cea2522cc3e670b231a1a2810c4c615f27c.jpg` | 13 | Figure 1 | 真插图 | 单折线图。横轴 `Year`（2018–2021），纵轴 `Number of parameters (in billions)` 对数刻度 10⁻²–10³。标注点与文字：`ELMo (94M)`、`BERT-L (340M)`、`GPT-2 (1.5B)`、`Megatron-LM (8.3B)`、`Turing-NLG (17.2B)`、`GPT-3 (175B)`；蓝色实线为实测点连线，红色虚线为指数趋势外推 | 不采用（引言背景图，与「气泡 / 调度 / 双向布局」主线无承接关系） | — |
| `648f402f48c10c7b096f43f9b5f7947a16536f94118d70eba2b3d9b7c593fdd7.jpg` | 56 | **Figure 2**（草稿误标 Figure 4） | 真插图 | 左右两块对比结构图，标题 `Transformer layer #1` / `Transformer layer #2`，中间绿色粗箭头。每块内有三个虚线分组：`Tensor MP partition #1`、`Tensor MP partition #2`（各自含 Attention 与 MLP 的子方块）、以及最外层 `Pipeline MP partition #1` / `Pipeline MP partition #2`。左块含 `f`、`g`（行/列并行 GEMM）、`Dropout` 等算子方块，右块为同样的分块拓扑 | **建议采用** | §8（框架侧对照：PTD-P 把 tensor MP 放在节点内、pipeline MP 放在跨节点，是「Megatron combined 1F1B」的结构前提；也是解释「层内 all-reduce vs 层间 P2P」两种通信语义的对照图） |
| `769882e4379d4df19e671b3a12388798ff2df0563a290578b3bbdbd972141ce5.jpg` | 59 | Figure 3 | 真插图 | 4 行 `Device 1`–`Device 4` 的调度时间轴，左下角 `Time →`。蓝格=Forward Pass、绿格=Backward Pass、灰格=idle。全部 forward 先跑：Device 1 先 `1 2 3 4 5 6 7 8` 蓝格，随后是绿格 `1 2 3 4 5 6 7 8`；第二轮编号 `9 10 11 12 13 14 15 16` 重复同样形态；Device 4 起始最晚、尾部灰格最多；一条竖黑线之后转下一轮 | **建议采用** | §1（两种等待：先给出「pipeline bubble = 整段尾部空闲」的最直观形态）；§3（GPipe all-forward-all-backward 作为 1F1B 的对照基线） |
| `2a3ac338c206e25875fba03f81ced8e175be26545684dd22e150a7c9a8c0dd3d.jpg` | 62 | Figure 4 | 真插图 | 上下两面板。上：default/non-interleaved 1F1B，4 行 `Device 1`–`Device 4`，编号 `1 2 3 4` 起步、稳态段每行 F/B 交替（如 Device 3 出现 `1 2 3 5 4 6 5 6 7 8 7 8` 的穿插），中段有 `1 2 2 3 3 4 4 5 5 6 6 7 7 8 8` 长串，竖黑线后接 `9 10 11 12 12 13 14 13 14 15 16 9 10 11` 等第二轮。下：interleaved，两面板之间有一条粗黑箭头 + 文字 `Assign multiple stages to each device`；下方 4 行格内蓝/浅蓝、绿/深绿**四色**交替（v=2 的两个 model chunk），编号如 `1 2 3 4 1 2 3 4 5 6 5 6 7 8 1 2 …`，竖黑线明显更早；图例 `Forward Pass`（蓝）/ `Backward Pass`（绿），`Time →` | **建议采用**（本目录**最高优先级**的一张） | §3（1F1B → interleaved 1F1B / VPP 前身的核心对照图：同一 batch、同一 p，flush 提前、气泡缩小，代价是每设备多持一个 chunk）；§4 开篇也可复用 |
| `f034002f8cc83a1eda8e869907b3645c9520a10a2656a6d53959f497d820391c.jpg` | 125 | Figure 5(a) | 真插图 | 标题 `Y = GeLU(XA)` 与 `Z = Dropout(YB)` 左右两段。左段：`X` → `f`（绿）→ `A = [A₁, A₂]` 双列并行 → `XA₁`/`XA₂` → `GeLU` → `Y₁`/`Y₂`。右段：`Y₁B₁`/`Y₂B₂` → `g`（绿）→ `Dropout` → `Z`，下方标 `B = [B₁; B₂]`（行切分） | 不采用（张量并行 MLP 内部细节，属 Megatron-LM 2019 论文范畴；本文章讲的是层间流水，Figure 2 已足够表达「tensor MP 在节点内」） | —（若 §8 要展开 `f`/`g` 共轭算子语义，可回取；md 第 129 行文字已给出结论） |
| `bf616888f0af48fce20de2a5f397491cef4e0f8d551cd9ec31e76a766b3e091a.jpg` | 128 | Figure 5(b) | 真插图 | 标题 `Y = Self-Attention(X)` 与 `Z = Dropout(YB)`。左段：`X` → `f`（绿）→ `K₁/Q₁/V₁` 与 `K₂/Q₂/V₂` 两路 → `⊗` → `Softmax` → `Dropout` → `⊗` → `Y₁`/`Y₂`，图内文字 `Split attention heads ← Q = [Q₁,Q₂]; K = [K₁,K₂]; V = [V₁,V₂]`。右段与 5(a) 相同。图下方小标题 `(b) Self-Attention.` | 不采用（同上，多头切分细节与本文主线无关） | — |
| `0b96d327cec43357cc2271be149d6d69f90bbc9ef277dbeb3ecea526801ee5dc.jpg` | 155 | Figure 6 | 真插图 | 折线图。横轴 `Data-parallel size (d)`，取值 1/2/4/8/16/32/64（对数轴）；纵轴 `Pipeline bubble size`，0.00–1.00。四条曲线（图例原文）：`n=32, b'=32`（蓝圆点）、`n=32, b'=128`（橙菱形）、`n=128, b'=128`（绿三角）、`n=128, b'=512`（红方块）。可读数值：n=32,b'=32 从 d=1 的 ≈1.00 降到 d=32 的 ≈0.00；n=128,b'=512 全程 ≈0.25 → 0.13 | **建议采用** | §6（气泡公式重算与比较口径：解析式 (n−d)/b′ 的曲线形态，说明「加大 d 会压气泡」的边界条件；也用于 §9 选型误区「以为 pipeline size 越大越好」） |
| `252b2b41169960fb7a2432b2296e29bd91d6867706d32821450d66052823956b.jpg` | 174 | Figure 7 | 真插图 | 单折线图。横轴 `Microbatch size`（1/2/4/8/16），纵轴 `Achieved teraFLOP/s per GPU`（0–100）。曲线从 microbatch=1 的 ≈67 单调升到 16 的 ≈92，8→16 段已近似平台 | 不采用（单 GPU 上「microbatch 越大越吃满算力」的结论在 Figure 16 的完整配置里被更完整地表达，二者信息重叠） | —（若 §9 要区分「单卡算力效率」与「整条流水线吞吐」两个口径，可回取） |
| `82614767551c90653d6c4b93bc1e2796de2f46b5dd4d31493039f9c12c4c44a2.jpg` | 195 | Figure 8 | 真插图 | 折线图。横轴 `Microbatch size`（1/2/4/8/16），纵轴 `Normalized throughput`（0.00–1.25）。两条曲线：`Batch size = 128`（蓝圆点）在 4 处见顶 ≈1.09 后回落到 16 的 ≈0.75；`Batch size = 512`（橙菱形）在 4 处见顶 ≈1.22，16 处仍 ≈1.10 | 不采用（与 Figure 16 同为「microbatch size 影响吞吐」的实验，信息重叠；其解析模型含义已在 md 第 192 行式 (1) 与正文说明中） | —（若 §6 要展示「解析模型能复现实测拐点」，此图是唯一的模型校准图，可回取） |
| `8810fe26b6d870568e66bdf782f10b52532a8e9dcb34617b115e571368f44869.jpg` | 214 | Figure 9(a) + 9(b)（两子图在同一文件内） | 真插图 | 单张宽图，左右两面板。左：`InfiniBand` 下方两组节点（`1`/`2` 带浅蓝 NVLink 竖条，`3`/`4` 带深蓝竖条），节点间两条黑色箭头，上下各有一条等长红色粗条 = 冗余发送的同一张量；左侧文字 `NVLink` + 弧线箭头连接 1、2。右：`Scatter of`（上方浅红短条）与 `All-gather of`（上方一长一短红条），节点间箭头由长红条变为半长条 | 不采用（DGX A100 八张 IB 卡专属的 scatter/gather 工程优化，与 DualPipe 的 P2P 排布主线无关；§8 若讲「跨节点通信量」可另作脚注） | —（md 第 215 行给出子图说明文字，第 225 行给出 1/t 的量化结论） |
| `675b8a1b46a8a2d2b3a8ae152a874d90945bd0bf67ba277fa4f6f1e15468818c.jpg` | 279 | Figure 10 | 真插图 | 折线图。横轴 `Number of GPUs`（768/1152/1536/1920），纵轴 `Achieved teraFLOP/s per GPU`（0–200）。四条曲线：`ZeRO-3, 175B`（蓝虚线圆点，≈145 → ≈44）、`ZeRO-3, 530B`（蓝实线菱形，≈140 → ≈48）、`PTD-P, 175B`（橙虚线三角，≈152 → ≈141）、`PTD-P, 530B`（橙实线方块，≈170 → ≈160） | 不采用（ZeRO-3 对照属「无模型并行」路线，与流水线调度/双向布局主线无承接；其结论「少跨节点通信」已由 Figure 13/14 的更贴近主题的图表达） | — |
| `59d12b477b0866d52a1dc2b616057aa4566e927337afb0bf861ff670b6142c67.jpg` | 296 | Figure 11 | 真插图 | 折线图。横轴 `Pipeline-parallel size`（1/2/4/8），纵轴 `Achieved teraFLOP/s per GPU`（0–200）。`Batch size = 8`（蓝圆点）从 ≈162 单调降到 ≈88；`Batch size = 128`（橙菱形）从 ≈178 只降到 ≈160（近似平线） | **建议采用** | §6（气泡如何随 pipeline size 被摊薄：同一张图同时给出「小 batch 时 p 越大越亏、大 batch 时 p 几乎免费」两条曲线，是气泡公式 `(p−1)/m` 的实测口径）；§9（选型：p 的上限由 microbatch 数决定） |
| `334ef459964f9ba6db7a6a6b510aea873a95bf7b172004abad223516ca551de9.jpg` | 299 | Figure 12 | 真插图 | 折线图。横轴 `Batch size`（12/24/36/48/60），纵轴 `Achieved teraFLOP/s per GPU`（50–150）。`Non-interleaved`（蓝圆点）≈81 → ≈136；`Interleaved`（橙菱形）≈119 → ≈147。两线在 batch 增大时收敛（差距从 ≈38 缩到 ≈11） | **建议采用** | §3（interleaved 相对 1F1B 的收益量级 ≈10%，且随 batch 增大被通信量吃掉 —— 正是「为何还要 DualPipe」的动机铺垫）；§8 |
| `f3ea415d7b9cf5de4c13a9f73868a669dad7cbed910f54fcd60ae825abbf460c.jpg` | 302 | Figure 13 | 真插图 | 折线图。横轴 `(Pipeline-parallel size, Tensor-parallel size)`，取值 `(2,32) (4,16) (8,8) (16,4) (32,2)`；纵轴 `Achieved teraFLOP/s per GPU`（0–200）。`Batch size = 32`（蓝圆点）峰值在 `(8,8)` ≈142；`Batch size = 128`（橙菱形）峰值也在 `(8,8)` ≈166，两端 `(2,32)`/`(32,2)` 分别 ≈108/≈147 | **建议采用** | §8（框架侧对照：tensor MP 的度应等于节点内 GPU 数（8）—— 与 DualPipe 把 P2P 留在跨节点是同一套硬件约束）；§9（「极端 t 或极端 p」都属于误区） |
| `b6c8d7a05522215bc25eee68d6ba3879ad2c8128da45bb20d75dcaa435e8e3f0.jpg` | 313 | Figure 14 | 真插图 | 折线图。横轴 `(Pipeline-parallel size, Data-parallel size)`，取值 `(2,32) (4,16) (8,8) (16,4) (32,2)`；纵轴 `Achieved teraFLOP/s per GPU`（0–200）。`Batch size = 32`（蓝圆点）≈62 → ≈45 缓降；`Batch size = 512`（橙菱形）≈150 → ≈88 明显下降。两条线均随 pipeline size 增大而下降 | **建议采用** | §9（选型结论：pipeline parallelism 只用来「装下模型」，横向扩容应交给 data parallelism —— 这条正是 DualPipe/VPP 讨论里最常被搞反的一条） |
| `d3d57e1df1720f002db254b15db9830ababee51b143498d80ba462ba7ae4e14f.jpg` | 316 | Figure 15 | 真插图 | 折线图。横轴 `(Tensor-parallel size, Data-parallel size)`，取值 `(2,32) (4,16) (8,8) (16,4) (32,2)`；纵轴 `Achieved teraFLOP/s per GPU`（0–200）。`Batch size = 32`（蓝圆点）≈55 几乎水平；`Batch size = 128`（橙菱形）≈103 → ≈22；`Batch size = 512`（绿三角）≈127 → ≈21。三条线全部向右收敛到 ≈20 | 不采用（数据并行 × 张量并行的通信口径，与流水线气泡/双向布局主题无承接；同族的 Figure 13/14 已覆盖选型结论） | — |
| `231e06c3ea3e2972435388f81fb32418450cbacbacf6ea0d6192f92a18d78b46.jpg` | 319 | Figure 16 | 真插图 | 折线图。横轴 `Microbatch size`（1/2/4/8），纵轴 `Achieved teraFLOP/s per GPU`（0–200）。`Batch size = 128`（蓝圆点）≈153 → 峰值 ≈156（microbatch=2）→ ≈122；`Batch size = 512`（橙菱形）≈160 → 峰值 ≈172（microbatch=2）→ ≈152。图上最优 microbatch size 为 2 | **建议采用** | §9（常见误区：microbatch 越大越好 —— 实测在 (t,p)=(8,8) 下最优是 2；与 §6 的「气泡 ∝ 1/m」形成张力） |
| `d97fe77bf501c7b0c9c3abb20840c11650f208d6d377fdc9e458d2537fa28a76.jpg` | 334 | Figure 17 | 真插图 | 折线图。横轴 `Batch size`（1…256，对数轴），纵轴 `Throughput (sequences/second)`（0.0–10.0）。`Act. recomputation`（蓝圆点）从 ≈0.6 单调升到 ≈7.8；`W/o act. recomputation`（橙菱形）只画到 batch size=8（≈0.6 → ≈3.9）后终止 —— 即不重算时大 batch 直接 OOM 画不出来 | **建议采用**（全文唯一直接给出「激活重算 ↔ batch/显存」折算关系的图） | §6（显存公式重算：重算换来的不是吞吐而是「可用的 batch 上限」，气泡因 m 变大而缩小；口径统一时必须把「算力多跑一遍前向」与「气泡变小」两笔账分开算） |
| `12072e20fd8ce3cc1ba3e80ef1d6fdd4f0bb6cc54acff7cd478e515f81a8b1e2.jpg` | 337 | Figure 18 | 真插图 | 折线图。横轴 `Batch size`（12/24/36/48/60），纵轴 `Achieved teraFLOP/s per GPU`（50–150）。`Unoptimized`（蓝圆点）≈108 → ≈131；`Scatter/gather optimization`（橙菱形）≈119 → ≈147，全程领先 ≈11% | **建议采用** | §8（通信开销一侧的对照：interleaved 带来的额外 P2P 通信需要配套优化才划算；说明「调度更细 → 通信更密 → 必须有重叠/优化手段」，与 DualPipe 的 SM 划分动机同源） |
| `492129eeb72aff8b1054c01446739fc583bb8c13e9457f6efc905b6f0530d1e9.jpg` | —（md 未引用；正文以 HTML `<table>` 出现在第 257 行，图注在第 259 行） | Table 1 | 表格截图 | 11 列 × 11 行的截图表格，表头为 `Number of parameters (billion)`、`Attention heads`、`Hidden size`、`Number of layers`、`Tensor model-parallel size`、`Pipeline model-parallel size`、`Number of GPUs`、`Batch size`、`Achieved teraFLOP/s per GPU`、`Percentage of theoretical peak FLOP/s`、`Achieved aggregate petaFLOP/s`。数据行从 `1.7 / 24 / 2304 / 24 / 1 / 1 / 32 / 512 / 137 / 44% / 4.4` 直到 `1008.0 / 160 / 25600 / 128 / 8 / 64 / 3072 / 3072 / 163 / 52% / 502.0` | 可转写为 Markdown 表格，不直接引图 | §5 或 §8（弱扩展配置表：p 从 1 到 64、n 从 32 到 3072，是「pipeline size 与 GPU 数同步放大」的官方口径，可作为 §6 比较口径的基准表） |
| `4e6647c732ceed86a0326537a574acec8aabdcf850fce23cf10ee26ef0d4fea4.jpg` | —（md 未引用；正文以 HTML `<table>` 出现在第 292 行，图注在第 294 行） | Table 2 | 表格截图 | 8 列 × 13 行截图表格，含合并单元格。表头 `Scheme`、`Number of parameters (billion)`、`Model-parallel size`、`Batch size`、`Number of GPUs`、`Microbatch size`、`Achieved teraFLOP/s per GPU`、`Training time for 300B tokens (days)`。分两组：`ZeRO-3 without Model Parallelism`（174.6 / 529.6，`2560*` 行有星号脚注）与 `PTD Parallelism`（174.6 / 529.6）。可读数值：PTD 174.6B 在 384→1536 GPU 上 153/149/141 teraFLOP/s、84/43/23 天；ZeRO-3 对应 144/88/44 teraFLOP/s 与 90/74/74 天 | 可转写为 Markdown 表格，不直接引图 | §8（对照表：同一模型下 TP+PP+DP 组合与纯 ZeRO-3 的吞吐/训练时间对比；若文章要论证「模型并行不是为了省显存而是为了少跨节点通信」，这张表是原始出处） |
| `7736c2dd2a7181206a976cd61c34768d1047a383dd516fd8697fc154c0efb283.jpg` | 70（md 同式以 LaTeX 出现在 69–71 行） | 式：Bubble time fraction | 公式截图 | 单行公式截图：`Bubble time fraction (pipeline bubble size) = t_pb / t_id = (p − 1) / m.` | 不采用（公式截图，非插图） | — |
| `d0a831571d7fdd30e32889d878c25d7ebb68cda68b1e058b72a5afd3eee57388.jpg` | 150（md 第 149–151 行同一公式） | 式：气泡随 t 变化 | 公式截图 | 单行公式截图：`(p − 1)/m = (n/t − 1)/m.` | 不采用（公式截图，非插图） | — |
| `7be6bf7cff113985f7326191a47595f82aaef228b7c48884abc19a0dcb43a154.jpg` | 171（md 第 170–172 行同一公式） | 式：气泡随 d 变化 | 公式截图 | 单行公式截图：`(p − 1)/m = (n/d − 1)/(b′/d) = (n − d)/b′.` | 不采用（公式截图，非插图） | — |
| `5f5f7cd99e88ca1a373583aa1effbf760cb5b33b9847b11c1a099dfb0d1978e2.jpg` | 96（md 第 95–97 行同一公式） | 式：MLP 两段 | 公式截图 | 单行公式截图：`Y = GeLU(XA).   Z = Dropout(YB).` | 不采用（公式截图，非插图） | — |
| `c7246376bd5e6c0dcdcfdaceb2865a2cc2eb8833fc5a4274b1aea468139ef691.jpg` | 102（md 第 101–103 行同一公式） | 式：GeLU 逐列独立 | 公式截图 | 单行公式截图：`[Y₁, Y₂] = [GeLU(XA₁), GeLU(XA₂)].` | 不采用（公式截图，非插图） | — |
| `50a7f1f7ef170bde9cdb2db31b5ab5e838b5a0b9569954c1a39d3d27d7c0e925.jpg` | 110（md 第 109–111 行同一公式） | 式：B、Y 切分 | 公式截图 | 单行公式截图：`B = [B₁; B₂] ,  Y = [Y₁, Y₂].` | 不采用（公式截图，非插图） | — |
| `5a4b561596616ba7060032f94fe22f2b8699a530882013f7f3e6ba424a290e66.jpg` | 192（md 第 191–193 行，式 (1)） | 式 (1)：单 batch 计算时间 | 公式截图 | 单行公式截图：`(b′/b + p − 1) · (t_f(b) + t_b(b)).` | 不采用（公式截图，非插图） | — |
| `28651782c521a53f10d9d66b079eb2052f7591fe985233b1aa6710651f35e7ab.jpg` | 254（md 第 253–255 行，式 (2)） | 式 (2)：参数量 | 公式截图 | 单行公式截图：`P = 12lh²(1 + 13/(12h) + (V + s)/(12lh)).` | 不采用（公式截图，非插图） | — |
| `fc3551897e59bc0181505078cc5a897a16ca3c7f3c9f3faa270ebb421a2fbdd0.jpg` | 264（md 第 263–265 行，式 (3)） | 式 (3)：FLOPs | 公式截图 | 单行公式截图：`F = 96Bslh²(1 + s/(6h) + V/(16lh)).` | 不采用（公式截图，非插图） | — |
| `36265c3cd16188e83da739be664e0d6639d92e2b478fd63320f716e155a17652.jpg` | 399（附录，md 第 398–400 行） | 附录式：总 FLOPs | 公式截图 | 单行公式截图（带句点、无 `F =` 前缀）：`96Bslh²(1 + s/(6h) + V/(16lh)).` | 不采用（公式截图，非插图） | — |
| `d5abda272108e056d6e6e4757db9f222fdf7d9e629b164668a9f719de31273b0.jpg` | 274（md 第 273–275 行，式 (4)） | 式 (4)：端到端训练时间 | 公式截图 | 单行公式截图：`End-to-end training time ≈ 8TP/(nX).` | 不采用（公式截图，非插图） | — |

## 采用图的正文引用块（可直接粘贴）

以下 11 张为建议采用。落地时把源图从 `references/papers/megatron-lm-gpu-clusters/images/<原名>` 复制到 `torch/dualpipe/pics/<建议文件名>`（`pics/` 目录当前不存在，需新建；相对路径按 `deep-dive.md` 所在目录 `torch/dualpipe/` 计算）。所有文件扩展名沿用原文件的 `.jpg`。

### 1. `megatron-fig4-default-vs-interleaved-1f1b.jpg`（源 `2a3ac338c206e25875fba03f81ced8e175be26545684dd22e150a7c9a8c0dd3d.jpg`，第 62 行）

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/megatron-fig4-default-vs-interleaved-1f1b.jpg" alt="Megatron-LM Figure 4：上=default non-interleaved 1F1B（Device 1-4，micro-batch 1-16，稳态 F/B 交替，flush 竖线偏后）；下=interleaved 1F1B（Assign multiple stages to each device，蓝/浅蓝、绿/深绿四色两 chunk，flush 提前），蓝=Forward Pass，绿=Backward Pass，灰=idle" style="width: 100%;">
</div>

> **图片来源**：Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM（arXiv 2104.04473v5）Figure 4，§2.2.2 Schedule with Interleaved Stages。本地副本：`references/papers/megatron-lm-gpu-clusters/megatron-lm-gpu-clusters.md` 第 62 行引用图。

### 2. `megatron-fig3-gpipe-schedule-pipeline-bubble.jpg`（源 `769882e4379d4df19e671b3a12388798ff2df0563a290578b3bbdbd972141ce5.jpg`，第 59 行）

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/megatron-fig3-gpipe-schedule-pipeline-bubble.jpg" alt="Megatron-LM Figure 3：GPipe 调度，Device 1-4，先全部 forward（蓝，micro-batch 1-8）再全部 backward（绿），第二轮编号 9-16，尾部大片灰格=Devices idle/pipeline bubble" style="width: 100%;">
</div>

> **图片来源**：Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM（arXiv 2104.04473v5）Figure 3，§2.2.1 Default Schedule。本地副本：`references/papers/megatron-lm-gpu-clusters/megatron-lm-gpu-clusters.md` 第 59 行引用图。

### 3. `megatron-fig2-tensor-pipeline-mp-combination.jpg`（源 `648f402f48c10c7b096f43f9b5f7947a16536f94118d70eba2b3d9b7c593fdd7.jpg`，第 56 行）

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/megatron-fig2-tensor-pipeline-mp-combination.jpg" alt="Megatron-LM Figure 2：Transformer layer #1 与 #2 并列，每层内标 Tensor MP partition #1 / #2，外层粗框标 Pipeline MP partition #1 / #2，展示张量并行在节点内、流水线并行跨节点的两级切分" style="width: 100%;">
</div>

> **图片来源**：Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM（arXiv 2104.04473v5）Figure 2，§2 MODES OF PARALLELISM。本地副本：`references/papers/megatron-lm-gpu-clusters/megatron-lm-gpu-clusters.md` 第 56 行引用图。

### 4. `megatron-fig6-bubble-size-vs-data-parallel-size.jpg`（源 `0b96d327cec43357cc2271be149d6d69f90bbc9ef277dbeb3ecea526801ee5dc.jpg`，第 155 行）

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/megatron-fig6-bubble-size-vs-data-parallel-size.jpg" alt="Megatron-LM Figure 6：Pipeline bubble size vs Data-parallel size (d)，四条曲线 n=32/b'=32、n=32/b'=128、n=128/b'=128、n=128/b'=512；n=32,b'=32 从 ≈1.00 降至 d=32 的 ≈0.00，n=128,b'=512 全程 ≈0.25→0.13" style="width: 70%;">
</div>

> **图片来源**：Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM（arXiv 2104.04473v5）Figure 6，§3.2 Tensor and Pipeline Model Parallelism。本地副本：`references/papers/megatron-lm-gpu-clusters/megatron-lm-gpu-clusters.md` 第 155 行引用图。

### 5. `megatron-fig11-throughput-vs-pipeline-parallel-size.jpg`（源 `59d12b477b0866d52a1dc2b616057aa4566e927337afb0bf861ff670b6142c67.jpg`，第 296 行）

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/megatron-fig11-throughput-vs-pipeline-parallel-size.jpg" alt="Megatron-LM Figure 11：Achieved teraFLOP/s per GPU vs Pipeline-parallel size (1,2,4,8)，Batch size=8 从 ≈162 降至 ≈88，Batch size=128 从 ≈178 仅降至 ≈160" style="width: 70%;">
</div>

> **图片来源**：Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM（arXiv 2104.04473v5）Figure 11，§5.3.1 Weak Scaling。本地副本：`references/papers/megatron-lm-gpu-clusters/megatron-lm-gpu-clusters.md` 第 296 行引用图。

### 6. `megatron-fig12-interleaved-vs-noninterleaved-throughput.jpg`（源 `334ef459964f9ba6db7a6a6b510aea873a95bf7b172004abad223516ca551de9.jpg`，第 299 行）

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/megatron-fig12-interleaved-vs-noninterleaved-throughput.jpg" alt="Megatron-LM Figure 12：Achieved teraFLOP/s per GPU vs Batch size (12-60)，Non-interleaved 从 ≈81 升至 ≈136，Interleaved 从 ≈119 升至 ≈147，两线随 batch 增大而收敛" style="width: 70%;">
</div>

> **图片来源**：Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM（arXiv 2104.04473v5）Figure 12，§5.3.2 Interleaved versus Non-Interleaved Schedule。本地副本：`references/papers/megatron-lm-gpu-clusters/megatron-lm-gpu-clusters.md` 第 299 行引用图。

### 7. `megatron-fig13-pipeline-vs-tensor-parallel-size.jpg`（源 `f3ea415d7b9cf5de4c13a9f73868a669dad7cbed910f54fcd60ae825abbf460c.jpg`，第 302 行）

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/megatron-fig13-pipeline-vs-tensor-parallel-size.jpg" alt="Megatron-LM Figure 13：Achieved teraFLOP/s per GPU vs (Pipeline-parallel size, Tensor-parallel size) = (2,32)(4,16)(8,8)(16,4)(32,2)；Batch size=32 与 128 两条曲线峰值均在 (8,8)，分别为 ≈142 与 ≈166" style="width: 70%;">
</div>

> **图片来源**：Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM（arXiv 2104.04473v5）Figure 13，§5.4.1 Tensor versus Pipeline Parallelism。本地副本：`references/papers/megatron-lm-gpu-clusters/megatron-lm-gpu-clusters.md` 第 302 行引用图。

### 8. `megatron-fig14-pipeline-vs-data-parallel-size.jpg`（源 `b6c8d7a05522215bc25eee68d6ba3879ad2c8128da45bb20d75dcaa435e8e3f0.jpg`，第 313 行）

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/megatron-fig14-pipeline-vs-data-parallel-size.jpg" alt="Megatron-LM Figure 14：Achieved teraFLOP/s per GPU vs (Pipeline-parallel size, Data-parallel size) = (2,32)(4,16)(8,8)(16,4)(32,2)；Batch size=32 从 ≈62 缓降至 ≈45，Batch size=512 从 ≈150 降至 ≈88，均随 pipeline size 增大而下降" style="width: 70%;">
</div>

> **图片来源**：Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM（arXiv 2104.04473v5）Figure 14，§5.4.2 Pipeline versus Data Parallelism。本地副本：`references/papers/megatron-lm-gpu-clusters/megatron-lm-gpu-clusters.md` 第 313 行引用图。

### 9. `megatron-fig16-microbatch-size-throughput.jpg`（源 `231e06c3ea3e2972435388f81fb32418450cbacbacf6ea0d6192f92a18d78b46.jpg`，第 319 行）

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/megatron-fig16-microbatch-size-throughput.jpg" alt="Megatron-LM Figure 16：(t,p)=(8,8) 下 Achieved teraFLOP/s per GPU vs Microbatch size (1,2,4,8)；Batch size=128 峰值 ≈156 在 microbatch=2 后降至 ≈122，Batch size=512 峰值 ≈172 在 microbatch=2 后降至 ≈152" style="width: 70%;">
</div>

> **图片来源**：Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM（arXiv 2104.04473v5）Figure 16，§5.5 Microbatch Size。本地副本：`references/papers/megatron-lm-gpu-clusters/megatron-lm-gpu-clusters.md` 第 319 行引用图。

### 10. `megatron-fig17-activation-recomputation.jpg`（源 `d97fe77bf501c7b0c9c3abb20840c11650f208d6d377fdc9e458d2537fa28a76.jpg`，第 334 行）

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/megatron-fig17-activation-recomputation.jpg" alt="Megatron-LM Figure 17：Throughput (sequences/second) vs Batch size (1-256)；Act. recomputation 从 ≈0.6 升至 ≈7.8 覆盖全部 batch size，W/o act. recomputation 仅到 batch size=8（≈3.9）即终止" style="width: 70%;">
</div>

> **图片来源**：Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM（arXiv 2104.04473v5）Figure 17，§5.6 Activation Recomputation。本地副本：`references/papers/megatron-lm-gpu-clusters/megatron-lm-gpu-clusters.md` 第 334 行引用图。

### 11. `megatron-fig18-scatter-gather-optimization.jpg`（源 `12072e20fd8ce3cc1ba3e80ef1d6fdd4f0bb6cc54acff7cd478e515f81a8b1e2.jpg`，第 337 行）

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/megatron-fig18-scatter-gather-optimization.jpg" alt="Megatron-LM Figure 18：175B 模型 + interleaved 调度下 Achieved teraFLOP/s per GPU vs Batch size (12-60)；Unoptimized ≈108→≈131，Scatter/gather optimization ≈119→≈147，全程领先约 11%" style="width: 70%;">
</div>

> **图片来源**：Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM（arXiv 2104.04473v5）Figure 18，§5.7 Scatter-Gather Optimization。本地副本：`references/papers/megatron-lm-gpu-clusters/megatron-lm-gpu-clusters.md` 第 337 行引用图。

## 存疑与分歧

1. **草稿把 `648f402f…jpg` 当成 Figure 4，读图后判定为 Figure 2 —— 以读图为准，草稿需改。** 该图画面为 `Transformer layer #1/#2` + `Tensor MP partition #1/#2` + `Pipeline MP partition #1/#2`，与 md 第 58 行图注 `Figure 2: Combination of tensor and pipeline model parallelism (MP)…` 完全对应；而真正的 Figure 4（default vs interleaved 1F1B，含 `Assign multiple stages to each device` 箭头）是 `2a3ac338…jpg`（md 第 62 行）。两张图在 md 中都能被正确引用（第 56、62 行），只是**哈希名容易被认错**，正文引用块按本清单第 1、3 条走。
2. **md 第 58 行的图注是两段图注被 MinerU 粘连的结果。** 第 58 行写作 `Figure 2: Combination of tensor and pipeline model parallelism (MP) used in this work for transformer-based models. Pipeline flush` —— 其中 `Pipeline flush` 实际是 Figure 3 图注串上来的残段（Figure 3 原文图注为 `... leading to idle devices and a pipeline bubble.`，第 60 行已完整给出）。这不影响图号判定（图在注前，与 Figure 1 的排布一致），但**不能**拿第 58 行当 Figure 2 的完整图注引用。
3. **Figure 9 的两张子图被打包进同一个文件。** `8810fe26…jpg` 一张宽图同时包含「无 scatter/gather」与「有 scatter/gather」两面板，md 第 215 行的 `(a) … (b) …` 是文字说明而非另存图片；因此 19 张真插图里 Figure 9 只占 1 个文件，**不存在** Figure 9(a)/9(b) 两张独立文件。
4. **目录里没有「激活内存随 pipeline size 变化」的图，这一点与任务描述不符，需记录。** 通读 32 张后确认：本论文给出的气泡相关图是 Figure 6（bubble vs data-parallel size）与 Figure 11（吞吐 vs pipeline-parallel size）；显存相关只有 Figure 17（激活重算对可用 batch size 的影响，间接反映激活显存）与 §3.5 的文字/公式 `c·A^input + l/c·A^intermediate`。若文章 §6 需要「激活内存 ∝ pipeline size」的曲线，**megatron 这份资料给不出**，需要另找来源（Megatron-LM 后续论文、Zero Bubble 论文或本仓库 `references/papers/zero-bubble/`）。
5. **11 张公式截图与 2 张表格截图的「引用行」是反推的。** 它们在 md 中并无图片引用，表中的行号是 md 里同一内容的 LaTeX/HTML 出处（例如 `7736c2dd` ↔ 第 70 行公式、`492129ee` ↔ 第 257 行 `<table>`）。这批文件对本文章**没有引图价值**，但可作为「MinerU 会把行间公式/表格另外切图」的证据保留。
6. **两张表格截图内容完整、可直接转写。** `492129ee`（Table 1）与 `4e6647c7`（Table 2）的每一个可读数值都与 md 第 257、292 行的 HTML 表格一致（已逐格比对表头与首尾行）；因此结论是「转写为 Markdown 表格」，不要把截图贴进正文。
7. **边界候选（本次判为不采用，若章节需要可回取）**：`82614767`（Figure 8，解析模型 vs 实测的归一化吞吐曲线，若 §6 要展示「公式能复现拐点」则是唯一候选）；`252b2b41`（Figure 7，单卡算力随 microbatch 的单调上升，若 §9 要区分「单卡效率」与「流水线吞吐」两个口径可回取）。
