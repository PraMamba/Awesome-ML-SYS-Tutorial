# 中文社区文章配图逐张核对（6 篇 / 60 张）

核对范围：`torch/dualpipe/references/articles/<slug>/images/` 下全部 60 个文件，逐张用 `read_image` 实际读图（未依赖文件名猜测）。
对照物：`references/papers/`（MinerU 转换的论文 md + images）、`/workspace/algorithm/DualPipe/images/`（官方仓库图）。
核对日期：本轮会话；文中「官方图」指 `/workspace/algorithm/DualPipe/images/{dualpipe,dualpipev}.png` 与 `references/papers/` 中已归档的论文原图。

## 汇总

| slug | `ls | wc -l` | 清单行数 | 建议采用 |
| --- | --- | --- | --- |
| normaluhr-dualpipe-explained-zh | 5 | 5 | 0 |
| pipeline-parallelism-visualization | 5 | 5 | 5 |
| reku-dualpipe-thoughts | 8 | 8 | 0 |
| sea-ai-lab-cut-in-half | 5 | 5 | 4 |
| xiaodonggua-shousi-dualpipe | 24 | 24 | 9 |
| yeqianshu-dualpipe-source-walkthrough | 13 | 13 | 4 |
| **合计** | **60** | **60** | **22** |

- 合计文件数：60；合计清单行数：60（逐篇 `ls images/ | wc -l` 核对结果：5 / 5 / 8 / 5 / 24 / 13，与逐张清单行数一一相等；`find -path '*/images/*' -type f | wc -l` 亦为 60；60 行清单的文件名与本目录 60 个文件双向一一对应，无遗漏、无多余）。
- 建议采用合计：**36 张** = 可进正文的社区图 **22 张**（其中「第一手来源」21 张 +「官方图 + 社区标注」1 张，即 `xiaodonggua` 的 `v2-02221b07…`）+ 「与官方图/一手图重复、应改引原图」**14 张**（11 张应改引官方/论文原图 + 3 张应改引 Sea AI Lab 一手原图）。两类互斥；下表「建议采用」列是 22 张之和。
- 不采用（既非第一手、也非原图重复，按类别另作处理）：**24 张**（22 + 14 + 24 = 60）。
- 类别分布（60 张）：论文/官方图转贴 16（其中 3 张为「官方图 + 社区标注」）、作者自绘调度图 33、作者自绘示意图/类比图 11、公式截图 0、代码截图 0、表格截图 0、装饰/页眉 0。其中 8 张「逐步骤表格」是作者自绘的调度表（`yeqianshu` step1–step8），计入作者自绘调度图，不计入「表格截图」；60 张里没有任何公式截图、代码截图或页眉装饰图。

判断规则（本文件统一使用，便于复核）：

1. 社区图 = 官方/论文原图的**未改动复制** → 结论「不采用（改引官方）」，计入 14 张的「应改引原图」桶。
2. 社区图 = 官方/论文原图**叠加了作者自己的论点标注** → 保留采用，但必须在图注写明「官方图对应关系 + 标注为社区添加」。
3. 社区图 = 作者自绘且承载本文档需要的**一手信息**（其它资料没有同类表达）→ 采用。
4. 作者自绘但信息量低（手绘草图、逐步骤中间产物、自制排布的自证图）或与更权威一手资料同主题 → 不采用，在理由里写清替代来源。
5. 无公式截图、无代码截图、无装饰图（`grep -c '!\[\]('` 与 `ls` 数量一致，60 张全部被正文引用，无页眉/二维码类图片）；8 张「逐步骤表格」是作者自绘的调度表（`yeqianshu` step1–step8），计入作者自绘调度图，不计入「表格截图」。

## 逐张清单

| slug | 文件名 | md 引用行 | 类别 | 内容（读图后的事实描述） | 结论 | 建议章节 |
| --- | --- | --- | --- | --- | --- | --- |
| normaluhr-dualpipe-explained-zh | v2-581f35da90daf02c15f05619f488b0ec_r.png | 49 | 作者自绘调度图 | Device 1–4 × T0–T12 网格；每设备 1 格 Forward（黄）→ 大段灰格（idle）→ 1 格 Backward（蓝）→ Parameter Update（绿）；图例 "Forward / Backward / Parameter Update"；无 p、micro-batch 数、批次 ID 等口径标注；对应正文「未实行流水线系统之前」的车间示意 | 不采用：这是「完全没有流水线」的入门基线，正文可用 GPipe 论文 Figure 2/3 或 Megatron-LM 论文 Figure 3 表达同一件事；本图无 p/m 记号，接不上 §6 的比较口径 | —（§1 备选） |
| normaluhr-dualpipe-explained-zh | v2-58b9b289c095ef6417ba4a3ec90012b4_r.png | 55 | 作者自绘调度图 | Device 1–4 × T0–T36；warm-up 阶梯 → 稳定期 F/B 交替 → 尾部反向 → Optimization Step（绿）→ 下一轮 warm-up；micro-batch ID 1–8；灰格=idle，可读出气泡位置；图例 "Forward / Backward / Optimization Step" | 不采用：1F1B 基线已有更权威表达（Megatron-LM 论文 Figure 4 上半、`docs/aiinfra-pp-1f1b-interleaved/`），且本图记号与 §3 的 F/I/W 记号不通约 | —（§3 备选） |
| normaluhr-dualpipe-explained-zh | v2-4f06afb39ddb0b3d15ae01a90dc7a91a_r.png | 62 | 作者自绘调度图 | Device 1–4 × T0–T36；Forward（黄）/ Backward–Input（浅蓝）/ Backward–Weights（深灰蓝）/ Optimization Step（绿）；同一 iteration 内先完成 B-Input 再补 B-Weights，尾部 W 填气泡，形成被 optimizer step 分隔的两段；micro-batch ID 1–8 | 不采用：与 Zero Bubble 论文 Figure 3（ZB-H1 / ZB-H2）同主题的重绘；本图未标注自己对应 ZB1P 还是 ZB-H1/H2，引用会引入口径歧义，正文应引一手论文原图 | —（§3 备选） |
| normaluhr-dualpipe-explained-zh | v2-3d792b47f32f2275d837131843afd716_r.png | 138 | 论文/官方图转贴 | 逐格与官方 DualPipe 调度图一致：Device 0–7 × micro-batch ID 0–9；橙=Forward、绿=Backward、浅绿=Backward for input、蓝=Backward for weights、橙绿拼接=Overlapped forward & Backward；灰白格为气泡；图例文字与图内编号均与官方一致 | 不采用（改引官方）：与官方图重复，应优先引用官方原图 | 对应 `/workspace/algorithm/DualPipe/images/dualpipe.png`＝DeepSeek-V3 报告 Figure 5；§4 |
| normaluhr-dualpipe-explained-zh | v2-341bd07c8c699514aa9fab35d96e9797_r.png | 154 | 论文/官方图转贴 | 逐格与 DeepSeek-V3 报告 Figure 4 一致：两行（Computation / Communication）× 两段（MLP 段、ATTN 段），格内 MLP(B)/MLP(W)/MLP(F)/ATTN(B)/ATTN(W)/ATTN(F)、DISPATCH(F)/DISPATCH(B)/COMBINE(F)/COMBINE(B)/PP；△=Forward chunk、▲=Backward chunk；两端橙/绿/紫/红竖条＝barrier | 不采用（改引官方）：与官方图重复，应优先引用官方原图 | 对应 DeepSeek-V3 报告 Figure 4；§2/§7 |
| pipeline-parallelism-visualization | v2-895d80daa4ea431e561fd9e5ac95b374_r.jpg | 27 | 作者自绘调度图 | matplotlib「Activation Memory Timeline」：纵轴每设备在飞激活数 0–5，横轴 Schedule Timeline 0–50；Device 0–3 每设备连续 4 格 F0–F3（蓝，阶梯错开），随后各自连续 4 格 B0–B3（橙）；backward 阶段设备间有明显 idle 空隙（Device 0 的 B 到 t≈27 才开始）；四条包络线均由 0 升到峰值 4 再落回 0；右下角图例 Device 0–3 / Forward / Backward / SendRecv | 采用：全部 60 张里唯一带「激活显存时间线」叠加的基线图，直接服务 §6 的显存口径统一；与论文原图不重复（论文图不叠显存包络） | §1、§3、§6 |
| pipeline-parallelism-visualization | v2-663e063329c2940f297fe0bc55743cde_r.jpg | 39 | 作者自绘调度图 | 同款 matplotlib：Device 0–3 各连续 4 格 F0–F3（蓝）后接 4 格 B0–B3（橙），backward 自 Device 3 向 Device 0 反向推进、在 t≈18–20 收尾（flush）；四条包络线在 t≈4–5 同时到达峰值 4，符合「各设备峰值激活一致」；图例同上 | 采用：GPipe 基线的显存/时间线一手渲染，公式与图可以互相验证；不替换为 GPipe 论文 Figure 2，因为论文图没有显存包络 | §3、§6 |
| pipeline-parallelism-visualization | v2-3f3c2f8b447143d60fb104d87c49e3e7_r.jpg | 53 | 作者自绘调度图 | 同款 matplotlib：Device 0–3，micro-batch 0–7；warm-up 阶梯 → 稳定期严格 1F1B（F/B 交替，可读出 F4/B1、F5/B2 …）→ 尾部反向；激活包络呈「锯齿平台」，峰值 4 并保持到 t≈15 后下降；可见每设备稳定期错位一格 | 采用：1F1B 是 §3 的基准调度，本图同时给出气泡形状与激活包络，是社区侧唯一可作横向比较的一手图 | §3、§6 |
| pipeline-parallelism-visualization | v2-20e1678a5d804874102a944b8a5b4565_r.jpg | 75 | 作者自绘调度图 | 同款 matplotlib：「对称 1F1B」——Device 0–3 每行按 F0（空一格）F1（空一格）F2 … 的间隔排布，稳定期 B 与 F 交替，包络呈连续三角锯齿，峰值 4；对应正文「仅用于理解，实际没有该调度」 | 采用（必须标注）：图中调度并不存在，作者本人已注明；它的唯一价值是解释 warm-up 次数 p−p_i−1 的下界，正文引用时必须在图注写明「非真实调度，仅用于说明 warm-up 下界」 | §3 |
| pipeline-parallelism-visualization | v2-93b117ea7ca1d90b96d3adb97b531ebd_r.jpg | 83 | 作者自绘调度图 | 同款 matplotlib，纵轴上限 6：Device 0–3，每设备两个 chunk（F-Chunk0 深蓝 / F-Chunk1 浅蓝 / B-Chunk0 深绿 / B-Chunk1 浅绿），micro-batch 0–7；可见 chunk 交错与更早的 flush；包络峰值 6（比 1F1B 更高），对应正文「激活峰值显存增加」；图例含 F-Chunk0/F-Chunk1/B-Chunk0/B-Chunk1/SendRecv | 采用：这是 60 张里唯一按 vpp/g 记号画出 interleaved-1F1B 并同时给出显存代价的图，直接服务 §8（Megatron combined 1F1B）与 §6 | §8、§6 |
| reku-dualpipe-thoughts | v2-4a744711b2f5129cd3f36bd2674f8f64_r.jpg | 9 | 作者自绘示意图/类比图 | 手绘：标题「mini batch=2」，上行 4 个圆角框 Attention → AllToAll → MLP → AllToAll，下行同样 4 框错位半格；表达「把一层拆成两个无数据依赖的小层互相掩盖」 | 不采用：手绘草图，只表达一个已在正文说清的想法，且「切 bs / 切 seq 的代价」无法从图读出；正文用文字或 Mermaid 表达更省版面 | — |
| reku-dualpipe-thoughts | v2-f8ece5e658a6cbdce5c024264d2b75c7_r.jpg | 15 | 作者自绘示意图/类比图 | 手绘：上行 Attention → AllToAll → MLP → AllToAll（正向），下行 AllToAll(B) → MLP(B) → AllToAll(B) → Attention(B)（反向）；表达「一个正向能不能和一个反向互相掩盖」 | 不采用：信息量与上一张同级，正文一句话可替代 | — |
| reku-dualpipe-thoughts | v2-903c53e90f817ab377412da44d48dc3b_r.jpg | 17 | 作者自绘调度图 | 手绘 1F1B 草图：三行设备，格内数字 0/1/2/3/4，上行 0-0-1-1-2-2-3-3-4（正反向交替），下行 0-1 / 0-2-1-3-2-4-3（B 与 F 交错），最下行 0-1-2 / 0-3-1-4-2；无时间轴、无颜色区分 F/B | 不采用：无图例、无时间轴的手绘 1F1B，无法作为正文插图；§3 用 `pipeline-parallelism-visualization` 的 1F1B 图 | — |
| reku-dualpipe-thoughts | v2-3b27856696b7fa51a21d4a4291c3fec2_r.jpg | 21 | 作者自绘调度图 | 手绘：三行设备错位排布，格内 0/1/2/3/4/5，行间留大段空白；表达「强行掩盖相邻部分的正反向通信会导致 bubble 变大」的结论 | 不采用：手绘且没有量化 bubble，正文用 §4 的官方调度图 + 文字即可说明 | — |
| reku-dualpipe-thoughts | v2-c1cba8a88d50f7644083d4d8c71150f4_r.jpg | 25 | 论文/官方图转贴 | 论文风格调度图，P0–P7 八行，P3/P4 之间一条水平虚线；部分格内为两个数字（重叠格，浅蓝底）；右侧两张竖直柱状图分别标 M_W（绿，轴 1 2）与 M_A（粉，轴 2 4 6 8）；图下方保留原图注 "(c) Chimera" | 不采用（改引官方）：这是论文图的裁切转贴（保留 "(c) Chimera" 图注），应改引 Chimera 一手图；但本仓库 `papers/chimera/` 的 Figure 2/3/8 排版与该图均不一致，无法确认具体 Figure 编号，正文若要用须先回溯原文 | 疑似 Chimera 论文（Li & Hoefler, SC21）；§4/§9 |
| reku-dualpipe-thoughts | v2-7e33dc1655added04ef3d4bb8d92ec63_r.jpg | 29 | 论文/官方图转贴 | 与官方 DualPipe 调度图逐格一致（Device 0–7、micro-batch 0–9、同一图例与橙/绿/浅绿/蓝/橙绿拼接配色），仅分辨率更低并带知乎水印 | 不采用（改引官方）：与官方图重复，应优先引用官方原图 | 对应 `/workspace/algorithm/DualPipe/images/dualpipe.png`＝DeepSeek-V3 报告 Figure 5；§4 |
| reku-dualpipe-thoughts | v2-7c04292e8b38a6cbb2434e34d7667bc3_r.jpg | 43 | 论文/官方图转贴 | 与 Megatron-LM 论文 Figure 4 下半（interleaved 1F1B）逐格一致：Device 1–4、每设备多 chunk、深/浅色区分 chunk0/chunk1、黑色竖线＝pipeline flush、图例 "Forward Pass / Backward Pass"；作者在可掩盖区间上叠加了 3 个红色矩形框 | 不采用（改引官方）：底图是论文图转贴，应改引 Megatron-LM 论文 Figure 4 原图；正文若要表达「VPP 中这些区间可掩盖」，建议在官方图上自行标注，避免引用二手截图 | 对应 `papers/megatron-lm-gpu-clusters/images/2a3ac338c206…jpg`＝Megatron-LM 论文 Figure 4；§8/§9 |
| reku-dualpipe-thoughts | v2-e3657f716555455f55acc73d74bb26b3_r.jpg | 49 | 论文/官方图转贴 | FLUX（arXiv 2406.06858）风格的两段对照图：上半 "Original"、下半 "Overlapped"；左侧是 A/B/C 分块的矩阵乘示意，右侧是两条时间线（GPU0/GPU1 上 GEMM(A00,A10,B0)->[C00,C10] 与 ReduceScatter，Overlapped 版插入 Send C0 Recv C1 / Send C1 Recv C0 并标注 "Time Saved"） | 不采用（改引官方）：论文图转贴（FLUX，本仓库未归档该论文），且主题是 TP 侧 reduce-scatter 与 GEMM 的 kernel 融合，属 §9 选型的边缘材料；若要用必须引 FLUX 原文 | —（§9 备选） |
| sea-ai-lab-cut-in-half | v2-3da883a2e88124b91ba12d642d9e3ba9_r.png | 19 | 论文/官方图转贴（官方图 + 社区标注） | 底图是 DualPipe 调度图的改写版：Device 0–7，每格内改为**两行编号**（上=upside-down 方向、下=down-to-up 方向 micro-batch），图例明确写出 "Micros from up to down / Micros from down to up"，并在右侧加注 "The two parts are mirrored."；Device 0–3 与 Device 4–7 两半逐格镜像 | 采用：这是论文「反向 micro-batch 对称」一句话的**可视化证据**，也是本文档 §4「镜像两半」论证的一手来源；必须同时标注底图对应官方 `dualpipe.png`（V3 报告 Figure 5），标注内容为 Sea AI Lab 添加 | §4（草稿称「镜像对称示意」——核实成立，见「存疑与分歧」1） |
| sea-ai-lab-cut-in-half | v2-7c797d3cc827149eadb7beb740d5889f_r.jpg | 26 | 作者自绘示意图/类比图 | 上下两栏对照：上栏 "DualPipe"（Model Layers 列：Device 0=0,7；Device 1=1,6；Device 2=2,5；Device 3=3,4；Device 4=4,3 … Device 7=7,0；右侧 8 行橙色 F 块排成 V 形，Device 3/4 之间一条水平虚线），下栏 "Cut-in-half"（只保留 Device 0–3，Model Layers 仍为 0,7 / 1,6 / 2,5 / 3,4，右下方橙色 F 块排成更窄的 V 形） | 采用：**cut-in-half / V 形由来的第一手来源**（作者即 Zero Bubble 团队），也是官方 `dualpipev.png` 的设计依据；仓库内没有等价图（官方仓库只给最终 DualPipeV 调度图，没有这个「裁剪前后对照」） | §5、§4 |
| sea-ai-lab-cut-in-half | v2-e77aeb55ddbe50cb37dfeb213c6ac054_r.png | 31 | 论文/官方图转贴（官方图 + 社区标注） | Cut-in-half 的完整调度图：Device 0–3，每格两行编号，图例 "The first half layers / The second half layers" + Forward / Backward / Backward for input / Backward for weights / Overlapped forward & backward；逐格与官方 `dualpipev.png` 一致 | 不采用（改引官方）：与官方图重复，应优先引用官方原图 | 对应 `/workspace/algorithm/DualPipe/images/dualpipev.png`（官方 README 明确写该图即 Sea AI Lab 的 cut-in-half）；§5 |
| sea-ai-lab-cut-in-half | v2-4c4a1cc8bf1e26ee6e2b93be719f4070_r.png | 61 | 作者自绘调度图 | 「解耦 F/B 并 squeeze」中间调度：Device 0–3，每格两行编号，**不再有橙绿拼接的重叠格**（图例只剩 Forward / Backward / Backward for input / Backward for weights），F 与 B 被拆开重新排布以获得更灵活依赖 | 采用：cut-in-half → ZBV 之间的关键中间态，第一手（官方仓库无此图，`dualpipev.png` 只给最终态）；是 §5「同一理念下继续挤气泡」的直接证据 | §5、§6 |
| sea-ai-lab-cut-in-half | v2-1247bd394c4ed000ce98345349d82f04_r.png | 65 | 作者自绘调度图 | ZB-V 调度：Device 0–3，每格两行编号，格子被重新排成更宽的阶梯，无重叠格、无 optimizer step 分隔；图例同上（Forward / Backward / Backward for input / Backward for weights） | 采用：第一手 ZB-V 调度图（Zero Bubble 团队自己给出），是 §5 的终点态；正文若同时要论文口径，应与 `papers/zero-bubble/images/536478895e0a…jpg`（Zero Bubble 论文 Figure 8）并列引用并说明两者是同一调度的不同渲染 | §5、§6 |
| xiaodonggua-shousi-dualpipe | v2-1d033c926efa7dd4386d269e1470255c_r.jpg | 20 | 作者自绘示意图/类比图 | 两张并排示意：左「流水线并行」（GPU0 上 W1、W2，GPU1 上 W3、W4，竖向箭头串行，两 GPU 之间一个红色双向箭头＝设备间通信）；右「张量并行」（W1-a/W1-b … W4-a/W4-b 两列，每层左右各有红箭头，底部 GPU0/GPU1） | 不采用：与 Megatron-LM 论文 Figure 2（TP+PP 组合）表达同一件事，属入门对比；正文若要横向对照应引论文原图 | —（§1 备选） |
| xiaodonggua-shousi-dualpipe | v2-78c60e5c84bc64d10f302e9eaad427a3_r.jpg | 24 | 作者自绘示意图/类比图 | 左右两栏（前向 / 反向）的 EP 通路示意：Attention → LayerNorm →（All-to-All Dispatch，虚线分隔）→ Expert 0 / Expert 1 →（All-to-All Combine，虚线分隔）→ LayerNorm；token 集合标出 X0={0,1,2}、X1={10,11,12}，dispatch 上标 {0,2}、{11}，expert 侧标 {1,11}、{0,2,10,12}；底部 GPU0 / GPU1 | 采用：把「stage 内通信等待」具体化成可核对的 token 集合，是 §1/§2 讲清 All-to-All dispatch/combine 代价的最省字数的图，官方论文没有等价图 | §1、§2 |
| xiaodonggua-shousi-dualpipe | v2-384a158ddab66a184b1254bf6f6f74c6_r.jpg | 28 | 作者自绘示意图/类比图 | 「MoE训练 PP+EP」示意：左绿框 PP Stage1（GPU0/GPU1）内部有两次 All-to-All（标「(内部) stage 内部通信 All-to-All」），右蓝框 PP Stage2（GPU2/GPU3），两框之间用红箭头标「(外部)PP stage 之间通信 当前stage传入到下一个stage」 | 采用：唯一把「stage 内 All-to-All」与「stage 间 P2P」画在同一张图上的社区图，正好是 §1 两种等待的分界，也是 §8 框架侧对照的公共词汇 | §1、§8 |
| xiaodonggua-shousi-dualpipe | v2-701ccef478c1f397adf1324177a517bc_r.jpg | 51 | 作者自绘示意图/类比图 | 左「Dense Block 执行周期」：Attention → LayerNorm → MLP → LayerNorm；右「MoE Block EP执行周期」：Attention → LayerNorm → All-to-All Dispatch → Expert0/Expert1 → All-to-All Combine → LayerNorm（带 token 集合标注）；底部两行彩色块图例：Dense = Attn / MLP / MLP(B) / Attn(B) / PP，MoE = Attn / All-to-All / MLP / All-to-All / All-to-All(B) / MLP(B) / All-to-All(B) / Attn(B) / PP | 采用：底部图例行就是 §2 需要的 F / I / W / B_full / C 记号与 MoE 操作块的对照表（社区版），比纯文字更快建立记号 | §2 |
| xiaodonggua-shousi-dualpipe | v2-b1cfba984fe157dec26a9c5fa0251e0c_r.jpg | 55 | 作者自绘调度图 | 上下两组 1F1B 排布对照：上「Dense Block 1F1B执行周期」（GPU0–3，格内 Attn+MLP / MLP+Attn(B)），下「MoE EP 1F1B执行周期」（GPU0–3，格内 Attn+A2A+MLP+A2A / A2A+MLP+A2A+Attn(B)）；右侧注「由于A2A的存在，拉长了一个Stage的计算周期 导致 EP+PP 气泡率更高」 | 采用：**§1 两种等待的核心论据图**——同一 1F1B 下，A2A 让单 stage 周期变长从而放大气泡；官方论文没有这种对照 | §1 |
| xiaodonggua-shousi-dualpipe | v2-33affd2027bc0a57b5c8df95ced57d32_r.jpg | 59 | 作者自绘示意图/类比图 | 三组色块对照：上「EP 1F1B 计算+通信串行」= Attn / All-to-All dispatch / MLP / All-to-All Combine / All-to-All(B) Dispatch / MLP(B) / All-to-All(B) Combine / Attn(B) / PP 一条链；下左「EP 1F1B 通信计算重叠」= 计算行 Attn / MLP(B) / MLP / Attn(B) 与通信行 All-to-All(B) Combine / All-to-All(B) Dispatch / All-to-All(B) Dispatch / All-to-All combine / PP 上下对齐，并标出 t_overlap；下右「Dense 1F1B」= Attn / MLP / MLP(B) / Attn(B) / PP | 采用：把「重叠前 / 重叠后」的周期长度直接画成可比的色块，是 §6 讨论「F&B 重叠后气泡公式里为什么出现 F&B 项」的直观依据 | §6、§2 |
| xiaodonggua-shousi-dualpipe | v2-a6e680b8933e67c09a3d9c10763bcb5c_r.jpg | 63 | 作者自绘调度图 | 上下对照：上「MoE EP 1F1B执行周期 Non-Overlap-1F1B」（GPU0–3 紧密排布，右侧注「非overlaped耗时是 overlaped 的2倍」），下「MoE EP 1F1B执行周期 Overlap-1F1B」（GPU0–3，用粉色 "Overlap 1F1B" 格 + 蓝色 "bubble" 标注指出重叠后新出现的空洞） | 不采用：结论与 `v2-b1cfba98…` 和 `v2-33affd20…` 重复（同一「重叠换来更长 stage 周期 / 新气泡」论点第三次出现），三张取二已足够；保留 `33affd20`（周期可比）与 `b1cfba98`（气泡成因） | — |
| xiaodonggua-shousi-dualpipe | v2-630ec2414473a93f83a8d43b6b2f1e81_r.jpg | 67 | 作者自绘示意图/类比图 | 上下对照「EP-Overlaped 里的 1F 完成时机」与「Dense 1F1B 里的 1F 完成时机」：Stage1 一排实线块（Attn / MLP(B) / MLP / Attn(B)）与 Stage2 的虚线块（Attn / MLP(B) / MLP / Attn(B)），红色向下箭头标出「1F完成」「1B完成」两个时刻；右侧注「更严格需要完成Attn(B) 才发送数据到下一个Stage」 | 不采用：结论已由 `v2-b1cfba98…`（stage 周期变长）覆盖，本图是同一论点的时序细化；正文用一句话说明「overlap 块的完成时刻后移」即可 | — |
| xiaodonggua-shousi-dualpipe | v2-00c117ca4e6e1796255943a076242e27_r.jpg | 71 | 作者自绘调度图 | 上下两幅 V 形调度：上「DualPipe-Phase1」（GPU0–7，左侧从 GPU0 起 1F 阶梯上行、右侧 1B 阶梯下行，中间为深绿 1F1B 带），下「DualPipe-Phase2」（同结构，改用蓝紫色系，1F 阶梯从 GPU7 下行、1B 上行，中间为深蓝 1F1B 带）；只有 1F / 1F1B / 1B 三种标签 | 采用：把「双向注入」拆成两个方向分别可见，是 §4 讲 (F0,B1,F1,B0) 配对前最直观的一步；官方调度图把两个方向叠在一起，看不出这个分解 | §4 |
| xiaodonggua-shousi-dualpipe | v2-926b3cfd48afb50720e51bce240f4f7c_r.jpg | 75 | 作者自绘调度图 | 把 Phase1 与 Phase2 融合成一张完整 DualPipe 排布（GPU0–7，浅绿=1F、深绿/深蓝=1F1B、浅蓝/浅绿边=1B），底部用红色双向箭头标 "Remove Bubble" 并写出 `1F1B1F1B`，指出两个 1F1B 相邻即可消掉中间气泡 | 不采用：融合后的结论已由 `v2-00c117ca…`（两方向分解）+ 采用的官方图 `xiaodonggua-official-schedule-eight-steps.jpg` 覆盖，本图是同一论证的第三次表达；且它是作者为「消除 stage 外气泡」自建的排布，非官方 | —（§4 备选） |
| xiaodonggua-shousi-dualpipe | v2-a9359b74e6853e63aaad939ce98dd84c_r.png | 94 | 论文/官方图转贴 | 与 DeepSeek-V3 报告 Figure 4 逐格一致：两行（Computation / Communication）× MLP 段与 ATTN 段，格内 MLP(B)/MLP(W)/MLP(F)/ATTN(B)/ATTN(W)/ATTN(F)、DISPATCH(F)/DISPATCH(B)/COMBINE(F)/COMBINE(B)/PP，△=Forward chunk、▲=Backward chunk，两端为 barrier 色条；正文明确写「DeepSeek-V3 论文给出了通信-计算重叠示意图」 | 不采用（改引官方）：与官方图重复，应优先引用官方原图 | 对应 DeepSeek-V3 报告 Figure 4；§2/§7 |
| xiaodonggua-shousi-dualpipe | v2-4ef39e5d6049e4c22955fe9c0cb3825d_r.jpg | 102 | 论文/官方图转贴 | 与官方 DualPipe 调度图逐格一致（Device 0–7、micro-batch 0–9、橙/绿/浅绿/蓝/橙绿拼接、同一图例文字），仅叠加了知乎水印 | 不采用（改引官方）：与官方图重复，应优先引用官方原图 | 对应 `/workspace/algorithm/DualPipe/images/dualpipe.png`＝DeepSeek-V3 报告 Figure 5；§4 |
| xiaodonggua-shousi-dualpipe | v2-b923066c55af15aed38713513da35083_r.jpg | 133 | 作者自绘调度图 | GPU 0–7 × 时间网格，只用 4 种颜色：青=F0、黄=F10、蓝=B0、红=B10，格内写 0 或 10；右下角图例明写 "F0 / F10 / B0 / B10"；局部同时出现 0 与 10 两格相邻（t≈4–5、t≈15–17） | 采用：**最简双向布局图（每方向 1 个 micro-batch）**，正好对应 §4 要点名的 (F0, B1, F1, B0) 配对；官方图是 20 个 micro-batch，无法看清配对关系 | §4 |
| xiaodonggua-shousi-dualpipe | v2-34eaa5c3e4d7133f3cb82850f05bd866_r.jpg | 137 | 作者自绘调度图 | 上下两幅同一张「单边 10 个 micro-batch」自制 DualPipe 排布（GPU0–7，青/浅青/黄/红/蓝/浅蓝多色，格内 0–3、10–19），上图每格更小更稠密、下图放大以便阅读；正文说明「在顶部和底部单边处理10个micro batch」 | 不采用：这是作者按自己的理解还原的 20-micro-batch 排布（非官方），已被 `v2-02221b07…`（官方图 + 8 步标注）取代；引用自制排布容易与官方调度混淆 | —（§4 备选） |
| xiaodonggua-shousi-dualpipe | v2-ed6a8b52676467ae03580972eb835944_r.jpg | 143 | 作者自绘调度图 | 单边减到 4 个 micro-batch 的自制排布：GPU0–7，青/黄/红/蓝四色，格内 0/1/2/3 与 10/11/12/13；可见两方向在中间交错 | 不采用：自制排布的中间产物，信息被下一张（`482f980f`/`b96b60cf`）包含 | — |
| xiaodonggua-shousi-dualpipe | v2-482f980f0371bc35500bb184c8150e9a_r.jpg | 147 | 作者自绘调度图 | 在上一张基础上「进一步调整」的排布：GPU0–7，格内 0–3 与 10–13；底部 4 个 device 的斜向 1F1B 带更整齐 | 不采用：同上，自制排布的中间产物；被 `b96b60cf` 的操作标注版取代 | — |
| xiaodonggua-shousi-dualpipe | v2-b96b60cfceadaf71392646c42d476b0c_r.jpg | 153 | 作者自绘调度图 | 只保留 GPU0–3 四行，每格下方另起一行写上操作名：F0 / F1F0 / F1B1 / B0B1 / B0（GPU0 行：F0 F0 F0 F0 → F1B1 F1B1 F1B1 F1B1 → B0 B0 B0 B0；GPU1 行：F0 F0 F0 → F1F0 F1B1 F1B1 F1B1 → B0B1 B0 B0 B0；GPU2/GPU3 行同规律逐步增加 F1F0/B0B1） | 采用：直接给出 §4 需要的 5 个复合操作（F0、F1F0、F1B1、B0B1、B0）及其出现规律，是社区侧唯一把「配对」写成可数操作的图 | §4 |
| xiaodonggua-shousi-dualpipe | v2-cbcfa5a771d3e777a705dd778fab67a8_r.jpg | 278 | 作者自绘调度图 | 自制实现的打印结果排布：GPU0–7，格内 0–3 与 10–13；与上一张结构一致但边界更整齐，作为「打印结果符合预期」的自证 | 不采用：代码输出的自证图，不是独立信息；正文若要展示自制实现结果，应直接引 `community/easy-dualpipe/` 的可复现输出 | — |
| xiaodonggua-shousi-dualpipe | v2-52de2c6bec9e992eeddd5fe182e12ce9_r.png | 596 | 作者自绘调度图 | GPU0–7 自制排布，格内 0–3 与 10–19；图上有绿色圆圈和绿色箭头，圈出同时存在 F0/F1 或 B0/B1 的位置，指向「应为 F1B1 而非 F0F1」的缺陷 | 不采用：作者在「分析」节自述这套排布就是 Chimera；结论是「我的实现有缺陷」，作为正文插图价值为负；Chimera 对应关系应引 Chimera 论文原图 | —（§9 备选） |
| xiaodonggua-shousi-dualpipe | v2-02221b07d4e4e5cfb10e355dbbd2c686_r.jpg | 606 | 论文/官方图转贴（官方图 + 社区标注） | 底图是官方 DualPipe 调度图（Device 0–7、micro-batch 0–9、同一图例与配色），作者用红色斜线／矩形把它切成 8 段并逐段标注：F0、F0F1、B1W1F1、F0B1F1B0、B1F1B0、B1B0、WB0、W | 采用：这 8 段命名与 `yeqianshu` 文章 step1–step8（nF0 / nF0F1 / nB1W1F1 / nF0B1F1B0 / nB1F1B0 / nB1B0 / nWB0 / nW）逐字对应，两篇独立文章互证，是 §4/§7 讲官方 8 步划分的最佳社区图；引用时必须同时给出官方 `dualpipe.png` 并说明红框为社区添加 | §4、§7 |
| xiaodonggua-shousi-dualpipe | v2-26b1578916680d21cccff770e0a750b5_r.jpg | 866 | 论文/官方图转贴 | 与 `sea-ai-lab-cut-in-half` 的 `v2-7c797d3c…` 为**同一张图**：上「DualPipe」8 设备 V 形、下「Cut-in-half」4 设备 V 形，Model Layers 列 0,7 / 1,6 / 2,5 / 3,4，DualPipe 中 Device 3/4 间有水平虚线 | 不采用（改引一手）：同一张图已有第一手来源（Sea AI Lab 作者自己的博客），应引 `sea-ai-lab-cut-in-half` 那一份，不要经本文二手转贴 | 对应 `sea-ai-lab-cut-in-half/images/v2-7c797d3cc827149eadb7beb740d5889f_r.jpg`；§5 |
| xiaodonggua-shousi-dualpipe | v2-45e9c6fe9eabed217a5391b3be80f953_r.png | 870 | 论文/官方图转贴 | Cut-in-half 完整调度：Device 0–3，每格两行编号，图例 "The first half layers / The second half layers" + Forward / Backward / Backward for input / Backward for weights / Overlapped forward & backward；与 `sea-ai-lab-cut-in-half` 的 `v2-e77aeb55…` 为同一张图 | 不采用（改引官方）：内容与官方 `dualpipev.png` 重复，应引官方原图；如需「前半层/后半层」编号说明则引 Sea AI Lab 一手版 | 对应 `/workspace/algorithm/DualPipe/images/dualpipev.png`；§5 |
| xiaodonggua-shousi-dualpipe | v2-9dd1e161bc926aa7b00530f65a26c330_r.png | 872 | 论文/官方图转贴 | 「解耦 F/B 并 squeeze」中间调度：Device 0–3，每格两行编号，无重叠格，图例只剩 Forward / Backward / Backward for input / Backward for weights；与 `sea-ai-lab-cut-in-half` 的 `v2-4c4a1cc8…` 为同一张图 | 不采用（改引一手）：同一张图应引 Sea AI Lab 一手版（`v2-4c4a1cc8…`），避免二手转贴 | 对应 `sea-ai-lab-cut-in-half/images/v2-4c4a1cc8bf1e26ee6e2b93be719f4070_r.png`；§5 |
| xiaodonggua-shousi-dualpipe | v2-bb4b8477f726117c93d1ff408782bbe8_r.png | 874 | 论文/官方图转贴 | ZB-V 调度：Device 0–3，每格两行编号，宽阶梯、无重叠格、无 optimizer step；与 `sea-ai-lab-cut-in-half` 的 `v2-1247bd39…` 为同一张图 | 不采用（改引一手）：同一张图应引 Sea AI Lab 一手版（`v2-1247bd39…`） | 对应 `sea-ai-lab-cut-in-half/images/v2-1247bd394c4ed000ce98345349d82f04_r.png`；§5 |
| yeqianshu-dualpipe-source-walkthrough | v2-a9aa0c9ef6aed7ad1e1b74a981adc176_r.jpg | 13 | 作者自绘调度图 | 图 1，三栏 (a)(b)(c)：左侧纵轴标 pipeline / 1F1B / dualpipe，device 行分别写 device0/device1（(a)(b)）与 device0/device2（(c)）；(a) 是 F 段（黄，0-1-2-3）接 B 段（绿，0-1-2-3）的全前向全反向；(b) 是 1F1B，顶部横条标 nF / n1F1B / nB；(c) 是 dualpipe，顶部横条标 F0F1 / F0B1 / F1B0 / B1B0，格内 0-2-1-2-3-0-3-1 / 2-0-3-0-1-2-1-3；底部图例「前向（黄）/ 后向（绿）」 | 采用：一张图覆盖「全前向全反向 → 1F1B → DualPipe」三段演进，正是 §3→§4 的过渡；图内 device 行标签不一致（(c) 用 device0/device2 却只有两行）需在图注说明 | §3、§4 |
| yeqianshu-dualpipe-source-walkthrough | v2-4fd75112270832090a24aee8a3382ff2_r.jpg | 23 | 作者自绘示意图/类比图 | 图 2：左为前向计算图（X → W → output），右为反向计算图（Grad_output 同时送 W 与 X，分别得 Grad_x 与 Grad_w）；下方给出公式与形状：`output = X @ W`，`Grad_x = Grad_output @ W`，`Grad_w = Grad_output.flatten(0,1).T @ X.flatten(0,1)`，X:[b,L,h]、W:[h,h]、output:[b,L,h]、Grad_output:[b,L,h] | 采用：§2「反向拆分 I=GΘ、W=GᵀX」的社区版推导示意，含 flatten 后的形状约定；这是自绘推导图而非论文公式截图，可作插图（若正文已用 LaTeX 写出公式，则只引其形状约定即可） | §2 |
| yeqianshu-dualpipe-source-walkthrough | v2-86d6d62375e52bc9ad2e15ffbfd42810_r.jpg | 59 | 作者自绘示意图/类比图 | 图 3：左侧「1F1B Pipeline」= pipe 0–7 各持一个 Layer 0–7，箭头标 forward（蓝）/ backward（绿）；右侧「DualPipe」= Dual pipe 0–7 各持两个 Layer（0 与 7、1 与 6 … 7 与 0），上方标 F0B0 与 F1B1 两个方向，左右两列的 forward/backward 箭头方向相反 | 采用：把「每 rank 两份参数、两个方向在同一 rank 上对推」画得最清楚，直接支撑 §4 与 §6 里 2× 参数开销的叙述；官方调度图不含层级放置信息 | §4、§6 |
| yeqianshu-dualpipe-source-walkthrough | v2-c23c7161317f8f32d47ccb037e66e1c3_r.jpg | 113 | 作者自绘调度图 | 图 4（step1 nF0，正向 forward）：表格 dual pipe × device0–7，device0 填 0–5、device1 填 0–3、device2 填 0–1、device5 填 10–11、device6 填 10–13、device7 填 10–15，其余空；橙色底 | 不采用：8 个逐步骤表格中的第一个中间态，信息被图 11（完整演算）包含；正文用文字描述 step1 即可 | —（§7 备选） |
| yeqianshu-dualpipe-source-walkthrough | v2-3266f49aeafec4730cfe8756d8df0ae5_r.jpg | 230 | 作者自绘调度图 | 图 5（step2 nF0F1）：表格 device0 填 0-1-2-3-4-5-6-10、device1 填 0-1-2-3-4-10-5-11、device2 填 0-1-2-10-3-11、device3 填 0-10-1-11-2-12-3-13、device4 填 10-0-11-1-12-2-13-3、device5 填 10-11-12-0-13-1-14-2、device6 填 10-11-12-13-14-0-15-1、device7 填 10-11-12-13-14-15-16-0；橙色底 | 不采用：同上，逐步骤中间态；图 11 已给出合并后的完整结果 | — |
| yeqianshu-dualpipe-source-walkthrough | v2-4a63e189d276f0a3b5bb98498707d521_r.jpg | 250 | 作者自绘调度图 | 图 6（step3 nB1W1F1）：表格 device0 行前半为橙色 forward（0…13），后半为绿（10/11/12）与蓝（13）格；右侧图例：橙=Forward计算、绿=输入梯度计算、蓝=权重梯度计算、紫=权重梯度+输入梯度计算 | 不采用：逐步骤中间态；`v2-87c35c47…`（step4）与图 11 已覆盖同类信息 | — |
| yeqianshu-dualpipe-source-walkthrough | v2-87c35c47d0765cb704cd8357b4b02a78_r.png | 412 | 作者自绘调度图 | 图 7（step4 nF0B1F1B0）：表格 device0–7，橙格 forward 编号 0–19 铺满，绿格（输入梯度）与紫格（权重梯度+输入梯度）出现在中后段；同一图例（Forward计算 / 输入梯度计算 / 权重梯度计算 / 权重梯度+输入梯度计算） | 不采用：逐步骤中间态；step4 已在图 11 中体现，且正文重点是「计算通信层叠」而非这张表本身 | —（§7 备选） |
| yeqianshu-dualpipe-source-walkthrough | v2-5444c3b3b992fc4531fa4040638a2b6d_r.png | 456 | 作者自绘调度图 | 图 8（step5 nB1F1B0）：同款表格与图例，橙格到 19，绿/紫格继续右移，全部 forward 在本步完成 | 不采用：逐步骤中间态 | — |
| yeqianshu-dualpipe-source-walkthrough | v2-5c5b248918b6a99f5cc4672c27b00869_r.png | 470 | 作者自绘调度图 | 图 9（step6 nB1B0）：同款表格与图例，绿格（输入梯度）铺满 device0–4，紫格（权重梯度+输入梯度）集中在 device5–7 中后段 | 不采用：逐步骤中间态 | — |
| yeqianshu-dualpipe-source-walkthrough | v2-39ceff9ad80e74e8adf71393c2af1981_r.png | 488 | 作者自绘调度图 | 图 10（step7 nWB0）：同款表格与图例，橙格填满，本步以蓝格（权重梯度）为主，绿格（新输入梯度）收尾 | 不采用：逐步骤中间态 | — |
| yeqianshu-dualpipe-source-walkthrough | v2-c499392417507ebce669c019d20132ad_r.png | 500 | 作者自绘调度图 | step8 nW：同款表格与图例，橙格填满，仅剩 device4–7 的蓝格（权重梯度）收尾，对应正文「为了 zero bubble 的收尾权重梯度计算」 | 不采用：逐步骤中间态 | — |
| yeqianshu-dualpipe-source-walkthrough | v2-8ba9f543c3733d0c162a1a565aff39f5_r.png | 517 | 作者自绘调度图 | **图 11**：把 step1–step8 合并成一张完整演算表——dual pipe × device0–7，每行按时间顺序排列橙格（Forward计算，编号 0–19 与 0–9/10–19 交错）、绿格（输入梯度）、蓝格（权重梯度）、紫格（权重梯度+输入梯度）；右侧保留四色图例；**没有时间轴刻度、没有 overlaps 拼接格、没有空白气泡格标注** | 采用：§7 需要的「一个完整 step 的逐步演算结果」，作者自绘且与官方图有明确差异（见「存疑与分歧」2）；引用时必须写成「社区自绘演算，非官方调度图」 | §7 |
| yeqianshu-dualpipe-source-walkthrough | v2-74cee9df9927e93361c7c8c585215883_r.png | 519 | 论文/官方图转贴 | **图 12**：与官方 DualPipe 调度图逐格一致（Device 0–7、micro-batch 0–9、橙/绿/浅绿/蓝/橙绿拼接、同一图例文字） | 不采用（改引官方）：与官方图重复，应优先引用官方原图 | 对应 `/workspace/algorithm/DualPipe/images/dualpipe.png`＝DeepSeek-V3 报告 Figure 5；§4 |

## 采用图的正文引用块（可直接粘贴）

以下 22 张为「可进正文的社区图」。使用时请把图片复制到 `torch/dualpipe/pics/` 下并使用建议文件名；`<L>` 为社区文章 md 中该图的引用行。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/sea-ai-lab-cut-in-half-mirrored-schedule.png" alt="DualPipe 调度图，每格补全 down-to-up 方向的 micro-batch 编号，右侧标注 The two parts are mirrored，Device 0-3 与 Device 4-7 逐格镜像" style="width: 100%;">
</div>

> **图片来源**：Sea AI Lab《双流并行(DualPipe) 没有双流会更好》（Penghui Qi、Xinyi Wan、Guangxing Huang、Min Lin，2025-02-27，<https://hackmd.io/@ufotalent/S1N_ay0ckx>，英文版 <https://hackmd.io/@ufotalent/r1lVXsa9Jg>）。本地副本：`references/articles/sea-ai-lab-cut-in-half/sea-ai-lab-cut-in-half.md` 第 19 行引用图。
> **层级说明**：底图是官方 DualPipe 调度图（＝`/workspace/algorithm/DualPipe/images/dualpipe.png`＝DeepSeek-V3 报告 Figure 5）的社区改写版，双向 micro-batch 编号与 "The two parts are mirrored." 标注为 Sea AI Lab 作者添加；正文若只讲调度本身，请改引官方原图。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/sea-ai-lab-cut-in-half-vshape.jpg" alt="上栏 DualPipe：8 个 device 各持两个 model layer（0,7 / 1,6 / 2,5 / 3,4 …），F 块排成 V 形；下栏 Cut-in-half：只保留 Device 0-3，同样 layer 配对，V 形变窄" style="width: 85%;">
</div>

> **图片来源**：Sea AI Lab《双流并行(DualPipe) 没有双流会更好》（Penghui Qi、Xinyi Wan、Guangxing Huang、Min Lin，2025-02-27，<https://hackmd.io/@ufotalent/S1N_ay0ckx>）。本地副本：`references/articles/sea-ai-lab-cut-in-half/sea-ai-lab-cut-in-half.md` 第 26 行引用图。
> **层级说明**：作者（Zero Bubble 团队）自绘的**第一手来源**，是 cut-in-half / V 形布局的原始出处，官方 `dualpipev.png` 由该做法导出；不是论文原图。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/sea-ai-lab-cut-in-half-fb-squeeze.png" alt="Cut-in-half 中间调度：Device 0-3，每格两行编号，已无重叠格，F 与 B 解耦后重新挤压排布" style="width: 100%;">
</div>

> **图片来源**：Sea AI Lab《双流并行(DualPipe) 没有双流会更好》（Penghui Qi、Xinyi Wan、Guangxing Huang、Min Lin，2025-02-27，<https://hackmd.io/@ufotalent/S1N_ay0ckx>）。本地副本：`references/articles/sea-ai-lab-cut-in-half/sea-ai-lab-cut-in-half.md` 第 61 行引用图。
> **层级说明**：作者自绘的第一手中间调度图（cut-in-half → ZBV 的过渡态），官方仓库无对应图。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/sea-ai-lab-cut-in-half-zbv.png" alt="ZB-V 调度：Device 0-3，每格两行编号，宽阶梯排布，无重叠格与 optimizer step 分隔" style="width: 100%;">
</div>

> **图片来源**：Sea AI Lab《双流并行(DualPipe) 没有双流会更好》（Penghui Qi、Xinyi Wan、Guangxing Huang、Min Lin，2025-02-27，<https://hackmd.io/@ufotalent/S1N_ay0ckx>）。本地副本：`references/articles/sea-ai-lab-cut-in-half/sea-ai-lab-cut-in-half.md` 第 65 行引用图。
> **层级说明**：作者自绘的第一手 ZB-V 调度图；同一调度在 Zero Bubble 论文中另有渲染（`references/papers/zero-bubble/images/536478895e0abbcea17d6178866439923d5a02d53f94cfd02f19ea5706b189e0.jpg`＝Figure 8），引用论文口径时应并列给出。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/pp-visualization-naive-mp.jpg" alt="Activation Memory Timeline：Device 0-3 各连续 4 格 F0-F3 后接 4 格 B0-B3，backward 阶段设备间有明显 idle 空隙，四条激活包络线均峰值 4" style="width: 100%;">
</div>

> **图片来源**：`flyinghu123/PipelineParallelismDemo` 配套文章《流水线并行可视化demo》（作者 flyinghu123，发布日未在快照中标注，<https://github.com/flyinghu123/PipelineParallelismDemo>）。本地副本：`references/articles/pipeline-parallelism-visualization/pipeline-parallelism-visualization.md` 第 27 行引用图。
> **层级说明**：作者用仓库自带脚本生成的自绘图（含激活显存时间线），是社区材料，不是论文原图；正文引用时请写明「非真实性能测量，为调度模拟的显存/时间占位」。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/pp-visualization-gpipe.jpg" alt="Activation Memory Timeline：Device 0-3 各连续 4 格 F0-F3 后接 4 格 B0-B3，backward 由 Device 3 反向推进到 Device 0，四条包络线在 t≈4-5 同时达到峰值 4" style="width: 100%;">
</div>

> **图片来源**：`flyinghu123/PipelineParallelismDemo` 配套文章《流水线并行可视化demo》（作者 flyinghu123，发布日未在快照中标注，<https://github.com/flyinghu123/PipelineParallelismDemo>）。本地副本：`references/articles/pipeline-parallelism-visualization/pipeline-parallelism-visualization.md` 第 39 行引用图。
> **层级说明**：作者脚本生成的自绘图；调度本身即 GPipe 论文 Figure 2 描述的全前向—全反向调度，但显存包络为社区新增，不属论文原图。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/pp-visualization-1f1b.jpg" alt="Activation Memory Timeline：Device 0-3，micro-batch 0-7，warm-up 阶梯后进入严格 1F1B 稳定期，尾部反向收尾，激活包络呈锯齿平台并峰值为 4" style="width: 100%;">
</div>

> **图片来源**：`flyinghu123/PipelineParallelismDemo` 配套文章《流水线并行可视化demo》（作者 flyinghu123，发布日未在快照中标注，<https://github.com/flyinghu123/PipelineParallelismDemo>）。本地副本：`references/articles/pipeline-parallelism-visualization/pipeline-parallelism-visualization.md` 第 53 行引用图。
> **层级说明**：作者脚本生成的自绘图；调度布局对应 Megatron-LM 论文 Figure 4 上半的 1F1B（PipeDream-Flush），但显存包络与记号（p / vpp / g）为社区自定。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/pp-visualization-symmetric-1f1b.jpg" alt="Activation Memory Timeline：Device 0-3，F 与 B 之间留空的对称 1F1B 排布，激活包络呈连续三角锯齿，峰值 4（作者注明实际不存在该调度）" style="width: 100%;">
</div>

> **图片来源**：`flyinghu123/PipelineParallelismDemo` 配套文章《流水线并行可视化demo》（作者 flyinghu123，发布日未在快照中标注，<https://github.com/flyinghu123/PipelineParallelismDemo>）。本地副本：`references/articles/pipeline-parallelism-visualization/pipeline-parallelism-visualization.md` 第 75 行引用图。
> **层级说明**：作者自绘，且**作者本人在正文注明「实际没有该调度，仅用于理解」**；引用时必须在图注保留这句限定，否则会被误读为真实调度。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/pp-visualization-interleaved-1f1b.jpg" alt="Activation Memory Timeline：Device 0-3，每设备两个 chunk（F-Chunk0/F-Chunk1/B-Chunk0/B-Chunk1），micro-batch 0-7，激活包络峰值升到 6" style="width: 100%;">
</div>

> **图片来源**：`flyinghu123/PipelineParallelismDemo` 配套文章《流水线并行可视化demo》（作者 flyinghu123，发布日未在快照中标注，<https://github.com/flyinghu123/PipelineParallelismDemo>）。本地副本：`references/articles/pipeline-parallelism-visualization/pipeline-parallelism-visualization.md` 第 83 行引用图。
> **层级说明**：作者脚本生成的自绘图；对应 Megatron-LM 论文 Figure 4 下半的 interleaved 1F1B（VPP），显存包络为社区新增。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/xiaodonggua-ep-alltoall-routing.jpg" alt="专家并行前向/反向通路：Attention → LayerNorm → All-to-All Dispatch → Expert0/Expert1 → All-to-All Combine → LayerNorm，标注 token 集合 X0={0,1,2}、X1={10,11,12} 与 dispatch {0,2}、{11}" style="width: 95%;">
</div>

> **图片来源**：知乎 小冬瓜AIGC《【手撕DualPipe】让我们一步步把 MoE EP 通信"消除"（附代码）》（发布日未在快照中标注，<https://zhuanlan.zhihu.com/p/1910995677451912435>）。本地副本：`references/articles/xiaodonggua-shousi-dualpipe/xiaodonggua-shousi-dualpipe.md` 第 24 行引用图。
> **层级说明**：作者自绘示意图（社区解读），不是 DeepSeek 官方图。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/xiaodonggua-moe-pp-ep-comm-scope.jpg" alt="MoE 训练 PP+EP：绿框 PP Stage1（GPU0/GPU1）内部含两次 All-to-All，蓝框 PP Stage2（GPU2/GPU3），两框之间红箭头标 PP stage 之间通信" style="width: 95%;">
</div>

> **图片来源**：知乎 小冬瓜AIGC《【手撕DualPipe】让我们一步步把 MoE EP 通信"消除"（附代码）》（发布日未在快照中标注，<https://zhuanlan.zhihu.com/p/1910995677451912435>）。本地副本：`references/articles/xiaodonggua-shousi-dualpipe/xiaodonggua-shousi-dualpipe.md` 第 28 行引用图。
> **层级说明**：作者自绘示意图（社区解读）；区分「stage 内 All-to-All」与「stage 间 P2P」的画法为本图独有。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/xiaodonggua-dense-vs-moe-block-ops.jpg" alt="左 Dense Block 执行周期（Attention→LayerNorm→MLP→LayerNorm），右 MoE Block EP 执行周期（Attention→LayerNorm→All-to-All Dispatch→Expert0/1→All-to-All Combine→LayerNorm）；底部两行彩色操作块图例：Dense=Attn/MLP/MLP(B)/Attn(B)/PP，MoE=Attn/A2A/MLP/A2A/A2A(B)/MLP(B)/A2A(B)/Attn(B)/PP" style="width: 100%;">
</div>

> **图片来源**：知乎 小冬瓜AIGC《【手撕DualPipe】让我们一步步把 MoE EP 通信"消除"（附代码）》（发布日未在快照中标注，<https://zhuanlan.zhihu.com/p/1910995677451912435>）。本地副本：`references/articles/xiaodonggua-shousi-dualpipe/xiaodonggua-shousi-dualpipe.md` 第 51 行引用图。
> **层级说明**：作者自绘示意图（社区解读）；底部操作块图例是社区版的 F/I/W/B_full/C 记号对照，官方论文没有等价图。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/xiaodonggua-dense-vs-moe-1f1b-bubble.jpg" alt="上 Dense Block 1F1B 执行周期（GPU0-3，格内 Attn+MLP / MLP+Attn(B)），下 MoE EP 1F1B 执行周期（GPU0-3，格内 Attn+A2A+MLP+A2A / A2A+MLP+A2A+Attn(B)）；右侧注：由于 A2A 的存在拉长了一个 Stage 的计算周期，导致 EP+PP 气泡率更高" style="width: 100%;">
</div>

> **图片来源**：知乎 小冬瓜AIGC《【手撕DualPipe】让我们一步步把 MoE EP 通信"消除"（附代码）》（发布日未在快照中标注，<https://zhuanlan.zhihu.com/p/1910995677451912435>）。本地副本：`references/articles/xiaodonggua-shousi-dualpipe/xiaodonggua-shousi-dualpipe.md` 第 55 行引用图。
> **层级说明**：作者自绘调度图（社区解读）；用于说明「stage 内通信等待会被放大成流水线气泡」，非官方性能数据。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/xiaodonggua-ep-serial-vs-overlap.jpg" alt="上 EP 1F1B 计算+通信串行（Attn/A2A dispatch/MLP/A2A Combine/A2A(B) Dispatch/MLP(B)/A2A(B) Combine/Attn(B)/PP），下左 EP 1F1B 通信计算重叠（计算行与通信行上下对齐并标 t_overlap），下右 Dense 1F1B（Attn/MLP/MLP(B)/Attn(B)/PP）" style="width: 100%;">
</div>

> **图片来源**：知乎 小冬瓜AIGC《【手撕DualPipe】让我们一步步把 MoE EP 通信"消除"（附代码）》（发布日未在快照中标注，<https://zhuanlan.zhihu.com/p/1910995677451912435>）。本地副本：`references/articles/xiaodonggua-shousi-dualpipe/xiaodonggua-shousi-dualpipe.md` 第 59 行引用图。
> **层级说明**：作者自绘示意图（社区解读）。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/xiaodonggua-dualpipe-phase1-phase2.jpg" alt="上 DualPipe-Phase1：GPU0-7，1F 阶梯自 GPU0 上行、1B 阶梯下行，中间为深绿 1F1B 带；下 DualPipe-Phase2：同结构，1F 阶梯自 GPU7 下行、1B 上行" style="width: 95%;">
</div>

> **图片来源**：知乎 小冬瓜AIGC《【手撕DualPipe】让我们一步步把 MoE EP 通信"消除"（附代码）》（发布日未在快照中标注，<https://zhuanlan.zhihu.com/p/1910995677451912435>）。本地副本：`references/articles/xiaodonggua-shousi-dualpipe/xiaodonggua-shousi-dualpipe.md` 第 71 行引用图。
> **层级说明**：作者自绘调度图（社区解读）；把两个注入方向拆开画出，官方调度图没有这个视角。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/xiaodonggua-dualpipe-two-microbatch-layout.jpg" alt="GPU0-7 × 时间网格，青=F0、黄=F10、蓝=B0、红=B10，右下角图例 F0/F10/B0/B10，可看到 0 与 10 两格相邻出现的配对关系" style="width: 95%;">
</div>

> **图片来源**：知乎 小冬瓜AIGC《【手撕DualPipe】让我们一步步把 MoE EP 通信"消除"（附代码）》（发布日未在快照中标注，<https://zhuanlan.zhihu.com/p/1910995677451912435>）。本地副本：`references/articles/xiaodonggua-shousi-dualpipe/xiaodonggua-shousi-dualpipe.md` 第 133 行引用图。
> **层级说明**：作者自绘的最简双向布局图（社区解读），每方向 1 个 micro-batch；官方图是每方向 10 个 micro-batch。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/xiaodonggua-dualpipe-five-ops.jpg" alt="仅 GPU0-3 四行，每格下方标注操作名 F0 / F0F0 / F1B1 / B0B1 / B0，展示 5 个复合操作按 rank 递增的出现规律" style="width: 100%;">
</div>

> **图片来源**：知乎 小冬瓜AIGC《【手撕DualPipe】让我们一步步把 MoE EP 通信"消除"（附代码）》（发布日未在快照中标注，<https://zhuanlan.zhihu.com/p/1910995677451912435>）。本地副本：`references/articles/xiaodonggua-shousi-dualpipe/xiaodonggua-shousi-dualpipe.md` 第 153 行引用图。
> **层级说明**：作者自绘调度图（社区解读），给出 {F0, F1F0, F1B1, B0B1, B0} 五操作记号。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/xiaodonggua-official-schedule-eight-steps.jpg" alt="官方 DualPipe 调度图（Device 0-7、micro-batch 0-9）上叠加红色斜线与矩形，把整张调度切成 F0 / F0F1 / B1W1F1 / F0B1F1B0 / B1F1B0 / B1B0 / WB0 / W 八段" style="width: 100%;">
</div>

> **图片来源**：知乎 小冬瓜AIGC《【手撕DualPipe】让我们一步步把 MoE EP 通信"消除"（附代码）》（发布日未在快照中标注，<https://zhuanlan.zhihu.com/p/1910995677451912435>）。本地副本：`references/articles/xiaodonggua-shousi-dualpipe/xiaodonggua-shousi-dualpipe.md` 第 606 行引用图。
> **层级说明**：底图是**官方图**（＝`/workspace/algorithm/DualPipe/images/dualpipe.png`＝DeepSeek-V3 报告 Figure 5），八段红框划分为社区作者添加；引用时请同时给出官方图，并注明标注来源与 `yeqianshu` 走读的 step1–step8 命名一致。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/yeqianshu-three-schedules-compare.jpg" alt="图 1：三栏对比 (a) 先全前向后全反向 (b) 1F1B（顶栏标 nF / n1F1B / nB）(c) dualpipe（顶栏标 F0F1 / F0B1 / F1B0 / B1B0），底部图例前向（黄）/后向（绿）" style="width: 90%;">
</div>

> **图片来源**：知乎 叶千树《理解DualPipe源码和pipeline实现逻辑》（2025-04-08，<https://zhuanlan.zhihu.com/p/1892343442778091583>）。本地副本：`references/articles/yeqianshu-dualpipe-source-walkthrough/yeqianshu-dualpipe-source-walkthrough.md` 第 13 行引用图。
> **层级说明**：**作者自绘**调度演算图（社区解读），不是论文原图；图内 (c) 的设备行标签只有 device0/device2 两行，需在图注说明其含义。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/yeqianshu-fwd-bwd-grad-formulas.jpg" alt="图 2：左侧前向计算图 X→W→output；右侧反向计算图 Grad_output 分送 W 与 X 得到 Grad_x 与 Grad_w；下方列出 output=X@W、Grad_x=Grad_output@W、Grad_w=Grad_output.flatten(0,1).T@X.flatten(0,1) 及 X/W/output 的形状 [b,L,h]、[h,h]" style="width: 90%;">
</div>

> **图片来源**：知乎 叶千树《理解DualPipe源码和pipeline实现逻辑》（2025-04-08，<https://zhuanlan.zhihu.com/p/1892343442778091583>）。本地副本：`references/articles/yeqianshu-dualpipe-source-walkthrough/yeqianshu-dualpipe-source-walkthrough.md` 第 23 行引用图。
> **层级说明**：**作者自绘**推导示意图（社区解读），不是论文公式截图；正文的 I=GΘ / W=GᵀX 记号以 Zero Bubble 论文为准，本图只用于给出形状约定。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/yeqianshu-1f1b-vs-dualpipe-layers.jpg" alt="图 3：左 1F1B Pipeline，pipe 0-7 各持一个 Layer 0-7；右 DualPipe，Dual pipe 0-7 各持两个 Layer（0 与 7、1 与 6 … 7 与 0），上方标 F0B0 与 F1B1 两个相反方向" style="width: 90%;">
</div>

> **图片来源**：知乎 叶千树《理解DualPipe源码和pipeline实现逻辑》（2025-04-08，<https://zhuanlan.zhihu.com/p/1892343442778091583>）。本地副本：`references/articles/yeqianshu-dualpipe-source-walkthrough/yeqianshu-dualpipe-source-walkthrough.md` 第 59 行引用图。
> **层级说明**：**作者自绘**示意图（社区解读），不是官方图。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/yeqianshu-dualpipe-full-derivation.png" alt="图 11：把 step1-step8 合并成一张完整演算表，dual pipe × device0-7，橙格为 Forward 计算（编号 0-19）、绿格为输入梯度、蓝格为权重梯度、紫格为权重梯度+输入梯度" style="width: 100%;">
</div>

> **图片来源**：知乎 叶千树《理解DualPipe源码和pipeline实现逻辑》（2025-04-08，<https://zhuanlan.zhihu.com/p/1892343442778091583>）。本地副本：`references/articles/yeqianshu-dualpipe-source-walkthrough/yeqianshu-dualpipe-source-walkthrough.md` 第 517 行引用图。
> **层级说明**：**作者自绘**的完整演算结果（社区解读），非官方调度图；与官方图（`./pics/dualpipe-official.png`）的差异见「存疑与分歧」2，引用时请与官方图并列并说明差异。

## 与官方图的重复关系

「社区图 ↔ 官方图」对应清单，供正文决定改引官方原图（共 14 张，与汇总里的「应改引原图」一致：11 张改引官方/论文原图 + 3 张改引 Sea AI Lab 一手原图）。

| # | 社区图 | 判定 | 官方对应物 | 处理建议 |
| --- | --- | --- | --- | --- |
| 1 | normaluhr `v2-3d792b47…` | 未改动的官方图复制 | `/workspace/algorithm/DualPipe/images/dualpipe.png` ＝ DeepSeek-V3 报告 Figure 5（`references/papers/deepseek-v3-report/images/e6f2d6c79a43060cbcca6151b7ab9f65a07022a2b6785819b193e4c6639157a4.jpg`） | 改引官方 `dualpipe.png` |
| 2 | normaluhr `v2-341bd07c…` | 未改动的官方图复制 | DeepSeek-V3 报告 Figure 4（`references/papers/deepseek-v3-report/images/ea1a52decfb55f603ceea7de1e14bbb72c8e161ed45f792c75afdcfc85a8b59b.jpg`） | 改引论文 Figure 4 |
| 3 | reku `v2-7e33dc16…` | 未改动的官方图复制（多一道知乎水印） | 同 #1 | 改引官方 `dualpipe.png` |
| 4 | reku `v2-7c04292e…` | 论文图复制 + 作者红框标注 | Megatron-LM 论文 Figure 4 下半（interleaved 1F1B，`references/papers/megatron-lm-gpu-clusters/images/2a3ac338c206e25875fba03f81ced8e175be26545684dd22e150a7c9a8c0dd3d.jpg`） | 改引论文 Figure 4；红框所标的「可掩盖区间」在正文用文字或自行标注表达 |
| 5 | reku `v2-c1cba8a8…` | 论文图的裁切转贴（保留 "(c) Chimera" 图注） | 疑似 Chimera（Li & Hoefler, SC21）中一块含 M_W / M_A 显存柱状图的双向流水线调度图；`references/papers/chimera/` 的 Figure 2（`aee6624e…jpg`）、Figure 3（`270ff454…jpg`）、Figure 8（`5242db91…jpg`）排版均与该图**不一致**，**无法确认具体 Figure 编号** | 正文若要用，先回溯 Chimera 原文确认编号后再引一手图；不要引用该裁切转贴 |
| 6 | xiaodonggua `v2-a9359b74…` | 未改动的官方图复制 | 同 #2（V3 报告 Figure 4） | 改引论文 Figure 4 |
| 7 | xiaodonggua `v2-4ef39e5d…` | 未改动的官方图复制（多一道知乎水印） | 同 #1 | 改引官方 `dualpipe.png` |
| 8 | xiaodonggua `v2-26b15789…` | 与 Sea AI Lab 一手图同图（二手转贴） | 一手版 `references/articles/sea-ai-lab-cut-in-half/images/v2-7c797d3cc827149eadb7beb740d5889f_r.jpg` | 改引 Sea AI Lab 一手图（该图本身建议采用） |
| 9 | xiaodonggua `v2-45e9c6fe…` | 与 Sea AI Lab 一手图同图，且其内容＝官方 DualPipeV | 一手版 `…/sea-ai-lab-cut-in-half/images/v2-e77aeb55ddbe50cb37dfeb213c6ac054_r.png`；官方 `/workspace/algorithm/DualPipe/images/dualpipev.png` | 改引官方 `dualpipev.png`（官方 README 第 16 行说明该调度即 Sea AI Lab 的 cut-in-half） |
| 10 | xiaodonggua `v2-9dd1e161…` | 与 Sea AI Lab 一手图同图（二手转贴） | 一手版 `…/sea-ai-lab-cut-in-half/images/v2-4c4a1cc8bf1e26ee6e2b93be719f4070_r.png` | 改引 Sea AI Lab 一手图 |
| 11 | xiaodonggua `v2-bb4b8477…` | 与 Sea AI Lab 一手图同图（二手转贴） | 一手版 `…/sea-ai-lab-cut-in-half/images/v2-1247bd394c4ed000ce98345349d82f04_r.png` | 改引 Sea AI Lab 一手图 |
| 12 | sea-ai-lab `v2-e77aeb55…` | 未改动的官方图复制（仅加了「前半层/后半层」编号） | `/workspace/algorithm/DualPipe/images/dualpipev.png`（官方 README：「DualPipeV is a concise V-shape schedule derived from DualPipe using a "cut-in-half" procedure, introduced by Sea AI Lab」） | 改引官方 `dualpipev.png`；若要保留「前半层/后半层」编号，须在图注写明为社区标注 |
| 13 | yeqianshu `v2-74cee9df…` | 未改动的官方图复制 | 同 #1 | 改引官方 `dualpipe.png` |
| 14 | reku `v2-e3657f71…` | 外部论文图转贴（本仓库 `papers/` **未**归档该论文） | FLUX: Fast Software-based Communication Overlap On GPUs Through Kernel Fusion（arXiv `2406.06858`）中「Original vs Overlapped」的 TP reduce-scatter/GEMM 融合图；**无法确认具体 Figure 编号** | 正文若要用须引 FLUX 原文（并建议顺手把该论文归档到 `references/papers/`），不要引用知乎截图 |

补充对应（**不构成重复、但引用时必须并列说明**）：

- `sea-ai-lab-cut-in-half/v2-3da883a2…`（采用）：底图＝官方 `dualpipe.png`／V3 报告 Figure 5，Sea AI Lab 补全了双向 micro-batch 编号并加 "The two parts are mirrored." 标注，故保留采用，但正文同时给出官方图。
- `xiaodonggua/v2-02221b07…`（采用）：底图＝官方 `dualpipe.png`／V3 报告 Figure 5，八段红框为社区添加，故保留采用，但正文同时给出官方图。
- `sea-ai-lab-cut-in-half/v2-1247bd39…`（采用，ZB-V）↔ Zero Bubble 论文 Figure 8（`references/papers/zero-bubble/images/536478895e0abbcea17d6178866439923d5a02d53f94cfd02f19ea5706b189e0.jpg`）：同一调度、不同渲染，不是重复；若要引论文口径（$p M_B$ 峰值激活等）应并列给论文 Figure 8。
- `normaluhr/v2-4f06afb3…`（不采用）↔ Zero Bubble 论文 Figure 3（ZB-H1 / ZB-H2，`references/papers/zero-bubble/images/e2c1ba8e49b0c79994431d4c405667b4dfb08dcbaa0d535cfc2dd803e8622016.jpg`）：同主题重绘，正文应引一手。
- `pipeline-parallelism-visualization` 的 5 张 ↔ GPipe 论文 Figure 2、Megatron-LM 论文 Figure 4：调度布局同源，但这 5 张的**激活显存时间线**是社区新增，故判定为采用而非重复。

## 存疑与分歧

1. **`sea-ai-lab-cut-in-half/v2-3da883a2…` 是否「镜像对称示意」——核实成立，但需精确表述。**
   它**不是**一张独立的"镜像示意图"，而是在官方 DualPipe 调度图（`dualpipe.png`）上做了两处改动：(a) 每格从单编号改成上下两行编号，图例写明 `Micros from up to down` / `Micros from down to up`，把 V3 报告 Figure 5 图注里"省略反向 micro-batch ID"的部分补全；(b) 右侧加注 `The two parts are mirrored.`，并在 Device 0–3 与 Device 4–7 之间体现出逐格镜像。所以草稿的说法方向正确，但引用时应写成「官方调度图 + Sea AI Lab 补全的双向编号与镜像对称标注」，避免让读者以为这是官方原图或一张全新的示意图。

2. **`yeqianshu` 图 11 / 图 12 的差异——可从图中确认的部分。**
   可确认：
   - 图 12 是官方调度图的复制（Device 0–7、micro-batch ID 只标单方向 0–9、格子按"共享黑边"表示重叠、图例含 `Overlapped forward & Backward`）；V3 报告 Figure 5 的图注明确说反向 micro-batch 是对称的、故省略其编号。
   - 图 11 把**两个方向的 micro-batch 编号都排在同一条时间线上**（每行可见 0–19 与 0–9/10–19 交错），且用颜色区分 Forward 计算 / 输入梯度 / 权重梯度 / **权重梯度+输入梯度（紫色合并块）**，而官方图用的是 `Backward for input` / `Backward for weights` / `Overlapped forward & Backward` 三分类，并**没有**"权重梯度+输入梯度合并"这一类。
   - 图 11 没有时间轴刻度，也没有官方图那种"共享黑边＝重叠"的记号，因此**无法从图 11 本身确认**它的逐格时间边界是否与官方图一一对齐（这正是作者在图注里提醒"图 12 省略了节点 7 的输入及相关梯度计算演示（对称），一定程度上带来一些混淆性"的原因）。
   - 结论：图 11 的定位是"社区自绘的逐步演算"，与官方调度图是**同一实验的两种表达**，不是同一张图的改写；正文并置时必须写清哪一张是官方、哪一张是社区自绘，且气泡/时间结论以官方图与论文 Table 2 为准。

3. **`reku/v2-c1cba8a8…` 的出处无法确认。** 图内保留 "(c) Chimera" 图注，右侧有 M_W / M_A 两张显存柱状图，P0–P7 八行、P3/P4 间一条虚线。`references/papers/chimera/` 中 Figure 2/3/8 的排版都与它不一致，也不属于 `zero-bubble`、`terapipe`、`pipedream(-2bw)` 的已归档图。保守表述：**疑似转贴自 Chimera（Li & Hoefler, SC21）方向的双向流水线调度图（含 M_W / M_A 显存柱状图），具体 Figure 编号无法确认**；正文若要使用，须先回溯 Chimera 原文。

4. **`reku` 文章的发布时间与内容不一致。** `references/README.md` 记录该文为 2025-02-06，但正文讨论的是 DeepSeek 开源周第 4 天（2025-02-27）才放出的 DualPipe，且反复以"猜想"口吻推演构造动机。引用时必须把它标注为**事前/事后推演的个人判断**，不能当作官方设计说明；其 8 张图也全部是手绘草图或转贴，无一手数据。

5. **`normaluhr` 正文里的 Table 2 被改写错了口径——引用该表必须以 V3 报告 Table 2 为准。**
   社区文正文写「1F1B 激活数据则为 1 x PP」「DualPipe 激活…提升到了 2 x PP + 1」，表格里也只有一列 `1 x` / `1 x` / `2 x + 1`。而 V3 报告 Table 2 是四列（Method / Bubble / Parameter / Activation）：1F1B = 1× / PP，ZB1P = 1× / PP，DualPipe = 2× / PP+1。可见社区文把 **Parameter 列与 Activation 列合并后再误读**，得到"激活 2×PP+1"这种不存在的量。仓库内 `papers/deepseek-v3-report/deepseek-v3-report.md` 第 310 行的原表可作更正依据。
   另外该文第 138 行前一段还写「W 则是激活数据累积的窗口大小」，与论文中 W = "backward for weights" 的定义不符，同样属社区误读。

6. **`yeqianshu` 图 1(c) 的设备行标签不自洽。** 图 1 的 (a)(b) 用 device0/device1 两行，(c) 用 device0/device2 两行，但正文描述的是"device 2 同时存有 layer 1 和 layer 2 的参数，device 1 也同时拥有 layer 1 和 layer 2 的参数"。引用该图时需在图注说明行标签含义，或直接改用官方调度图 + `yeqianshu` 图 3（本文件建议采用的 `yeqianshu-1f1b-vs-dualpipe-layers.jpg`）来讲参数放置。

7. **`pipeline-parallelism-visualization` 的图是调度模拟，不是性能测量。** 5 张图的纵轴是"每设备在飞激活数"、横轴是抽象时间格，作者未给出 p / vpp / g / micro-batch 数与该图的具体配置，也没有 GPU 实测。正文引用时必须声明"调度层面的占位模拟，非真实显存/吞吐测量"，避免与论文 Table 2 的公式口径混用。

8. **`xianodonggua` 自制排布（`v2-34eaa5c3…`、`v2-ed6a8b52…`、`v2-482f980f…`、`v2-cbcfa5a7…`、`v2-52de2c6b…`）与官方调度的关系未交叉验证。** 这 5 张是作者按自己的理解还原/简化出的 20-micro-batch 排布，作者在"分析"节自述第 2 条缺陷是"存在过多的 B0B1 和 F0F1，而非构造成 F1B1"，并自述发现自己的排布就是 Chimera。换言之**它们是被作者自己否定的中间产物**，不能作为官方调度的事实来源。本文件已在结论里统一判为不采用。

9. **未核实项：** 6 篇文章的配图未与原始网页逐张比对（本轮只做了本地文件 ↔ 论文/官方图的对照）；`reku`、`xiaodonggua`、`pipeline-parallelism-visualization` 三篇的确切发布日期在快照中无标注，引用块里已按「发布日未在快照中标注」处理；`sea-ai-lab-cut-in-half` 的知乎转载页（`zhuanlan.zhihu.com/p/26915547331`）与 hackmd 原文是否逐字一致，本轮未核对。
