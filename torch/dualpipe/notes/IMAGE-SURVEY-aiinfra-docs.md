# AIInfra 教材页《PP 并行：1F1B/1F1B Interleaved》图片逐张核对

核对对象：`torch/dualpipe/references/docs/aiinfra-pp-1f1b-interleaved/images/` 下的全部 20 张 `10pipeline*.png`。
配图来源页：`torch/dualpipe/references/docs/aiinfra-pp-1f1b-interleaved/aiinfra-pp-1f1b-interleaved.md`（805 行，页面快照另有同名 .html）。
核对方式：20 张全部用 `read_image` 逐张打开读图，表中「内容」列全部为**看图后**的事实描述（含图中可读文字/数值），未依据文件名或上下文顺序推断。

## 汇总

- `images/` 实际文件数（`ls | wc -l`）：20
- 本清单覆盖行数：20
- 教材示意调度图：20；公式截图：0；表格截图：0；装饰/logo：0
- 建议采用：10 张
- 细分（按图的体裁，避免把 20 张当成同一类东西）：
  - 完整调度时间轴（多 stage × 多 micro-batch）：6 张 —— 01、02、04、05、06、07
  - 排布/结构示意图（无时间轴）：2 张 —— 03、08
  - warmup 阶梯示意：1 张 —— 10
  - 双 rank 分步执行快照（每张只画 1～4 格）：11 张 —— 12、13、14、15、16、17、18、19、20、21、22
- 重要事实：本目录**没有任何公式截图与表格截图**。正文的公式（`(p-1)/m`、`(p-1)(t_f+t_b)/v`、`(1/v)·(p-1)/m`）都是页面 KaTeX 文本而非图片；而第 142、346 行 alt 写作「args 超参设置」的 `10pipeline08/10`，实际是 VPP 排布图与 warmup 阶梯图（详见「存疑与分歧」）。因此本页**不存在**「把公式/表格截图误当插图」的风险，唯一的误配风险来自 alt 文本与画面不符。
- 编号不连续：文件名序列为 01–08、10、12–22，缺 `10pipeline09`、`10pipeline11`（正文本也无对应引用），引用行共 21 处、覆盖 20 个唯一文件（`10pipeline19.png` 被第 654、717 行重复引用两次）。


> **〔第 3 轮订正〕** `10pipeline06.png` 被本文件按教材页标题记为「ZB-V 调度，排布呈 V 形」，这是错的。第 3 轮用同一套像素颜色分类方法把它与 Zero Bubble Figure 3 下栏逐格比对，四行的 F/B/W/Optimizer 序列 **39/39、39/39、39/39、39/39 完全相同**（含 7/5/3/1 的 warm-up 与逐行错位的 optimizer 列），画面上每格只有一个数字、也没有白/黑两色文字，看不出 V 形。**结论：这张图的内容是 ZB-H2，教材页把它放在了「ZB-V schedule」标题下。** 正文因此不再引用它（要看 ZB-V 用 Zero Bubble Figure 8，要看 ZB-H2 用 Figure 3 下栏）。
>
> 这一条也说明一件事：**教材页、社区文章里的图，标题与内容可能不一致**，采用前至少要与一手论文图做一次逐格比对。


## 逐张清单

| 文件名 | md 引用行 | 所在小节 | 类别 | 内容（读图后的事实描述，含图中可读的关键标注/数值） | 结论 | 建议章节 |
| --- | --- | --- | --- | --- | --- | --- |
| `10pipeline01.png` | 13 | `## PipeDream 基本原理 #`（第 9 行，图注却写「Gpipeline 原理」） | 教材示意调度图（时间轴） | GPipe 调度时间轴，4 行 Device 1–4。图中文字：`warmup`（黄底）、`cooldown`（黄底）、`Pipeline flush`（一条竖黑粗线）、`Devices idle`（指向灰格）、`Time →`；图例 `Forward Pass`（蓝）/`Backward Pass`（绿）。micro-batch 编号两轮：`1…8` 与 `9 10 11 12 13 14 15 16`；Device 1 从第 1 列起、Device 4 从第 4 列起逐级右移，尾部大片灰格 | **采用** | 第 3 章（1F1B → Zero Bubble 的起点对照：先给出「n=8 份激活全驻留 + 尾部 flush」的 Gpipe 基线）；第 1 章讲「流水线气泡」时也可用 |
| `10pipeline02.png` | 29 | `## PipeDream 基本原理 #` | 教材示意调度图（时间轴） | PipeDream/1F1B 调度时间轴，4 行 Device 1–4。图中文字：`warmup`（黄底）、`稳定阶段`（黄底，在 Device 4 下方）、`cooldown`（黄底）、`Time →`；图例 `Forward Pass`（蓝）/`Backward Pass`（绿）。micro-batch 编号 `1…12`；稳态段每行呈 `F/B` 交替（如 Device 1：`1` 蓝 → `5` 蓝旁夹绿 `1`），Device 4 稳态段为连续 `1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8` | **采用** | 第 3 章（1F1B 的标准时间轴，是「气泡仍是 (p-1)/m」的直接视觉证据） |
| `10pipeline03.png` | 37 | `## Virtual pipeline 基本原理 #`（第 31 行，图注只写「原理」） | 教材示意调度图（排布结构，无时间轴） | 上下两栏对比图。上栏标题 `Pipeline`：GPU1={L1,L2}、GPU2={L3,L4}、GPU3={L5,L6}、GPU4={L7,L8}，斜箭头 L2→L3、L4→L5、L6→L7。下栏标题 `Virtual Pipe`：GPU1={L1,L5}、GPU2={L2,L6}、GPU3={L3,L7}、GPU4={L4,L8}，箭头 L1→L2→L3→L4 直行、L4 折返指回 GPU1 的 L5 再 L5→L6→L7→L8 | **采用** | 第 3 章（Interleaved 1F1B/VPP 的层归属主图）；第 5 章讲 DualPipeV「一个 device 持有多段」时可作 VPP 侧对照，但须注明这是 Megatron VPP 而非 DualPipeV |
| `10pipeline04.png` | 43 | `## Virtual pipeline 基本原理 #` | 教材示意调度图（时间轴） | Interleaved 1F1B / VPP 调度时间轴，4 行 Device 1–4，同样有 `Pipeline flush` 竖黑线、`Time →`。与 02 的差别可见于配色：蓝/浅蓝、绿/深绿**四种**格子（对应 v=2 的两个 model chunk），每行出现「浅色小块 + 深色大块」交替；micro-batch 编号仍为 `1…12`，稳态段每行同时存在两套编号序列（如 Device 1 出现 `7 1 8 2 5 3 6 4 7 1 8 2 3 4`） | **采用** | 第 3 章（VPP 时间轴，与 03 配对：03 讲层归属、04 讲层归属落到时间上后气泡如何变小）；第 5 章也可复用 |
| `10pipeline05.png` | 67 | `### PipeDream-2BW #`（第 63 行） | 教材示意调度图（时间轴 + 权重版本标注） | PipeDream-2BW 调度时间轴，4 行 Worker 1–4，图例 `Forward Pass`（蓝）/`Backward Pass`（绿）、`Time →`。稳态段每行 `F/B` 交替，编号 1–8 循环。右上方标注：`Before: W₁⁽⁰⁾, W₁⁽⁰⁾` / `After: W₁⁽⁰⁾, W₁⁽⁴⁾`；右下方标注：`Before: W₄⁽⁰⁾, W₄⁽⁰⁾` / `After: W₄⁽⁰⁾, W₄⁽⁴⁾`；横轴末端标注 `t = 21`。Worker 1–4 各有一个**棋盘格纹**的绿格（格内编号 4），即被双缓冲/延迟应用的权重版本落点 | **采用** | 第 3 章（PipeDream-2BW = 权重双缓冲，是「延迟 W」家族的重要一环）；第 6 章讨论显存/版本口径时也可引用 |
| `10pipeline06.png` | 73 | `### ZB-V schedule #`（第 69 行） | 教材示意调度图（时间轴） | **实为 ZB-H2 调度（教材页标题误标为 ZB-V）**，4 行 Device 1–4，图例四种：`F`（蓝）、`B`（青）、`W`（深绿）、`Optimizer step`（米黄）。排布呈 V 形：Device 1 先跑 `1 2 3 4 5 6 7`（F/B 交替，蓝青相间）后接 `1 1 8 2 2 3 3 4 4 5 5 6 6 7 7 8 8`，Device 2/3/4 起点依次右移一格；每行末端各有一个米黄 `Optimizer step` 格，且落在不同时刻（Device 1 最早、Device 4 最晚）。`Time →` | **采用** | 第 3 章（Zero Bubble 家族 + V 形回绕的直接图解）；第 5 章讲 V 形时可作「ZB-V ≠ DualPipeV」的对照，必须显式区分 |
| `10pipeline07.png` | 79 | `### Hanayo wave-like pipeline #`（第 75 行） | 教材示意调度图（时间轴，双面板） | 两面板。(a) 标题 `(a) wave=2, devices=8`：8 行 P0–P7，格内编号 0–7，浅绿/橙/粉三色，中段起出现波状错位，右上标注 `Flush`。(b) 标题 `(b) wave=2 and wave=4, devices=4`：左半 4 行 P0–P3（wave=2，编号 0–3，末端 `Flush`），右半 4 行 P0–P3（wave=4，编号 0–3 与 `0 1 2 3` 长串混排，末端 `Flush`） | **采用** | 第 3 章（波浪式调度，谱系的最新一支）；第 9 章选型讨论「气泡降到 1/(2W) 且不复制模型」时引用 |
| `10pipeline08.png` | 142 | `## 分布式框架里的 PP 实现 #`（第 81 行；图注写「args 超参设置」，与画面不符） | 教材示意调度图（排布结构，无时间轴） | 上下两栏。上栏标题 `非VPP场景`：GPU1→GPU2→GPU3→…→GPUP 五块，各块内标 `L1`、`L2`、`L3`、`LP`，块间箭头分别标 `1`、`2`、`P-1`。下栏标题 `VPP场景`：虚线框 `V1` 内为 GPU1..GPUP 依次 `L1`、`L2`、`L3`、`LP1`（箭头 `1`、`2`、`P-1`），虚线框 `V2` 内为 `L_{P+1}`、`L_{P+2}`、`L_{P+3}`、`L_{P2}`（箭头同样 `1`、`2`、`P-1`），V1 右端有一条线回绕指向 V2 左端；下方还有省略号表示更多 V 段 | **采用**（作为 03 的补充，不是主图） | 第 3 章（补足 03 没有的「层区间记号」：`L1..LP1` 与 `L_{P+1}..L_{P2}` 两个 chunk 的划分与回绕顺序，正是正文 `(1/v)·(p-1)/m` 的图形前提）。若第 3 章只留一张 VPP 图，留 03 |
| `10pipeline10.png` | 346 | `## 分布式框架里的 PP 实现 #`（图注写「args 超参设置」，与画面不符） | 教材示意调度图（warmup 阶梯） | 标题 `每个NPU num_warmup_microbatches不同`。5 行 `NPU0`、`NPU1`、`NPU2`、`…`、`NPUP-1`，用细横线作各自时间轴。每行左侧是一串蓝色小方块（格内 `1`、`2`、`3`、`…`、`P-1`），方块起点逐行右移一格；随后错位出现黄色单格（格内 `1`），也逐行右移。最底部注释：`NPUP-1 上 num_warmup_microbatches=0，执行1个F后进入1F1B状态`，并用红色箭头指向 NPUP-1 行第一个蓝格与紧邻黄格的交界 | **采用** | 第 8 章（Megatron 1F1B 对照：`num_warmup_microbatches = pp_size - rank - 1` 的图形化）；第 7 章源码走读讲 `forward_backward_pipelining_without_interleaving` 的 warmup 计算时也可引用 |
| `10pipeline12.png` | 379 | `## 分布式框架里的 PP 实现 #` | 教材示意调度图（双 rank 分步快照） | 只有两行 `NPU0`、`NPU1` 的细时间轴。NPU0 上 1 个蓝格 `1`，右侧文字 `NPU0 完成 F1`；NPU1 行为空。图例：`Forward 计算1个 Micro Batch`（蓝）/`Backward 计算1个 Micro Batch`（黄） | **不采用** | — |
| `10pipeline13.png` | 430 | `## 分布式框架里的 PP 实现 #` | 教材示意调度图（双 rank 分步快照） | 同 12 的版式。NPU0 上 1 个蓝格 `1`，文字 `NPU0 发送 F1`；NPU1 行为空；图例同 12 | **不采用** | — |
| `10pipeline14.png` | 516 | `## 分布式框架里的 PP 实现 #` | 教材示意调度图（双 rank 分步快照） | 同版式。NPU0 上 2 个相邻蓝格 `1`、`2`；NPU1 行为空；图例同 12 | **不采用** | — |
| `10pipeline15.png` | 520 | `## 分布式框架里的 PP 实现 #` | 教材示意调度图（双 rank 分步快照） | 同版式。NPU0 上 2 个蓝格 `1`、`2`，文字 `NPU0 完成 F1 执行 F2`；NPU1 上 1 个蓝格 `1`（与 NPU0 的第 2 格对齐），文字 `NPU1 执行 F1`；图例同 12 | **不采用** | — |
| `10pipeline16.png` | 554 | `## 分布式框架里的 PP 实现 #` | 教材示意调度图（双 rank 分步快照） | 与 15 画面几乎一致（同一版式与格子位置）。差异只在文字：`NPU0 完成 F1 完成 F2`、`NPU1 完成 F1`；图例同 12 | **不采用** | — |
| `10pipeline17.png` | 473 | `## 分布式框架里的 PP 实现 #` | 教材示意调度图（双 rank 分步快照） | 同版式。NPU0 上 2 个蓝格 `1`、`2`，文字 `NPU0 完成 F2`；NPU1 行为空；图例同 12 | **不采用** | — |
| `10pipeline18.png` | 601 | `## 分布式框架里的 PP 实现 #` | 教材示意调度图（双 rank 分步快照） | 同版式。NPU0 上 3 个蓝格 `1`、`2`、`3`；NPU1 上 1 个蓝格 `1` 紧跟 1 个黄格 `B1`，红色箭头指向两者交界，文字 `NPU1 进入 IF1B 状态`（图中写作 `IF1B`）；图例同 12 | **不采用** | — |
| `10pipeline19.png` | 654 与 717（同一文件被引两次） | `## 分布式框架里的 PP 实现 #` | 教材示意调度图（双 rank 分步快照） | 同版式。NPU0 上 `1`、`2`、`3`（蓝）+ `B1`（黄），文字 `NPU0 进入 IF1B 状态`；NPU1 上 `1`（蓝）+ `B1`（黄），文字 `NPU1 进入 IF1B 状态`；图例同 12 | **不采用** | — |
| `10pipeline20.png` | 723 | `## 分布式框架里的 PP 实现 #` | 教材示意调度图（双 rank 分步快照） | 同版式。NPU0 上 `1`、`2`、`3`（蓝）+ `B1`、`B2`（黄），文字 `NPU0 进入 cooldown by pass`；NPU1 上 `1`（蓝）`B1`（黄）`2`（蓝）`B2`（黄），文字 `NPU1 进入 IF1B 状态`；图例同 12 | **不采用** | — |
| `10pipeline21.png` | 729 | `## 分布式框架里的 PP 实现 #` | 教材示意调度图（双 rank 分步快照） | 同版式。NPU0 上 `1`、`2`、`3`（蓝）+ `B1`、`B2`（黄），文字 `NPU0 进入 cooldown by pass`；NPU1 上 `1 B1 2 B2 3 B3`（蓝黄交替），文字 `NPU1 进入 IF1B 状态`；图例同 12 | **不采用**（其中 `cooldown by pass` 这条文字标注若确需引用，见采纳的 22 的说明） | — |
| `10pipeline22.png` | 735 | `## 分布式框架里的 PP 实现 #` | 教材示意调度图（双 rank 分步快照） | 同版式中信息最完整的一张。NPU0 上 `1`、`2`、`3`（蓝）+ `B1`、`B2`、`B3`（黄）；NPU1 上 `1 B1 2 B2 3 B3`（蓝黄交替）；两条时间轴上无「IF1B」错别字，只在 NPU1 时间轴末端画一条**青色竖线**并标注 `梯度更新`；图例同 12 | **采用**（分步快照系列只保留这一张） | 第 8 章（`p=2` 的 1F1B 全流程紧凑示意：warmup 1 个 F → 稳态 F/B 交替 → cooldown 收尾 → 梯度更新时刻，是整个「分步快照」小系列唯一承载完整生命周期的图） |

## 采用图的正文引用块（可直接粘贴）

以下 10 张为建议采用。落地时请把源图从 `references/docs/aiinfra-pp-1f1b-interleaved/images/<原名>` 复制到 `torch/dualpipe/pics/<建议文件名>`（`pics/` 目录当前不存在，需新建；相对路径按 `deep-dive.md` 所在目录 `torch/dualpipe/` 计算）。

### 1. `aiinfra-pp-gpipe-flush-idle.png`（源 `10pipeline01.png`，第 13 行）

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/aiinfra-pp-gpipe-flush-idle.png" alt="GPipe 调度时间轴：Device 1-4，micro-batch 1-8 与 9-16 两轮，warmup/稳定/cooldown 三段，Pipeline flush 竖线与 Devices idle 灰格，蓝=Forward Pass，绿=Backward Pass" style="width: 100%;">
</div>

> **图片来源**：Infrasys-AI AIInfra 教材页《PP 并行：1F1B/1F1B Interleaved》（https://infrasys-ai.github.io/aiinfra-docs/04Train02ParallelAdv/08PPInterleaved.html）。本地副本：`references/docs/aiinfra-pp-1f1b-interleaved/aiinfra-pp-1f1b-interleaved.md` 第 13 行引用图。该图为教材示意图，非论文原图。

### 2. `aiinfra-pp-1f1b-pipedream-timeline.png`（源 `10pipeline02.png`，第 29 行）

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/aiinfra-pp-1f1b-pipedream-timeline.png" alt="PipeDream 1F1B 调度时间轴：Device 1-4，micro-batch 1-12，warmup/稳定阶段/cooldown 三段，稳态段每行 Forward 与 Backward 交替，蓝=Forward Pass，绿=Backward Pass" style="width: 100%;">
</div>

> **图片来源**：Infrasys-AI AIInfra 教材页《PP 并行：1F1B/1F1B Interleaved》（https://infrasys-ai.github.io/aiinfra-docs/04Train02ParallelAdv/08PPInterleaved.html）。本地副本：`references/docs/aiinfra-pp-1f1b-interleaved/aiinfra-pp-1f1b-interleaved.md` 第 29 行引用图。该图为教材示意图，非论文原图。

### 3. `aiinfra-pp-vpp-layer-device-mapping.png`（源 `10pipeline03.png`，第 37 行）

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/aiinfra-pp-vpp-layer-device-mapping.png" alt="Pipeline 与 Virtual Pipe 层归属对比：上栏 GPU1=L1,L2 / GPU2=L3,L4 / GPU3=L5,L6 / GPU4=L7,L8；下栏虚拟化后 GPU1=L1,L5 / GPU2=L2,L6 / GPU3=L3,L7 / GPU4=L4,L8，L4 折返回 GPU1 的 L5" style="width: 45%;">
</div>

> **图片来源**：Infrasys-AI AIInfra 教材页《PP 并行：1F1B/1F1B Interleaved》（https://infrasys-ai.github.io/aiinfra-docs/04Train02ParallelAdv/08PPInterleaved.html）。本地副本：`references/docs/aiinfra-pp-1f1b-interleaved/aiinfra-pp-1f1b-interleaved.md` 第 37 行引用图。该图为教材示意图，非论文原图。

### 4. `aiinfra-pp-interleaved-1f1b-vpp-layout.png`（源 `10pipeline04.png`，第 43 行）

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/aiinfra-pp-interleaved-1f1b-vpp-layout.png" alt="Interleaved 1F1B / VPP 调度时间轴：Device 1-4，每个 device 有两种深浅的 Forward/Backward 格子（v=2 的两个 model chunk），micro-batch 1-12，含 Pipeline flush 竖线" style="width: 100%;">
</div>

> **图片来源**：Infrasys-AI AIInfra 教材页《PP 并行：1F1B/1F1B Interleaved》（https://infrasys-ai.github.io/aiinfra-docs/04Train02ParallelAdv/08PPInterleaved.html）。本地副本：`references/docs/aiinfra-pp-1f1b-interleaved/aiinfra-pp-1f1b-interleaved.md` 第 43 行引用图。该图为教材示意图，非论文原图。

### 5. `aiinfra-pp-pipedream-2bw-double-buffered-weight.png`（源 `10pipeline05.png`，第 67 行）

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/aiinfra-pp-pipedream-2bw-double-buffered-weight.png" alt="PipeDream-2BW 调度时间轴：Worker 1-4，Forward/Backward 交替，棋盘格标出权重版本落点，并标注 Before W1(0),W1(0) / After W1(0),W1(4)，Before W4(0),W4(0) / After W4(0),W4(4)，t=21" style="width: 85%;">
</div>

> **图片来源**：Infrasys-AI AIInfra 教材页《PP 并行：1F1B/1F1B Interleaved》（https://infrasys-ai.github.io/aiinfra-docs/04Train02ParallelAdv/08PPInterleaved.html）。本地副本：`references/docs/aiinfra-pp-1f1b-interleaved/aiinfra-pp-1f1b-interleaved.md` 第 67 行引用图。该图为教材示意图，非论文原图。

### 6. `aiinfra-pp-zbv-schedule.png`（源 `10pipeline06.png`，第 73 行）

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/aiinfra-pp-zbv-schedule.png" alt="ZB-V 调度时间轴：Device 1-4 的 V 形排布，图例 F/ B/ W/ Optimizer step 四色，各 device 末端在不同时刻触发 Optimizer step" style="width: 100%;">
</div>

> **图片来源**：Infrasys-AI AIInfra 教材页《PP 并行：1F1B/1F1B Interleaved》（https://infrasys-ai.github.io/aiinfra-docs/04Train02ParallelAdv/08PPInterleaved.html）。本地副本：`references/docs/aiinfra-pp-1f1b-interleaved/aiinfra-pp-1f1b-interleaved.md` 第 73 行引用图。该图为教材示意图，非论文原图。

### 7. `aiinfra-pp-hanayo-wave-like-wave2-wave4.png`（源 `10pipeline07.png`，第 79 行）

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/aiinfra-pp-hanayo-wave-like-wave2-wave4.png" alt="Hanayo 波浪式流水线：(a) wave=2, devices=8，P0-P7 波状交错含 Flush；(b) wave=2 与 wave=4, devices=4，P0-P3 两种波数的排布对比" style="width: 100%;">
</div>

> **图片来源**：Infrasys-AI AIInfra 教材页《PP 并行：1F1B/1F1B Interleaved》（https://infrasys-ai.github.io/aiinfra-docs/04Train02ParallelAdv/08PPInterleaved.html）。本地副本：`references/docs/aiinfra-pp-1f1b-interleaved/aiinfra-pp-1f1b-interleaved.md` 第 79 行引用图。该图为教材示意图，非论文原图。

### 8. `aiinfra-pp-vpp-chunk-notation-non-vpp-contrast.png`（源 `10pipeline08.png`，第 142 行）

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/aiinfra-pp-vpp-chunk-notation-non-vpp-contrast.png" alt="非VPP 场景与 VPP 场景对比：非VPP 下 GPU1..GPUP 依次持 L1..LP；VPP 下虚线框 V1 为 L1..LP1、V2 为 L_{P+1}..L_{P2}，V1 右端回绕接入 V2 左端" style="width: 60%;">
</div>

> **图片来源**：Infrasys-AI AIInfra 教材页《PP 并行：1F1B/1F1B Interleaved》（https://infrasys-ai.github.io/aiinfra-docs/04Train02ParallelAdv/08PPInterleaved.html）。本地副本：`references/docs/aiinfra-pp-1f1b-interleaved/aiinfra-pp-1f1b-interleaved.md` 第 142 行引用图。该图为教材示意图，非论文原图。**注意**：该页第 142 行的 alt 文本写作「args 超参设置」与画面内容不符，引用时以本表描述为准。

### 9. `aiinfra-pp-1f1b-warmup-microbatches-staircase.png`（源 `10pipeline10.png`，第 346 行）

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/aiinfra-pp-1f1b-warmup-microbatches-staircase.png" alt="1F1B warmup 阶梯示意：NPU0 到 NPUP-1 每行的 num_warmup_microbatches 不同，蓝色 Forward 格编号 1..P-1 逐行右移，黄色 Backward 格错位出现；底部标注 NPUP-1 上 num_warmup_microbatches=0，执行 1 个 F 后进入 1F1B 状态" style="width: 85%;">
</div>

> **图片来源**：Infrasys-AI AIInfra 教材页《PP 并行：1F1B/1F1B Interleaved》（https://infrasys-ai.github.io/aiinfra-docs/04Train02ParallelAdv/08PPInterleaved.html）。本地副本：`references/docs/aiinfra-pp-1f1b-interleaved/aiinfra-pp-1f1b-interleaved.md` 第 346 行引用图。该图为教材示意图，非论文原图。**注意**：该页第 346 行的 alt 文本同样写作「args 超参设置」，与画面不符。

### 10. `aiinfra-pp-1f1b-steady-cooldown-two-rank-trace.png`（源 `10pipeline22.png`，第 735 行）

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/aiinfra-pp-1f1b-steady-cooldown-two-rank-trace.png" alt="两个 rank 的 1F1B 全流程快照：NPU0 为 F1 F2 F3 + B1 B2 B3，NPU1 为 F1 B1 F2 B2 F3 B3，末端青色竖线标注梯度更新时刻；蓝=Forward 计算 1 个 Micro Batch，黄=Backward 计算 1 个 Micro Batch" style="width: 80%;">
</div>

> **图片来源**：Infrasys-AI AIInfra 教材页《PP 并行：1F1B/1F1B Interleaved》（https://infrasys-ai.github.io/aiinfra-docs/04Train02ParallelAdv/08PPInterleaved.html）。本地副本：`references/docs/aiinfra-pp-1f1b-interleaved/aiinfra-pp-1f1b-interleaved.md` 第 735 行引用图。该图为教材示意图，非论文原图。

## 落地建议（本次未执行）

```bash
cd torch/dualpipe
mkdir -p pics
S=references/docs/aiinfra-pp-1f1b-interleaved/images
cp $S/10pipeline01.png pics/aiinfra-pp-gpipe-flush-idle.png
cp $S/10pipeline02.png pics/aiinfra-pp-1f1b-pipedream-timeline.png
cp $S/10pipeline03.png pics/aiinfra-pp-vpp-layer-device-mapping.png
cp $S/10pipeline04.png pics/aiinfra-pp-interleaved-1f1b-vpp-layout.png
cp $S/10pipeline05.png pics/aiinfra-pp-pipedream-2bw-double-buffered-weight.png
cp $S/10pipeline06.png pics/aiinfra-pp-zbv-schedule.png
cp $S/10pipeline07.png pics/aiinfra-pp-hanayo-wave-like-wave2-wave4.png
cp $S/10pipeline08.png pics/aiinfra-pp-vpp-chunk-notation-non-vpp-contrast.png
cp $S/10pipeline10.png pics/aiinfra-pp-1f1b-warmup-microbatches-staircase.png
cp $S/10pipeline22.png pics/aiinfra-pp-1f1b-steady-cooldown-two-rank-trace.png
```

## 存疑与分歧

1. **本章图集完全不含 DualPipe / DualPipeV 自身画面（最重要）**。20 张全部来自 Gpipe / PipeDream / Megatron VPP / 2BW / ZB-V / Hanayo 这条线，没有一张画双向布局或 `(F₀,B₁,F₁,B₀)` 配对，也没有一张画 cut-in-half 的 V 形切分。因此拟定章节结构中的**第 4 章（DualPipe 双向布局）与第 5 章（DualPipeV cut-in-half）无法靠这批图支撑**，仍需使用 DualPipe 论文原图或仓库自有绘图（`references/articles/` 下有 `sea-ai-lab-cut-in-half`、`normaluhr-dualpipe-explained-zh` 等带 `images/` 的文章可另作来源，但同样需按其原始出处标注）。本清单中的 03/06/08 只能作为「VPP / V 形」的**对照**，不能代替 DualPipeV 的图，引用时必须写明二者不是同一机制。
2. **alt 文本与画面不符两处**：第 142 行（`10pipeline08.png`）与第 346 行（`10pipeline10.png`）的 md alt 都写作「args 超参设置」，但实际画面分别是 VPP 排布图和 warmup 阶梯图，两张都不是超参表截图。推测教材页原文此处应有配置表格截图（对应缺失的 `10pipeline09` / `10pipeline11`？），本快照中该图被替换或错位。若后续引用这两张图，不要沿用原 alt 文本。
3. **`10pipeline19.png` 被重复引用**：第 654 行与第 717 行两处都引用同一文件，但两处正文描述的是不同事件（「NPU0 反向执行 Stage0」与「NPU1 反向执行 Stage1」）。这属于原页面的图文错配，不构成我们这边的重复采用问题（该图已判为不采用）。
4. **`IF1B` 拼写错误**：`10pipeline18/19/20/21` 图中的文字标注写作 `IF1B`（应为 `1F1B`）。采纳的 `10pipeline22.png` 没有这处错字，这也是它优先于 21 的原因之一；若最终必须使用 18–21 中任一张，图注需自行纠正。
5. **`10pipeline22.png` 与 `10pipeline21.png` 的取舍**：两张画面高度相似，差别是 21 在 NPU0 行标注 `NPU0 进入 cooldown by pass`、22 在 NPU1 行末端画青色竖线并标注 `梯度更新`。本清单选 22，因为它同时呈现了两个 rank 的完整 F/B 序列与梯度更新时刻；若正文需要 `cooldown by pass` 这一措辞（它对应源码里 cooldown 阶段关闭/开启 grad sync 的语义），可从 21 取词而不必再贴一张图。
6. **`10pipeline05.png` 的权重上标辨识度有限**：图中 `Before/After` 处的 `W₁⁽⁰⁾`、`W₁⁽⁴⁾`、`W₄⁽⁰⁾`、`W₄⁽⁴⁾` 上标字号很小，上述读法为放大后的判读结果；引用时图注建议只写「权重版本号（0）→（4）的双缓冲切换，t=21」，不要写死具体下标含义。
7. **`t = 21` 未经独立验证**：该数值是图中文字，本文档只记录「图中写了什么」，未核对 2BW 原论文里同一示意图的步数口径。若要写进正文，需另找 2BW 原论文核对。
8. **`10pipeline03.png` 的分辨率**：711×522，是全部采用图中最小的一张（约 30 KB）。放大到正文宽度会有明显模糊，建议按 45% 宽使用；若不能接受，替代方案是用 `10pipeline08.png`（777×546）顶替，但会失去「L2→L3 斜箭头」这层直观的跨 device 依赖。
9. **未核对的来源链**：本页参考文献（第 799–805 行）只给了 3 篇知乎/CSDN 博客与 Megatron-LM 仓库，未逐图标注原始出处。因此本清单只能确认「图出自 AIInfra 教材页」，**无法进一步确认教材页自己画的是否与论文原图一致**（例如 01 与 GPipe 论文 Figure、05 与 2BW 论文 Figure、07 与 Hanayo 论文 Figure 的对应关系）。按任务要求，引用时统一标注为「教材示意图，非论文原图」。
10. **画像分辨率与清晰度**：`10pipeline06.png`（1202×180）、`10pipeline18.png`（1174×315）等高宽比极扁，正文排版时需注意纵向留白；`10pipeline04.png` 原图 2117×390 在读取时被下采样，实际使用不受影响（本地文件为原始尺寸）。
