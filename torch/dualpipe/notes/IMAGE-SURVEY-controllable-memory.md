# Pipeline Parallelism with Controllable Memory（arXiv 2405.15362v1）图片逐张核对

## 汇总

- `images/` 实际文件数（`ls | wc -l`）：**49**
- 本清单覆盖行数：**49**
- 真插图：**37**；公式截图：**3**；表格截图：**9**；页眉/装饰/碎片：**0**
- 建议采用：**23 张**（另 14 张真插图为「可选」，见逐张清单「结论」列）

补充事实（核对用）：

- `controllable-memory.md` 全文 418 行，共 **37** 处 `![](images/...)` 引用，涉及 **37 个不同文件**（无重复引用）。
- `49 − 37 = 12` 个文件**未被 md 引用**；逐张读图后确认这 12 张**全部**是公式截图（3）或表格截图（9），**没有一张是丢失的插图**。原因是 MinerU 已把公式转成 `$$…$$` LaTeX、把表格转成内联 `<table>` HTML，同时又保留了 PDF 里的裁切图。
- 分类统计：49 = 37 真插图 + 9 表格截图 + 3 公式截图 + 0 页眉/装饰/碎片。**本目录不存在页眉、logo、装饰条或图内碎片类图片**（最小文件 3 483 B 的 `54b87dad…jpg` 也是公式，不是装饰）。
- 建议采用的 23 张覆盖 Figure 1、2、3(a–d)、4、5(a–b)、6、9(a–c)、10、11、14(a–e)、16、17、18。
- 逐张清单首列为了表格可读性写成缩写 `<前 8 位十六进制>…<后 6 位>.jpg`（49 个前缀两两互不相同，无歧义）；**完整文件名见文末「全量文件名索引」**，可用于 `ls` 逐一核对。图片实际路径为 `torch/dualpipe/references/papers/controllable-memory/images/<完整文件名>`。

## 逐张清单

按 `controllable-memory.md` 中引用行号升序排列，未被引用的 12 张列在末尾。

| 文件名 | md 引用行 | 图号/表号 | 类别 | 内容（读图后的事实描述，含图中可读的关键标注/数值） | 结论 | 建议章节 |
| --- | --- | --- | --- | --- | --- | --- |
| `84cacb5f…c1ad0c.jpg` | 36 | Figure 1 | 真插图 | 左右两栏。左「(a) 1F1B」、右「(b) Eager 1F1B」。每栏上排＝building block 重复铺开（不同深浅灰表示不同 minibatch，彩色高亮出被复用的那个 block），中间向下箭头标 `Squeeze`，下排＝挤压后的调度。栏内标注两处横向量：`Lifespan`（跨 stage 的长横线）与 `Interval`（起点到起点的短横线）。彩色格子只有蓝 `F`、绿 `B`，**没有 W**——印证 1F1B 把 B 与 W 合并 | 采用 | 2（最小概念模型）、3（1F1B→ZB 演进） |
| `7f4661e0…f4c40.jpg` | 77 | Figure 2 | 真插图 | 左「Parallel」、右「V-Shape」两个 device 放置对比。两侧都画 3 条红色 device 时间轴（上 l₁、中 l₂、下 l₃），蓝方块＝F、绿方块＝B，虚线箭头表示 stage 间依赖。左侧 device 1 上同时挂 **l₁ 与 l₄**、device 2 挂 l₂+l₅、device 3 挂 l₃+l₆；右侧 device 1 挂 **l₁ 与 l₆**。这正是正文「parallel 瓶颈 ∝ l₁+l₄，V-Shape 为 l₁+l₆」的配图 | 采用 | 5（DualPipeV cut-in-half 与 V 形布局，核心论据） |
| `8e41c496…e624e.jpg` | 86 | Figure 3(a) | 真插图 | 1F1B 的 building block。蓝色 `F` 阶梯下行 6 格、绿色 `B` 阶梯上行 6 格，左右端与 F/B 交界处各一条竖直虚线；底部两段横向量分别标 **`2d`**（F 段）与 **`4d`**（B 段），合计 **6d**，与 Table 1 中 1F1B 的 `l_max = 6d` 精确吻合。**无 W 方块** | 采用 | 5、6（气泡/显存公式重算） |
| `5abf8895…5a18.jpg` | 89 | Figure 3(b) | 真插图 | V-Half 的 building block。含蓝 `F`、绿 `B`、青 `W`，F 阶梯之间有明显空隙（对应 Table 9 的 δF⁰=2）；底部四段横向量标 **`2d` / `d` / `2d-1` / `d`**，合计 **6d−1 ≈ 6d**，与 Table 1 中 V-Half 的 `l_max = 6d` 吻合。结构中出现一块灰色空格（气泡/空档） | 采用 | 5、6 |
| `596bbd1a…dec6b.jpg` | 91 | Figure 3(c) | 真插图 | V-Min 的 building block。F 阶梯紧密相扣（δ=1，最紧），底部出现并排的 `F F`；含 `B`/`W` 块，底部四段横向量标 **`d` / `d` / `d` / `d`**，合计 **4d**，与 Table 1 中 V-Min 的 `l_max = 4d` 唯一吻合 | 采用 | 5、6 |
| `bb02e3eb…4d35.jpg` | 95 | Figure 3(d) | 真插图 | V-ZB 的 building block。F 阶梯之间空隙最大、整体最宽，含 `F`/`B`/`W`；底部四段横向量标 **`4d-3` / `2d-1` / `4d-3` / `2d-1`**，合计约 **12d**，对应 Table 1 中 V-ZB 的 `l_max = 12d`（该图标注与正文「精确值 12d−2」略有出入，见「存疑与分歧」第 7 条） | 采用 | 5、6 |
| `40d18d25…c6a66.jpg` | 111 | Figure 4 | 真插图 | 四联整幅调度图，面板自上而下标 **(a) 1F1B / (b) V-Min / (c) V-Half / (d) V-ZB**。每个面板用彩色格阵表示 warm-up→stable→cool-down，格内数字为 microbatch 编号 1–10，蓝＝F 段、青＞蓝＝B/W 混合段、绿＝稳定段，右下角灰色梯形＝bubble。四者可直观对比气泡斜边长度与稳定区形状 | 采用 | 5（核心）、3、6 |
| `09572d6c…2c9b31.jpg` | 114 | Figure 5(a) | 真插图 | 上排＝从 V-Min 完整调度里切出一个 `d×T` 的重复网格（绿框高亮）；下方箭头标 `Tf=3,Tb=4,Tw=2`（把 F/B/W 换成不等长），右侧重排结果中明确写出 **`bubble`** 并留空档——说明 V-Min 每个重复周期都会留一个气泡 | 采用 | 6（气泡随 microbatch 增长） |
| `055d481b…3f55.jpg` | 117 | Figure 5(b) | 真插图 | 与 (a) 同格式的 V-Half 版本：同样取 `d×T` 网格、同样用 `Tf=3,Tb=4,Tw=2` 重排，右侧结果**铺满、没有 `bubble` 标注**，即 V-Half 在真实不等长 F/B/W 下仍能自洽拼接 | 采用 | 6 |
| `6d0222dd…ea69c.jpg` | 143 | Figure 6 | 真插图 | 2 行 × 3 列 = 6 面板。上排 MFU (%) vs Number of Microbatches，标题分别为「9.6B Model on 16 Pipelines」「21B Model on 24 Pipelines」「38.5B Model on 32 Pipelines」，图例 V-ZB / ZB-1P / V-Half / 1F1B / V-Min / 1F1B-R。下排 Activation Memory (GB) 同标题，可读出 1F1B≈45–49 GB、ZB-1P≈30 GB、V-Half≈20 GB、V-Min≈15 GB（38.5B 面板为 35/30/20/15 GB）——即 V-Half≈1/2、V-Min≈1/3 的实测证据 | 采用 | 6（显存公式的实测对照）、9 |
| `a3f33e81…9e443.jpg` | 152 | Figure 7(a) | 真插图 | 散点＋Pareto 前沿。「Model: 9.6B; p=16; m=64; mb=2」。x＝Activation Memory (GB)，y＝MFU (%)。红线（Pareto）串联 1F1B-R(≈32%,4GB) → V-Min(≈40.5%,10GB) → V-Half(≈46.5%,14GB) → V-ZB(≈51%,23GB) → ZB-2P(≈52.5%,48GB)；非前沿点：1F1B(≈43.5%,23GB)、ZB-1P(≈48.5%,25GB)、Interleaved 1F1B(≈48.5%,36GB) | 可选（Pareto 结论已由 Figure 6/10 覆盖，篇幅紧张时可省） | 9（选型） |
| `57595b74…1e76f.jpg` | 154 | Figure 7(b) | 真插图 | 同 (a) 格式。「Model: 21B; p=24; m=96; mb=1」。Pareto 红线：1F1B-R(≈32%,4GB) → V-Min(≈40.5%,8GB) → V-Half(≈45.5%,12GB) → V-ZB(≈50%,21GB) → ZB-2P(≈52%,42GB)；非前沿：1F1B(≈42%,20GB)、ZB-1P(≈47.5%,22GB)、Interleaved 1F1B(≈47%,30GB) | 可选 | 9 |
| `a46d3cbf…bae3.jpg` | 157 | Figure 7(c) | 真插图 | 同 (a) 格式。「Model: 38.5B; p=32; m=128; mb=1」，x 轴下方额外标注 **`* ZB-2P OOM`**（该配置下 ZB-2P 显存溢出，故图上无此点）。点：1F1B-R(≈32%,4GB)、V-Min(≈41%,15GB)、V-Half(≈46.5%,20GB)、V-ZB(≈51.5%,35GB)、1F1B(≈43%,35GB)、ZB-1P(≈47.5%,36GB)、Interleaved 1F1B(≈47.5%,46GB)。**OOM 标注是「失败模式」章节的好素材** | 可选（若第 9 章要写失败模式，建议优先选这一张） | 9、6 |
| `0bb85b23…022e.jpg` | 165 | Figure 8(a) | 真插图 | 「9.6B Model on 16 Pipelines」，MFU (%) vs Number of Microbatches（16–256）。图例带 **各自不同的 microbatch size**：`V-ZB mbs:4`、`ZB-1P mbs:4`、`V-Half mbs:8`、`1F1B mbs:4`、`V-Min mbs:12` | 可选（Figure 8 是「比较口径」的关键，但 6 张全开会占很大篇幅；建议至少选 8(c)+8(f)） | 6（比较口径统一） |
| `54b18b0c…a604.jpg` | 167 | Figure 8(b) | 真插图 | 「21B Model on 24 Pipelines」，MFU vs #Microbatches（24–384）。图例 `V-ZB mbs:2`、`ZB-1P mbs:2`、`V-Half mbs:4`、`1F1B mbs:2`、`V-Min mbs:6` | 可选 | 6 |
| `4d60dbc7…66a7.jpg` | 169 | Figure 8(c) | 真插图 | 「38.5B Model on 32 Pipelines」，MFU vs #Microbatches（32–512）。图例 `V-ZB mbs:1`、`ZB-1P mbs:1`、`V-Half mbs:2`、`1F1B mbs:1`、`V-Min mbs:3`。在 256–512 处**绿线 V-Half 反超 V-ZB 与 1F1B**，是「省显存换取更大 microbatch→更高算术强度」的直接证据 | 可选（推荐入选） | 6、9 |
| `20d18354…ad75.jpg` | 171 | Figure 8(d) | 真插图 | 「9.6B Model on 16 Pipelines」，Activation Memory (GB) vs #Microbatches。y 轴 0–60。因 V-Half/V-Min 把 microbatch 翻倍/三倍，**V-Min mbs:12 反而最高（≈56 GB）**，高于 1F1B mbs:4（≈46 GB）；V-Half mbs:8≈50 GB、ZB-1P≈48 GB、V-ZB≈48 GB、1F1B-R≈51 GB | 可选（**是「口径统一」最有力的反面教材**，推荐入选） | 6 |
| `d17efa8d…14f8e.jpg` | 173 | Figure 8(e) | 真插图 | 「21B Model on 24 Pipelines」，Activation Memory (GB) vs #Microbatches。V-Min mbs:6≈50 GB 最高，V-Half mbs:4≈46 GB，ZB-1P≈45 GB，V-ZB≈44 GB，1F1B mbs:2≈42 GB | 可选 | 6 |
| `3b523d5e…98ff.jpg` | 175 | Figure 8(f) | 真插图 | 「38.5B Model on 32 Pipelines」，Activation Memory (GB) vs #Microbatches。V-Min mbs:3≈41 GB、V-Half mbs:2≈38 GB、ZB-1P≈37 GB、V-ZB≈36 GB、1F1B mbs:1≈35 GB——**「省一半显存」结论在补偿 microbatch 后被大幅抵消**，且随 device 数增大而缓和 | 可选（推荐入选，与 4d60dbc7 成对） | 6 |
| `27b93e2f…3e336.jpg` | 256 | Figure 9(a) | 真插图 | 「9.6B Model, p=16」。y＝Bubble Rate（0–0.35），x＝Memory Limit（0.25–2.00），4 条曲线对应 `# Microbatches=32/64/128/256`。全部曲线在 **Memory Limit≈0.5 处出现垂直陡降**，之后随显存上限增大缓慢趋零 | 采用 | 6（0.5 悬崖＝V-Half 拐点）、9 |
| `8e0695c8…1812.jpg` | 258 | Figure 9(b) | 真插图 | 「21B Model, p=24」，同 (a) 格式，曲线 `# Microbatches=48/96/192/384`，同样在 Memory Limit≈0.5 陡降 | 采用 | 6、9 |
| `aad3d1ea…d19be.jpg` | 260 | Figure 9(c) | 真插图 | 「38.5B Model, p=32」，同 (a) 格式，曲线 `# Microbatches=64/128/256/512`，同样在 Memory Limit≈0.5 陡降。三张一起构成「气泡率随显存上限的通用曲线族」 | 采用 | 6、9 |
| `56322bd9…f2633.jpg` | 265 | Figure 10 | 真插图 | 4 行 × 3 列 = 12 个面板。行＝(9.6B p=16 / 21B p=24 / 38.5B p=32) 各自的 m=32/48/64、m=64/96/128、m=128/192/256、m=256/384/512；每面板 y＝Bubble Rate（0–0.35）、x＝Memory Limit（0–2.0）。图例三条：`V Scheduler`（蓝）、`ZB Scheduler`（橙）、`1F1B`（绿色水平虚线参考线）。可读出 **V Scheduler 在相同 Memory Limit 下气泡率一致低于 ZB Scheduler**，且在 ≈0.5 处下降更早 | 采用 | 6、9（V 调度谱系优于 Zero Bubble 自适应调度的定量证据） |
| `827f6617…e56051.jpg` | 272 | Figure 11 | 真插图 | 上下两幅时间线格阵，行标 `Device 1`–`Device 4`，横轴 `Time →`，格内数字为 microbatch 编号，蓝/青/绿分色。**上＝ squeezing 之后**（warm-up/cool-down 处仍有成片空格），**下＝ reordering 之后**（空格被回填、尾部呈阶梯收拢） | 采用 | 3（调度演进：挤压与重排）、5 |
| `c88a0fa7…3926a.jpg` | 325 | Figure 12(a) | 真插图 | 分组柱状图。「Single-pass MFU under different TP degree」，x 轴注 `SequenceLength=1024; BS=160`，y＝MFU（0.40–0.80），分组 t=1/2/4/8，组内柱 mb=1/2/4/8。读数：t=1 时 ≈0.675–0.70，t=2 ≈0.62–0.675，t=4 ≈0.55–0.625，t=8 ≈0.45–0.57——**TP 度越高单趟 MFU 越低** | 可选（服务于第 8 章框架侧对照） | 8 |
| `d793ec78…f767e6.jpg` | 327 | Figure 12(b) | 真插图 | 同 (a) 格式，x 轴注 `SequenceLength=3072; BS=640`。t=1 mb=1≈0.72；t=2 ≈0.66；t=4 ≈0.60（mb=1/2 几乎重合）；t=8 ≈0.515/0.53/0.55 | 可选 | 8 |
| `1c91ffb2…50636c.jpg` | 356 | Figure 13 | 真插图 | 单条横贯时间线（4 行 device，`Time →`）。蓝＝F、绿＝B 密铺，其中一串绿色格被单独标号 **1,1,1,1,1,5,2,2,2,2,6,6,3,3,3,7,7,4,4,4,8,8**，构成一条跨 device 的 dependency path，用于给出运行时间下界 `6n+6p−3k−1` | 可选 | 6（气泡下界） |
| `a2e16ea4…90a28.jpg` | 365 | Figure 14（第 1 幅） | 真插图 | 左侧竖排文字标 **`V-Min`**，图下方注 **`d mod 3 = 0`**。虚线框内为完整 building block：蓝 F 阶梯（含并排 `F F`）＋青 B/绿 W 块 | 采用 | 5、6（building block 随 d 变化、重复不碰撞） |
| `b0bca8e1…33aa4.jpg` | 367 | Figure 14（第 2 幅） | 真插图 | 同族第 2 幅，图下方注 **`d mod 3 = 1`**。F 阶梯中段多出一个空格，B/W 块错位方式与上一幅不同 | 采用 | 5、6 |
| `b36ca860…2f608.jpg` | 369 | Figure 14（第 3 幅） | 真插图 | 同族第 3 幅，图下方注 **`d mod 3 = 2`**。F 阶梯最完整、无并排 `F F`，W 块呈单列下行 | 采用 | 5、6 |
| `74db387f…d4718.jpg` | 371 | Figure 14（第 4 幅） | 真插图 | 左侧竖排文字标 **`V-Half`**，图下方注 **`d mod 2 = 0`**。F 阶梯间距明显比 V-Min 大三组更大（δF⁰=2），右侧 `B B W` 并排 | 采用 | 5、6 |
| `ff1550bc…2c2494.jpg` | 373 | Figure 14（第 5 幅） | 真插图 | 同族第 5 幅，图下方注 **`d mod 2 = 1`**。与上一幅同为 V-Half，仅偏移奇偶分支不同 | 采用 | 5、6 |
| `e8be317a…2f971.jpg` | 392 | Figure 15(a) | 真插图 | 单条横向时间线（`1 2 3 … 7` 蓝格为被高亮的 microbatch 起始格，其余为不同深浅灰的重复块）。表现 Interleaved 1F1B **官方实现的非均匀重复间隔** | 可选 | 3、8（Interleaved 1F1B 对照） |
| `5e578554…1e0ad.jpg` | 395 | Figure 15(b) | 真插图 | 同 (a) 格式的变体，重复间隔改为均匀，说明**同一 building block 换一种重复间隔可得到等价的显存/气泡表现** | 可选 | 3、8 |
| `2889cdd7…dc2ce.jpg` | 409 | Figure 16 | 真插图 | 三联 building block：(a) `1F1B-V`——只有 F 阶梯与 B 阶梯、**不做 B-W 拆分**；(b) `Zero Buble Schedule with 2/3 1F1B memory`——含并排 `F F`、`B B`、`W W` 的复杂交错块；(c) `A Variation of Interleaved 1F1B with The Same Bubble Rate but Lower Memory`——F 下行／B 上行的双阶梯中间夹 W。印证 §3.3「1F1B-V 达到 1F1B 的 2/3 激活显存且不需要拆分 B/W」 | 采用 | 5（V 形思想的推广）、6 |
| `016422f8…f1ad0c.jpg` | 412 | Figure 17 | 真插图 | 一整幅三联完整调度图，面板标签 **(a) 1F1B-V / (b) Zero Buble Schedule with 2/3 1F1B memory / (c) A Variation of Interleaved 1F1B with The Same Bubble Rate but Lower Memory**（图中 "Buble" 为原文拼写）。每面为彩色格阵，格内数字 1–8 为 microbatch 编号，白色空格＝bubble | 采用 | 5、6 |
| `d10a100e…9dc76.jpg` | 416 | Figure 18 | 真插图 | **整幅画廊，不是单一 (i) 子图**（743×1723，极高）。9 组「上排 building block＋重复方式 / 下排 squeeze+reorder 后的完整调度」，可辨认标签依次为 **(a) 1F1B、(b) Eager 1F1B、(c) ZB-H1、(d) ZB-H2、(e) GPipe、(f) GEMS、(g) Chimera、(h) Interleaved 1F1B**，最末一组 (i) 的标签被图片下边缘裁掉。这是**调度谱系图**，一图容纳 1F1B→Eager→ZB-H1/H2→GPipe/GEMS/Chimera→Interleaved 的全家谱 | 采用 | 3（调度演进总览，核心）、8 |
| `0772eb6e…568c3.jpg` | 无（未引用） | Table 2 | 表格截图 | 「Models used in experiments」表头 Model / Layers / Attention Heads / Hidden Size / GPUs，四行：9.6B 30 40 5120 16；21B 46 48 6144 24；38.5B 62 64 7168 32；98.5B 78 80 10240 40。与 md 第 137 行内联 HTML 表逐字一致 | 可转写为 Markdown 表格，不直接引图 | 6（实验口径） |
| `49171ef4…32285.jpg` | 无（未引用） | Table 1 | 表格截图 | 「Activation Memory (Asymptotic) of V-Shape Building Blocks Compared to 1F1B」。列 Interval T / l_max / m / Peak Memory。1F1B：6 / 6d / M/d / **M**；V-Min：6 / 4d / M/2d / **M/3**；V-Half：6 / 6d / M/2d / **M/2**；V-ZB：6 / 12d / M/2d / **M**。与 md 第 101 行内联 HTML 表逐字一致 | 可转写为 Markdown 表格，不直接引图（**第 6 章「气泡/显存公式重算」应直接转写此表**） | 6（核心） |
| `0c4d7b9f…51c51.jpg` | 无（未引用） | Table 8 | 表格截图 | 「Offsets for V-Min」。两行：`0 ≡ d mod 3` → δF⁰=1, δF¹=1, δB⁰=1, δB¹=1, δ(F_d⁰,F_d¹)=1, δ(F_0¹,B_0¹)=**3**, δ(B_d⁰,B_d¹)=1；`0 < d mod 3` → 全为 1。与 md 第 378 行内联 HTML 表逐字一致 | 可转写为 Markdown 表格，不直接引图 | 5（V-Min 实现细节） |
| `4c0179d3…14264.jpg` | 无（未引用） | Table 9 | 表格截图 | 「Offsets for V-Half」。两行：`0 ≡ d mod 2` → δF⁰=**2**, δF¹=1, δB⁰=**2**, δB¹=1, δ(F_d⁰,F_d¹)=2, δ(F_0¹,B_0¹)=**4**, δ(B_d⁰,B_d¹)=1；`1 ≡ d mod 2` → 2,1,2,1,2,**1**,1。与 md 第 382 行内联 HTML 表逐字一致 | 可转写为 Markdown 表格，不直接引图 | 5 |
| `afb606f5…c8749.jpg` | 无（未引用） | Table 3 | 表格截图 | 「V Schedules Combined with Other Memory Saving Methods」。列 Common Setup / PP Method / Best MFU (%) / Best Parameters (DP,TP,PP,mb)。98.5B SeqLength 1024 BatchSize 160 组：1F1B 49.81(2,4,5,2)、1F1B-R 37.15、ZB-1P 54.52、V-Half 52.60、V-Min 48.02、**V-ZB 57.58**；98.5B SeqLength 3072 BatchSize 640 组：1F1B 57.25、1F1B-R 45.81、ZB-1P 56.55、**V-Half 60.34**、V-Min 55.51、V-ZB 56.89。与 md 第 186 行内联 HTML 表逐字一致 | 可转写为 Markdown 表格，不直接引图（**第 9 章「选型」宜转写此表**：短序列选 V-ZB、长序列选 V-Half） | 9（核心）、6 |
| `c226e135…6e20.jpg` | 无（未引用） | Table 4 | 表格截图 | 「Comparing Pipeline Schedules」，本目录信息量最大的一张表。表头 Setup / Model(9.6B,21B,38.5B) / #GPU(16,24,32) / Microbatch(4,2,1) / #Microbatch(16…512)。四组指标行块：**Samples per second per GPU**（V-ZB 2.59→3.15 等）、**MFU (%)**（如 38.5B/512：V-ZB 53.2、V-Half 51.8、1F1B 47.1、V-Min 43.7、1F1B-R 37.2）、**Total memory (GB)**（V-Min 29–33、1F1B-R 14–24、其余 38–59）、**Activation memory (GB)**（V-Min 14–19、V-Half 19–28、ZB-1P 36–48、V-ZB/1F1B 35–56、1F1B-R 恒为 1）、**Bubble rate (%)**（V-ZB 最低 1.09–8.61，V-Min 20.1–36.5，1F1B 8.76–34.6，1F1B-R 8.44–50.7）。与 md 第 295 行内联 HTML 表一致 | 可转写为 Markdown 表格，不直接引图（**第 6 章「比较口径统一」应以本表为基准数据源，尤其 Activation memory 与 Bubble rate 两行**） | 6（核心） |
| `cc3f6cf2…5b8ed.jpg` | 无（未引用） | Table 5 | 表格截图 | 「Single-pass MFU Gain When Increasing Microbatch Size」。列 Model(9.6B/21B/38.5B) × Microbatch Size，行 F Pass (ms)、B Pass (ms)、W Pass (ms)、FBW Average MFU (%)。读数：9.6B mb=4/8/12 时 F=12.96/26.30/39.45 ms；FBW Average MFU≈65%。与 md 第 303 行内联 HTML 表一致 | 可转写为 Markdown 表格，不直接引图 | 6（算术强度论证） |
| `263ffd64…27bdb.jpg` | 无（未引用） | Table 6 | 表格截图 | 「Raw MFU Data of Grid Search」。列 Parallelisation / MicroBS / 1F1B / 1F1B-R / ZB-1P / V-Half / V-Min / V-ZB，按 DP/TP/PP 组合分行（DP=1 TP=4 PP=10 … DP=2 TP=4 PP=5），大量 `-` 表示 OOM 未跑。与 md 第 313 行内联 HTML 表一致 | 可转写为 Markdown 表格，不直接引图 | 9（可作为附录数据，正文不必引） |
| `63d856fa…9c911.jpg` | 无（未引用） | Table 7 | 表格截图 | 「Raw MFU Data of Grid Search on Increased Sequence Length and Batch Size」。同 Table 6 结构，行标签为 d=1/t=4/p=10 等，如 d=1 t=4 p=5 时 V-Half **60.34**、V-Min 13.27（异常低值）。与 md 第 321 行内联 HTML 表一致 | 可转写为 Markdown 表格，不直接引图 | 9（附录数据） |
| `54b87dad…c994.jpg` | 无（未引用） | §2.2 无编号公式 | 公式截图 | 纯公式裁图：`peak memory ≤ ⌈l/T⌉ m`。md 第 55–57 行已用 `$$…$$` 排版同一公式 | **不采用：公式截图，非插图** | — |
| `45a96dd9…97c69.jpg` | 无（未引用） | 式 (1) | 公式截图 | 纯公式裁图：`peak memory of device i ≤ Σ_{s∈S_i} ⌈l^s/T⌉ m^s`。md 第 61–63 行已用 `$$…\tag{1}$$` 排版 | **不采用：公式截图，非插图** | — |
| `de562485…5e720.jpg` | 无（未引用） | 式 (2) | 公式截图 | 纯公式裁图：自适应调度器的 δ 约束四式（`δF_i^0 = δB_i^1 = δ_{<K}^0, ∀1≤i<K` 等）。md 第 244–246 行已用 `$$…\tag{2}$$` 排版 | **不采用：公式截图，非插图** | — |

## 采用图的正文引用块（可直接粘贴）

对每张「采用」的图给出：

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/controllable-memory-fig1-how-to-build-a-pipeline.jpg" alt="Figure 1 如何搭一条流水线：(a) 1F1B 与 (b) Eager 1F1B，上排 building block 重复铺开、Squeeze 后得下排调度，横向量标注 Lifespan 与 Interval" style="width: 100%;">
</div>

> **图片来源**：Pipeline Parallelism with Controllable Memory（arXiv 2405.15362v1）Figure 1，§2 How to build a Pipeline。本地副本：`references/papers/controllable-memory/controllable-memory.md` 第 36 行引用图。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/controllable-memory-fig2-parallel-vs-vshape.jpg" alt="Figure 2 Parallel 与 V-Shape 放置对比：左 Parallel 下 device1 同时挂 l1 与 l4，右 V-Shape 下 device1 挂 l1 与 l6，长短 lifespan 配对后峰值显存更均衡" style="width: 100%;">
</div>

> **图片来源**：Pipeline Parallelism with Controllable Memory（arXiv 2405.15362v1）Figure 2，§3 Memory Efficient Building Blocks。本地副本：`references/papers/controllable-memory/controllable-memory.md` 第 77 行引用图。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/controllable-memory-fig3a-1f1b-building-block.jpg" alt="Figure 3(a) 1F1B 的 building block：F 阶梯下行、B 阶梯上行，底部标注 2d 与 4d，合计 6d 等于 Table 1 的 l_max，且不拆分 W" style="width: 72%;">
</div>

> **图片来源**：Pipeline Parallelism with Controllable Memory（arXiv 2405.15362v1）Figure 3(a)，§3.1 V-Shape Building Blocks。本地副本：`references/papers/controllable-memory/controllable-memory.md` 第 86 行引用图。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/controllable-memory-fig3b-vhalf-building-block.jpg" alt="Figure 3(b) V-Half 的 building block：含 F/B/W 三类方块，底部标注 2d、d、2d-1、d，合计约 6d" style="width: 72%;">
</div>

> **图片来源**：Pipeline Parallelism with Controllable Memory（arXiv 2405.15362v1）Figure 3(b)，§3.1 V-Shape Building Blocks。本地副本：`references/papers/controllable-memory/controllable-memory.md` 第 89 行引用图。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/controllable-memory-fig3c-vmin-building-block.jpg" alt="Figure 3(c) V-Min 的 building block：F 阶梯最紧、底部出现并排 F F，底部标注 d、d、d、d，合计 4d 是 V-Min 独有的 l_max" style="width: 72%;">
</div>

> **图片来源**：Pipeline Parallelism with Controllable Memory（arXiv 2405.15362v1）Figure 3(c)，§3.1 V-Shape Building Blocks。本地副本：`references/papers/controllable-memory/controllable-memory.md` 第 91 行引用图。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/controllable-memory-fig3d-vzb-building-block.jpg" alt="Figure 3(d) V-ZB 的 building block：F 阶梯间距最大、整体最宽，底部标注 4d-3、2d-1、4d-3、2d-1，合计约 12d" style="width: 72%;">
</div>

> **图片来源**：Pipeline Parallelism with Controllable Memory（arXiv 2405.15362v1）Figure 3(d)，§3.1 V-Shape Building Blocks。本地副本：`references/papers/controllable-memory/controllable-memory.md` 第 95 行引用图。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/controllable-memory-fig4-vshape-full-schedules.jpg" alt="Figure 4 V-Shape 完整调度四联图：(a) 1F1B、(b) V-Min、(c) V-Half、(d) V-ZB，格内数字为 microbatch 编号，右下灰色梯形为 bubble" style="width: 100%;">
</div>

> **图片来源**：Pipeline Parallelism with Controllable Memory（arXiv 2405.15362v1）Figure 4，§3.2 V-Shape Pipeline Schedules。本地副本：`references/papers/controllable-memory/controllable-memory.md` 第 111 行引用图。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/controllable-memory-fig5a-vmin-repeating-bubble.jpg" alt="Figure 5(a) V-Min 的重复气泡：从 d×T 网格重排，取 Tf=3,Tb=4,Tw=2 后右侧出现标注为 bubble 的空档" style="width: 95%;">
</div>

> **图片来源**：Pipeline Parallelism with Controllable Memory（arXiv 2405.15362v1）Figure 5(a)，§3.2 V-Shape Pipeline Schedules。本地副本：`references/papers/controllable-memory/controllable-memory.md` 第 114 行引用图。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/controllable-memory-fig5b-vhalf-repeat-tessellates.jpg" alt="Figure 5(b) V-Half 在同样 Tf=3,Tb=4,Tw=2 下重复网格可以铺满，不产生每周期气泡" style="width: 95%;">
</div>

> **图片来源**：Pipeline Parallelism with Controllable Memory（arXiv 2405.15362v1）Figure 5(b)，§3.2 V-Shape Pipeline Schedules。本地副本：`references/papers/controllable-memory/controllable-memory.md` 第 117 行引用图。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/controllable-memory-fig6-mfu-and-activation-memory.jpg" alt="Figure 6 相同 microbatch size 下的吞吐与激活显存：上排 MFU vs 微批数（9.6B/16、21B/24、38.5B/32），下排 Activation Memory (GB)，可读出 V-Half≈1/2、V-Min≈1/3 于 1F1B" style="width: 100%;">
</div>

> **图片来源**：Pipeline Parallelism with Controllable Memory（arXiv 2405.15362v1）Figure 6，§4.2 Comparing Pipeline Schedules。本地副本：`references/papers/controllable-memory/controllable-memory.md` 第 143 行引用图。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/controllable-memory-fig9a-bubble-rate-vs-memory-limit-9.6b.jpg" alt="Figure 9(a) 9.6B p=16 下 V 调度器的气泡率随显存上限变化，四条曲线对应 32/64/128/256 个微批，均在 Memory Limit≈0.5 处陡降" style="width: 80%;">
</div>

> **图片来源**：Pipeline Parallelism with Controllable Memory（arXiv 2405.15362v1）Figure 9(a)，附录 B Bubble Rate Evaluation of Adaptive Scheduler。本地副本：`references/papers/controllable-memory/controllable-memory.md` 第 256 行引用图。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/controllable-memory-fig9b-bubble-rate-vs-memory-limit-21b.jpg" alt="Figure 9(b) 21B p=24 下气泡率 vs 显存上限，四条曲线对应 48/96/192/384 个微批，同样在 0.5 处陡降" style="width: 80%;">
</div>

> **图片来源**：Pipeline Parallelism with Controllable Memory（arXiv 2405.15362v1）Figure 9(b)，附录 B Bubble Rate Evaluation of Adaptive Scheduler。本地副本：`references/papers/controllable-memory/controllable-memory.md` 第 258 行引用图。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/controllable-memory-fig9c-bubble-rate-vs-memory-limit-38.5b.jpg" alt="Figure 9(c) 38.5B p=32 下气泡率 vs 显存上限，四条曲线对应 64/128/256/512 个微批，同样在 0.5 处陡降" style="width: 80%;">
</div>

> **图片来源**：Pipeline Parallelism with Controllable Memory（arXiv 2405.15362v1）Figure 9(c)，附录 B Bubble Rate Evaluation of Adaptive Scheduler。本地副本：`references/papers/controllable-memory/controllable-memory.md` 第 260 行引用图。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/controllable-memory-fig10-v-scheduler-vs-zero-bubble.jpg" alt="Figure 10 V Scheduler 与 Zero Bubble Scheduler 对比：4×3 共 12 面板，每面板气泡率 vs Memory Limit，V Scheduler 在相同显存上限下气泡率一致更低" style="width: 100%;">
</div>

> **图片来源**：Pipeline Parallelism with Controllable Memory（arXiv 2405.15362v1）Figure 10，附录 B Bubble Rate Evaluation of Adaptive Scheduler。本地副本：`references/papers/controllable-memory/controllable-memory.md` 第 265 行引用图。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/controllable-memory-fig11-squeeze-vs-reorder.jpg" alt="Figure 11 上：Squeezing 之后仍留空格的调度；下：Reordering 之后空档被回填的同一调度，行标 Device 1–4，横轴 Time" style="width: 100%;">
</div>

> **图片来源**：Pipeline Parallelism with Controllable Memory（arXiv 2405.15362v1）Figure 11，附录 C Reordering。本地副本：`references/papers/controllable-memory/controllable-memory.md` 第 272 行引用图。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/controllable-memory-fig14a-vmin-building-block-dmod3eq0.jpg" alt="Figure 14 第 1 幅：V-Min 在 d mod 3 = 0 时的 building block，蓝 F 阶梯含并排 F F，右侧为 B 与 W 块" style="width: 70%;">
</div>

> **图片来源**：Pipeline Parallelism with Controllable Memory（arXiv 2405.15362v1）Figure 14，附录 F Building Blocks of V-Min and V-Half for all values of d。本地副本：`references/papers/controllable-memory/controllable-memory.md` 第 365 行引用图。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/controllable-memory-fig14b-vmin-building-block-dmod3eq1.jpg" alt="Figure 14 第 2 幅：V-Min 在 d mod 3 = 1 时的 building block，F 阶梯中段留出空格以避开碰撞" style="width: 70%;">
</div>

> **图片来源**：Pipeline Parallelism with Controllable Memory（arXiv 2405.15362v1）Figure 14，附录 F。本地副本：`references/papers/controllable-memory/controllable-memory.md` 第 367 行引用图。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/controllable-memory-fig14c-vmin-building-block-dmod3eq2.jpg" alt="Figure 14 第 3 幅：V-Min 在 d mod 3 = 2 时的 building block，F 阶梯完整无并排，W 呈单列" style="width: 70%;">
</div>

> **图片来源**：Pipeline Parallelism with Controllable Memory（arXiv 2405.15362v1）Figure 14，附录 F。本地副本：`references/papers/controllable-memory/controllable-memory.md` 第 369 行引用图。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/controllable-memory-fig14d-vhalf-building-block-dmod2eq0.jpg" alt="Figure 14 第 4 幅：V-Half 在 d mod 2 = 0 时的 building block，F 阶梯间距为 2 明显大于 V-Min" style="width: 70%;">
</div>

> **图片来源**：Pipeline Parallelism with Controllable Memory（arXiv 2405.15362v1）Figure 14，附录 F。本地副本：`references/papers/controllable-memory/controllable-memory.md` 第 371 行引用图。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/controllable-memory-fig14e-vhalf-building-block-dmod2eq1.jpg" alt="Figure 14 第 5 幅：V-Half 在 d mod 2 = 1 时的 building block，仅偏移奇偶分支与上一幅不同" style="width: 70%;">
</div>

> **图片来源**：Pipeline Parallelism with Controllable Memory（arXiv 2405.15362v1）Figure 14，附录 F。本地副本：`references/papers/controllable-memory/controllable-memory.md` 第 373 行引用图。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/controllable-memory-fig16-other-memory-efficient-building-blocks.jpg" alt="Figure 16 其他省显存 building block：(a) 1F1B-V 不做 B-W 拆分即可把峰值显存降到 1F1B 的 2/3；(b) Zero Buble Schedule with 2/3 1F1B memory；(c) 与 Interleaved 1F1B 同气泡率但更低显存的变体" style="width: 80%;">
</div>

> **图片来源**：Pipeline Parallelism with Controllable Memory（arXiv 2405.15362v1）Figure 16，附录 I Other Memory Efficient Building Blocks。本地副本：`references/papers/controllable-memory/controllable-memory.md` 第 409 行引用图。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/controllable-memory-fig17-other-memory-efficient-schedules.jpg" alt="Figure 17 三联完整调度：(a) 1F1B-V、(b) Zero Buble Schedule with 2/3 1F1B memory、(c) 与 Interleaved 1F1B 同气泡率但更低显存的变体，白格为 bubble" style="width: 90%;">
</div>

> **图片来源**：Pipeline Parallelism with Controllable Memory（arXiv 2405.15362v1）Figure 17，附录 I。本地副本：`references/papers/controllable-memory/controllable-memory.md` 第 412 行引用图。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/controllable-memory-fig18-schedule-gallery.jpg" alt="Figure 18 调度画廊：自上而下 (a) 1F1B、(b) Eager 1F1B、(c) ZB-H1、(d) ZB-H2、(e) GPipe、(f) GEMS、(g) Chimera、(h) Interleaved 1F1B，每组的上一行为 building block 及其重复方式、下一行为 squeeze 与 reorder 后的最终调度" style="width: 70%;">
</div>

> **图片来源**：Pipeline Parallelism with Controllable Memory（arXiv 2405.15362v1）Figure 18，附录 G A Gallery of Pipeline Parallel Schedules and Their Building Blocks。本地副本：`references/papers/controllable-memory/controllable-memory.md` 第 416 行引用图。

## 存疑与分歧

1. **草稿的两条既有判断均被读图证实。** `7f4661e0…` 确为 **Figure 2「Parallel vs V-Shape」**（§3），图内左栏 device 1 挂 l₁+l₄、右栏挂 l₁+l₆，与正文「瓶颈 ∝ l₁+l₄／l₁+l₆」一致；`40d18d25…` 确为 **Figure 4「V-Shape Full Schedules」**（§3.2），是含 (a) 1F1B、(b) V-Min、(c) V-Half、(d) V-ZB 四面板的整幅调度对比图。两者的图号与内容都与草稿描述相符，无需更正。
2. **md 第 89–96 行的 Figure 3 子图标注错位（MinerU 版面问题）。** md 的文本顺序是「图@86 → (a) 1F1B@87 → 图@89 → 图@91 → (c) V-Min@92 → (b) V-Half@94 → 图@95 → (d) V-ZB@96」，(b) 与 (c) 的题注顺序颠倒。读图后按**底部横向量标注之和与 Table 1 的 l_max 逐一对齐**判定归属，结论与 md 一致、只是位置错乱：`8e41c496` = (a) 1F1B（2d+4d=6d，无 W）、`5abf8895` = (b) V-Half（2d+d+2d−1+d≈6d，δF⁰=2 有大空隙）、`596bbd1a` = (c) V-Min（d+d+d+d=4d，F 阶梯最紧）、`bb02e3eb` = (d) V-ZB（4d−3+2d−1+4d−3+2d−1≈12d，最宽）。
3. **`596bbd1a…` 与 `bb02e3eb…` 图片上边缘各残留一行被裁切的题注，字形近似「(b) V-Half」。** 分辨率不足（原图仅 331×173 / 740×179，裁切带高 34 px）无法可靠 OCR。若该残留确实读作「(b) V-Half」，则原 PDF 的 Figure 3 是 2×2 网格（上行 (a) 1F1B / (b) V-Half，下行 (c) V-Min / (d) V-ZB），与第 2 条的读图结论自洽。**该残留未被用作判据**，仅作记录。
4. **`d10a100e…` 是 Figure 18 整幅画廊，而非单一 (i) 子图。** md 第 417 行的题注「(i) Interleaved 1F1B with Uniform Interval」只对应这张高图（742×1723）最末一组；图中可辨认标签为 (a) 1F1B 到 (h) Interleaved 1F1B，**(i) 自己的标签被图片下边缘裁掉**。若文章只想要 (i) 面板，需要自行裁切。
5. **md 第 156 行的「Figure 7」题注位置夹在两张子图之间。** 实际 Figure 7 = `a3f33e81`(9.6B/16) + `57595b74`(21B/24) + `a46d3cbf`(38.5B/32) 三张，题注被 MinerU 排在 (b) 与 (c) 之间。
6. **Figure 8 六张子图的 (a)–(f) 编号为推断。** md 的引用顺序是「0bb85b23 → 54b18b0c → 4d60dbc7 → 20d18354 → d17efa8d → 3b523d5e」，即先 3 张 MFU 面板、再 3 张 Activation Memory 面板，属**先行后列**读取；而从图像尺寸配对看（`0bb85b23`348×251 与 `20d18354`345×245、`54b18b0c`343×251 与 `d17efa8d`345×254、`4d60dbc7`348×251 与 `3b523d5e`348×254）原 PDF 很可能是**先列后行**的 2 行 3 列。两种编号下的**内容配对关系一致**（同模型同 GPU 数的 MFU 面板与显存面板成对），仅 (a)–(f) 的字面序号存疑。
7. **V-ZB building block 的图内标注与正文数值有轻微出入。** `bb02e3eb` 底部四段读作 `4d-3 / 2d-1 / 4d-3 / 2d-1`，合计 12d−8；而 §3.1 正文写「accurately for V-ZB l_max = 12d − 2 but we use l_max = 12d」，Table 1 取 12d。引用图中数值时不要与 Table 1 的 12d 直接画等号。
8. **12 张未引用图片全部是公式/表格裁图，不是丢失的插图。** 3 张公式（`54b87dad` = `peak memory ≤ ⌈l/T⌉m`，`45a96dd9` = 式(1)，`de562485` = 式(2)）与 9 张表格（Table 1–9 各一张）在 md 中都已分别用 `$$…$$` 与内联 `<table>` 重新排版，因此它们成为孤儿是 MinerU 的**有意行为**。核对方式：9 张表格截图的表头与数据同 md 内联 HTML 表逐字一致（Table 1 见第 101 行、Table 2 见第 137 行、Table 3 见第 186 行、Table 4 见第 295 行、Table 5 见第 303 行、Table 6 见第 313 行、Table 7 见第 321 行、Table 8 见第 378 行、Table 9 见第 382 行）。
9. **本目录没有任何页眉/logo/装饰/图内碎片。** 最小文件为 3 483 B（`54b87dad`，公式），无纯色块、无横幅、无 logo 类资源，因此第 4 类计数为 0。
10. **「采用」口径说明。** 本清单把 14 张真插图列为「可选」而非「采用」，判据是：Figure 7（3 张 Pareto）、Figure 8（6 张配套显存/吞吐）、Figure 12（2 张 TP 单趟 MFU）、Figure 13（1 张 dependency path 下界）、Figure 15（2 张 Interleaved 1F1B 重复间隔）都属于「结论已被更核心的图覆盖，或只服务单一附论」的类别。其中**建议优先补入的是 `4d60dbc7`+`3b523d5e`（Figure 8(c)/(f)）与 `20d18354`（Figure 8(d)）**——它们是第 6 章「比较口径统一」唯一的一手证据（省显存换来更大的 microbatch 后，V-Min 的激活显存反而超过 1F1B），以及 `a46d3cbf`（Figure 7(c)，含 `* ZB-2P OOM` 标注，适合第 9 章失败模式）。

## 全量文件名索引

逐张清单的排序与此表一致。核对方式：在
`torch/dualpipe/references/papers/controllable-memory/` 下执行 `ls images/ | wc -l` 应得 **49**；
本表行数 **49**；下表的 49 个文件名去重后应仍为 **49**。类别列用于交叉核对汇总里的 37/3/9/0。

| # | 完整文件名 | 图号/表号 | 类别 |
| --- | --- | --- | --- |
| 1 | `84cacb5f3a17558cf0849433896c024cf139c69cb46d11792953556462804799.jpg` | Figure 1 | 真插图 |
| 2 | `7f4661e035fa5b72abe32a5777663cd8701da5687051adfe5cd13ef11a9f4c40.jpg` | Figure 2 | 真插图 |
| 3 | `8e41c496029d0371ef6b3191b281a3e8faf990ce83470efde8a02cf1651e624e.jpg` | Figure 3(a) | 真插图 |
| 4 | `5abf88956627057b8da0cdced231ac6088c6ffb1162b229e13e52203ce755a18.jpg` | Figure 3(b) | 真插图 |
| 5 | `596bbd1a0a185ad722f5cdafb7ae677558c29b6c56adea68db4721c295bdec6b.jpg` | Figure 3(c) | 真插图 |
| 6 | `bb02e3eb74733e775ddaebb0b98d736e5d59f4344e4b36a07af0e026fc324d35.jpg` | Figure 3(d) | 真插图 |
| 7 | `40d18d252f7d6470f7140a7622b62634ba51963b085de5c7020b87f2a99c6a66.jpg` | Figure 4 | 真插图 |
| 8 | `09572d6c9c44334e95603f1c2c4257d90cbe8b6ec2201273305c4b70402c9b31.jpg` | Figure 5(a) | 真插图 |
| 9 | `055d481b7a75d2bc928c778ce598b7475617ca1affc8cd2c87fa9c7de5bf3f55.jpg` | Figure 5(b) | 真插图 |
| 10 | `6d0222dda41626e3feed3fce587f946460d388cb790c926574f994a04d2ea69c.jpg` | Figure 6 | 真插图 |
| 11 | `a3f33e814a7d1fd4373ab1f7a100e9c3e5af771d72481e27572a67768069e443.jpg` | Figure 7(a) | 真插图 |
| 12 | `57595b749570b586ce22d2c252f850fce502bbf7961dc66d93e0065c6aa1e76f.jpg` | Figure 7(b) | 真插图 |
| 13 | `a46d3cbfc161d475377d86c34bbcc781f4d19fb51edd588a2ab147802b6dbae3.jpg` | Figure 7(c) | 真插图 |
| 14 | `0bb85b23c6d47381604c23b406249779362eb73c427352169d85e4c49cb3022e.jpg` | Figure 8(a) | 真插图 |
| 15 | `54b18b0cea306550e4854f9fd5ee262a3c5c2c6dfd47ead6e6d289ffbab5a604.jpg` | Figure 8(b) | 真插图 |
| 16 | `4d60dbc73a511859ffa7e9060d396e64b7ae74f228ca1f353bb2017d6fa966a7.jpg` | Figure 8(c) | 真插图 |
| 17 | `20d18354038e9a7f30adb19771dee2ca1d33e12baca45ede20b89034990cad75.jpg` | Figure 8(d) | 真插图 |
| 18 | `d17efa8d66e67f3955b1b747beea236eb7ab531e4a8fe1fa552f46fa6ed14f8e.jpg` | Figure 8(e) | 真插图 |
| 19 | `3b523d5e5c298a88c9f94a7c698c416da1f93dfea96168cb4fb59ae1bdd398ff.jpg` | Figure 8(f) | 真插图 |
| 20 | `27b93e2f11ef19b3f1f59ed23265899e77dd150e9528f040eeea38d4e573e336.jpg` | Figure 9(a) | 真插图 |
| 21 | `8e0695c81bfcc88123972f02c0244006bb9b069d4088577db36b555b68531812.jpg` | Figure 9(b) | 真插图 |
| 22 | `aad3d1eaa8ae84eb0b7c3e332a9a47916ce46cd381c05b67b09f0ec8d5ad19be.jpg` | Figure 9(c) | 真插图 |
| 23 | `56322bd9dcbaffecaaabb66185f915a51f2ea2e96b87bff93212b4bf6f1f2633.jpg` | Figure 10 | 真插图 |
| 24 | `827f6617a248fdb558f10cb5a762564434cbf22ff424de872426d59158e56051.jpg` | Figure 11 | 真插图 |
| 25 | `c88a0fa72a29f911cb6be9d6d9246ba52061abf7e3948d5f94355ddb1103926a.jpg` | Figure 12(a) | 真插图 |
| 26 | `d793ec78a6154da8c545db398dea35c1240fb14e5a4104ec9c1e298048f767e6.jpg` | Figure 12(b) | 真插图 |
| 27 | `1c91ffb277f7a646ea110976a7228b9ef90037366cfec06602496df81550636c.jpg` | Figure 13 | 真插图 |
| 28 | `a2e16ea4f42046ede9d61ab31467d340cc81fc9402cc171fc89507c13aa90a28.jpg` | Figure 14（第 1 幅，V-Min d mod 3 = 0） | 真插图 |
| 29 | `b0bca8e14c488fb3c6f47ca82fe01a6703475e28d77d30405df8033633e63aa4.jpg` | Figure 14（第 2 幅，V-Min d mod 3 = 1） | 真插图 |
| 30 | `b36ca860e963b37f0e71cc02d0f8b6e1ff3c7ed122b23e96bd55470d1a82f608.jpg` | Figure 14（第 3 幅，V-Min d mod 3 = 2） | 真插图 |
| 31 | `74db387fb1392d59ae9052f23a0a4f6a3fe262890e7098957f2c4a04832d4718.jpg` | Figure 14（第 4 幅，V-Half d mod 2 = 0） | 真插图 |
| 32 | `ff1550bcea1bbe833619b9ec1150ef26964013a47aeacae1ea1e8530082c2494.jpg` | Figure 14（第 5 幅，V-Half d mod 2 = 1） | 真插图 |
| 33 | `e8be317a41f7a1a4b596858663150770da26c2900b71b5d6fe407cc04502f971.jpg` | Figure 15(a) | 真插图 |
| 34 | `5e5785544226a24896a19d3fdf619456c473f8a567211b2af2242fd29ac1e0ad.jpg` | Figure 15(b) | 真插图 |
| 35 | `2889cdd71f4c7d851453c994e1fbef4812c162d72b83a9fc2dbeb90bafcdc2ce.jpg` | Figure 16 | 真插图 |
| 36 | `016422f874a802b5701876cc8b7458b47db4b683c1d4e8ddc2c9897dffc1ad0c.jpg` | Figure 17 | 真插图 |
| 37 | `d10a100e2437141bdcbeb485e519199f4f3855984241e329b4370d9b6f49dc76.jpg` | Figure 18 | 真插图 |
| 38 | `0772eb6ec144ef19c7289b4fdee9b34f858e97fe69395f5a74e9ce3d074568c3.jpg` | Table 2 | 表格截图 |
| 39 | `49171ef4c355b0d12d9081f7f71a1142750ed4da55f05ecc3523e381e8232285.jpg` | Table 1 | 表格截图 |
| 40 | `0c4d7b9f90c7ede8e527aa7bfe11b059dbb9aa2d231ddcaa22f1eed18dc51c51.jpg` | Table 8 | 表格截图 |
| 41 | `4c0179d3240752300298e273a87292eff8eeddc04717f9b8b02c17f784214264.jpg` | Table 9 | 表格截图 |
| 42 | `afb606f52daf0c448b2af5d4659c7f71275e4dc7d2508c5e5a097e4e05fc8749.jpg` | Table 3 | 表格截图 |
| 43 | `c226e135addd5efed9d163807fbe429790123a488af3beb96e5e81c200b16e20.jpg` | Table 4 | 表格截图 |
| 44 | `cc3f6cf24e9c32a9409de809d78935d91aebc8afd5abe77ca2c251efcf05b8ed.jpg` | Table 5 | 表格截图 |
| 45 | `263ffd64ae242286f2520f781623808a944e15d2fc556353a9aed1f249927bdb.jpg` | Table 6 | 表格截图 |
| 46 | `63d856fad2c59d541f9db981b0b3334e76d431c868428c762d77f8356079c911.jpg` | Table 7 | 表格截图 |
| 47 | `54b87dad82dd3dfa2744f6ee20303f135c2333f885f56e5b6d250b8f3eacc994.jpg` | §2.2 无编号公式 `peak memory ≤ ⌈l/T⌉m` | 公式截图 |
| 48 | `45a96dd96bf76058a2f29109c4d213eae6017c5a83d9da7025d7a51c69e97c69.jpg` | 式 (1) | 公式截图 |
| 49 | `de5624854c47c5cafc85db5f2d8fa9e80f2fa52198635194414b97ffd665e720.jpg` | 式 (2) | 公式截图 |
