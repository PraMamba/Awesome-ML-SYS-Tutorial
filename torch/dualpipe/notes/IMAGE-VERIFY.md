# DualPipe 文章逐图核对记录

对象：`torch/dualpipe/deep-dive.md` 正文引用的图片（`pics/` 下同名副本）。**当前为 30 张**：第 3 轮删除了两张——`aiinfra-pp-zbv-schedule.png`（内容实为 ZB-H2 而非 ZB-V）与 `aiinfra-pp-interleaved-1f1b-vpp-layout.png`（与 Megatron-LM Figure 4 下栏逐格重复）。

记录分两层：

1. **全量核对**（433 张写作素材图）：`references/` 下每个目录的每一张抽取图都逐张用 `read_image` 实看，记录在 `IMAGE-SURVEY-<slug>.md`（12 个文件，每份表格的行数经脚本校验等于 `ls | wc -l`，逐行给出「文件名 / md 引用行 / 图号 / 类别 / 读图后的事实描述 / 采用结论与理由 / 建议章节」）。**未采用的图也逐张写明了理由。**
2. **本文采用图的独立复核**（当前 30 张）：本文件。复核项包括文件存在性、sha256、体积、以及**文章 alt 文本与图注所述内容是否与画面一致**。

---

## 一、文件存在性与完整性

```bash
cd torch/dualpipe
grep -oE 'src="\./pics/[^"]+"' deep-dive.md | sed 's/src="\.\/pics\///;s/"//' | sort > /tmp/used.txt
ls pics/*.png pics/*.jpg | sed 's|pics/||' | sort > /tmp/have.txt
comm -23 /tmp/used.txt /tmp/have.txt   # 被引用但不存在的文件
comm -13 /tmp/used.txt /tmp/have.txt   # 存在但未被引用的文件
```

结果：被引用 30 张，`pics/` 下非 README 文件 30 张，两个 `comm` 均无输出（第 3 轮删图后重跑）。**不存在正文引用了却找不到的图，也不存在放了但没用上的图。**

第一轮检查曾出现一处不匹配：附录 A 引用的 `terapipe-fig1d-token-pipeline.jpg` 在被补入 `pics/` 之前是「被引用但不存在」，第 2 轮已改从 `references/papers/terapipe/images/a73bba64…jpg`（真正的 Figure 1(d)）复制。

每张图的来源路径与 sha256 记录在 `pics/README.md` 第 1 节。

---

## 二、原文读图复核（本轮独立重看）

以下 18 张由本轮独立用 `read_image` 重看（其中 4 张在第 2 轮因独立审查提出异议而重新核对），核对文章的 alt 与图注是否与画面相符。

| 文件 | 文章所述 | 读图结果 | 结论 |
| --- | --- | --- | --- |
| `gpipe-fig2-naive-vs-microbatch-pipeline.jpg` | (a) 朴素模型并行链、(b) 单 microbatch 流水线、(c) micro-batch 流水线并在网格中部标注 `Bubble` | 画面确实是三联图：(a) 为 Device 0-3 各一个 \(F_i\) 紧跟 \(B_i\) 的链式图，箭头标 `Loss` / `Gradients`；(b) 为 \(F_0\)/\(B_0\) 阶梯加右侧 `Update` 列；(c) Device 0-3 上 \(F_{i,j}\)/\(B_{i,j}\) 密排，中偏左有一个显式标注 `Bubble` 的空框，右侧 `Update` 列 | **相符**；\((p-1)(F+B)\) 只对应 (c) 的判断成立 |
| `dualpipe.png` | 8 个 Device、双向合计 20 个 micro-batch（每方向 10 个）、五类图例、白格为气泡 | Device 0-7；图例为 Forward（橙）/ Backward（宽绿）/ Backward for input（窄绿，与上一项 RGB 相同）/ Backward for weights（蓝）/ Overlapped forward & Backward（橙绿拼接）；格子内数字为 micro-batch 编号（两个方向各 0-9）；存在白色空格 | **相符**。原 alt 里「被同一黑框圈住的两格表示重叠」是从 README 文字转述的，画面本身以橙绿拼接格表达重叠，alt 已改写为只描述图例与格子编号，README 的原话保留在正文 |
| `dualpipev.png` | 4 个 Device（8 个逻辑 stage）、10 个 micro-batch | Device 0-3；格子内 micro-batch 编号 0-9；图例与 `dualpipe.png` 相同；各 Device 的编号序列互不相同 | **相符** |
| `deepseek-v3-fig5-dualpipe-schedule-8pp-20mb.jpg` | 8 个 PP rank、双向合计 20 个 micro-batch（每方向 10 个），五类图例，**图块内不含 Table 2** | Device 0-7、编号 0-9、五类图例，画面除调度网格与图例外没有表格文字 | **相符**，并且**再次确认「MinerU 把 Table 2 并到同一图块」的说法不成立** |
| `zero-bubble-fig3-handcrafted-zb-h1-h2.jpg` | 上栏 ZB-H1 的 warm-up F 数为 4/3/2/1；下栏 ZB-H2 为 7/5/3/1 | 用像素颜色分类复核（格宽 24.5 px、首条竖线 x=72，取每格上边线下 2-4 px 的背景像素以避开数字字形）：上栏四行前导 F 数为 **4/3/2/1**，下栏为 **7/5/3/1**。结构上也只有 7/5/3/1 自洽：H2 每行的前导 F 数正好等于 H1 同一行的「F 数 + 白格数」（4+3=7、3+2=5、2+1=3、1+0=1），这正是「用多铺的前向把 H1 的白格逐个填掉」 | **第 2 轮修正**：最初按 `IMAGE-SURVEY-zero-bubble.md` 写成 7/6/5/1，独立审查提出 7/5/3/1，本轮像素复核确认 **7/5/3/1 正确**，文章已改（核对文件 `IMAGE-SURVEY-zero-bubble.md` 的同一处仍是旧值，引用时以本条为准） |
| `zero-bubble-fig8-zbv-schedule.jpg` | 4 个 Device 各一行时间网格，格内数字用白色或黑色区分两个 chunk | 论文 Figure 8 的图注原文写明「Each device is assigned to exactly 2 chunks, where white text colors represent the first chunk and black text colors represent the second chunk」；放大后可在同一行内看到白字与黑字两种数字，而每格是单色单格 | **第 2 轮修正**：最初写成「每个 Device 下有上下两行小格」，与图注和画面都不符；alt 已按图注口径改写为「单行 + 文字白/黑区分两个 chunk」 |
| `controllable-memory-fig2-parallel-vs-vshape.jpg` | 左 Parallel 下 device 1 挂 \(l_1\) 与 \(l_4\)；右 V-Shape 下 device 1 挂 \(l_1\) 与 \(l_6\) | 左图三行分别标 \(l_1\)/\(l_2\)/\(l_3\)，虚线箭头指向同一行右侧的 \(l_4\)/\(l_5\)/\(l_6\)；右图三行仍为 \(l_1\)/\(l_2\)/\(l_3\)，但 \(l_6\) 出现在 \(l_1\) 所在行的右侧、\(l_4\) 出现在 \(l_3\) 所在行 | **相符** |
| `megatron-fig4-default-vs-interleaved-1f1b.jpg` | default 1F1B 与 interleaved 1F1B 的双面板对照，interleaved 把多个 model chunk 交错排布、气泡显著变小 | 上图 Device 1-4 为标准排布，灰格（空闲）占比很大；两图之间有一个 `Assign multiple stages to each device` 的粗箭头；下图同样的 Device 1-4 上密排深蓝 `Forward Pass` 与浅绿 `Backward Pass`，灰格大幅减少 | **相符** |
| `pipedream-2bw-fig2-2bw-timeline-two-weight-versions.jpg` | 4 个 worker、棋盘格标出新版本权重的落点、标注 `Before/After W_i^(0) → W_i^(4)` 与 `t=21` | Worker 1-4；右上角文字为 `Before: W_1^(0), W_1^(0)` / `After: W_1^(0), W_1^(4)`，右下角为 `Before: W_4^(0), W_4^(0)` / `After: W_4^(0), W_4^(4)` 与 `t = 21`；棋盘格绿格标出新版本落点 | **相符** |
| `controllable-memory-fig18-schedule-gallery.jpg` | 一整幅画廊图（不是单一子图），面板为 (a) 1F1B、(b) Eager 1F1B、(c) ZB-H1、(d) ZB-H2、(e) GPipe、(f) GEMS、(g) Chimera、(h) Interleaved 1F1B | 画面共九组，标签为 (a) 1F1B 到 (h) Interleaved 1F1B，再加最下一组的 (i) Interleaved 1F1B with Uniform Interval；每个面板上排是 building block 的重复、下排是挤压重排后的调度 | **相符** |
| `controllable-memory-fig4-vshape-full-schedules.jpg` | 四联完整调度，面板为 (a) 1F1B、(b) V-Min、(c) V-Half、(d) V-ZB | 四个面板的标签与文章列举完全一致；每面板为多行时间网格，含 I / F / B 三类小格 | **相符** |
| `chimera-fig2-pipeline-schemes-comparison.jpg` | 六种调度的同轴对照，出现 replica0 与 replica1，以及 Bubble 与内存口径的图例框 | 面板自上而下为 `PipeDream`、`PipeDream-2BW`、`GPipe`、`GEMS`、`DAPPLE`、`Chimera`；右侧另有 `M_θ` 与 `M_a` 的柱状对照；图例框注明 Bubble、`model replica0` 的 x/y、`model replica1` 的 x/y，并标注「a backward pass is about 2 times workload of a forward pass」 | **相符** |
| `aiinfra-pp-1f1b-warmup-microbatches-staircase.png` | 各 rank 的前向方块逐行右移形成阶梯，最后一个 PP rank 上 `num_warmup_microbatches = 0` | 纵轴为 NPU0 …NPUP-1，各行方块逐行右移；图上方标「每个NPU num_warmup_microbatches不同」，最后一行右侧标「NPUP-1 上 num_warmup_microbatches=0，执行1个F后进入1F1B状态」 | **原 alt 称「蓝格 Forward、绿格 Backward」，但画面只有蓝、黄两色且没有图例**；已改写为只描述阶梯结构与图内文字，不再断言颜色语义 |
| `zero-bubble-fig1-mlp-computation-graph-f-b-w.jpg` | MLP 计算图，左栏 Forward，右栏 Backward 内部再分 B 与 W，橙色框标出矩阵乘 | 左栏 `Wx`（橙）→ `σ(z)`；右栏左半为 `Wᵀ∇_zL`（橙）与 `(dσ/dz)∇_yL`，右半为 `∇_zLxᵀ`（橙）；底部标注 `F` / `B` / `W` | **相符** |
| `deepseek-moe-fig2-fine-grained-segmentation-shared-expert.jpg` | 三栏：(a) 传统 top-2（N 个专家、K=2）；(b) 细粒度切分后 2N 个专家、K=4；(c) 共享专家隔离，1 个 shared expert 参与、K=3；图例区分 Routed 与 Shared | 三栏标签为 `(a) Conventional Top-2 Routing`、`(b) + Fine-grained Expert Segmentation`、`(c) + Shared Expert Isolation (DeepSeekMoE)`；K 分别为 2 / 4 / 3；图例框为 Routed Expert（浅蓝）与 Shared Expert（浅绿），(c) 中第 1 个专家为绿色 | **相符** |
| `xiaodonggua-ep-serial-vs-overlap.jpg` | 上排串行、下左重叠（计算行与通信行对齐并标 `t_overlap`）、下右 Dense 1F1B | 上排 `EP 1F1B 计算+通信串行`；下左 `EP 1F1B 通信计算重叠`，两行分别为「计算」与「通信」，下方有 `t_overlap` 双向箭头；下右 `Dense 1F1B` | **相符**；图右下角带「知乎 @小冬瓜AIGC」水印，属原页面自带 |
| `terapipe-fig1d-token-pipeline.jpg` | TeraPipe Figure 1(d)：token 级流水线，格子对应很细的 token 分段 | Device 1 到 Device 5 自下而上，每条设备标出负责的 Transformer layer，layer 条带下方是切成很细的 token 分段，橙色细箭头表示 token 分段之间的依赖跨设备传递 | **第 2 轮修正**：最初误取成同图的 Figure 1(c)（`b78a683e…`，GPipe 的 microbatch 流水线，整块条带、无细分段）；已换成真正的 1(d)（`a73bba64…`），alt 同步改写 |
| `sea-ai-lab-cut-in-half-vshape.jpg` | 上栏 DualPipe：8 个 device 各持两个 model layer（0 与 7、1 与 6…），F 块排成 V 形；下栏 Cut-in-half：只保留 Device 0-3，同样的 layer 配对，V 形变窄 | 上栏左侧列出 Device 0-7 的 `Model Layers` 为 0/7、1/6、2/5、3/4、4/3、5/2、6/1、7/0，中间虚线分隔两半，右侧 F 块确呈 V 形；下栏 Device 0-3、层配对相同（0/7、1/6、2/5、3/4），V 形更窄 | **相符**；另注：本地副本右下角带「知乎 @庞天宇」水印，说明该图是从知乎转载页取的图片文件，`references/` 记录的文章来源（Sea AI Lab 博客）本身不变 |

---

## 三、图注陷阱清单（文章已逐条处理）

0. **与核对文件的一处措辞差异（第 3 轮已撤回原判定）**：本文件原先判定 `notes/IMAGE-SURVEY-gpipe.md` 的「(b) 才是朴素模型并行」是措辞差异，并让文章改写成「(a) 是朴素模型并行链、(b) 是只有一个 microbatch 的流水线」。**第 3 轮复核论文图注后确认原核对文件是对的**：GPipe Figure 2 的图注写 (a) 是「an example neural network with sequential layers is partitioned across four accelerators」（即**网络结构图**，不是时间轴）、(b) 是「the naive model parallelism strategy leads to severe under-utilization」（**朴素模型并行**）、(c) 是「pipeline parallelism divides the input mini-batch into smaller micro-batches」。`IMAGE-SURVEY-gpipe.md` 与 `learn-plan.md` 原本的写法正确，文章已在第 3 轮改回。**教训：读图要分两步——先读原文图注与引用该图的正文段落，再看画面；只看画面容易把结构图当成时间轴。**
1. **GPipe Figure 2 是三联合图**，(b) 才是气泡几乎占满的单 microbatch 流水线、(c) 才是 micro-batch 流水线并显式标了 `Bubble`；文章引用时点名了子图 (c)。
2. **Megatron-LM Figure 4 是 `2a3ac338…`**，不是常见的 `648f402f…`（后者是 Figure 2：Transformer layer + Tensor MP + Pipeline MP 两级切分）。
3. **PipeDream-2BW Figure 2 是 `ec0c258b…`**，不是 `e1dfea35…`（后者是 Figure 3：GPipe 与 PipeDream-Flush 的对照）。
4. **Zero Bubble Figure 3 是 `e2c1ba8e…`**（`7828028c…` 是 Figure 2 的 1F1B），**Figure 8 是 `53647889…`**（`3955441b…` 是 Figure 7 的气泡率曲线）。
5. **DeepSeek-V3 Figure 5 的图块内不含 Table 2**，需要 Table 2 时直接转写为 Markdown 表格。
6. **公式截图与表格截图一律不进正文**。`papers/` 下共 41 张纯公式截图与 15 张表格截图全部判为不采用，理由逐张写在核对文件里；`controllable-memory` 的 9 张表格截图内容与其 md 内联 HTML 表逐字一致，`zero-bubble` 的 12 张表格截图同理。
7. **教材示意图与论文原图不混引**：`aiinfra-pp-*` 全部来自 AIInfra 开源教材页，图注均写明「该图为教材示意图，不是论文原图」；该页第 142、346 行的 alt 文本与画面不符（都写成「args 超参设置」），文章未沿用。
8. **社区图标注层级**：`sea-ai-lab-*`、`xiaodonggua-*`、`yeqianshu-*` 的图注都写明了「社区作者自绘」或「官方图 + 社区标注」，其中 Sea AI Lab 那两张注明是 cut-in-half 的第一手来源。
9. **两张官方调度图的 microbatch 数不同**：`dualpipe.png` 是 8 台设备 × 双向合计 20 个，`dualpipev.png` 是 4 台设备 × 10 个；文章在 `dualpipev.png` 的图注里明确提示不能横向比格子数量。

---

## 四、结论边界

- **第 3 轮新增的三条复核结论**：`zero-bubble-fig3` 下栏的 warm-up F 数是 **7/5/3/1**（像素分类复核，原记 7/6/5/1）；该图上栏 warm-up 后的 3/2/1/0 个白格**一直保留**、构成 ZB-H1 剩余的 $(p-1)t$ 气泡（不是被 $W$ 填掉）；`dualpipe.png` 与 `dualpipev.png` 逐列取色均为**每行 66 个单位、前向 20、空闲 6**，这一读数直接决定了第七章的主口径。
- 本记录的「相符」只表示**画面内容与文章所述一致**，不表示图本身的数据是本文测得的。所有图表都是他人论文、官方仓库或社区文章的素材，文章在每张图下都给出了来源与版本。
- 全量核对的 433 张里有相当一部分只做了内容判断而没有做尺寸与裁切层面的逐像素复核；`notes/IMAGE-SURVEY-*.md` 的「存疑与分歧」小节记录了各处细节（例如 `controllable-memory` 的图 3(b)/(c) 版面顺序、`chimera` 的图例错位、`terapipe` 的裁切越界），引用时以那些记录为准。
- 本记录未与原始 PDF 逐页比对图片裁切范围；`references/raw/` 下保留了原始 PDF，需要时可复查。
