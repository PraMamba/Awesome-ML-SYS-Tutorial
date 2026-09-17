# GPipe：Easy Scaling with Micro-Batch Pipeline Parallelism（arXiv 1811.06965）图片逐张核对

资料定位（对 `torch/dualpipe/deep-dive.md` 而言）：**基线来源**。文章用它的朴素 micro-batch 流水线图作为气泡基线（\((p-1)(F+B)\) 的来源），以及「重计算换激活显存」的原始出处。

- 本地副本：`torch/dualpipe/references/papers/gpipe/gpipe.md`（MinerU 产物，266 行）
- 图片目录：`torch/dualpipe/references/papers/gpipe/images/`

## 汇总

- `images/` 实际文件数（`ls torch/dualpipe/references/papers/gpipe/images/ | wc -l`）：**9**
- 本清单覆盖行数：**9**
- 真插图：**4**（Figure 1a、Figure 1b、Figure 2、Figure 3）；公式截图：**0**；表格截图：**5**（Table 1–5）；页眉/装饰/碎片：**0**
- 建议采用：**1 张**（`8f3b60…jpg` = Figure 2）
- md 中带图引用的行：第 37、39、53、136 行；**另外 5 张图在 md 中没有任何引用**（均为表格截图，且表格内容已在 md 中以 HTML `<table>` 另行转录，属重复产物）

## 逐张清单

（按文件名字典序排列，便于与 `ls` 输出逐行对照）

| 文件名 | md 引用行 | 图号/表号 | 类别 | 内容（读图后的事实描述，含图中可读的关键标注/数值） | 结论 | 建议章节 |
| --- | --- | --- | --- | --- | --- | --- |
| `10411567182bc7e9b338117076e11ff551935e8c56c3ce7a33abbd198d95a371.jpg` | —（md 未引用） | Table 3 | 表格截图 | 表头 `GPU / AmoebaNet / Transformer`；`K =` 2、4、8 两列组；唯一数据行 `M = 32`：AmoebaNet 1 / 1.7 / 2.7，Transformer 1 / 1.8 / 3.3。即「无高速互联的 P100 上，K 从 2 增到 8 仍有近线性加速」 | 可转写为 Markdown 表格，不直接引图（md 第 107 行已有同内容 HTML 表） | §6（若要论证「流水线的通信量只在 stage 边界，慢互联不是瓶颈」可复述该数字） |
| `6b3a66413fb64a5b3ba5529bcc8ec981cd90073f27e4d3abb49483fc70b215ab.jpg` | —（md 未引用） | Table 2 | 表格截图 | 表头 `TPU / AmoebaNet / Transformer`；`K =` 2、4、8；三行 `M = 1`→1 / 1.13 / 1.38 与 1 / 1.07 / 1.3；`M = 4`→1.07 / 1.26 / 1.72 与 1.7 / 3.2 / 4.8；`M = 32`→1.21 / 1.84 / 3.48 与 1.8 / 3.4 / 6.3 | 可转写为 Markdown 表格，不直接引图（md 第 97 行已有同内容 HTML 表） | §6（气泡公式中 `M` 的原始实验依据：「`M ≥ 4K` 时气泡几乎可忽略」） |
| `8f3b6054d2586fe0af6cbb7013dd5f4fe4751ca2227d61a94498d6df47eb379e.jpg` | 53（图注在第 51 行） | Figure 2 | 真插图 | 三联图。(a) 左侧纵向 4 个 device：Device 0 持 `F₀/B₀`，Device 1 `F₁/B₁`，Device 2 `F₂/B₂`，Device 3 `F₃/B₃`，顶端箭头汇入 `Loss`，底端箭头汇出 `Gradients`。(b) 右侧上半「朴素模型并行」时间线：`F₀→F₁→F₂→F₃→B₃→B₂→B₁→B₀` 完全串行，每个 block 后跟一个 `Update`，中间白色箭头标注 `Time`——任一时刻只有 1 个 device 在工作。(c) 右侧下半 micro-batch 流水线：4 个 device × 4 个 micro-batch 的网格，标注形如 `F0,0 F0,1 F0,2 F0,3`、`F1,0…F1,3`、`B2,3…B2,0`、`B1,3…B1,0`、`B0,3…B0,0`，网格中部留白区被显式标注为 **`Bubble`**，最右侧一列 4 个 `Update` | **采用** | **§1**（流水线气泡的原始基线图，主用）；**§6**（重算气泡公式时复引同一张图对齐记号） |
| `9f2064ce937b5da44459693607793fa9acf682f758ee4874747c4771fad4e475.jpg` | —（md 未引用） | Table 5 | 表格截图 | 表头 `Batch Size / 260K / 1M / 4M`；`BLEU` 30.92 / 31.86 / 32.71；`Loss (NLL)` 2.58 / 2.51 / 2.46 | 可转写为 Markdown 表格，不直接引图；但内容（大 batch 训练）与本文主线无关，建议也不转写 | — |
| `ad4c547d0425499202922a113623b13c2dfd1f30bf6fc1db011e7e1d85fdd6bc.jpg` | 39（图注在第 35 行） | Figure 1(b) | 真插图 | 散点图：纵轴 `Average BLEU` 30.0–37.0，横轴 `Number of Parameters (Billions)` 对数刻度 0.5→8.0。可读标注：`T(6, 8192, 16)`≈30.4、`T(12, 16384, 32)`≈32.4、`T(24, 8192, 16)`≈34.3、`T(32, 16384, 32)`≈35.4，红点 `T(64, 16384, 32)`≈36.4（6B 参数） | 不采用（内容为「模型越大、多语言翻译质量越好」的动机曲线，与流水线调度无关） | — |
| `b8888643d0b7a3273f10a99e684205e576c342fd5497e3883e230f62f0a65371.jpg` | 37（图注在第 35 行） | Figure 1(a) | 真插图 | 散点图：纵轴 `Top-1 Accuracy` 0.73–0.85，横轴 `Number of Parameters (Millions)` 对数刻度 →600。可读标注：`GoogleNet`≈0.745、`Inception3`≈0.781、`ResNet-152`≈0.799、`ResNeXt-101`≈0.807、`SENet`≈0.827、`NasNetA`≈0.828、`AmoebaNetC(6, 228)`≈0.833，红点 `AmoebaNetB(18, 512)`≈0.843（约 550M） | 不采用（内容为「容量 ↔ 精度强相关」的动机图，与流水线调度无关） | — |
| `ddebd5874e7cbcf34b3de29fd976ab0548686f6377a9486d4f3865a696066c82.jpg` | —（md 未引用） | Table 4 | 表格截图 | 表头 `Dataset / # Train / # Test / # Classes / Accuracy (%) / Previous Best (%)`；ImageNet-2012 84.4 vs 83.9、CIFAR-10 99.0 vs 98.5、CIFAR-100 91.3 vs 89.3、Stanford Cars 94.6 vs 94.8\*、Oxford Pets 95.9 vs 93.8\*、Food-101 93.0 vs 90.4\*、FGVC Aircraft 92.7 vs 92.9\*、Birdsnap 83.6 vs 80.2\* | 可转写为 Markdown 表格，不直接引图；但内容（迁移学习精度）与本文主线无关，建议也不转写 | — |
| `ec1670b027184197af350e13b7fd7d848c6b32974b641dc7ac01fe2df07dc703.jpg` | 136（图注在第 135 行） | Figure 3 | 真插图 | 折线+散点图：横轴 `Languages` 0–100（按训练数据量递减排列），纵轴 `ΔBLEU` −10…25。图例：`Bilingual Baselines`（黑，≈0 水平线）、`T(6, 8192, 16)`（绿）、`T(24, 8192, 16)`（蓝）、`T(12, 16384, 32)`（黄）、`T(32, 16384, 32)`（紫）、`T(64, 16384, 32)`（红）。右端低资源语言区红/蓝曲线升至 ≈14–16，绿色仍在 ≈10 | 不采用（内容为多语言翻译质量评估，与流水线调度无关） | — |
| `f01ecd02de5914a4697813b20908ba127bb8d9411a1ef67f7a3c376f58df6478.jpg` | —（md 未引用） | Table 1 | 表格截图 | 上半：`NVIDIA GPUs (8GB each)`，列 `Naive-1 / Pipeline-1 / Pipeline-2 / Pipeline-4 / Pipeline-8`；AmoebaNet-D (L, D) 为 (18,208)/(18,416)/(18,544)/(36,544)/(72,512)；`# of Model Parameters` 82M→318M→542M→1.05B→1.8B；`Total Model Parameter Memory` 1.05GB→24.62GB；**`Peak Activation Memory` 6.26GB→3.46GB→8.11GB→15.21GB→26.24GB**。下半：`Cloud TPUv3 (16GB each)`，列 `Naive-1 / Pipeline-1 / Pipeline-8 / Pipeline-32 / Pipeline-128`；Transformer-L 3/13/103/415/1663；参数量 282.2M→83.9B；参数显存 11.7G→937.9G；Peak Activation Memory 3.15G→6.4G→50.9G→199.9G→796.1G | 可转写为 Markdown 表格，不直接引图（md 第 71 行已有同内容 HTML 表） | §6（这是本文可引的唯一量化出处：「重计算使 `Naive-1` 峰值激活显存 6.26GB→3.46GB（Pipeline-1），但峰值激活显存随 `K` 增大而回升」；注意这是激活显存口径，不是气泡口径） |

## 采用图的正文引用块（可直接粘贴）

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/gpipe-fig2-naive-vs-microbatch-pipeline.jpg" alt="GPipe Figure 2：(a) 四设备层划分 F0/B0…F3/B3；(b) 朴素模型并行时间线完全串行；(c) micro-batch 流水线 4 设备 × 4 micro-batch，中部留白标注 Bubble、右端 Update" style="width: 100%;">
</div>

> **图片来源**：GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism（arXiv 1811.06965）Figure 2，§2 The GPipe Library / §2.2 Algorithm。本地副本：`references/papers/gpipe/gpipe.md` 第 53 行引用图（图注在第 51 行）。

建议文件名：`gpipe-fig2-naive-vs-microbatch-pipeline.jpg`（原文件 `8f3b6054…jpg`，1050×489，`.jpg` 沿用原扩展名）。

**引用时需要注意（读图结论，与草稿说法有细微差异）**：Figure 2 是三联图，真正对应「气泡基线」的是 **(c)**——4 个 device 的 micro-batch 流水线，网格中部明确写着 `Bubble`；而 **(b)** 是「朴素模型并行」（device 之间完全串行，\(4F+4B\)，不是 \((p-1)(F+B)\)）。正文若要写 \((p-1)(F+B)\)，必须点名 (c)，否则会把 (b) 的「几乎零利用率」误当成气泡基线的量级。

## 存疑与分歧

1. **草稿对 `8f3b60…jpg` 的图号判断正确，但对内容的概括需要收紧。** 草稿称它是「朴素 micro-batch 流水线与气泡基线」；核实结果：图号确为 **Figure 2**（图注 md 第 51 行，图片引用紧随其后第 53 行），但它同时包含 (a) 层划分示意图、(b) **朴素模型并行**（无 micro-batch）时间线、(c) micro-batch 流水线 + `Bubble` 标注。只有 (c) 是气泡基线；草稿描述里的「朴素」和「micro-batch」分属 (b) 和 (c) 两个不同子图。已在「采用图的正文引用块」下写明修正建议。
2. **5 张未引用图全部是表格截图，且是重复产物。** `10411567…`(Table 3)、`6b3a6641…`(Table 2)、`9f2064ce…`(Table 5)、`ddebd587…`(Table 4)、`f01ecd02…`(Table 1) 在 md 中没有任何 `![]()` 引用；其内容已在 md 里以 HTML `<table>` 形式出现在第 107、97、148、119、71 行。若文章要引这些数据，直接从 md 的 HTML 表转写即可，不需要引图。
3. **「重计算换激活显存」在本目录没有对应插图。** 该结论的原始出处是 §2.3 的**正文段落**（md 第 75 行）：`O(N + L/K × N/M)` 对比 `O(N × L)`。唯一可引的量化数据在 Table 1 的 `Peak Activation Memory` 行（图片 `f01ecd02…` / md 第 71 行 HTML 表）。本目录 4 张真插图中没有任何一张讲 re-materialization。
4. 本目录 9 张里没有公式截图（MinerU 把 GPipe 的行间公式都识别成了正文文本或 HTML 表），也没有页眉/logo/装饰/图内碎片。
5. **`torch/dualpipe/pics/` 目录当前不存在**——`torch/dualpipe/` 下只有 `deep-dive_draft-1.md`、`deep-dive_draft-2.md`、`PROMPT-learn-pipeline.md`、`notes/`、`references/`。粘贴上面的引用块前需要先建 `pics/` 并按建议文件名把原图复制过去；本次任务只产出核对表，未创建或复制任何图片文件。
6. 两个草稿（`deep-dive_draft-1.md`、`deep-dive_draft-2.md`）目前**都没有出现 `GPipe`/`gpipe` 字样**，也就是说本目录尚未被正式引用；上面「建议章节」是按下达的 9 节拟定结构（§1 两种等待 / §6 公式重算）给出的，不是从草稿既有引用反推的。
