# TeraPipe：Token-Level Pipeline Parallelism for Training Large-Scale Language Models（arXiv 2102.07988v2）图片逐张核对

资料定位（对 `torch/dualpipe/deep-dive.md` 而言）：**谱系来源**。文章只用它说明**调度粒度谱系**（从 layer 级 stage 到 token 级流水线），说明「流水线的划分粒度」是与「反向前后拆分」正交的另一个设计维度。因此本目录绝大多数图不采用，属预期结果。

- 本地副本：`torch/dualpipe/references/papers/terapipe/terapipe.md`（MinerU 产物，379 行）
- 图片目录：`torch/dualpipe/references/papers/terapipe/images/`

## 汇总

- `images/` 实际文件数（`ls torch/dualpipe/references/papers/terapipe/images/ | wc -l`）：**32**
- 本清单覆盖行数：**32**
- 真插图：**17**（Figure 1a–d、Figure 2a–c、Figure 3、Figure 4、Figure 5a–d、Figure 6a–b、Figure 7、附录图）；公式截图：**11**（式 (1)–(9) 及 §2 的 XAB 分块式）；表格截图：**4**（Table 1 与 3 张附录表）；页眉/装饰/碎片：**0**
- 建议采用：**2 张**（`b78a683e…jpg` = Figure 1(c)、`a73bba64…jpg` = Figure 1(d)），二者是同一张 Figure 1 里同一套 5 设备布局的两种粒度，并置才构成「粒度谱系」的对照；若正文只想要一张，保留 Figure 1(d)、删除 Figure 1(c)（理由见「存疑与分歧」第 3 条）
- md 中带图引用的行：第 19、22、25、28、64、67、69、117、128、217、220、223、226、244、247、257、356 行；**另外 15 张图在 md 中没有任何引用**（11 张公式截图 + 4 张表格截图）

## 逐张清单

（按文件名字典序排列，便于与 `ls` 输出逐行对照）

| 文件名 | md 引用行 | 图号/表号 | 类别 | 内容（读图后的事实描述，含图中可读的关键标注/数值） | 结论 | 建议章节 |
| --- | --- | --- | --- | --- | --- | --- |
| `01f9d94ac1e5f9eb42a9fd2c130205f612e7c6c5a0de216b3b41dcbe992f206e.jpg` | —（md 未引用） | 附录表（对应 Figure 7） | 表格截图 | `Model / Input Sequence Length / Algorithm / Slicing Scheme / Latency (s) / TFLOps (per GPU)`；GPT3-13B，2048：w/o 1.863±0.007 → w/ 1.328±0.037（slicing `[(1, [704, 688, 656])] * 32`）；4096：2.526 → 0.913；6144：3.754 → 0.756；8192：4.978 → 0.636；TFLOPS(per GPU) 8.5792/12.0354、1.5819/4.3765、0.5322/2.6427、0.2007/1.5707 | 可转写为 Markdown 表格，不直接引图 | §9（若正文要引「序列越长，粗粒度流水线的每 GPU 算力利用率越低」的量化依据：8192 时 w/o 仅 0.2007 TFLOPS） |
| `0434aba3cc3f1925afce3a3a29720dbf08bb61d660882519af300b5198e25237.jpg` | —（md 未引用） | 公式 (5) | 公式截图 | `T* = min_{l₁,…,l_M} { Σ_{i=1}^{M} t_i + (K−1) · max_{1≤j≤M} {t_j} }` | 不采用（公式截图，非插图） | — |
| `070bc956e847c6fd0e845e746f603816b9c6cadcb02ad5c29a1a44901db9f379.jpg` | 69 | Figure 2(c) | 真插图 | 4 个 GPU（GPU 1→GPU 4）的时间线，粒度极细：每行由大量红/黄交替小格组成（红、黄各代表一条输入序列，说明一条序列被切成很多 token 片），阶段标题依次为 `Forward` / `Backward` / `Forward`，灰色空闲带只出现在上三角 ramp-up 与个别尾部格，整体气泡明显小于 (a)/(b)。**本图顶部裁进了相邻子图注文本 `(b) Microbatch-based pipeline parallelism with small batch size`（MinerU 裁切越界，不是本子图的标题）** | 不采用（图号与内容已核实无误，但结论是：(i) 它讲的「更细粒度 → 更小气泡」正是 Figure 1(d) 要讲的同一件事，重复；(ii) 它与 GPipe Figure 2(c) 的 micro-batch 网格语义相近，同时采用会让读者分不清两张图的差别；(iii) `Bubble` 的显式标注在 GPipe 图里有，这张没有） | — |
| `11193006c1d2266c111ca2afedc8ae5cff1548c7a500c5ba119f3f3af3cd86e9.jpg` | 257 | Figure 7 | 真插图 | 柱状图：纵轴 `Latency (s)` 0–5，横轴 `Input sequence length` = 2048 / 4096 / 6144 / 8192；`w/o TeraPipe`（蓝）≈1.86 / 2.53 / 3.76 / 4.98，`w/ TeraPipe`（橙）≈1.33 / 0.93 / 0.76 / 0.64；带误差棒 | 不采用（性能评估曲线，服务于 TeraPipe 自身的收益论证，不承担粒度谱系说明） | — |
| `1b325fa630c417260175a16a313c8e830f711a63c65d0fa84b29bbb702975787.jpg` | 356 | 附录图（md 中未编号；正文第 354 行称 "similar to Figure 2 in the main paper"） | 真插图 | 三行时间线 (a)(b)(c)，每行都是 GPU 1–3 × 6 条训练序列（格内标 1…6）；右下角图例：蓝 = `Forward`，绿 = `Backward`。(a) 每个 GPU 最多存 3 条序列的激活，GA（DAPPLE 式调度）可行；(b) 每个 GPU 只能存 2 条序列，序列 3 的前向必须等序列 1 的反向完成并释放激活显存，于是任一时刻只有两个 GPU 在工作；(c) 同 (b) 但启用 TeraPipe，把一条序列切成 a/b 两半（格内标注 `1a 1b 2a 2b …6a 6b`），3 个 GPU 可同时工作 | 不采用（附录里 GA × TeraPipe 的组合论证，需要正文额外引入 GA/DAPPLE 才能读懂，脱离本文主线；其「显存约束反过来限制流水线效率」的论点已在 §6 用 DualPipe 自己的显存公式覆盖） | — |
| `2e555c448c703e87207d74b501aaf7f43d05e1f3a9d5efa4f1ce4dee07015d22.jpg` | 217 | Figure 5(a) | 真插图 | 分组柱状图，标题场景 GPT3-1B：纵轴 `Latency (s)` 0–1.6，横轴配置号 (1)(2)(3)；`w/o TeraPipe` ≈1.53 / 1.03 / 0.93，`w/ TeraPipe` ≈1.26 / 1.03 / 0.92；带误差棒 | 不采用（逐配置的性能柱状图，只服务 TeraPipe 收益论证） | — |
| `3df22840e8870883f07e2004737e4b226350ac8375a8a77f779c2fc4277b903e.jpg` | 244 | Figure 6(a) | 真插图 | 柱状图 GPT3-44B setting (8)：纵轴 `Latency (s)` 0–2.5，横轴 `#Slices` = 1 / 4 / 8 / 16 / DP；≈2.66 / 1.24 / 1.26 / 1.24 / 1.11 | 不采用（DP 切分消融图；「切片数不是越细越好」这条与本文 §9 有一点呼应，但 Figure 6(b) 更清楚，且两图都需正文展开 DP 算法） | — |
| `4dfb84ec919765ca472c078223ca9ab0109197df85ec5ecbb1f32b82badc4ca5.jpg` | 22 | Figure 1(b) | 真插图 | 两个虚线框 `Device 1` / `Device 2`（中间 `…`），各含 `Layer 1 part 1` / `Layer 2 part 1` / `Layer 3 part 1`；每层内用红/黄/绿三段横条表示被切开的矩阵块，层与层之间用双向箭头表示跨设备 AllReduce 同步 | 不采用（这是「操作切分/Megatron 式张量并行」的示意图，本文 §9 若要引它，是为了说明被 DualPipe 排除的另一条路线；不需要图） | — |
| `59bcc6b28162c1f890ef4d29f8e12cbe22cfec8005265480b0dbe391c2f01513.jpg` | 67 | Figure 2(b) | 真插图 | 4 个 GPU（GPU 1→GPU 4）时间线，全程只有 **2 个 microbatch**（红、黄）：Forward 段是两级阶梯后接一大片灰色空闲，Backward 段前半整片为灰，只有右端少量红/黄格。直观呈现「固定显存下序列变长 → minibatch 变小 → 气泡急剧放大」 | 不采用（默认不采用，但与 Figure 2(a)/(c) 相比它是「microbatch 数 M 变小 → 气泡变大」最直观的一张，作为 §6 备选保留，见「存疑与分歧」第 4 条） | — |
| `5b34cd9b19b93395d7de65889e20dbff3f6a5886a385cec8b690367c7b88f389.jpg` | —（md 未引用） | 公式 (4) | 公式截图 | `t_i = t_fwd( l_i, Σ_{j=1}^{i−1} l_j )` | 不采用（公式截图，非插图） | — |
| `664594d57b7774bcc38e939b0d80b6e39d3d0de4840a2ec56511a07415052c0a.jpg` | —（md 未引用） | 公式 (2) 第二行 | 公式截图 | `where α_ts = softmax( ((W_Q h_t)^⊤ (W_K h_s)) / √H )` | 不采用（公式截图，非插图） | — |
| `703551b720ca88a507f5f5dcc62b6645c2585dceeb29aeb5cf43ec874097a091.jpg` | —（md 未引用） | 公式 (8) | 公式截图 | `S*(i; t_max) = min_{1≤k≤i} { S*(i−k; t_max) + t_fwd(k, i−k) \| t_fwd(k, i−k) ≤ t_max }` | 不采用（公式截图，非插图） | — |
| `76b3f22ccab59d336cab01b8048b70645391e3c4c3e924d899c1b2fa2bf99f7e.jpg` | —（md 未引用） | 公式 (3) | 公式截图 | `FFN(h_t) = W₂ σ(W₁ h_t + b₁) + b₂` | 不采用（公式截图，非插图） | — |
| `8c314aa88d219c50bd721357278f73d2f2a984c6193896424f4b6bb5422bba53.jpg` | 223 | Figure 5(c) | 真插图 | 分组柱状图 GPT3-44B：纵轴 `Latency (s)` 0–14，配置 (6)(7)(8)；`w/o TeraPipe` ≈13.3 / 4.3 / 2.66，`w/ TeraPipe` ≈7.1 / 2.8 / 1.1 | 不采用（同 Figure 5(a) 理由） | — |
| `9fba6cd9c289cda26f70a92c232a99f307ec2ac5c473f46afdb6441e003468dc.jpg` | —（md 未引用） | 公式 (7) | 公式截图 | `S*(L; t_max) = min_{l₁+…+l_M = L} { Σ_{i=1}^{M} t_i \| t_i ≤ t_max }` | 不采用（公式截图，非插图） | — |
| `a2f647f313686b7cd9cdc9204cb9a88253d4dcd8f99ebb1b5a628151bb2a831a.jpg` | 247 | Figure 6(b) | 真插图 | 柱状图 GPT3-175B setting (9)：纵轴 `Latency (s)` 0–10，横轴 `#Slices` = 1 / 4 / 8 / 16 / 32 / 64 / 128 / DP；≈9.99 / 2.90 / 1.89 / 1.55 / 1.59 / 2.23 / **3.25** / 1.48。即切得太细（128）反而比 16 更慢 | 不采用（DP 消融图；若要讲「粒度不是越细越好」，§9 用文字 + 该组数字即可，不必引图） | — |
| `a73bba64824371103963da88558e60812c9ebcfbdce52ff1aee1353e24f435ae.jpg` | 28（子图注在第 29 行，Figure 1 总图注在第 30 行） | Figure 1(d) | 真插图 | 自上而下 5 个虚线框：`Device 5` 持 `Transformer layer 5`、`Device 4` 持 `Transformer layer 4`、`Device 3` 持 `Transformer layer 3`、`Device 2` 持 `Transformer layer 2`、`Device 1` 持 `Transformer layer 1`。每个 device 内部放的是**同一层被沿 token 维切开的多段小块**（橙色，越靠后的 position 块越小），虚线块之间用细箭头表示 token 块从上一个 layer 传到下一个 layer，箭头自上而下贯到底 | **采用** | **§2**（最小概念模型：说明「划分粒度」是与反向前后拆分正交的第二个维度，主用）；**§9**（选型时复引，说明粒度受硬件利用率限制） |
| `a7ed7499a31b6b8a0e35a16018c556a1154c9feb16e05279710deeb6bd64cc2e.jpg` | 226 | Figure 5(d) | 真插图 | 分组柱状图 GPT3-175B：纵轴 `Latency (s)` 0–10，配置 (9)(10)；`w/o TeraPipe` ≈9.9 / 5.8，`w/ TeraPipe` ≈1.5 / 1.1。是四张 Figure 5 中差距最大的一格 | 不采用（同 Figure 5(a) 理由；「模型越大收益越大」是 TeraPipe 的结论，不是本文要引的谱系事实） | — |
| `aa3ef9dac81a6beacdb86ae775f3bb6c9b5020256062f75af5360871f7cd55b7.jpg` | 128（图注在第 129 行） | Figure 4 | 真插图 | 上下两条 4-GPU 时间线，横轴 `Time`。上：把序列**均匀**切成 4 段但各段耗时 `t₁…t₄` 不等（`t₁<t₂<t₃<t₄`），红色块标注，慢段后面拖出大片灰色气泡；下：把序列**按耗时**切成 4 段使各段耗时相等，红块整齐对齐，灰色只在上三角。说明流水线总时延由最慢 stage 决定，非均匀切分产生更大气泡 | 不采用（DP 切分动机图，服务于 TeraPipe 的动态规划算法，不承担粒度谱系说明） | — |
| `ab1b203a2a6b17d486e85f626218c7c7c1c970c28c5b6734bac32fb50b2e5d10.jpg` | 64（图注在第 71 行） | Figure 2(a) | 真插图 | 4 个 GPU（GPU 1→GPU 4）时间线，4 个 microbatch 各用一种颜色（红、黄、绿、蓝），阶段划分 `Forward` / `Backward`；灰色空闲构成上三角 ramp-up 与下三角 ramp-down，中部无空隙 | 不采用（与 GPipe Figure 2(c) 语义重复——都是「4 GPU × 4 microbatch 的时间线网格」，而 GPipe 那张还额外标注了 `Bubble`；重复引两张会稀释图的作用） | — |
| `b2b50d24687f151f439137976baef8c556b653669c34fb3b51bbffd596b753ef.jpg` | —（md 未引用） | 附录表（对应 Figure 6） | 表格截图 | `Model / Setting / Algorithm / Slicing Scheme / Latency (s) / TFLOps (per GPU)`。GPT3-44B 6(a)：`#Slices=1/4/8/16/DP` → 2.662/1.241/1.255/1.241/1.111 s，TFLOPS 4.2995/9.2226/9.1197/9.2226/10.3018；GPT3-175B 6(b)：1/4/8/16/32/64/128/DP → 9.990/2.902/1.892/1.547/1.593/2.227/3.252/1.481 s，TFLOPS 1.1300/3.8900/5.9667/7.2973/7.0866/5.0691/3.4714/7.6225；含完整 slicing scheme（如 `[(1, [384, 384, 368, 320, 296, 296])] * 8`） | 可转写为 Markdown 表格，不直接引图 | §9（若正文要给「切片过细反而变慢（128 片 3.252 s 差于 16 片 1.547 s）」的完整数据） |
| `b78a683e0a3b8eee6d67a9733ef0708c4bf996c0bbf75ac21882a5dc07339edb.jpg` | 25（子图注在第 26 行，Figure 1 总图注在第 30 行） | Figure 1(c) | 真插图 | 与 (d) 完全相同的 5 设备布局：`Device 1`–`Device 5` 各持 `Transformer layer 1`–`Transformer layer 5`；差别在每个 device 内部放的是**红、绿两整条 microbatch**（沿时间错位排布），device 之间用弧线箭头自下而上串联，表示「layer 级 stage + microbatch 流」 | **采用** | **§2**（最小概念模型：与 (d) 并置，同一布局下只换「切分维」，把「粒度」这个正交维度讲清楚）；**§1**（layer 级 micro-batch 流水线的另一种画法） |
| `b91d5d46089724b5b147fefb6fe4cffd9ade8d815b36250f2887e6831041a8eb.jpg` | —（md 未引用） | Table 1 | 表格截图 | `Model / N / H / #Params / L / #GPUs / B / #Data / #Pipe / #Op`，10 个配置：(1)(2)(3) GPT3-1B，N=24、H=2048、L=2048、#GPUs=192，B=128 / 72 / 72，#Data=8 / 2 / 1，#Pipe=24 / 12 / 24，#Op=1 / 8 / 8；(4)(5) GPT3-13B，40、5120、#GPUs=320，B=32，#Pipe=20 / 40；(6)(7)(8) GPT3-44B，96、6144、#GPUs=384，B=8，#Pipe=96 / 24 / 48；(9)(10) GPT3-175B，96、12288、#GPUs=384，B=2，#Pipe=96 / 48、#Op=4 / 8 | 可转写为 Markdown 表格，不直接引图（md 第 215 行已有同内容 HTML 表） | §2 / §9（若要举「模型越大 → batch size 只能从 128 降到 2 → 粗粒度流水线越难填满」的例子） |
| `b91f1ff8054ce0092977d6d7e179ba41e7927cdeae93887c92c2b2e3cfef05ff.jpg` | 19（子图注在第 20 行，Figure 1 总图注在第 30 行） | Figure 1(a) | 真插图 | 纵向堆叠的 `Transformer layer 1` … `Transformer layer N`（中间 `…`）；底部是 token 序列 `<sos> Cats are the best`，顶部输出 `Cats are the best <eos>`；层内橙色细箭头画出因果注意力连接（每个位置只连自己及其左侧位置，形成典型下三角） | 不采用（背景示意图，讲的是 LM 的因果性质；本文 §2 的 `I = GΘ`、`W = GᵀX` 数学不需要它，且读图也读不出 DualPipe 需要的 F/B/W 记号） | — |
| `b9fb8ecd590fd0e1623b180b068e207c76702d6befb51f9e8949c67a0dcce8b6.jpg` | 117（图注在第 118 行） | Figure 3 | 真插图 | 上下两幅、共用横轴 `# Input tokens` 0–1024。上：`Time (ms)` 0–2.7，从 ≈1.45 ms 起步，0–256 token 区间近似持平，之后缓升至 ≈2.7 ms。下：`Throughput (tokens / ms)` 0–400，从 0 近似线性上升到 ≈400 后饱和；虚线网格 | 不采用（默认不采用：它论证的是「序列太短打不满 GPU」，属于「粒度不是越细越好」的支撑证据，但该论点在本文里可由 §9 文字承担。若 §9 要为该论点配图，这是首选，见「存疑与分歧」第 4 条） | — |
| `c1fa75aee4b796892ac9d1590a7d1813e818497d0f72367be062d4455a4fd1bd.jpg` | —（md 未引用） | 公式 (9) | 公式截图 | `t_fwd(i, j) = t_fwd(i, 0) + t_ctx(i, j)` | 不采用（公式截图，非插图） | — |
| `deb3060ef859355d4f7cb4ded61787472af029b6a853b4d629b05b532625323b.jpg` | —（md 未引用） | §2 未编号公式（XAB 分块） | 公式截图 | `X A B = X · [A₁ A₂] · [[B₁],[B₂]] = X A₁ B₁ + X A₂ B₂` | 不采用（公式截图，非插图） | — |
| `e993e73210f433d19b390b25745e43e8100dbfdcb7c48604854db71bd7297b0f.jpg` | 220 | Figure 5(b) | 真插图 | 分组柱状图 GPT3-13B：纵轴 `Latency (s)` 0–2.5，配置 (4)(5)；`w/o TeraPipe` ≈2.6 / 1.85，`w/ TeraPipe` ≈1.9 / 1.33 | 不采用（同 Figure 5(a) 理由） | — |
| `efa0417c3968f05bc81e66ea8250f0cc93f52c314ec0f53d64d83d7a4d3c599c.jpg` | —（md 未引用） | 公式 (2) 第一行 | 公式截图 | `SelfAtt(h_t; h₁, …, h_{t−1}) = Σ_{s=1}^{t} α_ts · (W_V h_s)` | 不采用（公式截图，非插图） | — |
| `f7bd9255b2e927d5f6ef73b69f6be76c09eb30db52e02b39777aecfdd4bcfab3.jpg` | —（md 未引用） | 公式 (6) | 公式截图 | `T* = min_{t_max} { S*(L; t_max) + (K−1) · t_max }` | 不采用（公式截图，非插图） | — |
| `fb93f9bce172fd315246d4d9be9141c899b34f178cc3df047f6a09045e454c66.jpg` | —（md 未引用） | 附录主结果表 | 表格截图 | `Model / Setting / Algorithm / Slicing Scheme / Latency (s) / TFLOps (per GPU)`，覆盖 GPT3-1B (1)–(3)、GPT3-13B (4)(5)、GPT3-44B (6)–(8)、GPT3-175B (9)(10) 的 w/o 与 w/ TeraPipe 全量：如 (1) 1.517±0.107 → 1.254±0.160 s；**(6) 13.319±0.067 → 7.103±0.243 s，TFLOPS 0.2148 → 0.4028**；**(9) 9.990±0.005 → 1.481±0.002 s，TFLOPS 1.1300 → 7.6225**；slicing 如 `[(1, [120]) * 4 + [112] * 6 + [104] * 8 + [64])] * 2` | 可转写为 Markdown 表格，不直接引图 | §9（若正文要给「同一模型在不同并行配置下最优粒度不同，粒度选型必须按配置调」的量化依据） |
| `fe9857daa8af6597d61452a27f4affd257c6d1acf370386acb76d0f182d9d489.jpg` | —（md 未引用） | 公式 (1) | 公式截图 | `P(x) = Π_{t=1}^{L} P(x_t \| x₁, …, x_{t−1})` | 不采用（公式截图，非插图） | — |

## 采用图的正文引用块（可直接粘贴）

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/terapipe-fig1c-microbatch-pipeline.jpg" alt="TeraPipe Figure 1(c)：layer 级 micro-batch 流水线，Device 1–5 各持 Transformer layer 1–5，每个 device 内是错位排布的红/绿两条 microbatch" style="width: 46%;">
</div>

> **图片来源**：TeraPipe: Token-Level Pipeline Parallelism for Training Large-Scale Language Models（arXiv 2102.07988v2）Figure 1(c)，§1 Introduction。本地副本：`references/papers/terapipe/terapipe.md` 第 25 行引用图（子图注在第 26 行，Figure 1 总图注在第 30 行）。

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/terapipe-fig1d-token-pipeline.jpg" alt="TeraPipe Figure 1(d)：token 级流水线，同样的 Device 1–5 与 Transformer layer 1–5，但每个 device 内把该层的 token 维切成多段，token 块自上而下逐级传递" style="width: 46%;">
</div>

> **图片来源**：TeraPipe: Token-Level Pipeline Parallelism for Training Large-Scale Language Models（arXiv 2102.07988v2）Figure 1(d)，§1 Introduction。本地副本：`references/papers/terapipe/terapipe.md` 第 28 行引用图（子图注在第 29 行，Figure 1 总图注在第 30 行）。

建议文件名：`terapipe-fig1c-microbatch-pipeline.jpg`（原文件 `b78a683e…jpg`，326×331）、`terapipe-fig1d-token-pipeline.jpg`（原文件 `a73bba64…jpg`，326×340），均沿用原 `.jpg` 扩展名。两张宽度都是 326px 的竖长图，建议并排放在同一行（各 46%）；若 Markdown 渲染器不支持并排，各自单独一行也可以，只是会显得偏窄。

**引用时需要注意**：(c) 与 (d) 是**同一套 5 设备布局**（Device 1–5 各持 Transformer layer 1–5），唯一差别是「device 内部切开的是什么维度」——(c) 切 batch/microbatch，(d) 切 token。这正是文章要说的「划分粒度是与反向前后拆分正交的另一个维度」，因此两张必须配对出现；只引一张会丢掉对照。

## 存疑与分歧

1. **Figure 2 的子图号存在「图内文字与 md 标注不一致」的现象，已核实并裁定为 md 标注正确。**
   - `070bc956…jpg`（md 第 69 行，md 标为 `(c) TeraPipe`）的**图片顶部裁进了一行文字**：`(b) Microbatch-based pipeline parallelism with small batch size`。
   - 裁定依据（两条独立证据）：① **内容**——该图内是极细粒度、红/黄两条序列的小格时间线，气泡很小，与图注「(c) TeraPipe. Pipeline bubbles are substantially reduced because of the improved pipelining granularity」吻合；而 `59bcc6b2…jpg` 只有 2 个 microbatch、大片灰区，与「(b) … 气泡显著增加」吻合。② **版式**——Figure 2 的三个 panel 是全宽纵排、子图注在各 panel 下方；`59bcc6b2…jpg` 顶部同样裁进了上一行子图注的残片，(a)/(c) 的子图注在 md 第 66、70 行有文本、(b) 的却缺失，说明 MinerU 把 (b) 的这行子图注丢进了下一张图的裁切区域。**结论：`070bc956…jpg` 就是 Figure 2(c)，读图与图注一致；顶部那行是相邻子图注，不是本图标题。**
   - 附带分歧：图片内子图注写作 `… with small batch size`，md 总图注（第 71 行）写作 `… with longer sequence (hence smaller minibatch size due to fixed GPU memory). Pipeline bubbles significantly increase.`。两者语义一致（都指 minibatch 变小），措辞不同；已记录，不影响子图归属。
2. **md 未引用的 15 张图构成已全部查清：11 张公式截图 + 4 张表格截图。** 公式截图对应式 (1)、(2) 两行、(3)、(4)、(5)、(6)、(7)、(8)、(9) 与 §2 的 `XAB` 分块式；表格截图是 Table 1（md 第 215 行已有 HTML 表）与 3 张附录表（主结果表、Figure 6 消融表、Figure 7 序列表，md 正文只在第 251 行用一句 "supplementary material" 指代）。这些都不是插图，一律不采用。
3. **Figure 1(c) 与 GPipe Figure 2(c) 有信息重叠，需要正文分工。** 两者都在讲「layer 级 stage + microbatch 流」；区别是 GPipe 那张是时间线网格并显式标了 `Bubble`，TeraPipe 这张是架构式示意图。若文章同时采用 `gpipe-fig2-…` 与 `terapipe-fig1c-…`，正文必须说明前者承担气泡基线、后者只作为与 (d) 配对的那一半；**如果只想保留一张 layer 级图，建议保留 `gpipe-fig2-…` 并删掉 `terapipe-fig1c-…`，此时本目录建议采用数降为 1 张（仅 Figure 1(d)）**。
4. **两张「备选但默认不采用」的图，理由已写明，供文章后续按需启用：**
   - `59bcc6b2…jpg`（Figure 2(b)）：若 §6 需要一张图直观说明「固定显存下 `M` 变小 → 气泡急剧变大」，这张比任何文字都直接。
   - `b9fb8ecd…jpg`（Figure 3）：若 §9 要论证「粒度不是越细越好」（切得太细打不满 GPU），下幅吞吐曲线在 0–256 token 区间近似持平、256 之后才线性上升，是现成证据。
   两者都不是「类别存疑」，而是「内容成立但不属本文主线」，所以结论写「不采用」并在正文里给出可启用的条件。
5. **性能评估类图（Figure 5a–d、Figure 6a–b、Figure 7、附录 GA 图）全部不采用。** 原因统一为：它们度量的是 TeraPipe 相对 GPipe 的加速比（如 (9) 9.99 s → 1.48 s、5.0×–6.75×），属于 TeraPipe 自身的收益论证；本文引用 TeraPipe 只用其「粒度维度」这一概念贡献，不需要复现它的实验结论。附录 GA 图还额外依赖 GA/DAPPLE 背景，引用成本高于收益。
6. 本目录 32 张里**没有**页眉/logo/装饰/图内碎片类图片；除第 1 条所述的图注裁切越界外，没有其他一致性问题。
7. **`torch/dualpipe/pics/` 目录当前不存在**——`torch/dualpipe/` 下只有 `deep-dive_draft-1.md`、`deep-dive_draft-2.md`、`PROMPT-learn-pipeline.md`、`notes/`、`references/`。粘贴上面的引用块前需要先建 `pics/` 并按建议文件名把原图复制过去；本次任务只产出核对表，未创建或复制任何图片文件。
8. 两个草稿（`deep-dive_draft-1.md`、`deep-dive_draft-2.md`）目前**都没有出现 `TeraPipe`/`terapipe` 字样**，本目录尚未被正式引用；上面「建议章节」是按下达的 9 节拟定结构（§2 最小概念模型 / §9 选型）给出的，不是从草稿既有引用反推的。
