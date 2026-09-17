# PipeDream-2BW（arXiv 2006.16668）图片逐张核对

核对对象：`torch/dualpipe/references/papers/pipedream-2bw/images/` 下的全部 30 张 `.jpg`。
配图来源稿：`torch/dualpipe/references/papers/pipedream-2bw/pipedream-2bw.md`（441 行，MinerU 产物；原始 PDF 在同目录 `references/raw/pipedream-2bw/PipeDream-2BW.pdf`）。
核对方式：30 张**全部**用 `read_image` 逐张打开读图，「内容」列全部为看图后的事实描述（含图中可读文字/数值）；图号归属用 md 引用行 + 图注文本 + **PDF 版面坐标**三方交叉验证，分歧记录在文末。
本文定位：`torch/dualpipe/deep-dive.md` 用这篇论文讲「以 1F1B 换取显存与参数一致性的权衡（double-buffered weights）」，说明「同时驻留多少个 weight 版本 → 显存」这条线索与后来 Zero Bubble 的 W 延迟问题同源。因此本目录有明确的两张候选，其余为不采用。
⚠️ **重要更正**：草稿 `PROMPT-learn-pipeline.md` 第 79 行把 `e1dfea35…jpg` 当作 **Figure 2**，经读图 + PDF 版面核对，**该文件是 Figure 3**；真正的 Figure 2 是 `ec0c258b…jpg`。详见「存疑与分歧」第 1 条。

## 汇总

- `images/` 实际文件数（`ls | wc -l`）：**30**
- 本清单覆盖行数：**30**（逐张一一对应，无跳过、无合并）
- 真插图：**21**；公式截图：**8**；表格截图：**1**；页眉/装饰/碎片：**0**
- 建议采用：**2** 张
- 交叉核对：md 中 `images/` 引用共 **21** 处（`grep -c 'images/' pipedream-2bw.md`），恰好等于真插图数；其余 9 张（8 公式 + 1 表格）在 md 里被 MinerU 以 LaTeX / HTML 形式复现，故未被 `<img>` 引用。`references/README.md` 第 174 行记录的「pipedream-2bw 21 处引用」与本目录核对一致。
- 图号覆盖：论文 Figure 1–12 全部到位（Fig 1 两张子图、Fig 5 四张、Fig 6 三张、Fig 10 三张、Fig 12 两张，其余各一张），**没有缺图**。
- 读图数值说明：表中柱高、曲线终点等带「≈」的数字均为**读图估计**（±1～2 单位），不是论文原文数字；要写进正文的数字请以论文正文与 Table 1 为准。

## 逐张清单

| 文件名 | md 引用行 | 图号/表号 | 类别 | 内容（读图后的事实描述，含图中可读的关键标注/数值） | 结论 | 建议章节 |
| --- | --- | --- | --- | --- | --- | --- |
| `3f3c4cf3b976c84ea604eab2fbfcbdb1f544a411a2cf32be7ebd3389cce140ab.jpg` | 41 | Figure 1(a) | 真插图（调度时间轴） | 4 行 `Worker 1`–`Worker 4` 的调度矩阵；左上标注 `Operations use weight version from last flush`（一条弧线从首格跨到 flush 前最后一格），右侧竖线标 `Pipeline flush`；图例 `Forward Pass`（深蓝）/`Backward Pass`（绿），底部 `Time →`。前段 micro-batch `1 2 3 4`（蓝，逐行右移一格）→ `1 2 3 4`（绿）→ 竖线 flush → `5 6`；图下方标 `(a) GPipe.` | **不采用**：图内文字已经点明主题是「GPipe 靠周期性 flush 保证权重版本一致」。本文第 3 章的 GPipe 基线已采用 GPipe 论文 Figure 2 与 AIInfra 教材 `10pipeline01.png`，本图功能重复 | — |
| `d4e4622f471ade85d38cc119eb9d9a4b1134378aa62a031272c398950eb8c0d2.jpg` | 43 | Figure 1(b) | 真插图（调度时间轴） | 同版式 4 行 `Worker 1`–`4`，**没有** flush 竖线；图例 `Forward Pass`/`Backward Pass` + `Time →`。右上标注 `Before: W_1^{(1)}, W_1^{(2)}, W_1^{(3)}, W_1^{(4)}` / `After: W_1^{(2)}, W_1^{(3)}, W_1^{(4)}, W_1^{(5)}`；右下标注 `Before: W_4^{(4)}` / `After: W_4^{(5)}`。稳态段每行 F/B 交替、micro-batch 编号连续到 8；图下方标 `(b) PipeDream.` | **不采用**：这是「worst case 驻留 d 个权重版本」的时间轴，恰是本文要对比的反面。但同一对比由同页的 Figure 2（只驻留 2 个版本）承担更贴题，且两图画面高度雷同，**若采用 Figure 2 就不应再同时放本图** | §3/§6（若只想要「d 个版本」这一侧的对照，可与 Figure 2 二选一） |
| `e1dfea353f4b22452d870c15cf39f8c320d8280e6653b54da2fa31aac0fe0ae0.jpg` | 62 | **Figure 3**（草稿误记为 Figure 2） | 真插图（调度时间轴，双面板） | 上下两块，各为 2 行 `Worker 1`/`Worker 2` 的调度矩阵。上块：`Operations use weight version from last flush` + 竖线 `Pipeline flush`，`1 2 3 4`（蓝）→ `1 2 3 4`（绿）→ flush → `5 6 7 8`（绿）与 `5 6`，下方标 `(a) GPipe.`；下块：同样有 `Operations use weight version from last flush` + `Pipeline flush` 竖线，稳态段 F/B **交替**（Worker 1 为 `1 2 1 3 2 4 3 4 5 6 5 7 6`，Worker 2 为 `1 1 2 2 3 3 4 4 5 5 6 6 7`），下方标 `(b) PipeDream-Flush.` | **不采用**：内容（flush 式 1F1B 的显存/吞吐权衡）在本文第 3 章由 Megatron combined 1F1B 与 Zero Bubble 时间轴覆盖；**且必须先改正图号**（它是 Figure 3，不是 Figure 2）——原草稿若按 Figure 2 引用会引错 | —（若 §3 要「flush 派」对照图可备选，须同时改图号与文件名） |
| `ec0c258b472ef66da4148bdd10d376d6db66777cc15ef85a1023d2d1d9d7e81a.jpg` | 64（md 第 65 行的图注块把 Figure 2、Figure 3 两条图注合并在这里） | **Figure 2** | 真插图（调度时间轴） | 4 行 `Worker 1`–`4` 的 2BW 时间轴，`Time →`，图例 `Forward Pass`（蓝）/`Backward Pass`（绿）。各 worker 前段只跑 `1 2 3`（蓝，逐行右移）；**棋盘格（checkered）绿格**出现在 Worker 1–4 的同一列（格内数字 `4`）＝新权重版本生成点；右上标注 `Before: W_1^{(0)}, W_1^{(0)}` / `After: W_1^{(0)}, W_1^{(4)}`；右下标注 `Before: W_4^{(0)}, W_4^{(0)}` / `After: W_4^{(0)}, W_4^{(4)}`；右端标注 `t = 21` | **采用**：本文要讲的「同时驻留几个 weight 版本 → 显存」与「延迟 W」两条线索，在这张图上一次性给出——只 stash 2 个版本、新版本落地位置（棋盘格）、以及 in-flight 的 input 7 仍用旧版本（`t = 21`）。是 PipeDream-2BW 在本文章里唯一的正面证据图 | **§3**（1F1B → Zero Bubble 的调度演进：weight 版本数从 d 降到 2 的转折点）+ **§6**（显存口径来源） |
| `aa4b4d45c5e04927a46b55da5a7192d2508a2224e7d2e4dc29fdf2141233d1cf.jpg` | 74 | Figure 4 | 真插图（划分结构示意） | 左：一组竖条标 `Original DNN model`；中间大黑箭头，箭头上下分别写 `Partitioned into parallel pipelines` 与 `Input minibatch split over pipelines`；右：三列 `Stage 1`/`Stage 2`/`Stage 3` × 两行（上行 `GPU 1`/`GPU 2`/`GPU 3` 黄色竖条 + GPU 卡图标，下行 `GPU 4`/`GPU 5`/`GPU 6` 蓝色竖条）；右侧竖排标注 `Width w = 2`，底部标注 `Depth d = 3` | **不采用**：讲的是 (w,d) 等宽等深的「平行流水线」划分卖点（§3.3/§4 规划器）。本文第 4/5 章讲的是 DualPipe 的 (F0,B1,F1,B0) 配对与 V 形布局，这张 2×3 网格图不但无关，还容易被误当成 DualPipe 的 V 形排布 | — |
| `dff98b0b76e56fed74ecb26f121f91405bcdbd74036dcb5bb8b4af59cedf0bce.jpg` | 129 | Figure 5(a) 左 | 真插图（折线图） | 纵轴 `Training loss` 1.5–4.5+，横轴 `Iteration` 0–400000+；两条几乎重合的曲线 `2BW`（蓝）与 `Vanilla`（橙），前 100k 迭代 2BW 略高，之后贴合到 ≈1.9 | **不采用**：收敛性曲线，论证「1-stale 的权重更新语义与 vanilla 收敛轨迹接近」（§5.1）。本文只需一句文字结论，曲线不提供额外信息 | — |
| `2447edc0c15b064360459592dbc53f52e2a8e910625be5421a592a62d9dc9597.jpg` | 131 | Figure 5(a) 右 | 真插图（折线图） | 同版式：纵轴 `Validation loss` 1.5–4.5+，横轴 `Iteration` 0–400000+；`2BW`（蓝）与 `Vanilla`（橙）在 100k 后基本重合（≈1.9），前段 2BW 略高 | **不采用**：同上 | — |
| `5a9a71d01d2e28d24346a3142d342171537d39d65107f34e1050bb2de84c506f.jpg` | 134 | Figure 5(b) 左 | 真插图（折线图） | 纵轴 `Training loss` 2.5–5.0，横轴 `Iteration` 0–300000；`2BW`（蓝）与 `Vanilla`（橙）几乎重合，300k 时 ≈2.75 | **不采用**：同上 | — |
| `043b19e4228192eefec2472764e24b717ec5a1952ce6dea418c57086375d3da1.jpg` | 136 | Figure 5(b) 右 | 真插图（折线图） | 纵轴 `Validation loss` 2.5–5.0，横轴 `Iteration` 0–300000；两条曲线在 100k 后贴合（≈2.75） | **不采用**：同上 | — |
| `677ea82411b01e698e9b1da873feb501a133723a629f1b79a0bc017c34d7add3.jpg` | 166 | Figure 6(a) | 真插图（柱状图） | 纵轴 `Throughput (seqs/second)` 0–45，横轴 `Batch size`（64 / 256）；图例 `Inter-layer MP`（蓝斜纹）/`Tensor MP`（橙格）/`GPipe`（绿格）/`PipeDream-Flush`（红）/`PipeDream-2BW`（紫）。读图估计——64：≈7 / ≈25 / ≈18 / ≈26 / ≈43；256：≈7 / ≈26 / ≈18 / ≈38 / ≈44 | **不采用**：跨系统吞吐 benchmark（GPT 2.2B、8×V100）。本文只要「以 1F1B 换显存」的定性权衡，不需要把 2BW 的吞吐优势搬进来 | — |
| `ccee2353594bcceb976c816307dbd55601c930c460f4eac837cf793d7ba34d34.jpg` | 169 | Figure 6(b) | 真插图（柱状图） | 同版式：纵轴 0–160（刻度 0/40/80/120/160），横轴 512 / 2048；512：≈11 / ≈88 / ≈80 / ≈110 / ≈160；2048：≈12 / ≈155 / ≈80 / ≈145 / ≈168 | **不采用**：同上 | — |
| `93b9aa95a22ee8fbb65548ebe9a9aa3995d1910359d1209cb6c34ef257dc1089.jpg` | 172 | Figure 6(c) | 真插图（柱状图） | 同版式：纵轴 0–130（刻度 0/30/60/90/120），横轴 512 / 2048；512：≈10 / ≈8 / ≈38 / ≈75 / ≈127；2048：≈12 / ≈8 / ≈38 / ≈115 / ≈135 | **不采用**：同上（GPT 3.8B、16-way model parallelism） | — |
| `d16b6456d4433851a245d1d22a82876b63fc0537b5679670b87b6761abaff595.jpg` | 182 | **Figure 7** | 真插图（柱状图） | 纵轴 `Memory footprint (GB)` 0–15（刻度 0/3/6/9/12/15），横轴 `Batch size`（64 / 256）；图例同上五色。**`GPipe`（绿格）在 64 与 256 两处都没有柱，只有竖排文字 `OOM`**；读图估计——64：Inter-layer MP ≈8.6、Tensor MP ≈7.4、PipeDream-Flush ≈10.1、PipeDream-2BW ≈11.3；256：≈8.5 / ≈7.4 / ≈10.1 / ≈11.3 | **采用**：本文第 6 章要统一「气泡/显存」的比较口径，这张图正是口径差别的实证：同样 8×16GB V100、同样 batch 64，activation stash 口径不同的 GPipe 直接 `OOM`，而 2BW（多一份 weight 版本）反而能跑。用来支撑「先声明驻留了哪些版本，再比显存」这条方法论 | **§6**（显存比较口径统一） |
| `ab90d8756538de2dcce616a35c5094c26ec2d39876ddf0975c31dccef597a0b6.jpg` | 185 | Figure 8 | 真插图（折线图） | 纵轴 `Throughput (seqs/second)` 0–300，横轴 `Batch size` 为**对数轴** 2^6…2^11；曲线 `(4, 1)`（蓝圆点，≈85 → ≈235，单调上升）与 `(8, 1)`（橙菱形，≈120 → ≈160，趋平）；另有一个孤立绿方块 `(8, 32)` 落在 2^11、≈285 | **不采用**：讲 (d, b) 配置与 global batch size 的调参权衡（§5.4），与本文主线无关 | — |
| `fc1a75f5944b38dd35673d0a260694c932d0d40c869e7ea4e9e6e6cf28e8eb27.jpg` | 196 | Figure 9 | 真插图（散点图） | 纵轴 `Maximum model size (billion parameters)` 0–30（刻度每 5），横轴 `Model parallel size`（1/2/4/8/16/32/64）；读图估计的散点：(1, ≈1.2)、(2, ≈1.7)、(4, ≈2.5)、(8, ≈4.5)、(16, ≈8.6)、(32, ≈15.5)、(64, ≈28.5) | **不采用**：论证「更深的流水线能装更大的模型」（§5.5）。本文第 5/6 章讨论的是 DualPipe 的气泡与激活驻留口径，不是 maximum model size 曲线 | — |
| `5d755a38f381636abc346e3bc5b677d98331b274ba9a6e8d7608d83310739a37.jpg` | 419 | Figure 10(a) | 真插图（柱状图） | 纵轴 `Throughput (seqs/second)` 0–50，横轴 `Batch size` 64 / 256；读图估计——64：≈7 / ≈26 / ≈23 / ≈24 / ≈46；256：≈7 / ≈26 / ≈22 / ≈38 / ≈47（五色图例同 Figure 6） | **不采用**：与 Figure 6 同族的 BERT 吞吐图（附录 §B.1），信息重复且与主线无关 | — |
| `a83c551bc9f812012c7626af765f1b8ee546b23393df65e091b667c4f229f87c.jpg` | 423 | Figure 10(b) | 真插图（柱状图） | 同版式：纵轴 0–160，横轴 512 / 2048；读图估计——512：≈30 / ≈88 / ≈122 / ≈110 / ≈160；2048：≈30 / ≈158 / ≈122 / ≈145 / ≈168 | **不采用**：同上（附录 §B.1） | — |
| `092360a79992e7ef7f58c087b29e3495223f681553facbfca187002dc3447741.jpg` | 427 | Figure 10(c) | 真插图（柱状图） | 同版式：纵轴 0–160，横轴 512 / 2048；读图估计——512：≈10 / ≈8 / ≈58 / ≈75 / ≈140；2048：≈12 / ≈8 / ≈58 / ≈115 / ≈145 | **不采用**：同上（附录 §B.1） | — |
| `a3259bf456a84ebf714d5f6a606752b15f82b7c3b63f2f06afc3487872e19c82.jpg` | 432 | Figure 11 | 真插图（柱状图） | 纵轴 `Memory footprint (GB)` 0–15，横轴 64 / 256；与 Figure 7 的关键差别：**`GPipe`（绿格）在 64 处有柱 ≈13.3（不是 OOM）**，只在 256 处是竖排 `OOM`；其余读图估计——64：Inter-layer ≈8.2、Tensor ≈7.3、Flush ≈9.0、2BW ≈10.5；256：≈8.2 / ≈7.3 / ≈9.2 / ≈10.6 | **不采用**：BERT 版显存图，与 Figure 7（GPT 版）信息重复。第 6 章若只放一张显存对照，用 Figure 7（它同时给出「同 batch 下 GPipe OOM」这个更锋利的对照） | — |
| `ecc84ab4b6b70c1cd061731832c3f3d150548aeb258a6e201305d2db828db4f7.jpg` | 435 | Figure 12(a) | 真插图（折线图） | 纵轴 `Throughput (seqs/second)` 0–70+，横轴 `Microbatch size` 1/2/4/8/16；两条曲线 `Act. recomp.`（蓝圆点：1→≈31、2→≈47、4→≈58、8→≈63、16→≈66）与 `W/o act. recomp.`（橙圆点：1→≈40、2→≈57、4→≈67，其后无点）。mb=4 时橙线高于蓝线 | **不采用**：讲 activation recomputation 的吞吐取舍（附录 §B.2）。本文第 6 章讲显存公式口径，不展开 recomputation 的吞吐实验 | — |
| `4f3b9cfed56f33bb11571954913fc8d6c32f55c8fb6638f2e90d14b539b7095e.jpg` | 438 | Figure 12(b) | 真插图（折线图） | 同版式：纵轴 0–45，横轴 1/2/4/8/16；`Act. recomp.`（蓝）从 1→≈22 升到 16→≈43；`W/o act. recomp.`（橙）**只有一个点**，落在 mb=1、≈29 —— 即不重算时 microbatch 只能开到 1，重算后可开到 16，最终吞吐反而更高（正文：重算在 Figure 12b 中更快，在 12a 中不然） | **不采用**：同上 | — |
| `25a269074cc229033bc1b37fb9165f4a3c67917a75c4e627128815c3fdc1f196.jpg` | 未引用（md 第 317–319 行为同一式的 LaTeX） | 公式（§A.1.1，无流水线时每个 microbatch 的 t） | 公式截图 | `t = Σ_i max( T_i^{comp}(b, w, d) + Σ_j T_{j→i}^{comm}(b, w, d), (1/m(b)) · T_i^{comm}(b, w, d) )`（两行） | **不采用**：公式截图，非插图 | — |
| `e1329d9690ba762c1a4bab5c1c36a8c5bb7fb62e9e9e6e900a321a1426ae7823.jpg` | 未引用（md 第 323–325 行） | 公式（§A.1.1，有流水线时，用 b′） | 公式截图 | `t = max_i max( T_i^{comp}(b', w, d) + Σ_j T_{j→i}^{comm}(b', w, d), (1/m(b')) · T_i^{comm}(b', w, d) )`（两行） | **不采用**：公式截图，非插图 | — |
| `0875ea4255cbe73ab482760987e5ca5c407a73c0d4591606f63a9790e3d0fa25.jpg` | 未引用（md 第 329–331 行） | 公式（§A.1.1，含 activation recomputation 的 t） | 公式截图 | `t = max_i max( c^{extra} · T_i^{comp}(b, w, d) + Σ_j T_{j→i}^{comm}(b, w, d), (1/m(b)) · T_i^{comm}(b, w, d) )`（两行，多出常数 `c^{extra}`） | **不采用**：公式截图，非插图 | — |
| `4c149d10f43346de899b37b0c1b3febff22c06492fadd84fa4589b84a81cfe6c.jpg` | 未引用（md 第 375–377 行） | 公式（§A.1.1，跨 stage 通信时间） | 公式截图 | `T_{i→j}^{comm}(b, w, d) = 2 · |A^{inp.+out.}(b)| · 𝕀(d > 1) / bwdth_depth(d)`（分子含指示函数，2 倍是为反向的梯度） | **不采用**：公式截图，非插图 | — |
| `714055638da492aefffa10d069f7a31a899cecd88d929403d4cd0eefe7b8d670.jpg` | 未引用（md 第 383–385 行） | 公式（§A.1.1，带宽函数分段定义） | 公式截图 | `bwdth_width(w) = { B_high  if w < number of GPUs in server ;  B_low  otherwise }`（右大括号两行） | **不采用**：公式截图，非插图 | — |
| `d971897a87b9f34fb6cad22e855e432f1669cc811e4451a3fd2c01106f19b300.jpg` | 未引用（md 第 393–395 行） | 公式（§A.1.2，**无** activation recomputation 的显存） | 公式截图 | `2|W| / d + d|A^{total}(b)| / d + d|A^{input}(b)|` —— 中间项带因子 d（d 份激活全驻留） | **不采用**：公式截图，非插图。**注**：这一条与下一条是本文第 6 章「显存口径」最该复现的一对式子，请在正文用 LaTeX 写出（并与 Zero Bubble 的显存式并列），不要引截图 | §6（用 LaTeX 复现） |
| `aa277bf6c0d09e45500af1651d081b53ae0398b1e1a4216c438f30def726349a.jpg` | 未引用（md 第 399–401 行） | 公式（§A.1.2，**有** activation recomputation 的显存） | 公式截图 | `2|W| / d + |A^{total}(b)| / d + d|A^{input}(b)|` —— 中间项没有因子 d（只留 1 份激活） | **不采用**：公式截图，非插图。与上一条配对读：weight 版本数 `2` 不变、激活版本数 `d → 1`，正是「驻留几份 weight / 几份 activation 决定了显存口径」的最小示例 | §6（用 LaTeX 复现） |
| `f34c26205a688a1c80e64d27b36cf222d444ae32dd7582e285bec2e398e93899.jpg` | 未引用（md 第 81–83 行） | 公式（§3.1，2BW 的权重更新语义） | 公式截图 | `W^{(t+1)} = W^{(t)} − ν · ∇f( W^{(t−1)} ).` —— 全 stage 统一的 1-step 延迟 | **不采用**：公式截图，非插图。**注**：这正是本文第 3/6 章用来对照 Zero Bubble「W 延迟」的基准式，建议正文用 LaTeX 写出 | §3（用 LaTeX 复现） |
| `a3d522397ffb1931ea047ae18c9ada1ac4e5eb9219932dadb6fb4d97af0a20be.jpg` | 未引用（md 第 156 行为同内容的 HTML 表） | Table 1 | 表格截图 | 5 列表头 `Task / Metric / Vanilla / Vanilla (90%) / 2BW`；`MNLI  Overall Acc.  87.77% / N/A / 87.82%`；`RACE  Overall Acc.  80.06% / 79.30% / 79.48%`。与 md 第 156 行 HTML 表**逐个数字一致** | **可转写为 Markdown 表格，不直接引图**（若正文要给出「1-stale 语义不损下游精度」的实证，3 行 Markdown 表即可，不必引截图） | §3（用 3 行 Markdown 表复现） |

## 采用图的正文引用块（可直接粘贴）

落地时请把源图从 `references/papers/pipedream-2bw/images/<原名>` 复制到 `torch/dualpipe/pics/<建议文件名>`（`pics/` 目录当前不存在，需新建；相对路径按 `deep-dive.md` 所在目录 `torch/dualpipe/` 计算）。

### 1. `pipedream-2bw-fig2-2bw-timeline-two-weight-versions.jpg`（源 `ec0c258b472ef66da4148bdd10d376d6db66777cc15ef85a1023d2d1d9d7e81a.jpg`，第 64 行）

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/pipedream-2bw-fig2-2bw-timeline-two-weight-versions.jpg" alt="PipeDream-2BW 的 2BW 双缓冲权重时间轴：Worker 1-4 四行，Forward Pass 蓝格 / Backward Pass 绿格，棋盘格绿格标出新权重版本落点，右上标 Before W1(0),W1(0) / After W1(0),W1(4)，右下标 Before W4(0),W4(0) / After W4(0),W4(4)，右端标 t=21" style="width: 100%;">
</div>

> **图片来源**：PipeDream-2BW《Memory-Efficient Pipeline-Parallel DNN Training》（arXiv `2006.16668`）Figure 2，§3.1 Double-Buffered Weight Updates (2BW)。本地副本：`references/papers/pipedream-2bw/pipedream-2bw.md` 第 64 行引用图。

### 2. `pipedream-2bw-fig7-gpt2b-memory-footprint-gpipe-oom.jpg`（源 `d16b6456d4433851a245d1d22a82876b63fc0537b5679670b87b6761abaff595.jpg`，第 182 行）

<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/pipedream-2bw-fig7-gpt2b-memory-footprint-gpipe-oom.jpg" alt="GPT 2.2B 在 8×16GB V100 上的最坏显存占用对比：纵轴 Memory footprint (GB) 0-15，横轴 Batch size 64/256，五色柱为 Inter-layer MP / Tensor MP / GPipe / PipeDream-Flush / PipeDream-2BW，其中 GPipe 在 64 与 256 两处均为竖排 OOM 无柱" style="width: 80%;">
</div>

> **图片来源**：PipeDream-2BW《Memory-Efficient Pipeline-Parallel DNN Training》（arXiv `2006.16668`）Figure 7，§5.3 Memory Footprint。本地副本：`references/papers/pipedream-2bw/pipedream-2bw.md` 第 182 行引用图。

## 存疑与分歧

1. **（必须更正）草稿把 Figure 3 当成了 Figure 2。** `PROMPT-learn-pipeline.md` 第 79 行写「PipeDream-2BW Figure 2 = `papers/pipedream-2bw/images/e1dfea35…jpg`，double-buffered weight 与 1F1B 的显存/一致性权衡」。核实结论：**不成立**。
   - 读图事实：`e1dfea35…` 画的是 2 行 worker 的 `(a) GPipe.` 与 `(b) PipeDream-Flush.` 两块时间轴（都带 `Pipeline flush` 竖线），即论文 §3.2 的 **Figure 3**「Timelines of GPipe and PipeDream-Flush for 2 stages」。
   - 读图事实：`ec0c258b…` 画的是 4 行 worker、带**棋盘格绿格**与 `Before/After: W_1^{(0)},W_1^{(0)} → W_1^{(0)},W_1^{(4)}`、`t = 21` 的 2BW 时间轴，与 Figure 2 图注「New weight versions are generated in checkered green boxes; `W_4^{(4)}` is first used for input 9's forward pass」逐项吻合；正文 §3.1 也提到「input 7 on worker 3 at `t = 21`」，只有这张图有 `t = 21`。
   - 版面佐证：PyMuPDF 读 PDF 第 3 页，`ec0c258b…` 位于页面上方（bbox y≈84–231，横跨两栏），`e1dfea35…` 位于页面下部左栏（bbox y≈320–532）；`pdftotext` 文本流中 `Figure 2. Timeline showing PipeDream-2BW's double-buffered weight update (2BW)…` 出现在 `Figure 3. Timelines of GPipe and PipeDream-Flush for 2 stages.` **之前**。MinerU 的 `content_list_v2.json` 也把 `Figure 2.` 图注挂在 `ec0c258b…` 上。
   - 结论：Figure 2 = `ec0c258b472ef66da4148bdd10d376d6db66777cc15ef85a1023d2d1d9d7e81a.jpg`；若要引用「double-buffered weight 与 1F1B 的权衡」，用这一张，并同步修正草稿第 79 行。
2. **md 第 62–66 行把 Figure 2 与 Figure 3 的图注合并成一个文本块，且都放在 `ec0c258b…` 之后。** 这是 MinerU 的版面读取顺序问题（左栏图先出，再出跨栏图）。**单看 md 极易把 `e1dfea35…` 当成 Figure 2**——这正是草稿出错的原因。规范做法：图号一律以「读图内容 + PDF 版面」为准。
3. **Figure 5 的 (a)/(b) 归属靠 PDF 坐标判定，图内没有模型名。** 四张都是 `2BW` vs `Vanilla` 的 loss 曲线，图内只有 `Training loss`/`Validation loss` 与 `Iteration` 轴，看不出 BERT/GPT。用 PyMuPDF 取 PDF 第 6 页坐标：`(a) BERT, 355M (batch size = 1024).` 在 y=144–153（位于第一排图之下、第二排图之上），`(b) GPT, 355M (batch size = 512).` 在 y=230–240（位于第二排图之下），`Figure 5.` 总图注在 y=248–257。故：**第一排 = (a) BERT 355M**（`dff98b0b…` 训练 / `2447edc0…` 验证，迭代数到 400k+），**第二排 = (b) GPT 355M**（`5a9a71d0…` 训练 / `043b19e4…` 验证，迭代数到 300k）。两组曲线都不带 (a)/(b) 角标，若正文要引用必须自己注明模型与 batch size。
4. **Figure 6 与 Figure 10 的 (a)/(b)/(c) 同样只能靠纵轴量级 + md 位置区分。** Figure 6 三张分别是 GPT 2.2B/8×V100s（纵轴到 ≈45）、GPT 2.2B/64×V100s（到 ≈170）、GPT 3.8B/16-way/64×V100s（到 ≈130）；Figure 10 三张是 BERT 版，纵轴量级同样递增。图内均无 (a)/(b)/(c) 文字，我已核对与 md 引用行顺序一致，但**引用时勿把 8 卡与 64 卡的结果混用**。
5. **Figure 7 与 Figure 11 的差别只在 GPipe 一项，容易被当成重复图。** Figure 7（GPT 2.2B）里 GPipe 在 batch 64 就是 `OOM`；Figure 11（BERT 2.2B）里 GPipe 在 64 处**有柱 ≈13.3 GB**，只在 256 处 `OOM`。这与正文 §B.1「一个差别是 GPipe 在 batch 64 没有 OOM」完全对应，**不是截图损坏**。若采用 Figure 7 作 §6 的口径例证，就不要同时放 Figure 11。
6. **论文自身图注有一处笔误，转写时不要照抄。** Figure 1 图注原文写「forward and backward passes are assumed to take twice as long as forward passes」（应为 backward passes）；两张时间轴图内实际只标 `Forward Pass` / `Backward Pass` 图例。md 第 44 行原样保留了这句 OCR 文本。
7. **9 张未被 md 引用的图不是「漏图」。** 8 张公式截图与 1 张表格截图的内容，在 md 里分别以 `$$…$$`（第 81–83、317–319、323–325、329–331、375–377、383–385、393–395、399–401 行）与 HTML `<table>`（第 156 行）复现；Table 1 已逐项核对数字一致。`images/` 30 张 > md 引用 21 处属正常。
8. **本目录没有页眉/logo/装饰图，也没有图内碎片。** 30 张全部可归类（21 插图 / 8 公式 / 1 表格），不存在残余类别；论文 Figure 1–12 无缺号。
