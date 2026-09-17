# DualPipe 深度解析：第三轮深度审核报告

> 审核对象：`torch/dualpipe/deep-dive.md`，**读取时的行数：1372**（`wc -l` 与 `awk 'END{print NR}'` 一致，文件以换行结尾；`REVIEW.md`、`REVISION.md` 与本轮任务说明写的 1373 不准确），**md5：`30dac685147e335da31a79354e2091f2`**（审核开始 22:23 与写报告前 22:54 两次计算一致；文件 mtime 为 2026-09-10 11:52:04）
> （行号只作定位线索，引用时请以小节号与原文内容为准）
>
> **报告文件名说明**：审核开始（22:23）时 `notes/` 下没有 `REVIEW-deep-r3.md`；22:33:10 另一个会话创建了该文件，因此本报告按规则写入 `REVIEW-deep-r3b.md`。为保持独立性，**我没有阅读 `REVIEW-deep-r3.md`**，只在一次对 `notes/*.md` 的 `grep 7/6/5/1` 中看到过它的 5 行命中。

## 〇、审核环境记录

```text
$ pwd
/workspace/algorithm/Awesome-ML-SYS-Tutorial

$ git -C /workspace/algorithm/DualPipe rev-parse HEAD
030ce4325f4ebeb437da4ebc6d00a70469dd58ae        # 与 refs/remotes/origin/main 相同

$ wc -l torch/dualpipe/deep-dive.md && md5sum torch/dualpipe/deep-dive.md
1372 torch/dualpipe/deep-dive.md
30dac685147e335da31a79354e2091f2  torch/dualpipe/deep-dive.md
```

<details>
<summary><code>git status --short --branch</code> 完整输出（审核开始时）</summary>

```text
## main...origin/main
 M transformers/kda_linear_attention/kda_linear_attention-deep-dive.md
?? .agent-teams/
?? rlhf/agent-harness-rl/
?? torch/deepep/
?? torch/dualpipe/
?? torch/parallel_dims_lab/PROMPT-learn-review-v2.md
?? transformers/compressed_sparse_attention/PROMPT-learn-pipeline.md
?? transformers/compressed_sparse_attention/PROMPT-review-v2.md
?? transformers/compressed_sparse_attention/PROMPT-review.md
?? transformers/compressed_sparse_attention/deepseek-v4-csa-deep-dive_draft-1.md
?? transformers/compressed_sparse_attention/deepseek-v4-csa-deep-dive_draft-2.md
?? transformers/compressed_sparse_attention/learn-plan.md
?? transformers/compressed_sparse_attention/references/
?? transformers/gated_residual/PROMPT-deep-review-gated-residual-v2.md
?? transformers/gated_residual/PROMPT-deep-review-gated-residual-v3.md
?? transformers/gated_residual/PROMPT-deep-review-gated-residual.md
?? transformers/gated_residual/PROMPT-learn-pipeline-gated-residual.md
?? transformers/gated_residual/gated_residual-deep-dive_draft-1.md
?? transformers/gated_residual/gated_residual-deep-dive_draft-2.md
?? transformers/gated_residual/gated_residual-deep-dive_draft-3.md
?? transformers/gated_residual/learn-plan.md
?? transformers/gated_residual/learn-review-report-v2.md
?? transformers/gated_residual/learn-review-report-v3.md
?? transformers/gated_residual/learn-review-report.md
?? transformers/gated_residual/references/
?? transformers/kda_linear_attention/PROMPT-deep-review-kda-r3.md
?? transformers/kda_linear_attention/PROMPT-deep-review-kda-r4.md
?? transformers/kda_linear_attention/PROMPT-deep-review-kda-r5.md
?? transformers/kda_linear_attention/PROMPT-deep-review-kda.md
?? transformers/kda_linear_attention/PROMPT-learn-pipeline.md
?? transformers/kda_linear_attention/REFERENCES_AUDIT.md
?? transformers/kda_linear_attention/REVIEW-deep-r3.md
?? transformers/kda_linear_attention/REVIEW-deep-r4.md
?? transformers/kda_linear_attention/REVIEW-deep-r5.md
?? transformers/kda_linear_attention/REVIEW-deep.md
?? transformers/kda_linear_attention/REVIEW.md
?? transformers/kda_linear_attention/REVISION.md
?? transformers/kda_linear_attention/kda_linear_attention-deep-dive_draft-1.md
?? transformers/kda_linear_attention/kda_linear_attention-deep-dive_draft-2.md
?? transformers/kda_linear_attention/kda_linear_attention-deep-dive_draft-3.md
?? transformers/kda_linear_attention/kda_linear_attention-deep-dive_draft-4.md
?? transformers/kda_linear_attention/learn-plan.md
?? transformers/kda_linear_attention/pics/
?? transformers/kda_linear_attention/references/
```

</details>

其余基线：Megatron-LM 工作树 HEAD `0d7ecc5d6117ba8eae08d0bd2c7e2a0d9d28f3d6`，`git rev-parse core_v0.19.0` = `5be9626709af2722333bf54797c954c09edeada3`（本轮一律 `git show core_v0.19.0:<path>` 读取）；PyTorch `2.10.0+cu129`（`/usr/local/lib/python3.12/dist-packages/torch/distributed/pipelining/`）；SGLang HEAD `7399c2b5587e1559f3e5a26566ed322e81e1433a`；解释器 `/usr/bin/python`（3.12.3）。

**工作树与并发情况（只记录，未改动）**：

- 既有的 `M transformers/kda_linear_attention/kda_linear_attention-deep-dive.md` 与本任务无关，未触碰。
- `torch/dualpipe/` 整个目录是 untracked，因此**任何一轮的改动都无法用 git 回溯**；`REVIEW.md`（11:51:11）与 `REVISION.md`（11:50:28）写完之后，`deep-dive.md` 在 11:52:04 又被改过一次，这次改动没有记录。
- 审核期间目录里出现了两个**不是本轮产生**的文件：`notes/REVIEW-deep-r3.md`（22:33:10）与 `codes/__pycache__/02_schedule_sim.cpython-312.pyc`（22:26:16）。本轮所有 `python` 调用都带 `PYTHONDONTWRITEBYTECODE=1`，唯一一次以模块方式加载 `02_schedule_sim.py` 发生在 22:54 之后且设置了 `sys.dont_write_bytecode = True`；该 `__pycache__` 按「非本会话创建不动」处理，建议由其创建者清理。
- 本轮临时产物只有会话 scratchpad 里的 `sweep.txt`（`--sweep` 的完整输出），仓库内除本报告外没有新建或修改任何文件。
- 关于 `python-uv` 规则：`uv run --no-project python` 解析到 uv 管理的 CPython 3.11（无 torch），无法复现文章声明的 `torch 2.10.0+cu129` 环境，因此按任务说明直接使用 `python`。

## 本轮严重度口径

任务给出的定义在执行时有一处需要落细，这里先写清，便于第四轮复核同一把尺子：

- **P0**：（i）对被引来源「说了什么 / 画了什么」的陈述，与来源原文或像素直接矛盾（读图、读来源类，本主题的底线）；（ii）承载性结论里的数学或逻辑错误；（iii）把没有成立的验证、复现或交叉验证写成已成立。每条 P0 另标**影响面**（承载性 / 局部）与**改动量**，便于排序。
- **P1**：口径不清、记录与正文漂移、源码摘录被改写、渲染问题，以及不影响任何结论方向的数值旁注错误（这类严格说也是事实错误，逐条注明「可升 P0」）。
- **P2**：措辞、风格、计数表述等细节。

## 总评

**尚未达到可发布水平。** 第 2 轮声称修复的 7 条 P0 在正文里都已修对（逐项独立复核见后），公式代数与三个框架的源码行号也再次全部通过；但第七章承载「DualPipe 比别人好多少」的**口径解释**是错的：官方调度图本身按 `F&B = F + B_full` 绘制（逐列取色：每台设备空闲 6 个单位、总长 66），DeepSeek-V3 Figure 4 画的也是计算串行、通信被隐藏，而文章把 `max(F, B_full)` 当成「重叠成立」的主口径，把 `F + B_full` 当成「串行执行、气泡翻倍」，并用一个发送时序与官方代码不一致的模拟器给这一读法做了「很漂亮的交叉验证」。其余 8 条 P0 仍然集中在「读图 / 读来源」这一前两轮已经暴露的盲区：Figure 3 上栏白格被说成由 W 填掉、AIInfra 那张「ZB-V」其实逐格等于 ZB-H2、Megatron Figure 6 是解析曲线不是实测、Chimera 与 TeraPipe 的内容被说错、附录 A 的加粗结论与 §4/§6 自相矛盾、`codes/README.md` 写了脚本不会打印的输出，外加 5 处 alt 细节。前两轮修复的质量在「被点名的那一句」上是好的，但修复只落在正文，没有传播到被正文当作核对索引的 `notes/`、`pics/README.md`、`codes/README.md` 与 `learn-plan.md`，其中 `MATH-VERIFY.md` 仍把 PP 通信翻倍记在官方 README 头上。预计 P0 的改动集中在 §4.1/§4.4、§5.1、§6.1–6.2、§7 与附录 A，约 30 句加一张图的删改，模拟器可选一处小改。

## P0

| # | 位置（小节号 + 行号） | 问题 | 证据（命令输出／读图结果／论文行号） | 建议改法 |
| --- | --- | --- | --- | --- |
| P0-1 | §2.1 L118；§7.2 L725、L747；§7.3 L774；§7.4 L791、L801；「本文的验证范围」L1350；`codes/README.md` L73、L87。**影响面：承载性**；改动量约 12 句 | `F&B` 两端的语义被读反，因而「DualPipe 相对 1F1B 的气泡优势完全建立在配对块真的重叠了这个前提上」（L747）、「完全落在 C 与 F+B_full 的差上」（L118）、「如果执行器实际是串行的，两者的气泡都要翻一倍」这些核心判断不成立，主口径的数字（`(p-2)t/2`、`(p-1)t/2`、§7.5 的 3 / 4.8% 与 3.5 / 5.5%）只是官方图所画气泡的一半 | ① **文章自己的 §7.4 表**：「配对块串行执行」列 DualPipe `(p-2)t`，1F1B `3(p-1)t`，p=8 时 6t 对 21t，优势依然是 3.5 倍，与 L118、L747 直接矛盾。<br>② **DeepSeek-V3 Figure 4**（读图，且文章 L72 的 alt 也这样写）只有一条 Computation 行，`MLP(B)→MLP(W)→MLP(F)→ATTN(B)→ATTN(W)→ATTN(F)` 串行；报告 `deepseek-v3-report.md` 第 304 行说的是 all-to-all 与 PP 通信「can be fully hidden」。重叠盖住的是通信，两块计算并不并行；在全文「忽略通信」的时间模型里自然值是 `F&B = F + B_full`。L725 把左端解释为「两个 chunk…可以真正并排跑」没有任何来源。社区图 `xiaodonggua-ep-serial-vs-overlap.jpg` 的「通信计算重叠」一栏同样是计算行串行。<br>③ **官方调度图就是按 F+B 画的**：`dualpipe.png` 图例色块 Forward 宽 49 px（x=328–377）、Backward 99 px（633–732）、Overlapped 为 49+100 px（1902–2052）；按网格 [174,3526] ÷ 66 = 50.8 px/单位逐列取色，`dualpipe.png` 的 8 行与 `dualpipev.png` 的 4 行**每行 F 单位 20、空闲 6、总长 66**。6 = `(p-2)t`（p=8），在 DualPipeV 的半 chunk 单位下 = `2(p-1)`（p=4），即 README 式取 `F&B = F + B`；模拟器给出 63 / 3，恰为一半。<br>④ **按官方提交语义逐 op 重放 `step()`**（发送只在下一次 `_commit_and_wait_comm()` 生效、按 (src,dst) FIFO 匹配、通信耗时记 0，见逐项核对 2.3）：C=3 时 DualPipe p=4/6/8 → 气泡 2/4/6，DualPipeV p=2/4/8 → 2/6/14（半 chunk），与 m 无关，正是 README 式在 `F&B=F+B` 下的值；C=2 时各 rank busy 不再相等（p=8,m=20 为 49/50/51，中间 rank 第 4 步 i=0 特判不重叠），只有**最大**空闲才等于 `(p-2)/2`。<br>⑤ 连带的数学错误：L774「DualPipe 与 DualPipeV 的气泡恰好是 ZB1P 的一半」对 DualPipe 不成立（`(p-2)t/2 ≠ (p-1)t/2`）；在官方图口径下 DualPipe 气泡率为 `(p-2)/(3m+p-2)`、DualPipeV 为 `(p-1)/(3m+p-1)`，分母首项与 ZB1P 同为 `3m`，「3m 与 6m 是最容易出错的地方」这段论述随之失效；L791「三者相同不是巧合，而是必然结果」只在 `F&B=F+B` 的会计下成立 | 以 `F&B = F + B_full`（计算串行、通信被完全隐藏，与官方图和 V3 Figure 4 一致）作为主口径，改写 L118、L725、L747、L774、L801、L1350 与 `codes/README.md` L73、L87；`max(F,B_full)` 若保留，只能作为「两块计算真正并行」的假想下界并写明无来源；§7.3、§7.4、§7.5 的主数字换成 `(p-2)t`、`(p-1)t`、6/66、7/67；删除或重写 3m/6m 一段 |
| P0-2 | §7.3 L751–L757、L774（「很漂亮的交叉验证」）；`codes/README.md` L45–L52、L87；「本文的验证范围」L1350。**影响面：承载性**；改动量约 6 句，脚本可选改 10 行 | 模拟器不是官方 `step()` 在依赖层面的忠实镜像：它让配对块里 F 的输出在 F 一结束就放行，而官方代码要等两块计算都做完、到下一个提交点才发送。它与 README 式 `max` 端的吻合是这个未声明的乐观假设造成的数值巧合，不是「不建模重叠」的自然结果 | 源码：`dualpipe.py:205-214` 的 `_forward_backward_chunk` 在 `_forward_backward_compute_chunk` 之后才 `_send_forward(phase0)`、`_send_backward(phase1)`，真正提交发生在下一次调用开头的 `_commit_and_wait_comm()`（L188/L198/L209/L220）；模拟器 `02_schedule_sim.py:244-265` 让下游 F 依赖上游 F 任务的结束时刻。**反例**：p=4、m=8，rank 0 第 4 步 i=0 的配对 `(F0_3, B1_1)`（与脚本 steady_probe 打印的 `F0_3, B1_1, F1_2, B0_0` 一致），官方代码里 `F0_3` 的激活要等 `B1_1` 的 I 与 W（示例 hook 是顺序执行的）做完，在 `_forward_backward_chunk(1, 0)` 的提交点才发往 rank 1；模拟器里 rank 1 的 `F0_3` 可以早 2 个单位开工。整体上：同一组 (p, m) 按提交语义重放得气泡 2（p=4）、6（p=8），模拟器为 1、3。模拟器的循环次数、enable_zb 切换点、W 队列 FIFO、跨 rank 反向只挂 I，都与源码逐项一致（见逐项核对 2.2），问题只在这一条释放时序 | 把「逐行镜像官方 step()」限定为「逐行镜像循环结构与任务次序」，并在 §7.3 与 `codes/README.md` 写明「配对块内 F 输出提前释放」这一乐观假设；删除「模拟器给出的是配对块重叠成立那一端」及其因果解释。若要让脚本对齐官方语义，可把配对块的 F/I 释放时刻改为块结束时刻（改动约 10 行），此时脚本输出应与官方图的 66/6 一致 |
| P0-3 | §4.1 L344；§4.4 Figure 3 alt L397。**影响面：承载性（全文最关键的一张图）**；改动量 2 句 | 「warm-up 结束后每行仍留 3/2/1/0 个白格，而那些白格由尾部起跑的 W 填掉」 | Figure 3 逐格分类（x0=72、pitch 24.5，取格内左上角背景）：上栏四行 `FFFF...BWFBW…`、`.FFF..BFBW…`、`..FF.BFBF…`、`...FBFBF…`，warm-up 与第一个 B 之间的空格 3/2/1/0 **一直保留**，第一个 B 之后到 optimizer 之间空格 0，每台设备空格合计 3 = `(p-1)t`，正是 ZB-H1 剩下的气泡。论文 `zero-bubble.md` 第 46 行原文是「the tail-end bubbles are filled by the later-starting W passes」，被填的是尾部气泡，不是这些白格；文章自己 L402 又说「H1 上栏在 warm-up 之后每行还留 3/2/1/0 个白格…H2 下栏同一位置没有白格」，与 L344 自相矛盾。`IMAGE-SURVEY-zero-bubble.md` 第 54 行写的是「行末的 W（绿）填掉一部分尾部气泡」，所以这处错误是写作时新引入的，不是照抄核对文件 | 改为「warm-up 后每行留下 3/2/1/0 个空格，这就是 ZB-H1 剩余的 `(p-1)t` 气泡；1F1B 冷却期在尾部的空闲则被后起的 W 填掉」 |
| P0-4 | §6.2 L620–L626（`aiinfra-pp-zbv-schedule.png` 的引入句、alt 与图注）。**影响面：局部（装错语义的图）**；改动量：删图或改 3 句 | 这张被当作「ZB-V 的四色排布」「整体呈 V 形」的教材图，内容实际是 **ZB-H2**，与 Zero Bubble Figure 3 下栏逐格相同 | 用同一取色方法（格内 15%/85% 横向、20%/80% 纵向四点中位数）对两图分类，四行 F/B/W/Optimizer 序列与 Figure 3 下栏 **39/39、39/39、39/39、39/39 完全一致**（含 7/5/3/1 的 warm-up 与 optimizer 列 24/25/26/27 的逐行错位）；读图可见每格只有一个数字、没有白/黑两色文字，看不出 V 形。教材页 `aiinfra-pp-1f1b-interleaved.md` 第 69–73 行把这张 `10pipeline06.png` 放在「### ZB-V schedule」标题下，`IMAGE-SURVEY-aiinfra-docs.md` 第 30 行照单全收并写了「排布呈 V 形」，文章沿用 | 删除这张图（与 §4.4 的 Figure 3 下栏重复，§6.1 已有真正的 ZB-V 即 Figure 8）；若保留，图注改为「教材页把 ZB-H2 调度图放在了 ZB-V 标题下」并重写 alt |
| P0-5 | §7.5 L814、alt L817、图注 L820。**影响面：局部**；改动量 3 句 | Megatron-LM Figure 6 被写成「气泡率随数据并行度变化的**实测曲线**」「论文作者的实测数据…曲线口径为 GPT-3 类模型在特定 (p, m, d) 组合下」「趋于平台」，并据此「校准这条量级判断在真实系统里的位置」 | `megatron-lm-gpu-clusters.md` 第 147–150 行 bubble = `(p-1)/m`，第 168–177 行 `(p-1)/m = (n-d)/b′` 且「Figure 6 shows the behavior of the pipeline bubble size for various values of d, n, and b′」，第 123 行「We present analytical models where relevant for the pipeline bubble size」。读图：图例 `n=32,b'=32`、`n=32,b'=128`、`n=128,b'=128`、`n=128,b'=512`；蓝线在 d=1/16/32 为 0.97/0.50/0，恰为 `(n-d)/b′` 的 31/32、16/32、0，两条线降到 0 而不是平台。纵轴是相对理想时长的 `(p-1)/m`，不是文章的 `β=(p-1)/(m+p-1)` | 改为「解析曲线（由 `(n-d)/b′` 画出），横轴数据并行度 d，图例为总 GPU 数 n 与 b′=B/b，纵轴是 `(p-1)/m`」，删除「实测」与「校准真实系统」 |
| P0-6 | 附录 A L1316（加粗结论）。**影响面：承载性（全文收束句）**；改动量 1 句 | 「DualPipe 主要动的是第一个维度（拆分反向），而它的双向布局恰好是第一个维度能成立的前提」 | 文章 §4 的 ZB-H1/H2 在单向布局下就拆分了反向；§6 的 DualPipeV 去掉双向布局后仍在第 3/6/7 步使用 zb 拆分（`dualpipev.py:344-390`）；§5.1 L428 自己写「反向拆分这一条明确继承自 Zero Bubble」；DeepSeek-V3 报告第 304 行把 DualPipe 的关键写成「overlap the computation and communication within a pair of individual forward and backward chunks」，拆分反向是「like in ZeroBubble」 | 改为「DualPipe 的新意在计算通信重叠与双向调度；拆分反向继承自 Zero Bubble，也不依赖双向布局（DualPipeV 就是反例）」 |
| P0-7 | §5.1 L412、L420。**影响面：局部**；改动量 2 句 | Chimera 的两处描述与原文不符：① Figure 2 并排比较了「气泡、显存、吞吐三套口径」；② 「核心做法是把下行的 forward 与上行的 forward 合并成一个 forward doubling 段…用前向加倍、反向减半来配平各设备的工作量」 | ① 读图：Figure 2 只有五组调度网格、Bubble 图例与右侧 `M_θ` / `M_a` 显存柱，没有吞吐；「bubble ratio, memory cost … best throughput」是 **Figure 1** 的图注（`chimera.md` 第 38 行）。② `chimera.md` 第 168–175 行（§3.5 Scale to More Micro-Batches）：forward doubling / backward halving 是 N>D 时拼接多个基本调度单元、去掉中间气泡的方法，「increase the number of micro-batches for each forward pass to two」（Figure 7(d)），「key idea is to equalize the workloads of forward and backward passes」，配平的是前向与反向的工作量；Figure 3（第 81–84 行）画的是两份副本的 down/up pipeline 合并，与 forward doubling 无关 | ① 删去「吞吐」或改引 Figure 1；② 改为「Chimera 用两份副本分别跑 down/up 两条流水线并合并调度（Figure 3）；forward doubling 是 §3.5 在 N>D 时拼接调度单元的技巧」 |
| P0-8 | 附录 A L1306（TeraPipe 图注）。**影响面：局部**；改动量 1 句 | 「TeraPipe 论文自己的 Figure 3 显示，token 粒度低于某个阈值之后吞吐不再提升而通信开销继续上升」 | `terapipe.md` 第 118 行：Figure 3 是「Forward propagation time and throughput for a single layer of GPT3-1B model … on a single NVIDIA V100 GPU」；第 115 行：1 个 token 与 256 个 token 的前向时间相同，「the GPU is not being fully utilized for input sequence lengths less than 256」，切得太细吞吐下降；另一侧代价是「longer input slices lead to fewer pipeline stages … increase the pipeline bubble」。单卡实验里没有通信，「通信开销继续上升」无来源，「吞吐不再提升」把方向说反了 | 改为「Figure 3 显示单层在 256 token 以下 GPU 利用不足，切得太细吞吐反而下降；切得太粗则序列内 stage 变少、气泡变大（第 115 行）」 |
| P0-9 | `codes/README.md` L78；§5.4 L489。**影响面：局部（复现性声明）**；改动量 2 句 | README 写「`p=8, r=1` 时 DualPipe 给出 `(F0_8, B1_4, F1_6, B0_2)`」；正文写「这两组数都可以用 `codes/02_schedule_sim.py` 直接复现」 | 实跑 `python torch/dualpipe/codes/02_schedule_sim.py --p 8 --m 20` 打印 `rank 1: F0_6, B1_2, F1_4, B0_0`；脚本只在 step 4 的 i=0 记录 `steady_probe`（`02_schedule_sim.py:122-128`），任何参数都不会打印 `(F0_8, …)`。x=8 那组需要在 i=0 的基础上各加 2 自行推出 | README 改为实际输出；正文改为「x=6 那组可直接复现，x=8 由主循环每步四个计数器同步 +1 推出」 |
| P0-10 | 5 处 alt / 图注细节（每处 1 句）：§5.5 alt L498、§6.2 alt L615；§6.1 alt L578；§6.1 alt L552；§3.1 L222、alt L225、图注 L228；§3.1 alt L233。**影响面：局部** | ① 「绿格 Backward、浅绿 Backward for input」；② Controllable Memory Figure 4「含 F/B/W 与 optimizer step 方块」；③ Sea AI Lab 图「上栏 DualPipe…F 块排成 V 形；下栏…V 形变窄」；④ GPipe Figure 2「(a) 是朴素的模型并行链，设备之间完全串行；(b) 是只有一个 microbatch 的流水线」以及「把 (a) 或 (b) 的近乎零利用率当成气泡量级」；⑤ AIInfra GPipe 图「分 warmup / 稳定 / cooldown 三段」 | ① `dualpipe.png` 图例 Backward（x=633–732）与 Backward for input（x=988–1037）逐行取样 RGB 同为 (127,214,80)，只靠宽度区分（2 单位 / 1 单位），`dualpipev.png` 相同；② 读图：四个面板没有任何米色 optimizer 格，(a) 1F1B 面板只有 F 与 B 两色；③ 读图：上栏 F 块是上下镜像的两个 V（X 形），下栏保留上半个 V，横向跨度与上栏相同（显示坐标 x≈440–775），只是设备数减半；④ `gpipe.md` 第 51 行图注：「(a) An example neural network with sequential layers is partitioned across four accelerators … (b) The naive model parallelism strategy leads to severe under-utilization … (c) Pipeline parallelism divides the input mini-batch into smaller micro-batches」，(a) 是计算图而非时间轴，无所谓利用率；`IMAGE-SURVEY-gpipe.md` 与 `learn-plan.md` 第 294 行原本的「(b) 才是朴素模型并行」是对的，被 `IMAGE-VERIFY.md` 第 59 行「判定」成了现在的写法；⑤ 读图：只标了 warmup 与 cooldown，GPipe 没有稳态段（「稳定阶段」标在另一张 `10pipeline02` 上） | ① 改为「绿色宽格为完整 Backward、绿色窄格为 Backward for input」；② 删去 optimizer step，(a) 注明只有 F/B；③ 改为「上栏 F 块呈上下镜像的两个 V；下栏只保留上半，设备数减半」；④ 按原图注改为「(a) 四卡切分的网络结构、(b) 朴素模型并行的时间轴、(c) micro-batch 流水线」，删去「(a) 的利用率」；⑤ 删去「稳定」 |

## P1

| # | 位置 | 问题 | 证据 | 建议改法 |
| --- | --- | --- | --- | --- |
| P1-1 | §1 L69；§5.4 L468；总结 Mermaid L1237；总结 L1244 | 驱动问题的答案把「存在互不依赖、可配对的前向与反向」归功于双向布局（「DualPipe 的双向布局就是为了持续提供它」「双方向布局提供了一批天然独立的计算」），但这种独立性并不来自双向 | 1F1B 稳态 `(F_{k+p-r-1}, B_k)` 在 `r < p-1` 时本来就是两个不同 microbatch、无数据依赖，只有最后一个 stage 是 `(F_k, B_k)`；Megatron `core_v0.19.0` 的 `schedules.py:939-943` 注释：「When enabling overlap_moe_expert_parallel_comm, we schedule one extra micro-batch forward step before the 1f1b stages. This is needed to ensure the forward and backward computations are independent in all 1f1b steps.」即单向 interleaved 1F1B 多做一个 warm-up 前向就能在所有 rank 上得到独立配对，文章 §9.2 也写了 Megatron 没有采用双副本拓扑。`(F_0, B_1)` 本身确实无数据依赖（见对抗性证伪第 1 条），问题只在归因 | 把双向布局的贡献写准：不增加 warm-up 就让两条方向的末端 stage 都有另一方向的任务可配，且依赖深度按 `p/2` 计（README 式的 `PP/2-1`）；引用 Megatron 这段注释说明「单向 + 多一个 warm-up 前向」是另一条得到独立配对的路 |
| P1-2 | `notes/`、`pics/README.md`、`codes/README.md`、`learn-plan.md`、`REVIEW.md`、`REVISION.md`、`references/README.md`（明细见下表） | 第 2 轮修复只改了正文，被正文附录 B 当作核对索引的记录仍保留旧错，其中两条是第 2 轮 P0 的回潮；另有几处计数断言本身不成立 | 见下表 | 在记录里就地加「〔第 3 轮订正〕」而不是静默改写；下表作为第四轮的修复传播清单 |
| P1-3 | 全文 486 处 `\(…\)` 行内公式 | GitHub 与 CommonMark 不识别 `\(…\)`，`\(` 按转义字符处理成普通括号，全文几乎所有行内公式（包括第七章每个表格单元格）在 GitHub 上都会显示成原始 TeX | 本地 `markdown-it-py 4.0.0`（CommonMark）把 `设 \(X\in\mathbb{R}\)` 渲染为 `设 (X\in\mathbb{R})`；抽查 6 篇同仓库文章（`torch/fsdp2/readme.md`、`rlhf/sys-design/readme-4.md`、`rlhf/sys-design/readme-2.md`、`sglang/scheduler/readme.md`、`torch/nccl/readme.md`、`transformers/kda_linear_attention/kda_linear_attention-deep-dive.md`），`\(` 均为 0 次，其中 4 篇用 `$`（共 1238 个 `$` 字符）。GitHub 官方数学文档只列 `$…$`、`$$…$$` 与 math 代码块，本轮未联网复核，记 `[UNVERIFIED]` | 统一改成 `$…$`；表格单元格里的 `\|` 需要单独检查 |
| P1-4 | §2.2 L162 | 「其余配置的偏差在 14% 到 17% 之间」（第 2 轮 P2-2 修复时新写入） | 按 `zero-bubble.md` 第 302 行 Table 9 逐行计算 `(2T_F-(T_B+T_W))/2T_F`：1.5B 三行 26.0%/26.0%/26.1%，6.2B 16.9%/17.9%/18.1%，14.6B 14.6%/14.4%/14.3%，28.3B 14.0%/14.0%/13.7%；「其余」11 行的范围是 13.7%–26.1%，其他模型是 13.7%–18.1%；而且 12 行**全部** `T_B < T_F`，不是「还出现了」。严格按定义是事实错误（可升 P0），但不影响任何结论 | 改为「1.5B 三行都约 26%，其余模型在 13.7% 到 18.1% 之间，且 12 行全部 `T_B < T_F`」 |
| P1-5 | §9.1 L1087–L1089 | `register_custom_function` 的「源码摘录」被改写：去掉 `_ComputationType`、`_CustomFunctionProtocol` 两个类型注解并把多行签名并成一行，没有标注 | 本地 2.10.0 `schedules.py:1887-1891` 是带注解的多行签名；本轮对正文 14 段 Python 摘录逐行比对，其余 13 段逐字一致或已用 `...`／注释标出省略 | 按原文贴出，或写明「签名（省略类型注解）」 |
| P1-6 | 参考 · 论文 L1256 | Controllable Memory 条目写「§3、§6、Figure 2/4/6/8」，该论文没有 §6（可能与 Zero Bubble §6 混了） | `controllable-memory.md` 正文只有 1 Introduction 到 5 Conclusion And Future Work，附录 A–I（第 236–399 行）；Figure 6 在 §4.2，Figure 8 在 §4.3，Figure 18 在附录 G | 改为「§3、§4.2–4.3、附录 G，Figure 2/4/6/8/18」 |
| P1-7 | 开篇 L7；文末注释 L1369 | DeepEP 链接与 readme-4 的处理：① 链接指向未发布文章，与 `.learn/config.md`「引用 published 状态的文章」字面冲突；② 注释同一句里既说 readme-4 的主题「改由…`torch/deepep/` 中可复现的通信结论承担」，又说 DeepEP「不是知识来源」；③ 排除 readme-4 的前提「该文在 README.md 中标记为 [Pending Review]」不完整 | `README.md` 第 51 行把 `readme-4-en.md` 列为已发布，并附「Chinese version（`./rlhf/sys-design/readme-4.md`）」与知乎链接；`README-cn.md` 第 64 行也把 `readme-4.md` 列为已发布；只有 `README.md` 第 150 行一条标题为「Expert Parallelism」的旧条目带 `[Pending Review]`；知识图谱 `status: published` 与两处已发布条目一致 | 见「争议项裁决」第 2、3 条 |
| P1-8 | §5.4 L491（第 2 轮 P1-3 修复时新写入） | 「在上半区里，方向 1 的反向进度（下标 `x-p+1+r`）总是不晚于方向 0 的反向进度（下标 `x-p/2`）」把两个下标挂反了，原句按字面不成立 | 同一节 L478–L487 的四元组是 `(F_{0,x}, B_{1,x-p/2}, F_{1,x-p/2+1+r}, B_{0,x-p+1+r})`：方向 1 反向的下标是 `x-p/2`，方向 0 反向的下标是 `x-p+1+r`；以 p=8、r=1、x=6 代入分别为 2 与 0（与 `--p 8 --m 20` 打印的 `B1_2`、`B0_0` 一致），原句读作「0 不晚于 2」。改正后的关系 `x-p/2 ≥ x-p+1+r` 等价于 `r ≤ p/2-1`，这才是「只看上半区」的来由。严格按定义属数学表述错误（可升 P0），不影响四元组本身 | 改为「方向 1 的反向进度（下标 `x-p/2`）不落后于方向 0 的反向进度（下标 `x-p+1+r`），二者之差 `p/2-1-r ≥ 0` 正是 `r < p/2` 的来由」 |

**P1-2 明细（记录与正文 / 来源的漂移）**

| 文件 | 行 | 仍然写着 | 应为 |
| --- | --- | --- | --- |
| `notes/MATH-VERIFY.md` | 101 | 「PP 边界通信约 `2x` ｜ 官方 README 与 Sea AI Lab 博客」 | 只有 Sea AI Lab 博客（`sea-ai-lab-cut-in-half.md` 第 40–49 行）；README 全文 69 行没有 PP 通信量表述。**第 2 轮 P0-6 的回潮** |
| `notes/MATH-VERIFY.md` | 127 | 「它给出的是串行执行下的气泡，是重叠收益的参考上界」 | 第 2 轮 P1-1 已判两义；本轮 P0-1、P0-2 表明它既不是串行口径，也不是按官方语义的重叠口径 |
| `notes/MATH-VERIFY.md` | 77、99 | `b_{1,i} ≥ b_{0,i}`（未定义记号）；「ZB1P 无配对块 → 分母恒为 3m」 | 正文第 2 轮已改（P1-3、P2-3），但 P2-3 的新写法本身有误（P0-1 ⑤） |
| `codes/README.md` | 73、87 | 「同两个气泡值都要翻倍」（病句）；「『串行执行 + 依赖等待』下的气泡，是重叠收益的上界参考」 | 见 P0-1、P0-2 |
| `notes/IMAGE-VERIFY.md` | 39、68 | 「8 个 PP rank、双向各 20 个 micro-batch」「`dualpipe.png` 是 8 台设备 × 双向各 20 个」 | 双向合计 20、每方向 10。**第 2 轮 P0-5 的回潮** |
| `pics/README.md` | 66 | 「`dualpipe.png` 是 8 PP rank × 双向各 20 micro-batch」 | 同上 |
| `notes/IMAGE-VERIFY.md` | 24 | 「已从 `references/papers/terapipe/images/b78a683e…jpg` 复制补齐」 | 现为 `a73bba64…`（同一文件第 52 行已写第 2 轮修正，前后矛盾） |
| `notes/IMAGE-VERIFY.md` | 37 | 「Backward（绿）/ Backward for input（浅绿）」 | 两块图例 RGB 相同（P0-10 ①） |
| `notes/IMAGE-VERIFY.md` | 45 | Controllable Memory Figure 18「画面自 (a) 到 (h) 共八个面板」 | 画面共 9 组，最下一组的标签「(i) Interleaved 1F1B with Uniform Interval」在 md 第 417 行 |
| `notes/IMAGE-VERIFY.md` | 46 | Controllable Memory Figure 4「含 I / F / B 三类小格」 | 画面是 F（蓝）、B（青）、W（绿）；(a) 面板只有 F、B |
| `notes/IMAGE-VERIFY.md` | 53 | Sea AI Lab 图「右侧 F 块确呈 V 形…V 形更窄」 | P0-10 ③ |
| `notes/IMAGE-VERIFY.md` | 59 | 把 `IMAGE-SURVEY-gpipe.md` 的「(b) 才是朴素模型并行」判为措辞差异，改采相反写法 | 论文图注支持原核对文件（P0-10 ④） |
| `notes/IMAGE-VERIFY.md` 7、75；`pics/README.md` 5、77；`REVIEW.md` 152 | — | 「453 张」 | 写作素材 433 张（353+20+60），`references/` 下图片总数 435 |
| `pics/README.md` | 76 | 「31 张图之外」 | 32 |
| `notes/IMAGE-VERIFY.md` 65；`REVIEW.md` 152；`learn-plan.md` 249、299 | — | 「`papers/` 下共 41 张纯公式截图与 15 张表格截图」 | 按 10 份论文核对表逐行统计：公式截图 94 行、表格截图 57 行（`learn-plan.md` 第 236–245 行自己的汇总表两列相加也是 94 与 57）。表格截图的结论写的是「可转写为 Markdown 表格，不直接引图」，与不采用同义；正文 L1332 没有写具体数字，所以这是记录错误 |
| `learn-plan.md` | 249、359 | 「88/31/49/26/32/9/22/30/32/34/20/60，合计 435」「435 张逐张核对已完成」 | 这 12 个数相加是 433；435 含未逐张核对的 community 2 张 |
| `learn-plan.md` | 23、143、339 | 「`torch/deepep/` 的已发布结论」「（均已发布）」 | DeepEP 一篇不在两个 README 中（正文开篇已改） |
| `learn-plan.md` | 102 | 「固定设备数时 DualPipe 需要 4× 而 DualPipeV 需要 2×」 | 正文 §6.3 的 2× / 1× 正确（Sea AI Lab 第 34 行「每个设备的参数量降至原来的 50%」） |
| `learn-plan.md` 94、154；`references/README.md` 44 | — | 「Controllable Memory 论文 §6」「并提供 ZB-V」「含 ZB-V」 | CM 没有 §6，全文 0 处「ZB-V」；这正是第 2 轮 P0-1 的源头 |
| `notes/IMAGE-SURVEY-zero-bubble.md` | 54、80、87 | 「7 / 6 / 5 / 1」 | 7/5/3/1（争议项 1） |
| `notes/IMAGE-SURVEY-aiinfra-docs.md` | 30 | `10pipeline06.png`「ZB-V 调度…排布呈 V 形」 | 内容是 ZB-H2（P0-4） |
| `REVIEW.md` 3、200 | — | 「1350 行左右」「本轮修改后文件变为 1373 行」 | 1372 行；且 11:52:04 的最后一次改动在记录写完之后，未留痕 |
| `REVISION.md` | 142 | P1-5 行称附录 B 已改为「435 张（论文 353、教材页 20、社区文章 60、社区复现代码 2）」 | 正文实际是「433 张（论文 353、教材页 20、社区文章 60）」，正文对，记录与之不一致 |

## P2

- **P2-1**（§2.2 L120、§2.3 L164、§2.4 L176、§5.3 L450、§5.4 L460、§6.5 L681、§8.3 L869）章节标题是问句（「为什么…」），style-guide §3.7 要求概念章节用描述性名词。
- **P2-2**（§3.4 L322）「第三章讲完了外层调度能做的事，接下来该看反向拆分能做的事」接近 §6.6 禁止的「接下来讨论…」，建议改为引用上一章那个未解决的具体问题（完整 B 拉长了跨 stage 依赖）。
- **P2-3**（L114、L368、L1014、L1033）「有两点必须在这里说清…」「这里有两点必须自己算一遍才不会读错。」「把上面几节拼起来，可以得到一个明确判断。」「由此可以得出两条经得起复核的结论。」属于 §9.7 所说的短促独立断言句，建议并回上下文。
- **P2-4**（开篇 L20）致谢对象是「团队」「Sea AI Lab 的几位作者」，style-guide §1.5 要求「各位大哥」式点名、不标组织；全文也几乎没有 §9.2 的幽默与 §9.9 的括号吐槽。
- **P2-5**（§1.2 L46）`**点对点通信（P2P）**完全是两件事`：右侧 `**` 前是全角括号、后接汉字，不满足 CommonMark 右定界规则，`markdown-it` 实测输出字面 `**`（全文唯一一处，对 485 个 `**` 行逐行渲染得出）。
- **P2-6**（附录 A 图注 L1298）「743×1723」实为 742×1723（PIL）；alt 列到 (h)「等」，画面共 9 组，可补 (i)。
- **P2-7**（§3.2 alt L282）「micro-batch 1-12」：画面 Device 3 出现 13。
- **P2-8**（§9.2 L1113–L1117）`aiinfra-pp-interleaved-1f1b-vpp-layout.png` 与两段之前的 Megatron Figure 4 下栏逐格同内容（重复引用），且单面板图上看不出 alt 所说的「交错排布后气泡变小」。
- **P2-9**（§6.1 图注 L547）「底图是官方 DualPipe 调度图」：该图图例是两行带编号的色块（Micros from up to down / down to up），尺寸 3234×504，与官方 3548×590 不同，更可能是按官方调度重绘，建议写「按官方调度重绘并补全反向编号」。
- **P2-10**（§6.1 L557）「半年之后，同一批作者」：arXiv 2401.10241（2024-01）到 2405.15362（2024-05）约 4 个月；两文作者 4 人中 3 人重合（ZB 有 Guangxing Huang，CM 有 Nyamdavaa Amar）。
- **P2-11**（§3.4 L318）用附录第 348 行「m ≤ p 且 `T_W < T_B` 的粗略分析」为 `(m+p-1)(F+B_full)` 背书，而全文在 m ≥ p 下使用（§7.5 取 p=8、m=20）；改引 Table 2（第 70 行）的 1F1B 气泡再加 `m(F+B_full)` 更干净。
- **P2-12**（§8.4 L930、§8.7 L1035）对 `req.wait()` 的解释：本地 `Work.wait` docstring 写的是「calling wait() is the same as calling synchronize(): Letting the current stream block on the completion of the NCCL work」，结论（当前 stream 上的计算不会与这一批 P2P 重叠）成立，但 CPU 并不阻塞，建议写明是 stream 级同步，并限定为「在当前 stream 上排队的计算」。
- **P2-13**（§9.2 L1139）「语义与 Zero Bubble 的延迟 W 同源」无来源：Megatron `megatron/core/transformer/moe/README.md` 第 563 行只写「Split dgrad/wgrad compute」，`model_chunk_schedule_plan.py:244-246` 显示它是层内把 `mlp_bwd_dw` 挪到与 dispatch 重叠的位置，与 Zero Bubble 的调度级延迟 W 作用粒度不同；建议改为「思路同类（都拆 dgrad/wgrad）」。
- **P2-14**（§9.2 L1111）「代价是边界通信变多、峰值激活变高」：Megatron-LM 第 85 行支持通信增加 v 倍，但第 32 行说 interleaved 调度「with comparable memory footprint」，「峰值激活变高」需另给来源或删去。
- **P2-15**（§6.5 L694）「这一点在第七章重算气泡率时会用到，因为它意味着 m 可以取奇数」：第七章与扫掠都只用偶数 m，`02_schedule_sim.py:431-433` 对 DualPipeV 的单次运行也断言 `m % 2 == 0`，承诺没有兑现。
- **P2-16**（§8.5 L959）「第 3 步…是整段 step() 里唯一一处通信与计算被刻意排在一起的地方」：第 2 步 `dualpipe.py:365、368` 同样提前 `_recv_forward(0)`。
- **P2-17**（§5.3 L455）`g_ℓ = (1/m)(…)` 与官方示例逐 microbatch `F.mse_loss` 求和、不除以 m（`example_dualpipe.py:86-100`）的口径不同，§10.1 已提到，建议公式旁注明归一化前提。
- **P2-18**（§8.1 L838、L849）先说「四个功能模块文件…加两个示例」，表里 6 行，再说「这四类组件分别回答…三个不同的问题」，计数对不上。
- **P2-19**（§2.5 L188–L203）`SplitLinear` 是自编教学代码（相对 `codes/01` 还删了 docstring、改了注释标点），style-guide §5.1 更推荐官方 `examples/example_dualpipe.py:13-34` 的 `LinearFunc`：那段同时能讲清 `WeightGradStore.enabled` 的开关，以及「示例里 W 先于 dgrad」这一与模拟器建模相关的事实。
- **P2-20**（§7.3 L759、L774）「实测结果」「与模拟器实测的…」：CPU 依赖模拟的输出不宜称实测，改「模拟输出」。
- **P2-21**（§7.5 L830）「六个面板是用各不相同的 microbatch size」：读图 V-ZB、ZB-1P、1F1B 同为 mbs 4（38.5B 面板为 1），只有 V-Half（8 / 2）与 V-Min（12 / 3）不同；「V-Min 的激活显存反而高于 1F1B」成立（38.5B 面板约 41 GB 对 35 GB）。
- **P2-22**（§9.3 L1159）对 `core_v0.19.0` 配置检查的描述**已核实成立**（`megatron/core/transformer/transformer_config.py:2669-2746`：torch ≥ 2.6、PP>1 时必须指定 VPP、EP>1、dispatcher 为 alltoall/flex、禁用 full 重计算、bf16/fp16、与 `moe_shared_expert_overlap` 互斥等），建议补行号。
- **P2-23**（§1.2 L56）「通信从可有可无的边界开销变成了与计算同量级的关键路径」挂在 DeepSeekMoE 图之后，但「同量级」出自 DeepSeek-V3 报告第 302 行「computation-to-communication ratio of approximately 1:1」，建议就近引用。
- **P2-24**（参考 · 源码 L1271）SGLang 文档只记「本地 clone 的快照日期」，建议与全文 pin 纪律一致，记 commit（本地 HEAD `7399c2b5587e1559f3e5a26566ed322e81e1433a`）。
- **P2-25**（§4.3 L390）「H2 用大约两倍的激活换来了零气泡，这也解释了『H2』这个名字里的 2」：论文只说 ZB-H1 是「Our first handcrafted schedule」（`zero-bubble.md` 第 46 行），Figure 3 图注为「Handcrafted pipeline schedules, top: ZB-H1; bottom: ZB-H2」（第 42 行），并没有解释编号含义；H1 / H2 与 1× / 2× 内存预算恰好对应，也与 §5 的 ZB-1p / ZB-2p 命名平行，但这只是推测，建议写成「编号或许对应…」或删去。
- **P2-26**（§1.2 L48–L56）style-guide §4.5 要求模型介绍先交代名称、来源、参数量级与链接；文章对 DeepSeekMoE / DeepSeek-V3 只给了 arXiv 与图，没有给出模型规模，也没有给出「计算与通信同量级」的出处（见 P2-23），可补一句全貌。

## 第 2 轮 7 项修复的独立核验

| # | 声称的修复 | 我的核验方式 | 结论 |
| --- | --- | --- | --- |
| 1 | ZB-V 归属改为 Zero Bubble §6 / Figure 8，并说明 Controllable Memory 更晚 | 读 `zero-bubble.md` 第 156 行节标题「6 MEMORY EFFICIENT ZERO BUBBLE SCHEDULE」、第 158–159 行 Figure 8 与图注、第 161 行「we design ZB-V」与 16 层 / 4-stage 例子、第 163 行「both the forward pass and backward pass for each microbatch originate from the same worker」、第 165 行 warm-up `2p-1` 次；`grep -c ZB-V controllable-memory.md` = 0，V-Min / V-Half / V-ZB 出现在第 82、107 行；arXiv 2401.10241（2024-01）早于 2405.15362（2024-05）；对照正文 §6.1 L557、L567、L573 | **已修好**。残留：「半年之后，同一批作者」不精确（P2-10）；`learn-plan.md` 94、154 与 `references/README.md` 44 仍写 CM「§6」「含 ZB-V」（P1-2）；正文参考条目也写了不存在的「§6」（P1-6） |
| 2 | terapipe 图换成真正的 1(d) | sha256 全量反查：`pics/terapipe-fig1d-token-pipeline.jpg` 与 `references/papers/terapipe/images/a73bba64….jpg` 字节相同，该文件在 `terapipe.md` 第 28 行被引用，第 29 行子图注为「(d) Token-based pipeline parallelism (TeraPipe)」；`read_image`：Device 1–5 自下而上，每台一条 Transformer layer 条带，条带下方是细分的 token 段，橙色细箭头跨设备传递 | **已修好**。残留：`IMAGE-VERIFY.md` 第 24 行仍写从 `b78a683e…` 复制（P1-2）；同一图注里对 TeraPipe Figure 3 的描述另有错误（P0-8） |
| 3 | Figure 3 下栏 warm-up F 数 7/6/5/1 → 7/5/3/1 | 自写像素分类（x0=72、pitch 24.5；每格取内侧左上角 3–5 px 与格内下部背景点的中位数，避开居中的数字字形）：下栏四行 `FFFFFFFBWFB…`、`.FFFFFBFBFB…`、`..FFFBFBFBF…`、`...FBFBFBF…`，前导 F 为 **7/5/3/1**；上栏 4/3/2/1、warm-up 与首个 B 之间空格 3/2/1/0。「7=4+3、5=3+2、3=2+1、1=1+0」成立，而且是一般恒等式：worker i 的 H2 前导前向数 `2p-2i+1` = H1 前导前向数 `p-i+1` + 空格数 `p-i`，与论文第 50 行「we introduce more F passes during the warm-up phase to fill the bubble preceding the initial B」及第 62 行 H2 激活公式一致 | **已修好**（数字与自证关系都对）。但同一 alt 与 §4.1 对上栏白格的解释是错的（P0-3）；核对文件里的 7/6/5/1 未订正（争议项 1） |
| 4 | Figure 8 alt 改为单行网格 + 白 / 黑字区分两个 chunk | 论文第 159 行图注原文「Each device is assigned to exactly 2 chunks, where white text colors represent the first chunk and black text colors represent the second chunk」；像素统计每格数字字形亮度（与格背景色差 > 90 的像素，均值 > 170 记白、< 100 记黑）：Device 1–4 分别为白 31 / 黑 40、白 33 / 黑 36、白 32 / 黑 37、白 32 / 黑 34，每行都同时有白字与黑字；水平网格线 y = 9/32/56/80/103，共 4 行，每个 Device 一行 | **已修好** |
| 5 | 「双向各 20」→「双向合计 20（每方向 10）」 | 官方 `README.md` 第 9 行「8 PP ranks and 20 micro-batches in two directions」、第 10–11 行反向对称省略编号；`examples/example_dualpipe.py` 第 118 行 `num_chunks = 20`、第 146/149 行 `full_x.chunk(2)`；`dualpipe.py:336` `half_num_chunks = num_chunks // 2`，第 345 行按 `half_num_chunks` scatter；读图 `dualpipe.png` 与 V3 Figure 5 格内编号 0–9；逐列取色每台设备 F 单位 20（两个方向各 10） | 正文 L495、L498、L501、L506、L618 **已修好**；**修得不彻底**：`IMAGE-VERIFY.md` 第 39、68 行与 `pics/README.md` 第 66 行仍写「双向各 20」（P1-2） |
| 6 | 删去「官方 README 与 Sea AI Lab 博客都提到」PP 通信翻倍 | 通读官方 `README.md` 全文 69 行：涉及通信的只有第 3 行「full overlap of forward and backward computation-communication phases」与第 11–12 行「mutually overlapped computation and communication」，没有 PP 通信量或 PP / EP 开销比较；Sea AI Lab 本地副本第 36 行「所有方案均基于相同数量的设备」、第 40–45 行表头「PP 通信」列（Cut-in-half 为 2x）、第 49 行「对半裁剪方案的 PP 通信量是其他方法的两倍…相较于 EP 通信，PP 通信的开销较小」 | 正文 §6.3 L643、§7.4 L797 **已修好**；**修得不彻底**：`notes/MATH-VERIFY.md` 第 101 行仍把来源写成「官方 README 与 Sea AI Lab 博客」（P1-2） |
| 7 | 八处「本地副本第 N 行引用图」行号改正 | 全量脚本：对正文 32 个 `src="./pics/…"` 逐个取紧随其后的「图片来源」块，解析「本地副本：`<md>` 第 N 行」，用 sha256 在 `references/` 下反查同字节文件，再在该 md 中找到引用该文件名的真实行号 | **已修好，100% 通过**：30 处本地副本行号全部命中；其余 2 张 `dualpipe.png`、`dualpipev.png` 没有本地副本声明，sha256 与 `/workspace/algorithm/DualPipe/images/` 同名文件相同；`comm` 双向比对 32/32，无缺失、无多余；`pics/README.md` 32 行的体积与 sha256 前 16 位全部一致 |

## 争议项裁决

### 1. `notes/IMAGE-SURVEY-zero-bubble.md` 的 `7/6/5/1` 未订正，是否可接受

**不可接受，建议就地加注订正。** 理由有三。其一，该文件是第 2 轮 P0-3 的直接成因，而文章附录 B 把 `notes/` 作为逐图复核索引对外引用，读者或后续轮次按索引核对时会先读到错误值。其二，错误不止一处：第 54 行（表格）、第 80 行（建议 alt）、第 87 行（「引用时需要注意…务必按读图结果写」）三处都是 7/6/5/1，第 87 行恰恰是一句「照抄指令」。其三，只在 `IMAGE-VERIFY.md` 里写「以本条为准」属于隐式勘误，要求读者两份文件都读到。建议三处改为 7/5/3/1，并在第 54 行末追加「〔第 3 轮像素复核订正，原记 7/6/5/1，见 IMAGE-VERIFY.md 与 REVIEW-deep-r3b.md〕」；`IMAGE-SURVEY-aiinfra-docs.md` 第 30 行（P0-4）同样处理。流程上建议把「记录文件的勘误」明确列入每轮允许的改动范围，否则「修正文、留旧账」会继续发生，本轮 P1-2 明细里的 20 余条就是这样积累出来的。

### 2. `../deepep/deep-dive.md` 的引用合规性

**正文加注是诚实的，但不满足 `.learn/config.md`「引用 published 状态的文章」这条字面规则。** 核实结果：`torch/deepep/deep-dive.md` 存在（102715 字节），`README.md`、`README-cn.md` 里 `grep -i deepep` 均无结果，知识图谱也没有该条目。我的判断是发布时二选一：若 DeepEP 与本文同批或先发布，保留链接并删去括号说明；若本文先发布，改为不带链接的纯文字提及，并删掉 L7 对其内容「已经讲到拓扑、显存布局与 SM 占用这一层」的断言，因为未发布文章的结论不应作为背景事实被引用。另需修正文末注释 L1369 的自相矛盾，以及 `learn-plan.md` 第 23、143、339 行「已发布」的说法。

### 3. `rlhf/sys-design/readme-4.md` 的状态矛盾

**文章不引用它的做法可以接受，但给出的理由与对矛盾的描述都不准确；矛盾应由 README 维护者（仓库作者）修，且需先征得同意。** 核实结果：`README.md` 第 51 行把 `readme-4-en.md` 列为已发布，并注明「Also available in Chinese version（`./rlhf/sys-design/readme-4.md`）and zhihu」；`README-cn.md` 第 64 行把 `readme-4.md` 列为已发布；只有 `README.md` 第 150 行一条标题为「Expert Parallelism」的条目带 `[Pending Review]`；知识图谱中该文 `status: published`。所以矛盾实际上是 **`README.md` 内部第 51 行与第 150 行两条目冲突**，知识图谱与已发布的两条一致，第 150 行更像一条遗留占位。按 AGENTS.md「文章是否已正式发布，先查 README.md 和 README-cn.md」，三个条目里有两个是已发布；保守起见本文暂不引用不算错，但文末注释应改为「README.md 第 150 行的遗留 [Pending Review] 条目与第 51 行、README-cn 第 64 行冲突，发布状态待作者确认，本文暂不引用」，并删掉「改由 torch/deepep 承担」。清理第 150 行属于改变 README 发布状态，按 AGENTS.md 属 Ask First，应由仓库作者决定。

### 4. 审查报告与被审对象的版本漂移：建议约定

本轮观察到三种漂移：第 2 轮报告以 md5 `9174861c…`（1368 行）为基准，之后正文被改到 1372 行；`REVIEW.md`、`REVISION.md` 写完后正文又改过一次（11:52:04）且无记录；审查期间另一个会话在同一 `notes/` 目录写入报告、在 `codes/` 生成 `__pycache__`。`torch/dualpipe/` 整体 untracked，git 无法兜底。建议后续轮次沿用以下约定：

1. **报告头写被审对象的四元组**：路径、`wc -l`、md5、mtime，外加是否被 git 跟踪；报告写完前再算一次 md5，不一致就注明「审查期间被修改」并重跑自动核对。
2. **每条问题用「小节号 + 原文锚句」定位**，锚句取 10–30 字的逐字引文，行号只作线索；复核时用 `grep -nF '<锚句>'` 重新定位，找不到锚句即视为该条已被改动、需要重新判断。
3. **修复记录（REVISION）逐轮记录修改前后的 md5，并附传播清单**：每条修复要列出正文之外还有哪些文件写着同一事实（`notes/`、`pics/README.md`、`codes/README.md`、`learn-plan.md`），逐一勾掉。
4. **审查期间冻结被审对象**：开始审查前由用户做一次快照（commit，或把 md5 写入报告），审查结束前不改正文；多会话并行审查时约定报告文件名带会话后缀、只写自己的文件，运行 `codes/` 下脚本统一带 `PYTHONDONTWRITEBYTECODE=1`。
5. **只读核对脚本入库**：本轮与第 2 轮都各自手写了「sha256 反查行号」「像素分类」脚本，建议经用户授权后放进 `notes/checks/`，下一轮直接复跑，避免每轮重新踩取样位置的坑。

### 5. 「非 zb 反向统一记成 I → W 只影响 rank 内松弛量」是否正确

**数值结论成立，理由表述不严谨，而且这不是模拟器最大的偏差。** 我把模拟器里非 zb 反向的顺序改成 W → I（与示例 hook `example_dualpipe.py:26-33` 先 `grad_weight_fn()` 后 `grad_output @ weight` 一致），在 DualPipe `(p, m)` = (4,8)、(4,20)、(6,12)、(8,16)、(8,20)、(8,32) 与 DualPipeV (2,4)、(4,8)、(4,10)、(8,16)、(8,20) 共 11 组配置下重算，气泡**全部不变**（DualPipe 1、1、2、3、3、3；DualPipeV 1、3、3、7、7）。但「不影响跨 rank 依赖」的说法不准：依赖边的结构不变，边被满足的时刻（I 的完成时刻）后移 1 个单位，只是没有落在关键路径上，应写成「依赖边不变、满足时刻后移 1 个单位，经扫掠验证不改变关键路径」。更重要的是，文章为这处小取舍专门写了一段说明，却没有提到会让气泡减半的「配对块内 F 输出提前释放」这一建模假设（P0-2）。

## 对抗性证伪尝试

1. **「双向布局提供了天然独立、可配对的前向与反向任务」**：配对本身**未能推翻**。上半区 rank r 第 4 步里，`F_0` 处理方向 0 的 microbatch x，输入激活来自 rank r−1；`B_1` 处理方向 1 的 microbatch x−p/2，输出梯度同样来自 rank r−1（方向 1 的前向从 rank p−1 流向 0，反向从 0 流向 p−1）；两个 microbatch 分属两份不相交的数据（`example_dualpipe.py:145-150` 的 `full_x.chunk(2)[0]` 与 `[1]`），在同一个 `batch_isend_irecv` 里接收，计算上互不依赖。**「独立性来自双向布局」这一归因已推翻**：1F1B 稳态 `(F_{k+p-r-1}, B_k)` 在 r<p−1 时就是不同 microbatch；Megatron `schedules.py:939-943` 用单向 interleaved 1F1B 多做一个 warm-up 前向，就在所有 rank 上得到独立配对（P1-1）；附录 A「双向布局是拆分反向成立的前提」也被 §4 与 §6 推翻（P0-6）。
2. **「官方式在 `F&B = max(F, B_full)` 下等于 `(p-2)t/2`，并与模拟器实测一致」**：代数**未能推翻**（`(p/2-1)(2t+2t-3t) = (p-2)t/2`，`--sweep` 32 组全部命中）；但作为「交叉验证」**已推翻**：按官方提交语义重放，C=3 得 `(p-2)t`（p=4/6/8 → 2/4/6），C=2 时各 rank busy 不等、只有最大空闲等于 `(p-2)/2`；官方 `dualpipe.png` 按 `F&B = F + B` 绘制，逐列取色每台设备空闲 6 = `(p-2)t`。模拟器与 `max` 端一致，靠的是「配对块内 F 输出提前释放」的假设（P0-1、P0-2）。
3. **气泡率分母 `3m` / `6m` 的区分**：在 `max` 口径内算术成立，但第 2 轮 P2-3 修出来的新解释「DualPipe 与 DualPipeV 的气泡恰好是 ZB1P 的一半」对 DualPipe 不成立（`(p-2)/2` 对 `(p-1)`，p=8 时 3 对 7）；在与官方图一致的 `F+B` 口径下，三者分母首项都是 `3m`（DualPipe `(p-2)/(3m+p-2)`，DualPipeV `(p-1)/(3m+p-1)`），「3m 与 6m 是最容易出错的地方」这条论述**已推翻**（P0-1 ⑤）。第 2 轮 P2-3 指出的「解释牵强」没有真正修好，而是换成了一个错误的解释。
4. **「官方骨架不含细粒度重叠，`_weight_chunk()` 里的 `_commit_and_wait_comm()` 是硬同步点」**：**未能推翻**。本地 `torch.distributed.distributed_c10d.Work.wait.__doc__`：「calling wait() is the same as calling synchronize(): Letting the current stream block on the completion of the NCCL work」，所以在当前 stream 上排队的计算不会与这一批 P2P 重叠；PyTorch NCCL 进程组在发起通信前让通信 stream 等待当前 stream 上已排队的计算，这一点本地没有 C++ 源码可查，记 `[UNVERIFIED]`。需要补的限定只有两条：CPU 不阻塞（stream 级同步）；结论只针对骨架自身的 P2P，使用方在 `overlapped_forward_backward` 里用其他 stream 发起的 All-to-All 不在此列（P2-12）。另外，V3 Figure 4 被隐藏的通信里包括 PP，而骨架里 PP 永远在两次计算之间提交并等待，这反而加强了「骨架不含细粒度重叠」的结论。
5. **「固定设备数下 DualPipeV 每设备参数份额 1×、DualPipe 2×」**：**未能推翻**。Sea AI Lab 本地副本第 34 行「设计一个与 DualPipe 设备数量一致的对半裁剪调度方案…每个设备的参数量降至原来的 50%」，第 42–45 行（同设备数）DualPipe 2×、Cut-in-half 1×。隐含前提是各 stage 参数均匀、不计 embedding 与输出头；DeepSeek-V3 报告第 330 行说 DualPipe 把最浅层（含 embedding）与最深层（含输出头）放在同一 PP rank，V 形布局下 rank 0 同样同时持有二者，倍数关系不变，但建议在 §6.3 表下注明「均匀划分、不计 embedding / head」。
6. **「ZB-H1 峰值激活等于 `pM_B`，与 1F1B 相同」**：**未能推翻**。`zero-bubble.md` 第 70 行 Table 2 两行都是 `pM_B`；第 62 行 worker i 激活 `(p-i+1)M_B+(i-1)M_W` 在 `M_W < M_B`（第 60 行 Table 1：`32sbh < sb(34h+5as)`）下于 i=1 取峰值；第 46 行「the first worker has the maximum peak memory usage which is consistent with 1F1B」。
7. **「1F1B 一次迭代时长 `(m+p-1)(F+B_full)`」的适用前提**：**未能推翻公式**，文章也确实写了「在 m ≤ p 的粗略分析里」（L318）；但所引附录第 348 行的前提是「ignoring communication and assuming m <= p and T_W < T_B」，文章随后在 m ≥ p（§3.3、§7.5 取 p=8、m=20）下使用。该式对非交错 1F1B 在 m > p 时同样成立（Table 2 第 70 行的气泡 `(p-1)(T_F+T_B+T_W)` 不带此前提，再加 `m(F+B_full)` 的有效工作即得），建议改引 Table 2（P2-11）。

## 逐项核对记录

### 1. 数学逐式复核（公式 → 来源 → 我的复算 → 结论）

| 公式或断言（文章位置） | 来源（实际读到的位置） | 我的复算 | 结论 |
| --- | --- | --- | --- |
| `Y = XΘᵀ`，`I = GΘ`，`W = GᵀX`，例子 `X[4,8]`、`Θ[16,8]`、`G[4,16]`（§2.2 L124–L136） | 矩阵微分；Zero Bubble Figure 1（`zero-bubble.md` 第 23–24 行，读图可见 `Wᵀ∇_zL` 与 `∇_zLxᵀ`） | `[4,16]×[16,8]=[4,8]`，`[16,4]×[4,8]=[16,8]` | 一致 |
| 「I 优先、W 可延迟是调度策略，不是数学必然」（§2.3 L174、§4.4 L404） | 论文第 46 行「adjusts the starting points of W」；示例 hook 先算 W 闭包再算 dgrad（`example_dualpipe.py:26-33`） | 单个线性算子两式互不依赖；整段 stage 的 dX 在最后一个算子后才产出 | 一致，表述收紧得当 |
| Table 1 与 `T_W < T_F < T_B`、`T_B + T_W = 2T_F`（§2.2 L156–L162） | 第 56、60 行 | `(24h+8s)+24h = 2(24h+4s)` | 一致 |
| Table 9 「相差约 26%」「其余配置 14%–17%」（L162） | 第 302 行 | 12 行逐行计算见 P1-4 | 首行一致；「其余」范围错误（P1-4） |
| `n_warmup(r) = min(p-r-1, m)`（§3.2 L243） | Megatron `core_v0.19.0` `schedules.py:930` 与 949–951 | p=4 时 3/2/1/0 | 一致 |
| 稳态 `(F_{k+p-r-1}, B_k)`（L266） | 由 warm-up 计数推出 | p=4 表逐行成立 | 一致 |
| 峰值激活 `min(m,p)A`、`pA = A_all`（§3.3） | Table 2 第 70 行 `pM_B` | 前提（均匀、无重计算、忽略 buffer）已写全 | 一致 |
| `T_1F1B = (m+p-1)(F+B_full)`、`T_bubble = (p-1)(F+B_full)`、`β = (p-1)/(m+p-1)`（§3.4） | 附录第 348 行（前提 m ≤ p、`T_W < T_B`）；Table 2 第 70 行 | 相除成立 | 一致（引用前提见 P2-11） |
| ZB-H1 三元组 `(F_x, I_{x-(p-1-r)}, W_{x-(p-1)})` 与区间推导（§4.1 L333–L342） | 论文未给此式；与 Figure 3 上栏逐格分类比对 | r=0：`F0…F3`、空 3、`B0 W0 F4 B1 W1 F5…`，即 `(F3,I0,W0)`、`(F4,I1,W1)`；r=1：`(F2,I0)` 后接 `(F3,I1,W0)`；r=3：`(F0,I0)…(F3,I3,W0)` | 与图逐格一致（本轮新增证据） |
| ZB-H1 / H2 气泡与等时代入（§4.2 L350–L366） | Table 2 第 70 行；第 46 行「reduced to a third of 1F1B's size」、第 50 行 parallelogram | `3(p-1)t → (p-1)t → 0`；Figure 3 上栏每台设备空格合计 3 = `(p-1)t`（p=4） | 一致 |
| worker i 激活与峰值 `pM_B`、`(2p-1)M_B`（§4.3） | 第 62 行 | `M_W < M_B` 下两式随 i 递减 | 一致 |
| README 的 ZB1P 行等于 ZB-H1（§7.1 L713） | README 第 29 行 | 代入 `B = I + W` | 一致 |
| DualPipe 四元组（§5.4 L478–L487） | `dualpipe.py:358-396` 循环次数 | p=8、r=1 进入第 4 步时 F0=6、B1=2、F1=4、B0=0，x=6，与 `--p 8 --m 20` 的 rank 1 输出一致；x=8 由主循环每步四个计数器同步 +1 得 `(8,4,6,2)` | 一致（复现声明见 P0-9） |
| 「方向 1 的反向进度（下标 `x-p+1+r`）总是不晚于方向 0 的反向进度（下标 `x-p/2`）」（§5.4 L491，第 2 轮 P1-3 修复时新写入） | 同一节 L478–L487 的四元组 | 四元组里 `B_1` 的下标是 `x-p/2`、`B_0` 的下标是 `x-p+1+r`，这句把两个下标挂反了；以 p=8、r=1、x=6 代入，方向 1 为 2、方向 0 为 0，原句读作「0 不晚于 2」不成立 | **错误**（P1-8） |
| DualPipeV 四元组（§6.4 L660–L679） | `dualpipev.py:330-367` | p=4、r=1 时 F0=6、F1=4、B1=2、B0=0，与 `--p 4 --m 10` 输出一致；代入得 2/4/0 | 一致 |
| 八步循环次数（§6.4 表、§8.5 Mermaid） | `dualpipe.py:358-425`、`dualpipev.py:330-396` | 逐项相同 | 一致 |
| 「第 6 段恰好产生 r+1 组待完成的 W」（§6.5 L692） | `dualpipe.py:404-425`，断言在 L425 | 第 6 段共 `2(r+1)` 次反向，按奇偶切换点其中 `r+1` 次为 zb；第 3、7 段入队出队相抵；第 8 段出队 `r+1`；模拟器逐设备断言队列清空；本轮重放中 zb 次数等于 `_weight_chunk` 次数 | 一致 |
| `max(F,B) ≤ F&B ≤ F+B` 及两端解释（§2.1 L118、§7.2 L721–L747） | README 第 33–36 行；V3 报告第 304 行与 Figure 4 | 区间本身成立；端点语义与「优势完全建立在重叠上」不成立 | **修正**（P0-1） |
| 两端代入：DualPipe `(p-2)t/2`、`(p-2)t`，DualPipeV `(p-1)t/2`、`(p-1)t`（§7.2 L727–L745） | README 式 | 代数成立；按提交语义重放与官方图对应 `F+B` 端 | 代数一致，口径解释错（P0-1） |
| 每设备有效工作 `3mt`（§7.3、§7.4 L791） | 模拟器 busy 30 / 60 | 在 `F&B = F+B` 的会计下成立；C=2 重放时各 rank busy 不再相等 | 有条件成立（P0-1） |
| 气泡率 `(p-2)/(6m+p-2)`、`(p-1)/(6m+p-1)`（§7.3 表） | `--sweep` 32 组 | `max` 端成立；`F+B` 端为 `(p-2)/(3m+p-2)`、`(p-1)/(3m+p-1)` | 见 P0-1 |
| 「DualPipe 与 DualPipeV 的气泡恰好是 ZB1P 的一半」（L774） | — | DualPipe `(p-2)/2` 对 ZB1P `(p-1)`，p=8 时 3 对 7 | **错误**（P0-1 ⑤） |
| 参数份额的两种基准（§6.3 表） | README 第 26–31 行；Sea AI Lab 第 34、42–45 行 | 固定 stage 数时均为 2×；固定设备数时 2× 对 1× | 一致 |
| `(2p+1)·A/2 = (p+1/2)A`（§7.4 L795） | README `PP+1`；Sea AI Lab 第 45 行 `d+1/2` | 成立，前提已写明 | 一致 |
| `2×2048×4096×2 B = 32 MiB`（§8.6 L1006–L1008） | — | 33,554,432 B | 一致 |
| §7.5 表 | — | 21、7/27≈25.9%、7、7/67≈10.4%、3、6/126≈4.8%、6/66≈9.1%、3.5、7/127≈5.5%、7/67≈10.4% 算术全部正确 | 算术一致，主列口径见 P0-1 |
| `g_ℓ = (1/m)(ΣA + ΣB)`（§5.3 L455） | `example_dualpipe.py:86-100`、`168-176` | 与示例的逐 microbatch 求和相差归一化 | 见 P2-17 |

### 2. 可复现性（命令 → 实际输出 → 与文章数字是否一致）

#### 2.1 四条命令实跑（均带 `PYTHONDONTWRITEBYTECODE=1`，退出码全部为 0）

| 命令 | 实际输出要点 | 与文章是否一致 |
| --- | --- | --- |
| `python torch/dualpipe/codes/01_split_backward_minimal.py` | `torch 2.10.0+cu129`、`torch.float64`、`(8, 8) / (16, 8)`、microbatch 2、`backward 返回后 dTheta is None : True`、dX 最大误差 `0.000e+00`、dTheta 最大误差 `1.110e-16`、`PASS` | 一致（L205–L212、L1342；`codes/README.md` L32–L39） |
| `python torch/dualpipe/codes/02_schedule_sim.py --p 4 --m 10` | DualPipe busy 30、关键路径 31、气泡 1、3.2258%；DualPipeV 60、63、3、4.7619%，换算为 30、31.5、1.5；闭式复核四项「一致」；DualPipe rank 1 首组 `F0_2, B1_0, F1_2, B0_0`，DualPipeV rank 1 首组 `F0_6, B1_2, F1_4, B0_0` | 一致（L759–L766、L660–L666） |
| `python torch/dualpipe/codes/02_schedule_sim.py --p 8 --m 20` | DualPipe 60 / 63 / 3 / 4.7619%；DualPipeV 120 / 127 / 7 / 5.5118%（换算 60 / 63.5 / 3.5）；DualPipe rank 1 首组 `F0_6, B1_2, F1_4, B0_0` | 一致；任务说明列出的数字全部对上 |
| `python torch/dualpipe/codes/02_schedule_sim.py --sweep` | 「一致」96 次（32 组 × 3 项）、「不一致」0 次，汇总表 32 行 | 一致。文章写的「p∈{2,4,6,8}、m∈{8,10,16,20,32}」在跳过 m<2p 后实为 32 组；第 2 轮报告写「30 组」是记录笔误 |

`codes/README.md` 的「已验证」三条在其自身模型内成立，但第三条漏写了「配对块内 F 输出提前释放」这一假设；「未验证」一节对 GPU、通信、等时假设与缺失的张量级执行器的声明是诚实的，唯独 L87 把模拟结果称为「串行执行 + 依赖等待下的气泡、重叠收益的上界参考」不成立（P0-1、P0-2）。

#### 2.2 模拟器与官方 `step()` 的逐点对照

| 检查点 | 官方源码 | 模拟器 | 结论 |
| --- | --- | --- | --- |
| 八步循环次数 | `dualpipe.py:358-425`、`dualpipev.py:330-396` | `02_schedule_sim.py:108-154`、`169-215` | 逐步一致 |
| step 3 / step 7 的 `enable_zb=True` | `dualpipe.py:376、419`；`dualpipev.py:347、390` | L117、L151；L178、L212 | 一致 |
| step 6 的奇偶切换点 | `dualpipe.py:404-413`；`dualpipev.py:375-384` | L139–L147；L200–L208 | 逐字同构 |
| `_weight_chunk()` 与 flush 的先后、`forward_only` 提前返回 | `dualpipe.py:216-223` 先提交通信再 `pop()`，推理模式提前返回；`_backward_compute_chunk` 在 zb 时于 L113–L114 flush | `_backward` 在 zb 时立即 flush（L74–L77），`_weight_chunk` 按 FIFO `popleft`（L86–L90）；训练模式 `forward_only=False`，提前返回分支不触发，模拟器不建模推理模式 | 训练模式下一致 |
| 跨 rank 反向依赖只挂 I、不挂 W | 只有输入梯度跨 rank 发送（`_send_backward` 发的是 `input_grad_chunks`） | L267–L287 只用 `b.inp` | 一致 |
| V 底部本地交接 | `dualpipev.py:79-80、115-118、171-172、182-185` | L256–L258、L273–L276 | 一致 |
| 中间 rank 第 4 步 i=0 特判 | `dualpipe.py:384-393` | 未单独建模（任务次序同为 F0→B1→F1→B0） | 在模拟器语义下无影响；在提交语义下它让中间 rank 的这一块不重叠，C<3 时各 rank busy 不再相等 |
| **跨 rank 发送的释放时刻** | `dualpipe.py:205-214`：配对块两段计算都结束后才追加发送，下一次 `_commit_and_wait_comm()` 才提交；`_forward_chunk`、`_backward_chunk` 同理（L185–L203） | 下游直接依赖上游 F / I 任务的结束时刻（L244–L287） | **不一致**：配对块里的 F 输出乐观 2 个单位，非 zb 反向的 I 输出乐观 1 个单位（反例见 P0-2） |

W 队列会计本身没有找到反例：第 3、7 段逐次入队出队，第 6 段 zb 次数 = 第 8 段出队次数，脚本与重放两边都满足。

#### 2.3 按官方提交语义重放 `step()`（本轮独立模型）

建模方式：把 `dualpipe.py:294-440` 与 `dualpipev.py:288-411` 逐调用转写为每个 rank 的操作序列（追加接收、追加发送、提交、计算）；计算耗时 F=1、非 zb 反向 2、zb 反向 1、`_weight_chunk` 的 W 为 1、配对块为 C（示例 hook 顺序执行时 C=3）；NCCL 按 (src, dst) FIFO 匹配，并逐条核对消息的种类、方向与序号；一次提交的完成时刻 = max(本方发起时刻, 所有配对消息所在批次的发起时刻)，迭代到不动点；通信耗时记 0。

| 方法 | p | m | C=3：busy / makespan / 气泡 | C=2：busy / makespan / 最大空闲 | 模拟器气泡 |
| --- | ---: | ---: | --- | --- | ---: |
| DualPipe | 4 | 8 | 24 / 26 / 2 | 21 / 22 / 1 | 1 |
| DualPipe | 4 | 20 | 60 / 62 / 2 | 45 / 46 / 1 | 1 |
| DualPipe | 6 | 12 | 36 / 40 / 4 | 31–32 / 33 / 2 | 2 |
| DualPipe | 8 | 16 | 48 / 54 / 6 | 41–43 / 44 / 3 | 3 |
| DualPipe | 8 | 20 | 60 / 66 / 6 | 49–51 / 52 / 3 | 3 |
| DualPipe | 8 | 32 | 96 / 102 / 6 | 73–75 / 76 / 3 | 3 |
| DualPipeV | 2 | 4 | 24 / 26 / 2 | 21 / 22 / 1 | 1 |
| DualPipeV | 4 | 8 | 48 / 54 / 6 | 41–43 / 44 / 3 | 3 |
| DualPipeV | 4 | 10 | 60 / 66 / 6 | 49–51 / 52 / 3 | 3 |
| DualPipeV | 8 | 16 | 96 / 110 / 14 | 81–87 / 88 / 7 | 7 |
| DualPipeV | 8 | 20 | 120 / 134 / 14 | 97–103 / 104 / 7 | 7 |

所有配置的消息种类 / 方向 / 序号错配为 0、收发条数不等为 0，2–8 次迭代收敛。结论：C=3 的气泡与 m 无关、在各 rank 上相等，恰为 README 式取 `F&B = F + B` 的值，也与官方 `dualpipe.png`（p=8, m=20）、`dualpipev.png`（p=4, m=10）逐列取色得到的 makespan 66、空闲 6 完全一致；模拟器的 `(p-2)/2` 只与 C=2 时的最大空闲相等。局限：通信耗时记 0；配对块内部当作不可分的整体；按「批次发起即可配对」近似 NCCL 分组语义，没有建模其内部调度。

#### 2.4 旧声明与「实测」措辞

正文开篇 L11 与文末 L1349 各声明一次「没有任何 GPU 实测」。「实测」出现在 L11、118、162、205、759、774、814、820、822、1285、1342、1349、1350、1358：L205、L1285、L1342 明确限定为 CPU 脚本；L759、L774 把模拟输出称为实测（P2-20）；L814、L820 把 Megatron-LM Figure 6 称为实测，不成立（P0-5）；L822 的 Controllable Memory Figure 6 确为实验数据。

### 3. 源码行号核对（引用 → revision → 实际位置 → 一致？）

| 引用（文章位置） | revision | 实际位置与内容 | 一致？ |
| --- | --- | --- | --- |
| 六个文件的行数 440 / 411 / 80 / 38 / 202 / 183（§8.1 表） | `030ce432` | `wc -l` 相同；另有 `dualpipe/__init__.py` 17 行 | 一致 |
| `step()` 起点与模拟器注释引用的范围 | 同上 | `dualpipe.py:294`、`dualpipev.py:288` | 一致 |
| 入口断言（§6.5 L694） | 同上 | `dualpipe.py:332-333`；`dualpipev.py:318` 无偶数要求 | 一致 |
| `half_rank = min(rank, num_ranks - 1 - rank)`（L864） | 同上 | `dualpipe.py:335` | 一致 |
| `is_in_second_half`、`is_middle_rank`（§8.2、§8.5 L986） | 同上 | `dualpipe.py:44-45` | 一致 |
| phase 语义注释原文（L853） | 同上 | `dualpipe.py:355-356` 逐字 | 一致 |
| `phase ^= self.is_in_second_half`（L856） | 同上 | `dualpipe.py:68、91、132、147、232、242、260、273` | 一致 |
| `_recv_forward` / `_send_forward` 的判断（L859） | 同上 | `dualpipe.py:233`、`243` | 一致 |
| 六个计数器（§8.3） | 同上 | `dualpipe.py:58-63` | 一致 |
| `WeightGradStore`（§8.4：26 行，`put/flush/pop` 15 行） | 同上 | `utils.py:8-33` 共 26 行，三个方法在 14–28 行共 15 行；摘录逐行一致 | 一致 |
| `_weight_chunk()` 摘录（L920–L928） | 同上 | `dualpipe.py:216-223` 逐行一致 | 一致 |
| `_commit_and_wait_comm()` 的语义（L930） | 同上 | `dualpipe.py:285-292` | 一致 |
| step 3 循环摘录（L962–L967） | 同上 | `dualpipe.py:375-379` 逐行一致 | 一致 |
| step 4 中间 rank 特判与 NOTE 注释（L974–L984、§5.6 L529） | 同上 | `dualpipe.py:384-393` 逐行一致，注释原文逐字 | 一致 |
| `_forward_backward_chunk` 摘录（L1021–L1031） | 同上 | `dualpipe.py:205-214` 逐行一致 | 一致 |
| `comm.py` 摘录（L993–L1000） | 同上 | `comm.py:7-8、21-22`，省略的两个 setter 已用注释标出 | 一致 |
| 重叠钩子为类方法、docstring、调用方式（§8.7 L1016） | 同上 | `example_dualpipe.py:54-83`，docstring 第 66–69 行逐字；调用在 `dualpipe.py:168` | 一致 |
| README Quick Start 下的提醒（L1043） | 同上 | 官方 `README.md` 第 47 行逐字 | 一致 |
| DualPipeV 本地交接（§6.2 L610） | 同上 | `dualpipev.py:79-80`、`115-118` | 一致 |
| `full_modules[rank]`、`[pp_size - 1 - rank]`、`[pp_size * 2 - 1 - rank]`（L446、L597） | 同上 | `example_dualpipe.py:138`；`example_dualpipev.py:127、137` | 一致 |
| P2P 形状 `(3, 256, 512)`、float32（L1002） | 同上 | `example_dualpipe.py:118-125` | 一致 |
| 示例的梯度合并方式（§5.3 L458） | 同上 | `example_dualpipe.py:168-176` | 一致 |
| `3da1bbea…`（2025-03-06）与 HEAD 的差异只有 `__init__.py` 10 行（L849） | git | `3da1bbe` 提交于 2025-03-06 09:41 +0700；`git diff --stat` 为 `dualpipe/__init__.py` 1 个文件、5 增 5 删；HEAD `030ce43` 提交于 2025-12-26，与 `origin/main` 相同 | 一致 |
| 官方 Python 源码不含非 ASCII 字符（草稿里的「中文注释」已删） | 同上 | `grep -nP '[^\x00-\x7F]' --include=*.py -r .` 无输出 | 一致 |
| Megatron `schedules.py:926-950`（§3.2 L246） | `core_v0.19.0` | 第 930 行取 `pipeline_parallel_size - pipeline_parallel_rank - 1`，949–951 行截断 | 一致 |
| `combined_1f1b_schedule_for_interleaved_pipelining` 138、`combined_forward_backward_step` 281、docstring 164（§9.2 L1119） | 同上 | 三处命中；调用点在 `schedules.py:1458-1459`，位于 L992 起的 `forward_backward_pipelining_with_interleaving` 内 | 一致 |
| `checkpoint_activations_microbatch` 与 overlap 不兼容（§9.3） | 同上 | `combined_1f1b.py:344-345` | 一致 |
| `TransformerLayerSchedulePlan` 30、节点树 docstring 37–43、`_build_callable_nodes` 112、`delay_wgrad_compute` 133 及 stream 分配（§9.2 表） | 同上 | 全部命中；`create_node` 在 160–161（comp）、163（comp）、165–166（comm）、172–173（comp） | 一致 |
| 执行链 Mermaid（§9.2 L1144–L1153） | 同上 | `GPTModel.build_schedule_plan`（`gpt_model.py:798`）、`TransformerModelChunkSchedulePlan.run`（`model_chunk_schedule_plan.py:500`）、`TransformerLayerSchedulePlan.run`（236）；`combined_1f1b.py:377-382、455-456` 的注释印证调用顺序 | 一致 |
| 两个开关（§9.3 L1159） | 同上 | `megatron/core/transformer/moe/README.md` 第 20、251、434、562–563 行 | 一致 |
| 配置兼容性检查（§9.3 L1159） | 同上 | `transformer_config.py:2669-2746`、`arguments.py:1426-1434` | 一致（建议补行号，P2-22） |
| PyTorch 三个调度类 2493 / 2808 / 2994、`register_custom_function` 1887、`OVERLAP_F_B` 默认分支 2257–2260、自定义函数检查在 2246（§9.1、参考 L1269） | 本地 2.10.0 | 全部命中（`schedules.py` 3438 行） | 一致 |
| `BACKWARD_INPUT = 2`、`BACKWARD_WEIGHT = 3`、`OVERLAP_F_B = 11`（L1056–L1062） | 同上 | `schedules.py:47、48、56` | 一致 |
| `sub_actions` 构造摘录（L1066–L1072） | 同上 | `schedules.py:3093-3097` 逐行一致 | 一致 |
| `register_custom_function` 接受 `OVERLAP_F_B`（L1091） | 同上 | `schedules.py:1901-1913`；**摘录的签名被改写**（P1-5） | 行号一致，摘录不一致 |
| `ScheduleDualPipeV` 的 V 映射调用与两条 raise（L1093–L1095） | 同上 | 3026–3027 `style="v"`、3033 `n_local_stages != 2`、3038 `n_microbatches < self._num_stages` | 一致 |
| `_utils.py` 的 `generate_stage_to_rank_mapping` 91、`style == "v"` 分支 104–119 与注释原文（L1093） | 同上 | 91、104–119，注释在 113 行 | 一致 |
| `_backward.py` 先保存中间量再单独算权重梯度（L1097） | 同上 | `stage_backward_input` 143–157 行 docstring、`stage_backward_weight` 226 行 | 一致 |
| V 底部相邻本地 stage 不走 P2P（L1097） | 同上 | `stage.py:339-348`（`set_local_fwd_input` 等）、`schedules.py:2161-2220` | 一致 |
| SGLang TBO / SBO（§9.4 L1165） | `7399c2b5`（2026-08-29 −0700） | `docs/docs/advanced_features/expert_parallelism.mdx` 第 218–233 行（TBO，yield 点）、235–237 行（SBO，「These hooks execute before and after the dispatch and combine operations」，示例为共享专家与 combine 重叠） | 一致 |

**结论**：本轮没有发现源码行号错误；唯一的源码摘录问题是 P1-5。

### 4. 图片（存在性 / 元数据行号全量比对 / 读图结果）

#### 4.1 存在性与元数据

- `grep -oE 'src="\./pics/[^"]+"' deep-dive.md` 去重得 32 个路径，`pics/` 下除 `README.md` 外 32 个文件，双向 `comm` 均无输出。
- `pics/README.md` 第 1 节 32 行的体积与 sha256 前 16 位，逐一与文件实算值一致。
- 尺寸（PIL）：Zero Bubble Figure 2 为 1026×159、Figure 3 为 1032×310（L277 正确）；DeepSeek-V3 Figure 5 为 1256×212；Controllable Memory Figure 18 为 742×1723（L1298 写 743，P2-6）。
- `torch/deepep/pics/deepseek-moe-fig2-fine-grained-shared-expert.jpg` 与本目录副本 sha256 相同，L54「同一份素材的两份副本」属实。

#### 4.2 全部 32 张：本地副本行号全量比对 + 逐张读图

| # | 文件（正文行） | 本地副本行（sha256 反查） | 读图结果 | 结论 |
| ---: | --- | --- | --- | --- |
| 1 | `deepseek-moe-fig2-fine-grained-segmentation-shared-expert.jpg`（L51） | `deepseek-moe.md` 74 ✓ | 三栏：(a) Top-2，N 个专家、K=2；(b) 2N 个专家、K=4；(c) 专家 1 为绿色共享专家、K=3；图例 Routed / Shared | 相符 |
| 2 | `deepseek-v3-fig4-overlap-forward-backward-chunk-pair.jpg`（L72） | `deepseek-v3-report.md` 291 ✓ | Computation 行依次 MLP(B)▲、MLP(W)▲、MLP(F)△、ATTN(B)▲、ATTN(W)▲、ATTN(F)△，单行串行；Communication 行 DISPATCH(F)、DISPATCH(B)、COMBINE(F)、PP、COMBINE(B)；另有红色 barrier | 相符；同时是 P0-1 的证据 |
| 3 | `xiaodonggua-ep-serial-vs-overlap.jpg`（L80） | `xiaodonggua-shousi-dualpipe.md` 59 ✓ | 上排串行 9 块；下左「计算」「通信」两行加 t_overlap，计算行仍是 Attn、MLP(B)、MLP、Attn(B) 串行；下右 Dense 1F1B 加 t_dense；知乎水印 | 相符 |
| 4 | `zero-bubble-fig1-mlp-computation-graph-f-b-w.jpg`（L149） | `zero-bubble.md` 23 ✓ | Forward 栏 `Wx → σ(z)`；Backward 栏 `Wᵀ∇_zL`；W 栏 `∇_zLxᵀ`；底部标 F / B / W | 相符 |
| 5 | `gpipe-fig2-naive-vs-microbatch-pipeline.jpg`（L225） | `gpipe.md` 53 ✓ | (a) Device 0–3 上 `F_k` / `B_k` 的网络结构图，Loss 在上、Gradients 在下；(b) `F_0` 阶梯、`B_0` 与 Update 列的时间轴；(c) `F_{i,j}` / `B_{i,j}` 密排、中部 Bubble、Update 列 | (c) 相符；(a)(b) 语义与原图注不符（P0-10 ④） |
| 6 | `aiinfra-pp-gpipe-flush-idle.png`（L233） | 教材 md 13 ✓ | Device 1–4，1–8 与 9–16 两轮；只有 warmup、cooldown 两个标签；Pipeline flush 竖线、Devices idle | 「稳定」段不存在（P0-10 ⑤） |
| 7 | `aiinfra-pp-1f1b-warmup-microbatches-staircase.png`（L249） | 教材 md 346 ✓ | NPU0…NPUP-1 蓝色前向阶梯、黄色反向；顶部「每个NPU num_warmup_microbatches不同」，底部 NPUP-1 的说明文字 | 相符 |
| 8 | `zero-bubble-fig2-1f1b-schedule.jpg`（L274） | `zero-bubble.md` 38 ✓ | 只有 Forward（蓝）、Backward（橙，两格宽）、Optimizer step（米色），没有 W | 相符 |
| 9 | `aiinfra-pp-1f1b-pipedream-timeline.png`（L282） | 教材 md 29 ✓ | warmup、稳定阶段、cooldown 三个标签；第二轮出现编号 13 | 细节（P2-7） |
| 10 | `zero-bubble-fig4-optimizer-post-validation.jpg`（L375） | `zero-bubble.md` 94 ✓ | 浅橙 1–4 沿对角向下、深橙 5–8 回传、米色 optimizer 逐行错位、红色 rollback 细条 | 相符 |
| 11 | `zero-bubble-fig3-handcrafted-zb-h1-h2.jpg`（L397） | `zero-bubble.md` 41 ✓ | 见 4.3：上栏前导 F 4/3/2/1 且空格 3/2/1/0 一直保留；下栏前导 F 7/5/3/1、无空格、米色逐行错位 | 数字相符；白格解释不符（P0-3） |
| 12 | `chimera-fig2-pipeline-schemes-comparison.jpg`（L415） | `chimera.md` 74 ✓ | PipeDream 与 PipeDream-2BW 共用一组、GPipe、GEMS、DAPPLE、Chimera 共五组网格；Bubble 与 replica0 / replica1 图例；右侧 `M_θ` / `M_a` 显存柱；没有吞吐 | alt 相符；正文「吞吐」不符（P0-7） |
| 13 | `chimera-fig3-bidirectional-pipeline-schedule.jpg`（L423） | `chimera.md` 81 ✓ | replica0 的 down pipeline 与 replica1 的 up pipeline 合并为 Chimera 调度，另附「backward is 2× workload of forward」一版 | alt 相符（正文 forward doubling 说法见 P0-7） |
| 14 | `dualpipe.png`（L498） | 无本地副本声明；sha256 等于官方 `images/dualpipe.png` | Device 0–7、编号 0–9；图例五项，Backward 与 Backward for input 同色；每台设备 F 20 单位、空闲 6 单位（4.3） | 数量相符；「浅绿」不符（P0-10 ①） |
| 15 | `deepseek-v3-fig5-dualpipe-schedule-8pp-20mb.jpg`（L506） | `deepseek-v3-report.md` 306 ✓ | 8 行、编号 0–9、五项图例，块内没有表格 | 相符 |
| 16 | `xiaodonggua-official-schedule-eight-steps.jpg`（L514） | `xiaodonggua-shousi-dualpipe.md` 606 ✓ | 官方调度底图上用红线分出 F0、F0F1、B1W1F1、F0B1F1B0、B1F1B0、B1B0、WB0、W 八段 | 相符 |
| 17 | `yeqianshu-1f1b-vs-dualpipe-layers.jpg`（L522） | `yeqianshu-dualpipe-source-walkthrough.md` 59 ✓ | 左 pipe 0–7 各一层；右 Dual pipe 0–7 各两层（0/7 至 7/0），标 F0B0 与 F1B1 | 相符 |
| 18 | `sea-ai-lab-cut-in-half-mirrored-schedule.png`（L544） | `sea-ai-lab-cut-in-half.md` 19 ✓ | 全部格子补了编号，Device 3/4 之间双线，右侧「The two parts are mirrored.」；图例两行带编号色块 | 相符（「底图是官方图」的措辞见 P2-9） |
| 19 | `sea-ai-lab-cut-in-half-vshape.jpg`（L552） | `sea-ai-lab-cut-in-half.md` 26 ✓ | 上栏 Device 0–7 层配对 0/7 至 7/0，F 块呈上下镜像的两个 V；下栏 Device 0–3 只保留上半个 V，横向跨度不变；知乎 @庞天宇 水印 | 形状描述不符（P0-10 ③） |
| 20 | `controllable-memory-fig2-parallel-vs-vshape.jpg`（L562） | `controllable-memory.md` 77 ✓ | Parallel：l1+l4、l2+l5、l3+l6；V-Shape：l1+l6、l2+l5、l3+l4 | 相符 |
| 21 | `zero-bubble-fig8-zbv-schedule.jpg`（L570） | `zero-bubble.md` 158 ✓ | 4 行单格网格，每行白字与黑字并存（4.3）；图例 F / B / W / Optimizer step；米色逐行错位 | 相符 |
| 22 | `controllable-memory-fig4-vshape-full-schedules.jpg`（L578） | `controllable-memory.md` 111 ✓ | (a) 1F1B（只有 F、B）、(b) V-Min、(c) V-Half、(d) V-ZB（F / B / W），各 5 行；没有 optimizer step | optimizer 描述不符（P0-10 ②） |
| 23 | `dualpipev.png`（L615） | 无本地副本声明；sha256 等于官方 `images/dualpipev.png` | Device 0–3、编号 0–9；4 行与 `dualpipe.png` 前 4 行逐列相同（cut-in-half 的直观印证）；每台空闲 6 单位 | 数量相符；「浅绿」不符（P0-10 ①） |
| 24 | `aiinfra-pp-zbv-schedule.png`（L623） | 教材 md 73 ✓ | 与 Zero Bubble Figure 3 下栏（ZB-H2）逐格 39/39 相同；每格一个数字，没有白 / 黑两色文字，看不出 V 形 | 不符（P0-4） |
| 25 | `megatron-fig6-bubble-size-vs-data-parallel-size.jpg`（L817） | `megatron-lm-gpu-clusters.md` 155 ✓ | 图例 `n=32,b'=32`、`n=32,b'=128`、`n=128,b'=128`、`n=128,b'=512`；蓝线 0.97→0.50→0；纵轴 Pipeline bubble size | 不符（P0-5） |
| 26 | `controllable-memory-fig6-mfu-and-activation-memory.jpg`（L825） | `controllable-memory.md` 143 ✓ | 2 行 × 3 列：上排 MFU、下排激活显存；曲线 V-ZB、ZB-1P、V-Half、1F1B、V-Min、1F1B-R | 相符 |
| 27 | `megatron-fig4-default-vs-interleaved-1f1b.jpg`（L1106） | `megatron-lm-gpu-clusters.md` 62 ✓ | 上栏 default 1F1B 灰格多；中间箭头「Assign multiple stages to each device」；下栏 interleaved 灰格明显少 | 相符 |
| 28 | `aiinfra-pp-interleaved-1f1b-vpp-layout.png`（L1114） | 教材 md 43 ✓ | 深浅蓝、深浅绿四色；与 `megatron-fig4` 下栏逐格同内容；单面板，无对照 | 细节（P2-8） |
| 29 | `xiaodonggua-ep-alltoall-routing.jpg`（L1168） | `xiaodonggua-shousi-dualpipe.md` 24 ✓ | 前向 Attention→LayerNorm→Dispatch→Expert0/1→Combine→LayerNorm；X0={0,1,2}、X1={10,11,12}、{0,2}、{11} | 相符 |
| 30 | `controllable-memory-fig18-schedule-gallery.jpg`（L1295） | `controllable-memory.md` 416 ✓ | 9 组画廊，标签 (a) 1F1B 至 (h) Interleaved 1F1B，最下一组的 (i) 标签在 md 第 417 行；每组上排 building block、下排挤压后的调度 | 基本相符（P2-6） |
| 31 | `terapipe-fig1d-token-pipeline.jpg`（L1303） | `terapipe.md` 28 ✓ | Device 1–5 自下而上，layer 条带与细分 token 段，橙色细箭头跨设备 | 相符 |
| 32 | `pipedream-2bw-fig2-2bw-timeline-two-weight-versions.jpg`（L1311） | `pipedream-2bw.md` 64 ✓ | Worker 1–4、棋盘格 4、Before / After `W_1`、`W_4`、`t = 21` | 相符 |

另外为核对 L830，读了未进入 `pics/` 的 Controllable Memory Figure 8 面板：9.6B MFU 面板图例为 V-ZB mbs 4、ZB-1P mbs 4、V-Half mbs 8、1F1B mbs 4、V-Min mbs 12；38.5B 激活显存面板中 V-Min（mbs 3）约 41 GB，1F1B（mbs 1）约 35 GB。

#### 4.3 像素级复核的方法与原始读数

1. **Zero Bubble Figure 3**：网格水平线 y = 11/35.5/60/84.5/109（上栏）与 158/182.5/207/231.5/256（下栏），首条竖线 x=72，格宽 24.5 px，共 39 列；每格取内侧左上角 (3–5, 3–4) px 与格内下部两点的中位数，按最近色分到 `F`（蓝）、`B`（青）、`W`（绿）、`.`（空格）、`O`（米色 optimizer）。

   ```text
   H1 Device1 FFFF...BWFBWFBWFBWFBWBWBWBWOOFFFF...BWF
   H1 Device2 .FFF..BFBWFBWFBWFBWFBWBWBWWOO.FFF..BFBW
   H1 Device3 ..FF.BFBFBWFBWFBWFBWFBWBWWWOO..FF.BFBFB
   H1 Device4 ...FBFBFBFBWFBWFBWFBWFBWWWWOO...FBFBFBF
   H2 Device1 FFFFFFFBWFBWBWBWBWBWBWBWOOFFFFFFFBWFBWB
   H2 Device2 .FFFFFBFBFBWFBWBWBWBWBWWWOOFFFFFBFBFBWF
   H2 Device3 ..FFFBFBFBFBFBWFBWBWBWWWWWOOFFFBFBFBFBF
   H2 Device4 ...FBFBFBFBFBFBFBWFBWWWWWWWOOFBFBFBFBFB
   ```

   读数：上栏前导 F 4/3/2/1，warm-up 与首个 B 之间空格 3/2/1/0，首个 B 之后到 optimizer 之前空格 0，optimizer 起始列四行都是 27；下栏前导 F 7/5/3/1，中间空格 0，optimizer 起始列 24/25/26/27。

2. **Zero Bubble Figure 8 的文字颜色**：水平线 y = 9/32/56/80/103（4 行），竖线 77 条、间距约 12 px；每格以上下边缘像素中位数为背景，取与背景色差 > 90 的像素作为字形，均值 > 170 记白、< 100 记黑。四行分别为白 31 / 黑 40、白 33 / 黑 36、白 32 / 黑 37、白 32 / 黑 34。

3. **`dualpipe.png` 与 `dualpipev.png`**：
   - 图例（y=542/548/554 三条扫描线一致）：Forward x=328–377，RGB (240,178,56)；Backward x=633–732，(127,214,80)；Backward for input x=988–1037，(127,214,80)；Backward for weights x=1445–1494，(118,190,251)；Overlapped 为橙 1902–1951 加绿 1952–2052。
   - 网格：用 Device 0 行的竖线确定左右边界 [174, 3526]（`dualpipev.png` 为 [174, 3524]）；按竖线分段得到每行总长 66 单位，再按 50.79 / 50.76 px 每列在格内上部取 5 点中位数分类。
   - `dualpipe.png` 各行空闲列：Device 0 与 7 为 20、21、22、38、44、50；Device 1 与 6 为 0、18、19、41、47、53；Device 2 与 5 为 0、1、16、44、50、53；Device 3 与 4 为 0、1、2、47、50、53；每行 F 单位 20、空闲 6。`dualpipev.png` 的 4 行与 `dualpipe.png` 前 4 行逐列相同。
   - 本轮自己踩过、值得后续轮次避开的坑：只用贯穿全部行的竖线找网格左边界，会漏掉下面几行开头没有竖线的空白格（少 3 列）；用名义格宽 52 px 取样，到行尾会漂半格；按竖线分段数空格，会把没有竖线的空白格并进相邻格而少数空闲。先用分段确定总列数、再按精确格宽逐列取样，读数才稳定。

4. **`aiinfra-pp-zbv-schedule.png` 对 Figure 3 下栏**：水平线 y = 6.5/34.5/62/90/117.5，首末竖线 x = 90 / 1170，共 39 列（格宽 27.69 px）；两图用同一方法（格内横向 15%、85% 与纵向 20%、80% 四点中位数）分类，四行序列逐字相同，均为 39/39。

#### 4.4 未采用图的逐张理由（覆盖率）

- 论文 10 份核对表的行数与图片目录 `ls` 一致：deepseek-v3-report 88、zero-bubble 31、controllable-memory 49、chimera 26、megatron-lm-gpu-clusters 32、gpipe 9、pipedream 22、pipedream-2bw 30、terapipe 32、deepseek-moe 34，合计 353；aiinfra-docs 20 = 20；文章类 6 篇按文件名逐一核对，5/5/8/5/24/13 共 60 张全部出现在清单里。
- 论文核对表中公式截图 94 行、表格截图 57 行，每一行都写了「不采用」或「可转写为 Markdown 表格，不直接引图」，所以正文 L1332「纯公式截图与表格截图全部判为不采用」成立；记录里「41 张 / 15 张」的计数不成立（P1-2）。
- 覆盖完整不等于内容正确：本轮在核对文件里又确认了两处错误，`IMAGE-SURVEY-zero-bubble.md` 的 7/6/5/1（第 54、80、87 行）与 `IMAGE-SURVEY-aiinfra-docs.md` 把 ZB-H2 认成 ZB-V（第 30 行）；而 `IMAGE-SURVEY-gpipe.md` 本来正确的子图语义，被 `IMAGE-VERIFY.md` 第 59 行推翻了（P0-10 ④）。

### 5. 引用与 pin

- `blob/main`、`blob/master`、`tree/main` 链接 0 处；`sandbox:` 0 处；`zhimg`、`src="http…"`、`![](http…)` 0 处。
- GitHub 链接共 13 个，全部在「参考 · 源码」L1266–L1270：DualPipe 1 个 tree 与 6 个 blob，全部是 `030ce4325f4ebeb437da4ebc6d00a70469dd58ae`；Megatron-LM 1 个 tree 与 2 个 blob，全部是 `5be9626709af2722333bf54797c954c09edeada3`，本地 `git rev-parse core_v0.19.0` 与之相同，`git cat-file -t` 为 commit；`sail-sg/zero-bubble-pipeline-parallelism` 为 40 位 commit `c5d50741…` 与 tag `zero-bubble-v0.1.0`（本地无 clone，tag 与 commit 的对应无法核实）；`deepseek-ai/profile-data` 为未 pin 的仓库根链接，L1267 已明确注明（第 2 轮 P1-10 的处理成立）。
- 版本分层前后一致：DualPipe 全文只用 `030ce432…`（`3da1bbea…` 只在 L849 作为历史说明出现）；Megatron 只用 `core_v0.19.0`；PyTorch 只以「本地安装版 2.10.0 + 路径 + 行号」出现，`v2.9.0` 只在 L1051、L1269 两处否定式声明里出现，没有任何 `v2.9.0` 或 `main` 的行号，**不触发任务说明中的 P0 条件**；但「上游 v2.9.0 的类集合与行号都不同」本身没有本地证据（见「我无法核实的事项」）。
- repo 内相对链接 11 处全部存在：L7 的 `../deepep/deep-dive.md`、`../torch-distributed/readme.md`、`../nccl/readme.md`，L641 的 `../fsdp2/readme.md`，以及 L1283–L1286、L1324、L1332 的 `./codes/…`、`./notes/`、`./notes/MATH-VERIFY.md`、`./pics/README.md`。发布状态：`torch-distributed`（`README.md` 第 153 行、`README-cn.md` 第 166 行，知识图谱 published）、`nccl`（第 151 / 164 行，知识图谱 published）、`fsdp2`（第 148 / 162 行，无 `[Pending Review]`；知识图谱无条目，按 config 知识图谱只收录作者本人文章，不影响发布判断）均为已发布；`deepep` 未发布（P1-7）。
- 外部文章 URL：Sea AI Lab 本地副本第 11、71 行含英文版 `r1lVXsa9Jg`，第 75 行含中文版 `S1N_ay0ckx`，与正文一致（第 2 轮报告说中文版无本地证据，现在可以自证）；叶千树、HF 博客的日期与 `references/README.md` 第 61、59 行一致；两篇知乎文章与 AIInfra 页面本轮未联网核对。

### 6. 比较口径

| 检查项 | 结果 |
| --- | --- |
| 官方表「based on the same number of PP stages」与 DualPipeV 的 `#Devices = PP/2` | §7.1 L702 原样引用表头并逐列照抄，随后专门说明两行的参数列与设备列不在同一预算下 ✓ |
| 每张表是否写明基准 | §6.3 表按「官方表格（固定逻辑 stage 数）」与「固定设备数」两行分列 ✓；§7.3 表列出逻辑 stage 数，表后说明 DualPipeV 的半 chunk 换算 ✓；§7.4 表前写明「同设备数、同模型、同 microbatch 数」与等时、激活单位等前提 ✓；§7.5 表写明 p=8、m=20，沿用 §7.4 的前提 ✓ |
| 「参数 2x」与「删除参数副本」 | 两句的基准分别是固定 stage 数与固定设备数，§6.3 与 §7.1 前后一致，并与 README 第 26–31 行、Sea AI Lab 第 34、42–45 行吻合 ✓ |
| `F&B` 取 `max` 还是 `F+B` | 两端代数都算对，也都并列写出；但主口径选错、端点语义说反（P0-1），模拟器与 `max` 端的吻合被当成机制性验证（P0-2） |
| 分母 `3m` 与 `6m` | `max` 口径内算术成立；在与官方图一致的 `F+B` 口径下三者首项都是 `3m`，区分失去意义，且「恰好一半」对 DualPipe 不成立（P0-1 ⑤） |
| `PP+1 → (p+1/2)A` 的前提 | L795 写明「只有在写明按固定设备数、stage 减半时才成立」 ✓ |
| 时间单位 | `t` 为标准 stage chunk 的一次前向；DualPipeV 半 chunk 的换算在 §7.2 与 §7.3 表后都有说明 ✓ |
| 外部图表的口径 | Megatron-LM Figure 6 的纵轴是 `(p-1)/m` 而不是 `β`，且为解析曲线（P0-5）；Controllable Memory Figure 8 各方法 microbatch size 不同，文章把它作为「口径必须显式」的反例是恰当的，只有「各不相同」措辞过度（P2-21） |

### 7. 不可回溯断言与过度声称

| 断言（位置） | 能否回溯 | 判断 |
| --- | --- | --- |
| 官方注释「We don't overlap these two chunks to further reduce bubble size」与本文的解读（§5.6 L529、§8.5 L986） | 注释原文逐字在 `dualpipe.py:386`；「合并发送比多叠一个方块更划算」已明确标为本文解读 | 第 2 轮 P1-6 已修好 ✓ |
| 「DualPipe 相对 1F1B 的气泡优势完全建立在配对块真的重叠了这个前提上」（L118、L747） | 与本文 §7.4 表、V3 Figure 4、官方调度图矛盾 | P0-1 |
| 「本文这套离散模型恰好等价于官方公式里 `F&B = max` 那一端」（L774） | 数值成立，机制不成立 | P0-2 |
| 「双向布局恰好是第一个维度能成立的前提」（L1316） | 与 §4、§6 以及 V3 报告第 304 行矛盾 | P0-6 |
| 「DualPipe 的双向布局就是为了持续提供它」「双方向布局提供了一批天然独立的计算」（L69、L468） | 独立性并不来自双向（Megatron 注释） | P1-1 |
| Chimera 的「吞吐」与「forward doubling」（L412、L420） | 与 Figure 1 图注、§3.5 原文不符 | P0-7 |
| TeraPipe Figure 3「通信开销继续上升」（L1306） | 单卡实验，没有通信 | P0-8 |
| Megatron-LM Figure 6「实测」（L814、L820） | 解析曲线 | P0-5 |
| 「语义与 Zero Bubble 的延迟 W 同源」（L1139） | Megatron 文档只写「Split dgrad/wgrad compute」 | P2-13 |
| 「代价是边界通信变多、峰值激活变高」（L1111） | 前半有 Megatron-LM 第 85 行支持，后半与第 32 行相左 | P2-14 |
| 「这也解释了 H2 这个名字里的 2」（L390） | 论文没有解释编号 | P2-25 |
| 「通信…变成了与计算同量级的关键路径」（L56） | 出处在 V3 报告第 302 行，文中没有就近引用 | P2-23 |
| `profile-data` README 自述的边界（L1267） | 本地无副本、无 pin，文章已声明 | 无法核实，处理方式可接受 |
| verl 把训练与 rollout 交给不同后端（§9.4 L1175） | 本轮没有读 verl 源码 | 无法核实（常识性描述，风险低） |
| 草稿旧声明：`66 / 9.09%`、`6.939e-18`、`8.674e-19`、`2.10.0+cpu`、三行「官方中文注释」 | 对 `9.09`、`6.939`、`8.674`、`+cpu`、`在这里启用`、`total 66` 逐一 grep，正文无命中；官方 Python 源码无非 ASCII 字符 | 已删干净 ✓ |
| GPU 实测类声称 | 开篇 L11 与文末 L1349 各声明一次没有 GPU 实测；正文没有本文测得的吞吐、显存、带宽数字 | ✓（「实测」措辞问题见 2.4、P0-5、P2-20） |

### 8. 结构与风格

- **开篇**：以「用了 1F1B，为什么 GPU 仍然有大量等待」切入，没有「因为工作需要」「本文基于 commit」式模板句；第一人称动机（L7–L9）、系列回顾（DeepEP、PyTorch Distributed、NCCL，其中 DeepEP 未发布，见 P1-7）、4 条路线图（L15–L18）与无 GPU 实测声明（L11）齐全；致谢对象是团队（P2-4）。
- **章节顺序**：一至四章概念，五至七章调度模型，八、九章源码，十章收束，满足「概念 → 模型 / 场景 → 代码」；§2.5 较早出现一段自编验证代码（P2-19）。
- **驱动问题**：在第一章末（L85–L87）读者已握有两块背景后提出，并在 L406、L468、L1155 回收 ✓；答案的归因有偏差（P1-1）。
- **过渡句**：多数具体（如 L539「第五章最后留下了一个具体代价」、L1047「第八章的结论对使用者不太友好」）；L322 偏空泛（P2-2）。
- **推导链与表格**：没有独立的约束映射章节 ✓；§10.1 的误区对照表属于收束章节的总结，不计为违规。
- **设计分析**：演进 GPipe → 1F1B → ZB-H1/H2 → DualPipe → DualPipeV 完整 ✓；替代方案对比表在 §7.4 ✓（列语义见 P0-1）；「为什么不用 X」在 §9.4 按「TBO / SBO 解决什么 → 训练侧有 W 与优化器边界 → 不能直接类比」展开 ✓。
- **格式**：ASCII 字符画 0 处；全角引号 0 处；`——` 1 处（L1306），在 0–1 限内；Mermaid 11 处（`flowchart` 10、`stateDiagram-v2` 1），关键字合法；`[Pending Review]` 文章未被当作知识来源 ✓；行内公式分隔符（P1-3）、问句式标题（P2-1）、短促断言句（P2-3）、一处粗体不渲染（P2-5）、模型全貌（P2-26）。
- **深度校准**：understand-reproduce 的定位合适，调度与数学推导略深于原理级但有必要；第七章的问题是正确性，不是深度。
- **大纲完成度**（对照 `learn-plan.md` 十步与定稿大纲）：十步与附录 A、B 都有对应章节；计划「推荐资源」里的源码、论文与社区文章都已引用；驱动问题从计划的「第三章末」移到「第一章末」，`REVIEW.md` 已记录理由，我认为合理；计划「草稿完成度分析」里提到的 Hanayo 没有进入附录 A（可选，不另编号）。计划本身与正文不一致的旧说法见 P1-2 明细。

## 我无法核实的事项

1. **GPU 行为**：没有运行官方 `examples/example_dualpipe*.py`（需要多卡 NCCL），「示例在 GPU 上梯度数值正确」只能依据代码里的断言，本轮没有复现。
2. **NCCL 进程组在发起 P2P 前让通信 stream 等待当前 stream 的具体实现**：C++ 源码不在本地，只核对了 `Work.wait` 的 docstring。
3. **DeepSeek 内部真实重叠实现里配对块的实际耗时**：P0-1 的论证依据是 V3 Figure 4 与官方调度图的绘制方式，不是测量；它的结论是「官方材料与 `F+B` 一致，与把 `max` 解释为计算并行不一致」，不是对真实 GPU 时序的断言。
4. **按提交语义重放的模型本身**：通信耗时记 0，NCCL 分组语义按「批次发起即可配对」近似；我用 11 组配置、消息次序 0 错配、以及与官方图 66 / 6 的吻合做了交叉验证，但它仍然是模型。
5. **PyTorch 上游 `v2.9.0` 的类集合与行号**（L1051、L1269 的否定式说法）：本地没有该版本，本轮未联网。
6. **GitHub 对 `\(…\)` 数学分隔符的渲染**：本地只用 CommonMark 渲染器与仓库惯例佐证（P1-3 已标 `[UNVERIFIED]`）。
7. **外部页面**：`deepseek-ai/profile-data` README 的自述内容，hackmd、知乎、HuggingFace、AIInfra 页面的当前可达性与内容。
8. **Sea AI Lab 博客日期 2025-02-27**：本地副本只有 bibtex 年份 2025，具体日期只来自 `references/README.md` 第 62 行。
9. **`sail-sg/zero-bubble-pipeline-parallelism`** 的 tag `zero-bubble-v0.1.0` 与 commit `c5d50741…` 是否存在及其关系（本地无 clone）。
10. **`references/community/easy-dualpipe/`**：未运行，也未审阅其代码。
11. **verl 的后端组合描述**（§9.4 L1175）：未读 verl 源码。
12. **并发写入者**：`notes/REVIEW-deep-r3.md` 与 `codes/__pycache__/` 的创建者只能从时间戳推断为另一个审查会话；我没有读那份报告，因此也无法说明两份报告的结论是否一致。

## 流程层面的观察

1. **修复没有传播**：第 2 轮 7 条 P0 在正文里都修对了，但其中两条在 `MATH-VERIFY.md`、`IMAGE-VERIFY.md`、`pics/README.md` 里原样保留，`codes/README.md` 还有与脚本输出不符的声明。`/learn-review` 的 Step 3.3「引用正确性」只检查链接能否解析，建议增加「被正文引用的记录文件与正文事实一致」一项，并在修复流程里强制写传播清单（争议项 4 第 3 条）。
2. **核对记录会推翻正确结论**：`IMAGE-SURVEY-gpipe.md` 与 `learn-plan.md` 第 294 行原本写对了 GPipe 子图语义，`IMAGE-VERIFY.md` 却把它「判定」成错误写法，原因是只看画面、不读论文图注。建议把读图拆成两步并都留痕：**先读原文图注与引用该图的正文段落，再看画面**。本轮的 GPipe、Megatron Figure 6、TeraPipe Figure 3、Chimera Figure 2 四处错误，读一遍图注或相邻段落就能发现。
3. **「数字对得上」不等于「模型对」**：写作阶段与前两轮都用模拟器交叉验证闭式，数字全部吻合，却没有人把模拟器的依赖语义与源码的提交点逐条对照，也没有人去数官方调度图里的空闲格。建议 `/learn-review` 在可复现性维度区分三层：脚本能跑、输出与文章一致、脚本的建模假设与被模拟的源码一致，第三层要求逐条列出假设。
4. **二手素材需要反向校验**：AIInfra 教材把 ZB-H2 的图放在 ZB-V 标题下，核对文件与正文都沿用了标题。凡是教材页、社区文章里的图，建议至少与一手论文图做一次逐格或逐元素比对。
5. **像素级复核应当脚本化并复用**：本轮在 Figure 3、Figure 8 与官方调度图上各踩过一次取样位置、网格左边界或格宽漂移的坑（4.3 第 3 条）；建议经用户授权后把这类只读脚本存到 `notes/checks/`，报告里写明取样参数，下一轮直接复跑。
6. **计数断言需要附命令**：「41 张 / 15 张」「合计 435」「453 张」「31 张」这类数字在多份记录之间互相抄写，没有一份附上生成命令；建议所有计数都在记录里附一行命令。
7. **untracked 目录与并发会话**：`torch/dualpipe/` 没有进入 git，轮次之间无法 diff；本轮审查期间还有另一个会话写入同一目录。建议每轮开始前由用户授权做一次快照 commit，或至少在报告头记录 md5 并约定审查期间冻结正文（争议项 4 第 4 条）。
8. **严重度口径需要固定**：同类问题在不同轮次被分到不同等级（例如第 2 轮把 Table 9 的范围问题定为 P2，本轮定为 P1 并注明可升 P0）。建议把本报告开头的口径说明并入 `.claude/commands/learn-review.md` 的报告格式（修改命令定义需用户授权）。
