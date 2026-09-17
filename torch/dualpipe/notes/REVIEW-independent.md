# deep-dive.md 独立技术审查报告

审查对象：`torch/dualpipe/deep-dive.md`
审查时锁定 revision：**1368 行，md5 `9174861c479a70524921ff06fab92497`**（2026-09-10 11:43 的版本；本报告中每一条结论都已在该版本上重跑复验）
复核环境：DeepSeek `DualPipe` commit `030ce4325f4ebeb437da4ebc6d00a70469dd58ae`（本地 `/workspace/algorithm/DualPipe`）、Megatron-LM tag `core_v0.19.0` = `5be9626709af2722333bf54797c954c09edeada3`（`git show` 读取，未用工作树 HEAD）、本地安装版 PyTorch `2.10.0+cu129`（`/usr/local/lib/python3.12/dist-packages/torch/distributed/pipelining/`）。

> **审查过程中文章被并发修改过**：我首次读取时是 1350 行（md5 `2849ca2a6f316877ca206f52974cb008`），中途变为 1368 行（md5 `e1289c5420a58cb13295b7639765ab63`），成稿前又变为 md5 `9174861c479a70524921ff06fab92497`。**每一次变化后我都重跑了全部自动核对**：8 处行号错配、7 类 P0 文本、图片与 `pics/` 资产（terapipe 那张的 sha256 仍是 `a7c15872…`、图片数仍是 32）在三个版本上表现完全一致，因此本报告结论对最后一版同样成立；行号以最后一版为准。若文章再次变动，行号需重新对齐。

---

## 总评

文章的骨架是成立的：数学主线（1F1B → Zero Bubble → DualPipe → DualPipeV）逐式复算全部通过，源码事实（DeepSeek 440/411/80/38 行、入口断言、八步结构、`WeightGradStore` 会计、Megatron 与 PyTorch 的每个行号）逐条核对**无一处错误**，配套脚本可复跑且文中引用的每个数字（`1.110e-16`、busy 30/31/1、60/63/3、3.23%、4.76%、`2.10.0+cu129`）都与实际输出一致。也就是说，**它的问题不在公式和行号，而在「读图」与「读来源」这两类断言上**：我把文章引用的 11 张图逐张实看（其中 Figure 3 做了像素级逐格分类），发现 3 处图文不符（含 1 张图被放成了同图的另一个子图），另有 1 处把 A 论文的结论与例子记到了 B 论文头上、1 处把官方 README 没说过的话算在 README 头上，以及 8 处「本地副本第 N 行引用图」的行号全部对不上。

**最严重的三条**：(1) `terapipe-fig1d-token-pipeline.jpg` 里装的其实是 TeraPipe Figure 1(**c**)（GPipe microbatch 流水线），不是 1(d)，alt 与图注描述的「很细的 token 分段」在画面上根本不存在；(2) Zero Bubble Figure 3 下栏（ZB-H2）的 warm-up F 数被写成 `7/6/5/1`，逐格实测是 `7/5/3/1`，而这正是第四章「H2 靠多铺前向填气泡」的唯一读图证据；(3) ZB-V 的来源被记成「更早的 Controllable Memory 论文」，实际出自 Zero Bubble 论文 §6/Figure 8，两者时间顺序正好相反。这三条都属于「读起来很顺、落到来源站不住」，属于必须修复的 P0。

按现状，我认为文章**尚未达到可发布水平**：P0 修完之后（预计改动量不大，主要是换图、改 3 处图注数字、改 2 处来源归属、对齐 8 个行号），它是一篇质量相当高的源码级深度解析，数学与源码部分我可以签字。

---

## P0（事实/数学/引用错误，必须修复）

| # | 位置（行号/小节） | 问题 | 证据 | 建议改法 |
| --- | --- | --- | --- | --- |
| P0-1 | 557（§6.1 第一段） | 把 ZB-V 的来源记成「Zero Bubble 团队在**更早的** Controllable Memory 论文里就提出了 ZB-V」，并把「4-stage、16 层、worker 1 拿 1-2 与 15-16 层」的例子、「前向反向起止于同一设备」的性质一并记到该论文名下 | ZB-V 出自 Zero Bubble 论文 §6（`references/papers/zero-bubble/zero-bubble.md` 第 156 行节标题、第 163 行「we design ZB-V」、第 161 行逐字给出 16 层/4-stage 例子、第 158 行 Figure 8 标题），arXiv `2401.10241`（2024-01）；Controllable Memory 是 `2405.15362`（2024-05），**更晚**，且其正文只提 V-Min / V-Half / V-ZB（该 md 第 82 行），没有「ZB-V」这个名字。文章自己在 567、573 行又把 ZB-V 归给 Zero Bubble §6/Figure 8，前后矛盾 | 改成「ZB-V 出自 Zero Bubble 论文 §6（Figure 8）；Controllable Memory（更晚）把它推广成 V-Shape building block 家族（V-Min / V-Half / V-ZB）」，并把 16 层例子与「起止同设备」性质标注为 Zero Bubble §6 |
| P0-2 | 1300-1304（附录 A 第三条线索） | 引用 `pics/terapipe-fig1d-token-pipeline.jpg` 讲 TeraPipe Figure 1(d)，但该文件内容是同图 **1(c)**；alt「每个格子对应很细的 token 分段」与画面不符；且给出的「第 28 行引用图」与该文件实际来源行（第 25 行）不一致 | `sha256(pics/terapipe-fig1d-token-pipeline.jpg)` = `a7c15872…`，与 `references/papers/terapipe/images/b78a683e0a3b8eee6d67a9733ef0708c4bf996c0bbf75ac21882a5dc07339edb.jpg` **字节相同**，而该文件在 `terapipe.md` 第 **25** 行被引用、其后第 26 行的图注是「(c) Microbatch-based pipeline parallelism (GPipe)」；真正的 1(d) 是 `a73bba64…jpg`（326×340，md 第 28 行）。逐图对比：`b78a683e` 的每设备条带是整块、只有两条跨设备弯箭头（microbatch 流水线）；`a73bba64` 的条带被切成 4-6 段、有大量细箭头（token 级流水线）。`notes/IMAGE-SURVEY-terapipe.md` 第 69 行也记录 1(d) = `a73bba64…`，即复制时取错了源文件 | 用 `references/papers/terapipe/images/a73bba64…jpg` 替换 `pics/terapipe-fig1d-token-pipeline.jpg`（并同步 `pics/README.md` 的 sha256），引用行号改为 28；若只想保留 1(c)，则必须改标题、alt 与正文，说明它讲的是 GPipe 的 microbatch 流水线 |
| P0-3 | 397（Figure 3 alt）+ 402（正文） | 称 Zero Bubble Figure 3 下栏（ZB-H2）「warm-up 的 F 数增至 **7/6/5/1**」 | 对 `pics/zero-bubble-fig3-handcrafted-zb-h1-h2.jpg` 逐格做颜色分类（cell 宽 ≈24.6 px、网格原点 x≈74，上下栏各采样 4 个 y 偏移取多数），四行前导连续 F 数为：上栏 **4/3/2/1**（白格 3/2/1/0，与文章一致）、下栏 **7/5/3/1**。分类串：上栏 `FFFF...BWFBW…`、`.FFF..BFBW…`、`..FF.BFBF…`、`...FBFBF…`；下栏 `FFFFFFFBWFBW…`、`.FFFFFBFBFB…`、`..FFFBFBFBF…`、`...FBFBFBF…`。放大到 4× 复原后同样读作 7/5/3/1（下栏第 2 行是 `1 2 3 4 5` 后接青色 `1`）。**结构上也只有 7/5/3/1 自洽**：H2 每行的前导 F 数恰好等于 H1 同一行的「F + 白格」数（4+3=7、3+2=5、2+1=3、1+0=1），这正是「用多铺的前向把 H1 留下的白格填掉」；7/6/5/1 不构成任何等差结构。注：`notes/IMAGE-SURVEY-zero-bubble.md` 第 54 行也写 7/6/5/1，说明是**读图记录本身错了**，文章照抄了下来 | 两处都改为 7/5/3/1，并顺手在 IMAGE-SURVEY-zero-bubble.md 里订正（该文件不在本次允许改动的范围内，仅提示） |
| P0-4 | 570（Figure 8 alt） | 称 ZB-V 图「4 个 Device 的时间网格，**每个 Device 下有上下两行小格**分别对应它的两个 chunk」 | 逐像素检查 `pics/zero-bubble-fig8-zbv-schedule.jpg`（1035×159，4 行 Device 1–4）：每一行内每个格子的上半与下半颜色分类完全一致（18 格中 0–2 格因边界混色不一致），即**每格是单色单格、不存在上下两行**。论文 Figure 8 图注（`zero-bubble.md` 第 159 行）明确写两个 chunk 用**文字颜色**区分：「white text colors represent the first chunk and black text colors represent the second chunk」——这一点在放大图中可直接看到（同一行里数字有白字与黑字两种） | 改为「每个 Device 一行时间网格，格内数字的颜色（白/黑）区分它的两个 chunk」，或直接照抄论文图注口径 |
| P0-5 | 495（正文）+ 506（Figure 5 alt）+ 618（§6.2 口径提醒） | 三处把 `dualpipe.png` / DeepSeek-V3 Figure 5 说成「两个方向**各** 20 个 micro-batch」/「双向各 20 个 micro-batch」 | 官方图注是「8 PP ranks and 20 micro-batches **in two directions**」，且「The micro-batches in the reverse direction are symmetric …, so we omit their batch ID」；两张图（`pics/dualpipe.png`、`pics/deepseek-v3-fig5-dualpipe-schedule-8pp-20mb.jpg`）格内编号范围都只到 **9**（0–9），即每方向 10 个、合计 20 个。这与本文 448 行「两条路径通常各处理 \(m/2\) 个」、以及 489 行用 `--p 8 --m 20` 复现 `F0_6,B1_2,F1_4,B0_0`（模拟器里 \(m=20\) 的总量口径）自相矛盾；`notes/MATH-VERIFY.md` 也把 \(m\) 定义为「一次逻辑迭代的**总** microbatch 数，DualPipe 下两个方向各 \(m/2\)」 | 三处统一为「双向共 20 个（每方向 10 个）」，保留 \(m=20\) 的总量口径 |
| P0-6 | 643（§6.3）+ 795（§7.4） | 「**官方 README 与 Sea AI Lab 博客都提到** PP 边界通信翻倍 / 相对 EP 开销较小」 | `grep -i "comm" /workspace/algorithm/DualPipe/README.md` 只命中 2 处，都是图注里的「mutually overlapped computation and communication」，**全文没有任何关于 PP 通信量或 PP vs EP 开销的表述**。该说法只存在于 Sea AI Lab 博客（`references/articles/sea-ai-lab-cut-in-half/sea-ai-lab-cut-in-half.md` 第 40 行表头「PP 通信」列与第 49 行「对半裁剪方案的 PP 通信量是其他方法的两倍……相较于 EP 通信，PP 通信的开销较小」）与 V3 报告 Figure 5 附近的讨论之外 | 删掉「官方 README 与」，只保留 Sea AI Lab 博客（并给出第 49 行这一具体位置）；若要保留 README，需先找到 README 里对应句子 |
| P0-7 | 83、517、525、626、1113、1167、1302、1310（八处「本地副本：`<path>` 第 N 行引用图」） | 八处行号与「该图在被引 md 中出现的行」不一致，其中 1310 指向的还是**另一张图的图注** | 用 sha256 把 `pics/` 中每张图逐一匹配回 `references/` 原图，再在该 md 里定位该图像的行号：83 行 xiaodonggua-ep-serial-vs-overlap 声称 296 / 实际 **59**；517 行 official-schedule-eight-steps 声称 214 / 实际 **606**；525 行 yeqianshu-1f1b-vs-dualpipe-layers 声称 100 / 实际 **59**；626 行 aiinfra-pp-zbv-schedule 声称 137 / 实际 **73**；1113 行 aiinfra-pp-interleaved-1f1b-vpp-layout 声称 51 / 实际 **43**；1167 行 xiaodonggua-ep-alltoall-routing 声称 41 / 实际 **24**；1302 行 terapipe-fig1d 声称 28 / 实际 **25**；1310 行 pipedream-2bw-fig2 声称 66 / 实际 **64**（且第 66 行是 **Figure 3** 的图注「Figure 3. Timelines of GPipe and PipeDream-Flush…」，与本图无关）。其余 20 处行号全部正确，说明这是逐条抄录时出错而非口径不同；`notes/IMAGE-SURVEY-aiinfra-docs.md`（第 74、90 行）与 `IMAGE-SURVEY-pipedream-2bw.md`（第 58 行）记录的正是 43、73、64 | 逐条改成上列实际行号；建议加一条自动校验（按 sha256 反查被引 md 行号）写进 `codes/README.md` 的复核流程 |

---

## P1（结构、口径、可读性）

| # | 位置 | 问题 | 建议 |
| --- | --- | --- | --- |
| P1-1 | 772（§7.3）、780-785（§7.4 表「气泡（串行执行）」列）、797、1346 | 「串行执行」一词被用于两个互不相干的模型：§7.2/7.4 表里的「串行执行」= 配对块被执行器顺序执行（气泡翻倍，DualPipe `(p-2)t`）；§7.3 与 1346 行又把模拟器（**完全不建模通信重叠**）的结果叫「串行执行下的气泡」，而模拟器给出的是 `(p-2)t/2`（即 7.3 表与 7.2 的**重叠成立**那一端）。同一名词指向相差一倍的两个数，读者按表去核对脚本必然对不上 | 把两类口径改名区分，例如「配对块串行执行」（执行器口径）与「无通信重叠 + 依赖等待」（模拟器口径）；并在 7.4 表下补一句「模拟器给出的是『重叠成立』那一列，不是『串行执行』那一列」 |
| P1-2 | 795（§7.4 第四条） | 同一句话自相矛盾：先说「每次传输的数据量是减半的」，括号里又说「单次数据量不变、次数翻倍」。正确结论是**单次数据量不变、次数翻倍**（边界张量是同一个隐藏状态，与每 stage 层数无关） | 删掉「每次传输的数据量是减半的」，保留括号里的解释 |
| P1-3 | 476、491（§5.4） | 记号缺陷：公式里只有 \(x\)、\(r\)，正文却说「稳态的第 \(i\) 组配对」；491 行断言「在上半区里 \(b_{1,i} \ge b_{0,i}\) 恒成立」，其中 \(b_{1,i}\)、\(b_{0,i}\) 全文未定义，该断言因此无法核对 | 「第 \(i\) 组」改为「第 \(x\) 组」；491 行改用已定义量重述（例如「方向 1 的前向进度 ≥ 方向 0 的前向进度」并给出定义），或直接删去该句、只保留「跨中点后公式需换成对称形式」的结论 |
| P1-4 | 834（§8.1） | 「官方仓库只有四个 Python 文件加两个示例」，但仓库还有 `dualpipe/__init__.py`（本文 845 行自己还引用了它） | 改为「四个功能模块文件（另有 `dualpipe/__init__.py`）加两个示例」 |
| P1-5 | 1328（附录 B） | 两处计数对不上：①「本目录 `pics/` 下的 **31** 张图」实际是 **32** 张（`pics/` 下非 README 文件 32 个，正文 `src="./pics/…"` 去重后也是 32）；②「参考资料下 **453** 张抽取图（论文 353、教材页 20、社区文章 60）」——三项相加只有 **433**，`references/` 下图片实际总数 **435**（papers 353 + docs 20 + articles 60 + community 2） | 分别改为 32 与「435 张（论文 353、教材页 20、社区文章 60、社区复现代码 2）」 |
| P1-6 | 982（§8.5） | 把源码注释没说的理由算给了注释：注释原文是「NOTE: We don't overlap these two chunks to further reduce bubble size.」（`dualpipe.py:386`），文章写成「说明这一步是故意不重叠的……只为了让依赖消息更早发出去」，并称「这就是 5.6 节那句话的源码依据」 | 改成「注释给出的理由是进一步减小气泡；代码把 `_send_forward/_send_backward` 插在两次计算之间，因此也顺带更早释放依赖消息」，把「早释放」明确标为本文的机制解读 |
| P1-7 | 1360-1365（文末 `/learn-write` 注释）与 7（开篇） | 自相矛盾：开篇说 DeepEP 那篇「尚未进入两个 README 的索引」，文末却写交叉引用「均为 published 状态」并称「torch/deepep/ 的**已发布**结论」。按仓库规则，未进 README 索引的文章不宜当知识来源，两处口径必须统一 | 文末改为「/learn-review 的引用合规性以两个 README 为准；DeepEP 尚未进索引，仅作同期姊妹篇链接，不作为知识来源」 |
| P1-8 | 1101-1107（§9.2 节点表） | `mtp_post_process` 的 stream 列写「—」，但源码 `model_chunk_schedule_plan.py:172` 是 `create_node(comp_stream, mtp_post_process_module, …)`，即它挂在**计算 stream** 上 | 该单元格填「计算 stream」（或写明「非 MTP 层为 Noop」）；顺带在 570 行的 Figure 8 alt 里补上「Optimizer step」图例项（图中确实有） |
| P1-9 | 246（§3.2） | 引用「Megatron-LM 的非交错实现……`num_warmup_microbatches` 先取 `p - rank - 1` 再与 `num_microbatches` 取小」但未固定 revision（第九章才统一到 `core_v0.19.0`）。该说法在 `core_v0.19.0` 的 `schedules.py:930` 成立，但读者无法回溯到具体版本 | 在该处补一句「（以下行号按 tag `core_v0.19.0`：`megatron/core/pipeline_parallel/schedules.py:926-950`）」 |
| P1-10 | 1263（参考·源码） | `deepseek-ai/profile-data` 只给了仓库根链接、未锁 commit，而正文依赖「其 README 说明模拟了完全均衡的专家路由、训练 trace 省略了 PP 通信」这样的**可变内容** | 锁一个 commit（或注明「无 pin，按 2026-09 的 README 自述」），与全文其它引用的 pin 纪律保持一致 |

---

## P2（细节、措辞）

| # | 位置 | 问题 | 建议 |
| --- | --- | --- | --- |
| P2-1 | 679（§6.4） | 「\(p/2\) 全部换成了 \(p\)、**\(2p\) 换成了 \(2p\)**」——后半个分句是空操作，显然是笔误 | 改为「\(p/2\) 全部换成了 \(p\)、\(p\) 换成了 \(2p\)」 |
| P2-2 | 162（§2.2） | 「论文 Table 9 的 profiled 数据里 \(T_B < T_F\) 且 \(T_B + T_W\) 与 \(2T_F\) 相差约 26%」——26% 只对 1.5B / \(p=8\) / \(m=24\) 那一行成立（其余各行约 14%–17%） | 补上配置：「以 1.5B、\(p=8\)、\(m=24\) 为例（\(T_F=18.522\)、\(T_B=18.086\)、\(T_W=9.337\)）相差约 26%，其余配置 14%–17%」——`notes/MATH-VERIFY.md` 已经写成这个更精确的形式 |
| P2-3 | 772（§7.3） | 「ZB1P 的气泡与 \(F\&B\) 无关（它没有配对块），所以它的分母**永远**是 \(3m\)」——分母差别的真实原因是 DualPipe 的气泡只有一半（约分后出现 \(6m\)），与「有没有配对块」没有直接因果；且 ZB1P 的实际分母是 \(3m+p-1\)，「永远 3m」忽略了常数项 | 改为「ZB1P 的气泡是 \(O(p)\)，分母首项是 \(3m\)；DualPipe/DualPipeV 的气泡减半，通分后首项变成 \(6m\)」 |
| P2-4 | 807-808（§7.5 表） | 串行列写成 `12/132`、`14/134`，与同表其它分数（`7/27`、`6/126`）的约分口径不一致（数值等价，但读者会以为分母有别的含义） | 统一约分为 `6/66`、`7/67`（或统一不约分） |
| P2-5 | 986（§8.6） | 「`comm.py` 只用 38 行做了一件事」——实际做了四件事：设置形状/dtype 全局变量、`build_from_tensor_shapes` 建 buffer、`append_irecv`、`append_isend` | 改为「只用 38 行把『张量形状』变成全局契约」 |
| P2-6 | 552、555（sea-ai-lab-cut-in-half-vshape） | 图面带「知乎 @庞天宇」水印，而署名只写 Sea AI Lab 四位作者（本地 md 确为 Sea AI Lab 博客的知乎转载，标题即「给 Sea AI Lab 的同事大佬们宣传一下新的 blog」） | 图片来源改为「Sea AI Lab 博客（知乎转载，知乎账号 @庞天宇）」，避免署名与画面互相打脸 |
| P2-7 | 880（§8.4） | 「延迟参数梯度的实现不到 30 行」过于含糊：`WeightGradStore` 占 `utils.py` 第 8–33 行（26 行），`put/flush/pop` 三个方法 15 行 | 直接写「`WeightGradStore` 占 `utils.py:8-33`（26 行），其中 `put/flush/pop` 15 行」 |
| P2-8 | 753（§7.3 第 1 条）与 `codes/02_schedule_sim.py:68-80` | 模拟器对非 zb 反向按「先 \(I\) 后 \(W\)」建模；而官方示例 hook 里 `grad_weight_fn()` 是在算 `grad_input` **之前**执行的（`examples/example_dualpipe.py:24-33`），即示例的算子级顺序相反。这是合理的抽象取舍（文章明确说依赖模型只挂 \(I\)），但文章没写这一句 | 在 7.3 或 1350 行的「验证范围」里加一句：「模拟器把一次非 zb 反向统一记成 \(I \to W\) 两个任务，与示例 hook 算子级的 \(W\) 先于 dgrad 不同，只影响同 rank 内部的松弛量，不影响跨 rank 依赖」 |

---

## 逐项核对记录

### 1. 数学逐式复核

| 公式 | 来源（实际读到的位置） | 我的复算 | 结论 |
| --- | --- | --- | --- |
| \(Y=X\Theta^\top\)、\(I=G\Theta\)、\(W=G^\top X\)，例子 `[4,8]`/`[16,8]`/`[4,16]` | 标准矩阵微分；文章 §2.2 | \(G\Theta:[N,H_{out}]\times[H_{out},H_{in}]=[N,H_{in}]\) ✓；\(G^\top X:[H_{out},N]\times[N,H_{in}]=[H_{out},H_{in}]\) ✓；数值例子维度全部对得上 | 一致 |
| Table 1（F/B/W 的 FLOPs 与激活） | `zero-bubble.md:60` | 逐项转写一致；\(T_B+T_W=(24h+8s)+24h=48h+8s=2(24h+4s)=2T_F\) ✓；\(M_W=32sbh<M_B=sb(34h+5as)\) ✓；\(T_W<T_F<T_B\) ✓ | 一致 |
| 1F1B warmup \(\min(p-r-1,m)\) | 文章 §3.2；Megatron `core_v0.19.0` `schedules.py:930`（`pipeline_parallel_size - pipeline_parallel_rank - 1`，随后被 `total_num_microbatches` 截断） | p=4 时 3/2/1/0 ✓ | 一致 |
| 稳态配对 \((F_{k+p-r-1},B_k)\) | 文章 §3.2（由 warmup 数推出） | p=4：rank0 \(F_3\)、rank1 \(F_2\)、rank2 \(F_1\)、rank3 \(F_0\)，与文章表格逐行一致 | 一致 |
| \(T_{\mathrm{1F1B}}=(m+p-1)(F+B_{\mathrm{full}})\) | `zero-bubble.md:348`（附录 H 原文「an 1F1B iteration takes \((m+p-1)*(T_F+T_B+T_W)\)」，前提 \(m\le p\)、忽略通信） | 与我们熟悉的 \((m+p-1)(F+B)\) 同式；文章已注明论文的 \(T_B+T_W\) 即本文 \(B_{\mathrm{full}}\) | 一致 |
| \(T_{\mathrm{bubble}}=(p-1)(F+B_{\mathrm{full}})\)、\(\beta=(p-1)/(m+p-1)\) | `zero-bubble.md:70`（Table 2 第 1 行） | 相除即得 ✓ | 一致 |
| ZB-H1 / H2 气泡 \((p-1)(F+I-W)\)、\((p-1)(F+I-2W)\) | `zero-bubble.md:70` Table 2 + 记号映射 \(T_B=I,T_W=W\) | 等时代入：H1 \((p-1)t\)（1F1B \(3(p-1)t\) 的 1/3 ✓）、H2 \(0\) ✓；README 的 ZB1P \((PP-1)(F+B-2W)\) 代入 \(B=I+W\) 与 H1 同式 ✓ | 一致 |
| H1/H2 峰值激活 \(pM_B\)、\((2p-1)M_B\) | `zero-bubble.md:62`（§2.3 worker 公式） | H1: \((p-i+1)M_B+(i-1)M_W\) 在 \(M_W<M_B\) 时随 \(i\) 递减，峰值 \(i=1\) 得 \(pM_B\) ✓；H2: \((2p-2i+1)M_B+(2i-2)M_W\) 每步增 \(2(M_W-M_B)<0\)，峰值 \((2p-1)M_B\) ✓ | 一致 |
| ZB-V warm-up \(2p-1\) 次（chunk1 做 \(2p-i\)、chunk2 做 \(i-1\)）、等时零气泡、峰值 \(pM_B\) | `zero-bubble.md:163-165`（§6 原文） | 与文章 567 行逐字对应 ✓ | 一致 |
| DualPipe 稳态四元组 \((F_{0,x},B_{1,x-p/2},F_{1,x-p/2+1+r},B_{0,x-p+1+r})\) | 文章 §5.4（自有推导）；官方 `dualpipe.py:358-396` | 由 step1/2/3 的循环次数得进入 step4 时 \(F_0=2H-r-1=p-r-1\)、\(F_1=H=p/2\)、\(B_1=H-r-1\)、\(B_0=0\)，故 \(x_0=p-r-1\)；四条偏移量 \(-p/2,\ -p/2+1+r,\ -p+1+r\) 与 \(x_0\) 代入后精确复现模拟器 p=8,r=1 打印的 \(F0_6,B1_2,F1_4,B0_0\) ✓；每一步主循环四个计数器同步 +1，偏移量恒定 ✓ | 一致 |
| DualPipeV 稳态四元组 \((F_{0,x},B_{1,x-p},F_{1,x-p+1+r},B_{0,x-2p+1+r})\) | 文章 §6.4；官方 `dualpipev.py:330-367` | 进入 step4 时 \(F_0=2p-r-1\)、\(F_1=p\)、\(B_1=p-r-1\)、\(B_0=0\)，\(x_0=2p-r-1\)；代入 p=4,r=1 → \(F0_6,B1_2,F1_4,B0_0\)，与 `--p 4 --m 10` 的 DualPipeV rank 1 输出逐项相同 ✓ | 一致（但 679 行的文字说明有笔误，见 P2-1） |
| 每设备有效工作 \(3mt\) | 文章 §7.4 | DualPipe：\(2\times\frac{m}{2}\times 3t=3mt\)；DualPipeV：\(2\times m\times\frac{3t}{2}=3mt\)；模拟器 busy=30（p4,m10）、60（p8,m20）与 \(3mt\) 一致 ✓ | 一致 |
| DualPipe 气泡 \((p-2)t/2\)、DualPipeV \((p-1)t/2\)；气泡率 \((p-2)/(6m+p-2)\)、\((p-1)/(6m+p-1)\) | 文章 §7.3/7.4 | `--sweep`（p∈{2,4,6,8}、m∈{8,10,16,20,32}）全部一致（脚本内断言 + 我的实际运行输出）✓；解析：rate \(=\frac{(p-2)/2}{3m+(p-2)/2}=\frac{p-2}{6m+p-2}\) ✓ | 一致 |
| README 式 \((PP/2-1)(F\&B+B-3W)\) 在 \(F\&B=\max(F,B)\) 下 \(=(p-2)t/2\) | `/workspace/algorithm/DualPipe/README.md` 比较表 + 文章 §7.3 交叉验证 | 等时 \(F=I=W=t\Rightarrow B_{\mathrm{full}}=2t\)；\(F\&B=\max(t,2t)=2t\)；\(F\&B+B-3W=2t+2t-3t=t\)；\((PP/2-1)t=(p/2-1)t=(p-2)t/2\) ✓ 与模拟器实测完全相同——**文章这一步验算成立** | 一致 |
| DualPipeV 版 \((p-1)(F\&B'-\tau)\) 两端 \(=(p-1)t/2\) 与 \((p-1)t\) | 文章 §7.2 | 半 chunk \(F=I=W=\tau=t/2\)、\(B=2\tau=t\)、\(PP=2p\)：\(\max\) 端 \((p-1)(t-\tau)=(p-1)t/2\) ✓；串行端 \((p-1)(3\tau-\tau)=(p-1)t\) ✓ | 一致 |
| 分母 \(6m\) 与 \(3m\) 的区分 | 文章 §7.3 | 作为首项成立：ZB1P \( (p-1)/(3m+p-1)\) 首项 \(3m\)，DualPipe \((p-2)/(6m+p-2)\) 首项 \(6m\)；但「因为 ZB1P 没有配对块」的解释牵强（见 P2-3） | 数值一致、解释待改 |
| 激活换算 \((2p+1)\cdot A/2=(p+\tfrac12)A\) | 文章 §7.4 | 2048；\(2p+1\) 乘 \(A/2\) = \((p+0.5)A\) ✓ | 一致 |
| 隐藏状态 \(2\times2048\times4096\times2\) bytes \(=32\) MiB | 文章 §8.6 | \(2\cdot2048\cdot4096\cdot2=33{,}554{,}432\) B \(=32\) MiB ✓ | 一致 |
| ZB-H1 配对式 \((F_x,I_{x-(p-1-r)},W_{x-(p-1)})\) 与 §4.1 的分段计数 | 文章 §4.1（论文未给出该式） | 区间不等式自洽：\(x<p-1-i\) 时只有 F（共 \(p-1-i\) 个）、\(p-1-i\le x<p-1\) 时 \((F,I)\)（共 \(i\) 个）✓；与 Figure 3 上栏抽查一致（i=0 行出现同编号的 I/W 相邻、i=3 行出现同编号的 F/I 相邻） | 与来源不可比（论文无此式），内部自洽 |
| §7.5 表（p=8,m=20） | 文章 §7.5 | \(6m+p-2=126\)、\(3m+p-1=67\) ✓；1F1B 21 与 \(7/27\) ✓；ZB1P 7 与 \(7/67\) ✓；DualPipe 3 与 \(6/126\) ✓；DualPipeV 3.5 与 \(7/127\) ✓ | 一致（分数未约分见 P2-4） |
| 「论文 Table 9 的 \(T_B<T_F\) 且 \(T_B+T_W\) 与 \(2T_F\) 相差约 26%」 | `zero-bubble.md:302` Table 9 | 1.5B/8/24 行：27.423 vs 37.044 → 26.0% ✓；6.2B 16.9%、14.6B 14.6%、28.3B 14.0% | 仅 1.5B 行成立（见 P2-2） |
| §7.4「每个 microbatch 每次传输的数据量减半」 | 文章 §7.4 | 边界张量是同一份隐藏状态，单次大小不变；总量 ≈2× | **错误**（同句括号已给出正确说法，见 P1-2） |

### 2. 代码可复现性

命令与输出（`python torch/dualpipe/codes/...`，本机 `torch 2.10.0+cu129`）：

| 命令 | 实际输出要点 | 与文章数字是否一致 |
| --- | --- | --- |
| `01_split_backward_minimal.py` | `torch 2.10.0+cu129`；`float64`；`x/theta = (8,8)/(16,8)`；microbatch 2；`dTheta is None: True`；`max abs dX diff = 0.000e+00`；`max abs dTheta diff = 1.110e-16`；`PASS` | **一致**（205-212 行逐字对应） |
| `02_schedule_sim.py --p 4 --m 10` | DualPipe busy 30 / makespan 31 / bubble 1 / 3.2258%；DualPipeV busy 60 / makespan 63 / bubble 3 / 4.7619%；闭式复核全部「一致」；DualPipeV rank 1 稳态首组 `F0_6, B1_2, F1_4, B0_0` | **一致**（757-764 行、660-666 行） |
| `02_schedule_sim.py --p 8 --m 20` | DualPipe busy 60 / makespan 63 / bubble 3 / 4.7619%；DualPipeV 换算后 busy 60 / makespan 63.5 / bubble 3.5 / 5.5118%；DualPipe rank 1 稳态首组 `F0_6, B1_2, F1_4, B0_0` | **一致**（489 行、801-808 行） |
| `02_schedule_sim.py --sweep` | 30 组 (p,m,方法) 全部通过脚本内断言，闭式与实测一致 | **一致**（1339 行的验证范围声明成立） |

**对模拟器建模的独立审阅**（逐行对照 `dualpipe.py` / `dualpipev.py`）：

- **step 3/6/7 的 `enable_zb` 切换点**：模拟器 `02_schedule_sim.py:141-147`（DualPipe）与 `:202-208`（DualPipeV）的 `if i == step_6 // 2 and half_rank % 2 == 1 / == 0` 结构，与 `dualpipe.py:404-413`、`dualpipev.py:375-384` **逐字同构**（含「奇 rank 先翻转、偶 rank 后翻转」这一细节）✓；step 3、step 7 的 `enable_zb=True` 与官方一致 ✓。我另外独立推算了 step 6 的 zb 计数：无论 step_6 奇偶，每行恰好产生 `rank+1` / `half_rank+1` 个 zb 反向，与 step 8 的 pop 数相等 → `assert WeightGradStore.funcs_queue.empty()` 在真实代码里也成立 ✓。
- **`_weight_chunk()` 与 flush 的先后顺序**：模拟器 `_backward()` 在 `enable_zb=True` 时「追加到 cache 后立刻 flush」（对应 `_backward_compute_chunk` 末尾的 `WeightGradStore.flush()`），`_weight_chunk()` 做 FIFO `popleft` 并就地发 W 任务；调用位置与官方 `_weight_chunk()` 完全对齐（step3: `B(zb) → weight → F`；step7: `weight → B0(zb)`）✓。我按官方源码重算了 step3/6/7 的入队-出队次数，总入队 = 总出队，队列不会下溢 ✓。
- **跨 rank 依赖只挂 I 不挂 W**：`wire_dependencies` 第 3 段用 `b.inp`（I 任务表）建依赖，W 任务没有任何跨 rank 边 ✓；并且方向映射正确——方向 0 的 I 依赖 `rank+1`、方向 1 的 I 依赖 `rank-1`，与 DualPipeV 的 V 形映射（rank r 持有 stage r 与 2p-1-r）自洽 ✓；V 底部的本地交接（rank p-1 的 phase1 F 依赖本设备 phase0 F、phase0 的 I 依赖本设备 phase1 的 I）与 `dualpipev.py:79-80`、`:115-118`、`:171-172`、`:182-185` 的四处本地 append 对应 ✓。
- **未能找到反例**：p=2 与 p=6 的极端配置、m=2p 的下界配置都跑通且满足闭式；DualPipe 的 `is_middle_rank` 特判在任务级顺序上与非特判路径相同（F0→B1→F1→B0），因此模拟器忽略该分支不影响任务序列 ✓。
- 唯一保留意见：非 zb 反向的内部分工在官方示例里是「先 W 闭包、后 dgrad」（`examples/example_dualpipe.py:24-33`），模拟器统一成 `I→W`。这是文章明说的建模取舍，不构成会计错误（见 P2-8，建议在正文补一句）。

### 3. 源码行号核对

| 引用 | revision | 实际内容 | 一致？ |
| --- | --- | --- | --- |
| `dualpipe.py` 440 行 / `dualpipev.py` 411 / `utils.py` 80 / `comm.py` 38 / `example_dualpipe.py` 202 / `example_dualpipev.py` 183 | `030ce432…` | `wc -l` 全部相同 | ✓ |
| 入口断言：偶数设备、`num_chunks%2==0 and >= num_ranks*2`；DualPipeV 仅 `num_chunks >= num_ranks*2` | 同上 | `dualpipe.py:332-333`、`dualpipev.py:318` | ✓ |
| 八步注释与循环次数（step1 `(H-r-1)*2` … step8 `half_rank+1`） | 同上 | `dualpipe.py:358/363/373/381/398/404/415/421`、`dualpipev.py:330/335/344/352/369/375/386/392`，逐条相同 | ✓ |
| `_commit_and_wait_comm()` 位置与语义（`batch_isend_irecv` + 逐个 `req.wait()`） | 同上 | `dualpipe.py:285-292`、`_weight_chunk` 在 `:216-223` 先 commit 再 pop | ✓ |
| `WeightGradStore` 五个类方法 / `funcs_queue` / `put-flush-pop` 语义、`assert not empty` | 同上 | `utils.py:8-33`；文章第 883-909 行的代码块与该文件逐行相同（含 `clear`） | ✓ |
| `half_rank = min(rank, num_ranks - 1 - rank)` | 同上 | `dualpipe.py:335` | ✓ |
| `is_middle_rank` 取值 `p/2-1`、`p/2`；phase 注释原文；`phase ^= self.is_in_second_half` | 同上 | `:44-45`、`:355-356`、`:68/91/232/242/260/273` | ✓ |
| 第 4 步中间 rank 特判与 `NOTE` 注释原文 | 同上 | `:384-393` | ✓ |
| DualPipeV 本地交接（前向 `is_last_rank and phase==0`、反向 `is_last_rank and phase==1`）与「不断言偶数」 | 同上 | `:79-80`、`:171-172`、`:115-118`、`:182-185`、`:318` | ✓ |
| 更早提交 `3da1bbea53606543d7f5f232338fc58096db30e3` 与 HEAD 差异仅 `dualpipe/__init__.py` 的 `__all__` 10 行 | 同上 | `git diff --stat 3da1bbe 030ce432` = `dualpipe/__init__.py | 10 +++---`（5 增 5 删）；hash 存在且可解析 | ✓ |
| 重叠钩子契约：`overlapped_forward_backward` 为类方法、docstring 两句原文、示例体顺序执行 | 同上 | `dualpipe.py:23/168`、`example_dualpipe.py:59-62`、`:63-81` | ✓ |
| `comm.py` 38 行、`requires_grad=True`、形状 `(3,256,512)` + `float32` | 同上 | `comm.py:21-22`、`example_dualpipe.py:127-133` | ✓ |
| Megatron `combined_1f1b_schedule_for_interleaved_pipelining` 第 138 行、`combined_forward_backward_step` 第 281 行、docstring 第 164 行 | `core_v0.19.0` = `5be9626709af2722333bf54797c954c09edeada3`（`git show`） | 三条全部命中 | ✓ |
| `TransformerLayerSchedulePlan` 第 30 行、五个节点名与顺序、`_build_callable_nodes` 第 112 行、`delay_wgrad_compute` 第 133 行、stream 分配 | 同上 | `model_chunk_schedule_plan.py:30/38-43/54-58/112/133`；`:160`(comp)、`:163`(comp)、`:165`(comm)、`:166`(comm)、`:172`(comp，文章写「—」→ P1-8)；`:344-345` 的 `checkpoint_activations_microbatch` 约束存在 | ✓（除 stream 表单元格） |
| PyTorch `schedules.py` 3438 行；`ScheduleInterleaved1F1B` 2493、`ScheduleZBVZeroBubble` 2808、`ScheduleDualPipeV` 2994、`register_custom_function` 1887、`OVERLAP_F_B` 默认分支 2257-2260、`sub_actions` 构造 3093-3097（含 `FULL_BACKWARD`）、`OVERLAP_F_B = 11` | 本地 2.10.0 | `grep -n` 全部命中；2246 行的 `_comp_type_to_function_map` 分支确实排在 2257 行 `OVERLAP_F_B` 分支之前（文章结论成立） | ✓ |
| `_utils.py` 的 `generate_stage_to_rank_mapping` 第 91 行、`style=="v"` 分支 104-119 行、`dont change rank … (to keep v shape)` | 本地 2.10.0 | `_utils.py:91/104/113/119` | ✓ |
| `ScheduleDualPipeV` 的两条 `raise`（`n_local_stages != 2`、`n_microbatches < self._num_stages`） | 本地 2.10.0 | `schedules.py:3033`、`:3038` | ✓ |
| `_backward.py` 的 I/W 分离（`stage_backward_input` 143、`stage_backward_weight` 226） | 本地 2.10.0 | 存在 | ✓ |

**结论：文章给出的每一处源码行号都正确，没有发现一处错位**（P0-7 的 8 处错误全部属于「社区文章 / 教材页 / 论文 md 的本地副本引用行」，不是源码行号）。

### 4. 链接与 pin

| 检查项 | 结果 |
| --- | --- |
| `blob/main`、`blob/master`、`tree/main` 行号链接 | 0 处 ✓ |
| `sandbox:` 死链 | 0 处 ✓ |
| 裸外链图片（`![](http…)` 或 `src="http…"`） | 0 处 ✓ |
| GitHub 源码链接是否带 pin | 全部带 commit/tag：DualPipe `030ce432…`（6 个文件）、Megatron `5be9626709…`（tag `core_v0.19.0`）、sail-sg `c5d5074…` 与 tag `zero-bubble-v0.1.0` ✓ |
| tag 与 commit 对应 | `git rev-parse core_v0.19.0` = `5be9626709af2722333bf54797c954c09edeada3` ✓，与文章一致 |
| 版本分层一致性 | PyTorch 只以「本地安装版 2.10.0 + 路径」出现；`v2.9.0` 仅出现在两处**否定式声明**（「本文不引用上游 v2.9.0 或 main 的行号」），没有混入其行号 ✓ |
| 仓库内相对链接 | `./codes/*`、`./notes/`、`./notes/MATH-VERIFY.md`、`./pics/README.md`、`../deepep/deep-dive.md`、`../torch-distributed/readme.md`、`../nccl/readme.md`、`../fsdp2/readme.md` 全部存在 ✓ |
| 未 pin 的外链 | `https://github.com/deepseek-ai/profile-data`（仓库根，正文依赖其 README 的可变内容）→ P1-10 |
| hackmd 中文/英文两个 URL | 本地 md 自证英文版 `r1lVXsa9Jg` ✓；中文版 `S1N_ay0ckx` 无本地证据（见「无法核实」） |

### 5. 图片抽查

实际用 `read_image` 看过（含为核对读数而做的裁剪/放大）：**12 张正文用图 + 1 张对照原图**。全部 32 个 `src="./pics/…"` 目标文件存在，无缺失、无未使用文件（`comm -23` / `comm -13` 均为空）。

| 文件 | 读图结果 | 文章 alt/图注是否相符 |
| --- | --- | --- |
| `gpipe-fig2-naive-vs-microbatch-pipeline.jpg` | 三联图：(a) Device 0-3 朴素链，Loss 在顶、Gradients 在底；(b) 单 microbatch 阶梯 + Update 列；(c) F_{i,j}/B_{i,j} 密排 + 中部 `Bubble` 标注 + Update 列 | **相符**（222、225、228 行：(a)(b)(c) 的归属与「气泡基线只对应 (c)」都对） |
| `dualpipe.png` | Device 0-7 共 8 行；图例五项（Forward / Backward / Backward for input / Backward for weights / Overlapped forward & Backward = 橙绿拼接）；格内编号 0–9；有白色空格 | 图号、设备数、图例**相符**；但「双向各 20 个 microbatch」与编号只到 9 不符 → **P0-5** |
| `dualpipev.png` | Device 0-3 共 4 行，格内编号 0–9，Device 0 与 Device 3 序列不同 | **相符**（612-618 行：「4 PP rank（8 逻辑 stage）、10 个 micro-batch」✓） |
| `zero-bubble-fig3-handcrafted-zb-h1-h2.jpg` | 上栏 ZB-H1：前导 F 4/3/2/1，白格 3/2/1/0；下栏 ZB-H2：前导 F **7/5/3/1**，行内无白格；两栏图例 F/B/W/Optimizer step；下栏米色 optimizer 块逐行错位（列 24/25、25/26、26/27、27/28） | 上栏描述**相符**；下栏 F 数**不符**（P0-3）；「米色 optimizer 逐行错位」**相符**；「平行四边形」与论文 §2.2 文字一致 |
| `zero-bubble-fig8-zbv-schedule.jpg` | Device 1-4 共 4 行；每格单色单格、格内单个数字；数字有白/黑两种颜色；米色 optimizer 块错位 | 「4 个 worker」**相符**；「每格上下两行对应两个 chunk」**不符**（P0-4）；图例实际含 Optimizer step，alt 未写（P1-8） |
| `controllable-memory-fig2-parallel-vs-vshape.jpg` | 左 Parallel：上/l1+l4、中/l2+l5、下/l3+l6；右 V-Shape：l1+l6、l2+l5、l3+l4 | **相符**（559、562 行；与论文 md 第 73 行「parallel 是 l1+l4，V-Shape 是 l1+l6」一致） |
| `sea-ai-lab-cut-in-half-vshape.jpg` | 上栏 DualPipe：Device 0-7，layers (0,7)(1,6)(2,5)(3,4)…；下栏 Cut-in-half：Device 0-3，同样层配对；图面带「知乎 @庞天宇」水印 | 内容描述**相符**；署名未提转载者（P2-6） |
| `terapipe-fig1d-token-pipeline.jpg` | 4 条整块条带 + 2 条弯箭头（GPipe microbatch 型），**无 token 分段**；与对照图 `a73bba64…`（条带被切成多段、大量细箭头）相比明显不是 1(d) | **不符**（P0-2） |
| `megatron-fig4-default-vs-interleaved-1f1b.jpg` | 上栏 4 行粗粒度 F/B + 灰格 idle；下栏同样 4 行但格内多个 chunk 编号交错、idle 变少；中部箭头注 `Assign multiple stages to each device`；图例 Forward/Backward | **相符**（1084 行附近 alt + 正文「default 1F1B 与 interleaved 1F1B 时间轴对照、气泡显著变小」） |
| `pipedream-2bw-fig2-2bw-timeline-two-weight-versions.jpg` | Worker 1-4；棋盘格绿格标新权重版本；右上 `Before/After W_1^(0)→W_1^(4)`、右下 `W_4^(0)→W_4^(4)`、`t = 21` | **相符**（1310 行 alt）——但该行引用的「第 66 行」是真 Figure 3 的图注（P0-7） |
| `deepseek-v3-fig5-dualpipe-schedule-8pp-20mb.jpg` | 8 行 Device 0-7、编号 0–9、图例五项 | 图号/图例**相符**；「双向各 20 个」**不符**（P0-5） |
| `zero-bubble-fig2-1f1b-schedule.jpg` | Device 1-4，只有 Forward（蓝）/Backward（橙）/Optimizer step（米色）三色，无 W 格；1026×159 | **相符**（271-277 行：「没有任何 W 格」「与 Figure 3 是两个独立文件（1026×159 与 1032×310）」都对） |

### 6. 禁止项与过度声称

| 检查项 | 结果 |
| --- | --- |
| ASCII 字符画（`┌┐└┘│├─`） | **0 处** ✓（流程图全部 Mermaid/表格；仅 Figure 4 的 alt 里出现 `△/▲` 用于描述图例符号，不属于字符画） |
| 全角引号 `“”` | **0 处** ✓（统一用「」） |
| `——` 出现次数 | **1 处**（1302 行）✓ 未超过 1 |
| `[Pending Review]` 当作知识来源 | 未发现 ✓；1365 行只是把 `rlhf/sys-design/readme-4.md` 的 `[Pending Review]` 标记作为**排除理由**写出，且我核实：`README.md:150` 确为 `[Pending Review] [Expert Parallelism](./rlhf/sys-design/readme-4.md)`，`knowledge-graph.json` 里该条目 `"status": "published"` → 文章所指的「标记与 published 状态矛盾」**确实存在**，报告属实 |
| 「本文实测」式过度声称 | 未发现 GPU 类声称 ✓。开篇第 11 行与文末 1345 行都三次声明「本文没有任何 GPU 实测，也没有跑过官方示例」；正文中「实测/本机实测」只用于 `codes/` 的两个 CPU 脚本（205、757、772、1338 行），且 1328 行的表格明确写「不是硬件性能测试」，846 行明确写 `C` 只能由执行器与实测决定 ✓ |
| 性能数字是否都标了来源与配置 | 外部数字均标注来源（论文图号/表号 + 本地副本路径 + arXiv 版本）；第七章两张表都在表头/表下写明基准（「同一 stage 数」vs「同设备数、同模型、同 microbatch 数」）与前提假设（等时、均匀划分、忽略通信）✓ |

---

## 我无法核实的事项

1. **外部 URL 的可达性**：本次审查没有联网核对 4 个 hackmd / 知乎 / HF blog / aiinfra 页面是否仍然存在（环境未使用 `web_search`，避免引入不可复现的检索结果）。可本地自证的只有：Sea AI Lab 英文版 URL 与其本地 md 第 10 行一致；知乎与 HF 两个 URL 无本地快照可交叉验证。
2. **`references/community/easy-dualpipe/`**：文章声明「没有运行它，也没有用它作为事实来源」，我同样没有运行（其中含 3 个 `.py`），因此无法判断它是否与正文结论冲突。
3. **`profile-data` README 的自述内容**：仓库本地无副本、无 pin，文章 1263 行关于「模拟完全均衡的专家路由、trace 省略 PP 通信」的说法我无法核实（只能确认文章已声明其边界）。
4. **Figure 3 上栏 ZB-H1「W 在尾部把白格填掉」的逐格对应**：我用像素分类能确认白格数量（3/2/1/0）与后续出现 W 格，但无法逐格证明「填的正是那几个位置」；该结论本身与论文 §2.1 文字（"the tail-end bubbles are filled by the later-starting W passes"）一致。
5. **§4.1 的 ZB-H1 配对公式**：论文正文/附录没有给出该式子，只能与 Figure 3 的格子编号做抽查（i=0、i=3 两行一致），无法逐项与来源对照。
6. **DualPipe 官方调度图里「被同一个黑框圈住的两格」**：`pics/dualpipe.png` 在 2048 px 缩放下两格重叠格渲染为橙绿拼接格，黑框边界无法在现有分辨率下逐一确认；该说法与官方 README 与论文 Figure 5 图注文字一致，故按「来源一致」处理。
7. **并发修改**：审查期间文章与 `notes/` 被同时修改（新增 `MATH-VERIFY.md`、`IMAGE-VERIFY.md`，并有若干行被就地订正，例如 §8.4 的 `WeightGradStore` 代码块已从删节版补成与 `utils.py` 逐行一致）。我先后在 md5 `2849ca2a…`（1350 行）、`e1289c54…`（1368 行）、`9174861c…`（1368 行）三版上复验，本报告的 P0/P1/P2 在三版上均成立；行号对应 `9174861c…`。若在此之后文章再变动，行号需要重新对齐（结论本身对应的是文本内容，不受行号漂移影响）。
