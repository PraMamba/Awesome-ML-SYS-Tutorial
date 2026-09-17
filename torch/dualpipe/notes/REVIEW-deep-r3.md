# DualPipe 深度解析：第三轮深度审核报告

> 审核对象：`torch/dualpipe/deep-dive.md`，**我读取时的行数**：1372，**md5**：`30dac685147e335da31a79354e2091f2`
> （行号只作定位线索，引用时请以小节号与原文内容为准。本节末尾附完整 provenance。）

## 审核开始时的环境记录

```text
$ pwd
/workspace/algorithm/Awesome-ML-SYS-Tutorial

$ git status --short --branch
## main...origin/main
 M transformers/kda_linear_attention/kda_linear_attention-deep-dive.md
?? .agent-teams/
?? rlhf/agent-harness-rl/
?? torch/deepep/
?? torch/dualpipe/
?? torch/parallel_dims_lab/PROMPT-learn-review-v2.md
?? transformers/compressed_sparse_attention/…（下略，共 38 条 untracked）
（注：`transformers/kda_linear_attention/kda_linear_attention-deep-dive.md` 的 `M`
 与 `torch/dualpipe/` 整目录的 untracked 状态都是本轮之前就存在的，本轮未触碰。）

$ git -C /workspace/algorithm/DualPipe rev-parse HEAD
030ce4325f4ebeb437da4ebc6d00a70469dd58ae
（与审核基线一致；该 revision 同时是本地 clone 的 origin/main）

$ git -C /workspace/algorithm/Megatron-LM rev-parse HEAD
0d7ecc5d6117ba8eae08d0bd2c7e2a0d9d28f3d6
$ git -C /workspace/algorithm/Megatron-LM rev-parse core_v0.19.0
5be9626709af2722333bf54797c954c09edeada3   ← 与任务书一致，本轮一律用 git show core_v0.19.0:<path> 读取

$ python -c "import torch; print(torch.__version__)"
2.10.0+cu129

$ wc -l torch/dualpipe/deep-dive.md
1372 torch/dualpipe/deep-dive.md          ← 注意：不是任务书里写的「约 1373 行」
$ md5sum torch/dualpipe/deep-dive.md
30dac685147e335da31a79354e2091f2  torch/dualpipe/deep-dive.md
```

---

## 总评

**这篇文章的主体已经达到可发布水平，而且质量高于前两轮记录的印象：我逐式重推了全部 display 公式（含四组气泡率恒等式、Table 1/2、worker 激活公式、官方表换算），逐行核对了 DeepSeek/Megatron/PyTorch 三个 revision 的全部行号与代码摘录，四条命令实跑的输出与文章数字逐字对上，32 张图的元数据行号用 sha256 反查脚本全量复核、30/30 全对，并独立读了 17 张图。这些部分我**没有发现任何 P0**。**

**但第 2 轮「已全部处理」的自述有两处不成立**，而且这两处恰好落在第 2 轮自己定性为 P0 的那两类（读来源、可复现性）里：`codes/README.md` 声称的第一组稳态 chunk 序号与脚本实际输出相矛盾；`notes/MATH-VERIFY.md` 仍把「PP 通信翻倍」记在官方 README 名下——这正是第 2 轮 P0-6 判定为错误、并在文章里删掉的那句话。两处都在文章正文之外，但都在文章自己指定的复核链路上，所以我按 P0 计。

**第 2 轮 P1-1（「串行执行」一词两义）并未修干净**：文章 §7.3、`codes/README.md`、`notes/MATH-VERIFY.md` 三处仍用「串行执行」指代模拟器模型，而 §7.4 又把「配对块串行执行」定义为恰好相反的那一端（气泡差一倍）。此外我发现三处可核算但算错的数字（图 18 尺寸 743→742、Table 9 偏差范围 14%–17% → 13.74%–18.14%、`pics/README.md` 的 453/31 未随正文同步），以及一处**两轮都漏掉**的重要事实：`rlhf/sys-design/readme-4.md` 在 README.md 里有**两条互相矛盾**的条目（第 51 行 published、第 150 行 `[Pending Review]`），文章只报告了后者，结论因此偏严。

总评：**修完下面 2 条 P0 与 5 条 P1 即可发布**；数学、源码行号、图片装配与 pin 纪律这三块可以直接签字。

---

## P0

> 说明：**文章正文本身本轮没有发现新的 P0。**以下两条都落在 `deep-dive.md` 之外、但被文章明确指定为复核链路的文件上，且都是第 2 轮声称已修而实际未修/未同步的项。按「读者顺着文章给的指针就能读到错误断言」这一标准计为 P0。

| # | 位置（文件 + 行号） | 问题 | 证据 | 建议改法 |
| --- | --- | --- | --- | --- |
| P0-1 | `codes/README.md` 第 78 行（§「额外核对」第 2 条） | 写成「第一组稳态工作的 chunk 序号与草稿公式逐项吻合，例如 `p=8, r=1` 时 DualPipe 给出 `(F0_8, B1_4, F1_6, B0_2)`」。**这不是脚本对 p=8,r=1 打印的第一组稳态工作**，而且与文章 §5.4 的结论（第一组是 `(F0_6, B1_2, F1_4, B0_0)`，x 由 warmup 决定、不是自由参数）直接矛盾 | 实跑：`python torch/dualpipe/codes/02_schedule_sim.py --p 8 --m 20` 输出 `rank 1: F0_6, B1_2, F1_4, B0_0`。`(F0_8,B1_4,F1_6,B0_2)` 是把 x=8 代进闭式得到的**示例值**（文章 §5.4 明确写它只是「取 p=8、r=1、x=8」的代入演示），不是脚本的 steady_probe 输出 | 改写为「`p=8, r=1` 时脚本打印的第一组稳态工作是 `(F0_6, B1_2, F1_4, B0_0)`，与 warmup 计数推出的 x=6 一致；把 x=8 代入闭式另得 `(F0_8, B1_4, F1_6, B0_2)`」 |
| P0-2 | `notes/MATH-VERIFY.md` 第 101 行（§2.5 表末第 4 行「PP 边界通信约 `2x`」） | 「来源」列写作「**官方 README 与 Sea AI Lab 博客**」。这正是第 2 轮 P0-6 判定为错误、并从文章中删去的那句话（`REVISION.md` 第 131 行自述：「README 全文没有任何关于 PP 通信量或 PP vs EP 开销的表述」）。文章修了，逐式复核记录没修，而文章附录 B 又把这份记录作为「第三方复核入口」 | `grep -in comm /workspace/algorithm/DualPipe/README.md` 仅命中第 3 行（`full overlap of forward and backward computation-communication phases`）与第 12 行（`have mutually overlapped computation and communication`），**没有任何 PP 通信量的表述**；文章自身第 643 行也已写明「官方 README 只给出比较表与致谢，没有对 PP 通信量做任何定量或定性的表述」 | 把该行来源改为「Sea AI Lab cut-in-half 博客的对照表与结论（arXiv/README 无对应表述）」 |

---

## P1

| # | 位置 | 问题 | 证据 | 建议改法 |
| --- | --- | --- | --- | --- |
| P1-1 | `deep-dive.md` §7.3（第 774 行）+ `codes/README.md` 第 87 行 + `notes/MATH-VERIFY.md` 第 127 行 | **第 2 轮 P1-1 没修干净。** §7.3 仍写「本文这套**「串行执行 + 依赖等待」的离散模型**，恰好等价于官方公式里 \(F\&B=\max(F,B)\) 那一端」；`codes/README.md` 写「它给出的是**「串行执行 + 依赖等待」下的气泡**，是重叠收益的上界参考」；`MATH-VERIFY.md` 写「它给出的是**串行执行**下的气泡」。而 §7.4（第 801 行）刚刚定义「**「配对块串行执行」**指的是执行器把 F 与 B 顺序跑完（\(F\&B=F+B_{\mathrm{full}}\)）」，并且明确说模拟器「落在**配对块重叠成立**那一列，不是『配对块串行执行』那一列」。同一个词在两节里指相差一倍的两端 | 模拟器实跑 p=8,m=20 → 气泡 3（\(=(p-2)t/2\)）；若真按「配对块串行执行」应为 6（\(=(p-2)t\)）。`codes/README.md` 同一文件第 73 行还写着「若按串行执行取 \(F\&B=F+B_{\mathrm{full}}=3t\)，同两个气泡值都要翻倍」——与第 87 行自相矛盾 | §7.3 改为「本文这套**「设备内任务串行 + 跨 stage 依赖等待」**的离散模型」（或直接说「不建模内部重叠的依赖模型」）；`codes/README.md` 第 87 行、`MATH-VERIFY.md` 第 127 行同步 |
| P1-2 | `deep-dive.md` 第 1358 行（自动检查报告注释）与 `learn-plan.md` 第 13 行 | 自述不实 + 与计划不符。注释写「驱动问题（『这段独立计算从哪里来』）在**第一章末尾**、读者同时握有**气泡公式**与 MoE 通信结构之后出现」。第一章**没有任何气泡公式**（\(n_{\mathrm{warmup}}=\min(p-r-1,m)\)、\((m+p-1)(F+B_{\mathrm{full}})\)、\(\beta_{\mathrm{1F1B}}\) 全在第三章）；`learn-plan.md` 第 13 行原话是「驱动问题放在**第三章末尾**、DualPipe 的自我介绍之前出现——那正是读者同时握有『气泡公式』和『MoE 通信结构』两块背景的位置」，即计划本身就要求放在气泡公式之后。文章实际放在 §1 末尾（第 87 行），而该问句的措辞「调度图上的稳态区域可以被 F、B 方块填满」预设了第三章才引入的稳态与 F/B 方块概念 | `grep -n "beta_{\\\\mathrm{1F1B}}\|n_{\\\\mathrm{warmup}}" deep-dive.md` 只命中第 243/315 行（均在第三章）；§1 只有定性描述。驱动问题在文中被回收 3 次（第 406、468、1155 行），回收本身没问题 | 二选一：①把驱动问题移到 §3.4 末尾（符合计划，但会削弱第五章的悬念）；②保留在 §1 末尾，但把第 1358 行的自述改为「读者同时握有『两类等待的定性区分』与 MoE 通信结构」，并在 `learn-plan.md` 里把该句要求改成实际落点。**我倾向 ②**，因为 §1 末尾的两类等待对照表已经给出足够背景，真正要修的是自述与计划文本 |
| P1-3 | `deep-dive.md` 第 1369 行（自动检查报告注释） | 「该文（`rlhf/sys-design/readme-4.md`）**在 README.md 中标记为 `[Pending Review]`**」——**不完整，会误导。** README.md 对同一篇文章有**两条互相矛盾的条目**：第 51 行 `[Deep Dive into DeepSeek MoE with Classic Secondary Development of EP on FSDP](./rlhf/sys-design/readme-4-en.md)`（**无**标记，并链到中文版），第 150 行 `[Pending Review] [Expert Parallelism](./rlhf/sys-design/readme-4.md)`（**有**标记）；而 `README-cn.md` 第 64 行 `[深入浅出 DeepSeek MoE，EP 与 FSDP 经典二次开发](./rlhf/sys-design/readme-4.md)` **无标记** | `grep -n "sys-design" README.md README-cn.md` 输出如上；`knowledge-graph.json` 该条目 `status: published`。文章正文并未引用该文（`grep -n "Pending Review" deep-dive.md` 只命中第 1369 行这一处注释），所以对正文知识无影响 | 把该句改为「该文在 `README.md` 第 150 行有一条 `[Pending Review]`、但同一文件第 51 行与 `README-cn.md` 第 64 行都按 published 列出，README 内部自相矛盾；本文因此按最保守处置不引用，并只作报告」。见「争议项裁决」第 3 条 |
| P1-4 | `pics/README.md` 第 5、76、77 行；`notes/IMAGE-VERIFY.md` 第 7、75 行 | **计数未随正文同步。** 正文已统一为「433 张写作素材图（论文 353 + 教材页 20 + 社区文章 60）」（第 1286、1332 行，与 `learn-plan.md` 第 5、230 行一致），但 `pics/README.md` 第 5 行、第 77 行仍写 **453 张**，`IMAGE-VERIFY.md` 第 7、75 行仍写 **453 张**；`pics/README.md` 第 76 行仍写「**31 张**图之外」（第 19、72 行同文件已写 32） | 实际清点：`references/` 下图片文件 **435** 个 = papers 353 + docs 20 + articles 60 + community 2；433 = 前 353+20+60 三项（写作素材）。`pics/` 下非 README 文件 **32** 个（`comm` 双向比对与正文 32 个引用完全一致）。第 2 轮 P1-5 只改了正文与 `learn-plan.md`，未改这两个文件 | 两处统一为 433（或明确写「435 个文件，其中 433 张为写作素材、2 张为 `community/easy-dualpipe/` 仓库自带图」）；`31` 改 `32` |
| P1-5 | `learn-plan.md` 第 249 行 | 括号内列举的 12 个目录图片数逐个正确（与 `ls` 实测完全一致），但**「合计 435」是错的**：`88/31/49/26/32/9/22/30/32/34/20/60` 逐项相加 = **433**；同一句后半又说「另有社区复现代码目录 2 张**未纳入**逐张核对」，即这 2 张不应进合计 | 实际：papers 353（=88+31+49+26+32+9+22+30+32+34）+ docs 20 + articles 60 = 433 ✓，额外 community 2 张 = 435（总文件数）。`REVISION.md` 第 142 行又声称 P1-5 已改为「435 张（论文 353、教材页 20、社区文章 60、社区复现代码 2）」，与正文实际的 433 口径**又不一样**——同一轮的三份文件出现三个口径 | `learn-plan.md` 第 249 行「合计 435」改为「合计 433」；`REVISION.md` 第 142 行的自述改为与正文一致的口径 |

---

## P2

| # | 位置 | 问题 | 证据 | 建议改法 |
| --- | --- | --- | --- | --- |
| P2-1 | `deep-dive.md` §7.4（第 787 行表「PP 边界通信」列，DualPipeV 行的 `/ 约 2x`） | 表里 `/ 约 2x` 这一列本身正确（第 797 行也正确注明「这一条来自 Sea AI Lab 博客的对照表，官方 README 没有对应表述」）；**但同一论断在 `notes/MATH-VERIFY.md` 第 101 行与 `learn-plan.md` 第 249 行的来源列里仍带 README**，与正文口径不一致 | 见 P0-2 的 grep 证据 | 只改那两份配套文件，正文无需动 |
| P2-2 | `deep-dive.md` 附录 A（第 1298 行图注） | 「这是一整幅画廊图（**743×1723**）」 | 实读 `PIL.Image.open(...).size` = **(742, 1723)** | 改为 742×1723；对照：第 277 行的「1026×159 与 1032×310」我实测完全正确 |
| P2-3 | `deep-dive.md` §2.2（第 162 行） | 「其余配置的偏差在 **14% 到 17%** 之间」。逐行复算 Zero Bubble Table 9：1.5B 三行为 25.95%/25.97%/26.08%；其余 9 行为 **13.74%（28.3B/m=256）～18.14%（6.2B/m=64）**，上下界都越界 | 我用 `(2T_F-(T_B+T_W))/(2T_F)` 逐行算：6.2B 三行 16.93/17.86/18.14；14.6B 三行 14.60/14.41/14.27；28.3B 三行 13.99/13.97/13.74。核心论断（26%、且出现 \(T_B<T_F\)）**正确** | 改为「其余配置的偏差约在 14% 到 18% 之间」。（这是第 2 轮 P2-2 补写的那句话，属于该条修复的残留偏差） |
| P2-4 | `deep-dive.md` §8.4（第 884 行） | 「`put` / `flush` / `pop` 三个方法只有 **15 行**」 | `utils.py` 第 14–28 行 = 15 行，但其中含 2 行空行分隔；三个方法本体（含 `@classmethod`）= 13 行（put 3 + flush 4 + pop 6）。「第 8 到 33 行的 26 行」我实测为 33−8+1 = 26 ✓ 正确 | 改为「三个方法占第 14–28 行」或「13 行」 |
| P2-5 | `deep-dive.md` §8.7（第 1016 行） | 「钩子的 docstring **只有一句**：『You should implement custom forward-backward overlap strategy. The code below is just an example.』」 | `examples/example_dualpipe.py` 第 66–69 行是**两句、两行**。引文本身逐字正确（只把换行接成了空格） | 改为「只有两句话」；或在引文里保留换行 |
| P2-6 | `deep-dive.md` §7.3（第 774 行） | 「DualPipe 与 DualPipeV 的气泡**恰好是 ZB1P 的一半**」。对 DualPipeV 成立（\((p-1)/2\) 对 \(p-1\)）；对 DualPipe **不成立**（\((p-2)/2\) 对 \(p-1\)，不是精确一半），\(6m\) 分母其实来自通分时清掉分子的 \(1/2\)，与「是否恰好一半」无关 | \(p=8\)：ZB1P \(=7t\)，DualPipe \(=3t\neq 3.5t\)，DualPipeV \(=3.5t\) ✓ | 直接写推导：「气泡率 = Bubble/(3mt + Bubble)；DualPipe 的 Bubble 带 \(1/2\) 因子，乘 2 通分即得首项 \(6m\)」。这正是第 2 轮 P2-3 想修的那处「解释牵强」，本轮看**换了说法但没换掉牵强** |
| P2-7 | `deep-dive.md` §3.4（第 318 行起）与 §7.4/§7.5 | 「\(T_{\mathrm{1F1B}}=(m+p-1)(F+B_{\mathrm{full}})\)」的来源被正确标注为「论文附录在 \(m\le p\) 的粗略分析里」（前提**带出来了**，这点比预期好），但 §7.4/§7.5 随后在 \(p=8,m=20\)（即 \(m>p\)）使用它，没有说明该式对任意 \(m\ge1\) 都成立、论文只是在一个更窄的前提里用了它 | 论文附录 H（`zero-bubble.md` 第 348 行）原文：「assuming \(m<=p\) and \(T_W<T_B\), an 1F1B iteration takes \((m+p-1)*(T_F+T_B+T_W)\)」 | 加一句「该式对任意 \(m\ge1\) 都成立，论文只是在 \(m\le p\) 的粗略分析里使用它」 |
| P2-8 | `notes/IMAGE-VERIFY.md` 第 33、35 行 | 「文章所述」列仍写 `dualpipe.png` / `deepseek-v3-fig5` 是「**两个方向各** 20 个 micro-batch」——这是第 2 轮 P0-5 判定为错误、并在文章中改成「双向合计 20 个（每方向 10 个）」的旧说法。同一行「读图结果」列写的是「两个方向各 0-9」，与结论列自相矛盾 | 文章第 495、498、501、506 行现在都写「双向合计 20 个（每方向 10 个）」；官方 README 第 9 行原文 `8 PP ranks and 20 micro-batches in two directions` | 该列改写为「双向合计 20 个（每方向 10 个）」 |
| P2-9 | `notes/MATH-VERIFY.md` 第 93 行 | 「模拟器实测：\(p=2,4,6,8,10\) → \(0,1,2,3,4\)」。内置 `--sweep` 只覆盖 \(p\in\{2,4,6,8\}\)（`02_schedule_sim.py` 第 410 行），\(p=10\) 需要手工 `--p 10` | 我实跑 `--sweep` 确认覆盖面为 {2,4,6,8}×{8,10,16,20,32}；(p−2)/2 对 p=10 的算术本身正确 | 注明 p=10 是手工单跑，或把 p=10 加进 `--sweep` |
| P2-10 | `notes/MATH-VERIFY.md` 第 99 行 vs `deep-dive.md` 第 774 行 | 对 \(3m/6m\) 给了**两套不同解释**：`MATH-VERIFY` 说「ZB1P 无配对块 → 3m；DualPipe/DualPipeV 有配对块 → 6m」，文章说「气泡是 \(O(p)\) 量级…DualPipe 的气泡恰为一半，通分后首项变成 6m」。两套都不是严格推导 | 严格推导见 P2-6 | 两处统一为同一句推导 |
| P2-11 | `deep-dive.md` §5.5（第 495 行） | 「**两个被同一个黑框圈住的格子表示这一段计算与通信是相互重叠的**」是在转述 README 第 11–12 行（`Two cells enclosed by a shared black border have mutually overlapped computation and communication`），但在这张图的可用分辨率下我**没能确认画面里存在这种共享黑框**，画面表达重叠的方式是图例里的橙绿拼接格（`Overlapped forward & Backward`）。第 1 轮已把该说法从 alt 移出，但正文里这句仍以陈述句出现 | 我读图结果与 `IMAGE-VERIFY.md` 第 34 行一致（该行也承认「画面本身以橙绿拼接格表达重叠」） | 加限定：「README 对这张图的说明是『两个被同一黑框圈住的格子表示…』（画面以橙绿拼接格表达同类含义）」 |

---

## 第 2 轮 7 项修复的独立核验

| # | 声称的修复 | 我的核验方式 | 结论 |
| --- | --- | --- | --- |
| 1 | ZB-V 归属改为 Zero Bubble §6/Figure 8，并说明 Controllable Memory 更晚 | 读 `references/papers/zero-bubble/zero-bubble.md` 第 156（`## 6 MEMORY EFFICIENT ZERO BUBBLE SCHEDULE`）、159（Figure 8 图注）、161–165 行：ZB-V 的定义、\(2p\) chunk、16 层例（「layers 1-2 and layers 15-16 to worker 1, layers 3-4 and layers 13-14 to worker 2」）、「forward and backward pass … originate from the same worker」、「Under the condition \(T_F=T_B=T_W\), ZB-V achieves zero bubble with a peak activations memory of \(pM_B\)」全部在 Zero Bubble §6；读 `controllable-memory.md` 第 82、107 行确认 V-Min/V-Half/V-ZB 三个名字出自该文，且该文正文不出现「ZB-V」。arXiv：2401.10241（Zero Bubble）早于 2405.15362（Controllable Memory） | **已修好。** 文章第 557 行的每一处归属与论文原文逐句对得上 |
| 2 | `terapipe-fig1d-token-pipeline.jpg` 从 1(c) 换成真正的 1(d) | `sha256` 比对：该文件 = `references/papers/terapipe/images/a73bba64824371103963da88558e60812c9ebcfbdce52ff1aee1353e24f435ae.jpg` ✓；`read_image` 确认画面为 Device 5→Device 1 五条 Transformer layer 条带 + 条带下方细密 token 分段 + 跨设备橙色细箭头（token 级），**不是** GPipe 式 microbatch 流水线；并核对 `terapipe.md` 第 28 行引用确为 1(d) | **已修好** |
| 3 | Figure 3 下栏 warm-up F 数 7/6/5/1 → 7/5/3/1 | 自己写像素分类脚本（PIL + 最近邻色分类，`x0=72`、pitch `24.4`，取每格上边线下 3–8 px、水平靠左 8 px 的区域取中位色以避开数字字形；图例色实测 F=(66,133,245)、B=(70,188,198)、W=(52,168,85)、OPT=(255,242,205)）。结果：上栏前导 F = **4/3/2/1**（各行前置位移 0/1/2/3 空格），位移后白格 = **3/2/1/0**；下栏前导 F = **7/5/3/1**，白格 = **0/0/0/0**。自证关系 **7=4+3、5=3+2、3=2+1、1=1+0 全部成立** | **已修好，且自证关系成立。** 注意核对文件 `notes/IMAGE-SURVEY-zero-bubble.md` 第 54、80、87 行**三处仍是 7/6/5/1**（见争议项 1） |
| 4 | Figure 8 的 alt 改为「单行网格 + 格内数字用白/黑区分两个 chunk」 | 读论文第 159 行图注原文：「Each device is assigned to exactly 2 chunks, where **white text colors** represent the first chunk and **black text colors** represent the second chunk」；`read_image` 画面：4 个 Device 各**一行**时间网格，同一行内确实同时存在白字与黑字数字（如 Device 2 行中段的 `6 2 1 8` 为白字，其余为黑字），图例为 F（蓝）/B（青）/W（绿）/Optimizer step（米色），米色格逐 Device 右移错位 | **已修好** |
| 5 | 三处「双向各 20 个 micro-batch」→「双向合计 20 个（每方向 10 个）」 | 官方 README 第 9–12 行原文 `8 PP ranks and 20 micro-batches in two directions`；`examples/example_dualpipe.py` 第 118 行 `num_chunks = 20`、第 146/149 行 `full_x.chunk(2)[0] / [1]`、`dualpipe.py` 第 336 行 `half_num_chunks = num_chunks // 2`；`read_image` 确认 `dualpipe.png` 与 `dualpipev.png` 格内编号最大均为 **9** | **已修好**（`pics/README.md` 与 `IMAGE-VERIFY.md` 的对应表述未同步，见 P2-8） |
| 6 | 两处删去「官方 README 与 Sea AI Lab 博客都提到」PP 通信翻倍 | `grep -in comm /workspace/algorithm/DualPipe/README.md` → 只有第 3 行与第 12 行，均与 PP 通信量无关；文章第 643 行已删并显式写出「官方 README 只给出比较表与致谢，没有对 PP 通信量做任何定量或定性的表述」，第 797 行同样只引博客 | **文章已修好；`notes/MATH-VERIFY.md` 第 101 行未同步（P0-2）** |
| 7 | 八处「本地副本第 N 行引用图」行号改正 | 全量脚本：对 `pics/` 下 32 张图逐个算 sha256 → 在 `references/` 全部 435 个图片文件中反查源文件 → 在源 md 里用 `![]()` / `<img src>` 正则定位真实引用行号 → 与文章 30 条「本地副本…第 N 行引用图」逐条比对 | **30/30 全部通过，0 处残留。** 该条修复彻底 |

---

## 争议项裁决

### 1. `notes/IMAGE-SURVEY-zero-bubble.md` 里的 `7/6/5/1` 未订正，是否可接受？

**不可接受，建议就地订正（最小改动是加行内批注，最好直接改值）。**

理由：
- 这份文件不是普通记录，而是**错误的发源地**：第 2 轮 P0-3 的成因正是「文章照抄了核对文件的错误读数」。同一轮把文章改对、把源头留着，等于把同一个坑保留给下一轮——本项目已经有先例（`IMAGE-SURVEY-zero-bubble.md` 第 87 行还在指导写作「warm-up 的 F 数从 `4/3/2/1` 增到 `7/6/5/1`」，任何按核对文件写作的会话都会再次写错）。
- 「不在本轮允许改动的范围内」是本轮**自己设定的范围**，不是不可抗力。第 2 轮已经改过 `learn-plan.md`（同属交付物），说明范围本身就是可调的。
- `notes/IMAGE-VERIFY.md` 第 40 行（`zero-bubble-fig3` 那一行的结论列）确实写了「核对文件 `IMAGE-SURVEY-zero-bubble.md` 的同一处仍是旧值，引用时以本条为准」，但那是**反向指针**：读者/agent 打开核对文件时看不到它，只有先读 IMAGE-VERIFY 才知道要去覆盖。
- **建议写法**：在第 54、80、87 行三处的 `7/6/5/1` 后就地改为 `7/5/3/1`，并加「（第 3 轮像素复核订正：上栏 F 数 + 白格数 = 下栏 F 数）」；若坚持不改核对文件，则至少在这三行行首加 `> ⚠️ 已作废`。核对文件的「不可变性」不该凌驾于「不得留存已知错误读数」之上。

### 2. `../deepep/deep-dive.md` 的引用合规性

**当前处理方向正确但不够，建议补一句边界说明，而不是删链接。**

- `.learn/config.md` 要求内部引用指向 published 状态的文章；`torch/deepep/deep-dive.md` 确实不在 README.md / README-cn.md 的索引里（`grep -i deepep README.md README-cn.md` 无输出），文件存在 ✓，链接可解析 ✓。
- 文章第 7 行给的括号说明只交代了「同期写作，尚未进入两个 README 的索引」，**没有说清「因此不作为知识来源」**；而文末注释第 1364/1369 行才写出「按仓库规则暂不作为知识来源，仅作同期姊妹篇链接」。对只读正文的读者来说，边界是缺失的。
- **裁决**：删链接是过度反应（读者确实需要 EP 那一层的前序背景，而文件真实存在）；但正文必须把边界写在读者看得到的地方。建议第 7 行括号改为「（该篇与本文同期写作，尚未进入两个 README 的索引，因此本文只把它当作同期姊妹篇链接，不引用其结论作为知识来源）」。这样既满足 config 的「不作为知识来源」，也不产生死链。

### 3. `rlhf/sys-design/readme-4.md` 的 `[Pending Review]` 与知识图谱 `published` 的矛盾

**文章的处置（不引用 + 只报告）方向正确，但报告的事实基础不完整；矛盾本身应由仓库主人改 README，不应由本文作者改知识图谱。**

核查结果（这是本轮新发现，前两轮与任务书都没提到）：

| 文件 | 行 | 条目 | 标记 |
| --- | ---: | --- | --- |
| `README.md` | 51 | `[Deep Dive into DeepSeek MoE with Classic Secondary Development of EP on FSDP](./rlhf/sys-design/readme-4-en.md)`（并链到中文版 `readme-4.md`） | **无标记** |
| `README.md` | 150 | `[Expert Parallelism](./rlhf/sys-design/readme-4.md)` | **`[Pending Review]`** |
| `README-cn.md` | 64 | `[深入浅出 DeepSeek MoE，EP 与 FSDP 经典二次开发](./rlhf/sys-design/readme-4.md)` | **无标记** |
| `knowledge-graph.json` | — | `sys-design-moe-ep-fsdp`, `path: rlhf/sys-design/readme-4.md` | `status: published` |

所以真正的矛盾**不在 README 与知识图谱之间，而在 README.md 内部**：同一篇文章有两条条目，一条按 published 列（第 51 行，英文版为主条目），一条按 `[Pending Review]` 列（第 150 行，中文版，挂在别的分类下）。`README-cn.md` 与知识图谱都站在「published」一侧，也就是 3 比 1。

- **对文章的裁决**：文章「不把该文当知识来源」是**安全的**（AGENTS.md：`[Pending Review]` 不得作为知识来源，冲突时保守），而且正文确实没引用它，改由 DeepSeekMoE 论文承担对应内容——**这个替换是正确的**。但第 1369 行「在 README.md 中标记为 [Pending Review]」应改写成「README.md 内部对同一篇文章有两条互相矛盾的条目」，否则读者会以为该文明确未发布（见 P1-3）。
- **该由谁修**：README.md 第 150 行是重复且状态过期的条目，应由仓库主人决定是删除它、还是把它并入第 51 行；`knowledge-graph.json` 的 `status` 只有在 README 统一之后才需要跟改，**都不属于本文的改动范围**。文章现在「只报告、不改动」在纪律上是对的，只需把报告写准。

### 4. 审查报告与被审对象的版本漂移，应立什么约定？

**建议在本仓库的 `/learn-review` 流程里加一条硬约定（可直接写进 `.claude/commands/learn-review.md` 的 Step 4 报告格式）。**

第 2 轮报告以 md5 `9174861c…`／1368 行为基准，文章现在是 `30dac685…`／1372 行，两者对不上——而第 1 轮的报告（`REVIEW.md`）同样留下了行号基准不明的问题。这类漂移在本仓库反复发生（可对照 `transformers/kda_linear_attention/` 的 r3–r5 报告），原因是修复轮次必然改行号，而报告里的定位只有行号。

可操作约定（按优先级）：

1. **报告头部加 provenance 块（机器可读）**：`被审对象路径 / wc -l / md5sum / 该文件所属仓库的 rev-parse HEAD / 审核时刻`。建议直接写成 YAML front-matter（`object: …`、`lines: …`、`md5: …`、`repo_head: …`、`reviewed_at: …`），这样下一轮可以用一条命令校验「这份报告是否还适用于当前文件」。
2. **行号降级为线索，主键改为「小节号 + 该小节标题原文」**：引用一律写成「§7.4（配对块串行执行 vs 配对块重叠成立）」，行号只在括号里备注。本轮报告已按此执行。
3. **报告只读一次快照**：报告开写之后被审对象不得再改；若必须改，则新增 `revision 2` 小节，禁止原地改行号（否则历史行号失去意义）。
4. **修复记录（`REVISION.md`）引用「报告小节号 + 问题编号」**，不要引用行号，也不必复述报告原文——现在 `REVISION.md` 里同时写着结论与依据，容易出现自述与实物不一致（本轮 P1-5 就是实例：`REVISION.md` 说改成了 435，正文实际是 433）。
5. **附带效应**：`notes/MATH-VERIFY.md`、`notes/IMAGE-VERIFY.md`、`IMAGE-SURVEY-*.md` 也应带同一套 provenance 头（尤其 `IMAGE-SURVEY-*.md` 的「md 引用行」列——本轮全量复核证明这 30 个行号目前全对，但它们的有效性依赖被引用的 `references/*.md` 不变）。

### 5. `codes/02_schedule_sim.py` 把非 zb 反向统一记成 `I → W`：文章的断言是否正确？

**断言正确，我未能证伪。**

文章的断言（§7.3 第 757 行）：这处建模取舍「只影响同一个 rank 内部两个任务之间的松弛量（因此不影响跨 rank 的依赖结构，也不影响任何一台设备的总工作量）」。

我的独立验证：

```text
方法：把 ProgramBuilder._backward 换成「非 zb 反向先 emit W 再 emit I」（与 example_dualpipe.py
第 29–33 行 grad_weight_fn() 排在 grad_input = grad_output @ weight 之前一致），
其余依赖装配不动，比较 makespan / busy / bubble。

扫描 p ∈ {2,4,6,8,10,12}, m ∈ [2p, 40]：共 162 组 (p,m) × 2 方法 = 324 次比较
→ I→W 与 W→I 两种 rank 内顺序结果不同的组数 = 0
（抽样：p=4,m=10 DP (31,30,1) / DPV (63,60,3)；p=8,m=20 DP (63,60,3) / DPV (127,120,7)）
```

结构性理由（为什么必然如此）：① 两种顺序的任务多重集完全相同，故每设备总工作量不变；② 跨 rank 反向边**只挂在 `I` 上**（`wire_dependencies` 第 267–287 行用的是 `b.inp`），边集不变；③ 同一 rank 里相邻两个跨 rank 同步点之间的程序链长度不随 `I`/`W` 互换而改变。

**一处保留意见（措辞层面，非错误）**：「只影响松弛量」成立，但方向要说对——在 `W→I` 下 W 不再被迫等待上游的 `I`，即 W 的松弛量**变大**；模拟器把 W 排在 I 之后会让 W 更紧。文章没有说反，但读者容易把「只影响松弛量」误读成「两种顺序完全等价」，建议补一句「模拟器给出的是两种顺序中较紧的那一种，因此它的气泡是偏保守的估计」。

---

## 对抗性证伪尝试

| # | 待证伪的结论 | 结论 |
| --- | --- | --- |
| 1 | 驱动问题的答案：「双向布局提供了天然独立、可配对的前向与反向任务」——\((F_0,B_1)\) 里两者真的没有数据依赖吗？\(B_1\) 需要的是哪个 microbatch 的梯度？ | **未能推翻。** \(B_1\) 需要的是**反方向**第 \(x-p/2\) 个 microbatch（即较早的一个）的梯度：稳态式 \((F_{0,x},B_{1,x-p/2},F_{1,x-p/2+1+r},B_{0,x-p+1+r})\) 里的 \(B_1\) 下标就是 \(x-p/2\)。\(B_1\) 的输入是 chunk 1（`module[1]`）在该 microbatch 上前向保存的激活 + 从 rank \(r+1\) 收到的 output-grad；\(F_{0,x}\) 的输入是从 rank \(r-1\) 收到的激活、作用于 `module[0]`。**模块不同、microbatch 不同、张量不同，数据流图上不存在边**（对照 `dualpipe.py` 第 121–183 行 `_forward_backward_compute_chunk` 的 pre-forward / pre-backward 两段，它们各自从 `input_chunks[phase0]` 与 `output_grad_chunks[phase1]` 取数）。<br>**但我记一处可加强的地方（我的判断，不是推翻）**：1F1B 稳态里的 \((F_k, B_{k-(p-r-1)})\) 同样是数据流独立的，所以「存在可配对的前反向」并非双向布局独有；双向布局真正的区别是**配对的两个 chunk 属于不同参数集**（§5.2 已写「每台设备持有两个模型层段」、§6.1 已写「让设备上同时存在两个可配对的 chunk」）。建议在 §5.4 补一句，否则懂 1F1B 的读者会觉得答案不够锋利 |
| 2 | 官方 \((PP/2-1)(F\&B+B-3W)\) 在 \(F\&B=\max(F,B_{\mathrm{full}})\) 下等于 \((p-2)t/2\)，并与模拟器实测一致 | **未能推翻。** 用 `fractions.Fraction` 精确复算：\((p/2-1)(2t+2t-3t)=(p/2-1)t=(p-2)t/2\) ✓；模拟器 p=2,4,6,8 的气泡依次为 0,1,2,3 ✓；两端代入 \(F\&B=3t\) 得 \((p-2)t\)，恰为两倍 ✓ |
| 3 | 气泡率分母 \(3m\) / \(6m\) 的区分（含第 2 轮 P2-3 的「解释牵强」是否修好） | **未能推翻结论，但解释仍未修好。** 恒等式全部成立且精确：1F1B \(\frac{3(p-1)}{3m+3(p-1)}=\frac{p-1}{m+p-1}\)、ZB1P \(\frac{p-1}{3m+p-1}\)、DualPipe \(\frac{(p-2)/2}{3m+(p-2)/2}=\frac{p-2}{6m+p-2}\)、DualPipeV \(\frac{(p-1)/2}{3m+(p-1)/2}=\frac{p-1}{6m+p-1}\)（我用 Fraction 在 p∈{2,4,6,8}×m∈{8,10,16,20,32} 全量断言通过）。但「DualPipe 与 DualPipeV 的气泡恰好是 ZB1P 的一半」对 DualPipe 不成立（见 P2-6），\(6m\) 的真正来源是通分时清掉分子的 \(1/2\) |
| 4 | 「官方骨架不含细粒度重叠，`_weight_chunk()` 里的 `_commit_and_wait_comm()` 是硬同步点」有没有被过度推广？ | **未能推翻，但应加范围限定。** 已核对 `dualpipe.py` 第 216–223 行：`_weight_chunk()` 先 `_commit_and_wait_comm()`（第 285–291 行：`dist.batch_isend_irecv` → 逐个 `req.wait()` → 清空列表 → `_free_tensors()`），**再** `WeightGradStore.pop()` ✓；第 185–214 行三个 chunk 函数的形状也确实是「append → commit_and_wait → compute → send」✓；第 168 行确为 `type(module0).overlapped_forward_backward(...)` ✓；示例钩子（`example_dualpipe.py` 第 70–83 行）确实是顺序执行 ✓。<br>需要限定的两点：① `req.wait()` 阻塞的是 **host 线程**，它保证「没有新的计算被发射到这段通信上」，但**不能推出「GPU 在等待期间空闲」**——`wait` 之前已发射的 kernel 仍可能在跑；文章现有措辞（「没有哪个计算被摆在通信进行中执行」「`WeightGradStore.pop()` 排在这些通信完成之后」）是站得住的，建议把这个区分明写一次。② `_commit_and_wait_comm()` 在 `comm_ops` 为空时**提前返回**（第 286–287 行），所以「每个计算 chunk 之前都有一次 `req.wait()`」是对代码路径的概括，不是逐 chunk 的事实 |
| 5 | 固定设备数下 DualPipeV 每设备参数份额 `1×`、DualPipe `2×` | **未能推翻。** DualPipe 切 \(p\) 段、每设备 2 个 chunk → \(2\cdot\frac1p=\frac2p\)；DualPipeV 切 \(2p\) 段、每设备 2 个 chunk → \(2\cdot\frac{1}{2p}=\frac1p\)；比值 2 ✓ 与 §6.3 表一致。且「以 chunk 为单位两行都写 `2x`」也与 README 第 30–31 行原文一致（我逐字读了官方表格） |
| 6 | ZB-H1 峰值激活等于 \(pM_B\)，与 1F1B 相同 | **未能推翻。** 论文 §2.1 第 46 行：「ensures that the maximum peak memory usage over all workers doesn't exceed that of 1F1B」；§2.3 第 62 行给 worker 公式 \((p-i+1)M_B+(i-1)M_W\)，在 \(i=1\) 取 \(pM_B\)；Table 2（第 70 行）1F1B 与 ZB-H1 的峰值列都是 \(pM_B\) ✓。文章 §4.3 与 §7.4 第二点的表述与之一致 |
| 7 | 「1F1B 一次迭代时长 \((m+p-1)(F+B_{\mathrm{full}})\)」的适用前提 | **未能推翻——文章把前提带出来了。** 论文附录 H（`zero-bubble.md` 第 348 行）原文：「assuming \(m<=p\) and \(T_W<T_B\), an 1F1B iteration takes \((m+p-1)*(T_F+T_B+T_W)\)」，文章第 318 行写「论文附录在 \(m\le p\) 的粗略分析里明确写…」，前提在场 ✓。唯一可挑的是 §7.4/§7.5 随后在 \(m>p\) 使用它而没有补一句「该式对任意 \(m\ge1\) 成立」（P2-7） |

---

## 逐项核对记录

### 1. 数学逐式复核（公式 → 来源 → 我的复算 → 结论）

全部用 `fractions.Fraction` 精确复算，未用浮点近似。

| 公式 | 来源 | 我的复算 | 结论 |
| --- | --- | --- | --- |
| \(Y=X\Theta^\top\)；\(I=G\Theta\)、\(W=G^\top X\) | Zero Bubble §2/Fig 1 | \((N\times H_{out})(H_{out}\times H_{in})\to N\times H_{in}\)；\((H_{out}\times N)(N\times H_{in})\to H_{out}\times H_{in}\) | 一致 |
| 「\(I\) 优先、\(W\) 可延迟」是调度策略而非数学必然 | 论文 §2 + 文章判断 | 单算子确实互不依赖；整段 stage 里某些参数梯度可用时间晚于输入梯度——文章 §2.3 已显式写成「策略」并给出反例方向（沿序列维归约） | 一致（且是**修正后的收紧表述**） |
| Table 1：F \(sbh(24h+4s)\)/0；B \(sbh(24h+8s)\)/\(sb(34h+5as)\)；W \(sbh(24h)\)/\(32sbh\) | `zero-bubble.md` 第 60 行 | 逐字比对 | 一致 |
| \(T_W<T_F<T_B\)、\(T_B+T_W=2T_F\) | 第 56 行 | 第 1、3 行相加 = \(sbh(48h+4s)\) = 2×第 2 行 ✓ | 一致 |
| Table 9 偏差「1.5B/p=8/m=24 约 26%、且 \(T_B<T_F\)」 | 第 302 行 | 27.423 / 37.044 → 25.97% ✓；18.086 < 18.522 ✓ | 一致 |
| Table 9 偏差「其余配置 14%–17%」 | 文章自算 | 实测 13.74%–18.14% | **不一致 → P2-3** |
| \(n_{\mathrm{warmup}}(r)=\min(p-r-1,m)\) | Megatron `schedules.py` 第 930/949–950 行（tag `core_v0.19.0`） | 源码 `num_warmup_microbatches = pipeline_parallel_size - pipeline_parallel_rank - 1` 再被 `total_num_microbatches` 截断 ✓ | 一致 |
| 稳态 \((F_{k+p-r-1},B_k)\) | 文章推导 | p=4 → 3/2/1/0 warmup，配对 \((F_3,B_0),(F_2,B_0),(F_1,B_0),(F_0,B_0)\) ✓ | 一致 |
| \(T_{\mathrm{1F1B}}=(m+p-1)(F+B_{\mathrm{full}})\) | 附录 H 第 348 行 | 论文 \(T_B+T_W=B_{\mathrm{full}}\)，替换一致 | 一致（前提见 P2-7） |
| \(\beta_{\mathrm{1F1B}}=(p-1)/(m+p-1)\) | 文章 | \(\frac{3(p-1)}{3m+3(p-1)}\) ✓ | 一致 |
| Table 2 三行（1F1B / ZB-H1 / ZB-H2 的气泡与峰值） | 第 70 行 | 逐字比对 | 一致 |
| ZB-H1 气泡 \((p-1)(F+I-W)=(p-1)(F+B_{\mathrm{full}}-2W)\) | Table 2 + 代入 | \(B_{\mathrm{full}}=I+W\) ✓ | 一致 |
| ZB-H2 气泡 \((p-1)(F+I-2W)=(p-1)(F+B_{\mathrm{full}}-3W)\)、等时下 0 | Table 2 + 第 50 行 | \((p-1)(t+t-2t)=0\) ✓ | 一致 |
| worker 激活 \((p-i+1)M_B+(i-1)M_W\)、\((2p-2i+1)M_B+(2i-2)M_W\) | 第 62 行 | 与原文逐字一致；\(M_W<M_B\) → 峰值在 \(i=1\) → \(pM_B\)、\((2p-1)M_B\) ✓ | 一致 |
| DualPipe 稳态 \((F_{0,x},B_{1,x-p/2},F_{1,x-p/2+1+r},B_{0,x-p+1+r})\) | `dualpipe.py` L381–396 | p=8,r=1,x=8 → \((F_{0,8},B_{1,4},F_{1,6},B_{0,2})\) ✓；warmup 计数 → 第一组 x=6 → \((F_{0,6},B_{1,2},F_{1,4},B_{0,0})\)，与脚本 rank 1 输出一致 ✓；p=4,m=10 四行全部代入通过 ✓ | 一致 |
| DualPipeV 稳态 \((F_{0,x},B_{1,x-p},F_{1,x-p+1+r},B_{0,x-2p+1+r})\) | `dualpipev.py` L352–367 | p=4/m=10 四个 rank、p=8/m=20 八个 rank 全部代入通过 ✓ | 一致 |
| 八步循环次数（两套） | L358–425 / L330–395 | 逐行对照源码，16 个 step 计数全对 ✓ | 一致 |
| 官方表换算：ZB1P \((PP-1)(F+B-2W)=(p-1)(F+I-W)\) | README 第 29 行 | \((F+I+W-2W)\) ✓ | 一致 |
| 4 组气泡率的 4 个恒等式 | 文章 + 模拟器 | Fraction 在 p∈{2,4,6,8}×m∈{8,10,16,20,32} 全量断言通过 ✓ | 一致 |
| \(F\&B\) 两端：\((p-2)t/2\) 与 \((p-2)t\)；\((p-1)t/2\) 与 \((p-1)t\) | README 定义 + 文章 | 精确复算 ✓ | 一致 |
| p=8,m=20 的 7.5 节表（21 / 7 / 3 / 3.5；7/27、7/67、6/126、6/66、7/127、7/67） | 文章 | 全部逐一复算 ✓（25.93% / 10.45% / 4.76% / 9.09% / 5.51% / 10.45%） | 一致 |
| 激活换算 \((2p+1)\cdot A/2=(p+1/2)A\) | 文章 | Rounding 一致；p=8 → 8.5A ✓ | 一致（前提已在 §7.4 写明） |
| 隐藏状态 \(2\times2048\times4096\times2\,\mathrm{B}=32\,\mathrm{MiB}\) | 文章 | 33,554,432 B = 32.0 MiB ✓ | 一致 |
| `pA = A_all` 的成立前提 | 文章 §3.3 | 前提（均匀划分 + 无重计算 + 忽略通信 buffer）已显式列出 ✓ | 一致 |
| 「每设备有效工作 \(3mt\)」四种方法相同 | 文章 §7.4 | 1F1B \(m\cdot3t\)；DualPipe \(2\cdot\frac m2\cdot3t\)；DualPipeV \(2\cdot m\cdot\frac{3t}2\) ✓ | 一致 |

**本章未发现问题**（除 P2-3、P2-6、P2-7 三条已在上面标出）。

### 2. 可复现性（命令 → 实际输出 → 与文章数字是否一致）

四条命令全部实跑。

| 命令 | 实际输出（关键行） | 文章数字 | 结论 |
| --- | --- | --- | --- |
| `01_split_backward_minimal.py` | `torch : 2.10.0+cu129`；`dtype : torch.float64`；`x/theta : (8,8)/(16,8)`；`microbatch 数 : 2`；`dTheta is None : True`；`max\|dX err\| : 0.000e+00`；`max\|dTheta err\| : 1.110e-16`；`PASS` | 第 205–211 行：环境 `2.10.0+cu129`、float64、`[8,8]`/`[16,8]`、2 个 microbatch、`None`、`0.000e+00`、`1.110e-16`、PASS | **逐字一致** |
| `02_schedule_sim.py --p 4 --m 10` | DualPipe busy 30 / makespan 31 / bubble 1 / `0.032258 = 3.2258%`；DualPipeV busy 60 / makespan 63 / bubble 3 / `0.047619 = 4.7619%`；闭式复核 4 项全「一致」 | §7.3 表：30/31/1/3.23%、60/63/3/4.76%；§7.3 正文「换算到 t 之后 busy 30、关键路径 31.5、气泡 1.5」✓ | **一致** |
| `02_schedule_sim.py --p 8 --m 20` | DualPipe 60/63/3；DualPipeV 120/127/7；rank 1 稳态 `F0_6, B1_2, F1_4, B0_0` | §5.4、§6.6 表格与正文 | **一致**（`codes/README.md` 第 78 行的表述不一致，见 P0-1） |
| `02_schedule_sim.py --sweep` | 覆盖 p∈{2,4,6,8}×m∈{8,10,16,20,32}，30 组全部打印「一致」，闭式 \(3mt\)、\((p-2)t/2\)、\((p-1)t/2\)、\((p-2)/(6m+p-2)\)、\((p-1)/(6m+p-1)\) 全部命中 | §7.3 表与文末「已验证」段 | **一致** |

**模拟器本身的忠实度审阅（比核数字更重要）**——我逐行对照 `02_schedule_sim.py` 与官方 `step()`：

- **step 3 / 6 / 7 的 `enable_zb` 切换点**：脚本第 117 行（step3 `enable_zb=True`）、第 142–146 行（step6 的 `if i == step_6 // 2 and half_rank % 2 == 1` / `== 0` 两处切换，含顺序）、第 151 行（step7 `enable_zb=True`）与 `dualpipe.py` 第 376、408–413、419 行逐行一致；DualPipeV 版第 178、203–207、212 行与 `dualpipev.py` 第 347、379–384、390 行一致。✓
- **`_weight_chunk()` 与 `flush()` 的先后**：官方在 `_backward_compute_chunk` 末尾对每次 `enable_zb=True` 的反向调用一次 `WeightGradStore.flush()`（`dualpipe.py` 第 113–114 行）✓，脚本第 74–77 行正是「append 到 cache 后立即 flush」✓；官方 `_weight_chunk()` 是「`_commit_and_wait_comm()` → `WeightGradStore.pop()`」（第 220–223 行），脚本 `_weight_chunk` 只做 `popleft()` 并把组内任务就地发射（第 86–90 行）——**通信等待被有意省略**（模拟器不建模通信），这不是依赖模型的错误，且脚本第 216/155 行的 `assert not b._queue` 复现了官方第 425/396 行的 `assert WeightGradStore.funcs_queue.empty()` 会计 ✓。
- **`forward_only` 提前返回**：官方 `_weight_chunk()` 在 `forward_only` 时直接 return（第 217–218 行），脚本只建模训练态、没有 forward_only 分支。这**不是**缺陷（脚本只在训练态下断言会计），但 `codes/README.md` 与脚本 docstring 都没提这一点，可以补一句。
- **跨 rank 的反向依赖只挂在 `I` 上、不挂 `W`**：脚本第 267–287 行用 `b.inp[...]` 建立反向边，而 `inp` 只在 `_backward` 里对 `I` 任务登记（第 73 行），`W` 从不登记 ✓。这正确复现了 Zero Bubble 剪掉的那条边。
- **跨 rank 前向依赖**：第 244–265 行按 `(direction, chunk)` 配对上/下游第 k 个 F ✓；DualPipeV 在 `rank == p-1` 的上升段改为本地交接（第 256–258 行 `dep = b.fwd.get((0, c))`），与 `dualpipev.py` 第 79–80 行的本地入队一致 ✓；DualPipeV 反向底部同理（第 274–275 行 vs `dualpipev.py` 第 115–116 行）✓。
- **结论：我没有找到依赖模型或 W 队列会计的反例。** 想构造反例的三个方向（把反向边挂到 `W`、把 V 底部当 P2P、把 step6 切换点提前/延后半个循环）都会立刻破坏脚本第 360–364 行的计数断言（每 rank F=I=2·per_rank_f、全设备 I 总数 = W 总数）或第 395 行的气泡闭式断言。

**「已删除」的旧声明是否真的删干净**：`6.939e-18`、`8.674e-19`、`2.10.0+cpu`、`sandbox:`、`main`/`master` 行号链接、`zhimg` 裸外链在 `deep-dive.md` 中全部为 0 命中 ✓。`66 / 9.09%` 这一组**仍然存在但含义已变**：第 811 行的 `6/66 ≈ 9.1%` 是 §7.5 表里「配对块串行执行」那一端的正确取值（\(3mt+(p-2)t=60+6=66\Rightarrow 6/66\)），与草稿的 `busy 60 / total 66 / 9.09%` 不是同一件事——草稿是用串行口径冒充模拟器结果，现在是被显式标注的对照列 ✓。三行伪造的「官方代码中文注释」在 `deep-dive.md` 与两份草稿中都不再出现，且 `git grep -nP '[^\x00-\x7F]' 030ce432 -- '*.py'` 返回空，确认该 revision 的 Python 源码没有任何非 ASCII 字符 ✓。

### 3. 源码行号核对（引用 → revision → 实际位置 → 一致？）

**DeepSeek DualPipe @ `030ce4325f4ebeb437da4ebc6d00a70469dd58ae`**（本地 HEAD = origin/main，已 `git rev-parse` 复核）

| 文章引用 | 实际 | 结论 |
| --- | --- | --- |
| `dualpipe.py` 440 行、`dualpipev.py` 411 行、`utils.py` 80 行、`comm.py` 38 行、`example_dualpipe.py` 202 行、`example_dualpipev.py` 183 行 | `wc -l` 完全一致 | ✓ |
| `step()`：`dualpipe.py` L294–440、`dualpipev.py` L288–411 | `def step(` 分别在 294 / 288，文件末行 440 / 411 | ✓ |
| 入口断言：`num_ranks % 2 == 0`、`num_chunks % 2 == 0 and >= num_ranks*2`（dualpipe.py L332–333）；DualPipeV 只有 `num_chunks >= num_ranks*2`（L318） | 逐行一致 | ✓ |
| `half_rank = min(rank, num_ranks - 1 - rank)` | L335 | ✓ |
| `is_middle_rank` = `rank == num_ranks//2 - 1 or rank == num_ranks//2` | L45 | ✓ |
| `phase ^= self.is_in_second_half` 作为「一次异或」 | 出现在 L68、91、232、242、260、273（6 处） | ✓ |
| 中间 rank 特判注释原文 `# NOTE: We don't overlap these two chunks to further reduce bubble size.` | L386（DualPipeV 对应 L357） | ✓ 逐字一致 |
| `assert WeightGradStore.funcs_queue.empty()` | L425 / L396 | ✓ |
| `utils.py` 第 8–33 行 26 行；`put`/`flush`/`pop`/`clear` 四个类方法 | L8–33 = 26 行 ✓；四个 `@classmethod` 齐全 ✓（「三个方法 15 行」的口径见 P2-4） | ✓ |
| `run_backward` 的四个 kwargs（`keep_graph`/`create_graph`/`allow_unreachable`/`accumulate_grad`） | L37–42 | ✓ |
| `comm.py` 的 `TENSOR_SHAPES`/`TENSOR_DTYPE`/`build_from_tensor_shapes`（`requires_grad=True`）、`append_irecv`/`append_isend` | L7–8、21–22、25–38 | ✓ |
| `full_modules[rank]` / `full_modules[pp_size - 1 - rank]` | `example_dualpipe.py` L138 | ✓ |
| `full_modules[rank]` / `full_modules[pp_size*2 - 1 - rank]`、`pp_size * 2` 个 stage | `example_dualpipev.py` L127、L137 | ✓ |
| 示例声明形状 `(3,256,512)`、`float32` | `example_dualpipe.py` L119–125 | ✓ |
| 梯度合并方式：`all_gather_into_tensor` + 索引 `pp_size-1-rank` | L171–174 | ✓ |
| `criterion` 逐 microbatch `F.mse_loss` 且不缩放；参考实现逐 microbatch 累加 | L86–87、L90–100 | ✓ |
| `overlapped_forward_backward` 注册为类方法、调度器用 `type(module0).overlapped_forward_backward(...)` 调用 | L54–55（类方法定义）、`dualpipe.py` L168 | ✓（docstring 句数见 P2-5） |
| 钩子体顺序：先 `module0(*inputs0)`，再 `loss1.backward()` / `run_backward(...)` | L70–81 | ✓ |
| 示例钩子里 `grad_weight_fn()` 排在 `grad_input = grad_output @ weight` 之前 | L29–33 | ✓ |
| 与 `3da1bbea…` 的差异「只有 `__init__.py` 的 `__all__` 从类对象改成字符串这 10 行」 | `git diff --stat 3da1bbea 030ce432` → `dualpipe/__init__.py | 10 +++++-----`（1 file, 5 insertions, 5 deletions）；diff 内容确为 5 个条目加引号 | ✓ |
| 草稿里的 `3da1bbea…`（2025-03-06） | `git log -1 --format=%ai` = **2025-03-06 09:41:54 +0700**（commit date 是 03-10，作者日期才是 03-06） | ✓ 文章用的是作者日期，正确 |
| 「`030ce432…` 也是上游 `main` 当时的 HEAD」 | 本地 clone 的 `origin/main` 指向它 | ✓（离线可验证的部分成立；无法验证远端是否已被推进） |

**PyTorch 本地安装版 2.10.0+cu129，`/usr/local/lib/python3.12/dist-packages/torch/distributed/pipelining/`**

| 文章引用 | 实际 | 结论 |
| --- | --- | --- |
| `schedules.py` 3438 行 | 3438 | ✓ |
| `ScheduleInterleaved1F1B` 第 2493 行、`ScheduleZBVZeroBubble` 第 2808 行、`ScheduleDualPipeV` 第 2994 行 | 2493 / 2808 / 2994 | ✓ |
| `register_custom_function` 第 1887 行 | 1887 | ✓ |
| `OVERLAP_F_B` 默认执行分支第 2257–2260 行 | 2257–2260，内容 `elif … OVERLAP_F_B:` / `assert action.sub_actions is not None` / `for sub_a in action.sub_actions:` / `_perform_action(sub_a)` | ✓ |
| 「`_comp_type_to_function_map` 的检查排在 `OVERLAP_F_B` 分支之前」 | L2246 `if action.computation_type in self._comp_type_to_function_map:` 在 L2257 `elif … OVERLAP_F_B` 之前 | ✓ |
| `_ComputationType`：`BACKWARD_INPUT = 2`、`BACKWARD_WEIGHT = 3`、`OVERLAP_F_B = 11` | L47 / L48 / L56 | ✓ |
| `sub_actions` 四行摘录 | L3093–3097 逐字一致（`FORWARD` + `FULL_BACKWARD` + `_Action(-1, OVERLAP_F_B, None, sub_actions)`） | ✓ |
| `generate_stage_to_rank_mapping` 第 91 行，`style == "v"` 分支第 104–119 行 | L91 / L104–119 | ✓ |
| 注释 `dont change rank if we are on the border (to keep v shape)` | L113 | ✓ |
| `n_local_stages != 2` 报错、`n_microbatches < self._num_stages` 报错 | L3033–3036 / L3038–3042 | ✓（`_num_stages = 2p`，故等价于 \(m\ge2p\)） |
| `_backward.py` 不是把同一个 `.backward()` 调两次 | `stage_backward_input`（L143）与 `stage_backward_weight`（L226）分离，docstring L153–157 明说「we save the intermediate nodes in `param_groups` for later use in `stage_backward_weight`」 | ✓ |
| **未出现 `v2.9.0` 或 `main` 的行号** | `grep -n "v2\.9\.0\|main"` 在 `deep-dive.md` 中 0 命中 | ✓ |

**Megatron-LM tag `core_v0.19.0`（`git show` 读取，非工作树）**

| 文章引用 | 实际 | 结论 |
| --- | --- | --- |
| `combined_1f1b_schedule_for_interleaved_pipelining` 第 138 行、`combined_forward_backward_step` 第 281 行 | 138 / 281 | ✓ |
| docstring「This method is called only if `overlap_moe_expert_parallel_comm` is true.」 | 第 164 行，逐字一致 | ✓ |
| `checkpoint_activations_microbatch` 与 `overlap_moe_expert_parallel_comm` 不兼容 | 第 344–345 行 assert | ✓ |
| `TransformerLayerSchedulePlan` 第 30 行、`_build_callable_nodes` 第 112 行、`delay_wgrad_compute` 第 133 行 | 30 / 112 / 133 | ✓ |
| 五个节点与 stream 分配：`pre_dispatch_computation`→comp、`mlp`→comp、`moe_dispatch`→comm、`moe_combine`→comm、`mtp_post_process`→comp | 第 160–176 行完全一致 | ✓ |
| `schedules.py:926-950` 的 `num_warmup_microbatches` | 赋值在 L930，截断在 L949–950，区间 926–950 覆盖 | ✓ |
| `tag core_v0.19.0 = 5be96267…` | `git rev-parse core_v0.19.0` = `5be9626709af2722333bf54797c954c09edeada3` | ✓ |

**代码摘录是否被改写**：把文章里 8 段较长的 Python 摘录（`SplitLinear` 教学片段、`WeightGradStore` 类、`_weight_chunk`、step3 循环、step4 中间 rank 特判、`_forward_backward_chunk`、`comm.py` 片段、`sub_actions`、默认执行分支、`register_custom_function` 签名、`_ComputationType` 片段、Megatron 的 `delay_wgrad_compute` 行）与源文件逐行 diff：**除整体去缩进外，标识符、语句顺序、guard 全部一致；`comm.py` 与 `_ComputationType` 两处有省略，都已用「（… 在此省略）」/`...` 显式标出**；`SplitLinear` 是文章自编的教学最小示例而非官方代码，文章第 186 行已说明「它不是 DualPipe」并在第 205 行指向 `codes/01`，符合「不得用自编示例冒充源码」。**未发现改写。**

### 4. 图片（存在性 / 元数据行号全量比对 / 读图结果）

**存在性（`comm` 双向，全量）**：

```text
grep -oE 'src="\./pics/[^"]+"' deep-dive.md | sed 's|src="\./pics/||;s|"$||' | sort | uniq > /tmp/ref_imgs.txt
→ 32 条（无重复）
ls pics/ | grep -v '^README.md$' | sort → 32 条
comm -23（被引用但不存在）= 空
comm -13（存在但未被引用）= 空
```

✓ **32/32 完全对齐，无缺失、无多余、无重复引用。**

**元数据行号（全量脚本，30 条）**：脚本流程 = 对 `pics/` 下每张图算 sha256 → 在 `references/` 下全部 435 个图片文件里按 sha256 反查源文件 → 在源 md 中用 `![]()`/`<img src=…>` 正则定位该文件的真实引用行号 → 与文章 30 条「本地副本：`…` 第 N 行引用图」逐条比对。

结果：**30 / 30 全部命中，0 处不一致**（含 `zero-bubble.md` 第 23/38/41/94/158 行、`gpipe.md` 第 53 行、`deepseek-moe.md` 第 74 行、`deepseek-v3-report.md` 第 291/306 行、`chimera.md` 第 74/81 行、`controllable-memory.md` 第 77/111/143/416 行、`megatron-lm-gpu-clusters.md` 第 62/155 行、`pipedream-2bw.md` 第 64 行、`terapipe.md` 第 28 行、`aiinfra-pp-1f1b-interleaved.md` 第 13/29/43/73/346 行、`xiaodonggua-…md` 第 24/59/606 行、`yeqianshu-…md` 第 59 行、`sea-ai-lab-…md` 第 19/26 行）。另外 sha256 反查同时证明了**装图正确**（每张图与它声称的源文件字节相同）。

**读图（独立 `read_image` 实读 17 张）**：

| 文件 | 我的读图结果 | 与 alt/图注 |
| --- | --- | --- |
| `dualpipe.png` | Device 0–7；五类图例（Forward 橙 / Backward 绿 / Backward for input 浅绿 / Backward for weights 蓝 / Overlapped forward & Backward 橙绿拼接）；白格为气泡；**格内编号最大 9** | 相符 |
| `dualpipev.png` | Device 0–3；同一套图例；**格内编号最大 9**；各行序列不同 | 相符 |
| `gpipe-fig2-…jpg` | 三联：(a) Device 0–3 的 F_i→B_i 链 + Loss/Gradients 箭头；(b) 单 microbatch 阶梯 + Update 列；(c) F_{i,j}/B_{i,j} 密排 + 中部显式 `Bubble` 框 | 相符；(p−1)(F+B) 只对应 (c) |
| `zero-bubble-fig2-…jpg` | Device 1–4，三色（Forward 蓝 / Backward 橙 / Optimizer step 米色），**无 W 格** | 相符 |
| `zero-bubble-fig3-…jpg` | 像素分类：上栏 F = 4/3/2/1、白格 3/2/1/0；下栏 F = 7/5/3/1、白格全 0；米色 optimizer 逐行右移错位 | 相符（**7/5/3/1 正确**） |
| `zero-bubble-fig4-…jpg` | 图例：橙 `1 2 3 4`= 沿对角线传播局部归约值；米色 = Optimizer step；橙 `5 - 8` = 把全局归约值回传各 stage；深红 = Rollback if validation fails；1–4 与 5–8 都在对角线上；米色逐行错位 | 逐条相符 |
| `zero-bubble-fig8-…jpg` | 4 个 Device 各**一行**网格；同一行内同时有白字与黑字数字；图例 F 蓝 / B 青 / W 绿 / Optimizer step 米色；米色逐 Device 右移 | 相符 |
| `controllable-memory-fig2-…jpg` | 左 `Parallel`：3 条红色 device 轴（l₁/l₂/l₃）另加 l₄/l₅/l₆ 标注，虚线箭头表示 stage 依赖；右 `V-Shape`：同结构但第二半的层位反序 | 相符（放置规则另与论文 §3 第 75 行「second half of stages … in reverse order」互相印证） |
| `controllable-memory-fig4-…jpg` | 四个面板自上而下标 **(a) 1F1B / (b) V-Min / (c) V-Half / (d) V-ZB**，均为多 device 时间网格，含 F/B/W 彩色方块 | 相符 |
| `controllable-memory-fig18-…jpg` | 八组：(a) 1F1B、(b) Eager 1F1B、(c) ZB-H1、(d) ZB-H2、(e) GPipe、(f) GEMS、(g) Chimera、(h) Interleaved 1F1B；每组上排为 building block/repeat、下排为挤压重排后的调度 | 相符；**但实测尺寸 742×1723，文章写 743×1723（P2-2）** |
| `chimera-fig2-…jpg` | 六个方案（PipeDream / PipeDream-2BW / GPipe / GEMS / DAPPLE / Chimera）同轴对照；图例框含 `Bubble`、replica0（黑 x）与 replica1（蓝 y）定义；右侧 `M_θ` / `M_a` 小柱 | 相符（\(M_\theta\) 是论文记号表中「weights」的符号，见 `chimera.md` 第 62 行；我在低分辨率下曾把它读成 `M_g`，正文用的 \(M_\theta\) 正确） |
| `pipedream-2bw-fig2-…jpg` | 4 个 Worker 的时间网格；棋盘格标出新版本权重落点；标注 `Before: W_1^(0), W_1^(0) / After: W_1^(0), W_1^(4)`（下同 W_4）、`t = 21` | 相符 |
| `terapipe-fig1d-…jpg` | Device 5→Device 1 五条 Transformer layer 条带；条带下方细密 token 分段；跨设备橙色细箭头 | 相符（确为 token 级 1(d)） |
| `sea-ai-lab-cut-in-half-mirrored-schedule.png` | Device 0–7；每格上下两个数字（上＝up-to-down、下＝down-to-up）；右侧 `The two parts are mirrored.` | 相符 |
| `sea-ai-lab-cut-in-half-vshape.jpg` | 上栏 DualPipe 8 device（Model Layers 0/7、1/6、2/5、3/4、4/3、5/2、6/1、7/0）F 块成 V 形；下栏 Cut-in-half 只留 Device 0–3，layer 配对相同；带「知乎 @庞天宇」水印 | 相符 |
| `megatron-fig4-…jpg` | 双面板；上＝default 1F1B，中间黑箭头标 `Assign multiple stages to each device`，下＝interleaved 1F1B；图例 Forward Pass（深蓝）/ Backward Pass（浅绿） | 相符 |
| `deepseek-v3-fig4-…jpg` | 上行 Computation：MLP(B)▲ / MLP(W)▲ / MLP(F)△ / ATTN(B)▲ / ATTN(W)▲ / ATTN(F)△；下行 Communication：DISPATCH(F)△ / DISPATCH(B)▲ / COMBINE(F)△ / PP / COMBINE(B)▲；图例 △ Forward chunk、▲ Backward chunk | **逐项相符** |

**带读数的图注像素级复核**：见上表 `zero-bubble-fig3`（自写色分类脚本，见「第 2 轮修复核验」第 3 项）。这是我唯一发现「读数」类断言的一张图，而它现在是**对的**。

**433 张写作素材图的逐张覆盖（全量计数）**：

```text
notes/IMAGE-SURVEY-*.md 的逐条清单行数（按每份文件的清单格式分别计数）：
aiinfra-docs 20 | chimera 26 | controllable-memory 49 | deepseek-moe 34 | deepseek-v3-report 88
gpipe 9 | megatron-lm-gpu-clusters 32 | pipedream 22 | pipedream-2bw 30 | terapipe 32 | zero-bubble 31
= papers+docs 部分 373
articles 60（汇总表 6 个 slug：normaluhr 5 / pipeline-parallelism-visualization 5 / reku 8 /
           sea-ai-lab 5 / xiaodonggua 24 / yeqianshu 13 = 60）
合计 433，与 references/ 实测的 papers 353 + docs 20 + articles 60 = 433 逐目录相等
```

✓ **覆盖是全量的**，每个目录的清单行数都等于 `ls | wc -l`。纯公式截图与表格截图在每份文件的「类别」列被判为「不采用」并给出理由（`learn-plan.md` 第 249 行记录 papers 下 41 张公式截图 + 15 张表格截图全部不采用，`articles/` 下 14 张与官方图重复的转贴改引原图）。**未发现问题。**

### 5. 引用与 pin

| 检查项 | 结果 |
| --- | --- |
| `main`/`master` 行号链接 | **0 处** ✓ |
| `sandbox:` 死链 | **0 处** ✓ |
| 裸外链图片（`zhimg` 等） | **0 处** ✓（`grep -oE 'https?://…'` 得到的全部是 arxiv abs、GitHub 带 40 位 commit、hackmd、zhihu 文章页、huggingface 博客、infrasys 教材页，没有图片直链） |
| 外部代码链接带 40 位 commit | 6 条 DualPipe blob 链接均为 `030ce4325f4ebeb437da4ebc6d00a70469dd58ae` ✓；2 条 Megatron blob 链接均为 `5be9626709af2722333bf54797c954c09edeada3` ✓；2 条 `tree/` 链接分别为该 commit 与 tag `core_v0.19.0` ✓；`sail-sg/zero-bubble-pipeline-parallelism` 用 commit `c5d5074132dd47aec5a92b8753a56d808a109eda` 与 tag `zero-bubble-v0.1.0`，均未引行号 ✓ |
| tag 与 commit 对得上 | `git rev-parse core_v0.19.0` → `5be96267…` ✓ 与文章一致 |
| 版本分层前后一致 | DualPipe 一律 `030ce432…`（8.1 节显式说明、8.5/8.6/参考节一致）✓；Megatron 一律 tag `core_v0.19.0`（3.2 节、9.2 节、参考节）✓；PyTorch 一律「本地安装版 2.10.0 + 文件路径 + 类行号」（9.1 节、参考节，且参考节显式写「本文不引用上游 `v2.9.0` 或 `main` 的行号」）✓ |
| repo 内 `.md`/脚本相对链接 | 10 条全部解析成功（`./codes/01_split_backward_minimal.py`、`./codes/02_schedule_sim.py`、`./codes/README.md`、`./notes/`、`./notes/MATH-VERIFY.md`、`./pics/README.md`、`../deepep/deep-dive.md`、`../fsdp2/readme.md`、`../nccl/readme.md`、`../torch-distributed/readme.md`）✓ |
| 被引用文章的 published 状态 | `../torch-distributed/readme.md`、`../nccl/readme.md`、`../fsdp2/readme.md` 都在 README.md 里按 published 列出 ✓；`../deepep/deep-dive.md` 不在两个 README 索引里（`grep -i deepep README.md README-cn.md` 无输出），文章第 7 行已加括号说明 ✓（是否足够见争议项 2） |
| `profile-data` 未 pin | 参考节已明确写「该仓库本地没有副本、也没有 pin 到具体 commit，以下描述按 2026-09 检索时的 README 自述，读者复核时请注意它可能已经变化」✓，且不引行号 ✓ |
| `sail-sg/zero-bubble-pipeline-parallelism` 本地无副本 | 只引 URL 与 tag，不引行号 ✓ |
| SGLang 路径 | `docs/docs/advanced_features/expert_parallelism.mdx` —— 本地确有该文件（`docs/docs/` 而非 `docs/`）✓，并注明快照日期 ✓ |

### 6. 比较口径

| 检查项 | 结果 |
| --- | --- |
| 官方比较表按「相同 PP stage 数」给出，DualPipeV 的 `#Devices` 是 `PP/2` | 我逐字读了 README 第 24–36 行：表头 `(based on the same number of PP stages)`、`DualPipeV` 行 `#Devices` = `*PP*/2` ✓，文章第 702、709 行转写正确 |
| 每张表是否写明当前基准 | §7.1 表：正文第 702 行先说明表头基准 ✓；§7.4 表：第 780 行显式列出全部前提假设（等时、同设备数 \(p\)、同一份模型、同一 \(m\)、\(A\) 的定义）✓；§7.5 表：第 805 行说明 \(p=8,m=20\) 与两个分母的取值，列名写明「配对块重叠成立/串行执行」✓；§6.3 表：两个基准分列两行并写明「每设备参数份额」✓。**未发现未标基准的表。** |
| 「参数 `2×`」与「DualPipeV 删除参数副本」各自的基准 | §6.3 解释正确：官方表的 `2x` 以「一个 stage 大小的 chunk = `1x`」计，两行每设备都放 2 个 chunk 所以都写 `2x`；差别在设备预算（PP vs PP/2）。固定设备数下 DualPipe `2/p` vs DualPipeV `1/p` ✓。前后一致（§6.3 与 §7.1 第二处、§7.4 表列一致）✓ |
| \(F\&B\) 取 \(\max\) 与取 \(F+B\) 两端，气泡各是多少 | §7.2 完整代入两端：DualPipe \((p-2)t/2\) 与 \((p-2)t\)；DualPipeV \((p-1)t/2\) 与 \((p-1)t\) ✓ 我用 Fraction 复算一致。**没有混用**：§7.3 表、§7.4 表（两列并列）、§7.5 表（两列并列）三处都分开列；§7.3 第 801 行与文末第 1350 行都明确说模拟器落在「配对块重叠成立」那一列 ✓（唯一残留是 §7.3 第 774 行的用词，见 P1-1） |
| 气泡率分母 \(3m\) 与 \(6m\) 的区分是否成立、解释是否牵强 | 区分**成立且精确**（四个恒等式全部验证）。解释**仍然牵强**：见 P2-6 |
| 「激活 `PP+1` → \((p+1/2)A\)」的换算前提是否写明 | §7.4 第三点写明「在固定设备数的口径下，DualPipeV 的逻辑 stage 是标准 stage 的一半，所以单个 stage 的激活单位也减半」并加粗「这个换算只有在写明『按固定设备数、stage 减半』时才成立」 ✓ |
| DualPipe 与 DualPipeV 两张官方图的 microbatch 口径 | 第 618 行明确写出「这张图是 4 台设备 × 单方向 10 个 microbatch，第五章的 `dualpipe.png` 是 8 台设备 × 双向合计 20 个（每方向 10 个），两张图的设备数与 microbatch 口径都不同，不能横向比格子数量」✓ |

### 7. 不可回溯断言与过度声称

我逐条挑的是「读起来很顺、但落不到论文/README/固定 commit 源码/可跑脚本上」的句子。结论：**文章在这一维度上出乎意料地干净**——绝大部分疑似断言都能落到具体行号上。以下是仍然偏软的三处（都不构成事实错误）：

| 句子 | 问题 | 建议 |
| --- | --- | --- |
| §8.5（第 986 行）「注释本身没有展开论证，『合并发送比多叠一个方块更划算』是本文对这段代码的机制解读，不是官方注释的原话」 | 这句**已经**主动把原文与解读分开，是正面例子。但前半句第 529 行「**这说明**『把两个方向配成一对』并不总是最优」把机制解读写成了推论 | 第 529 行的「这说明」可改为「本文据此推测」，与 8.5 的谨慎口径统一 |
| §8.7（第 1035 行）「每个计算 chunk 之前都有一次 `req.wait()`，所以没有哪个计算被摆在通信进行中执行」 | 「每个」略强（`comm_ops` 为空时 `_commit_and_wait_comm()` 提前返回）；且 `wait()` 阻塞 host 线程 ≠ GPU 空闲 | 见「对抗性证伪」第 4 条 |
| §9.2（第 1139 行）「Megatron 里也有一份『延迟 wgrad』的配置，**语义与 Zero Bubble 的延迟 \(W\) 同源**」 | 「同源」是本文的类比判断，不是 Megatron 文档或注释的表述；两处的推迟对象（ZeRO/FSDP 场景下的 wgrad 计算重排 vs Zero Bubble 的 I/W 拆分）机制并不相同 | 标成「机制上可以互相对照」而非「同源」；或补一句说明二者的差别 |

未发现任何「本文实测 GPU」「实测吞吐/显存」「验证过重叠收益」的表述 ✓。开篇第 11 行与文末第 1349 行各声明一次「没有任何 GPU 实测」✓（任务书要求的两次都在）。所有性能数字都标了来源与配置，估算量都写了假设 ✓。

### 8. 结构与风格

| 检查项 | 结果 |
| --- | --- |
| 开篇非模板化 + 个人动机 | ✓ 第 7 行从「每次有人问我 DeepSeek 那个 DualPipe 到底厉害在哪」与「我能给出的回答仍然停留在…没有信息量的层面」切入；第 9 行有真实阅读感受 |
| 系列回顾 | ✓ 第 7 行依次回顾 `../deepep/deep-dive.md`、`../torch-distributed/readme.md`、`../nccl/readme.md`，并说明各自交代了哪一层 |
| 路线图 ≤ 4 条 | ✓ 第 13–18 行是 4 条编号列表 |
| 致谢自然 | ✓ 第 20 行「照例感谢把 DualPipe、Zero Bubble、Controllable Memory 与 Megatron 完整开源的这些团队…」，无公司/组织标注 |
| 章节顺序严格「概念 → 模型/场景 → 代码」 | ✓ 一～四章概念与基线、五～七章调度模型与重算、八～九章源码与框架分层、第十章选型 |
| 驱动问题在有足够背景后出现并在后文回收 | 回收 ✓（第 406、468、1155 行三次）；出现在第一章末尾——**背景是否足够见 P1-2** |
| 过渡句是否具体（禁止「有了这些基础，我们来看」） | ✓ `grep` 全部空命中；实际过渡如第 322 行「第三章讲完了外层调度能做的事，接下来该看反向拆分能做的事」、第 406 行「这正是第一章末尾那个驱动问题的落点」、第 539 行「第五章最后留下了一个具体代价」，都具体引用前节结论 |
| checklist 式罗列 / 独立「约束映射表」章节 | ✓ 没有独立的约束映射章节；第 60、101、350、649、704、782、1123、1185、1200 行的表格都嵌在推导中，每张表前后都有解释性段落 |
| 设计方案展示演进（baseline → 中间 → 最终）+ 替代方案对比表 | ✓ GPipe → 1F1B → Zero Bubble(H1/Hr2) → DualPipe → DualPipeV 的演进链完整；§6.3 与 §7.4 给出同口径四方案对比表；中间方案（ZB-H1、interleaved 1F1B）都单独展开过 |
| 「为什么不用 X」遵循「X 解决什么 → 本场景为什么不需要 → 结论」 | ✓ §9.4 对 TBO/SBO 的处理：先讲 TBO 的 yield 点与请求级 micro-batch 拆分解决了什么，再讲推理侧没有 \(W\) 与优化器同步边界所以那套设计没有对应物，最后给结论「把 TBO 说成『推理版的 DualPipe』会把两者的约束条件混淆」；§9.2 对「Megatron 不等于原始 DualPipe」也是同一模式 |
| 无 ASCII 字符画 | ✓ `grep -P '[│┌┐└┘├┤┬┴┼─━┃╔╗╚╝]'` 空命中；Megatron 的节点树已改为 Markdown 表格（第 1123–1129 行） |
| 全角引号 `“”` | ✓ **0 处** |
| `——` 数量 ≤ 1 | ✓ **1 处**（第 1306 行） |
| `[Pending Review]` 被当作知识来源 | ✓ 正文 0 引用；只在文末注释里报告（内容准确性见 P1-3） |
| 短促独立断言句 / 戏剧化动词 | ✓ 抽查未发现；`grep` 未命中「真正的麻烦在这里」一类 |
| 括号收纳个人吐槽的频率（style-guide §9.9 建议 2–4 处） | 偏少：正文里带判断的括号只有第 7 行「（该篇与本文同期写作…）」、第 529 行的一处引用式说明等少数几处，且基本是元信息而非吐槽。属**可选**改进，不计入 P 级 |
| 行内补充用圆括号 / 段落级补充 | ✓ 一致 |

---

## 我无法核实的事项

诚实列出本轮**没查到**或**查不动**的部分（不代表「没问题」）：

1. **`deepseek-ai/profile-data` 的 README 自述**：本地无副本、无网络，我无法验证文章第 1267 行转述的「模拟了完全均衡的专家路由、训练 trace 为简化省略了 PP 通信」是否仍是该仓库 README 的当前内容。文章已自行标注「按 2026-09 检索时的自述，可能已变化」，处理得当。
2. **上游 `main` 是否仍等于 `030ce432`**：本地 `origin/main` 指向它，但无网络，无法确认远端未被推进。文章第 849 行的措辞是「也是上游 `main` 当时的 HEAD」，属可回溯的历史陈述，可接受。
3. **`dualpipe.png` 中「被同一个黑框圈住的格子」**：在可用分辨率下我没能确认画面里存在共享黑框（画面表达重叠的方式是橙绿拼接格）。该说法来自 README 第 11–12 行，文章是转述，见 P2-11。**我没有把它升级为 P0，因为它是忠实转述而非误读。**
4. **`sail-sg/zero-bubble-pipeline-parallelism` 的 tag/commit 内容**：本地无 clone，我无法验证「tag `zero-bubble-v0.1.0` 下的 `handcrafted_zb_v.py` 是 ZB-V 的手工实现」。文章只引 URL 与 tag、不引行号，风险已由写法本身限制。
5. **SGLang 文档中 TBO/SBO 的具体章节**：我核对了路径存在（`docs/docs/advanced_features/expert_parallelism.mdx`），但**没有**读该文件确认 `### Two-Batch Overlap (TBO)` / `### Single-Batch Overlap (SBO)` 两个小节标题与文章第 1271 行的转述完全对应；文章 §9.4 的机制描述我只做了常识级判断。
6. **`references/community/easy-dualpipe/` 那份教学复现**：文章声明「没有运行它，也没有用它作为事实来源」，我同样没有读它——所以也无法判断它是否值得在别处被引用。
7. **两张社区图的原始 URL 是否仍然有效**（知乎 / hackmd / HuggingFace 博客）：无网络，未验证链接可达性，只验证了本地副本存在且 sha256 与 `references/` 一致。
8. **官方两张调度图的原始生成脚本**：未重建（`pics/README.md` 第 77 行也把这条列为「未做」）。我读图只能确认「编号到 9」「图例五类」「有白格」等可直接观察的事实，无法验证格子总数是否精确等于官方声明的调度长度。
9. **`notes/IMAGE-SURVEY-*.md` 里每张图的细节判断**：我只做了**行数覆盖**的全量校验与「是否采用」的抽样阅读，没有逐张复核 433 条的图片描述文字。`IMAGE-VERIFY.md` 第 75 行自己也声明「全量核对的 453 张里有相当一部分只做了内容判断而没有做尺寸与裁切层面的逐像素复核」——这个边界是诚实的。

---

## 流程层面的观察

这套 `/learn-review` 工作流在本轮暴露出四个可操作的改进点，按价值排序：

1. **「修文章」与「修记录」必须同批完成，否则错误会以记录为宿主复活。** 本轮两条 P0 都是这个形态：文章改对了，`codes/README.md` 与 `notes/MATH-VERIFY.md` 留着旧说法；而下一轮审查（包括本轮）恰恰要**通过记录**去核对文章，于是记录本身成了污染源。`pics/README.md` 的 453/31 是同一现象的第三次出现。**建议**：在 `learn-review.md` 的 Step 4 报告格式里固定加一节「受影响的附属记录清单」（文章 / codes README / notes 记录 / pics README / learn-plan / REVISION），修复时逐项打勾。仅凭「本文件不在改动范围内」放行，实际成本会在下一轮以 P0 的形式还回来。

2. **报告的 provenance 必须机器可读（见争议项 4）。** 本轮的补救成本很低（我用一条命令复算了 `md5` 与 `wc -l`），但因为不知道第 2 轮的基准是什么时候、以哪个 revision 记的，我不得不**重做**了全部行号核对——而不是只核对「报告里列出的那几处」。在报告头部加 md5/行数/审核时刻的 YAML front-matter，可以让下一轮把「全量重算」降级为「增量核对」。

3. **`REVISION.md` 的自述正在变成一个不可信层。** 本轮发现 `REVISION.md` 第 142 行声称 P1-5 把计数改成了「435 张（…社区复现代码 2）」，而正文实际写的是 433；第 128 行声称 P0-3 修好了，但同轮的 P2-2 补写的偏差范围本身是错的。这已经是「第三份关于同一件事的描述」，而三份互不一致。**建议**：`REVISION.md` 只记「改了什么文件、改成了什么字面值」，不记「依据」（依据放报告里），并对每轮修复附一条可跑的校验命令（例如 `grep -c '433 张' deep-dive.md`），让自述可被一条命令证伪。

4. **「隔离范围」应当按语义划定，而不是按目录划定。** 第 2 轮把 `notes/IMAGE-SURVEY-zero-bubble.md` 排除在改动之外，理由是「核对文件不在改动范围内」——但核对文件的全部价值就在于它所记录的事实；把它设为不可变，等于把「已证伪的读数」永久固化。**建议**：把「不可改动」的语义限定为「不得删改原始观测记录」，允许并**要求**在发现错误后就地加订正批注（保留原值与订正值、写明轮次与依据）。这既保住了可追溯性，也不会让错误活到下一轮。

---

## 附：本轮结论分级汇总

- **P0：2 条**（`codes/README.md` 第 78 行的稳态序号与脚本输出矛盾；`notes/MATH-VERIFY.md` 第 101 行保留了第 2 轮已撤销的 README 归属）。**均在文章正文之外，正文本身 0 条新增 P0。**
- **P1：5 条**（「串行执行」一词两义未修净；驱动问题自述不实且与计划不符；`readme-4` 的 `[Pending Review]` 报告不完整；`pics/README.md`/`IMAGE-VERIFY.md` 的 453 与 31 未同步；`learn-plan.md` 第 249 行「合计 435」算错）。
- **P2：11 条**（图 18 尺寸 743→742；Table 9 偏差范围 14%–17%→13.74%–18.14%；`put/flush/pop` 的 15 行口径；docstring「只有一句」实为两句；「恰好一半」不精确；1F1B 时长公式的适用前提未补；`IMAGE-VERIFY.md` 的旧 microbatch 口径；`MATH-VERIFY.md` 的 p=10 覆盖面；两套 \(3m/6m\) 解释不一致；共享黑框断言未限定）。
- **第 2 轮 7 项修复：7 项全部核验通过**，其中 6 项彻底、1 项（P0-6）文章已修但记录未同步。
- **对抗性证伪：7 条全部未能推翻**，其中 3 条给出了限定或加强建议。
