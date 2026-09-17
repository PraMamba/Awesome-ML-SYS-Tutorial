# DeepEP 文章配图独立核对报告

> 核对人：独立复核代理（未参与 `learn-plan.md` / `deep-dive.md` / `pics/README.md` 的写作）
> 核对日期：2026-09-10
> 核对对象：`torch/deepep/learn-plan.md`、`torch/deepep/deep-dive.md`、`torch/deepep/pics/README.md`、`torch/deepep/pics/`
> 证据方式：逐张打开图片查看内容 + `references/raw/*/*_content_list.json` 的 MinerU caption 与 `page_idx` + md 引用集合 + 文件尺寸/字节数 + 上游 `git show` 哈希比对

本报告只做核对与取证，不修改被核对文件。

---

## 一、结论摘要

| 检查项 | 结果 |
| --- | --- |
| 122 张抽取图是否逐张处置 | **通过**。`learn-plan.md` 与 `pics/README.md` 都建立了完整台账，`122 = 55 已引用 + 67 未引用` 的拆分可分步复算 |
| 台账分类是否与实际图片一致 | **不通过**。见 §三 第 2 条：把一批**完整可读的单层热力图小图**误判为「小于 15 KB 的碎片」 |
| 正文图片引用是否可解析 | **不通过（P0）**。2026-09-10 10:21 的 `deep-dive.md` 引用 10 张图，其中 **4 个文件名在 `pics/` 中不存在**，另有 6 张已存在的图成为孤儿，见 §四 |
| 命名方案是否单一 | **不通过（P0）**。`learn-plan.md`、`pics/README.md`、`deep-dive.md` 三份文件给出三套互不相同的文件名，见 §四 第 3 条 |
| 图片是否符合「论文原图直接复用」 | **通过**。正文用 `<div>` + `<img>` 引用原图，未重绘、未裁剪；Python 代码块与 Mermaid 混排，无 ASCII 字符画 |
| 上游 DeepEP 结构图是否可用 | **通过**，且比台账写的更强。见 §五 |

---

## 二、122 张抽取图的构成（独立复算）

方法：以 `references/raw/<slug>/*_content_list.json` 中 `type ∈ {image, chart}` 的 `img_path` 为「有图注」集合，以 `references/papers/<slug>/<slug>.md` 的 `images/*.jpg` 引用为「被正文引用」集合，再与 `images/` 目录清单做差集，最后逐张看图复核形态。

| 图片实际形态 | DeepSeekMoE | DeepSeek-V3 | 合计 |
| --- | ---: | ---: | ---: |
| 论文标号图 / 可独立成图的子图 | 7 | 16 | 23 |
| 表格截图（正文宜用 Markdown 重排） | 12 | 32 | 44 |
| 图内小图 / 版面切片（含被切散的附录热力图分栏） | 5 | 27 | 32 |
| 公式 / 表题碎片（正文宜用 LaTeX 重排） | 15 | 6 | 21 |
| 版式块（作者 / 贡献者名单） | 0 | 2 | 2 |
| **合计** | **34** | **88** | **122** |

闭合校验：`23 + 44 + 32 + 21 + 2 = 122`。

补充事实（用于判断「是否值得进正文」）：

- 未引用图共 **67** 张（MoE 27 + V3 40），其中 **45 张（67.2%）小于 15 KB**。
- 但**另有 22 张大于 15 KB**，其中 15 张是 `≥1100×120` 的窄条形热力图。因此「未引用的多数是小于 15 KB 的碎片」这句话对比例成立，对**个体**不成立——不能据此把整批 67 张都称为碎片。

---

## 三、两条重要修正（影响「哪些图值得用」的判断）

### 1. 论文标号图比台账估计的多，且有两张高价值图未被采用

`learn-plan.md` 的采用清单只列 7 张（MoE Figure 2 + V3 Figure 2/4/5/6 + DeepEP 两张结构图），`pics/README.md` 扩展为 12 张。独立复核认为**至少还有两张应当考虑进入正文**，理由不是「图多」，而是它们直接支撑本文的论证链：

| 图 | 源文件 | 支撑什么论点 | 建议 |
| --- | --- | --- | --- |
| DeepSeekMoE Figure 3（归一化消融，1228×551） | `papers/deepseek-moe/images/0c75664b014d1ebcd2304451abb55968b12f7d2c48a59126288d5e9b351bb2df.jpg` | 把「细粒度切分」与「共享专家隔离」两项设计的贡献分开，正好对应 `deep-dive.md` 第 4 章把两者分开讲的写法 | 建议采用；若不采用需在图注或台账中写明理由，而不是归入「表格/碎片」 |
| DeepSeek-V3 Figure 9（专家负载不均衡，1243×598） | `papers/deepseek-v3-report/images/5ed9b418dd78752d04a362f42e53abe50f3dfd6dea682e641aba9888e2ffab14.jpg` | 它是「`get_theoretical_num_sms` 假设 balanced gate distribution」这条局限的**唯一可引用视觉证据**；正文既然要讲这条局限，配图比纯文字更有说服力 | 建议采用（现落在 `pics/` 的补充素材里，已在 `pics/README.md` 登记，但未在正文使用） |

### 2. DeepSeek-V3 附录热力图不是「碎片」，而是一个被 MinerU 打散的多子图版面

`learn-plan.md` 写「DeepSeek-V3 那 38 张密集出现在附录 C 的逐层专家负载图组，属于附录级 ablation」，`pics/README.md` 写「多数小于 15 KB，属表格或子图碎片」。独立逐张查看后需要修正：

像 `c9d3e8c67ffd`（Aux-Loss-Free Layer 1）、`9e44145d2716`（Aux-Loss-Based Layer 3）、`aabc1c04`（Aux-Loss-Based Layer 24）、`347c9ac333c1`（Aux-Loss-Based Layer 19）这些 1190×137 上下的窄条，**打开后是完整、可读、自洽的单层热力图**：

- 纵轴三行齐全（Wikipedia (en) / Github / DM Mathematics）；
- 横轴 1–64 个专家的编号完整；
- 标题写明 layer 号，并有 `Relative Expert Load` 色标（部分带 0–10 图例）。

也就是说，它们不是被切坏的残片，而是**一张多子图版面被按行切开的结果**，每一行单独看都是有效子图。这对本文的意义是：

1. 「不采用」的理由应当改写为「附录逐层可视化的子图之一，本文只需一个整体代表」——这是**相关性**理由，而不是「是碎片所以不能用」；
2. 如果正文要论证专家负载不均衡（配合 `get_theoretical_num_sms` 的 balanced-gate 假设），完全可以从中挑 1–2 张完整小图（例如 Aux-Loss-Based 与 Aux-Loss-Free 各一张）作对照，效果比单引 Figure 9 更直接；
3. 台账里 `pics/README.md` 的一句「多数小于 15 KB」应改为「67 张未引用图中 45 张（67%）小于 15 KB」，并补一句「其余 22 张中多数为附录热力图的完整单层子图」。

---

## 四、P0：正文图片引用与 `pics/` 实物不匹配

### 1. 实测：`deep-dive.md` 引用的 9 个文件名中至少 4 个不存在

对 `deep-dive.md`（2026-09-10 10:21 快照）做 `\./pics/([^")\s]+)` 抽取，得到 10 个引用名；逐个在 `torch/deepep/pics/` 下查找，结果为 **6 个存在、4 个不存在**：

| `deep-dive.md` 中的引用 | 是否存在于 `pics/` |
| --- | --- |
| `./pics/deepseek-moe-fig2-architecture.jpg` | **不存在** |
| `./pics/deepseek-v3-fig2-architecture.jpg` | **不存在** |
| `./pics/deepep-fig-normal-pipeline.png` | **不存在** |
| `./pics/deepep-fig-low-latency-overlap.png` | **不存在** |
| `./pics/deepseek-moe-fig5-activated-experts.jpg` | 存在 |
| `./pics/deepseek-v3-fig4-dualpipe-overlap.jpg` | 存在 |
| `./pics/deepseek-v3-fig5-dualpipe-schedule.jpg` | 存在 |
| `./pics/deepseek-v3-fig6-fp8-mixed-precision.jpg` | 存在 |
| `./pics/deepseek-v3-fig7a-fp8-fine-grained-quant.jpg` | 存在 |

而 `pics/` 目录当前实际有 **12** 个文件。6 个成为孤儿（在目录里但正文没引）：

```
deepep-v1-low-latency-overlap.png
deepep-v1-normal-overlap.png
deepseek-moe-fig2-fine-grained-shared-expert.jpg
deepseek-moe-fig6-gshard-vs-deepseekmoe.jpg
deepseek-v3-fig2-mla-deepseekmoe-architecture.jpg
deepseek-v3-fig7b-fp8-accumulation-precision.jpg
```

即正文与目录**双向不匹配**：4 个引用落空、6 个文件闲置。

> 复现命令：
> ```bash
> cd torch/deepep && python3 -c "
> import re,os
> md=open('deep-dive.md',encoding='utf-8').read()
> refs=sorted(set(re.findall(r'\./pics/([^\")\s]+)', md)))
> print('missing:', [r for r in refs if not os.path.exists('pics/'+r)])
> print('orphans:', [f for f in os.listdir('pics') if f.endswith(('.jpg','.png')) and f not in refs])
> "
> ```

> 这一条是**发布级缺陷**：当前状态下 `deep-dive.md` 有 4 张图会渲染失败。

### 2. 三份文件给出三套命名，必须收敛为一套

同一张 DeepSeekMoE Figure 2，在三份文件里的名字各不相同（`pics/.superseded/` 里还留着第 4 种）：

| 实物/语义 | `learn-plan.md` | `pics/README.md` | `deep-dive.md` 正文 | `pics/.superseded/` |
| --- | --- | --- | --- | --- |
| DeepSeekMoE Fig 2 | `deepseek-moe-fig2-fine-grained-shared-expert.jpg` | 同左 | `deepseek-moe-fig2-architecture.jpg` | `deepseek-moe-fig2-fine-grained-expert-segmentation.jpg` |
| DeepSeek-V3 Fig 2 | `deepseek-v3-fig2-mla-deepseekmoe-architecture.jpg` | 同左 | `deepseek-v3-fig2-architecture.jpg` | `deepseek-v3-fig2-model-architecture.jpg` |
| DeepSeek-V3 Fig 4 | `deepseek-v3-fig4-dualpipe-overlap.jpg` | 同左 | 同左 | `deepseek-v3-fig4-chunk-overlap-strategy.jpg` |
| DeepSeek-V3 Fig 5 | `deepseek-v3-fig5-dualpipe-schedule.jpg` | 同左 | 同左 | `deepseek-v3-fig5-dualpipe-scheduling.jpg` |
| DeepEP V1 normal | `deepep-v1-normal-overlap.png` | `deepep-v1-normal-overlap.png` | `deepep-fig-normal-pipeline.png` | — |
| DeepEP V1 low-latency | `deepep-v1-low-latency-overlap.png` | 同左 | `deepep-fig-low-latency-overlap.png` | — |

**建议的收敛方案**（只需改一处，改动最小）：以 `learn-plan.md` + `pics/README.md` 已达成一致的名字为**唯一命名**（它们对 7 张核心图的命名完全相同），修改 `deep-dive.md` 的 4 个引用名，或按该名字补齐文件。无论选哪种，交付前必须满足：

1. `git diff --check` 通过；
2. 用脚本抽取 `deep-dive.md` 的全部 `./pics/...` 引用，逐个 `test -f` 通过；
3. `pics/` 下无正文未引用的孤儿文件；
4. `pics/.superseded/` 不出现在任何正文引用中（当前满足）。

### 3. `pics/README.md` 的计数与正文实际引用不同步

`pics/README.md` 第 1 节标题写「已复制到本目录的图片（12 张）」，与实际 `ls pics/*.jpg pics/*.png | wc -l` 的 12 一致，**这一条当前是对的**；但该文件在 10:20 之后仍被其他代理继续写入（已增至 337 行并含额外条目），且 `learn-plan.md` 的采用清单只列 7 张、`pics/README.md` 列 12 张、`.superseded/` 另有 4 张。交付前应以**目录实物**为唯一事实来源，把 `learn-plan.md` 的采用清单同步到最终采用集合，避免三处数字继续分叉。

---

## 五、上游 DeepEP 结构图的核对（结论：可用，且理由比台账更充分）

对 `pics/deepep-v1-normal-overlap.png` 与 `pics/deepep-v1-low-latency-overlap.png` 做 sha256，与上游仓库两个 revision 逐一比对：

| 文件 | `pics/` 副本（前 16 位） | tag `v1.2.1` | `01dc3aac`（当前工作树） |
| --- | --- | --- | --- |
| `figures/normal.png` | `cdaf96ab5e65c336` | `cdaf96ab5e65c336` | `cdaf96ab5e65c336` |
| `figures/low-latency.png` | `8694113abc564171` | `8694113abc564171` | `8694113abc564171` |

结论：**这两个文件在 `v1.2.1` 与 `01dc3aac` 下字节完全相同**，因此不存在"引用错 revision"的风险；台账写「来自 tag `v1.2.1`」与写「来自 `01dc3aac`」都对。建议在正文图注里同时给出两个 revision，或统一写「两代 revision 下字节一致」，以免读者误以为版本敏感性存在。

另一条同源事实：`v1.2.1` 下**没有** `docs/legacy.md`（该文件只存在于较新的 revision）。`learn-plan.md` 的「素材缺口」一节已正确识别并给出处理方式（V1 benchmark 引用 `01dc3aac` 下的该文件并写明它描述的是 legacy 路径），这一条**复核通过**。

---

## 六、逐张归类的完整性抽检

为保证"一张都不能漏"，对两份 difference 集合做了全量打印与归类，并抽查了容易误判的样本：

| 抽查对象 | 期望形态 | 实际看到 | 归类是否正确 |
| --- | --- | --- | --- |
| `deepseek-moe/images/f77205913b6f…` | Figure 2 三栏对比 | (a) 传统 top-2 / (b) 细粒度切分 K=4 / (c) 共享专家隔离 K=3 | 正确（已采用） |
| `deepseek-moe/images/ac61dc9d6ddc…` | Figure 1 散点 | 激活参数量—平均性能平面，DeepSeekMoE 16B 高于同量级模型 | 正确（未采用，属效果类） |
| `deepseek-moe/images/0546107fcba0…` | 公式碎片 | `h_t^l = Σ(g·FFN) + u_t^l` 的 LaTeX 截图 | 正确（碎片） |
| `deepseek-moe/images/0ef10fd83591…` | 表格截图 | Open LLM Leaderboard 分项对照表 | 正确（表格） |
| `deepseek-v3-report/images/434ce896e9c6…` | 热力图分栏 | Aux-Loss-Based Layer 2，含三领域纵轴与 1–64 专家横轴 | **需修正**（是完整子图，不是碎片） |
| `deepseek-v3-report/images/5b3ce649bffe…` | 作者名单版式块 | Research & Engineering 名单 | 正确（版式块） |
| `deepseek-v3-report/images/f1df2e9f8ff5…` | Figure 7(a) 细粒度量化 | 含 Fprop / 1×128 分块与 `1×128` 标注 | 正确（论文标号子图） |
| `deepseek-v3-report/images/8afcbbea5450…` | Figure 7(b) 精度提升 | Weight/Input Gradient 双路径、CUDA Core 提升区间 | 正确（论文标号子图） |

---

## 七、交付前必须闭环的动作清单

按优先级排列，全部为可机械验证项：

1. **[P0]** 收敛图片命名：修正 `deep-dive.md` 中 4 个不存在的引用，或按这些名字补文件；随后跑引用存在性脚本，要求 9/9（或最终引用数）全部 `test -f` 通过。
2. **[P0]** 重新校准 `pics/README.md` 的「已复制 N 张」计数，使其等于 `ls pics/*.jpg pics/*.png | wc -l`。
3. **[P1]** 修正台账对 67 张未引用图的描述：把「多数小于 15 KB 的碎片」改成带比例的准确表述，并承认其中 22 张（含 15 张窄条热力图）是**完整子图**而非碎片。
4. **[P1]** 决定 DeepSeekMoE Figure 3 与 DeepSeek-V3 Figure 9 是否进入正文；若决定不用，把理由从"碎片/无关"改为具体的相关性判断。
5. **[P1]** 在正文图注中把 DeepEP 两张结构图的 revision 口径统一为「`v1.2.1` 与 `01dc3aac` 字节一致」或等价表述。
6. **[P2]** 清理 `pics/.superseded/`：确认正文零引用后删除，或在 `pics/README.md` 保留说明并保留目录（当前已有说明，可接受）。

---

## 八、本次核对未覆盖的范围

- 未核对正文文字与源码的事实一致性（属 learn-write / learn-review 的其他维度）。
- 未核对数学公式（本项目另有数学核对线）。
- 未核对 `codes/` 下教学实现的正确性，仅确认目录存在。
- 未核对交叉引用与 README 发布状态。
