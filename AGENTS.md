# Awesome-ML-SYS-Tutorial

这是一个以 ML Systems 学习笔记为主、同时保留少量可运行练习的知识库。工作时先核对当前文件、对应源码和运行结果，再下结论；README、旧文章、学习计划和 proposal 可以帮助定位问题，但不能替代当前实现的事实。

> 本文件只规定仓库级边界和最容易出错的工作流。/learn 的详细行为继续以 .claude/commands/ 为准，写作细则继续以 .learn/ 为准，避免在多个入口复制同一套长规则。

## WHAT：项目边界

仓库内容主要包括 RLHF 系统、SGLang 推理、PyTorch/FSDP 分布式基础、Transformer/模型架构、工程工具，以及配套的源码走读和学习实验。它不是一个有统一构建流程的软件包：当前根目录没有 pyproject.toml、Makefile、统一测试入口或 CI 配置，代码验证必须按模块执行。

### 指导文件与事实来源

- 根目录的 AGENTS.md 是仓库级工作规范。当前未发现 AI.md；CLAUDE.md 只有 @AGENTS.md 指针，不要在两个文件中维护两份规则。
- 涉及 /learn 命令的执行行为，以 .claude/commands/*.md 为准；.learn/skill.md 是命令地图和约束的文档说明，不是另一套实现。
- 涉及文章风格、学习深度、引用和翻译，以 .learn/config.md、.learn/index/style-guide.md 和对应模板为准；.learn/readme.md 主要记录设计动机和演进背景。
- 文章是否已正式发布，先查 README.md 和 README-cn.md；[Pending Review] 文章不能作为 /learn 的风格参考或知识来源。knowledge-graph.json 只收录满足发布条件的文章，不能反过来替代 README 的发布判断。
- 源码事实以当前实际源码、固定的 commit 和可复现输出为准。文档与源码冲突时采用源码，并在相关笔记中记录有意义的差异，不要静默照抄旧文档。

## WHY：工作目标

这个仓库同时服务两种读者：希望沿学习路径理解系统设计的人，以及需要通过代码验证分布式概念的人。因此每次改动都要能回答三个问题：事实来自哪里，读者如何复现，哪些结论只是模拟或个人判断。文章、/learn 工作流和 runnable code 是三个相互关联但不能混为一谈的层次。

## HOW：开始、修改与交付

### 开始前

先确认位置和工作树，不要覆盖已有改动：

    pwd
    git status --short --branch
    find .. -name AGENTS.md -o -name AI.md -o -name CLAUDE.md

随后按任务读取最小但完整的上下文：

1. 改文章或索引时，先读对应的 README.md、README-cn.md、文章全文和相邻目录命名；需要 /learn 时再读对应 command、config.md、style-guide.md、知识图谱和模板。
2. 做源码分析时，先读实际源码和版本信息，再写解释或引用；外部 GitHub 代码链接必须锁定 commit hash 和行号，不能用 main/master 行号。
3. 改 runnable code 时，先读该目录的说明和已有测试，确认依赖、设备要求和入口，不把某个局部命令包装成全仓库验证。

### 修改后

- 只改用户要求的文件；保留无关的 staged、unstaged 和 untracked 内容，尤其不要用 reset --hard、checkout、清空目录或未经确认的 stash 来“整理”工作树。
- 文档改动至少执行 git diff --check，并检查新增的相对链接、图片路径和代码块引用；代码改动执行对应模块的测试或最小运行示例。
- 交付前再次运行 git status --short --branch，明确说明验证范围、未验证的外部依赖，以及是否有超出任务范围的现有改动。
- 默认不 commit、push、改写历史、发布文章、下载大型模型或启动高成本 GPU 任务；这些动作需要用户明确授权。

## 文件结构与知识地图

以下是当前主要目录，不是所有文章的穷举清单：

| 路径 | 作用 | 修改时先看 |
| --- | --- | --- |
| .claude/commands/ | /learn-plan、/learn-write、/learn-review、/learn-add 的实际命令定义 | 对应 command 文件 |
| .learn/ | /learn 的 proposal、共享上下文、风格、模板和知识图谱 | .learn/skill.md、config.md |
| README.md、README-cn.md | 文章目录、语言版本和发布状态入口 | 两个 README 的对应条目 |
| rlhf/ | slime、verl、AReal、OpenRLHF、RL 算法与系统设计 | 目标系列的前序文章 |
| sglang/ | SGLang 核心架构、scheduler、KV cache、Omni、量化和 RL 接口 | 目标模块的源码版本 |
| torch/ | PyTorch Distributed、NCCL、CUDA Graph、内存和并行策略 | 目标子目录的 readme/代码 |
| torch/fsdp2/ | 从分布式基础到 FSDP2、DTensor、checkpoint 和组合并行的教程与代码 | 章节 readme 和 codes/ |
| transformers/ | Attention、special token 和模型架构笔记 | 对应模型/机制的文章 |
| engineer/ | Docker、uv、缓存等开发实践 | 具体实践文档 |
| kernels/ | kernel/data layout 相关笔记和图示 | 目标主题文档 |

### torch/ 下的可运行内容

当前 Python 示例不只位于 torch/torch-distributed/codes/：

- torch/torch-distributed/codes/ 是 torch.distributed 通信 API 示例，按各自说明使用 python 或 torchrun。
- torch/parallel_dims_lab/ 是 ParallelDims 学习实验，包含 01 到 10 脚本、共享小工具、pytest 测试和 notes/README.md。除 01_collectives.py 的 CPU/Gloo 多进程演示外，其余脚本是 CPU 张量或本地列表模拟，不测量真实通信性能。
- 该实验的事实基线、源码 commit、文档差异和运行命令集中记录在 [torch/parallel_dims_lab/notes/README.md](torch/parallel_dims_lab/notes/README.md)，不要从旧学习文章猜测当前 API。

运行和验证命令以 [torch/parallel_dims_lab/notes/README.md](torch/parallel_dims_lab/notes/README.md) 为唯一入口；需要验证时进入该目录，按 notes 中的测试或脚本说明执行。

不要把这些脚本的模拟结果写成真实 GPU 通信、吞吐或显存 benchmark；01 的 Gloo 结果也只证明该演示的通信逻辑在 CPU 上成立。

## .claude/ 与 .learn/：/learn 工作流

这两个 module 不是普通文章目录。.claude/commands/ 描述“命令如何执行”，.learn/ 提供命令按需读取的知识和写作约束；并非每个命令都需要所有上下文文件，修改一处契约时要检查另一处是否仍然准确。

### 四个命令的职责边界

| 命令 | 实际定义 | 允许的核心动作 |
| --- | --- | --- |
| /learn-plan | .claude/commands/learn-plan.md | 读取知识图谱、风格、配置和草稿，定位前置知识，广泛查找资源，生成学习路线和驱动问题 |
| /learn-write | .claude/commands/learn-write.md | 先读源码和资料，再按“概念 → 模型/场景 → 代码”完成文章；涉及源码时不得凭记忆写逻辑 |
| /learn-review | .claude/commands/learn-review.md | 检查风格、计划完成度、引用、深度、交叉引用和推导链；翻译是单独步骤，必须得到用户明确授权 |
| /learn-add | .claude/commands/learn-add.md | 仅处理用户主动指定的已发布文章；展示待写入条目并获得确认后，才更新知识图谱 |

### 共享上下文的使用顺序

1. .learn/index/knowledge-graph.json：文章元信息、系列、前置关系和引用关系。
2. .learn/index/style-guide.md：开篇、递进推导、概念拆解、代码引用、格式、语言和翻译细则。
3. .learn/config.md：topic 的学习深度、目录分类、路径命名、引用和语言设置。
4. .learn/templates/：按文章类型选择 code-walkthrough、sys-design、paper-reading 或 tutorial，不要把模板占位符直接当成事实。
5. .learn/readme.md：/learn 的设计说明、历史例子和知识图谱字段解释。

### /learn 的硬约束

- 写作、计划和审查以中文为先，技术术语按仓库惯例保留英文；代码注释保留源码语言。
- 任何 [Pending Review] 文章都不能作为风格范本或知识来源。
- 外部源码引用必须使用具体 commit hash；翻译时 repo 内部链接必须切换到目标语言版本，外部 URL 保持不变。
- 需要外部资料时，应结合 PR、官方文档和固定版本的 GitHub 源码广泛核查，不要只根据用户给出的单一链接补全事实。
- 文章主线遵循“概念 → 模型/场景 → 代码”；源码分析先读源码，代码块后解释执行逻辑，不能用假想实现冒充生产源码。
- 不用 ASCII 字符画；流程图用 Mermaid，对比用 Markdown 表格。概念、约束和设计选择要通过推导连接，不能堆成无上下文的 checklist。
- /learn-write 和 /learn-review 不会自动更新 knowledge-graph.json；新文章只有在 README 正式列出、没有 [Pending Review] 且确属作者作品时，才可由用户主动调用 /learn-add 收录。
- 未经用户明确授权，不执行翻译；也不因为文章写完或 review 通过就自动发布、改 README 或更新知识图谱。

### 修改 /learn 规则时

- 改命令实际行为，先更新对应的 .claude/commands/*.md，再检查 .learn/skill.md、.learn/readme.md 和相关模板是否需要同步；不要只改说明而不改执行定义。
- 改风格或发布契约时，检查 style-guide.md、config.md、README 状态和 command 中的检查项，避免出现互相矛盾的硬约束。
- 知识图谱的新增、删除、路径变更和引用同步走 /learn-add 定义的确认流程；不要在文章写作过程中顺手直接编辑 JSON。

## Codex：/learn 工作流路由

Codex 不把 `.claude/commands/` 当作 shell 命令执行；这些文件是本仓库的 workflow specification。根据用户意图选择对应定义文件，按阶段推进，不要跳过发布和写入门槛。

用户只询问流程时，只解释并给出下一步，不写文件；只有出现“生成、完成、修改、审查、翻译、发布、收录”等明确意图时，才进入对应阶段。目标路径不明确时先报告候选路径，不覆盖已有草稿。

| 用户意图 | 路由文件 | 产物 | 默认写入边界 |
| --- | --- | --- | --- |
| 想学习一个 topic、整理路线或分析已有草稿 | `.claude/commands/learn-plan.md` | 学习计划、驱动问题、前置知识和草稿完成度 | 只写计划，不改草稿、README 或知识图谱 |
| 想完成、重写或补齐文章 | `.claude/commands/learn-write.md` | 中文文章草稿 | 只写用户指定的文章路径；源码分析必须先读源码 |
| 想审查、校对或判断文章是否完成 | `.claude/commands/learn-review.md` | 中文检查报告 | 默认只读和报告，不改稿、不翻译 |
| 明确要求翻译 | `.claude/commands/learn-review.md` 的翻译步骤 | 对应语言版本 | 需明确授权，且先处理 P0 问题并检查内部链接 |
| 明确要求正式收录文章 | `.claude/commands/learn-add.md` | 知识图谱条目 | 文章已发布且用户确认后才写 JSON |

### Codex 执行协议

1. **识别状态。** 先运行 `git status --short --branch`，确认草稿、计划和目标路径；通过两个 README 判断发布状态，不凭文件名猜测。
2. **加载上下文。** 读取对应 command；涉及写作时按需读取 `.learn/index/knowledge-graph.json`、`style-guide.md`、`config.md` 和文章模板，不把所有 module 内容一次性塞入上下文。
3. **锁定事实。** 涉及源码时先读取当前源码和 `git rev-parse HEAD`；外部代码引用使用固定 commit。文档与源码冲突时记录冲突并按源码处理。
4. **执行单一阶段。** 计划、写作、审查、翻译和收录是不同阶段；用户只要求审查时不得顺手改稿，用户只要求写作时不得顺手发布或更新知识图谱。
5. **验证并交接。** 输出本阶段读取的关键文件、实际改动、验证结果、未解决的事实冲突和下一阶段入口；文档改动执行 `git diff --check`，代码改动执行目标模块测试。

### 当前 ParallelDims 草稿的路由

`torch/parallel_dims_lab/01-parallel-dims-source-walkthrough.md` 当前不在两个 README 的正式目录中，因此状态是 draft。推荐先以 `code-walkthrough`、`understand-reproduce` 路由 `/learn-plan`，再以计划调用 `/learn-write` 和 `/learn-review`；只有正式发布后才考虑 `/learn-add`。源码基线和差异清单以 [`torch/parallel_dims_lab/notes/README.md`](torch/parallel_dims_lab/notes/README.md) 及其中记录的固定 commit 为准。

## 内容与写作规范

- 新文章放入与主题匹配的一级目录和 topic 子目录；新增或正式发布文章时同步检查 README.md 与 README-cn.md 的条目、语言链接和 [Pending Review] 标记。
- 文件命名沿用同目录已有惯例。英文后缀在本仓库并不统一，先查实际文件和知识图谱，不要凭直觉拼出 -en 或 _en。
- 图片放在文章附近已有的 pics/、pic/、img/、imgs/、figs/、assets/、static/ 或 image/ 目录中，并使用相对路径。
- 文章中的数字、性能、源码行为和版本号都要能回溯到源码、固定版本、运行输出或明确标注的假设；模拟、proxy、草稿和正式结果不能混写。
- 只做拼写或排版修改时，不要无关地重写文章结构、翻译全文或调整知识图谱关系。

## Always Do / Ask First / Never Do

### Always Do

- 先读最近的仓库规范和目标模块文档；只有涉及源码或代码时，再读实际源码和现有测试。
- 把路径、commit、命令、依赖、设备要求和验证结果写进需要长期复现的 notes 或文章中。
- 对文档与源码的差异给出明确记录；对未验证的外部行为保留不确定性，不把推测写成事实。
- 将代码改动和文档改动分别做最小验证，并在交付时报告未覆盖的范围和现有工作树状态。

### Ask First

- commit、push、rebase、reset、删除/移动文件、修改历史或替用户解决冲突。
- 下载超过本地已有范围的大型权重/数据、安装新的全局依赖、占用 GPU 或启动昂贵实验。
- 将草稿改为正式发布、改变 README 的发布状态、翻译文章、或修改 knowledge-graph.json 的核心关系。
- 对现有用户改动进行清理、格式化或批量重写，哪怕这些文件与当前主题相邻。

### Never Do

- 不把 [Pending Review] 内容当作已发布事实、风格标准或知识图谱输入。
- 不引用会漂移的 main/master 源码行号，不在未读源码时解释内部实现。
- 不把 CPU 模拟、单元测试通过或局部 smoke 结果包装成真实 GPU 性能、官方 benchmark 或科学有效性结论。
- 不通过复制一份新的平行指导文件来解决规则冲突；先更新现有权威文件并检查引用它的文档。
- 不覆盖、丢弃或“顺手整理”用户已有的提交、索引、未提交文件和 untracked 文件。

## Common Pitfalls

1. **把旧的 torch/parallel_dims 文章当成 API 文档。** 先查实验 notes 和固定源码版本，冲突处按源码并记录差异。
2. **把 .learn/skill.md 当成命令实现。** 它解释四个命令和共享上下文，实际执行约束在 .claude/commands/。
3. **把知识图谱当成发布入口。** 发布状态先从两个 README 判断，且 /learn-add 只在用户主动调用并确认后写入。
4. **把“仓库没有统一测试”误读成“仓库没有可运行代码”。** 进入具体模块后使用该模块的 readme、tests 和运行命令。
5. **把中文文章翻译成英文后保留中文内部链接。** 翻译前后逐条检查目标语言文件是否存在；没有对应版本时明确标注语言限制。

## Progressive Disclosure：详细指引

需要进一步判断时，按下面顺序下钻，不要一开始加载整个仓库：

| 任务 | 详细入口 |
| --- | --- |
| 了解 /learn 的设计背景与知识图谱字段 | [.learn/readme.md](.learn/readme.md) |
| 查看命令的实际流程和写入边界 | [.learn/skill.md](.learn/skill.md) 与 [.claude/commands/](.claude/commands/) |
| 写作、审查或翻译 | [.learn/index/style-guide.md](.learn/index/style-guide.md)、[.learn/config.md](.learn/config.md)、[.learn/templates/](.learn/templates/) |
| 判断文章是否可被 /learn-add 收录 | [README.md](README.md)、[README-cn.md](README-cn.md)、[.claude/commands/learn-add.md](.claude/commands/learn-add.md) |
| 运行或修改 ParallelDims 学习实验 | [torch/parallel_dims_lab/notes/README.md](torch/parallel_dims_lab/notes/README.md)、对应 .py 和 tests/ |
| 理解 PyTorch/FSDP 教程代码 | 对应 torch/fsdp2/**/readme.md、torch/**/readme.md 和同目录 codes/ |
