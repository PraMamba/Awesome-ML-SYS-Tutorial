# DeepSeek-V3 Technical Report（arXiv 2412.19437v2）图片逐张核对

对 `torch/dualpipe/references/papers/deepseek-v3-report/images/` 下每一张 MinerU 抽取图片逐个读图判断，
产出「参考图 → 建议章节」映射表，供 `torch/dualpipe/deep-dive.md` 选图使用。

读图方法：88 张全部用 `read_image` 实际查看；图号与语义以 `deepseek-v3-report.md` 的图片引用行号及前后上下文（图注 / `Figure N:` / 所属小节标题）为索引，再与读图结果互相印证。
md 自身的图号冲突另用 arXiv 官方 HTML 版（ar5iv，同一论文 `2412.19437`）核对，见「存疑与分歧」第 1 条。

## 汇总

- `images/` 实际文件数（`ls | wc -l`）：**88**
- 本清单覆盖行数：**88**
- 真插图：**48**；公式截图：**28**；表格截图：**9**；页眉/装饰/碎片：**3**
  - 48 张真插图 = 正文 11 张（Figure 1-10，其中 Figure 7 为 a/b 两个文件）+ 附录 Figure 11 的 37 张子图切片
- 建议采用：**2** 张（Figure 4、Figure 5）；另有 1 张表格内容（Table 2）建议转写为 Markdown 表格而不引图
- md 中实际被引用的图片：**48** 张；未被任何一行引用的游离图片：**40** 张（9 张表格截图 + 28 张公式截图 + 3 张名单/行内片段）
- 注意：`deepseek-v3-report.md` 共 48 处图片引用，但其附录 C 的 37 张子图与正文 11 张合计正好 48，即**所有表格与公式截图都没有被 md 引用**

## 逐张清单

分类口径：(a) 真插图 / 架构图 / 调度图 / 曲线图；(b) 公式截图；(c) 表格截图；(d) 页眉 / 装饰 / 图内碎片。
「md 引用行」中的 `—` 表示该文件在 `deepseek-v3-report.md` 中完全没有被引用。

| 文件名 | md 引用行 | 图号/表号 | 类别 | 内容（读图后的事实描述，含图中可读的关键标注/数值） | 结论 | 建议章节 |
| --- | --- | --- | --- | --- | --- | --- |
| `17d0e3198304f86d33133ce7d993f69759822d45545b139e485cd266c457c061.jpg` | 11 | Figure 1 | 插图 | 分组柱状图。x 轴 6 个基准：MMLU-Pro(EM)、GPQA-Diamond(Pass@1)、MATH 500(EM)、AIME 2024(Pass@1)、Codeforces(Percentile)、SWE-bench Verified(Resolved)；6 个模型并列：DeepSeek-V3(蓝色斜纹)、DeepSeek-V2.5、Qwen2.5-72B-Instruct、Llama-3.1-405B-Instruct、GPT-4o-0513、Claude-3.5-Sonnet-1022。图例中 DeepSeek-V3 柱顶标数值 75.9 / 59.1 / 90.2 / 39.2 / 51.6 / 42.0；y 轴 Accuracy / Percentile (%)。 | 不采用：真插图，但与 DualPipe 主题无关 | — |
| `d9af045bad6955d9d437d94a3e81725e70e117b0525d70b9ab0a982019177d77.jpg` | 127 | Figure 2 | 插图 | 模型总体架构图。左侧 Transformer Block ×L（Feed-Forward Network、RMSNorm、Attention、RMSNorm 各 2 层残差）；右上 DeepSeekMoE：输入 u_t → Router → Top-K_r → N_s 个共享专家(绿)+N_r 个路由专家(蓝) → 输出 h'_t；右下 Multi-Head Latent Attention(MLA)：Latent c_t^Q / c_t^KV、apply RoPE、concatenate、Multi-Head Attention，标注 Cached During Inference 的两个方框。 | 不采用：真插图，但与 DualPipe 主题无关 | — |
| `6a5b2be7a664fb9473828efd43ce95f9667fc2121f87a56dd4ed32925c244ad8.jpg` | 238 | Figure 3 | 插图 | MTP 实现图。三列：Main Model(Next Token Prediction)、MTP Module 1(Next² Token Prediction)、MTP Module 2(Next³ Token Prediction)；每列自下而上为 Input Tokens / Embedding Layer / Transformer Block ×L / Output Head / Cross-Entropy Loss，损失记为 L_Main、L^1_MTP、L^2_MTP；Embedding Layer 与 Output Head 之间有 Shared 虚线标注；上方 Target Tokens 依次右移 t_2..t_5 / t_3..t_6 / t_4..t_7。 | 不采用：真插图，与 DualPipe 调度无直接关系；仅在讲到 MTP 共享 embedding/head 时可作旁证 | — |
| `ea1a52decfb55f603ceea7de1e14bbb72c8e161ed45f792c75afdcfc85a8b59b.jpg` | 291 | Figure 4 | 插图 | 一对前向/反向 chunk 的重叠策略时间轴，两行：上行 Computation、下行 Communication，右下角标 Time →。上行从左到右依次为 MLP(B)▲、MLP(W)▲、MLP(F)△、ATTN(B)▲、ATTN(W)▲、ATTN(F)△；下行依次为 DISPATCH(F)△、DISPATCH(B)▲、COMBINE(F)△、PP(紫色格)、COMBINE(B)▲。底部图例：△ Forward chunk、▲ Backward chunk。左右两边缘与前后半段交界处各有红色竖条（barrier）。橙色=前向、绿色=backward for input、浅蓝=backward for weights、紫色=PP 通信、红色=barrier，与图注逐一对应。 | 采用 | §4 DualPipe 配对（同时服务 §1 两种等待、§6 口径） |
| `e6f2d6c79a43060cbcca6151b7ab9f65a07022a2b6785819b193e4c6639157a4.jpg` | 306 | Figure 5 | 插图 | 官方 DualPipe 调度示例。y 轴为 8 行 Device 0..Device 7，右下角标 Time →，每行由按时间排列的彩色方格构成；前向方向 micro-batch 标有数字 0-9，反向方向方格留空（图注说明省略 batch ID）。图例 5 项：Forward(橙)、Backward(绿)、Backward for input(绿)、Backward for weights(浅蓝)、Overlapped forward & Backward(橙绿斜纹)。可读细节：Device 0 行前段为连续编号 0 1 2 3 4 5 6 的单色块，中段出现 '0 8'、'1 9' 等深浅交错的成对块，末端出现 6/7/8/9 与浅蓝块；越靠后的 device 其编号段整体右移。图中不含任何表格。 | 采用 | §4 DualPipe 双向布局与 (F0,B1,F1,B0) 配对；§5 与 §6 复算气泡时作参照 |
| `fb915dd3aeea3d7d5996a4d9cbcad0acab98d9ca4cd190c80373101c459c4f82.jpg` | 336 | Figure 6 | 插图 | FP8 混合精度框架图（仅 Linear 算子）。Input(BF16) → Fprop 乘加(FP32 累加) → Output → To BF16；Weight 双向；Wgrad 乘加 → Weight Gradient(FP32) → Master Weight(FP32) → Optimizer States；Dgrad 由 Output Gradient(BF16) → To FP8 进入乘加 → Input Gradient(BF16)。各张量旁标注 FP8/BF16/FP32。 | 不采用：真插图，属 FP8 训练章节，与 DualPipe 无关 | — |
| `8afcbbea54504587dfc7fc155c4f5c6cf4263d8cab7c1d77e2be58235bf6f860.jpg` | 347 | Figure 7(a) | 插图 | 细粒度量化示意图（子图 a）。Input 按 1×N_C 分组、Weight 按 N_C×N_C 分块，各自带 Scaling Factor；下方 Tensor Core 与 CUDA Core 两组等式：Tensor Core 为『紫 = 绿 × 黄』，CUDA Core 为『紫 * 青 * 紫 = 紫』；图注 (a) Fine-grained quantization。 | 不采用：真插图，属 FP8 量化，与 DualPipe 无关 | — |
| `f1df2e9f8ff579bf54458b74531b9b78ce5282ae199c170e1ab612451553a31a.jpg` | 349 | Figure 7(b) | 插图 | 提升累加精度示意图（子图 b）。上半为 Tensor Core 内的 WGMMA 1 与 WGMMA 4 两级乘法（Low Prec Acc 紫块 / GEMM Input 黄绿块）；下半为 CUDA Core 侧 Output 沿 N_C Interval 推进，图例给出 Scaling Factor(青) 与 FP32 Register(粉)。图注 (b) Increasing accumulation precision。 | 不采用：真插图，属 FP8 累加精度，与 DualPipe 无关 | — |
| `b855efd83f9ebb922e3f6a2d332d7b594546daeab44403ee509f7ce33a093bb3.jpg` | 466 | Figure 8 | 插图 | NIAH 测试热力图。y 轴 Document Depth Percent (%) 0→100（刻度 0,7,14,...,100），x 轴 Context Length 2K→128K（刻度 2K,11K,20K,...,128K），右侧 colorbar Score 1→10。整片区域近乎全绿（Score≈10），说明各深度/长度均检索成功。注意：该抽取图顶部标题被裁掉，标题文字不完整。 | 不采用：真插图，属长上下文评测，与 DualPipe 无关 | — |
| `5ed9b418dd78752d04a362f42e53abe50f3dfd6dea682e641aba9888e2ffab14.jpg` | 543 | Figure 9 | 插图 | 专家负载热力图，4 个横向子条：Aux-Loss-Based Layer 9、Aux-Loss-Free Layer 9、Aux-Loss-Based Layer 18、Aux-Loss-Free Layer 18；每条的 y 轴为 Wikipedia (en) / Github / DM Mathematics，x 轴为专家索引 1→64；底部共享 colorbar Relative Expert Load 0→10。Based 版整体浅色（最大仅个别橙色块），Free 版出现明显深红/橙高负载块（如 Layer 9 的 26-27、45-47、58 号专家）。 | 不采用：真插图，属 MoE 负载均衡，与 DualPipe 无关 | — |
| `d94eaa157b4f5c553067a7459822672e35b118eb9148815929d4adcb204f9eb5.jpg` | 1009 | Figure 10 | 插图 | 损失曲线对比，左右两幅：左『BF16 v.s. FP8 on 16B DeepSeek-V2』（x 轴 Tokens/B 0→1200+，y 轴 Loss 1.8→2.5），右『BF16 v.s. FP8 on 230B DeepSeek-V2』（x 轴 0→800+）；两幅各含一个虚线放大窗，纵轴为 Loss diff，显示逐 step 的 BF16−FP8 差值波动；图例 BF16(蓝线) / FP8(红线)。 | 不采用：真插图，属 FP8 精度验证，与 DualPipe 无关 | — |
| `434ce896e9c65f2da4251f122a3d0956acd80b8dd04b466c08df98e9c43dd9d0.jpg` | 1024 | Figure 11 子图 | 插图 | Figure 11 的子图切片，抽取图中内嵌标题为「Aux-Loss-Based Layer 2」。每条 y 轴为 Wikipedia (en) / Github / DM Mathematics，x 轴为专家索引 1→64。Aux-Loss-Based 条整体浅黄、仅在少数专家上出现淡橙块；Aux-Loss-Free 条出现成片橙/深红高负载块（相对负载可达 8-10）。 | 不采用：真图但属附录 C 专家特化，与 DualPipe 主题无关 | — |
| `caa21ae99c296b14ba56c52aa1440de4b6b8fe858646d9dc27f1be399020c91f.jpg` | 1026 | Figure 11 子图 | 插图 | Figure 11 的子图切片，抽取图中内嵌标题为「Aux-Loss-Based Layer 1」。每条 y 轴为 Wikipedia (en) / Github / DM Mathematics，x 轴为专家索引 1→64。Aux-Loss-Based 条整体浅黄、仅在少数专家上出现淡橙块；Aux-Loss-Free 条出现成片橙/深红高负载块（相对负载可达 8-10）。 | 不采用：真图但属附录 C 专家特化，与 DualPipe 主题无关 | — |
| `9e44145d2716be3ebd5ef0a1b1c586c22f97b06cac7d24ca9cca6d66e2cfc4df.jpg` | 1028 | Figure 11 子图 | 插图 | Figure 11 的子图切片，抽取图中内嵌标题为「Aux-Loss-Based Layer 3」。每条 y 轴为 Wikipedia (en) / Github / DM Mathematics，x 轴为专家索引 1→64。Aux-Loss-Based 条整体浅黄、仅在少数专家上出现淡橙块；Aux-Loss-Free 条出现成片橙/深红高负载块（相对负载可达 8-10）。 | 不采用：真图但属附录 C 专家特化，与 DualPipe 主题无关 | — |
| `c9d3e8c67ffdbf880948229cab32e4b872d53bac7835e3afb73b985891c915a4.jpg` | 1030 | Figure 11 子图 | 插图 | Figure 11 的子图切片，抽取图中内嵌标题为「Aux-Loss-Free Layer 1」。每条 y 轴为 Wikipedia (en) / Github / DM Mathematics，x 轴为专家索引 1→64。Aux-Loss-Based 条整体浅黄、仅在少数专家上出现淡橙块；Aux-Loss-Free 条出现成片橙/深红高负载块（相对负载可达 8-10）。 | 不采用：真图但属附录 C 专家特化，与 DualPipe 主题无关 | — |
| `f0895ea100acb10d1d9c64713d2fd73a5e6f475c6dbce2138cc3ce367b890fe3.jpg` | 1032 | Figure 11 子图 | 插图 | Figure 11 的子图切片，抽取图中内嵌标题为「Aux-Loss-Based Layer 4」。每条 y 轴为 Wikipedia (en) / Github / DM Mathematics，x 轴为专家索引 1→64。Aux-Loss-Based 条整体浅黄、仅在少数专家上出现淡橙块；Aux-Loss-Free 条出现成片橙/深红高负载块（相对负载可达 8-10）。 | 不采用：真图但属附录 C 专家特化，与 DualPipe 主题无关 | — |
| `1065a3a9921a63bd05559b4aeee526e8f89195e4633735683e2f74d4828f9740.jpg` | 1034 | Figure 11 子图 | 插图 | Figure 11 的子图切片，抽取图中内嵌标题为「Aux-Loss-Free Layer 3」。每条 y 轴为 Wikipedia (en) / Github / DM Mathematics，x 轴为专家索引 1→64。Aux-Loss-Based 条整体浅黄、仅在少数专家上出现淡橙块；Aux-Loss-Free 条出现成片橙/深红高负载块（相对负载可达 8-10）。 | 不采用：真图但属附录 C 专家特化，与 DualPipe 主题无关 | — |
| `96c643d8d1606db82e793491afdddfdd76d45c762735d4f617a389ee09a15888.jpg` | 1037 | Figure 11 子图 | 插图 | Figure 11 的子图切片，抽取图中内嵌标题为「Aux-Loss-Based Layer 7」。每条 y 轴为 Wikipedia (en) / Github / DM Mathematics，x 轴为专家索引 1→64。Aux-Loss-Based 条整体浅黄、仅在少数专家上出现淡橙块；Aux-Loss-Free 条出现成片橙/深红高负载块（相对负载可达 8-10）。 | 不采用：真图但属附录 C 专家特化，与 DualPipe 主题无关 | — |
| `4062dfc1c5c64cd67e414f99109ace033c4403d2dd05c84728cdd6052558ecc1.jpg` | 1039 | Figure 11 子图 | 插图 | Figure 11 的子图切片，抽取图中内嵌标题为「Aux-Loss-Free Layer 7」。每条 y 轴为 Wikipedia (en) / Github / DM Mathematics，x 轴为专家索引 1→64。Aux-Loss-Based 条整体浅黄、仅在少数专家上出现淡橙块；Aux-Loss-Free 条出现成片橙/深红高负载块（相对负载可达 8-10）。 | 不采用：真图但属附录 C 专家特化，与 DualPipe 主题无关 | — |
| `bf214eb53764663996c3e3825d099d7c47cd45f1759ec0a2e01deda142a73cac.jpg` | 1041 | Figure 11 子图 | 插图 | Figure 11 的子图切片，抽取图中内嵌标题为「Aux-Loss-Based Layer 8」。每条 y 轴为 Wikipedia (en) / Github / DM Mathematics，x 轴为专家索引 1→64。Aux-Loss-Based 条整体浅黄、仅在少数专家上出现淡橙块；Aux-Loss-Free 条出现成片橙/深红高负载块（相对负载可达 8-10）。 | 不采用：真图但属附录 C 专家特化，与 DualPipe 主题无关 | — |
| `a98cf73f00b0ca719bc581476cfcb91b0e04c8cf19c880cd4bcdcbe7f8ea4962.jpg` | 1043 | Figure 11 子图 | 插图 | Figure 11 的子图切片，抽取图中内嵌标题为「Aux-Loss-Free Layer 8」。每条 y 轴为 Wikipedia (en) / Github / DM Mathematics，x 轴为专家索引 1→64。Aux-Loss-Based 条整体浅黄、仅在少数专家上出现淡橙块；Aux-Loss-Free 条出现成片橙/深红高负载块（相对负载可达 8-10）。 | 不采用：真图但属附录 C 专家特化，与 DualPipe 主题无关 | — |
| `904e82bce10a4e3f2503b7b725a5b48a718adf8a96b374f0f614473be3335c52.jpg` | 1045 | Figure 11 子图 | 插图 | Figure 11 的子图切片，抽取图中内嵌标题为「Aux-Loss-Based Layer 9」。每条 y 轴为 Wikipedia (en) / Github / DM Mathematics，x 轴为专家索引 1→64。Aux-Loss-Based 条整体浅黄、仅在少数专家上出现淡橙块；Aux-Loss-Free 条出现成片橙/深红高负载块（相对负载可达 8-10）。 | 不采用：真图但属附录 C 专家特化，与 DualPipe 主题无关 | — |
| `74bb586a6e298fb7f6ed0b4e2a3b17609e8d8b55b8b71fbda92ab448de87cc02.jpg` | 1047 | Figure 11 子图 | 插图 | Figure 11 的子图切片，抽取图中内嵌标题为「Aux-Loss-Free Layer 9」。每条 y 轴为 Wikipedia (en) / Github / DM Mathematics，x 轴为专家索引 1→64。Aux-Loss-Based 条整体浅黄、仅在少数专家上出现淡橙块；Aux-Loss-Free 条出现成片橙/深红高负载块（相对负载可达 8-10）。 | 不采用：真图但属附录 C 专家特化，与 DualPipe 主题无关 | — |
| `083af571af169ff7db196db0c47340238f019bca50f5d5f369ae1cbafce0fb83.jpg` | 1049 | Figure 11 子图 | 插图 | Figure 11 的子图切片，抽取图中内嵌标题为「Aux-Loss-Based Layer 10」。每条 y 轴为 Wikipedia (en) / Github / DM Mathematics，x 轴为专家索引 1→64。Aux-Loss-Based 条整体浅黄、仅在少数专家上出现淡橙块；Aux-Loss-Free 条出现成片橙/深红高负载块（相对负载可达 8-10）。 | 不采用：真图但属附录 C 专家特化，与 DualPipe 主题无关 | — |
| `34303830bc37868d3e28bf00aa859d48f04e32b227556bc7d846bba82f7af75b.jpg` | 1051 | Figure 11 子图 | 插图 | Figure 11 的子图切片，抽取图中内嵌标题为「Aux-Loss-Free Layer 10」。每条 y 轴为 Wikipedia (en) / Github / DM Mathematics，x 轴为专家索引 1→64。Aux-Loss-Based 条整体浅黄、仅在少数专家上出现淡橙块；Aux-Loss-Free 条出现成片橙/深红高负载块（相对负载可达 8-10）。 | 不采用：真图但属附录 C 专家特化，与 DualPipe 主题无关 | — |
| `94b1e4fedaddcb0e788281a956a0f52b6a2f669de9ad97f8fbc1f88fc9586b8c.jpg` | 1053 | Figure 11 子图 | 插图 | Figure 11 的子图切片，抽取图中内嵌标题为「Aux-Loss-Based Layer 11」。每条 y 轴为 Wikipedia (en) / Github / DM Mathematics，x 轴为专家索引 1→64。Aux-Loss-Based 条整体浅黄、仅在少数专家上出现淡橙块；Aux-Loss-Free 条出现成片橙/深红高负载块（相对负载可达 8-10）。 | 不采用：真图但属附录 C 专家特化，与 DualPipe 主题无关 | — |
| `f8601eb282e4398f669b3ea7c6a75bb1131ca5f406c27b88cb81a5266e725e5f.jpg` | 1055 | Figure 11 子图 | 插图 | Figure 11 的子图切片，抽取图中内嵌标题为「Aux-Loss-Free Layer 11」。每条 y 轴为 Wikipedia (en) / Github / DM Mathematics，x 轴为专家索引 1→64。Aux-Loss-Based 条整体浅黄、仅在少数专家上出现淡橙块；Aux-Loss-Free 条出现成片橙/深红高负载块（相对负载可达 8-10）。 | 不采用：真图但属附录 C 专家特化，与 DualPipe 主题无关 | — |
| `7f5cbc41fc27622543e25c6e5bc7aee6a634657cacc991cef2deaf79afc5cc0a.jpg` | 1058 | Figure 11 子图 | 插图 | Figure 11 的子图切片，抽取图中内嵌标题为「Aux-Loss-Based Layer 17」。每条 y 轴为 Wikipedia (en) / Github / DM Mathematics，x 轴为专家索引 1→64。Aux-Loss-Based 条整体浅黄、仅在少数专家上出现淡橙块；Aux-Loss-Free 条出现成片橙/深红高负载块（相对负载可达 8-10）。 | 不采用：真图但属附录 C 专家特化，与 DualPipe 主题无关 | — |
| `0685ee0097abcd2dcfd9bdee4c1ca7319625d34ddde23512256356d6d5b75577.jpg` | 1060 | Figure 11 子图 | 插图 | Figure 11 的子图切片，抽取图中内嵌标题为「Aux-Loss-Free Layer 14 + Aux-Loss-Based Layer 15（同图两栏）」。每条 y 轴为 Wikipedia (en) / Github / DM Mathematics，x 轴为专家索引 1→64。Aux-Loss-Based 条整体浅黄、仅在少数专家上出现淡橙块；Aux-Loss-Free 条出现成片橙/深红高负载块（相对负载可达 8-10）。该文件把同一子图内的多栏合并在一张图里，并包含 Relative Expert Load 0-10 的 colorbar 与子图题注。 | 不采用：真图但属附录 C 专家特化，与 DualPipe 主题无关 | — |
| `de33af3da0d8ac60e9ead385a79da0a397c637cc452579fb4ef8ddbf597421d0.jpg` | 1062 | Figure 11 子图 | 插图 | Figure 11 的子图切片，抽取图中内嵌标题为「Aux-Loss-Based Layer 13」。每条 y 轴为 Wikipedia (en) / Github / DM Mathematics，x 轴为专家索引 1→64。Aux-Loss-Based 条整体浅黄、仅在少数专家上出现淡橙块；Aux-Loss-Free 条出现成片橙/深红高负载块（相对负载可达 8-10）。 | 不采用：真图但属附录 C 专家特化，与 DualPipe 主题无关 | — |
| `2a2f4658736fa7592781a54e954f9e1c31d7a1b1413189728dc1a60220c5d1f1.jpg` | 1064 | Figure 11 子图 | 插图 | Figure 11 的子图切片，抽取图中内嵌标题为「Aux-Loss-Based Layer 14」。每条 y 轴为 Wikipedia (en) / Github / DM Mathematics，x 轴为专家索引 1→64。Aux-Loss-Based 条整体浅黄、仅在少数专家上出现淡橙块；Aux-Loss-Free 条出现成片橙/深红高负载块（相对负载可达 8-10）。 | 不采用：真图但属附录 C 专家特化，与 DualPipe 主题无关 | — |
| `c34db3843af7b4ffaefcdd56cc2d903db52a759e20c0838d93e92e34d704f7cc.jpg` | 1066 | Figure 11 子图 | 插图 | Figure 11 的子图切片，抽取图中内嵌标题为「Aux-Loss-Free Layer 15」。每条 y 轴为 Wikipedia (en) / Github / DM Mathematics，x 轴为专家索引 1→64。Aux-Loss-Based 条整体浅黄、仅在少数专家上出现淡橙块；Aux-Loss-Free 条出现成片橙/深红高负载块（相对负载可达 8-10）。 | 不采用：真图但属附录 C 专家特化，与 DualPipe 主题无关 | — |
| `b069fde214127abb996462bbd8229ece2076fa6204261674117079e830c19bf9.jpg` | 1068 | Figure 11 子图 | 插图 | Figure 11 的子图切片，抽取图中内嵌标题为「Aux-Loss-Free Layer 13」。每条 y 轴为 Wikipedia (en) / Github / DM Mathematics，x 轴为专家索引 1→64。Aux-Loss-Based 条整体浅黄、仅在少数专家上出现淡橙块；Aux-Loss-Free 条出现成片橙/深红高负载块（相对负载可达 8-10）。 | 不采用：真图但属附录 C 专家特化，与 DualPipe 主题无关 | — |
| `94ca9157b4a024b284a8ba632579c75389eeaca33324cf3cb30b945ecf5b4b21.jpg` | 1070 | Figure 11 子图 | 插图 | Figure 11 的子图切片，抽取图中内嵌标题为「Aux-Loss-Based Layer 16」。每条 y 轴为 Wikipedia (en) / Github / DM Mathematics，x 轴为专家索引 1→64。Aux-Loss-Based 条整体浅黄、仅在少数专家上出现淡橙块；Aux-Loss-Free 条出现成片橙/深红高负载块（相对负载可达 8-10）。 | 不采用：真图但属附录 C 专家特化，与 DualPipe 主题无关 | — |
| `ee0a56d90cb5c9822f255a9548322b45c043e1f564c35d149d9e20dd28228f73.jpg` | 1073 | Figure 11 子图 | 插图 | Figure 11 的子图切片，抽取图中内嵌标题为「Aux-Loss-Free Layer 16」。每条 y 轴为 Wikipedia (en) / Github / DM Mathematics，x 轴为专家索引 1→64。Aux-Loss-Based 条整体浅黄、仅在少数专家上出现淡橙块；Aux-Loss-Free 条出现成片橙/深红高负载块（相对负载可达 8-10）。 | 不采用：真图但属附录 C 专家特化，与 DualPipe 主题无关 | — |
| `347c9ac333c18490e0156773cf83ccaea2503f8af7358629065b74d80d96989d.jpg` | 1075 | Figure 11 子图 | 插图 | Figure 11 的子图切片，抽取图中内嵌标题为「Aux-Loss-Based Layer 18 + Aux-Loss-Free Layer 18（同图两栏，含 colorbar 与子图题注 (c) Layers 13-19）」。每条 y 轴为 Wikipedia (en) / Github / DM Mathematics，x 轴为专家索引 1→64。Aux-Loss-Based 条整体浅黄、仅在少数专家上出现淡橙块；Aux-Loss-Free 条出现成片橙/深红高负载块（相对负载可达 8-10）。该文件把同一子图内的多栏合并在一张图里，并包含 Relative Expert Load 0-10 的 colorbar 与子图题注。 | 不采用：真图但属附录 C 专家特化，与 DualPipe 主题无关 | — |
| `afbe971a78f0bf36c0bc7b01340b78c7c4dbc740ef2d7e34dc0328167bd83e5a.jpg` | 1077 | Figure 11 子图 | 插图 | Figure 11 的子图切片，抽取图中内嵌标题为「Aux-Loss-Free Layer 23」。每条 y 轴为 Wikipedia (en) / Github / DM Mathematics，x 轴为专家索引 1→64。Aux-Loss-Based 条整体浅黄、仅在少数专家上出现淡橙块；Aux-Loss-Free 条出现成片橙/深红高负载块（相对负载可达 8-10）。 | 不采用：真图但属附录 C 专家特化，与 DualPipe 主题无关 | — |
| `1d569fb8b03961ff305063a121c9e7442e99416275bc98ab0c7679dcd669fc65.jpg` | 1079 | Figure 11 子图 | 插图 | Figure 11 的子图切片，抽取图中内嵌标题为「Aux-Loss-Free Layer 20」。每条 y 轴为 Wikipedia (en) / Github / DM Mathematics，x 轴为专家索引 1→64。Aux-Loss-Based 条整体浅黄、仅在少数专家上出现淡橙块；Aux-Loss-Free 条出现成片橙/深红高负载块（相对负载可达 8-10）。 | 不采用：真图但属附录 C 专家特化，与 DualPipe 主题无关 | — |
| `c96980505211596b49709b9e46c3f00e09a1fba439f71de66b5952efa8ead4de.jpg` | 1082 | Figure 11 子图 | 插图 | Figure 11 的子图切片，抽取图中内嵌标题为「Aux-Loss-Based Layer 19」。每条 y 轴为 Wikipedia (en) / Github / DM Mathematics，x 轴为专家索引 1→64。Aux-Loss-Based 条整体浅黄、仅在少数专家上出现淡橙块；Aux-Loss-Free 条出现成片橙/深红高负载块（相对负载可达 8-10）。 | 不采用：真图但属附录 C 专家特化，与 DualPipe 主题无关 | — |
| `b1b270336cdff500f5504af4b454742b4e7023de996c973773861b09cdc9d690.jpg` | 1084 | Figure 11 子图 | 插图 | Figure 11 的子图切片，抽取图中内嵌标题为「Aux-Loss-Free Layer 19」。每条 y 轴为 Wikipedia (en) / Github / DM Mathematics，x 轴为专家索引 1→64。Aux-Loss-Based 条整体浅黄、仅在少数专家上出现淡橙块；Aux-Loss-Free 条出现成片橙/深红高负载块（相对负载可达 8-10）。 | 不采用：真图但属附录 C 专家特化，与 DualPipe 主题无关 | — |
| `07827c7cb49b9fbc24298d5160a05a7ed6c001c9cffa9e130e13babfe5223a5b.jpg` | 1086 | Figure 11 子图 | 插图 | Figure 11 的子图切片，抽取图中内嵌标题为「Aux-Loss-Based Layer 20」。每条 y 轴为 Wikipedia (en) / Github / DM Mathematics，x 轴为专家索引 1→64。Aux-Loss-Based 条整体浅黄、仅在少数专家上出现淡橙块；Aux-Loss-Free 条出现成片橙/深红高负载块（相对负载可达 8-10）。 | 不采用：真图但属附录 C 专家特化，与 DualPipe 主题无关 | — |
| `767da36642d5417f5e1ce6149bdc1b0ad342826ef2017afb1688f0bf3ac8ea68.jpg` | 1088 | Figure 11 子图 | 插图 | Figure 11 的子图切片，抽取图中内嵌标题为「Aux-Loss-Based Layer 21」。每条 y 轴为 Wikipedia (en) / Github / DM Mathematics，x 轴为专家索引 1→64。Aux-Loss-Based 条整体浅黄、仅在少数专家上出现淡橙块；Aux-Loss-Free 条出现成片橙/深红高负载块（相对负载可达 8-10）。 | 不采用：真图但属附录 C 专家特化，与 DualPipe 主题无关 | — |
| `fb75846ba4bd80e1a2bdac3af055410a0e195dcbb9731f0342954da95d98b14e.jpg` | 1090 | Figure 11 子图 | 插图 | Figure 11 的子图切片，抽取图中内嵌标题为「Aux-Loss-Free Layer 21」。每条 y 轴为 Wikipedia (en) / Github / DM Mathematics，x 轴为专家索引 1→64。Aux-Loss-Based 条整体浅黄、仅在少数专家上出现淡橙块；Aux-Loss-Free 条出现成片橙/深红高负载块（相对负载可达 8-10）。 | 不采用：真图但属附录 C 专家特化，与 DualPipe 主题无关 | — |
| `32bb7755661b6c8bffcfc7df8c0fbdaa17b27c45c4211105a8e7327639c9245b.jpg` | 1092 | Figure 11 子图 | 插图 | Figure 11 的子图切片，抽取图中内嵌标题为「Aux-Loss-Based Layer 22」。每条 y 轴为 Wikipedia (en) / Github / DM Mathematics，x 轴为专家索引 1→64。Aux-Loss-Based 条整体浅黄、仅在少数专家上出现淡橙块；Aux-Loss-Free 条出现成片橙/深红高负载块（相对负载可达 8-10）。 | 不采用：真图但属附录 C 专家特化，与 DualPipe 主题无关 | — |
| `4f244c1ed548b7dbee896ec916083e3b87c73270af0cfe342181d4354ea558cd.jpg` | 1094 | Figure 11 子图 | 插图 | Figure 11 的子图切片，抽取图中内嵌标题为「Aux-Loss-Free Layer 22」。每条 y 轴为 Wikipedia (en) / Github / DM Mathematics，x 轴为专家索引 1→64。Aux-Loss-Based 条整体浅黄、仅在少数专家上出现淡橙块；Aux-Loss-Free 条出现成片橙/深红高负载块（相对负载可达 8-10）。 | 不采用：真图但属附录 C 专家特化，与 DualPipe 主题无关 | — |
| `ab0f457994234caf4060eb3de1b3725fb2e5011e6f5c9a81fddc261679f5985b.jpg` | 1096 | Figure 11 子图 | 插图 | Figure 11 的子图切片，抽取图中内嵌标题为「Aux-Loss-Based Layer 23」。每条 y 轴为 Wikipedia (en) / Github / DM Mathematics，x 轴为专家索引 1→64。Aux-Loss-Based 条整体浅黄、仅在少数专家上出现淡橙块；Aux-Loss-Free 条出现成片橙/深红高负载块（相对负载可达 8-10）。 | 不采用：真图但属附录 C 专家特化，与 DualPipe 主题无关 | — |
| `aabc1c04542eb4599cde227e4a8e130bab4e04dfd8e3d0379f637b4476518a5f.jpg` | 1098 | Figure 11 子图 | 插图 | Figure 11 的子图切片，抽取图中内嵌标题为「Aux-Loss-Based Layer 24 + Aux-Loss-Free Layer 24（同图两栏，含 colorbar 与子图题注 (d) Layers 19-25）」。每条 y 轴为 Wikipedia (en) / Github / DM Mathematics，x 轴为专家索引 1→64。Aux-Loss-Based 条整体浅黄、仅在少数专家上出现淡橙块；Aux-Loss-Free 条出现成片橙/深红高负载块（相对负载可达 8-10）。该文件把同一子图内的多栏合并在一张图里，并包含 Relative Expert Load 0-10 的 colorbar 与子图题注。 | 不采用：真图但属附录 C 专家特化，与 DualPipe 主题无关 | — |
| `e1c0bdf1b5781c2e96ac5712357aebbd9811d44cc063c2d3d3f2af7dc8f29f24.jpg` | 1100 | Figure 11 子图 | 插图 | Figure 11 的子图切片，抽取图中内嵌标题为「Aux-Loss-Based Layer 25 + Aux-Loss-Free Layer 25 + Aux-Loss-Based Layer 26 + Aux-Loss-Free Layer 26（同图四栏，含 colorbar 与子图题注 (e) Layers 25-27）」。每条 y 轴为 Wikipedia (en) / Github / DM Mathematics，x 轴为专家索引 1→64。Aux-Loss-Based 条整体浅黄、仅在少数专家上出现淡橙块；Aux-Loss-Free 条出现成片橙/深红高负载块（相对负载可达 8-10）。该文件把同一子图内的多栏合并在一张图里，并包含 Relative Expert Load 0-10 的 colorbar 与子图题注。 | 不采用：真图但属附录 C 专家特化，与 DualPipe 主题无关 | — |
| `ebf9b32cb5ed5f83172f6c73615586c79707a90f1eaad508aa3675ae1e9283af.jpg` | — | Table 1 | 表格截图 | 训练成本表。列：Pre-Training / Context Extension / Post-Training / Total；行：in H800 GPU Hours = 2664K / 119K / 5K / 2788K，in USD = $5.328M / $0.238M / $0.01M / $5.576M。 | 可转写为 Markdown 表格，不直接引图（md 第 81 行已有等价 HTML 表格） | — |
| `131abc3537b583e3fe70f1edf3d33a5c201e10a2730bad358398adcf9994eb4f.jpg` | — | Table 2 | 表格截图 | 流水线方案对比表。列：Method / Bubble / Parameter / Activation；行：1F1B = (PP-1)(F+B), 1×, PP；ZB1P = (PP-1)(F+B-2W), 1×, PP；DualPipe (Ours) = (PP/2-1)(F&B+B-3W), 2×, PP+1。 | 可转写为 Markdown 表格，不直接引图（md 第 310 行已有等价 HTML 表格） | — |
| `d1ed1e21a117ac70f3293f21254671cf5d85bd6ecdee733a7e264b8fdaaaba2f.jpg` | — | Table 3 | 表格截图 | Base 模型对比表。列：# Shots / DeepSeek-V2 Base / Qwen2.5 72B Base / LLaMA-3.1 405B Base / DeepSeek-V3 Base；三行架构信息（MoE 21B/236B、Dense 72B/72B、Dense 405B/405B、MoE 37B/671B）＋ English/Code/Math/Chinese/Multilingual 分组共 40 余项指标（如 Pile-test(BPB) 0.606/0.638/0.542/0.548，MMLU(EM) 78.4/85.0/84.4/87.1）。 | 可转写为 Markdown 表格，不直接引图（md 第 507 行已有等价 HTML 表格） | — |
| `69e313a9796545e5ce4f3c53d804c9e8da75c10d9ca3dc5ca7055f79574b7ec0.jpg` | — | Table 4 | 表格截图 | MTP 消融表。列：Small MoE Baseline / Small MoE w/ MTP / Large MoE Baseline / Large MoE w/ MTP；激活参数 2.4B/20.9B，总参数 15.7B/228.7B，训练 token 1.33T/540B；指标含 Pile-test(BPB) 0.729/0.729/0.658/0.657、BBH 39.0/41.4/70.0/70.7、MMLU 50.0/53.3/67.5/66.6 等。 | 可转写为 Markdown 表格，不直接引图（md 第 515 行已有等价 HTML 表格） | — |
| `337acd5ae77313d81662fd74b148282556ef5de5b1b0c96ed0df77330ece66af.jpg` | — | Table 5 | 表格截图 | 无辅助损失负载均衡消融表。列：Small MoE Aux-Loss-Based / Aux-Loss-Free / Large MoE Aux-Loss-Based / Aux-Loss-Free；参数 2.4B/15.7B 与 20.9B/228.7B，token 1.33T 与 578B；指标含 Pile-test(BPB) 0.727/0.724/0.656/0.652、BBH 37.3/39.3/66.7/67.9、MATH(EM) 10.9/11.1/37.2/39.6。 | 可转写为 Markdown 表格，不直接引图（md 第 529 行已有等价 HTML 表格） | — |
| `289359f4d6559cf6257210e658be5e030647aab6dd2f800ac950b1054552ad19.jpg` | — | Table 6 | 表格截图 | Chat 模型对比表。列：DeepSeek V2-0506 / V2.5-0905 / Qwen2.5 72B-Inst. / LLaMA-3.1 405B-Inst. / Claude-3.5-Sonnet-1022 / GPT-4o-0513 / DeepSeek V3；分组 English(MMLU 78.2/80.6/85.3/88.6/88.3/87.2/88.5、MMLU-Pro 58.5/66.2/71.6/73.3/78.0/72.6/75.9 …)、Code、Math、Chinese；粗体为各列最优。 | 可转写为 Markdown 表格，不直接引图（md 第 606 行已有等价 HTML 表格） | — |
| `d7b4392924351f5c5ccb940d97aeb0d0e6adffbc5539f248cb473d4771b1a71b.jpg` | — | Table 7 | 表格截图 | 开放式对话评测表。列：Arena-Hard / AlpacaEval 2.0；行：DeepSeek-V2.5-0905(76.2/50.5)、Qwen2.5-72B-Instruct(81.2/49.1)、LLaMA-3.1 405B(69.3/40.5)、GPT-4o-0513(80.4/51.1)、Claude-Sonnet-3.5-1022(85.2/52.0)、DeepSeek-V3(85.5/70.0，粗体)。 | 可转写为 Markdown 表格，不直接引图（md 第 624 行已有等价 HTML 表格） | — |
| `30d70ee15c1ce8725b35869b103242837721a08cac08234f228e3c30e52b5d3f.jpg` | — | §5.3.4 生成式奖励模型表 | 表格截图 | 生成式奖励模型评分表。列：Chat / Chat-Hard / Safety / Reasoning / Average；行：GPT-4o-0513(96.6/70.4/86.7/84.9/84.7)、GPT-4o-0806、GPT-4o-1120、Claude-3.5-sonnet-0620、Claude-3.5-sonnet-1022(96.4/79.7/91.1/87.6/88.7)、DeepSeek-V3(96.9/79.8/87.0/84.3/87.0)、DeepSeek-V3 (maj@6)(96.9/82.6/89.5/89.2/89.6)。 | 可转写为 Markdown 表格，不直接引图（md 第 642 行已有等价 HTML 表格） | — |
| `22b20fa200493fcd3e8cdf3dfe9d9e48a1ba906db30bd95e8b988bdb5d078a49.jpg` | — | §5.4.1 R1 蒸馏表 | 表格截图 | 蒸馏效果表。列：LiveCodeBench-CoT (Pass@1 / Length) 与 MATH-500 (Pass@1 / Length)；行：DeepSeek-V2.5 Baseline(31.1/718, 74.6/769) 与 DeepSeek-V2.5 +R1 Distill(37.4/783, 83.2/1510)。 | 可转写为 Markdown 表格，不直接引图（md 第 646 行已有等价 HTML 表格） | — |
| `022638e9290d84620e0493558c4d774d25f47b9fceb5cc9c27c27d04980d270f.jpg` | — | 行间公式（MinerU 切图） | 公式截图 | MinerU 把行间公式单独切成图片，内容为：J_GRPO(θ) = E[q~P(Q), {o_i}_{i=1}^G ~ π_θold(O\|q)] · (1/G)Σ_i( min(π_θ/π_θold·A_i, clip(...,1-ε,1+ε)·A_i) − β·D_KL(π_θ\|\|π_ref) ) | 不采用：公式截图，非插图 | — |
| `2114795f81386e71b88daeac74e6f8987ec81366fcd08fc620644644822f1625.jpg` | — | 行间公式（MinerU 切图） | 公式截图 | MinerU 把行间公式单独切成图片，内容为：A_i = (r_i − mean({r_1,...,r_G})) / std({r_1,...,r_G}) | 不采用：公式截图，非插图 | — |
| `2a6d0d72558da4bd11a88487862154b3df435a8305ced8f5d3263dd212d2cdd6.jpg` | — | 行间公式（MinerU 切图） | 公式截图 | MinerU 把行间公式单独切成图片，内容为：g'_{i,t} = { s_{i,t},  s_{i,t} ∈ Topk({s_{j,t} \| 1≤j≤N_r}, K_r);  0, otherwise } | 不采用：公式截图，非插图 | — |
| `2abcf4b346a4644a0789e018161e8eab1bac20a5cb67cb5e44d6deb013dc467a.jpg` | — | 行间公式（MinerU 切图） | 公式截图 | MinerU 把行间公式单独切成图片，内容为：[k^C_{t,1}; k^C_{t,2}; ...; k^C_{t,n_h}] = k^C_t = W^{UK} c^{KV}_t | 不采用：公式截图，非插图 | — |
| `31dab03b8917743f107839bc1c3be02a84cfae73cab20e995833e2ca1711fb82.jpg` | — | 行间公式（MinerU 切图） | 公式截图 | MinerU 把行间公式单独切成图片，内容为：g_{i,t} = g'_{i,t} / Σ_{j=1}^{N_r} g'_{j,t} | 不采用：公式截图，非插图 | — |
| `3c3c04cef7636780525050dc0649e42674199cace2a64eec9c9990d1a83f357c.jpg` | — | 行间公式（MinerU 切图） | 公式截图 | MinerU 把行间公式单独切成图片，内容为：[v^C_{t,1}; v^C_{t,2}; ...; v^C_{t,n_h}] = v^C_t = W^{UV} c^{KV}_t | 不采用：公式截图，非插图 | — |
| `45f27319efb2c9e788facc83ebc144d533cc23781ae61d32f6517cdce9611e27.jpg` | — | 行间公式（MinerU 切图） | 公式截图 | MinerU 把行间公式单独切成图片，内容为：P_i = (1/T) Σ_{t=1}^{T} s'_{i,t} | 不采用：公式截图，非插图 | — |
| `4d6aacfc883c91178249a875d4ee04949ceef8332cf8eb49dd39d5229c77ca41.jpg` | — | 行间公式（MinerU 切图） | 公式截图 | MinerU 把行间公式单独切成图片，内容为：[q^C_{t,1}; q^C_{t,2}; ...; q^C_{t,n_h}] = q^C_t = W^{UQ} c^Q_t | 不采用：公式截图，非插图 | — |
| `7125e18453c79cc6f021d641f1d75a37495109540549e97d63b0bb0cd0829b37.jpg` | — | 行间公式（MinerU 切图） | 公式截图 | MinerU 把行间公式单独切成图片，内容为：h'^k_i = M_k[ RMSNorm(h^{k-1}_i) ; RMSNorm(Emb(t_{i+k})) ] | 不采用：公式截图，非插图 | — |
| `7381b2592b6d215b9d7d71d49127236d0903c39d1f9e7db3962d31534d96ad9c.jpg` | — | 行间公式（MinerU 切图） | 公式截图 | MinerU 把行间公式单独切成图片，内容为：c^{KV}_t = W^{DKV} h_t（左侧带方框标记） | 不采用：公式截图，非插图 | — |
| `77c62ad1cd88a781190fd90b5c95a094e68dcafe3dd001f2fd34664706ca1d2e.jpg` | — | 行间公式（MinerU 切图） | 公式截图 | MinerU 把行间公式单独切成图片，内容为：[q^R_{t,1}; q^R_{t,2}; ...; q^R_{t,n_h}] = q^R_t = RoPE(W^{QR} c^Q_t) | 不采用：公式截图，非插图 | — |
| `7976c3b944c859e07cf05f6c90c63a4eb65090e6b34a8287bfe8f33324822f32.jpg` | — | 行间公式（MinerU 切图） | 公式截图 | MinerU 把行间公式单独切成图片，内容为：s_{i,t} = Sigmoid(u_t^T e_i) | 不采用：公式截图，非插图 | — |
| `8472b5411fd3f7be3cc321f9b0e565570a5dd14ba034aca76a5bece13b460288.jpg` | — | 行间公式（MinerU 切图） | 公式截图 | MinerU 把行间公式单独切成图片，内容为：o_{t,i} = Σ_{j=1}^{t} Softmax_j( q_{t,i}^T k_{j,i} / sqrt(d_h + d_h^R) ) v^C_{j,i} | 不采用：公式截图，非插图 | — |
| `8d2abf899c0c5a925b9b7de57ac651294a380c9d754b83c74fd385f1b0ad5610.jpg` | — | 行间公式（MinerU 切图） | 公式截图 | MinerU 把行间公式单独切成图片，内容为：h'_t = u_t + Σ_{i=1}^{N_s} FFN^{(s)}_i(u_t) + Σ_{i=1}^{N_r} g_{i,t} FFN^{(r)}_i(u_t) | 不采用：公式截图，非插图 | — |
| `957e3ab9836bb3401fe0a8ef747bf128913525c80b00b22db0f99571647bd33c.jpg` | — | 行间公式（MinerU 切图） | 公式截图 | MinerU 把行间公式单独切成图片，内容为：P^k_{i+k+1} = OutHead(h^k_i) | 不采用：公式截图，非插图 | — |
| `98400019b93d636f60222739535a2f9b6c65f9ddc0f9fc3fcb2a6df6ca0b0548.jpg` | — | 行间公式（MinerU 切图） | 公式截图 | MinerU 把行间公式单独切成图片，内容为：c^Q_t = W^{DQ} h_t | 不采用：公式截图，非插图 | — |
| `9b25275d77ad14f8304421e0d91f311088e9811fa267b43d9617e442b767fe37.jpg` | — | 行间公式（MinerU 切图） | 公式截图 | MinerU 把行间公式单独切成图片，内容为：g'_{i,t} = { s_{i,t},  s_{i,t} + b_i ∈ Topk({s_{j,t}+b_j \| 1≤j≤N_r}, K_r);  0, otherwise } | 不采用：公式截图，非插图 | — |
| `9fbcbcc2c67f8859c46c5823fd9b20b379710fcfb3ed0c7c9d81cfdeb04ac55d.jpg` | — | 行间公式（MinerU 切图） | 公式截图 | MinerU 把行间公式单独切成图片，内容为：L_Bal = α Σ_{i=1}^{N_r} f_i P_i | 不采用：公式截图，非插图 | — |
| `c52bc9b75602d8834821cbd05f290d4fca4fb8d6b16e80b82c86b2091cacc2f4.jpg` | — | 行间公式（MinerU 切图） | 公式截图 | MinerU 把行间公式单独切成图片，内容为：L^k_MTP = CrossEntropy(P^k_{2+k:T+1}, t_{2+k:T+1}) = −(1/T) Σ_{i=2+k}^{T+1} log P^k_i[t_i] | 不采用：公式截图，非插图 | — |
| `da9f297966752a668daeeb9b5d0813a2324660d395a8a7514cbd998b6dccad0d.jpg` | — | 行间公式（MinerU 切图） | 公式截图 | MinerU 把行间公式单独切成图片，内容为：s'_{i,t} = s_{i,t} / Σ_{j=1}^{N_r} s_{j,t} | 不采用：公式截图，非插图 | — |
| `db34d36115566e051fb85f3e47d8dc823008e94fdd92233cf536722eba5bda7f.jpg` | — | 行间公式（MinerU 切图） | 公式截图 | MinerU 把行间公式单独切成图片，内容为：q_{t,i} = [q^C_{t,i} ; q^R_{t,i}] | 不采用：公式截图，非插图 | — |
| `e01f4b4f7a9ce24f4c79260f0e6541331263846186a124e3213f80054160b4be.jpg` | — | 行间公式（MinerU 切图） | 公式截图 | MinerU 把行间公式单独切成图片，内容为：h^k_{1:T-k} = TRM_k(h'^k_{1:T-k}) | 不采用：公式截图，非插图 | — |
| `e18ad197981e5e07bf15f2bf00b954ba6f074ca717f06a71750885c5d50a82bf.jpg` | — | 行间公式（MinerU 切图） | 公式截图 | MinerU 把行间公式单独切成图片，内容为：k^R_t = RoPE(W^{KR} h_t)（左侧带方框标记） | 不采用：公式截图，非插图 | — |
| `e73d4f85e368e74298f79e1c50e278e535c8a2357a3ceb940d703b3db66ec4a8.jpg` | — | 行间公式（MinerU 切图） | 公式截图 | MinerU 把行间公式单独切成图片，内容为：D_KL(π_θ\|\|π_ref) = π_ref(o_i\|q)/π_θ(o_i\|q) − log(π_ref(o_i\|q)/π_θ(o_i\|q)) − 1 | 不采用：公式截图，非插图 | — |
| `ee82bdd84580cf61589d41707251bc59a8260db585667cb0398d394ce146752b.jpg` | — | 行间公式（MinerU 切图） | 公式截图 | MinerU 把行间公式单独切成图片，内容为：L_MTP = (λ/D) Σ_{k=1}^{D} L^k_MTP | 不采用：公式截图，非插图 | — |
| `eea74d1d661de457abe0650684f71ab6da3aa8a3ad590e62e2b56439d9946aaa.jpg` | — | 行间公式（MinerU 切图） | 公式截图 | MinerU 把行间公式单独切成图片，内容为：k_{t,i} = [k^C_{t,i} ; k^R_t] | 不采用：公式截图，非插图 | — |
| `f99cfaccdead096f2cb3d648369249144fd3b4cdb46e471cd48375930aff21bc.jpg` | — | 行间公式（MinerU 切图） | 公式截图 | MinerU 把行间公式单独切成图片，内容为：u_t = W^O [o_{t,1}; o_{t,2}; ...; o_{t,n_h}] | 不采用：公式截图，非插图 | — |
| `4f232488611d38f3365af788e636e14e8a732254cda1ec0f4c1e1a56f6423d5a.jpg` | — | 行间公式（MinerU 切图） | 公式截图 | MinerU 把行间公式单独切成图片，内容为：f_i = (N_r / (K_r T)) Σ_{t=1}^{T} 1( s_{i,t} ∈ Topk({s_{j,t} \| 1≤j≤N_r}, K_r) ) | 不采用：公式截图，非插图 | — |
| `5b3ce649bffea69a1ba1c2888816c284c69f0a16021aaf1e719532956a263dfa.jpg` | — | 附录 A 贡献者名单 | 页眉/装饰/碎片 | 整页作者名单（两栏英文人名，标题 Research & Engineering，含 Aixin Liu、Bing Xue、… 与 Lecong Zhang、Liang Zhao、… 各约 60 行，星号标注已离职成员）。属于正文排版内容而非插图。 | 不采用：作者名单截图，非插图（md 第 898 行已有等价 HTML 表格） | — |
| `4fb1b0387cb77d468822629cbe889998030883319fd432f624292ec0a20b1728.jpg` | — | 附录 A 贡献者名单（Business & Compliance） | 页眉/装饰/碎片 | 作者名单续页片段，两栏人名 Jian Liang / Jin Chen / … / T. Wang 与 W.L. Xiao / Wei An / … / Zhen Zhang。属于正文排版内容而非插图。 | 不采用：作者名单截图，非插图（md 第 1003 行已有等价 HTML 表格） | — |
| `226afee2259f4ed85cfc6209de494ebe09448d0cfb66ef8b0898ffdc6d0aabc1.jpg` | — | §4.1 数据构造中的 FIM token 序列 | 页眉/装饰/碎片 | MinerU 切出的行内片段，内容为 FIM（Fill-In-Middle）拼接后的 token 序列：<\|fim_begin\|> f_pre <\|fim_hole\|> f_suf <\|fim_end\|> f_middle <\|eos_token\|>。既不是公式也不是插图，属正文行内片段截图。 | 不采用：行内片段截图（token 序列），非插图 | — |

## 采用图的正文引用块（可直接粘贴）

说明：`pics/` 为建议的落地目录，请按 `deep-dive.md` 所在目录的实际图片目录名替换；文件名按
`deepseek-v3-fig<N><子图>-<内容 slug>.<ext>` 规则，扩展名沿用原 `.jpg`。

### 1. Figure 4 — 一对前向/反向 chunk 的重叠策略

建议文件名：`deepseek-v3-fig4-overlap-forward-backward-chunk-pair.jpg`（原图 1134×129，极扁，建议给足宽度）

```html
<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/deepseek-v3-fig4-overlap-forward-backward-chunk-pair.jpg" alt="DeepSeek-V3 Figure 4：一对前向/反向 chunk 的重叠策略，上行 Computation 依次为 MLP(B)/MLP(W)/MLP(F)/ATTN(B)/ATTN(W)/ATTN(F)，下行 Communication 依次为 DISPATCH(F)/DISPATCH(B)/COMBINE(F)/PP/COMBINE(B)，△ 表示前向 chunk、▲ 表示反向 chunk" style="width: 96%;">
</div>
```

> **图片来源**：DeepSeek-V3 Technical Report（arXiv 2412.19437v2）Figure 4，§3.2.1 DualPipe and Computation-Communication Overlap。
> 本地副本：`references/papers/deepseek-v3-report/deepseek-v3-report.md` 第 291 行引用图，图注见第 292 行。
> 口径说明：该图是 DualPipe 全部重叠机制的来源，只讲「一对 chunk 内如何重排组件以藏住 all-to-all 与 PP 通信」；
> 它**不**给出全局调度，全局调度由 Figure 5 承担，两者不要混引。图注中的颜色约定（橙=前向、绿=backward for input、蓝=backward for weights、紫=PP 通信、红=barrier）与读图结果一致。

### 2. Figure 5 — DualPipe 官方调度示例（8 PP ranks × 20 micro-batches）

建议文件名：`deepseek-v3-fig5-dualpipe-schedule-8pp-20mb.jpg`（原图 1256×212，建议满宽；8×20 网格在正文列宽下会很密，必要时提示读者点开原图）

```html
<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/deepseek-v3-fig5-dualpipe-schedule-8pp-20mb.jpg" alt="DeepSeek-V3 Figure 5：DualPipe 在 8 个 PP rank、双向各 20 个 micro-batch 下的调度示意，横轴为时间，8 行 Device 0-7，图例为 Forward/Backward/Backward for input/Backward for weights/Overlapped forward %26 Backward" style="width: 100%;">
</div>
```

> **图片来源**：DeepSeek-V3 Technical Report（arXiv 2412.19437v2）Figure 5，§3.2.1 DualPipe and Computation-Communication Overlap。
> 本地副本：`references/papers/deepseek-v3-report/deepseek-v3-report.md` 第 306 行引用图，图注见第 308 行。
> 口径说明：**该图块不包含 Table 2。** Figure 5 的图片文件里只有 8 行 device × 时间轴的调度网格；
> Table 2（气泡/显存对比公式）在 md 中是第 310 行的独立 HTML 表格，MinerU 另存了一份独立的图片文件
> （`131abc3537b583e3fe70f1edf3d33a5c201e10a2730bad358398adcf9994eb4f.jpg`，md 未引用）。
> 引用本图时若需要 Table 2 的公式，请直接抄 Markdown 表格，不要把该表格截进这张图里一起引用。

### 3.（不引图，转写内容）Table 2 — 气泡与显存对比

来源：md 第 310 行 HTML 表格 / 第 312 行图注；对应的图片文件为上文提到的 `131abc…jpg`（未被 md 引用）。
建议在 §6「气泡 / 显存公式重算与比较口径统一」中直接写成 Markdown 表格：

| Method | Bubble | Parameter | Activation |
| --- | --- | --- | --- |
| 1F1B | $(PP-1)(F+B)$ | 1× | PP |
| ZB1P | $(PP-1)(F+B-2W)$ | 1× | PP |
| DualPipe (Ours) | $(\frac{PP}{2}-1)(F\&B+B-3W)$ | 2× | PP+1 |

## 存疑与分歧

1. **md 的 Figure 10 出现了两次，附录 C 的图号是错的。**
   `deepseek-v3-report.md` 第 1010 行（附录 B）与第 1101 行（附录 C）都标为 `Figure 10`，且第 1022 行正文也写「as demonstrated in Figure 10」。
   用 arXiv 官方 HTML 版核对，编号应为：Figure 10 = 附录 B 的 BF16/FP8 损失曲线，Figure 11 = 附录 C 的专家负载图。
   因此：`d94eaa157b4f…jpg` = **Figure 10**（正确）；附录 C 的 37 张子图属于 **Figure 11**，md 的 `Figure 10` 是 MinerU 抽取错误。
   本清单的「图号/表号」列以读图 + 官方编号为准，未沿用 md 的错误编号。
2. **草稿「MinerU 把 Table 2 并到了 Figure 5 同一图块」的说法不成立。**
   读 `e6f2d6c79a43…jpg`（1256×212）确认：图中只有 8 行 Device 与时间轴方格，没有任何表格文字或公式；
   Table 2 在 md 中是独立的第 310 行 HTML 表格，另有一张完全独立的 Table 2 图片 `131abc3537b5…jpg`（md 未引用）。
3. **Figure 11 的子图边界与 md 引用行不严格对应。** 图中内嵌的子图题注可读出 `(c) Layers 13-19`、`(d) Layers 19-25`、`(e) Layers 25-27`；
   官方 HTML 版把该图拆成 5 个资源（`_1-6` / `_7-12` / `_13-18` / `_19-24` / `_25-26`），共 (a)-(e) 五组。
   MinerU 把其中若干栏合并进同一张图，导致 md 第 1034 / 1055 / 1070 / 1079 行的组标题与相邻图片并不对齐。
4. **md 第 1070、1079 行的两行散落文字（`Aux-Loss-Based Layer 13` / `Aux-Loss-Based Layer 19`）位置错误。**
   它们紧跟的图片实际是 `94ca9157b4a0…jpg`（读图标题为 Aux-Loss-Based **Layer 16**）与 `1d569fb8b039…jpg`（Aux-Loss-Free **Layer 20**）。
   即 md 中这两行的文字是 OCR 漂移出来的浮动标签，不能当作相邻图片的图注使用。
5. **`7f5cbc41…jpg` 的标题是 `Aux-Loss-Based Layer 17`，但它被排在 (b) Layers 7-13 组之后、`Aux-Loss-Free Layer 14` 之前，顺序错乱。**
   该标题已用 OCR 二次复核确认，不是误读。
6. **有 4 张图是「一图多栏」的合并产物**（`0685ee00…` 两栏、`347c9ac3…` 两栏+colorbar+子图题注、`aabc1c04…` 两栏+colorbar+子图题注、`e1c0bdf1…` 四栏+colorbar+子图题注）。
   这也是 Figure 11 的 37 张切片数与 5 组 × 层数的理论面板数对不上的原因。
7. **Figure 8（`b855efd8…jpg`）顶部标题在抽取时被裁掉。** 图内只剩下 NIAH 热力图本体（y 轴 Document Depth Percent、x 轴 Context Length 2K-128K、colorbar Score 1-10），
   标题文字不完整。该图与本文主题无关，不影响选图。
8. **Figure 4 的读图结果与图注一致，无分歧。** 图注声称的颜色约定（橙=前向、绿=backward for input、蓝=backward for weights、紫=PP、红=barrier）在图中可逐格对应；
   ▲/△ 分别表示 backward / forward chunk 也由图中图例确认。
9. **未被 md 引用的 9 张表格截图全部与 md 中已有的 HTML 表格重复**（Table 1/2/3/4/5/6/7 分别对应 md 第 81/310/507/515/529/606/624 行，另有两张对应第 642、646 行的无编号表格）。
   因此这 9 张图在写作中一律不引图，只在需要时从 md 的 HTML 表格转写。
10. **引用行号的口径**：表中「md 引用行」指 `deepseek-v3-report.md` 中 `![](images/xxx.jpg)` 所在行；
    图注通常在同一行（行尾接 `Figure N | …`）或下一行，本清单逐行核对过。

