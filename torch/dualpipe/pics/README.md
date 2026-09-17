# torch/dualpipe/pics：图片资产清单

本目录保存 `torch/dualpipe/deep-dive.md` 正文实际引用的图片副本。命名沿用 `torch/deepep/pics/` 与 `transformers/kda_linear_attention/pics/` 的惯例：`<来源简称>-fig<N><子图>-<内容 slug>.<ext>`；官方仓库图保留原名。**所有图片都是从 `references/` 或上游锁定 revision 直接复制，不重绘、不裁剪。**

逐张选择依据与「未采用图的理由」记录在 `notes/IMAGE-SURVEY-*.md`（12 个文件，覆盖 `references/` 下全部 433 张写作素材图）。本目录只放正文真正引用的图。

## 0. 正文引用格式

居中 `<div>` + `<img>` + 紧随其后的「图片来源」引用块；正文一律使用相对路径 `./pics/<文件名>`，alt 文本需含「图号 + 内容 + 图中关键读数」。

```html
<div style="text-align: center; width: 100%; margin: 0 auto;">
    <img src="./pics/<文件名>" alt="<图号 + 内容 + 关键读数>" style="width: 70%;">
</div>

> **图片来源**：<来源名>（arXiv <编号> 或 commit）<图号>，§<小节>。本地副本：`references/<...>.md` 第 <L> 行引用图。<口径说明>
```

## 1. 图片清单（30 张）

| # | 文件名 | 体积 | sha256（前 16 位） | 来源 |
| ---: | --- | ---: | --- | --- |
| 1 | `aiinfra-pp-1f1b-pipedream-timeline.png` | 142,605 B | `96d4490c775d7793` | `references/docs/aiinfra-pp-1f1b-interleaved/images/10pipeline02.png` |
| 2 | `aiinfra-pp-1f1b-warmup-microbatches-staircase.png` | 56,367 B | `2c88a1613a551e52` | `references/docs/aiinfra-pp-1f1b-interleaved/images/10pipeline10.png` |
| 3 | `aiinfra-pp-gpipe-flush-idle.png` | 192,888 B | `75ca4ab7e2691f22` | `references/docs/aiinfra-pp-1f1b-interleaved/images/10pipeline01.png` |
| 4 | `chimera-fig2-pipeline-schemes-comparison.jpg` | 134,082 B | `065724ca21067067` | `references/papers/chimera/images/aee6624ecf6b7a7091069e7856d11eadb156d52b3be993e54e78783904dcf06b.jpg` |
| 5 | `chimera-fig3-bidirectional-pipeline-schedule.jpg` | 98,821 B | `06afcc5d082951a8` | `references/papers/chimera/images/270ff45471492c201af79e28da9e509b7beb940b080dd19a299f9cf3930b8471.jpg` |
| 6 | `controllable-memory-fig18-schedule-gallery.jpg` | 132,346 B | `de316bc61248c28b` | `references/papers/controllable-memory/images/d10a100e2437141bdcbeb485e519199f4f3855984241e329b4370d9b6f49dc76.jpg` |
| 7 | `controllable-memory-fig2-parallel-vs-vshape.jpg` | 23,024 B | `021d7591b884124d` | `references/papers/controllable-memory/images/7f4661e035fa5b72abe32a5777663cd8701da5687051adfe5cd13ef11a9f4c40.jpg` |
| 8 | `controllable-memory-fig4-vshape-full-schedules.jpg` | 130,834 B | `648457c7b490ea0b` | `references/papers/controllable-memory/images/40d18d252f7d6470f7140a7622b62634ba51963b085de5c7020b87f2a99c6a66.jpg` |
| 9 | `controllable-memory-fig6-mfu-and-activation-memory.jpg` | 100,178 B | `d9d5041ba5042919` | `references/papers/controllable-memory/images/6d0222dda41626e3feed3fce587f946460d388cb790c926574f994a04d2ea69c.jpg` |
| 10 | `deepseek-moe-fig2-fine-grained-segmentation-shared-expert.jpg` | 87,277 B | `f26e515b9bec9dd6` | `references/papers/deepseek-moe/images/f77205913b6fd151e304c5a70d7e631629fa19ca95d08f9fb369d1ef4762eb67.jpg` |
| 11 | `deepseek-v3-fig4-overlap-forward-backward-chunk-pair.jpg` | 24,049 B | `6897f5d6a94e028a` | `references/papers/deepseek-v3-report/images/ea1a52decfb55f603ceea7de1e14bbb72c8e161ed45f792c75afdcfc85a8b59b.jpg` |
| 12 | `deepseek-v3-fig5-dualpipe-schedule-8pp-20mb.jpg` | 68,536 B | `f21679903bd65d01` | `references/papers/deepseek-v3-report/images/e6f2d6c79a43060cbcca6151b7ab9f65a07022a2b6785819b193e4c6639157a4.jpg` |
| 13 | `dualpipe.png` | 805,564 B | `2b4b9eaf44cd82a7` | DeepSeek 官方仓库 `deepseek-ai/DualPipe`，commit `030ce4325f4ebeb437da4ebc6d00a70469dd58ae`，`images/dualpipe.png` |
| 14 | `dualpipev.png` | 535,991 B | `ae0bb45143b74319` | DeepSeek 官方仓库 `deepseek-ai/DualPipe`，commit `030ce4325f4ebeb437da4ebc6d00a70469dd58ae`，`images/dualpipev.png` |
| 15 | `gpipe-fig2-naive-vs-microbatch-pipeline.jpg` | 48,756 B | `5517a4fc80d26027` | `references/papers/gpipe/images/8f3b6054d2586fe0af6cbb7013dd5f4fe4751ca2227d61a94498d6df47eb379e.jpg` |
| 16 | `megatron-fig4-default-vs-interleaved-1f1b.jpg` | 94,840 B | `ceb8d2c762df0e74` | `references/papers/megatron-lm-gpu-clusters/images/2a3ac338c206e25875fba03f81ced8e175be26545684dd22e150a7c9a8c0dd3d.jpg` |
| 17 | `megatron-fig6-bubble-size-vs-data-parallel-size.jpg` | 27,030 B | `7cd7dd985e50db22` | `references/papers/megatron-lm-gpu-clusters/images/0b96d327cec43357cc2271be149d6d69f90bbc9ef277dbeb3ecea526801ee5dc.jpg` |
| 18 | `pipedream-2bw-fig2-2bw-timeline-two-weight-versions.jpg` | 42,770 B | `378cfcc39edcf1b3` | `references/papers/pipedream-2bw/images/ec0c258b472ef66da4148bdd10d376d6db66777cc15ef85a1023d2d1d9d7e81a.jpg` |
| 19 | `sea-ai-lab-cut-in-half-mirrored-schedule.png` | 197,265 B | `2c202c54658ad176` | `references/articles/sea-ai-lab-cut-in-half/images/v2-3da883a2e88124b91ba12d642d9e3ba9_r.png` |
| 20 | `sea-ai-lab-cut-in-half-vshape.jpg` | 50,542 B | `efa9a8ea20cbd4ad` | `references/articles/sea-ai-lab-cut-in-half/images/v2-7c797d3cc827149eadb7beb740d5889f_r.jpg` |
| 21 | `terapipe-fig1d-token-pipeline.jpg` | 23,079 B | `4f429f076391960d` | `references/papers/terapipe/images/a73bba64824371103963da88558e60812c9ebcfbdce52ff1aee1353e24f435ae.jpg` |
| 22 | `xiaodonggua-ep-alltoall-routing.jpg` | 133,769 B | `ca90d8e9a1b16c6c` | `references/articles/xiaodonggua-shousi-dualpipe/images/v2-78c60e5c84bc64d10f302e9eaad427a3_r.jpg` |
| 23 | `xiaodonggua-ep-serial-vs-overlap.jpg` | 106,435 B | `11aafe740f207ae6` | `references/articles/xiaodonggua-shousi-dualpipe/images/v2-33affd2027bc0a57b5c8df95ced57d32_r.jpg` |
| 24 | `xiaodonggua-official-schedule-eight-steps.jpg` | 137,451 B | `e21548a9d4d5cf63` | `references/articles/xiaodonggua-shousi-dualpipe/images/v2-02221b07d4e4e5cfb10e355dbbd2c686_r.jpg` |
| 25 | `yeqianshu-1f1b-vs-dualpipe-layers.jpg` | 162,353 B | `a5ba66292958dbc5` | `references/articles/yeqianshu-dualpipe-source-walkthrough/images/v2-86d6d62375e52bc9ad2e15ffbfd42810_r.jpg` |
| 26 | `zero-bubble-fig1-mlp-computation-graph-f-b-w.jpg` | 26,904 B | `ef548603484ac47d` | `references/papers/zero-bubble/images/a3f9d2a02a9d8cacb8bee2bd14c6ba0504b65af0704d3d4dec44581d27545d12.jpg` |
| 27 | `zero-bubble-fig2-1f1b-schedule.jpg` | 29,020 B | `a13c20cf7d051c9a` | `references/papers/zero-bubble/images/7828028ce7aed2d6d6439d6f62df284d8048a5ead32e6574fb00299cb51faaa1.jpg` |
| 28 | `zero-bubble-fig3-handcrafted-zb-h1-h2.jpg` | 61,193 B | `4ca2a78e4c447b5a` | `references/papers/zero-bubble/images/e2c1ba8e49b0c79994431d4c405667b4dfb08dcbaa0d535cfc2dd803e8622016.jpg` |
| 29 | `zero-bubble-fig4-optimizer-post-validation.jpg` | 30,253 B | `9e937c1e44266bc7` | `references/papers/zero-bubble/images/1cfd9a39e33e68f4174b3a672afbaaf75fdde78c7d720a6203a2014515c2c767.jpg` |
| 30 | `zero-bubble-fig8-zbv-schedule.jpg` | 39,715 B | `9c410bd2919278ec` | `references/papers/zero-bubble/images/536478895e0abbcea17d6178866439923d5a02d53f94cfd02f19ea5706b189e0.jpg` |

## 2. 与 `learn-plan.md` 的对应

`learn-plan.md` 的「参考图复用策略 → 正文采用清单」一节列出同一批文件；其中 DeepSeekMoE Figure 2 的文件名在计划里写作 `deepseek-moe-fig2-fine-grained-shared-expert.jpg`，本目录实际使用核对文件给出的 `deepseek-moe-fig2-fine-grained-segmentation-shared-expert.jpg`，**以本目录为准**。

`torch/deepep/pics/` 里另有一份同名内容的 DeepSeekMoE Figure 2（命名为 `deepseek-moe-fig2-fine-grained-shared-expert.jpg`），两篇文章各自持有副本，不互相引用，避免跨目录相对路径。

## 3. 已知的图注陷阱

1. **GPipe Figure 2 是三联图**：(a) 是朴素的模型并行链，(b) 是只有一个 microbatch、气泡几乎占满的流水线，(c) 才是 micro-batch 流水线并在网格中部显式标注 `Bubble`。\((p-1)(F+B)\) 的基线只对应 (c)，引用时必须点名子图。
2. **Zero Bubble Figure 2 与 Figure 3 是两个独立文件**：`zero-bubble-fig2-1f1b-schedule.jpg` 只有 1F1B、整图没有任何 W 格；`zero-bubble-fig3-handcrafted-zb-h1-h2.jpg` 才是上下两栏的 ZB-H1 / ZB-H2。
3. **DualPipe 与 DualPipeV 的官方图口径不同**：`dualpipe.png` 是 8 PP rank × 双向合计 20 个 micro-batch（每方向 10 个）；`dualpipev.png` 是 4 PP rank（8 个逻辑 stage）× 10 micro-batch。两张图的 microbatch 数不同，不能横向比较格子数量。
4. **社区图必须标注层级**：`sea-ai-lab-cut-in-half-*`、`xiaodonggua-*`、`yeqianshu-*` 是社区解读或作者自绘（其中 Sea AI Lab 那两张是 cut-in-half 的第一手来源），不是论文原图；图注里要写清。
5. **教材示意图不是论文原图**：`aiinfra-pp-*` 来自 AIInfra 开源教材页，与论文图不能混引。该页面第 142、346 行的 alt 文本与画面不符（都写成「args 超参设置」），引用时不要沿用原 alt。

## 4. 验证记录

- 30 张图全部从 `references/` 或 `/workspace/algorithm/DualPipe/images/` 复制，sha256 见第 1 节；源文件在复制后未改动。
- 文件名与 `notes/IMAGE-SURVEY-*.md` 的「建议文件名」逐一对齐；`dualpipe.png` / `dualpipev.png` 使用官方原名。
- 逐张核对结果（图号、内容、类别）见对应的 `notes/IMAGE-SURVEY-*.md`；正文引用的每个路径都已用 `ls` 校验存在。
- **第 3 轮修正**：删除了两张正文不再引用的图——`aiinfra-pp-zbv-schedule.png`（教材页把它标在「ZB-V schedule」标题下，但逐格颜色分类证明内容是 **ZB-H2**，与 Zero Bubble Figure 3 下栏 39/39 相同）与 `aiinfra-pp-interleaved-1f1b-vpp-layout.png`（与 Megatron-LM Figure 4 下栏逐格同内容，属重复引用）。同时 `terapipe-fig1d-token-pipeline.jpg` 的替代文件在第 2 轮已换好。
- **第 2 轮修正**：`terapipe-fig1d-token-pipeline.jpg` 最初误取成了 TeraPipe Figure 1(c)（`b78a683e…`，GPipe 的 microbatch 流水线）；已改为真正的 Figure 1(d)（`a73bba64…`），sha256 见第 1 节。该错误由独立审查发现，并在 `notes/IMAGE-VERIFY.md` 记录。
- 正文引用的图之外，正文引用的图片路径与 `pics/` 目录内容已用 `comm` 双向比对：无缺失、无多余。
- **未做**：没有把图片与原页面逐张肉眼比对（核对任务用 `read_image` 读过全部 433 张写作素材图，但只对其中一部分做了尺寸与裁切复核）；没有重建官方仓库图的原始生成脚本。
