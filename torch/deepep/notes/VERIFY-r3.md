# 复审记录 R3：终稿冻结版（2026-09-10）

> 对应报告：`torch/deepep/REVIEW-r2.md`（第二轮 `/learn-review`，八维）
> 本文件只记录**本轮实际跑过的命令与逐条结果**，便于第三方复跑。

## 0. 快照

| 文件 | 行数 | 字节 | sha256 | mtime |
| --- | ---: | ---: | --- | --- |
| `deep-dive.md` | 765 | 96,895 | `1473c2ef9c3007e4012f9a97a89f9e88846a68aad901bb876fb1b852785765f0` | 2026-09-10 10:57:46 -0400 |
| `learn-plan.md` | 526 | 65,158 | `5eb883baaa1e1178ef08e25147e36a077f5c7346a5029fb05cf7dc9e8c616d60` | 本轮修改后 |
| `PROMPT-learn-pipeline.md` | — | — | `e773784cea6209a7ca275d6ebb8d1ace5afff33a2285634ba5d0b0f04fbe1a7d` | 本轮回写后 |

**漂移记录**：本轮审查开始时 `deep-dive.md` 为 763 行 / `ebb6579c…`（10:31:35）。10:57:46 另一条并行轨道改写了 §8.1 关于 `topk_weights` 复用的段落（同时把 §10 标题的「三条边界」改为与 §10.4 一致的「四条」）。差异已定位、新证据已逐条重核（见第 2 节第 6 行），本文件与 `REVIEW-r2.md` 的结论均针对上述 765 行快照。

## 1. 机械检查（逐条命令）

| 项 | 命令 | 结果 |
| --- | --- | --- |
| 外部链接（本地可复现） | `python3 /tmp/check_refs.py`（抽 `blob/<sha>/<path>#Lx-Ly`，对每个本地 checkout 执行 `git show <sha>:<path>` 并比对该文件总行数） | **OK = 79，FAIL = 0**；`main`/`master` 链接 **0** |
| 外部链接（本地无仓库的 pin） | `gh api repos/<r>/contents/<p>?ref=<sha>` → base64 解码后比对行数与首末行内容 | **5/5 通过**：NCCL `gin.h@b0e84e58`（418 行，L150 `ncclGinPut` / L177 `ncclGinFlush`）、DeepSeek-V3 `model.py@b15f0dbb`（808 行，L566 `def forward` / L598 `return weights…`、L690 `z = self.shared_experts(x)` / L693 `return (y + z).view(shape)`）、PyTorch `distributed_c10d.py@e8430bed`（8187 行，L6129 `def all_to_all_single`）、verl `docs/perf/dpsk.md@3467d90a`（88 行，L5 与 L29 内容与正文引述一致） |
| 图片双向一致 | `grep -o 'src="[^"]*"' deep-dive.md` 对比 `ls pics/*.jpg pics/*.png` | **12 = 12**，0 死链、0 孤儿 |
| 论文图出处行号 | 逐行 `sed -n '<N>p' references/papers/.../*.md` | 10 处全部命中对应图片引用行（MoE L74/L249/L252–255；V3 L127/L291/L306/L336/L347/L349/L543） |
| 官方文档行号 | `git show <rev>:README.md \| sed -n '…'`、`docs/legacy.md` 同法 | V2 README L3/L9/L19/L22/L29–31/L53/L55/L344–356、V1 README L7/L15–37/L227–229/L243/L290–292、`docs/legacy.md` L11/L227/L288/L310 全部与正文引述一致 |
| 格式红线 | `grep -c '——'`、`grep -c '^```text'`、`grep -nP '[┌┐└┘│├─┬┴┼▶◀▲▼]'`、`grep -o <url> \| grep -cE '/(blob\|tree)/(main\|master)'` | `——` = **0**；```text 围栏 = **0**；ASCII 字符画 = **0**；main/master = **0** |
| 隔离检查（不应出现） | `grep -n 'rlhf/sys-design/readme-5' deep-dive.md`、`grep -n 'sandbox:/mnt/data' deep-dive.md`、`grep -n '实测' deep-dive.md` | 0 命中；正文只在「**不是本文的实测结果**」「本文没有做任何一层实测」这类否定句里出现「实测」，无一处把官方数字写成实测 |

## 2. 内容级抽查（行号在范围内 ≠ 内容正确）

以下每条都打开源码（或 `gh api` 取回原文）核对内容，而非只查行号：

1. `v1.2.1:deep_ep/buffer.py#L337-L360` —— `num_worst_tokens` docstring「for intranode only」+ `assert num_worst_tokens == 0, 'Internode dispatch does not support …'`。与 §6 的引述一致。
2. `v1.2.1:csrc/kernels/layout.cu#L85-L94` —— `shifted_is_token_in_rank[j] = (is_in_rank[j] > 0)` 与 `num_tokens_per_rank_per_thread[…] += (is_in_rank[j] > 0)`，支持 §6「布尔化后计数」。
3. `v1.2.1:csrc/kernels/internode.cu#L371-L375 / #L1008 / #L1039` —— 角色枚举（`kRDMASender` … `kNVLReceivers`）、`kNumDispatchRDMASenderWarps = 7`、`SETUP_LAUNCH_CONFIG(num_channels * 2, (7 + 1 + NUM_MAX_NVL_PEERS) * 32, stream)`；支持 §6「每 rank 通道两个 SM、发送侧 7 个 warp」。
4. `v1.2.1:csrc/config.hpp#L129-L166` —— `num_bytes_per_dispatch_msg = sizeof(int4) + std::max(hidden * sizeof(nv_bfloat16), hidden + num_scales * sizeof(float))`，收缓冲按 `num_experts * num_max_dispatch_tokens_per_rank * …` 且 `total_bytes += … * 2`（双缓冲）；支持 §7.1。
5. `01dc3aac:deep_ep/include/deep_ep/impls/combine.cuh#L68-L69 / #L215-L224` —— expanded send 下 `EP_DEVICE_ASSERT(topk_weights == nullptr)`；权重在 `not kDoExpandedSend and topk_weights != nullptr` 时只写入待发送 buffer；支持 §9「只搬运、不求和」。
6. `01dc3aac:deep_ep/buffers/elastic.py#L889-L890 / #L906-L908 / L976 / L995`、`01dc3aac:csrc/elastic/buffer.hpp#L1106-L1110`、`01dc3aac:README.md L213-L217 / L295-L301` —— 漂移后新增的三处证据逐条成立：参数 docstring 写「Must be `None` if `handle` is provided」、handle docstring 写「can be optionally provided」、runtime 在 `self.runtime.dispatch(x, sf, topk_idx, topk_weights, …)` 原样转发、`do_zero_padding` 在 L995 一并转发、`topk_weights.has_value()` 时按 expand / 非 expand 分配 `recv_topk_weights`；README 的反向示例确实把 `grad_recv_topk_weights` 交给 `combine`，cached-handle 分支确实只传 `handle` 并返回 `None` 权重。
7. `01dc3aac:deep_ep/include/deep_ep/impls/hybrid_dispatch.cuh#L54` —— `EP_STATIC_ASSERT(kNumScaleoutWarps == kNumForwardWarps, "Invalid warp size")`；支持 §8.6。
8. `01dc3aac:deep_ep/utils/envs.py#L193-L220 / #L246-L265` —— NVLink 求和后 `* factor`（默认 0.9）、RDMA 走 `ibstat` 的 `Rate` 后 `return rate / 8`；支持 §8.4 与 §11 的十进制 GB/s 口径。
9. `01dc3aac:csrc/elastic/buffer.hpp#L670-L676 / #L1064-L1078`、`deep_ep/include/deep_ep/common/compiled.cuh#L74-L80`、`math.cuh#L16` —— `num_sf_packs = ceil_div(hidden, 32)`（注释「An approximation for number of SF packs」）、容量三行 + `math::align`、`sf_pack_t` 为 `{float, int}`；支持 §5 的 56.25% 第二口径与 §8.3 的容量公式。
10. `01dc3aac:deep_ep/include/deep_ep/common/comm.cuh#L59-L60 / #L145-L152`、`handle.cuh#L12-L17 / #L98-L109 / #L178-L208` —— `NCCL_GIN_RESOURCE_SHARING_CTA` / `_GPU` 的选择、`ptx::fence_acq_rel_sys()` 出现在 GIN barrier 之前、team tag 宏与 `put` / `put_value` / `red_add_rel` 的对称指针优先逻辑；支持 §8.5 与 §12.1。
11. `b15f0dbb:inference/model.py` L560–600（`gh api` 取回）—— `original_scores = scores` 后加 bias、`weights = original_scores.gather(1, indices)`、sigmoid 分支 `weights /= weights.sum(...)`、`weights *= self.route_scale`；支持 §4.3 的三条结论。
12. `0d7ecc5d:megatron/.../fused_a2a.py` L12–13 / L106–108 / L142–163 / L272 —— `from deep_ep import Buffer`、CUDA graph 不兼容注释、`FusedDispatch.backward` 调 `buffer.combine(..., handle=handle, topk_weights=grad_token_probs.float(), …)`、`from deep_ep import HybridEPBuffer`；支持 §2、§10.1。`rg -c ElasticBuffer megatron/` = 0，支持「Megatron 仍停在 V1」。
13. `7399c2b5:sglang/.../deepep_v2.py` L27–31 / L316–341、`deepep.py` L175 / L887、`deep_gemm.py` L1564 / L1699 —— `_EXPERT_ALIGNMENT = 128`、`do_cpu_sync_val = True` → `use_masked` 置 `False`、`do_expand=use_expand_layout`、`DeepEPBuffer` / `DeepEPDispatcher` / `DeepEPv2Dispatcher` 类名与布局转换函数；支持 §10.2。
14. `d8d53f17:vllm/.../all2all.py` L1005 / L1013–1014 / L1061 / L1079 —— `DeepEPV2All2AllManager`、`DeepEP >= 2.0` 与 `NCCL >= 2.30.4` 要求、GIN 探测失败信息、`handle.get_theoretical_num_sms(...)` 结果被用作 `_num_sms`；支持 §10.4。

## 3. 教学代码实跑（本机，CPU/Gloo，非 GPU 实测）

```bash
cd torch/deepep/codes
python3 minimal_moe.py
torchrun --standalone --nproc-per-node=4 distributed_ep.py
```

- `minimal_moe.py`：4 个用例全部 `[PASS]`，最大绝对误差 ≤ 2.220e-16；脚本同时打印 `assignment = 8，按 (token, 目的 rank) 去重后 = 6，涉及的目的 rank = 4` 与权重算例 `12.0 / 30.0 / 7.2`。与 §1.1、§3、§9 正文一致。
- `distributed_ep.py`：4 个用例全部 `[PASS]`，最大绝对误差 4.441e-16（含「某个 rank 完全没有 token」「所有 rank 都没有 token」）；rank 0 在常规路由用例下打印「本实现发送行数=8，按 (token,目的 rank) 去重后本可为 **7** 行」，在「同一 token 命中同一 rank 的两个专家」用例下为 **4** 行。与 §3 第三条 bullet 逐字一致（此处是上一轮 P1 的修复点，本轮复跑确认已闭合）。
- 退出码均为 0。**这两个脚本不测量任何通信性能**，也不加载 DeepEP；正文与 `codes/readme.md` 均如此声明。

## 4. 本轮做出的写入（仅两处，均在授权范围内）

| 文件 | 改动 | 理由 |
| --- | --- | --- |
| `PROMPT-learn-pipeline.md` | 回写 7 条源码修正：事实表标注来源与修正点；新增「阶段产物现状」的当前状态行；把第 13 条清单里被源码推翻的两项（`handle` 契约、`min_s max` 模型）改写；修正「DeepEP V1 示意图」一行对两张官方图的错误描述 | 用户本轮明确要求把 7 条修正回写进事实表 |
| `learn-plan.md` | 第 95 / 108 / 109 行的「参考资源 / 数学复核方式」不再把 `docs/legacy.md` 挂在 tag `v1.2.1` 语境下，改指 tag 自带 `README.md` 的性能表并注明归档版属于 `01dc3aac` | P1：与用户确认的第 3 条修正属同一事实错误（该 tag 树里没有 `docs/`），属事实错误而非风格偏好 |

未写入：`deep-dive.md`（本轮 0 条 P0，正文未改）、`codes/`、`pics/`、两份 draft、`README.md` / `README-cn.md`、`knowledge-graph.json`。

## 5. 未覆盖 / 边界

- 任何 GPU、NVLink、RDMA、NCCL Gin 行为均**未实测**；正文里所有性能数字都只是官方 README / 官方文档口径的引用。
- Mermaid 只做了源码级检查（围栏配对、语法形态），**未渲染**。
- 图片只做了「引用 ↔ 文件」双向一致与命名/出处核对；与 `references/` 源图的字节一致性依据前几轮的记录（`notes/IMAGE-VERIFY.md`、`research/verify-track2-snapshot.md`），本轮未重复比对。
- 两个教学脚本的实跑环境是 Python 3.12 + PyTorch 2.10.0+cu129 的 CPU 路径；`torchrun` 用 Gloo 后端。
