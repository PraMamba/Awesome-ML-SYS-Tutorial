# 第二轮独立复核（终稿冻结前的链接 / 数学 / 图片 / 机械检查）

> 复核执行：`/root/track2_deepep_v2/track_math_verify`（未参与 `deep-dive.md`、`learn-plan.md`、`codes/` 的写作）
> 复核日期：2026-09-10
> 复核快照：`deep-dive.md` sha256 `ebb6579cff04813b8bd1498bc02248e76be2ac8308d5f1168d21ec29e25d58fe`（764 行）
> 对照基线：`/workspace/algorithm/DeepEP`（工作树 `01dc3aac…` + tag `v1.2.1`）、`Megatron-LM@0d7ecc5d`、`sglang@7399c2b5`、`vllm@d8d53f17`、`verl-project/verl@3467d90a`、`DeepSeek-V3@b15f0dbb` / `@4c2fdb8f`、`pytorch@e8430bed`、`NCCL@b0e84e58`
> 证据方式：`git show <sha>:<path>` 逐文件取回并比对行数；`gh api ...?ref=<sha>` 取回远程文件；本地实跑教学脚本；python3 复算

本文件只记录**我自己跑出来的**结论，不转载其他轨道的判断。第一轮的八维报告在 `REVIEW.md` 与 `REVIEW-independent-r1.md`，数学与图片的专项记录在 `notes/MATH-VERIFY.md` 与 `notes/IMAGE-VERIFY.md`。

---

## 一、外部源码引用：35 条去重目标，35 条通过

从终稿中抽出所有 `github.com/<org>/<repo>/(blob|tree)/<commit>/<path>[#Lx-Ly]`，按 `(repo, commit, path)` 去重得到 **35 个目标**（含 2 个只引用整文件的 `README.md`、`config_671B.json`）。逐条核对「文件在该 commit 下存在」且「文件行数 ≥ 引用到的最大行号」：

| 归属 revision | 目标数 | 核对方式 | 结果 |
| --- | ---: | --- | --- |
| DeepEP `01dc3aac…`（V2） | 14 | `git show`（本地工作树即该 revision） | 14/14 PASS |
| DeepEP `9af0e0d0…`（tag `v1.2.1`） | 7 | `git show <tag>:<path>` | 7/7 PASS |
| Megatron-LM `0d7ecc5d…` | 3 | `git show`（本地 HEAD 即该 commit） | 3/3 PASS |
| SGLang `7399c2b5…` | 4 | `git show`（本地 HEAD 即该 commit） | 4/4 PASS |
| vLLM `d8d53f17…` | 1 | `git show`（本地 HEAD 即该 commit） | 1/1 PASS |
| NCCL `b0e84e58…` | 1 | `gh api contents?ref=` | PASS（418 行 ≥ 177） |
| DeepSeek-V3 `b15f0dbb…` / `4c2fdb8f…` | 2 | `gh api contents?ref=` | 2/2 PASS（808 行 ≥ 693；21 行配置） |
| PyTorch `e8430bed…` | 1 | `gh api contents?ref=` | PASS（8187 行 ≥ 6136） |
| verl `3467d90a…` | 1 | `gh api contents?ref=` | PASS（88 行 ≥ 29） |

同时复核：全篇 **0 条** `blob/main` 或 `blob/master` 链接；`pics/` 之外的 7 条仓库内相对链接全部指向存在的文件。

## 二、数学与数值：逐项复算通过

| 终稿位置 | 声明 | 我的复算 |
| --- | --- | --- |
| §5 | BF16 dispatch 逻辑量 448 MiB | $4096\times8\times7168\times2=469{,}762{,}048$ B $=448.000$ MiB ✓ |
| §5 | FP8 + 每 128 元素一个 FP32 scale = 231 MiB，比例 51.6% | $32{,}768\times7392=242{,}221{,}056$ B $=231.000$ MiB；$0.515625$ ✓（scale 自身 224 B/条，占 3.03%） |
| §5 | SF-pack 口径的 56.25% | `ceil_div(7168,32)=224` 个 4 B pack → 每条 8064 B，总量 252.0 MiB，$252/448=56.25\%$ ✓ |
| §5 | `topk_idx` 元数据 0.25 MiB / 0.06% | $4096\times8\times8=262{,}144$ B $=0.25$ MiB，占 448 MiB 的 0.0558% ✓ |
| §8.3 | 容量公式与 SGLang 算例 12288（+50%） | $8\times128\times8=8192$；$(128-1)\times32=4064$；$12256\to$ 对齐 128 $=12288$，比最坏展开量多 50.0% ✓ |
| §8.4 | 「不是最少 SM，而是带 1.25 经验系数的建议值」 | 源码逐行核对：`align(max(4, math.ceil(num_sms * 1.25)), 2)`，再 `max(num_sms, 64)`（不重叠时），最后被设备 SM 数截断 ✓ |
| §8.4 | QP：direct `min(num_sms, 8 + 1)`；hybrid `num_sms * 16 + 1`；额外的 1 与 notify 有关 | `elastic.py` 源码逐字 ✓；「The extra QP is for notify warps」在上限分配处 ✓ |
| §9 | 12 / 30 / 7.2 | python3 复算 ✓，且 `minimal_moe.py` 实跑打印同值 |
| §11 | 400 Gb/s = 50 GB/s | $400/8=50$ ✓（十进制 GB/s，与 `envs.py` 的 `rate / 8` 一致） |

## 三、图片：双向一致

正文 12 处 `src="./pics/..."` 全部命中磁盘文件；`pics/` 下 12 张 jpg/png **全部被正文引用**，无孤儿、无死链。122 张抽取图的逐张处置在 `learn-plan.md` 的采用清单与 `pics/README.md` 第 2 节（MoE 7+17+10=34；V3 11+28+9+37+3=88）。

## 四、机械检查

| 检查 | 结果 |
| --- | --- |
| ASCII 字符画（`┌┐└┘│├┤┬┴┼─━┃`） | 0 |
| `——` | 0 |
| ` ```text ` 围栏 | 0（代码块均有语言标注，围栏配对：22/6/2） |
| Mermaid 图 | 4 |
| 行尾空白 | 0 |
| 性能数字被写成「实测」 | 0（开篇即声明「下表全部来自官方文档，不是本文的实测结果」） |

## 五、教学代码实跑（我自己这一轮跑的）

```bash
cd /workspace/algorithm/Awesome-ML-SYS-Tutorial/torch/deepep/codes
python3 minimal_moe.py
torchrun --standalone --nproc-per-node=4 distributed_ep.py
```

- `minimal_moe.py`：4 个用例全 PASS，最大绝对误差 ≤ 2.220e-16；打印 `assignment = 8，按 (token, 目的 rank) 去重后 = 6` 与 `12.0 / 30.0 / 7.2`。
- `distributed_ep.py`：4 个用例全 PASS，最大绝对误差 ≤ 4.441e-16；rank 0 的打印是 **8→7**（常规路由用例）与 **8→4**（同一 token 命中同一 rank 两个专家的用例），**不存在 8→6**——终稿已按此改正归属（8/6 出自单进程脚本）。

## 六、本轮发现并已修复的 3 处缺陷（第一轮报告未覆盖）

| 编号 | 缺陷 | 修复 |
| --- | --- | --- |
| R2-1（P1） | §10.3 把 `docs/perf/dpsk.md` 的引用写成 `#L25-L27`，但「Hopper + DeepEP」那句在第 29 行、「integrates Megatron」在第 5 行；组织名用的是会重定向的 `volcengine/verl` | 改为 `verl-project/verl` 的 `#L5-L29`，并在正文里写明两句话各自的行号；文末参考表同步 |
| R2-2（P1） | `learn-plan.md` 自检清单仍写「7 张进入 `pics/`」，与采用清单的 12 张不一致 | 改为 12 张，并写明「正文 12 处引用与目录双向一致」 |
| R2-3（P1） | `learn-plan.md` 把 verl 的 revision 写成「本地快照 `9ff05e32…`」，终稿实际引用的是文件级 pin `3467d90a…` | 版本分层表、P0-1 行、说明段三处同步为「verl 用文件级 pin `3467d90a…`（只引用 `docs/perf/dpsk.md`）」 |

## 七、未覆盖与残留

1. 全部性能数字仍是官方文档口径，本文**不存在任何 GPU / NVLink / RDMA 实测**；本轮也没有编译 CUDA、没有跑 DeepEP kernel。
2. 远程 5 个 pin 的核对只到「文件存在 + 行数足够」；行号处的**语义**已由第一轮与各研究轨道的抽读记录覆盖（`research/frameworks.md`、`research/facts-external-pins.md`）。
3. `pics/.superseded/` 里 4 个旧名副本与现役文件内容相同，按仓库规则未删除，只在本记录与 `pics/README.md` 中说明。
4. 仓库级不一致（`README.md:150` 的冗余 `[Pending Review]`、`readme-5` 的发布状态、`/workspace/algorithm/verl` remote 内嵌明文 token、`PROMPT-learn-pipeline.md` 事实表的两处过时表述）只上报，未擅自修改。
