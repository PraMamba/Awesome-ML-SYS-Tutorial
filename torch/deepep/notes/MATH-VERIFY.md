# 《DeepEP 深度学习笔记》终稿：数学与源码引用独立复核

> 复核人：`/root/track5_math/track_math_code`（独立于文稿写作轨道，未参与 `deep-dive.md`、`learn-plan.md`、`codes/` 的写作）
> 复核日期：2026-09-10
> 复核对象：`torch/deepep/deep-dive.md`，763 行，`sha256 = 92081431015034e46ecf89e57f1dfb2a5a794d794319d1e143e0572229f4474e`
> 对照基线：`/workspace/algorithm/DeepEP` 工作树 HEAD `01dc3aac…`（V2）+ tag `v1.2.1`（V1）、`/workspace/algorithm/sglang` HEAD `7399c2b5…`、`/workspace/algorithm/Megatron-LM` HEAD `0d7ecc5d…`
> 写入边界：本文件是唯一新增产物。未修改文稿、草稿、`codes/`、任何源码仓库，未 commit，未运行 GPU。

复核对象在写作过程中仍在变动：数学与引用结论先在 752 行版本（`sha256 f3a40458…`）上完成，随后在 763 行版本（上面的 sha256）上**逐条复核缺陷仍然存在**并重新对齐了行号。若文稿再次编辑，行号需重新对齐。

## 一、结论摘要

| 复核项 | 结论 |
| --- | --- |
| MoE 路由式与门控语义（§1、§4.3） | **成立**。`Gate.forward` 的 softmax/sigmoid、bias 只影响选择、`route_scale=2.5` 三项描述与源码逐句一致 |
| assignment 矩阵与反向（§2） | **成立**。$D^{\top}D=KI$、$Y=D^{\top}(p\odot F(DX))$、$dX=D^{\top}dZ$ 均可独立复算；Megatron 的 `FusedDispatch.backward → combine` 与 `topk_weights=grad_token_probs` 属实 |
| 通信量算例（§5） | **成立且可复算**。448 MiB / 231.0 MiB / 51.6% / 3.03% / 0.25 MiB / 0.06% 六个数全部精确复现 |
| V1 低延迟布局与容量契约（§7.1、§7.2） | **成立**。shape、`recv_count`、「容量 ≠ 有效 token 数」、send/recv 两阶段与 hook 约束均可在 tag `v1.2.1` 定位 |
| V2 最坏情况容量公式（§8.3） | **成立**。三行源码与公式逐项对应；SGLang 取值算例 8192 / 4064 / 12256 → 12288、+50% 复算无误 |
| SM / QP 估算模型（§8.4） | **公式结论成立，两处表述需修正**（见 D2、D3）。文稿没有使用「$\min_s\max(T_{\rm comm},T_{\rm compute})$」这一与实际代码不符的模型，处理正确 |
| 逻辑带宽 vs 物理带宽（§11） | **成立**。400 Gb/s = 50 GB/s、`ibstat Rate / 8`、README 第 53 行脚注、61–91 与 726–740 不可当硬件常数，均有据 |
| 权重乘法位置表（§9） | **成立**。V1 normal 不乘 / V1 低延迟乘 / V2 不乘，三条 docstring 与 kernel 断言逐条对上；12 / 30 / 7.2 算例实测复现 |
| handle 复用契约（§8.1、§8.2） | **成立**。`topk_idx is None`、`do_cpu_sync` 必须为假、psum 与 `expert_alignment` 语义与 docstring 一致 |
| 教学代码实测声明（§3） | **数值成立，一处归属写错**（见 D1）。两个脚本 4+4 用例全部通过，误差上界留有裕量 |
| 其他行号抽查（§6、§7.2、§8.5–§8.7、§10、§12） | **成立**。抽查的 20 余处 pin 均在对应 revision 下命中 |
| 缺陷总数 | 7 处（0 处致命、3 处需改、4 处细节），逐条见第三节 |

## 二、逐项复核

### 2.1 MoE 路由语义与门控

`y_t=\sum_k p_{t,k}f_{e_{t,k}}(x_t)` 与「只统计 routed experts、共享专家不过路由器」的界定成立：官方参考实现里 `y` 只累加 routed experts，`z = self.shared_experts(x)` 之后 `return (y + z).view(shape)`。

§4.3 对门控的三条描述逐句核对（`inference/model.py`，commit `b15f0dbb…`）：

1. `softmax` 分支直接归一化、`sigmoid` 分支不归一化 —— 源码 `if self.score_func == "softmax": scores.softmax(dim=-1, dtype=torch.float32) else: scores.sigmoid()`；
2. bias 只参与专家选择、不进入权重数值 —— `original_scores = scores` 取在加 bias 之前，`weights = original_scores.gather(1, indices)`；
3. 只有 sigmoid 分支再归一化，最后统一乘 `route_scale`，671B 配置为 2.5 —— `if self.score_func == "sigmoid": weights /= weights.sum(...)`，`weights *= self.route_scale`。

`config_671B.json`（commit `4c2fdb8f…`）字段核对：`dim=7168`、`n_routed_experts=256`、`n_shared_experts=1`、`n_activated_experts=8`、`n_expert_groups=8`、`n_limited_groups=4`、`route_scale=2.5`，与 §4.2 表格逐行一致。

### 2.2 assignment 矩阵、combine 与反向

- $D$ 每行恰有一个 1，故 $D^{\top}D=K I$（$K$ 为每个 token 的有效 assignment 数）：$F=\mathrm{id}$ 且不乘权重时 $\sum_k x_t=Kx_t$，即文稿「得到的是 $KX$ 而不是 $X$」**成立**。
- $Y_t=\sum_k p_{t,k}x_t=(\sum_k p_{t,k})x_t$，因此「只有权重满足归一化条件才可能回到 $X$」是**必要条件**而非修辞，与 §4.3 的 `route_scale` 事实互相支撑。
- $Z=DX$ 对 $X$ 求导得 $dX=D^{\top}dZ$，形状 $M\to N$ 的归约方向**成立**；前提「路由索引离散、不参与求导」文稿已显式写出。
- Megatron 侧核对（`fused_a2a.py`，HEAD `0d7ecc5d…`）：`FusedDispatch.backward` 调用 `buffer.combine(...)`（L152）并以 `topk_weights=grad_token_probs.float()` 传入（L155），§2 表格与 §9 结尾的解释**成立**。

### 2.3 通信量算例（§5）

按文稿给出的假设（每个 assignment 独立传输）逐式复算：

| 量 | 文稿值 | 复算 |
| --- | --- | --- |
| assignment 数 $T\cdot K$ | 32,768 | $4096\times8=32768$ |
| $V_{\rm BF16}$ | 469,762,048 B = 448.0 MiB | $32768\times7168\times2$，除以 $1024^2$ 得 448.0 |
| 单条 FP8 assignment | 7168 + 224 = 7392 B | $7168/128=56$ 个 scale，$56\times4=224$ |
| $V_{\rm FP8}$ | 242,221,056 B = 231.0 MiB | $32768\times7392$，除以 $1024^2$ 得 231.0 |
| 比值 | 51.6% | 0.515625 |
| scale 占比 | 3.03% | $224/7392=3.0303\%$ |
| `topk_idx` 元数据 | 0.25 MiB，0.06% | $4096\times8\times8=262{,}144$ B；$262144/1048576=0.25$；$262144/469762048=0.0558\%$ |

文稿已显式声明「只计激活值与 scale，不计索引与目的信息」，与算例口径一致；「多出来的 1.6 个百分点全部来自 scale」也准确（$51.5625\%-50\%$）。

### 2.4 V1 低延迟的布局与容量（§7.1、§7.2）

在 tag `v1.2.1`（`9af0e0d0…`）下逐条命中：

- `low_latency_dispatch` 的返回是 `[num_local_experts, num_max_dispatch_tokens_per_rank * num_ranks, hidden]`，即文稿的 $E_{\rm local}\times(P\cdot T_{\max})\times H$；`use_fp8=False` 时为 `bfloat16`，`True` 时为 FP8 加 `hidden // 128` 的 scale；
- `recv_count` 为 `[num_local_experts]`，docstring 原句「not all tokens are valid, only some of the `num_max_dispatch_tokens_per_rank * num_ranks` are, as we do not synchronize CPU received count with GPU」与 §7.1 的「容量 ≠ 有效 token 数」一致；
- 单条消息字节数 `sizeof(int4) + max(hidden * sizeof(nv_bfloat16), hidden + num_scales * sizeof(float))`、接收侧 `num_experts * num_max_dispatch_tokens_per_rank` 且整体两份 —— `csrc/config.hpp` 的 `LowLatencyLayout` 逐行一致；
- 每 (本地专家, 源 rank) 槽位的分配来自 `atomicAdd(atomic_counter_per_expert + dst_expert_idx, 1)`；
- `x.size(0) <= num_max_dispatch_tokens_per_rank` 的 host 断言存在；
- send / recv 两次启动：`LOW_LATENCY_SEND_PHASE` / `LOW_LATENCY_RECV_PHASE` 提前返回，`recv_hook = [=]() { launcher(LOW_LATENCY_RECV_PHASE); }`；hook 模式用 compute stream 且 `EP_HOST_ASSERT(not (async and return_recv_hook))`；
- tag 的 `README.md` 第 243 行确有「there is no SM control API for the low-latency kernels」，§7.2 对「0 SM」的收窄成立。
- §7.3 的 `allow_nvlink_for_low_latency_mode=True` 默认值与「somehow incompatible with the hook-based overlapping」提示均在构造函数中；同机 P2P 分支（`nvshmemi_get_p2p_ptr` 非 0 即走拷贝）确实存在，「低延迟只走 RDMA」的过度概括被正确否掉。

### 2.5 V2 最坏情况容量公式（§8.3）

`csrc/elastic/buffer.hpp`（`01dc3aac…`）三行与公式逐项对应：

```text
num_recv_tokens     = num_max_tokens_per_rank * nccl_context->num_ranks
num_expanded_tokens = nccl_context->num_ranks * num_max_tokens_per_rank * std::min(num_topk, num_local_experts)
num_expanded_tokens += (expert_alignment - 1) * num_local_experts
num_expanded_tokens = math::align(num_expanded_tokens, expert_alignment)
```

两点确认：`nccl_context->num_ranks` 是 `scaleout_ranks × scaleup_ranks` 的**全局 rank 数**（`api.cuh` 中与 `num_scaleup_ranks` 并列的两个独立字段），所以 $P$ 是 EP 组总规模而不是 NVLink 域规模，文稿没有混用；`math::align(a,b)=ceil\_div(a,b)*b` 的定义与 `math.cuh` 一致。

SGLang 算例复算：$8\times128\times8=8192$，$127\times32=4064$，和为 12256，对齐到 128 得 $96\times128=12288$，相对 8192 恰为 +50%。`_EXPERT_ALIGNMENT = 128` 在 `deepep_v2.py` 第 29 行属实。

### 2.6 SM / QP 估算（§8.4）

`get_theoretical_num_sms` 的真实结构是「按均衡路由假设估计期望命中 rank 数 → 把 scaleout/scaleup 流量分别折进 RDMA/NVLink 带宽 → 取折后比值较大的那条链路作为瓶颈 → 由瓶颈反推 SM 数，再乘 1.25、对齐到偶数、下限 4」：文稿的描述与代码一致，且没有写成「$\min_s\max(T_{\rm comm},T_{\rm compute})$」那种与实现不符的形式（这一点是本条复核的重点，结果通过）。

`get_expected_topk` 的组合数形式（$\text{groups}\cdot(1-\binom{E-E/G}{K}/\binom{E}{K})$）、`num_scaleout_topk == 0` 的断言、docstring 的「This assumes a balanced gate distribution.」与注释「For V3.0's group-limited gate, please do not use this function」、以及两个 TODO（`do_expand`、`allow_multiple_reduction`）均属实。

带宽探测：`get_rdma_gbs` 读 `ibstat` 的 `Rate` 后 `rate / 8`，`get_nvlink_gbs` 累加 `nvidia-smi nvlink -s` 的各链路带宽再乘 0.9 —— §8.4 末段与 §11 的单位换算链由此闭合。

QP 部分见 D2、D3。

### 2.7 逻辑带宽与物理带宽（§11）

$400\ \mathrm{Gb/s}/8=50\ \mathrm{GB/s}$ 成立；README 第 53 行脚注原文「the results are logical bandwidth. For example, under the `EP 8 x 2` case, 90 GB/s actually contains local rank traffic」确认「逻辑带宽分子含本地 rank 流量，因此可超过单链路物理上限」这一解释是官方口径而非本文推断；README 第 19、22、30、55 行的 EP2048、SM 24→4–6、0 SM RDMA 不再支持、1.3x/4x 四处引用全部命中。

§11 对 V1 那条数字的条件描述（H800 + CX7 400 Gb/s、4096 tokens/batch、7168 hidden、top-4 groups / top-8 experts、FP8 dispatch + BF16 combine）与 `docs/legacy.md` 第 18–19 行的测试条件原句一致；低延迟表的 EP64 行 173 μs / 314 μs 与 RDMA 43 / 46 GB/s 也逐格对上。

### 2.8 权重乘法位置（§9）

| 路径 | 文稿结论 | 核对结果 |
| --- | --- | --- |
| V1 `Buffer.combine` | 不乘 | docstring「addition **without** weights」，成立 |
| V1 `low_latency_combine` | 会乘 | docstring「reduce **with weights**」，成立 |
| V2 `ElasticBuffer.combine` | 不乘 | `combine_reduce` 签名只含 topk slot、buffer 指针、bias，无权重参数；kernel 把 `topk_weights` 原样写入待发送 buffer（`combine.cuh` L216–224）；expand 路径有 `EP_DEVICE_ASSERT(topk_weights == nullptr)`（L69）；`combined_topk_weights` 作为独立返回值存在 |

算例 $0.8\times10+0.2\times20=12$、$10+20=30$、$0.8^2\times10+0.2^2\times20=7.2$ 三式复算无误，且已由教学脚本实际打印（见 2.10）。

### 2.9 handle 复用契约（§8.1、§8.2）

- `assert topk_idx is None`、`assert do_cpu_sync is None or not do_cpu_sync, 'Cannot do CPU sync with cached handle'` 位于 `elastic.py` 第 938–939 行（文稿引用的 L929–L941 覆盖该区间）；docstring 另有「`topk_idx` must be `None` (reused from handle)」。
- `num_experts / expert_alignment / num_max_tokens_per_rank` 必须与 handle 一致，有独立断言。
- `EPHandle` 的 psum 语义（按 scaleup rank 去重、expand 模式下 `psum[i]` 为「对齐前置累计 + 本专家未对齐计数」、`num_unaligned_recv_tokens_per_expert` 仅 expand 填充）与 docstring 逐句一致；`# May not be accurate without CPU sync` 注释位于 `num_recv_tokens` 赋值前。
- 补一条口径提醒：`topk_weights` 在 handle 复用**没有**被断言为 `None`，docstring 明确「`topk_weights` can be optionally provided (e.g. for backward pass with cached expand)」。文稿没有写错，但若后续要引用「handle 复用时 topk_idx/topk_weights 都必须为 None」这类说法，需要按源码修正。

### 2.10 教学代码实测复核（§3）

本轮独立重跑（强制 CPU，未触碰 GPU）：

```bash
cd torch/deepep/codes && CUDA_VISIBLE_DEVICES="" python3 minimal_moe.py
cd torch/deepep/codes && CUDA_VISIBLE_DEVICES="" torchrun --standalone --nproc-per-node=4 distributed_ep.py
```

`minimal_moe.py`：四个用例全部 `[PASS]`，最大绝对误差 2.220e-16 / 0 / 2.220e-16 / 0，与文稿「≤ 2.3e-16」相符；并打印 `token-expert assignment = 8，按 (token, 目的 rank) 去重后 = 6，涉及的目的 rank = 4` 与 `12.0 / 30.0 / 7.2`。

`distributed_ep.py`：四个用例全部 `[PASS]`，最大绝对误差 4.441e-16（末例 0），与文稿「≤ 4.5e-16」相符；rank0 统计为 8/8/7/11、8/8/7/7、8/8/4/8、0/0/0/0。

由此：文稿与 `codes/readme.md` 中所有误差声明与「第 3 个用例 8 → 4 行」的声明都成立；但 §3 第三条 bullet 的「8 → 6 行」归属写错，见 D1。

### 2.11 其他行号抽查

| 文稿位置 | 抽查结果 |
| --- | --- |
| §6 V1 构造签名与 `get_dispatch_layout` 返回值 | tag `v1.2.1` 下逐字命中；`num_tokens_per_rdma_rank` 在 intranode 为 `None` |
| §6 `is_token_in_rank` 的布尔化 | `layout.cu` 中先 `(is_in_rank[j] > 0)` 再累加，另一处 `num_tokens_per_*` 同样按布尔量计数 |
| §6 warp 角色与通道资源 | `WarpRole` 五角色齐全；`kNumDispatchRDMASenderWarps = 7` 在第 1008 行、`SETUP_LAUNCH_CONFIG(num_channels * 2, (7 + 1 + NUM_MAX_NVL_PEERS) * 32, stream)` 在第 1039 行 —— 与「每通道两个 SM、发送侧 7 个 warp」精确对上 |
| §6 `num_worst_tokens` | docstring 说明其为 intranode-only，internode 分支有 `assert num_worst_tokens == 0` |
| §8.1 `ElasticBuffer.__init__` 签名 | 与 `elastic.py` 第 228–246 行逐参数一致 |
| §8.5 NCCL Gin 封装与 barrier | `handle.cuh` 的 `put` / `put_value` / `red_add_rel` 与三个 team tag；`comm.cuh` 的 `NCCL_GIN_RESOURCE_SHARING_CTA/GPU` 判据（`kNumSMs == 1` 用 CTA，否则 GPU）；barrier 由 flush + 网格同步组合而成 |
| §8.6 warp 参数与静态约束 | `kNumNotifyWarps` / `kNumDispatchWarps`、hybrid 的 `kNumScaleoutWarps` / `kNumForwardWarps` / `kNumScaleoutRanks` / `kNumScaleupRanks` 与 `kNumScaleoutWarps == kNumForwardWarps` 断言齐备 |
| §8.7 编译策略 | `setup.py` 的 `sources` 含 `csrc/python_api.cpp`、`legacy/{layout,intranode,internode,internode_ll}.cu` 与三个 backend（nvshmem / nccl / cuda_driver），AOT 说法成立；JIT 缓存默认 `$HOME/.deep_ep`、可由 `EP_JIT_CACHE_DIR` 覆盖；README 第 353–355 行确有 `EP_JIT_DUMP_ASM/PTX/SASS` 三个开关（附录另有四个 build-time persistent 变量） |
| §10 Megatron 配置面 | `moe_token_dispatcher_type`、`moe_flex_dispatcher_backend`、`moe_flex_dispatcher_num_sms`、`moe_enable_deepep` 及 `__post_init__` 中对 deprecated 字段的路由均命中 |
| §10 SGLang 片段 | 引用的 `do_cpu_sync_val` / `use_masked` / `do_expand` / `use_tma_aligned_col_major_sf` 与注释原句一致（文稿为节选排版，参数值未改） |
| §12.1 PR #715 | 改动确为单文件 +2 行；`ptx::fence_acq_rel_sys()` 落在 `comm.cuh` 第 151–152 行 |

## 三、缺陷清单

编号沿用复核顺序，**没有致命项**；D1–D3 建议在 `learn-review` 前修掉。

| 编号 | 级别 | 位置 | 原文 | 问题与建议 |
| --- | --- | --- | --- | --- |
| D1 | P1 | `deep-dive.md:210`（§3 第三条 bullet） | 「第 1 章那个例子在四进程脚本里可以读到「本实现发送 8 行，按 (token, 目的 rank) 去重后本可为 6 行」」 | 实测四进程脚本 rank0 的三个数字是 **8 / 7（常规路由用例）** 与 **8 / 4（同 token 同 rank 双专家用例）**；「8 → 6」出自单进程 `minimal_moe.py` 打印的 §1.1 例子。建议改为「`minimal_moe.py` 打印 8 → 6；四进程脚本的第一个用例打印 8 → 7，第三个用例打印 8 → 4」 |
| D2 | P1 | `deep-dive.md:527`（§8.4） | 「非 hybrid 模式取 `min(num_sms, 8) + 1`」 | 源码是 `num_qps = min(num_sms, 8 + 1)`，即 `min(num_sms, 9)`。当 `num_sms < 9` 时两者不等（例：`num_sms=4` → 源码 4，文稿 5）。建议直接照抄 `min(num_sms, 8 + 1)`（保留作者写成 `8 + 1` 的原意）并注明 hybrid 分支为 `num_sms * 16 + 1`、最后统一按 `num_allocated_qps` 截断 |
| D3 | P2 | `deep-dive.md:527` 同一段的引用 | 「注释解释了那个额外的 1：它是留给 notify warp 的（[elastic.py#L836-L853]）」 | 该注释原文是 `# The extra QP is for notify warps`，位置在 `elastic.py:330`（`__init__` 的自动 QP 分配），不在所引用的 `get_theoretical_num_qps` 区间内。建议补一处 `elastic.py:330` 的引用 |
| D4 | P2 | `deep-dive.md:103`（§1.2） | 引用 `model.py#L690` 说明「共享专家的结果 `z` 最后单独加上去」 | 在 commit `b15f0dbb…` 下，`z = self.shared_experts(x)` 在第 690 行，相加发生在第 693 行 `return (y + z).view(shape)`。建议改引 `#L690-L693` 或 `#L693` |
| D5 | P2 | `codes/readme.md:37-38` | 「它同时打印文章 §2.2 的例子」「以及 §8 的权重位置算例」 | 终稿里这两个内容分别在 **§1.1** 与 **§9**。建议同步为终稿章节号，避免文稿与配套说明互相指向不存在的章节 |
| D6 | P2 | `deep-dive.md:257`（§4.2） | 「671B 总参数、每 token 激活 37B，61 层里前 3 层是 dense……这些数字不是我数出来的，而是官方配置里的字段（config_671B.json）」 | `config_671B.json` 中没有层数、dense 层数与激活参数量字段（实测仅含 `dim`、`n_routed_experts`、`n_shared_experts`、`n_activated_experts`、`n_expert_groups`、`n_limited_groups`、`route_scale` 等）。建议把 671B / 37B / 61 层的出处指向 Technical Report §2.1，JSON 只支撑表格里那一列 |
| D7 | P2 | `deep-dive.md:688`（§12.1） | 「在 2026-08-04T01:25:51Z 被合并」 | Git 侧 committer date 为 `2026-08-04T09:25:50+08:00`（= `01:25:50Z`），秒级与文稿差 1 秒（后者更像 GitHub API 的 `merged_at`）。建议统一为「2026-08-04（+0800）」或注明时间戳来源，避免两个来源混用 |

## 四、复现命令

```bash
# 版本基线
git -C /workspace/algorithm/DeepEP rev-parse HEAD v1.2.1^{commit}

# §5 通信量复算
python3 -c "
T,H,K=4096,7168,8
bf16=T*K*H*2; fp8=T*K*(H+(H//128)*4)
print(bf16, bf16/2**20, fp8, fp8/2**20, fp8/bf16, 224/7392, 4096*8*8)"

# §7.1 低延迟布局契约
git -C /workspace/algorithm/DeepEP show v1.2.1:deep_ep/buffer.py | sed -n '560,585p'
git -C /workspace/algorithm/DeepEP show v1.2.1:csrc/config.hpp | sed -n '129,166p'

# §8.3 容量公式与 §8.4 SM/QP
sed -n '1066,1071p' /workspace/algorithm/DeepEP/csrc/elastic/buffer.hpp
sed -n '729,853p' /workspace/algorithm/DeepEP/deep_ep/buffers/elastic.py
sed -n '326,334p' /workspace/algorithm/DeepEP/deep_ep/buffers/elastic.py   # notify warp 的 QP 注释

# §9 权重位置
git -C /workspace/algorithm/DeepEP show v1.2.1:deep_ep/buffer.py | sed -n '395,406p;598,606p'
sed -n '55,80p' /workspace/algorithm/DeepEP/deep_ep/include/deep_ep/impls/combine_utils.cuh

# 教学代码
cd /workspace/algorithm/Awesome-ML-SYS-Tutorial/torch/deepep/codes
CUDA_VISIBLE_DEVICES="" python3 minimal_moe.py
CUDA_VISIBLE_DEVICES="" torchrun --standalone --nproc-per-node=4 distributed_ep.py
```

## 五、边界与未覆盖

1. 本记录只做**数学、引用与教学代码**的独立复核；图片线由 `notes/IMAGE-VERIFY.md` 负责，两者不互相替代。
2. 所有性能数字均来自官方 README / `docs/legacy.md`，本轮**没有**运行任何 GPU kernel、DeepEP benchmark、NVLink/RDMA 测速；本记录中不存在本机实测性能结论。
3. 外部 pin（DeepSeek-V3 `model.py` / `config_671B.json`、PyTorch `distributed_c10d.py`、NCCL `gin.h`）本轮以已落盘的本地副本文本核对，未重新发起网络请求；行号结论以文稿引用的 commit 为准。
4. `deep-dive.md` 仍在编辑中：本记录的结论与行号绑定 `sha256 92081431…`（763 行）。后续若继续修改，D1–D7 需按新行号重新定位。
5. 本文件未修改任何其他产物，也不代表 `learn-review` 的最终结论；P0/P1 的处理结果仍以复审记录为准。
