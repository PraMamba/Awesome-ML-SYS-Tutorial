# DualPipe 文章逐式复核记录

对象：`torch/dualpipe/deep-dive.md`。复核方式：每个公式先定位来源（论文公式号 / 官方 README 表格 / 固定 commit 的源码行号），再自己重推一遍，最后写清符号定义、单位与前提假设。

复核用到的固定 revision：

- DeepSeek DualPipe：commit `030ce4325f4ebeb437da4ebc6d00a70469dd58ae`（本地 `/workspace/algorithm/DualPipe`，HEAD，工作树干净）
- Zero Bubble 论文：本地快照 `references/papers/zero-bubble/zero-bubble.md`（arXiv `2401.10241v1`）
- Controllable Memory 论文：本地快照 `references/papers/controllable-memory/controllable-memory.md`（arXiv `2405.15362v1`）
- 可执行复核：`codes/01_split_backward_minimal.py`、`codes/02_schedule_sim.py`

结论列的含义：**一致** = 文章写法与来源一致且已独立复算；**修正** = 文章相对来源做了改写，已记录理由；**删除** = 无法核实，已从文章删除。

---

## 一、记号与前提

| 符号 | 定义 | 单位 | 备注 |
| --- | --- | --- | --- |
| \(p\) | 流水线组中的物理设备数 | 台 | 与官方 README 的比较表里的 \(PP\)（stage 数）**不是同一个量** |
| \(s\) | 一份完整模型经过的逻辑 stage 数 | 个 | DualPipe 下 \(s=p\)；DualPipeV 下 \(s=2p\) |
| \(m\) | 一次逻辑迭代的**总** microbatch 数 | 个 | DualPipe 下两个方向各 \(m/2\)（`half_num_chunks = num_chunks // 2`）；DualPipeV 下单方向 \(m\) |
| \(F\) / \(I\) / \(W\) | 前向 / 输入梯度 / 参数梯度的耗时 | 时间 | \(W\) 不是优化器更新 |
| \(B_{\mathrm{full}}\) | 完整反向耗时 | 时间 | 等时简化下 \(B_{\mathrm{full}} = I + W\) |
| \(C\) | 配对执行的一个前向加一个完整反向的耗时 | 时间 | 即官方 README 的 \(F\&B\) |
| \(A\) | 一个标准 stage、一个 microbatch 的激活占用 | 显存 | 不是「全部激活」 |

等时假设 \(F = I = W = t\) 是 **Zero Bubble 论文 §2 手工调度分析所用的理想化前提**，不是实测结论。论文 §2.3 的解析口径给出 \(T_W < T_F < T_B\) 且 \(T_B + T_W = 2T_F\)；论文 Table 9 的 profiled 数据与这两条并不完全自洽（例如 1.5B / \(p=8\) / \(m=24\) 一组是 \(T_F=18.522\)、\(T_B=18.086\)、\(T_W=9.337\)，\(T_B+T_W=27.423\) 对 \(2T_F=37.044\)，相差约 26%，且出现 \(T_B < T_F\)）。**文章中此二口径分开标注，未合并。**

---

## 二、逐式复核

### 2.1 反向拆分

| 公式 | 来源 | 复算 | 结论 |
| --- | --- | --- | --- |
| \(Y = X\Theta^\top\)，\(X\in\mathbb{R}^{N\times H_{\mathrm{in}}}\)，\(\Theta\in\mathbb{R}^{H_{\mathrm{out}}\times H_{\mathrm{in}}}\) | 通用线性代数 | 形状自洽 | 一致 |
| \(I: \partial L/\partial X = G\Theta\)，\(G\in\mathbb{R}^{N\times H_{\mathrm{out}}}\) | Zero Bubble §2、Figure 1 | \((N\times H_{\mathrm{out}})(H_{\mathrm{out}}\times H_{\mathrm{in}}) \to N\times H_{\mathrm{in}}\) | 一致 |
| \(W: \partial L/\partial\Theta = G^\top X\) | Zero Bubble §2、Figure 1 | \((H_{\mathrm{out}}\times N)(N\times H_{\mathrm{in}}) \to H_{\mathrm{out}}\times H_{\mathrm{in}}\) | 一致 |
| 数值例子 `X[4,8]`、`Θ[16,8]`、`Y,G[4,16]` → `dX[4,8]`、`dΘ[16,8]` | 文章自建 | 与两式一致 | 一致 |
| 「\(W\) 在数学上不依赖 dX」 | Zero Bubble §2 | 两式都只需 \(G\) 与保存的 \(X\)；但整段 stage 中某些参数梯度的可用时间晚于输入梯度，因此文章写成「调度策略」而非「数学必然」 | **修正**（收紧表述） |
| 「延迟 \(W\) 需要保留 \(X\) 与 \(G\)」 | Zero Bubble §2.3 | 论文明确区分「仍需输入梯度反向的数据」与「只为参数梯度保留的数据」，并给 \(M_W < M_B\) | 一致 |
| 数值等价性实测 | `codes/01_split_backward_minimal.py` | 本机 PyTorch 2.10.0+cu129，float64：`max abs dX err = 0.000e+00`，`max abs dTheta err = 1.110e-16`，且 `backward()` 后 `theta.grad is None` 为 `True` | 一致 |

### 2.2 1F1B

| 公式 | 来源 | 复算 | 结论 |
| --- | --- | --- | --- |
| \(n_{\mathrm{warmup}}(r) = \min(p-r-1,\ m)\) | Megatron 非交错实现（`num_warmup_microbatches = p - rank - 1` 再与 `num_microbatches` 取小） | 与实现一致 | 一致 |
| 稳态配对 \((F_{k+p-r-1}, B_k)\) | 文章由 warmup 计数推导 | \(p=4\) 时 rank 0/1/2/3 的 warmup 前向数 3/2/1/0，第一组稳态配对 \((F_3,B_0),(F_2,B_0),(F_1,B_0),(F_0,B_0)\) | 一致 |
| 峰值激活 \(\min(m,p)A\)，且 \(m\ge p\)、\(A=A_{\mathrm{all}}/p\) 时 \(pA=A_{\mathrm{all}}\) | 文章推导 + 论文 Table 2 的 \(pM_B\) | 一致；文章已显式列出成立前提 | 一致 |
| \(T_{\mathrm{1F1B}} = (m+p-1)(F+B_{\mathrm{full}})\) | Zero Bubble 附录：「an 1F1B iteration takes \((m+p-1)(T_F+T_B+T_W)\)」 | 论文 \(T_B+T_W = B_{\mathrm{full}}\)，替换后一致 | 一致 |
| \(T_{\mathrm{busy}} = m(F+B_{\mathrm{full}})\)，\(T_{\mathrm{bubble}} = (p-1)(F+B_{\mathrm{full}})\) | 论文 Table 2 的 1F1B 行 \((p-1)(T_F+T_B+T_W)\) | 时长口径一致 | 一致 |
| \(\beta_{\mathrm{1F1B}} = (p-1)/(m+p-1)\) | 文章由上式相除 | \(3(p-1)t / [3mt + 3(p-1)t] = (p-1)/(m+p-1)\) ✓ | 一致 |
| 与官方 README 的 \((PP-1)(F+B)\) 不矛盾 | README 表头「same number of PP stages」 | README 的 \(B\) 是完整反向，与论文 \(T_B+T_W\) 对应；README 给的是时长，\(\beta\) 是比率，两者相差一个 \(T_{\mathrm{busy}}\) | 一致（文章已写明是口径差异） |

### 2.3 ZB-H1 / ZB-H2

| 公式 | 来源 | 复算 | 结论 |
| --- | --- | --- | --- |
| ZB-H1 稳态 \((F_x,\ I_{x-(p-1-r)},\ W_{x-(p-1)})\) | 论文 §2.1 + 文章按区间推导 | warmup1 得 \(p-1-i\) 个纯 \(F\)，warmup2 得 \(i\) 个 \((F,I)\) | 一致 |
| ZB-H1 气泡 \((p-1)(T_F+T_B-T_W)\) | 论文 Table 2 | 直接引用 | 一致 |
| 换记号后 \((p-1)(F+I-W) = (p-1)(F+B_{\mathrm{full}}-2W)\) | 文章改写 | \(B_{\mathrm{full}} = I+W\)，代入即得 ✓ | 一致 |
| \(F=I=W=t\) 时 H1 气泡 \(=(p-1)t\) | 文章代入 | \((p-1)(t+t-t) = (p-1)t\) ✓；论文正文「reduced to a third of 1F1B's size」对应 \(3(p-1)t \to (p-1)t\) ✓ | 一致 |
| ZB-H2 气泡 \((p-1)(T_F+T_B-2T_W)\) | 论文 Table 2 | 直接引用 | 一致 |
| \(F=I=W=t\) 时 H2 气泡 \(=0\) | 文章代入 | \((p-1)(t+t-2t)=0\) ✓ 与论文「zero bubble」「平行四边形」一致 | 一致 |
| ZB-H1 峰值激活 \(pM_B\)、H2 峰值激活 \((2p-1)M_B\) | 论文 §2.3 与 Table 2 | worker \(i\) 的激活分别为 \((p-i+1)M_B+(i-1)M_W\) 与 \((2p-2i+1)M_B+(2i-2)M_W\)；因 \(M_W<M_B\)，峰值都在 \(i=1\)，即 \(pM_B\) 与 \((2p-1)M_B\) ✓ | 一致 |
| 「绕过优化器同步」是 H2 零气泡的前提 | 论文 §2.2、§4 与 Figure 4 | 论文明确写「the synchronization between the optimizer steps is removed here」 | 一致 |
| README 的 ZB1P 行 \((PP-1)(F+B-2W)\) 等于 ZB-H1 | 文章换算 | \((p-1)(F+I+W-2W) = (p-1)(F+I-W)\) ✓ | 一致 |

### 2.4 DualPipe / DualPipeV 调度

| 公式 | 来源 | 复算 | 结论 |
| --- | --- | --- | --- |
| DualPipe 稳态 \((F_{0,x},\ B_{1,x-p/2},\ F_{1,x-p/2+1+r},\ B_{0,x-p+1+r})\) | `dualpipe.py` L381–396 的循环结构 | **两条独立路径互相印证**：① 手算 `p=8,r=1,x=8` 得 \((F_{0,8},B_{1,4},F_{1,6},B_{0,2})\)，与草稿示例一致；② 由 warmup 计数得第一组稳态 \(x=6\)，代入得 \((F_{0,6},B_{1,2},F_{1,4},B_{0,0})\)，与 `codes/02_schedule_sim.py --p 8 --m 20` 对 rank 1 打印的结果一致。\(p=4\) 下 rank 1 得 \((F_{0,2},B_{1,0},F_{1,2},B_{0,0})\)，模拟器一致 | 一致 |
| \(0 \le r < p/2\) 的来源 | 文章说明 | 上半区里 \(b_{1,i} \ge b_{0,i}\) 恒成立；跨过中点后关系翻转，公式需换对称形式。官方用 `phase ^= self.is_in_second_half` 实现 | 一致 |
| DualPipeV 稳态 \((F_{0,x},\ B_{1,x-p},\ F_{1,x-p+1+r},\ B_{0,x-2p+1+r})\) | `dualpipev.py` L352–367 | \(p=4,m=10,r=1\)：warmup 后 \(F_0=6,F_1=4,B_1=2,B_0=0\) → 第一组 \((F_{0,6},B_{1,2},F_{1,4},B_{0,0})\)；模拟器 `--p 4 --m 10` 输出一致 | 一致 |
| DualPipeV 的 \(\text{stage}_0=r,\ \text{stage}_1=2p-1-r\) | `example_dualpipev.py` L137：`full_modules[rank]`、`full_modules[pp_size*2-1-rank]`；PyTorch `_utils.py` L104–119 的 `style="v"` | \(p=4\) 时 rank 3 持 stage 3 与 4，rank 0 持 stage 0 与 7 | 一致 |
| V 底部为本地交接 | `dualpipev.py` L79–80（前向）与 L115–116（反向） | 在 `is_last_rank and phase == 0` / `is_last_rank and phase == 1` 处直接追加进本地队列 | 一致 |
| DualPipe 的入口约束 | `dualpipe.py` L332–333 | `num_ranks % 2 == 0`；`num_chunks > 0 and num_chunks % 2 == 0 and num_chunks >= num_ranks * 2` | 一致 |
| DualPipeV 的入口约束 | `dualpipev.py` L318 | `num_chunks > 0 and num_chunks >= num_ranks * 2`，**无偶数要求** | 一致 |
| 八步循环次数（DualPipe） | `dualpipe.py` L358–425 | step1 \(=2(p/2-r-1)\)、step2 \(=r+1\)、step3 \(=p/2-r-1\)、step4 \(=m/2-p+r+1\)、step5 \(=p/2-r-1\)、step6 \(=r+1\)、step7 \(=p/2-r-1\)、step8 \(=r+1\)；与文章表格逐项一致 | 一致 |
| 八步循环次数（DualPipeV） | `dualpipev.py` L330–395 | step1 \(=2(p-r-1)\)、step2 \(=r+1\)、step3 \(=p-r-1\)、step4 \(=m-2p+r+1\)、step5 \(=p-r-1\)、step6 \(=r+1\)、step7 \(=p-r-1\)、step8 \(=r+1\) | 一致 |
| \(W\) 队列会计：入队数 = 出队数 | `dualpipe.py` L373–425（DualPipe）、`dualpipev.py` L344–396（DualPipeV）与末尾 `assert WeightGradStore.funcs_queue.empty()` | DualPipe：入队 \(=\) step3 \(+\) step6 中 \(enable\_zb\) 者 \(+\) step7；出队 \(=\) step3 \(+\) step7 \(+\) step8，故要求 step6 中恰有 \(r+1\) 个 \(enable\_zb\)，与 step8 \(=r+1\) 吻合。DualPipeV 同理（用 `rank` 代替 `half_rank`）。`codes/02_schedule_sim.py` 对两台设备逐一断言，不通过即报错 | 一致 |

### 2.5 气泡与显存重算（第七章）

> **本节在第 3 轮整体重算过一次。** 第 1、2 轮把 \(F\&B = \max(F, B_{\mathrm{full}})\) 当主口径，第 3 轮审核用「官方调度图逐格取色」与「按官方提交语义重放」两条独立证据推翻了这个读法，本节已换成与官方材料一致的 \(F\&B = F + B_{\mathrm{full}}\)。改动记录见 `REVISION.md` 第 3 轮。

| 公式 | 来源 | 复算 | 结论 |
| --- | --- | --- | --- |
| 官方表 DualPipe/DualPipeV 气泡 \((PP/2-1)(F\&B+B-3W)\) | 官方 README | 原文照抄；表头注明「based on the same number of PP stages」 | 一致 |
| **\(F\&B\) 取哪一端** | 官方调度图逐格取色 + DeepSeek-V3 报告 Figure 4 | `dualpipe.png`（p=8, m=20）共 8 行、每行 **66** 个单位，其中前向 20、其余 40 为反向三类、**空闲 6**；代入 \((PP/2-1)(F\&B+B-3W)\)，只有 \(F\&B = F+B_{\mathrm{full}} = 3t\) 才给出 6（取 \(\max = 2t\) 会给出 3）。V3 报告 Figure 4 也只有一条 Computation 行、内部串行，被藏起来的是通信 | **修正**（第 3 轮：主口径由 \(\max\) 改为 \(F+B_{\mathrm{full}}\)） |
| 每设备有效工作 \(3mt\)（三种方法相同） | 文章推导 + `codes/02` 模拟输出 | 1F1B：\(m\cdot3t\)；DualPipe：\(2\times(m/2)\times3t\)；DualPipeV：\(2\times m\times(3t/2)\)（半 chunk）。扫掠 32 组全部命中 \(3m\) | 一致 |
| DualPipe 气泡 \((p-2)t\) | 文章推导 + 模拟器 | 模拟器：\(p=2,4,6,8\) → \(0,2,4,6\)，即 \(p-2\) ✓，与 \(m\) 无关；官方图 p=8 的空闲 6 与之相同 | 一致 |
| DualPipeV 气泡 \((p-1)t\) | 同上 | 半 chunk 单位下实测 \(2(p-1)\)（p=2,4,6,8 → 2,6,10,14），换算到 \(t\) 单位即 \(p-1\) ✓；官方图 p=4 的空闲 6 与之相同 | 一致 |
| \(\beta_{\mathrm{DP}} = (p-2)/(3m+p-2)\) | 文章推导 + 模拟器 | \(p=4,m=10\)：\(2/32 = 6.25\%\)，模拟器 \(2/32\) ✓；\(p=8,m=20\)：\(6/66 = 9.0909\%\)，模拟器 \(6/66\) ✓；扫掠全部命中 | 一致 |
| \(\beta_{\mathrm{DPV}} = (p-1)/(3m+p-1)\) | 同上 | \(p=4,m=10\)：\(3/33 = 9.0909\%\)，模拟器 \(3/33\) ✓；\(p=8,m=20\)：\(7/67 = 10.4478\%\)，模拟器 \(14/134\) ✓ | 一致 |
| **交叉验证**：官方式在 \(F\&B = F+B_{\mathrm{full}} = 3t\) 下等于 \((p/2-1)\cdot 2t = (p-2)t\) | 文章推导 | \((p/2-1)(3t+2t-3t) = (p-2)t\) ✓，与模拟器一致 | 一致 |
| DualPipeV 同式在 \(F\&B' = 3\tau = 1.5t\) 下等于 \((p-1)t\) | 文章推导 | \((s/2-1)(F\&B'+B'-3W')\)，\(s=2p\)、\(\tau=t/2\) → \((p-1)(1.5t+t-1.5t) = (p-1)t\) ✓ | 一致 |
| **三种调度的气泡率分母首项都是 \(3m\)** | 文章论断 | 1F1B \((p-1)/(m+p-1)\)、ZB1P \((p-1)/(3m+p-1)\)、DualPipe \((p-2)/(3m+p-2)\)、DualPipeV \((p-1)/(3m+p-1)\)。第 1、2 轮曾提出「DualPipe/DualPipeV 的分母是 \(6m\)」，那是 \(\max\) 口径的产物，第 3 轮**删除** | **修正**（推翻第 2 轮的 \(6m\) 论述） |
| 峰值激活 \(pA\)、\((p+1)A\)、\((p+\tfrac12)A\) | 官方 README 的 \(PP\)、\(PP+1\)；文章换算 | DualPipeV 在固定设备数下 stage 减半，\((2p+1)\times A/2 = (p+\tfrac12)A\) ✓ | 一致 |
| PP 边界通信约 `2x` | **只有 Sea AI Lab 博客**（对照表与结论）；官方 README 全文无此表述 | 逻辑 stage 数由 \(p\) 变 \(2p\)，边界穿越次数翻倍 | 一致（来源已在第 2 轮改正） |
| 隐藏状态尺寸 \(2\times2048\times4096\times2 = 32\) MiB | 文章自算 | \(2\times2048\times4096 = 16{,}777{,}216\) 元素；bf16 每元素 2 字节 → \(33{,}554{,}432\) B \(= 32\) MiB ✓ | 一致 |

### 2.6 模拟器的释放语义（第 3 轮新增）

| 检查点 | 复算 | 结论 |
| --- | --- | --- |
| 跨 rank 依赖是否应挂到「提交点」而不是「计算任务结束时刻」 | 官方 `_send_forward` / `_send_backward` 只把 `P2POp` 追加进 `self.comm_ops`，真正发出在下一次 `_commit_and_wait_comm()`（`dualpipe.py` L188/L198/L209/L220，`step()` 结尾 L427 还有一次）。按提交点建模后，\(p=8,m=20\) 得 66/6、\(p=4,m=10\) 得 66/6，与官方两张调度图逐格取色一致；不按提交点建模会各自得到 63/3 与 63/3（气泡少一半） | **修正**（第 3 轮修复；此前版本的「与 \(\max\) 端吻合」正是这一处乐观假设造成的数值巧合） |
| V 底部的本地交接 | `dualpipev.py` L79-80 / L115-118 在计算结束时直接 `append` 进本地队列，不经过网络提交点，因此那条边挂计算任务本身 | 一致 |
| 非 zb 反向记成 \(I \to W\) 而非示例 hook 的 \(W \to \text{dgrad}\) | 在 11 组 \((p,m)\) 配置下重算，气泡全部不变（依赖边结构不变，只是 \(I\) 的完成时刻后移 1 个单位、未落在关键路径上） | 一致（文章已写明该取舍） |

---

## 三、已修正 / 已删除的内容

| 原草稿说法 | 处理 | 依据 |
| --- | --- | --- |
| draft-2 §8.2 假设「\(C = F + B_{\mathrm{full}} = 3t\)」并据此得 DualPipe 气泡 \((p-2)t\) | **第 1、2 轮误判为错，第 3 轮改回**：draft-2 的取值是对的。第 1、2 轮用 \(F\&B=\max\) 作主口径并得 \((p-2)t/2\)，第 3 轮由官方调度图逐格取色（每行 66 单位、空闲 6）与提交语义重放两条证据推翻 | 官方 `dualpipe.png`；`codes/02_schedule_sim.py` 修复后复跑 |
| draft-2 §8.3 「参考博客的 \(\beta=(p-1)/(p-1+6m)\) 不能从官方公式推出」 | **第 2 轮曾推翻，第 3 轮恢复原结论**：在 \(F\&B = F+B_{\mathrm{full}}\) 下 DualPipeV 的 \(\beta\) 是 \((p-1)/(3m+p-1)\)，\(6m\) 版本同样推不出来 | 同上 |
| draft-1 总结表 DualPipe 简化气泡率 \((d-2)/(d-2+3m)\) | **第 3 轮判定为正确**（分子 \((d-2)F\) 与分母 \(3mF\) 同口径）。第 2 轮曾改成 \((p-2)/(6m+p-2)\)，已撤回 | 同上 |
| draft-2 §8.4 CPU 执行器结果：`PyTorch 2.10.0+cpu`、busy 60 / total 66 / 9.09% | **第 1、2 轮误删，第 3 轮部分恢复**：`60 / 66 / 9.09%` 这组数在修正模拟器的释放语义后**可以复现**（环境声明 `+cpu` 仍需改为本机实际的 `2.10.0+cu129`） | `codes/02_schedule_sim.py` 本地复跑 |
| draft-2 §11 张量级 CPU 执行器的误差数字（`6.939e-18`、`8.674e-19`） | **删除**：该脚本不在仓库内也不在可获取路径上，本次未重建 | P0 待办第 1 条 |
| draft-1 把 `7828028c…` 当 Figure 3、`3955441b…` 当 Figure 8 | **修正**为 `e2c1ba8e…`（Figure 3）与 `53647889…`（Figure 8） | `notes/IMAGE-SURVEY-zero-bubble.md` 读图核对 |
| 草稿「MinerU 把 Table 2 并到了 Figure 5 同一图块」 | **推翻**：Figure 5 图块（1256×212）内只有调度网格 | `notes/IMAGE-SURVEY-deepseek-v3-report.md` 读图核对 |
| 草稿把 Megatron Figure 4 写作 `648f402f…` | **修正**为 `2a3ac338…`（`648f402f…` 是 Figure 2） | `notes/IMAGE-SURVEY-megatron-lm-gpu-clusters.md` |
| 草稿把 PipeDream-2BW Figure 2 写作 `e1dfea35…` | **修正**为 `ec0c258b…`（`e1dfea35…` 是 Figure 3） | `notes/IMAGE-SURVEY-pipedream-2bw.md`（另用 PDF 图块坐标复核） |
| draft-2 「`register_custom_function()` 只在 main」 | **修正**：本地 2.10.0 的 `schedules.py` 第 1887 行已有该接口 | 本地读取 |
| draft-1 引用的「官方代码注释」（`# 在这里启用zero bubble (enable_zb=True)…` 等三行中文注释） | **删除**：commit `030ce432…` 的 Python 源码不含任何非 ASCII 字符，那三行注释是草稿自行添加的 | `grep -P '[^\x00-\x7F]' --include=*.py` 在本地仓库返回空 |

---

## 四、结论边界

- 本文所有气泡与气泡率的绝对值都建立在「等时假设 \(F=I=W=t\) + stage 均衡 + 忽略通信」之上，它们是算术结果，不是实测。
- `codes/02_schedule_sim.py` 只算依赖与时间：它按官方提交语义建模发送的释放时刻、把通信耗时记为 0，因此给出的是「配对块内的计算按顺序做完、通信被忽略」这一时间模型下的气泡。它与官方调度图的绘制口径一致，但既不是实测值，也不说明真实 GPU 上重叠掉了多少。
- 文章没有任何 GPU 实测，也没有运行过官方示例或任何框架的 DualPipe 调度；所有性能数字都标注了来源与配置。
