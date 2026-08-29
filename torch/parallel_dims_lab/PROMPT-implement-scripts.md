# PROMPT：实现 parallel_dims_lab 学习脚本

## 目标

在 `torch/parallel_dims_lab/` 下实现下列 10 个学习脚本 + `tests/` + `notes/`，全部为无 GPU 即可运行的纯 Python 模拟（01 除外，可跑 CPU+Gloo）：

```
parallel_dims_lab/
├── 01_collectives.py            # 单元1：torchrun 4 进程 Gloo All-Reduce Sum 验证=6；子 ProcessGroup [0,2] 演示
├── 02_mesh_coordinates.py       # 单元2：rank_to_coord / coord_to_rank / axis_group，支持 (2,4) 与任意形状
├── 03_dp_fsdp_sim.py            # 单元3：参数分片/All-Gather/Reduce-Scatter 模拟；dpr=2,dps=4,local_batch=3 → dp=8, global_batch=24
├── 04_tp_mlp_sim.py             # 单元4：tp=2 列/行切分 MLP，assert_close 验证与密集结果一致
├── 05_cp_pp_planner.py          # 单元5：CP Token 划分、seq_len_divisor、PP stage 划分、microbatch 流水表
├── 06_parallel_dims_lite.py     # 单元6：ParallelDimsLite dataclass + validate()（含 dp_shard=-1 推导、EP 整除）
├── 07_mesh_inspector.py         # 单元7：固定 32-GPU 配置输出 batch/loss/fsdp/efsdp 与全部 mesh 形状并验证乘积
├── 08_moe_router_sim.py         # 单元8：16 token / 8 expert / ep=4 路由分桶→计算→还原→负载统计
├── 09_gradient_accounting.py    # 单元9：等量/不等量 Token 的全局均值 vs 均值之均值（实验 B 应得 4 vs 3）
├── 10_parallel_config_advisor.py # 单元10+最终项目：CLI 配置审计器，ERROR/WARN/CHECK/INFO 分级
├── tests/                       # pytest：test_validation.py / test_mesh_shapes.py / test_router.py，覆盖用例 A–F
└── notes/README.md              # 记录源码 commit、文档与源码差异清单、运行说明
```

## 输入材料

1. **需求规格**（每个脚本的"实践练习"小节即需求）：`torch/parallel_dims_lab/03-parallel-dims-20-hour-study-plan.md`
2. **基准源码**（唯一事实来源，冲突时以此为准）：`/workspace/algorithm/torchtitan/.worktrees/source_code_analysis/`
   核心文件：`torchtitan/distributed/parallel_dims.py`（当前 commit `d6555c4c3`，实现前自行 `git rev-parse HEAD` 复核并记入 notes）

## 铁律

- **文档可能不准确，一切以源码为准**：凡文档描述与源码冲突，按源码实现，并在 `notes/README.md` 记录差异，不得照抄错误。
- 已知易错点（已对照源码核实）：
  - 当前 `ParallelDims` 字段：`dp_replicate, dp_shard, cp, tp, pp, ep, world_size, spmd_backend`（默认 `"spmd_types"`）。**没有 `etp` 字段，也没有 `fsdp_gradient_divide_factor` 属性**（属历史版本，勿实现为当前 API）。
  - `sparse_mesh = (pp, dp_replicate, efsdp, ep)`，**无 etp 轴**；`efsdp = dp_shard * cp * tp // ep`。
  - `seq_len_divisor = tp * (cp * 2)`。
  - 02 文档示例中 axis 标签写反：形状 `(2,4)` 行主序下 rank 6 坐标为 `(1,2)`，axis 0（行）组 `[4,5,6,7]`，axis 1（列）组 `[2,6]`。
- 不要修改现有 `01/02/03-*.md` 文档；不做真实分布式通信（01 用 Gloo 真跑，其余纯模拟）。

## 关键公式（直接采用，已对照源码）

```
batch = dpr * dps;  loss = dpr * dps * cp;  fsdp = dps * cp;  efsdp = dps * cp * tp // ep
dense 校验:  dpr * dps * cp * tp * pp == world_size   （dp_shard == -1 时自动推导 dp_shard = world_size // (dpr*cp*tp*pp)）
EP 校验:     (dps * cp * tp) % ep == 0
mesh 形状:   dataloading (pp, batch, cp, tp)
             loss = (batch, cp) 展平
             spmd_types:  storage (pp, dpr, dps, cp, tp)；fwd/bwd (pp, dp, cp, tp)，dp = batch
             partial_dtensor:  dense (pp, dpr, fsdp, tp)
             sparse (pp, dpr, efsdp, ep)
固定示例:    world=32, dpr=2, dps=2, cp=2, tp=2, pp=2, ep=4 → batch=4, loss=8, fsdp=4, efsdp=2
```

## 验收标准

1. `cd torch/parallel_dims_lab && python -m pytest tests/ -q` 全绿（用例 A 合法 / B 乘积错 / C EP 非法 / D dp_shard=-1 / E seq_len 非法 / F singleton 轴）。
2. 每个 `0X_*.py` 可独立 `python 0X_*.py` 运行，输出与文档示例一致（已按源码修正处除外）。
3. `notes/README.md` 含：源码 commit SHA、文档↔源码差异清单、各脚本运行方法。
4. 全部实现不使用 GPU；torch 仅用于 CPU 张量与 01 的 Gloo 通信（环境已装 torch 2.10.0）。
