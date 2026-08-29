# ParallelDims Lab Notes

## Source of truth

The reference file was read from the read-only torchtitan worktree:

```text
/workspace/algorithm/torchtitan/.worktrees/source_code_analysis/
torchtitan/distributed/parallel_dims.py
```

The verified source commit is:

```text
d6555c4c35a10bebcce652b58374cdeb2ecbe527
```

The reference worktree had unrelated documentation edits, but the core
`parallel_dims.py` file was read at the commit above. This lab does not import
torchtitan and does not construct a real `DeviceMesh`.

## Documentation versus current source

The scripts follow the source when the study documents and source disagree:

| Topic | Current source | Lab decision |
| --- | --- | --- |
| `ParallelDims` fields | `dp_replicate`, `dp_shard`, `cp`, `tp`, `pp`, `ep`, `world_size`, and `spmd_backend` | `ParallelDimsLite` exposes these fields only. |
| `etp` | No `etp` field exists in the verified source. | No `etp` field or sparse axis is implemented. |
| Sparse mesh | `(pp, dp_replicate, efsdp, ep)` | `07` and `10` use this four-axis shape. |
| `efsdp` | `dp_shard * cp * tp // ep` | The formula is shared by the validation and inspection code. |
| Sequence divisor | `tp * (cp * 2)` | `05`, `06`, and `10` use this exact formula. |
| Gradient divisor | No `fsdp_gradient_divide_factor` property exists in this source. | The property is intentionally absent. |
| Axis labels in the `02` document | For row-major `(2, 4)`, rank 6 is `(1, 2)`; axis 0 is `[4, 5, 6, 7]` and axis 1 is `[2, 6]`. | `02_mesh_coordinates.py` prints the source-aligned labels. |
| Loss view product | The source flattens only `(batch, cp)` into the loss submesh. | `loss` has product `batch * cp` (8 in the fixed example), while full-world views have product 32. |
| Real mesh construction | `build_mesh()` creates DeviceMesh views and process groups. | `07` and `10` print the corresponding shapes only. |

The historical `etp` and gradient-divisor descriptions remain useful as
reading contrasts, but they are not treated as the current API.

## Run the lab

From this directory:

```bash
cd torch/parallel_dims_lab
python -m pytest tests/ -q
```

All scripts except the real collective launcher are ordinary CPU programs:

```bash
python 02_mesh_coordinates.py
python 03_dp_fsdp_sim.py
python 04_tp_mlp_sim.py
python 05_cp_pp_planner.py
python 06_parallel_dims_lite.py
python 07_mesh_inspector.py
python 08_moe_router_sim.py
python 09_gradient_accounting.py
python 10_parallel_config_advisor.py
```

`01_collectives.py` has a single-process explanation mode when run directly.
The required CPU/Gloo four-rank check is:

```bash
torchrun --standalone --nproc-per-node=4 01_collectives.py
```

The 10th script accepts the configuration flags from the study plan. For
example:

```bash
python 10_parallel_config_advisor.py \
  --world-size 32 --gpus-per-node 8 \
  --dp-replicate 2 --dp-shard 2 --cp 2 --tp 2 --pp 2 --ep 4 \
  --seq-len 8192 --num-layers 32 --num-heads 32 --num-experts 8 \
  --spmd-backend spmd_types --rank 0
```

`ERROR` findings make the advisor exit with status 1. `WARN`, `CHECK`, and
`INFO` findings are advisory; `VALID` marks a successful hard check.

## Scope boundary

Only CPU tensors are used. `01` uses the Gloo backend when launched with four
processes; the other exercises simulate collectives and routing with local
lists or CPU tensors. No script claims to measure real communication or
hardware performance.
