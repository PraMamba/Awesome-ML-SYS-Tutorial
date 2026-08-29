"""
DeviceMesh 多维配置演示

展示不同的 mesh 配置 (4,), (2,2), (1,4) 和子 mesh 分组。

运行方式:
    torchrun --nproc_per_node=4 codes/02_multi_dim_mesh.py

要求: PyTorch >= 2.4, 需要 4 GPU
"""

import os
import torch
import torch.distributed as dist

assert torch.__version__ >= "2.4", f"需要 PyTorch >= 2.4，当前版本: {torch.__version__}"

from torch.distributed.device_mesh import init_device_mesh


def main():
    dist.init_process_group(backend="nccl")

    try:
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)

        if world_size < 4:
            if rank == 0:
                print(f"此示例需要 4 个 GPU，当前只有 {world_size} 个")
            dist.destroy_process_group()
            return

        if rank == 0:
            print("=" * 60)
            print("DeviceMesh 多维配置演示")
            print(f"World Size: {world_size}")
            print("=" * 60)

        # === 配置 1: 1D mesh (4,) ===
        mesh_1d = init_device_mesh("cuda", mesh_shape=(4,))
        if rank == 0:
            print(f"\n--- 配置 1: 1D mesh (4,) ---")
            print(f"  所有 GPU 在同一维度")
            print(f"  用途: 标准 FSDP 或 DDP")
            print(f"  GPU: [0, 1, 2, 3]")

        dist.barrier()

        # === 配置 2: 2D mesh (2, 2) — dp × tp ===
        mesh_dp_tp = init_device_mesh("cuda", (2, 2), mesh_dim_names=("dp", "tp"))
        dp_mesh = mesh_dp_tp["dp"]
        tp_mesh = mesh_dp_tp["tp"]

        if rank == 0:
            print(f"\n--- 配置 2: 2D mesh (2, 2) — dp × tp ---")
            print(f"  适用于 FSDP + TP 2D 并行")
            print(f"  布局:")
            print(f"         tp=0  tp=1")
            print(f"  dp=0 [ GPU0  GPU1 ]")
            print(f"  dp=1 [ GPU2  GPU3 ]")
        print(f"  进程 {rank}: dp_rank={dp_mesh.get_local_rank()}, tp_rank={tp_mesh.get_local_rank()}")

        dist.barrier()

        # === 配置 3: 2D mesh (2, 2) — replicate × shard ===
        mesh_hsdp = init_device_mesh("cuda", (2, 2), mesh_dim_names=("replicate", "shard"))
        rep_mesh = mesh_hsdp["replicate"]
        shard_mesh = mesh_hsdp["shard"]

        if rank == 0:
            print(f"\n--- 配置 3: 2D mesh (2, 2) — replicate × shard ---")
            print(f"  适用于 HSDP（节点间 replicate + 节点内 shard）")
            print(f"  布局:")
            print(f"                shard=0  shard=1")
            print(f"  replicate=0 [  GPU0     GPU1  ]  ← shard 组")
            print(f"  replicate=1 [  GPU2     GPU3  ]  ← shard 组")
        print(f"  进程 {rank}: rep_rank={rep_mesh.get_local_rank()}, shard_rank={shard_mesh.get_local_rank()}")

        dist.barrier()

        # === 配置 4: 2D mesh (1, 4) — 全 shard ===
        if rank == 0:
            print(f"\n--- 配置 4: 2D mesh (1, 4) — 等价于 1D FSDP ---")
            print(f"  replicate=1 意味着无 replicate")
            print(f"  shard=4 意味着所有 GPU 都参与 shard")
            print(f"  等价于 1D mesh (4,)")

        # === 配置 5: 2D mesh (4, 1) — 全 replicate ===
        if rank == 0:
            print(f"\n--- 配置 5: 2D mesh (4, 1) — 等价于 DDP ---")
            print(f"  replicate=4 意味着所有 GPU 都 replicate")
            print(f"  shard=1 意味着无 shard")
            print(f"  等价于标准 DDP")

        dist.barrier()
        if rank == 0:
            print(f"\n{'=' * 60}")
            print(f"总结: DeviceMesh 配置决定了并行策略")
            print(f"  (N,):     1D — 纯 FSDP 或纯 DDP")
            print(f"  (R, S):   HSDP — R 路 replicate × S 路 shard")
            print(f"  (DP, TP): 2D 并行 — DP 路数据并行 × TP 路张量并行")
            print(f"  关键: mesh 的维度名和子 mesh 决定了通信组")
            print(f"{'=' * 60}")

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
