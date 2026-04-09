"""
DTensor 基础：手动创建 DTensor + Placement 操作

演示:
  - 从本地张量创建 DTensor (Shard / Replicate)
  - redistribute: 在不同 placement 间转换

运行方式:
    torchrun --nproc_per_node=2 codes/01_dtensor_basics.py

要求: PyTorch >= 2.4
"""

import os
import torch
import torch.distributed as dist

assert torch.__version__ >= "2.4", f"DTensor 需要 PyTorch >= 2.4，当前版本: {torch.__version__}"

from torch.distributed.tensor import DTensor, Shard, Replicate
from torch.distributed.device_mesh import init_device_mesh


def main():
    dist.init_process_group(backend="nccl")

    try:
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)

        # 创建 1D DeviceMesh
        mesh = init_device_mesh("cuda", mesh_shape=(world_size,))

        if rank == 0:
            print("=" * 60)
            print("DTensor 基础操作")
            print(f"DeviceMesh: {mesh}")
            print("=" * 60)

        # === 1. 从完整张量创建 Shard DTensor ===
        if rank == 0:
            print("\n--- 1. Shard(0): 沿第 0 维分片 ---")

        full_tensor = torch.arange(8, dtype=torch.float32, device="cuda").reshape(4, 2)
        # DTensor.from_local: 每个 rank 提供自己的本地分片
        local_shard = full_tensor.chunk(world_size, dim=0)[rank].contiguous()
        dt_shard = DTensor.from_local(local_shard, mesh, placements=[Shard(0)])

        print(f"  进程 {rank}: 本地分片 = {local_shard.tolist()}")
        print(f"  进程 {rank}: DTensor full_shape = {list(dt_shard.shape)}")
        print(f"  进程 {rank}: placements = {dt_shard.placements}")

        # === 2. 从完整张量创建 Replicate DTensor ===
        dist.barrier()
        if rank == 0:
            print("\n--- 2. Replicate: 每个 rank 持有完整副本 ---")

        local_full = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda")
        dt_replicate = DTensor.from_local(local_full, mesh, placements=[Replicate()])

        print(f"  进程 {rank}: 本地数据 = {local_full.tolist()}")
        print(f"  进程 {rank}: DTensor full_shape = {list(dt_replicate.shape)}")
        print(f"  进程 {rank}: placements = {dt_replicate.placements}")

        # === 3. redistribute: Shard → Replicate ===
        dist.barrier()
        if rank == 0:
            print("\n--- 3. redistribute: Shard(0) → Replicate (触发 AllGather) ---")

        dt_gathered = dt_shard.redistribute(mesh, placements=[Replicate()])
        print(f"  进程 {rank}: Shard → Replicate 后的完整数据 = {dt_gathered.to_local().tolist()}")

        # === 4. redistribute: Replicate → Shard ===
        dist.barrier()
        if rank == 0:
            print("\n--- 4. redistribute: Replicate → Shard(0) (本地 slice) ---")

        dt_resharded = dt_gathered.redistribute(mesh, placements=[Shard(0)])
        print(f"  进程 {rank}: Replicate → Shard 后的本地分片 = {dt_resharded.to_local().tolist()}")

        dist.barrier()
        if rank == 0:
            print("\n" + "=" * 60)
            print("总结:")
            print("  Shard(dim):  张量沿 dim 维分片到各设备")
            print("  Replicate(): 每个设备持有完整副本")
            print("  redistribute: 在不同 placement 间转换")
            print("    Shard → Replicate = AllGather")
            print("    Replicate → Shard = 本地 slice")
            print("=" * 60)

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
