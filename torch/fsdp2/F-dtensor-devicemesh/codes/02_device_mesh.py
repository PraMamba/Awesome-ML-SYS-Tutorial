"""
1D 和 2D DeviceMesh 创建 + 子 mesh 提取

DeviceMesh 是设备的逻辑组织:
  - 1D mesh: 标准 FSDP / DDP
  - 2D mesh: HSDP (节点内 shard + 节点间 replicate) 或 FSDP+TP

运行方式:
    torchrun --nproc_per_node=4 codes/02_device_mesh.py

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
            print(f"DeviceMesh 演示 (world_size={world_size})")
            print("=" * 60)

        # === 1. 1D DeviceMesh ===
        mesh_1d = init_device_mesh("cuda", mesh_shape=(world_size,))
        if rank == 0:
            print(f"\n--- 1D DeviceMesh ---")
            print(f"  mesh_shape = ({world_size},)")
            print(f"  用途: 标准 FSDP / DDP")
            print(f"  所有 {world_size} 个 GPU 在同一个分片组")

        dist.barrier()

        # === 2. 2D DeviceMesh (2x2) ===
        mesh_2d = init_device_mesh(
            "cuda",
            mesh_shape=(2, 2),
            mesh_dim_names=("replicate", "shard"),
        )

        if rank == 0:
            print(f"\n--- 2D DeviceMesh (2x2) ---")
            print(f"  mesh_shape = (2, 2)")
            print(f"  dim_names  = ('replicate', 'shard')")
            print(f"  用途: HSDP（节点内 shard，节点间 replicate）")

        # 提取子 mesh
        shard_mesh = mesh_2d["shard"]
        replicate_mesh = mesh_2d["replicate"]

        print(f"  进程 {rank}: shard 子 mesh 中的 local rank = {shard_mesh.get_local_rank()}")
        print(f"  进程 {rank}: replicate 子 mesh 中的 local rank = {replicate_mesh.get_local_rank()}")

        dist.barrier()

        if rank == 0:
            print(f"\n  设备分组示意 (2x2 mesh):")
            print(f"            shard_dim=1")
            print(f"          GPU0  GPU1")
            print(f"  rep=0 [  ■     ■  ]  ← shard 组 0 (GPU0, GPU1 互相 shard)")
            print(f"  rep=1 [  ■     ■  ]  ← shard 组 1 (GPU2, GPU3 互相 shard)")
            print(f"          ↑     ↑")
            print(f"     replicate  replicate")
            print(f"      组 0      组 1")
            print(f"  (GPU0,2)   (GPU1,3)")
            print(f"   互相        互相")
            print(f"  replicate  replicate")

        # === 3. 不同 mesh 配置 ===
        dist.barrier()
        if rank == 0:
            print(f"\n--- 不同 2D mesh 配置对比 ---")
            print(f"  (1, 4): 全 shard，无 replicate = 标准 FSDP")
            print(f"  (2, 2): HSDP，2 路 replicate × 2 路 shard")
            print(f"  (4, 1): 全 replicate，无 shard = 标准 DDP")

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
