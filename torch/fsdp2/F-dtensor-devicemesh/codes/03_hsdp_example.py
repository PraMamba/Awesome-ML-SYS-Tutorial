"""
HSDP 示例: 节点内 shard，节点间 replicate

使用 2D DeviceMesh + fully_shard 实现 HSDP:
  - shard 维度: 同组内的 GPU 互相分片参数
  - replicate 维度: 不同组之间 AllReduce 梯度

运行方式:
    torchrun --nproc_per_node=4 codes/03_hsdp_example.py

要求: PyTorch >= 2.4, 需要 4 GPU
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist

assert torch.__version__ >= "2.4", f"FSDP2 需要 PyTorch >= 2.4，当前版本: {torch.__version__}"

from torch.distributed.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh


class SimpleMLP(nn.Module):
    def __init__(self, hidden=512):
        super().__init__()
        self.layer1 = nn.Linear(hidden, hidden * 2)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden * 2, hidden)

    def forward(self, x):
        return self.layer2(self.relu(self.layer1(x)))


def main():
    dist.init_process_group(backend="nccl")

    try:
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)

        if world_size < 4:
            if rank == 0:
                print(f"HSDP 示例需要 4 个 GPU，当前只有 {world_size} 个")
            dist.destroy_process_group()
            return

        if rank == 0:
            print("=" * 60)
            print("HSDP: 节点内 Shard + 节点间 Replicate")
            print("=" * 60)

        # 创建 2D mesh: (replicate=2, shard=2)
        # 模拟 2 个节点，每节点 2 GPU
        mesh_2d = init_device_mesh(
            "cuda",
            mesh_shape=(2, 2),
            mesh_dim_names=("replicate", "shard"),
        )

        if rank == 0:
            print(f"\n2D DeviceMesh: (replicate=2, shard=2)")
            print(f"  GPU 0,1 → shard 组 0（节点内互相 shard）")
            print(f"  GPU 2,3 → shard 组 1（节点内互相 shard）")
            print(f"  GPU 0,2 → replicate 组 0（节点间互相同步）")
            print(f"  GPU 1,3 → replicate 组 1（节点间互相同步）")

        # 创建模型
        torch.manual_seed(42)
        model = SimpleMLP(hidden=512).cuda()

        if rank == 0:
            total_params = sum(p.numel() for p in model.parameters())
            print(f"\n模型参数总量: {total_params:,}")

        # HSDP: 将 2D mesh 传给 fully_shard
        fully_shard(model.layer1, mesh=mesh_2d)
        fully_shard(model.layer2, mesh=mesh_2d)
        fully_shard(model, mesh=mesh_2d)

        if rank == 0:
            print("fully_shard 完成 (HSDP 模式)")

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # 训练
        if rank == 0:
            print("\n--- 开始训练 ---")

        for step in range(5):
            torch.manual_seed(step * world_size + rank)
            inputs = torch.randn(4, 512, device="cuda")
            targets = torch.randn(4, 512, device="cuda")

            outputs = model(inputs)
            loss = nn.functional.mse_loss(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if rank == 0:
                print(f"  Step {step}: loss = {loss.item():.6f}")

        if rank == 0:
            print("\nHSDP 训练成功!")
            print()
            print("HSDP vs FSDP vs DDP:")
            print("  DDP:  全 replicate，每 GPU 持有完整参数")
            print("  FSDP: 全 shard，参数分片到所有 GPU")
            print("  HSDP: 节点内 shard（减少显存）+ 节点间 replicate（减少跨节点通信）")
            print("  HSDP 适合多节点场景：利用节点内高带宽做 shard，避免跨节点大量通信")

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
