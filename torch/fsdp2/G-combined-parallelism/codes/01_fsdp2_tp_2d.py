"""
FSDP2 + TP 2D 并行

使用 2D DeviceMesh 实现:
  - TP (Tensor Parallelism): 将层内参数拆分到 tp_mesh
  - FSDP: 在 dp_mesh 上做数据并行分片

运行方式:
    torchrun --nproc_per_node=4 codes/01_fsdp2_tp_2d.py

要求: PyTorch >= 2.4, 需要 4 GPU
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist

assert torch.__version__ >= "2.4", f"需要 PyTorch >= 2.4，当前版本: {torch.__version__}"

from torch.distributed.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)


class TransformerBlock(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        # 用两个 Linear 模拟 MLP (不用 nn.Sequential，方便 TP 包裹)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return self.norm(x + residual)


class SimpleModel(nn.Module):
    def __init__(self, d_model=256, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(d_model) for _ in range(num_layers)])
        self.head = nn.Linear(d_model, d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


def main():
    dist.init_process_group(backend="nccl")

    try:
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)

        if world_size < 4:
            if rank == 0:
                print(f"2D 并行示例需要 4 个 GPU，当前只有 {world_size} 个")
            dist.destroy_process_group()
            return

        if rank == 0:
            print("=" * 60)
            print("FSDP2 + TP: 2D 并行")
            print("=" * 60)

        # 创建 2D mesh: (dp=2, tp=2)
        # dp 维度: FSDP 数据并行
        # tp 维度: Tensor Parallelism
        mesh_2d = init_device_mesh(
            "cuda",
            mesh_shape=(2, 2),
            mesh_dim_names=("dp", "tp"),
        )

        if rank == 0:
            print(f"\n2D DeviceMesh: (dp=2, tp=2)")
            print(f"  GPU 0,1 → TP 组 0 (层内参数拆分)")
            print(f"  GPU 2,3 → TP 组 1 (层内参数拆分)")
            print(f"  GPU 0,2 → DP 组 0 (FSDP 数据并行)")
            print(f"  GPU 1,3 → DP 组 1 (FSDP 数据并行)")

        # 创建模型
        torch.manual_seed(42)
        model = SimpleModel(d_model=256, num_layers=4).cuda()

        if rank == 0:
            total_params = sum(p.numel() for p in model.parameters())
            print(f"\n模型参数总量: {total_params:,}")

        # 第一步: 对每层应用 TP（在 tp mesh 上拆分层内参数）
        tp_mesh = mesh_2d["tp"]
        for layer in model.layers:
            parallelize_module(
                layer,
                tp_mesh,
                {
                    "linear1": ColwiseParallel(),  # 第一个 Linear 按列拆分
                    "linear2": RowwiseParallel(),  # 第二个 Linear 按行拆分
                },
            )

        # 第二步: 对每层应用 FSDP（在 dp mesh 上做数据并行分片）
        dp_mesh = mesh_2d["dp"]
        for layer in model.layers:
            fully_shard(layer, mesh=dp_mesh)
        fully_shard(model, mesh=dp_mesh)

        if rank == 0:
            print("TP + FSDP 包裹完成")
            print("  每层: ColwiseParallel(linear1) + RowwiseParallel(linear2)")
            print("  数据并行: fully_shard on dp_mesh")

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # 训练
        if rank == 0:
            print("\n--- 开始训练 ---")

        for step in range(5):
            inputs = torch.randn(4, 16, 256, device="cuda")
            targets = torch.randn(4, 16, 256, device="cuda")

            outputs = model(inputs)
            loss = nn.functional.mse_loss(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if rank == 0:
                print(f"  Step {step}: loss = {loss.item():.6f}")

        if rank == 0:
            print("\n2D 并行训练成功!")
            print()
            print("2D 并行的优势:")
            print("  TP: 解决单层参数太大的问题（层内拆分）")
            print("  FSDP: 解决总参数太多的问题（跨层分片 + 数据并行）")
            print("  组合后: TP 减少每个 FSDP unit 的参数量，FSDP 减少数据并行的显存占用")

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
