"""
最小 FSDP2 实践：Toy MLP + fully_shard 跑通训练

演示 FSDP2 的核心 API: fully_shard（不是 wrapper，而是 in-place 注册 hooks）

运行方式:
    torchrun --nproc_per_node=2 codes/01_fully_shard_mlp.py

要求: PyTorch >= 2.4
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist

assert torch.__version__ >= "2.4", f"FSDP2 需要 PyTorch >= 2.4，当前版本: {torch.__version__}"

from torch.distributed.fsdp import fully_shard


class ToyMLP(nn.Module):
    """三层 MLP，用于演示 fully_shard"""

    def __init__(self, hidden_size=256):
        super().__init__()
        self.layer1 = nn.Linear(hidden_size, hidden_size * 2)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        return self.layer3(x)


def main():
    dist.init_process_group(backend="nccl")

    try:
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)

        if rank == 0:
            print("=" * 60)
            print("FSDP2 最小示例：Toy MLP + fully_shard")
            print(f"World Size: {world_size}")
            print("=" * 60)

        # 创建模型
        model = ToyMLP(hidden_size=256).cuda()

        if rank == 0:
            total_params = sum(p.numel() for p in model.parameters())
            print(f"\n模型参数总量: {total_params:,}")

        # 自底向上 fully_shard（FSDP2 的标准用法）
        # 注意: fully_shard 是 in-place 操作，不是 wrapper
        fully_shard(model.layer1)
        fully_shard(model.layer2)
        fully_shard(model.layer3)
        fully_shard(model)  # root 最后

        if rank == 0:
            print("fully_shard 完成（自底向上：layer1 → layer2 → layer3 → root）")

        # optimizer 必须在 fully_shard 之后创建
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # 训练 5 步
        if rank == 0:
            print("\n--- 开始训练 ---")

        for step in range(5):
            # 随机输入
            inputs = torch.randn(4, 256, device="cuda")
            targets = torch.randn(4, 256, device="cuda")

            # 前向 + 反向 + 更新
            outputs = model(inputs)
            loss = nn.functional.mse_loss(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if rank == 0:
                print(f"  Step {step}: loss = {loss.item():.6f}")

        if rank == 0:
            print("\nFSDP2 训练成功完成!")
            print("=" * 60)

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
