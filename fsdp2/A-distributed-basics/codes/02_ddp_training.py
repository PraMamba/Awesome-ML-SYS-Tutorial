"""
DDP 训练完整 loop：model → DDP wrap → train 10 steps → print loss

运行方式:
    torchrun --nproc_per_node=2 codes/02_ddp_training.py
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def main():
    dist.init_process_group(backend="nccl")

    try:
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)

        # 创建一个简单的 MLP 模型
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        ).cuda()

        # 用 DDP 包裹模型
        model = DDP(model, device_ids=[local_rank])

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()

        if rank == 0:
            print("=" * 50)
            print("DDP 训练开始")
            print(f"World Size: {world_size}")
            print("=" * 50)

        # 训练 10 步
        for step in range(10):
            # 每个 rank 生成不同的随机数据（模拟不同数据分片）
            torch.manual_seed(step * world_size + rank)
            inputs = torch.randn(8, 64, device="cuda")
            labels = torch.randint(0, 10, (8,), device="cuda")

            # 前向 + 反向 + 更新
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()  # DDP 在此步自动 AllReduce 梯度
            optimizer.step()

            if rank == 0:
                print(f"  Step {step:2d} | Loss = {loss.item():.4f}")

        if rank == 0:
            print("=" * 50)
            print("DDP 训练完成")

        # 验证：所有 rank 的模型参数应该一致
        for name, param in model.named_parameters():
            tensor = param.data.flatten()[:1].clone()
            gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
            dist.all_gather(gathered, tensor)
            if rank == 0:
                all_equal = all(torch.equal(gathered[0], g) for g in gathered)
                print(f"  参数 {name}: 所有 rank 一致 = {all_equal}")

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
