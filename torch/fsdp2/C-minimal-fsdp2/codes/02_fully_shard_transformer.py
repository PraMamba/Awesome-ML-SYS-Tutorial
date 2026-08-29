"""
Toy Transformer + fully_shard

使用 PyTorch 原生 TransformerEncoderLayer，每层 fully_shard → root fully_shard → 训练

运行方式:
    torchrun --nproc_per_node=2 codes/02_fully_shard_transformer.py

要求: PyTorch >= 2.4
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist

assert torch.__version__ >= "2.4", f"FSDP2 需要 PyTorch >= 2.4，当前版本: {torch.__version__}"

from torch.distributed.fsdp import fully_shard


class ToyTransformer(nn.Module):
    """简单的 Transformer Encoder 模型"""

    def __init__(self, d_model=256, nhead=4, num_layers=2, dim_feedforward=512):
        super().__init__()
        self.embedding = nn.Linear(d_model, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                batch_first=True,
                dropout=0.0,  # 关闭 dropout，方便验证
            )
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(d_model, d_model)

    def forward(self, x):
        x = self.embedding(x)
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

        if rank == 0:
            print("=" * 60)
            print("FSDP2: Toy Transformer + fully_shard")
            print(f"World Size: {world_size}")
            print("=" * 60)

        # 创建模型
        model = ToyTransformer(d_model=256, nhead=4, num_layers=2).cuda()

        if rank == 0:
            total_params = sum(p.numel() for p in model.parameters())
            print(f"模型参数总量: {total_params:,}")

        # 自底向上 fully_shard
        # 每个 TransformerEncoderLayer 是一个 FSDP unit
        for i, layer in enumerate(model.layers):
            fully_shard(layer)
            if rank == 0:
                print(f"  fully_shard(layer[{i}]) 完成")

        fully_shard(model)  # root
        if rank == 0:
            print("  fully_shard(model) 完成 (root)")

        # optimizer 必须在 fully_shard 之后创建
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # 训练
        if rank == 0:
            print("\n--- 开始训练 ---")

        for step in range(5):
            # [batch, seq_len, d_model]
            inputs = torch.randn(2, 16, 256, device="cuda")
            targets = torch.randn(2, 16, 256, device="cuda")

            outputs = model(inputs)
            loss = nn.functional.mse_loss(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if rank == 0:
                print(f"  Step {step}: loss = {loss.item():.6f}")

        if rank == 0:
            print("\n训练成功完成!")

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
