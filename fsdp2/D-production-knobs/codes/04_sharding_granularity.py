"""
Sharding 粒度实验

8 层 Transformer，对比 coarse（整个模型一个 unit）vs fine（每层一个 unit）

运行方式:
    torchrun --nproc_per_node=2 codes/04_sharding_granularity.py

要求: PyTorch >= 2.4
"""

import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist

assert torch.__version__ >= "2.4", f"FSDP2 需要 PyTorch >= 2.4，当前版本: {torch.__version__}"

from torch.distributed.fsdp import fully_shard


class TransformerBlock(nn.Module):
    def __init__(self, d_model=512, nhead=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=0.0)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 2048), nn.ReLU(), nn.Linear(2048, d_model))
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(x, x, x, need_weights=False)[0]
        x = self.norm1(x)
        x = x + self.ff(x)
        x = self.norm2(x)
        return x


class Transformer(nn.Module):
    def __init__(self, num_layers=8):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock() for _ in range(num_layers)])
        self.head = nn.Linear(512, 512)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


def run_experiment(rank, granularity, num_steps=5):
    model = Transformer(num_layers=8).cuda()

    if granularity == "coarse":
        fully_shard(model)
    elif granularity == "per_layer":
        for layer in model.layers:
            fully_shard(layer)
        fully_shard(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    times = []
    for step in range(num_steps):
        inputs = torch.randn(4, 32, 512, device="cuda")
        targets = torch.randn(4, 32, 512, device="cuda")

        start = time.time()
        outputs = model(inputs)
        loss = nn.functional.mse_loss(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        times.append(time.time() - start)

    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    avg_time = sum(times[1:]) / len(times[1:])
    return peak_mem, avg_time


def main():
    dist.init_process_group(backend="nccl")

    try:
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

        if rank == 0:
            print("=" * 60)
            print("Sharding 粒度实验: coarse vs per_layer")
            print("=" * 60)
            print("模型: 8 层 Transformer (d_model=512, ff=2048)")
            print()

        for granularity in ["coarse", "per_layer"]:
            torch.cuda.empty_cache()
            peak_mem, avg_time = run_experiment(rank, granularity)

            if rank == 0:
                print(f"  {granularity:12s}: Peak Memory = {peak_mem:8.1f} MB | Avg Step = {avg_time*1000:.1f} ms")

        if rank == 0:
            fsdp_unit_coarse = 1
            fsdp_unit_per_layer = 8 + 1  # 8 layers + root
            print()
            print(f"  coarse:    {fsdp_unit_coarse} 个 FSDP unit → forward 时所有参数同时 AllGather")
            print(f"  per_layer: {fsdp_unit_per_layer} 个 FSDP unit → 逐层 AllGather + 释放")
            print()
            print("  per_layer 是生产中最常用的粒度:")
            print("  - 足够细以控制 peak memory")
            print("  - 不过细以避免通信 overhead")

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
