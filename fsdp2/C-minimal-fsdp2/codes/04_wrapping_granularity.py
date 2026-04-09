"""
不同包裹粒度对比：coarse / medium / fine

同一个模型用三种 fully_shard 粒度，对比 peak memory 和 step time。

粒度越细 → 通信桶越小 → peak memory 越低，但 overhead 越大。

运行方式:
    torchrun --nproc_per_node=2 codes/04_wrapping_granularity.py

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
    def __init__(self, d_model=512, nhead=8, dim_ff=1024):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=0.0)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(x, x, x, need_weights=False)[0]
        x = self.norm1(x)
        x = x + self.ff(x)
        x = self.norm2(x)
        return x


class MultiLayerTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=6, dim_ff=1024):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_ff) for _ in range(num_layers)
        ])
        self.head = nn.Linear(d_model, d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


def measure_training(model, optimizer, num_steps=5):
    """测量训练的 peak memory 和平均 step time"""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    times = []
    for step in range(num_steps):
        inputs = torch.randn(2, 32, 512, device="cuda")
        targets = torch.randn(2, 32, 512, device="cuda")

        start = time.time()
        outputs = model(inputs)
        loss = nn.functional.mse_loss(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        times.append(time.time() - start)

    peak_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB
    avg_time = sum(times[1:]) / len(times[1:])  # 跳过第一步（warmup）
    return peak_mem, avg_time


def run_experiment(rank, granularity):
    """运行指定粒度的实验"""
    model = MultiLayerTransformer(d_model=512, nhead=8, num_layers=6).cuda()

    if granularity == "coarse":
        # 粗粒度: 只对 root 做 fully_shard（整个模型是一个 FSDP unit）
        fully_shard(model)
    elif granularity == "medium":
        # 中粒度: 每个 TransformerBlock 一个 FSDP unit
        for layer in model.layers:
            fully_shard(layer)
        fully_shard(model)
    elif granularity == "fine":
        # 细粒度: 每个子模块一个 FSDP unit
        for layer in model.layers:
            fully_shard(layer.attn)
            fully_shard(layer.ff)
            fully_shard(layer)
        fully_shard(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    peak_mem, avg_time = measure_training(model, optimizer, num_steps=5)
    return peak_mem, avg_time


def main():
    dist.init_process_group(backend="nccl")

    try:
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

        if rank == 0:
            print("=" * 60)
            print("fully_shard 包裹粒度对比实验")
            print("=" * 60)
            print(f"模型: 6 层 Transformer (d_model=512, ff=1024)")
            print()

        results = {}
        for granularity in ["coarse", "medium", "fine"]:
            torch.cuda.empty_cache()
            peak_mem, avg_time = run_experiment(rank, granularity)
            results[granularity] = (peak_mem, avg_time)

            if rank == 0:
                print(f"  {granularity:8s}: Peak Memory = {peak_mem:8.1f} MB | Avg Step Time = {avg_time*1000:.1f} ms")

        if rank == 0:
            print()
            print("分析:")
            print("  - coarse: 整个模型是一个 FSDP unit，AllGather 时恢复全部参数，peak memory 最高")
            print("  - medium: 每层一个 FSDP unit（推荐），逐层 AllGather/释放，memory 适中")
            print("  - fine:   每个子模块一个 FSDP unit，memory 最低但通信 overhead 最大")

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
