"""
reshard_after_forward 权衡实验

reshard_after_forward=True  (默认): forward 后释放完整参数 → 省显存，但 backward 时需再次 AllGather
reshard_after_forward=False:         forward 后保留完整参数 → 费显存，但 backward 时无需重新 AllGather

运行方式:
    torchrun --nproc_per_node=2 codes/01_reshard_after_forward.py

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


class SimpleTransformer(nn.Module):
    def __init__(self, num_layers=8):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock() for _ in range(num_layers)])
        self.head = nn.Linear(512, 512)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


def run_experiment(rank, reshard_after_forward, num_steps=5):
    model = SimpleTransformer(num_layers=8).cuda()

    for layer in model.layers:
        fully_shard(layer, reshard_after_forward=reshard_after_forward)
    fully_shard(model, reshard_after_forward=reshard_after_forward)

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
            print("reshard_after_forward 对比实验")
            print("=" * 60)
            print("模型: 8 层 Transformer (d_model=512, ff=2048)")
            print()

        for reshard in [True, False]:
            torch.cuda.empty_cache()
            peak_mem, avg_time = run_experiment(rank, reshard)

            if rank == 0:
                label = "True (省显存)" if reshard else "False(省通信)"
                print(f"  reshard_after_forward={label}: Peak Memory = {peak_mem:8.1f} MB | Avg Step = {avg_time*1000:.1f} ms")

        if rank == 0:
            print()
            print("分析:")
            print("  True  → forward 后释放参数 → peak memory 低 → backward 需重新 AllGather → step time 略长")
            print("  False → forward 后保留参数 → peak memory 高 → backward 直接用 → step time 略短")
            print("  推荐: 大模型用 True（默认），小模型/显存充裕用 False")

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
