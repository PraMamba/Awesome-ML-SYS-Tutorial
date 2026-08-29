"""
CPUOffloadPolicy 示例

将参数和梯度 offload 到 CPU，节省 GPU 显存，但增加 PCIe 传输开销。

运行方式:
    torchrun --nproc_per_node=2 codes/03_cpu_offload.py

要求: PyTorch >= 2.4
"""

import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist

assert torch.__version__ >= "2.4", f"FSDP2 需要 PyTorch >= 2.4，当前版本: {torch.__version__}"

from torch.distributed.fsdp import fully_shard, CPUOffloadPolicy


class LargeMLP(nn.Module):
    def __init__(self, hidden=2048, num_layers=4):
        super().__init__()
        layers = []
        for _ in range(num_layers):
            layers.extend([nn.Linear(hidden, hidden), nn.ReLU()])
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, hidden)

    def forward(self, x):
        return self.head(self.net(x))


def run_experiment(rank, use_offload, num_steps=3):
    model = LargeMLP(hidden=2048, num_layers=4).cuda()

    if use_offload:
        offload_policy = CPUOffloadPolicy(pin_memory=True)
        for i, module in enumerate(model.net):
            if isinstance(module, nn.Linear):
                fully_shard(module, offload_policy=offload_policy)
        fully_shard(model, offload_policy=offload_policy)
    else:
        for module in model.net:
            if isinstance(module, nn.Linear):
                fully_shard(module)
        fully_shard(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    times = []
    for step in range(num_steps):
        inputs = torch.randn(4, 2048, device="cuda")
        targets = torch.randn(4, 2048, device="cuda")

        start = time.time()
        outputs = model(inputs)
        loss = nn.functional.mse_loss(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        times.append(time.time() - start)

    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    avg_time = sum(times[1:]) / len(times[1:]) if len(times) > 1 else times[0]
    return peak_mem, avg_time


def main():
    dist.init_process_group(backend="nccl")

    try:
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

        if rank == 0:
            print("=" * 60)
            print("CPUOffloadPolicy 对比实验")
            print("=" * 60)
            print("模型: 4 层 MLP (hidden=2048)")
            print()

        for use_offload in [False, True]:
            torch.cuda.empty_cache()
            peak_mem, avg_time = run_experiment(rank, use_offload)

            if rank == 0:
                label = "CPU Offload" if use_offload else "GPU Only   "
                print(f"  {label}: Peak GPU Memory = {peak_mem:8.1f} MB | Avg Step = {avg_time*1000:.1f} ms")

        if rank == 0:
            print()
            print("分析:")
            print("  CPU Offload 显著降低 GPU peak memory")
            print("  代价是 PCIe 数据传输导致 step time 增加")
            print("  适用场景: 模型太大无法完全放入 GPU，愿意用速度换显存")
            print("  pin_memory=True 可以加速 CPU↔GPU 数据传输")

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
