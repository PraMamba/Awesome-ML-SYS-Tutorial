"""
MixedPrecisionPolicy 示例

典型策略: param_dtype=bf16 (计算用低精度) + reduce_dtype=fp32 (梯度归约用高精度)

运行方式:
    torchrun --nproc_per_node=2 codes/02_mixed_precision.py

要求: PyTorch >= 2.4, GPU 支持 bfloat16
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist

assert torch.__version__ >= "2.4", f"FSDP2 需要 PyTorch >= 2.4，当前版本: {torch.__version__}"

from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy


class SimpleMLP(nn.Module):
    def __init__(self, hidden=1024):
        super().__init__()
        self.layer1 = nn.Linear(hidden, hidden * 4)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden * 4, hidden)

    def forward(self, x):
        return self.layer2(self.relu(self.layer1(x)))


def run_experiment(rank, use_mixed_precision):
    model = SimpleMLP(hidden=1024).cuda()

    if use_mixed_precision:
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,    # 参数存储和计算用 bf16
            reduce_dtype=torch.float32,    # 梯度归约用 fp32（保证精度）
        )
        fully_shard(model.layer1, mp_policy=mp_policy)
        fully_shard(model.layer2, mp_policy=mp_policy)
        fully_shard(model, mp_policy=mp_policy)
    else:
        fully_shard(model.layer1)
        fully_shard(model.layer2)
        fully_shard(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # 检查参数 dtype
    if rank == 0:
        mode = "混合精度 (bf16/fp32)" if use_mixed_precision else "全精度 (fp32)"
        print(f"\n--- {mode} ---")
        for name, param in model.named_parameters():
            print(f"  {name}: dtype = {param.dtype}, shape = {list(param.shape)}")

    # 训练
    torch.cuda.reset_peak_memory_stats()
    for step in range(3):
        inputs = torch.randn(8, 1024, device="cuda")
        targets = torch.randn(8, 1024, device="cuda")

        # 混合精度模式下，输入也需要转为 bf16 以匹配参数 dtype
        if use_mixed_precision:
            inputs = inputs.to(torch.bfloat16)
            targets = targets.to(torch.bfloat16)

        outputs = model(inputs)
        loss = nn.functional.mse_loss(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if rank == 0:
            print(f"  Step {step}: loss = {loss.item():.6f}")

    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    return peak_mem


def main():
    dist.init_process_group(backend="nccl")

    try:
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

        if rank == 0:
            print("=" * 60)
            print("MixedPrecisionPolicy 对比实验")
            print("=" * 60)

        results = {}
        for use_mp in [False, True]:
            torch.cuda.empty_cache()
            peak_mem = run_experiment(rank, use_mp)
            label = "mixed_precision" if use_mp else "full_precision"
            results[label] = peak_mem

        if rank == 0:
            print(f"\n--- 显存对比 ---")
            for label, mem in results.items():
                print(f"  {label:20s}: Peak Memory = {mem:.1f} MB")
            print()
            print("分析:")
            print("  bf16 参数比 fp32 节省一半显存")
            print("  reduce_dtype=fp32 保证梯度归约的数值精度")
            print("  典型生产配置: param_dtype=bf16, reduce_dtype=fp32")

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
