"""
AllReduce 基础：模拟 DDP 梯度同步

每个 rank 有一个"梯度"张量，AllReduce 后所有 rank 得到相同的梯度之和。
这就是 DDP 在 backward 时做的事情。

运行方式:
    torchrun --nproc_per_node=2 codes/01_allreduce_basics.py
"""

import os
import torch
import torch.distributed as dist


def main():
    dist.init_process_group(backend="nccl")

    try:
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)

        # 模拟每个 rank 的梯度（不同的值）
        gradient = torch.tensor([rank * 1.0, rank * 2.0, rank * 3.0], device="cuda")
        print(f"进程 {rank}: AllReduce 之前的梯度 = {gradient.tolist()}")

        # AllReduce: 所有 rank 的梯度求和，结果广播到每个 rank
        dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
        print(f"进程 {rank}: AllReduce(SUM) 之后的梯度 = {gradient.tolist()}")

        # DDP 实际上会除以 world_size 得到平均梯度
        avg_gradient = gradient / world_size
        print(f"进程 {rank}: 平均梯度 (÷ world_size) = {avg_gradient.tolist()}")

        dist.barrier()

        if rank == 0:
            print("\n" + "=" * 60)
            print("总结: AllReduce = DDP 梯度同步的核心操作")
            print("  1. 每个 rank 独立计算本地梯度 (backward)")
            print("  2. AllReduce(SUM) 将所有 rank 的梯度求和")
            print("  3. 每个 rank 拿到相同的梯度和 → 除以 world_size → 更新参数")
            print("  4. 因此所有 rank 的模型参数始终保持一致")
            print("=" * 60)

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
