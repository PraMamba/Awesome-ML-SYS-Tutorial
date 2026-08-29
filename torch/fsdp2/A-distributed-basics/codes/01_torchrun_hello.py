"""
torchrun 基础：打印每个 rank 的分布式信息

运行方式:
    torchrun --nproc_per_node=2 codes/01_torchrun_hello.py
"""

import os
import torch
import torch.distributed as dist


def main():
    # 初始化进程组（torchrun 自动设置环境变量）
    dist.init_process_group(backend="nccl")

    try:
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()

        # 设置当前进程使用的 GPU
        torch.cuda.set_device(local_rank)

        # 打印分布式环境信息
        print(
            f"进程 {rank} 信息:\n"
            f"  LOCAL_RANK  = {local_rank}\n"
            f"  RANK        = {rank}\n"
            f"  WORLD_SIZE  = {world_size}\n"
            f"  MASTER_ADDR = {os.environ.get('MASTER_ADDR', 'N/A')}\n"
            f"  MASTER_PORT = {os.environ.get('MASTER_PORT', 'N/A')}\n"
            f"  GPU 设备    = {torch.cuda.get_device_name(local_rank)}\n"
            f"  GPU 编号    = cuda:{local_rank}\n"
        )

        # 简单通信验证：每个 rank 创建一个标识张量，然后 all_reduce 求和
        tensor = torch.tensor([rank], dtype=torch.float32, device="cuda")
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        expected = sum(range(world_size))
        print(f"进程 {rank}: all_reduce 验证 = {tensor.item()} (期望 {expected})")

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
