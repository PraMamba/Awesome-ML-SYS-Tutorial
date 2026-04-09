"""
故障诊断示例：故意制造 hang，演示超时行为

使用 mp.spawn 启动，rank 0 调用 barrier，rank 1 跳过 → 导致 hang → NCCL 超时

运行方式:
    NCCL_DEBUG=INFO python codes/04_fault_diagnosis.py

预期行为: rank 0 会等待 barrier，但 rank 1 跳过了 → 触发超时错误
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"

    # 设置较短的超时，以便快速看到错误
    import datetime
    timeout = datetime.timedelta(seconds=10)

    torch.cuda.set_device(rank)

    try:
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=timeout,
        )
        print(f"进程 {rank}: 已加入进程组")

        # 正常的 all_reduce（所有 rank 都参与，不会 hang）
        tensor = torch.tensor([rank], dtype=torch.float32, device="cuda")
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        print(f"进程 {rank}: 正常 all_reduce 完成，结果 = {tensor.item()}")

        # 故意制造不对称调用 → hang
        if rank == 0:
            print(f"\n进程 {rank}: 准备调用 barrier (rank 1 不会调用 → 即将 hang)")
            print(f"进程 {rank}: 等待中... (将在 {timeout} 后超时)")
            try:
                dist.barrier()
                print(f"进程 {rank}: barrier 完成（不应该执行到这里）")
            except Exception as e:
                print(f"\n进程 {rank}: 捕获到超时错误 ✓")
                print(f"  错误类型: {type(e).__name__}")
                print(f"  错误信息: {str(e)[:200]}")
                print(f"\n  诊断提示:")
                print(f"  1. hang 通常由不对称的 collective 调用引起")
                print(f"  2. 设置 NCCL_DEBUG=INFO 可以看到更多通信细节")
                print(f"  3. 设置 TORCH_DISTRIBUTED_DEBUG=DETAIL 可以看到哪个 rank 在等待")
                print(f"  4. 检查所有 rank 是否执行了相同的代码路径")
        else:
            print(f"进程 {rank}: 跳过 barrier（故意制造不对称调用）")
            print(f"进程 {rank}: 直接退出")

    except Exception as e:
        print(f"进程 {rank}: 发生错误 - {type(e).__name__}: {str(e)[:200]}")

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print(f"此示例需要至少 2 个 GPU，当前只有 {world_size} 个")
        print("请在多 GPU 环境运行")
        return

    print("=" * 60)
    print("故障诊断示例：不对称 collective 调用导致 hang")
    print("=" * 60)
    print(f"将使用 {world_size} 个 GPU（仅使用前 2 个）")
    print("rank 0 会调用 barrier，rank 1 会跳过 → 导致超时")
    print()

    mp.spawn(worker, args=(2,), nprocs=2, join=True)


if __name__ == "__main__":
    main()
