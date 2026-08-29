"""
AllGather + ReduceScatter 分解演示

关键洞察: AllReduce = ReduceScatter + AllGather

运行方式:
    torchrun --nproc_per_node=2 codes/02_allgather_reducescatter.py
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

        # === 1. AllGather 演示 ===
        # 每个 rank 有一个分片，AllGather 后每个 rank 拿到所有分片
        local_shard = torch.tensor([rank * 10.0, rank * 10.0 + 1.0], device="cuda")
        gathered = [torch.zeros(2, device="cuda") for _ in range(world_size)]
        dist.all_gather(gathered, local_shard)

        if rank == 0:
            print("=" * 60)
            print("1. AllGather: 每个 rank 的分片 → 每个 rank 拿到完整数据")
            print("=" * 60)
        print(f"  进程 {rank}: 本地分片 = {local_shard.tolist()}")
        print(f"  进程 {rank}: AllGather 后 = {[g.tolist() for g in gathered]}")

        dist.barrier()

        # === 2. ReduceScatter 演示 ===
        # 每个 rank 有完整数据，ReduceScatter 后每个 rank 拿到一个 reduced 分片
        full_data = torch.tensor(
            [rank + 1.0] * (2 * world_size), device="cuda"
        )
        output = torch.zeros(2, device="cuda")
        # reduce_scatter 需要输入列表，每个元素对应一个分片
        input_list = list(full_data.chunk(world_size))
        dist.reduce_scatter(output, input_list, op=dist.ReduceOp.SUM)

        if rank == 0:
            print("\n" + "=" * 60)
            print("2. ReduceScatter: 先 Reduce(SUM) 再 Scatter 到各 rank")
            print("=" * 60)
        print(f"  进程 {rank}: 输入数据 = {full_data.tolist()}")
        print(f"  进程 {rank}: ReduceScatter 结果 = {output.tolist()}")

        dist.barrier()

        # === 3. 验证 AllReduce = ReduceScatter + AllGather ===
        data = torch.tensor(
            [rank * 1.0 + i for i in range(2 * world_size)], device="cuda"
        )

        # 方法 A: 直接 AllReduce
        data_a = data.clone()
        dist.all_reduce(data_a, op=dist.ReduceOp.SUM)

        # 方法 B: ReduceScatter → AllGather
        data_b = data.clone()
        rs_output = torch.zeros(2, device="cuda")
        dist.reduce_scatter(rs_output, list(data_b.chunk(world_size)), op=dist.ReduceOp.SUM)
        ag_output = [torch.zeros(2, device="cuda") for _ in range(world_size)]
        dist.all_gather(ag_output, rs_output)
        result_b = torch.cat(ag_output)

        if rank == 0:
            print("\n" + "=" * 60)
            print("3. 验证: AllReduce ≡ ReduceScatter + AllGather")
            print("=" * 60)
        match = torch.allclose(data_a, result_b)
        print(f"  进程 {rank}: AllReduce 结果 = {data_a.tolist()}")
        print(f"  进程 {rank}: RS+AG 结果   = {result_b.tolist()}")
        print(f"  进程 {rank}: 两者一致 = {match}")

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
