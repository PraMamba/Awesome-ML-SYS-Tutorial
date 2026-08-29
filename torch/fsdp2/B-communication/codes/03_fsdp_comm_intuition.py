"""
手动模拟 FSDP 通信模式

FSDP 的核心通信模式:
  Forward:  AllGather 参数分片 → 拼成完整参数 → 计算
  Backward: 计算梯度 → ReduceScatter 梯度 → 每个 rank 只保留自己的梯度分片

本脚本手动实现这个流程，帮助理解 FSDP 的通信本质。

运行方式:
    torchrun --nproc_per_node=2 codes/03_fsdp_comm_intuition.py
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

        if rank == 0:
            print("=" * 60)
            print("手动模拟 FSDP 通信模式")
            print("=" * 60)

        # 模拟一个线性层的权重 W: [out_features, in_features] = [4, 4]
        # 假设完整权重已知（仅用于验证）
        torch.manual_seed(42)
        full_weight = torch.randn(4, 4, device="cuda")

        # === FSDP: 每个 rank 只持有权重的一个分片 ===
        shards = full_weight.chunk(world_size, dim=0)
        local_shard = shards[rank].clone().contiguous()
        print(f"\n进程 {rank}: 本地权重分片 shape = {list(local_shard.shape)}")
        print(f"  分片内容 = {local_shard.flatten()[:4].tolist()}")

        # === Forward: AllGather 收集完整权重 ===
        if rank == 0:
            print("\n--- Forward: AllGather 收集完整权重 ---")

        gathered_shards = [torch.zeros_like(local_shard) for _ in range(world_size)]
        dist.all_gather(gathered_shards, local_shard)
        reconstructed_weight = torch.cat(gathered_shards, dim=0)

        # 验证 AllGather 后得到的权重与原始权重一致
        weight_match = torch.allclose(reconstructed_weight, full_weight)
        print(f"进程 {rank}: AllGather 后权重 shape = {list(reconstructed_weight.shape)}, 与原始一致 = {weight_match}")

        # 使用完整权重做 matmul
        input_data = torch.randn(2, 4, device="cuda")  # batch=2, features=4
        output = input_data @ reconstructed_weight.t()  # [2, 4]
        print(f"进程 {rank}: Forward 输出 shape = {list(output.shape)}")

        # === Backward: ReduceScatter 梯度 ===
        if rank == 0:
            print("\n--- Backward: ReduceScatter 梯度 ---")

        # 模拟梯度（假设 loss 对 output 的梯度为全 1）
        grad_output = torch.ones_like(output)
        # 权重梯度 = grad_output^T @ input_data → shape [4, 4]
        full_grad = grad_output.t() @ input_data

        # ReduceScatter: 对梯度求和并分片
        grad_shards = list(full_grad.chunk(world_size, dim=0))
        local_grad_shard = torch.zeros_like(local_shard)
        dist.reduce_scatter(local_grad_shard, grad_shards, op=dist.ReduceOp.SUM)

        print(f"进程 {rank}: 完整梯度 shape = {list(full_grad.shape)}")
        print(f"进程 {rank}: ReduceScatter 后梯度分片 shape = {list(local_grad_shard.shape)}")

        # === 用梯度分片更新权重分片（模拟 optimizer.step()）===
        lr = 0.01
        local_shard -= lr * local_grad_shard
        print(f"进程 {rank}: 权重分片已更新 (lr={lr})")

        dist.barrier()

        if rank == 0:
            print("\n" + "=" * 60)
            print("总结: FSDP 通信模式")
            print("  Forward:  AllGather(权重分片) → 完整权重 → matmul")
            print("  Backward: 计算完整梯度 → ReduceScatter → 梯度分片")
            print("  Update:   每个 rank 用自己的梯度分片更新权重分片")
            print()
            print("  显存优势: 每个 rank 只常驻 1/N 的权重 + 1/N 的梯度")
            print("  通信代价: 每次 forward/backward 各一次 collective")
            print("=" * 60)

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
