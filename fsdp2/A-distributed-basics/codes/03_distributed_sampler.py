"""
DistributedSampler 数据切分演示

每个 rank 拿到不同的样本子集，保证数据不重叠。

运行方式:
    torchrun --nproc_per_node=2 codes/03_distributed_sampler.py
"""

import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler


def main():
    dist.init_process_group(backend="nccl")

    try:
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)

        # 创建一个简单数据集：20 个样本，每个样本用其 index 标识
        data = torch.arange(20, dtype=torch.float32).unsqueeze(1)  # shape [20, 1]
        labels = torch.arange(20, dtype=torch.long)
        dataset = TensorDataset(data, labels)

        # 创建 DistributedSampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=42,
        )

        # 创建 DataLoader
        dataloader = DataLoader(dataset, batch_size=4, sampler=sampler)

        print(f"\n{'=' * 40}")
        print(f"进程 {rank} 的数据分配情况 (共 {world_size} 个进程)")
        print(f"{'=' * 40}")

        # 模拟 2 个 epoch
        for epoch in range(2):
            # 每个 epoch 必须调用 set_epoch，否则 shuffle 不变
            sampler.set_epoch(epoch)

            all_indices = []
            print(f"\n--- Epoch {epoch} ---")
            for batch_idx, (batch_data, batch_labels) in enumerate(dataloader):
                indices = batch_labels.tolist()
                all_indices.extend(indices)
                print(f"  进程 {rank} | Batch {batch_idx}: 样本 indices = {indices}")

            print(f"  进程 {rank} | Epoch {epoch} 总共拿到 {len(all_indices)} 个样本: {sorted(all_indices)}")

        # 验证：不同 rank 拿到的样本不重叠
        dist.barrier()
        if rank == 0:
            print(f"\n{'=' * 40}")
            print("注意: DistributedSampler 保证每个 rank 拿到不同的数据分片")
            print("如果 shuffle=True, 每个 epoch set_epoch() 后会重新打乱")
            print(f"{'=' * 40}")

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
