"""
DTensor 属性检查：fully_shard 后参数变成 DTensor

验证 fully_shard 后:
  - type(param) 是 DTensor（而非普通 Tensor）
  - param.placements 显示分片策略
  - param.device_mesh 显示设备组织

运行方式:
    torchrun --nproc_per_node=2 codes/03_dtensor_verification.py

要求: PyTorch >= 2.4
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist

assert torch.__version__ >= "2.4", f"FSDP2 需要 PyTorch >= 2.4，当前版本: {torch.__version__}"

from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(128, 256)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(256, 128)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


def main():
    dist.init_process_group(backend="nccl")

    try:
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)

        model = SimpleMLP().cuda()

        if rank == 0:
            print("=" * 60)
            print("fully_shard 前后参数对比")
            print("=" * 60)

            # fully_shard 之前
            print("\n--- fully_shard 之前 ---")
            for name, param in model.named_parameters():
                print(f"  {name}:")
                print(f"    type  = {type(param).__name__}")
                print(f"    shape = {list(param.shape)}")
                print(f"    是 DTensor = {isinstance(param, DTensor)}")

        # 执行 fully_shard
        fully_shard(model.linear1)
        fully_shard(model.linear2)
        fully_shard(model)

        if rank == 0:
            print("\n--- fully_shard 之后 ---")

        for name, param in model.named_parameters():
            is_dtensor = isinstance(param, DTensor)
            if rank == 0:
                print(f"  {name}:")
                print(f"    type       = {type(param).__name__}")
                print(f"    是 DTensor  = {is_dtensor}")
            if is_dtensor:
                if rank == 0:
                    print(f"    placements = {param.placements}")
                    print(f"    device_mesh= {param.device_mesh}")
                    print(f"    full shape = {list(param.shape)}")
                    print(f"    local shape= {list(param._local_tensor.shape)}")

        # 验证本地分片大小
        dist.barrier()
        if rank == 0:
            print("\n" + "=" * 60)
            print("关键观察:")
            print(f"  1. fully_shard 后参数类型从 Parameter 变成 DTensor")
            print(f"  2. placements 为 (Shard(0),) 表示沿第 0 维分片")
            print(f"  3. 本地分片大小 = 完整大小 / world_size")
            print(f"  4. device_mesh 描述了参与分片的设备组")
            print("=" * 60)

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
