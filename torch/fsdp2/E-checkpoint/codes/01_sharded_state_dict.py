"""
FSDP2 sharded state dict 保存/加载

演示 FSDP2 的检查点基本流程:
  1. 训练几步
  2. 保存 state_dict（自动是 sharded 的）
  3. 重新创建模型 + fully_shard
  4. 加载 state_dict
  5. 验证参数一致

运行方式:
    torchrun --nproc_per_node=2 codes/01_sharded_state_dict.py

要求: PyTorch >= 2.4
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist

assert torch.__version__ >= "2.4", f"FSDP2 需要 PyTorch >= 2.4，当前版本: {torch.__version__}"

from torch.distributed.fsdp import fully_shard


class SimpleMLP(nn.Module):
    def __init__(self, hidden=512):
        super().__init__()
        self.layer1 = nn.Linear(hidden, hidden)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden, hidden)

    def forward(self, x):
        return self.layer2(self.relu(self.layer1(x)))


def main():
    dist.init_process_group(backend="nccl")

    try:
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

        if rank == 0:
            print("=" * 60)
            print("FSDP2 Sharded State Dict 保存/加载")
            print("=" * 60)

        # === 1. 创建模型并训练 ===
        torch.manual_seed(42)
        model = SimpleMLP(hidden=512).cuda()
        fully_shard(model.layer1)
        fully_shard(model.layer2)
        fully_shard(model)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        if rank == 0:
            print("\n--- 训练 3 步 ---")
        for step in range(3):
            inputs = torch.randn(4, 512, device="cuda")
            targets = torch.randn(4, 512, device="cuda")
            loss = nn.functional.mse_loss(model(inputs), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if rank == 0:
                print(f"  Step {step}: loss = {loss.item():.6f}")

        # === 2. 保存 state_dict ===
        state_dict = model.state_dict()
        if rank == 0:
            print(f"\n--- 保存 state_dict ---")
            print(f"  state_dict keys: {list(state_dict.keys())}")
            for key, val in state_dict.items():
                print(f"  {key}: shape={list(val.shape)}, dtype={val.dtype}")

        # === 3. 创建新模型并加载 ===
        if rank == 0:
            print(f"\n--- 创建新模型并加载 state_dict ---")

        torch.manual_seed(0)  # 不同的种子，确保参数不同
        new_model = SimpleMLP(hidden=512).cuda()
        fully_shard(new_model.layer1)
        fully_shard(new_model.layer2)
        fully_shard(new_model)

        new_model.load_state_dict(state_dict)

        # === 4. 验证参数一致 ===
        if rank == 0:
            print("\n--- 验证参数一致性 ---")

        test_input = torch.randn(2, 512, device="cuda")
        with torch.no_grad():
            out_old = model(test_input)
            out_new = new_model(test_input)

        match = torch.allclose(out_old, out_new, atol=1e-6)
        if rank == 0:
            print(f"  原始模型输出: {out_old.flatten()[:3].tolist()}")
            print(f"  加载模型输出: {out_new.flatten()[:3].tolist()}")
            print(f"  输出一致: {match}")
            print("\n检查点保存/加载验证完成!")

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
