"""
DCP (Distributed Checkpoint) save/load 示例

使用 torch.distributed.checkpoint 的 save/load API，
支持分布式保存和加载，每个 rank 保存自己的分片。

运行方式:
    torchrun --nproc_per_node=2 codes/02_dcp_save_load.py

要求: PyTorch >= 2.4
"""

import os
import shutil
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.distributed.checkpoint as dcp

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

        ckpt_dir = "/tmp/fsdp2_dcp_ckpt"

        if rank == 0:
            print("=" * 60)
            print("DCP (Distributed Checkpoint) save/load")
            print("=" * 60)

        # === 1. 创建模型并训练 ===
        torch.manual_seed(42)
        model = SimpleMLP(hidden=512).cuda()
        fully_shard(model.layer1)
        fully_shard(model.layer2)
        fully_shard(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        for step in range(3):
            inputs = torch.randn(4, 512, device="cuda")
            targets = torch.randn(4, 512, device="cuda")
            loss = nn.functional.mse_loss(model(inputs), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if rank == 0:
            print("训练完成 (3 步)")

        # 记录训练后的输出
        test_input = torch.randn(2, 512, device="cuda")
        torch.manual_seed(123)  # 固定测试输入
        test_input = torch.randn(2, 512, device="cuda")
        with torch.no_grad():
            original_output = model(test_input)

        # === 2. DCP 保存 ===
        if rank == 0:
            # 清理旧检查点
            if os.path.exists(ckpt_dir):
                shutil.rmtree(ckpt_dir)
            print(f"\n--- DCP 保存到 {ckpt_dir} ---")

        dist.barrier()

        state_dict = {"model": model.state_dict()}
        dcp.save(state_dict, checkpoint_id=ckpt_dir)

        if rank == 0:
            # 检查保存的文件
            files = os.listdir(ckpt_dir)
            print(f"  保存的文件: {files}")

        # === 3. 创建新模型并用 DCP 加载 ===
        if rank == 0:
            print(f"\n--- DCP 加载 ---")

        torch.manual_seed(0)  # 不同种子
        new_model = SimpleMLP(hidden=512).cuda()
        fully_shard(new_model.layer1)
        fully_shard(new_model.layer2)
        fully_shard(new_model)

        state_dict_to_load = {"model": new_model.state_dict()}
        dcp.load(state_dict_to_load, checkpoint_id=ckpt_dir)
        new_model.load_state_dict(state_dict_to_load["model"])

        if rank == 0:
            print("  DCP 加载完成")

        # === 4. 验证 ===
        with torch.no_grad():
            loaded_output = new_model(test_input)

        match = torch.allclose(original_output, loaded_output, atol=1e-6)
        if rank == 0:
            print(f"\n--- 验证 ---")
            print(f"  原始输出: {original_output.flatten()[:3].tolist()}")
            print(f"  加载输出: {loaded_output.flatten()[:3].tolist()}")
            print(f"  输出一致: {match}")

        # 清理
        dist.barrier()
        if rank == 0:
            if os.path.exists(ckpt_dir):
                shutil.rmtree(ckpt_dir)
                print(f"\n已清理检查点目录: {ckpt_dir}")

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
