"""
Resharding Checkpoint: N GPU 保存 → M GPU 加载

使用 DCP 保存的检查点支持弹性恢复：
  - 用 2 GPU 训练并保存
  - 用不同数量的 GPU 加载

使用 argparse --mode save/load 分步运行。

运行方式:
    # 第一步: 用 N 个 GPU 保存
    torchrun --nproc_per_node=2 codes/03_resharding_checkpoint.py --mode save

    # 第二步: 用 M 个 GPU 加载（M 可以 != N）
    torchrun --nproc_per_node=2 codes/03_resharding_checkpoint.py --mode load
    # 或者用不同数量的 GPU
    torchrun --nproc_per_node=1 codes/03_resharding_checkpoint.py --mode load

要求: PyTorch >= 2.4
"""

import os
import argparse
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


def create_fsdp_model():
    """创建并 fully_shard 模型"""
    model = SimpleMLP(hidden=512).cuda()
    fully_shard(model.layer1)
    fully_shard(model.layer2)
    fully_shard(model)
    return model


def save_checkpoint(ckpt_dir):
    """训练并保存检查点"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print(f"--- 保存模式 (world_size={world_size}) ---")

    torch.manual_seed(42)
    model = create_fsdp_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # 训练几步
    for step in range(3):
        inputs = torch.randn(4, 512, device="cuda")
        targets = torch.randn(4, 512, device="cuda")
        loss = nn.functional.mse_loss(model(inputs), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if rank == 0:
            print(f"  Step {step}: loss = {loss.item():.6f}")

    # 保存
    if rank == 0:
        if os.path.exists(ckpt_dir):
            shutil.rmtree(ckpt_dir)
    dist.barrier()

    state_dict = {"model": model.state_dict()}
    dcp.save(state_dict, checkpoint_id=ckpt_dir)

    if rank == 0:
        print(f"  检查点已保存到: {ckpt_dir}")
        print(f"  保存时的 world_size = {world_size}")


def load_checkpoint(ckpt_dir):
    """从检查点加载（可能是不同数量的 GPU）"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print(f"\n--- 加载模式 (world_size={world_size}) ---")

    if not os.path.exists(ckpt_dir):
        if rank == 0:
            print(f"  错误: 检查点目录不存在: {ckpt_dir}")
            print(f"  请先运行 --mode save")
        return

    # 创建新模型（可能在不同数量的 GPU 上）
    model = create_fsdp_model()

    # DCP 加载（自动处理 resharding）
    state_dict = {"model": model.state_dict()}
    dcp.load(state_dict, checkpoint_id=ckpt_dir)
    model.load_state_dict(state_dict["model"])

    if rank == 0:
        print(f"  检查点加载成功!")
        print(f"  当前 world_size = {world_size}")

    # 验证模型可以正常 forward
    with torch.no_grad():
        test_input = torch.randn(2, 512, device="cuda")
        output = model(test_input)
        if rank == 0:
            print(f"  Forward 验证通过, 输出 shape = {list(output.shape)}")
            print(f"  输出前 3 个值: {output.flatten()[:3].tolist()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["save", "load"], required=True,
                        help="save: 训练并保存检查点; load: 从检查点加载")
    parser.add_argument("--ckpt_dir", default="/tmp/fsdp2_reshard_ckpt",
                        help="检查点目录")
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")

    try:
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

        if rank == 0:
            print("=" * 60)
            print("Resharding Checkpoint: N GPU 保存 → M GPU 加载")
            print("=" * 60)

        if args.mode == "save":
            save_checkpoint(args.ckpt_dir)
        else:
            load_checkpoint(args.ckpt_dir)

    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
