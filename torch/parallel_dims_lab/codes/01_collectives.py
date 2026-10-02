"""CPU/Gloo collective communication demo.

Run the real four-process example with::

    torchrun --standalone --nproc-per-node=4 01_collectives.py

When invoked with plain ``python`` the script uses a one-process explanation
mode so that every lab script remains runnable without a launcher.  The
four-process path is the one that verifies the required sum of 6 and the
``[0, 2]`` subgroup.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


EXPECTED_WORLD_SIZE = 4
SUBGROUP_RANKS = [0, 2]


def _launched_by_torchrun() -> bool:
    """Return whether the launcher supplied the distributed environment."""

    return all(name in os.environ for name in ("RANK", "WORLD_SIZE"))


def run_gloo_demo() -> None:
    """Run the real CPU/Gloo all-reduce and subgroup demonstration."""

    dist.init_process_group(backend="gloo", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if world_size != EXPECTED_WORLD_SIZE:
        dist.destroy_process_group()
        raise RuntimeError(
            f"This lesson expects world_size={EXPECTED_WORLD_SIZE}; got {world_size}."
        )

    value = torch.tensor(float(rank), dtype=torch.float32, device="cpu")
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    assert value.item() == 6.0, f"rank {rank} saw {value.item()}, expected 6"

    # Every global rank must call new_group in the same order.  Non-members
    # receive GROUP_MEMBER.NON_GROUP_MEMBER and must not call a collective on
    # that subgroup.
    subgroup = dist.new_group(ranks=SUBGROUP_RANKS, backend="gloo")
    if rank in SUBGROUP_RANKS:
        subgroup_value = torch.tensor(float(rank), dtype=torch.float32, device="cpu")
        dist.all_reduce(subgroup_value, op=dist.ReduceOp.SUM, group=subgroup)
        assert subgroup_value.item() == 2.0

    # Keep the output readable and ensure subgroup participants have finished
    # before rank 0 prints the summary.
    dist.barrier()
    if rank == 0:
        print("world group: 0 + 1 + 2 + 3 = 6")
        print("all_reduce sum = 6 (verified on every rank)")
        print("subgroup [0, 2]: 0 + 2 = 2")
        print(
            "ranks [1, 3]: non-members; they do not participate in subgroup all_reduce"
        )
    dist.barrier()
    dist.destroy_process_group()


def run_single_process_explanation() -> None:
    """Keep ``python 01_collectives.py`` useful without a process launcher."""

    value = torch.tensor(0.0, dtype=torch.float32, device="cpu")
    print("single-process fallback: rank=0, world_size=1, Gloo demo not launched")
    print(f"local value = {value.item():.0f}")
    print("run with: torchrun --standalone --nproc-per-node=4 01_collectives.py")


def main() -> None:
    if _launched_by_torchrun():
        run_gloo_demo()
    else:
        run_single_process_explanation()


if __name__ == "__main__":
    main()
