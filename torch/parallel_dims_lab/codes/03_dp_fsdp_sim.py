"""CPU simulation of FSDP parameter and gradient movement."""

from __future__ import annotations

import torch


def split_equal(tensor: torch.Tensor, parts: int) -> list[torch.Tensor]:
    """Split a one-dimensional tensor into equal parameter shards."""

    if tensor.ndim != 1:
        raise ValueError("this lesson expects a one-dimensional parameter")
    if parts < 1 or tensor.numel() % parts:
        raise ValueError("tensor length must be divisible by a positive parts value")
    return list(torch.chunk(tensor, parts, dim=0))


def simulate_all_gather(shards: list[torch.Tensor]) -> torch.Tensor:
    """Reconstruct a full parameter from one shard per rank."""

    if not shards:
        raise ValueError("at least one shard is required")
    return torch.cat([shard.clone() for shard in shards], dim=0)


def simulate_reduce_scatter(local_gradients: list[torch.Tensor]) -> list[torch.Tensor]:
    """Sum full local gradients, then keep one equal shard per rank."""

    if not local_gradients:
        raise ValueError("at least one local gradient is required")
    first_shape = local_gradients[0].shape
    if any(gradient.shape != first_shape for gradient in local_gradients):
        raise ValueError("all local gradients must have the same shape")
    reduced = torch.stack(local_gradients, dim=0).sum(dim=0)
    return split_equal(reduced, len(local_gradients))


def compute_batch(
    dp_replicate: int, dp_shard: int, local_batch: int
) -> tuple[int, int]:
    """Return ``(data_parallel_size, global_batch)``."""

    if min(dp_replicate, dp_shard, local_batch) < 1:
        raise ValueError("batch and data-parallel degrees must be positive")
    data_parallel_size = dp_replicate * dp_shard
    return data_parallel_size, data_parallel_size * local_batch


def run_demo() -> None:
    parameter = torch.arange(16, dtype=torch.float32, device="cpu")
    parameter_shards = split_equal(parameter, parts=4)
    gathered_parameter = simulate_all_gather(parameter_shards)
    torch.testing.assert_close(gathered_parameter, parameter)

    # Rank r contributes a full gradient filled with r + 1.  The reduced
    # gradient is therefore filled with 10 before it is sharded again.
    local_gradients = [torch.full_like(parameter, rank + 1) for rank in range(4)]
    gradient_shards = simulate_reduce_scatter(local_gradients)
    expected_gradient = torch.full_like(parameter, 10)
    torch.testing.assert_close(simulate_all_gather(gradient_shards), expected_gradient)

    data_parallel_size, global_batch = compute_batch(
        dp_replicate=2,
        dp_shard=4,
        local_batch=3,
    )
    print("parameter = [0, 1, ..., 15]")
    print(f"all-gather restored parameter: {gathered_parameter.tolist()}")
    for rank, (parameter_shard, gradient_shard) in enumerate(
        zip(parameter_shards, gradient_shards)
    ):
        print(
            f"rank {rank}: parameter shard={parameter_shard.tolist()}, "
            f"gradient shard={gradient_shard.tolist()}"
        )
    print("reduce-scatter: each gradient shard contains the reduced value 10")
    print(f"data_parallel_size = 2 x 4 = {data_parallel_size}")
    print(f"global_batch = {data_parallel_size} x 3 = {global_batch}")


if __name__ == "__main__":
    run_demo()
