"""CPU matrix-multiplication simulation of two-way tensor parallelism."""

from __future__ import annotations

import torch


def dense_mlp(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
    """Compute the unsplit two-layer linear MLP used in the exercise."""

    return x @ w1 @ w2


def tp_mlp(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    tp: int = 2,
) -> torch.Tensor:
    """Compute column-parallel ``w1`` plus row-parallel ``w2``."""

    if tp < 1:
        raise ValueError("tp must be positive")
    if w1.shape[0] != x.shape[1]:
        raise ValueError("w1 input width must match x hidden width")
    if w2.shape[0] != w1.shape[1] or w2.shape[1] != x.shape[1]:
        raise ValueError("w2 must map the full intermediate width back to hidden")
    if w1.shape[1] % tp:
        raise ValueError("w1 intermediate width must be divisible by tp")

    w1_shards = torch.chunk(w1, tp, dim=1)
    w2_shards = torch.chunk(w2, tp, dim=0)
    partial_outputs = []
    for w1_part, w2_part in zip(w1_shards, w2_shards):
        partial_outputs.append(x @ w1_part @ w2_part)
    return torch.stack(partial_outputs, dim=0).sum(dim=0)


def run_demo() -> None:
    torch.manual_seed(0)
    batch, hidden, intermediate, tp = 3, 4, 16, 2
    x = torch.randn(batch, hidden, device="cpu")
    w1 = torch.randn(hidden, intermediate, device="cpu")
    w2 = torch.randn(intermediate, hidden, device="cpu")

    dense_result = dense_mlp(x, w1, w2)
    tp_result = tp_mlp(x, w1, w2, tp=tp)
    torch.testing.assert_close(tp_result, dense_result, rtol=1e-5, atol=1e-5)

    print(f"x shape  = {tuple(x.shape)}")
    print(f"w1 shape = {tuple(w1.shape)} -> {tp} column shards")
    print(f"w2 shape = {tuple(w2.shape)} -> {tp} row shards")
    print("tp result matches dense result: True")
    print(
        f"maximum absolute error = {(tp_result - dense_result).abs().max().item():.3e}"
    )


if __name__ == "__main__":
    run_demo()
