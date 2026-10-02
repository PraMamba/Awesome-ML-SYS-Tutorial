"""Pure-Python MoE token dispatch, expert computation, and combine demo."""

from __future__ import annotations

import argparse
from statistics import mean


def expert_to_ep_rank(expert: int, num_experts: int, ep: int) -> int:
    """Map contiguous expert blocks to EP ranks."""

    if num_experts < 1 or ep < 1 or num_experts % ep:
        raise ValueError("num_experts must be divisible by a positive ep")
    if not 0 <= expert < num_experts:
        raise ValueError(f"expert must be in [0, {num_experts}), got {expert}")
    experts_per_rank = num_experts // ep
    return expert // experts_per_rank


def default_expert_assignments(num_tokens: int, num_experts: int) -> list[int]:
    """Use a transparent round-robin router for the command-line demo."""

    if num_tokens < 1 or num_experts < 1:
        raise ValueError("num_tokens and num_experts must be positive")
    return [token_id % num_experts for token_id in range(num_tokens)]


def simulate_routing(
    expert_assignments: list[int],
    num_experts: int,
    ep: int,
) -> dict[str, object]:
    """Dispatch tokens, compute in expert order, and restore token order."""

    if not expert_assignments:
        raise ValueError("at least one token is required")
    buckets = {rank: [] for rank in range(ep)}
    token_to_ep_rank = []
    for token_id, expert in enumerate(expert_assignments):
        rank = expert_to_ep_rank(expert, num_experts, ep)
        token_to_ep_rank.append(rank)
        buckets[rank].append(token_id)

    # The value is intentionally simple: it makes expert computation visible
    # while keeping the dispatch/combine ordering easy to inspect.
    restored_values: list[int | None] = [None] * len(expert_assignments)
    for token_ids in buckets.values():
        for token_id in token_ids:
            expert = expert_assignments[token_id]
            restored_values[token_id] = (token_id + 1) * (expert + 1)

    assert all(value is not None for value in restored_values)
    rank_loads = [len(buckets[rank]) for rank in range(ep)]
    return {
        "assignments": expert_assignments,
        "token_to_ep_rank": token_to_ep_rank,
        "buckets": buckets,
        "restored_values": restored_values,
        "rank_loads": rank_loads,
    }


def load_statistics(rank_loads: list[int]) -> dict[str, float | int]:
    """Summarize token balance across EP ranks."""

    if not rank_loads:
        raise ValueError("at least one EP rank is required")
    return {
        "minimum": min(rank_loads),
        "maximum": max(rank_loads),
        "average": mean(rank_loads),
        "spread": max(rank_loads) - min(rank_loads),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-tokens", type=int, default=16)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--ep", type=int, default=4)
    args = parser.parse_args()

    assignments = default_expert_assignments(args.num_tokens, args.num_experts)
    result = simulate_routing(assignments, args.num_experts, args.ep)
    loads = result["rank_loads"]
    assert isinstance(loads, list)
    statistics = load_statistics(loads)

    print(f"token -> expert: {list(enumerate(assignments))}")
    print(
        "expert mapping: "
        + ", ".join(
            f"{expert}->{expert_to_ep_rank(expert, args.num_experts, args.ep)}"
            for expert in range(args.num_experts)
        )
    )
    print("EP buckets:")
    buckets = result["buckets"]
    assert isinstance(buckets, dict)
    for rank, token_ids in buckets.items():
        print(f"  EP rank {rank}: tokens {token_ids}")
    print(f"restored output order: {result['restored_values']}")
    print(f"tokens per EP rank: {loads}")
    print(
        "load statistics: "
        f"min={statistics['minimum']}, max={statistics['maximum']}, "
        f"average={statistics['average']:.2f}, spread={statistics['spread']}"
    )


if __name__ == "__main__":
    main()
