"""Compare a true global token mean with a mean of rank-local means."""

from __future__ import annotations

from collections.abc import Sequence
from statistics import mean


def global_mean(losses_by_rank: Sequence[Sequence[float]]) -> float:
    """Compute one mean over every valid token on every rank."""

    losses = [loss for rank_losses in losses_by_rank for loss in rank_losses]
    if not losses:
        raise ValueError("at least one token loss is required")
    return mean(losses)


def mean_of_means(losses_by_rank: Sequence[Sequence[float]]) -> float:
    """Average rank-local means, giving each rank equal weight."""

    local_means = [mean(rank_losses) for rank_losses in losses_by_rank if rank_losses]
    if not local_means:
        raise ValueError("at least one non-empty rank is required")
    return mean(local_means)


def audit_rows(losses_by_rank: Sequence[Sequence[float]]) -> list[tuple[str, str]]:
    """Return a compact checklist for a token-normalization review."""

    return [
        ("global training target", "mean over all valid tokens"),
        ("local loss", "mean on each rank before communication"),
        ("valid token counts", str([len(losses) for losses in losses_by_rank])),
        ("collective", "sum of token losses and token counts"),
        ("normalization", "divide global loss sum by global token count"),
        ("reference", "compare with the global token mean"),
    ]


def run_experiment(name: str, losses_by_rank: Sequence[Sequence[float]]) -> None:
    global_value = global_mean(losses_by_rank)
    local_value = mean_of_means(losses_by_rank)
    print(f"{name}:")
    print(f"  global mean = {global_value:g}")
    print(f"  mean of rank means = {local_value:g}")
    print(f"  equal = {global_value == local_value}")


def main() -> None:
    experiment_a = [[1, 3], [5, 7]]
    experiment_b = [[1], [3, 5, 7]]
    run_experiment("Experiment A: equal token counts", experiment_a)
    run_experiment("Experiment B: unequal token counts", experiment_b)
    print("gradient audit checklist:")
    for question, answer in audit_rows(experiment_b):
        print(f"  {question}: {answer}")


if __name__ == "__main__":
    main()
