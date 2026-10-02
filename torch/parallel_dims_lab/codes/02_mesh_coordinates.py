"""Pure-Python row-major mesh coordinate and axis-group exercises."""

from __future__ import annotations

import argparse
from math import prod
from typing import Sequence


def _validated_shape(shape: Sequence[int]) -> tuple[int, ...]:
    result = tuple(int(size) for size in shape)
    if not result:
        raise ValueError("shape must contain at least one axis")
    if any(size <= 0 for size in result):
        raise ValueError(f"shape entries must be positive, got {result}")
    return result


def rank_to_coord(rank: int, shape: Sequence[int]) -> tuple[int, ...]:
    """Convert a row-major flat rank to an N-dimensional coordinate."""

    normalized_shape = _validated_shape(shape)
    world_size = prod(normalized_shape)
    if not 0 <= rank < world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")

    coordinate = [0] * len(normalized_shape)
    remainder = int(rank)
    for axis in range(len(normalized_shape) - 1, -1, -1):
        coordinate[axis] = remainder % normalized_shape[axis]
        remainder //= normalized_shape[axis]
    return tuple(coordinate)


def coord_to_rank(coord: Sequence[int], shape: Sequence[int]) -> int:
    """Convert an N-dimensional row-major coordinate to a flat rank."""

    normalized_shape = _validated_shape(shape)
    normalized_coord = tuple(int(value) for value in coord)
    if len(normalized_coord) != len(normalized_shape):
        raise ValueError(
            f"coord has {len(normalized_coord)} axes, expected {len(normalized_shape)}"
        )
    for axis, (value, size) in enumerate(zip(normalized_coord, normalized_shape)):
        if not 0 <= value < size:
            raise ValueError(f"coord axis {axis} must be in [0, {size}), got {value}")

    rank = 0
    for value, size in zip(normalized_coord, normalized_shape):
        rank = rank * size + value
    return rank


def axis_group(rank: int, shape: Sequence[int], axis: int) -> list[int]:
    """Return all ranks sharing ``axis``'s coordinate with ``rank``."""

    normalized_shape = _validated_shape(shape)
    if not 0 <= axis < len(normalized_shape):
        raise ValueError(f"axis must be in [0, {len(normalized_shape)}), got {axis}")
    fixed_coordinate = rank_to_coord(rank, normalized_shape)[axis]
    return [
        candidate
        for candidate in range(prod(normalized_shape))
        if rank_to_coord(candidate, normalized_shape)[axis] == fixed_coordinate
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape",
        nargs="+",
        type=int,
        default=[2, 4],
        help="row-major mesh shape, for example --shape 2 4 or --shape 2 2 2",
    )
    parser.add_argument("--rank", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    shape = _validated_shape(args.shape)
    coordinate = rank_to_coord(args.rank, shape)
    print(f"shape = {shape}")
    print(f"rank = {args.rank}")
    print(f"coordinate = {coordinate}")
    print(f"round_trip_rank = {coord_to_rank(coordinate, shape)}")
    for axis in range(len(shape)):
        print(f"axis {axis} group = {axis_group(args.rank, shape, axis)}")


if __name__ == "__main__":
    main()
