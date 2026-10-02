"""A small CPU-only version of the current ``ParallelDims`` validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from parallel_dims_common import (
    derive_dimensions,
    seq_len_divisor,
    validate_parallel_dims,
)


@dataclass
class ParallelDimsLite:
    """The current source fields, without a real ``DeviceMesh``."""

    dp_replicate: int
    dp_shard: int
    cp: int
    tp: int
    pp: int
    ep: int
    world_size: int
    spmd_backend: Literal["partial_dtensor", "spmd_types"] = "spmd_types"

    def __post_init__(self) -> None:
        # The reference class validates during construction.
        self.validate()

    def validate(self) -> None:
        """Apply the source checks and resolve ``dp_shard=-1`` in place."""

        self.dp_shard = validate_parallel_dims(
            self.dp_replicate,
            self.dp_shard,
            self.cp,
            self.tp,
            self.pp,
            self.ep,
            self.world_size,
        )

    @property
    def batch(self) -> int:
        return self.dp_replicate * self.dp_shard

    @property
    def loss(self) -> int:
        return self.batch * self.cp

    @property
    def fsdp(self) -> int:
        return self.dp_shard * self.cp

    @property
    def efsdp(self) -> int:
        return self.fsdp * self.tp // self.ep

    @property
    def seq_len_divisor(self) -> int:
        return seq_len_divisor(self.tp, self.cp)

    def derived(self) -> dict[str, int]:
        """Return the four dimensions most useful in the lab output."""

        dimensions = derive_dimensions(
            self.dp_replicate,
            self.dp_shard,
            self.cp,
            self.tp,
            self.pp,
            self.ep,
            self.world_size,
        )
        return {
            "batch": dimensions.batch,
            "loss": dimensions.loss,
            "fsdp": dimensions.fsdp,
            "efsdp": dimensions.efsdp,
        }


def main() -> None:
    dims = ParallelDimsLite(
        dp_replicate=2,
        dp_shard=2,
        cp=2,
        tp=2,
        pp=2,
        ep=4,
        world_size=32,
    )
    print("ParallelDimsLite validation: VALID")
    print(f"fields = {dims}")
    print(f"derived = {dims.derived()}")
    print(f"seq_len_divisor = {dims.seq_len_divisor}")


if __name__ == "__main__":
    main()
