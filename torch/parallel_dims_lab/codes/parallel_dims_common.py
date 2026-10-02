"""Small, dependency-free helpers shared by the ParallelDims lab scripts.

The functions in this module deliberately mirror the arithmetic in
``torchtitan.distributed.parallel_dims.ParallelDims``.  They do not construct
real ``DeviceMesh`` objects or process groups; the lab is intended to make the
shape and validation rules inspectable on a CPU-only machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Mapping


def product(shape: tuple[int, ...]) -> int:
    """Return the product of a mesh shape."""

    return prod(shape)


def seq_len_divisor(tp: int, cp: int) -> int:
    """Return the sequence-length divisor used by the current source.

    Tensor parallelism contributes ``tp`` and the default load-balanced
    context-parallel path contributes ``2 * cp``.
    """

    if tp < 1 or cp < 1:
        raise ValueError("tp and cp must be >= 1")
    return tp * (cp * 2)


def resolve_dp_shard(
    dp_replicate: int,
    dp_shard: int,
    cp: int,
    tp: int,
    pp: int,
    world_size: int,
) -> int:
    """Resolve ``dp_shard=-1`` using the source-code formula.

    No validation is performed here.  This makes the helper useful to the
    advisor, which needs to report all errors instead of stopping at the first
    assertion.
    """

    if dp_shard < 0:
        return world_size // (dp_replicate * cp * tp * pp)
    return dp_shard


def validate_parallel_dims(
    dp_replicate: int,
    dp_shard: int,
    cp: int,
    tp: int,
    pp: int,
    ep: int,
    world_size: int,
) -> int:
    """Validate dimensions with the same checks as current ``ParallelDims``.

    The return value is the resolved ``dp_shard``.  Assertions and the EP
    ``ValueError`` intentionally follow the reference implementation's
    distinction, which makes this helper useful for teaching the API boundary.
    """

    for degree in (dp_replicate, cp, tp, pp, ep):
        assert degree >= 1, "Parallelism degree should be >= 1, except for dp_shard"
    assert dp_shard == -1 or dp_shard >= 1, "dp_shard must -1 or >=1."

    resolved_dp_shard = resolve_dp_shard(
        dp_replicate,
        dp_shard,
        cp,
        tp,
        pp,
        world_size,
    )
    assert resolved_dp_shard >= 1

    dense_product = dp_replicate * resolved_dp_shard * cp * tp * pp
    assert dense_product == world_size, (
        f"Invalid parallel dims: dp_replicate({dp_replicate}) * "
        f"dp_shard({resolved_dp_shard}) * cp({cp}) * tp({tp}) * "
        f"pp({pp}) != WORLD_SIZE({world_size})"
    )

    sparse_region = resolved_dp_shard * cp * tp
    if sparse_region % ep != 0:
        raise ValueError(
            f"expert_parallel_degree ({ep}) must divide "
            f"dp_shard * cp * tp ({sparse_region})"
        )
    return resolved_dp_shard


@dataclass(frozen=True)
class DerivedDimensions:
    """Dimensions derived after ``dp_shard`` has been resolved."""

    dp_replicate: int
    dp_shard: int
    cp: int
    tp: int
    pp: int
    ep: int
    world_size: int

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
    def sparse_region(self) -> int:
        return self.dp_shard * self.cp * self.tp

    @property
    def sequence_length_divisor(self) -> int:
        return seq_len_divisor(self.tp, self.cp)


def derive_dimensions(
    dp_replicate: int,
    dp_shard: int,
    cp: int,
    tp: int,
    pp: int,
    ep: int,
    world_size: int,
) -> DerivedDimensions:
    """Validate and return the dimensions used by the lab's mesh views."""

    resolved_dp_shard = validate_parallel_dims(
        dp_replicate,
        dp_shard,
        cp,
        tp,
        pp,
        ep,
        world_size,
    )
    return DerivedDimensions(
        dp_replicate=dp_replicate,
        dp_shard=resolved_dp_shard,
        cp=cp,
        tp=tp,
        pp=pp,
        ep=ep,
        world_size=world_size,
    )


def mesh_shapes(dims: DerivedDimensions) -> dict[str, tuple[int, ...]]:
    """Return the source-aligned mesh-view shapes.

    ``loss`` is intentionally a submesh: the source flattens only the
    ``(batch, cp)`` axes, so its product is ``batch * cp`` rather than the
    complete world size.  The other views cover all ranks.
    """

    return {
        "dataloading": (dims.pp, dims.batch, dims.cp, dims.tp),
        "loss": (dims.loss,),
        "spmd_types_storage": (
            dims.pp,
            dims.dp_replicate,
            dims.dp_shard,
            dims.cp,
            dims.tp,
        ),
        "spmd_types_fwd_bwd": (dims.pp, dims.batch, dims.cp, dims.tp),
        "partial_dtensor_dense": (
            dims.pp,
            dims.dp_replicate,
            dims.fsdp,
            dims.tp,
        ),
        "sparse": (dims.pp, dims.dp_replicate, dims.efsdp, dims.ep),
    }


def named_shape(shape: tuple[int, ...], names: tuple[str, ...]) -> str:
    """Format a shape as ``(name=value, ...)`` for CLI output."""

    if len(shape) != len(names):
        raise ValueError("shape and names must have the same length")
    return "(" + ", ".join(f"{name}={value}" for name, value in zip(names, shape)) + ")"


def mesh_shape_names() -> Mapping[str, tuple[str, ...]]:
    """Return the labels associated with :func:`mesh_shapes`."""

    return {
        "dataloading": ("pp", "batch", "cp", "tp"),
        "loss": ("loss",),
        "spmd_types_storage": ("pp", "dp_replicate", "dp_shard", "cp", "tp"),
        "spmd_types_fwd_bwd": ("pp", "dp", "cp", "tp"),
        "partial_dtensor_dense": ("pp", "dp_replicate", "fsdp", "tp"),
        "sparse": ("pp", "dp_replicate", "efsdp", "ep"),
    }
