"""CPU-only ParallelDims configuration auditor and mesh visualizer."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from math import prod

from parallel_dims_common import (
    DerivedDimensions,
    mesh_shape_names,
    mesh_shapes,
    named_shape,
    resolve_dp_shard,
    seq_len_divisor,
)


@dataclass(frozen=True)
class AdvisorConfig:
    """Cluster, model, and parallelism values accepted by the CLI."""

    world_size: int = 32
    gpus_per_node: int = 8
    dp_replicate: int = 2
    dp_shard: int = 2
    cp: int = 2
    tp: int = 2
    pp: int = 2
    ep: int = 4
    seq_len: int = 8192
    num_layers: int = 32
    num_heads: int = 32
    num_experts: int = 8
    spmd_backend: str = "spmd_types"
    rank: int = 0


@dataclass(frozen=True)
class Finding:
    level: str
    message: str


@dataclass
class AuditReport:
    config: AdvisorConfig
    findings: list[Finding] = field(default_factory=list)
    dimensions: DerivedDimensions | None = None
    shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)

    @property
    def has_errors(self) -> bool:
        return any(finding.level == "ERROR" for finding in self.findings)

    def add(self, level: str, message: str) -> None:
        self.findings.append(Finding(level, message))


def _coord(rank: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    """Return a row-major coordinate for one full-world mesh view."""

    if not 0 <= rank < prod(shape):
        raise ValueError(f"rank must be in [0, {prod(shape)}), got {rank}")
    values = [0] * len(shape)
    remainder = rank
    for axis in range(len(shape) - 1, -1, -1):
        values[axis] = remainder % shape[axis]
        remainder //= shape[axis]
    return tuple(values)


def _axis_group(rank: int, shape: tuple[int, ...], axis: int) -> list[int]:
    """Return ranks sharing one coordinate in a full-world mesh view."""

    coordinate = _coord(rank, shape)
    return [
        candidate
        for candidate in range(prod(shape))
        if _coord(candidate, shape)[axis] == coordinate[axis]
    ]


def rank_view(config: AdvisorConfig, dimensions: DerivedDimensions) -> list[str]:
    """Render the selected rank's coordinates and communication groups."""

    rank = config.rank
    if not 0 <= rank < config.world_size:
        return [f"rank {rank} is outside world_size={config.world_size}"]

    node = rank // config.gpus_per_node
    local_rank = rank % config.gpus_per_node
    lines = [
        f"rank {rank}: node={node}, local_rank={local_rank}",
        "mesh rank view (full-world views):",
    ]
    shapes = mesh_shapes(dimensions)
    names = mesh_shape_names()
    for mesh_name, shape in shapes.items():
        if mesh_name == "loss":
            dataloading_shape = shapes["dataloading"]
            dataloading_coord = _coord(rank, dataloading_shape)
            loss_coord = (dataloading_coord[1], dataloading_coord[2])
            loss_group = [
                candidate
                for candidate in range(config.world_size)
                if _coord(candidate, dataloading_shape)[0] == dataloading_coord[0]
                and _coord(candidate, dataloading_shape)[3] == dataloading_coord[3]
            ]
            lines.append(
                f"  loss: coord(batch,cp)={loss_coord}, "
                f"group in this pp/tp slice={loss_group}"
            )
            continue

        coordinate = _coord(rank, shape)
        axis_text = []
        for axis, axis_name in enumerate(names[mesh_name]):
            axis_text.append(f"{axis_name}={_axis_group(rank, shape, axis)}")
        lines.append(f"  {mesh_name}: coord={coordinate}; " + "; ".join(axis_text))
    return lines


def audit_config(config: AdvisorConfig) -> AuditReport:
    """Audit one configuration without throwing away independent findings."""

    report = AuditReport(config)
    positive_values = {
        "world_size": config.world_size,
        "gpus_per_node": config.gpus_per_node,
        "dp_replicate": config.dp_replicate,
        "cp": config.cp,
        "tp": config.tp,
        "pp": config.pp,
        "ep": config.ep,
        "seq_len": config.seq_len,
        "num_layers": config.num_layers,
        "num_heads": config.num_heads,
        "num_experts": config.num_experts,
    }
    for name, value in positive_values.items():
        if value < 1:
            report.add("ERROR", f"{name} must be >= 1; got {value}")

    if config.dp_shard < -1 or config.dp_shard == 0:
        report.add("ERROR", f"dp_shard must be -1 or >= 1; got {config.dp_shard}")
    if config.rank < 0 or config.rank >= config.world_size:
        report.add(
            "ERROR",
            f"rank must be in [0, {config.world_size}); got {config.rank}",
        )
    if config.spmd_backend not in {"spmd_types", "partial_dtensor"}:
        report.add("ERROR", f"unknown spmd_backend: {config.spmd_backend}")

    base_values_valid = all(value >= 1 for value in positive_values.values())
    if config.dp_shard == -1 and base_values_valid:
        denominator = config.dp_replicate * config.cp * config.tp * config.pp
        resolved_dp_shard = resolve_dp_shard(
            config.dp_replicate,
            config.dp_shard,
            config.cp,
            config.tp,
            config.pp,
            config.world_size,
        )
        report.add(
            "INFO",
            f"dp_shard=-1 -> dp_shard={resolved_dp_shard} "
            f"(world_size // {denominator})",
        )
    else:
        resolved_dp_shard = config.dp_shard

    dimensions_valid = base_values_valid and resolved_dp_shard >= 1
    dense_valid = False
    ep_valid = False
    if dimensions_valid:
        dense_product = (
            config.dp_replicate * resolved_dp_shard * config.cp * config.tp * config.pp
        )
        dense_factors = (
            config.dp_replicate,
            resolved_dp_shard,
            config.cp,
            config.tp,
            config.pp,
        )
        if dense_product == config.world_size:
            dense_valid = True
            report.add(
                "VALID",
                f"Dense world-size equation: {' x '.join(map(str, dense_factors))} "
                f"= {config.world_size}",
            )
        else:
            report.add(
                "ERROR",
                f"Dense world-size equation: "
                f"{' x '.join(map(str, dense_factors))} != {config.world_size}",
            )

        sparse_region = resolved_dp_shard * config.cp * config.tp
        if sparse_region % config.ep == 0:
            ep_valid = True
            report.add(
                "VALID",
                f"Expert region: dp_shard x cp x tp = "
                f"{resolved_dp_shard} x {config.cp} x {config.tp} = "
                f"{sparse_region}; {sparse_region} % ep({config.ep}) = 0",
            )
        else:
            report.add(
                "ERROR",
                f"Expert region {sparse_region} is not divisible by ep({config.ep})",
            )

        divisor = seq_len_divisor(config.tp, config.cp)
        if config.seq_len % divisor == 0:
            report.add(
                "VALID",
                f"Sequence length: divisor={divisor}; {config.seq_len} % {divisor} = 0",
            )
        else:
            report.add(
                "ERROR",
                f"Sequence length: divisor={divisor}; "
                f"{config.seq_len} % {divisor} = {config.seq_len % divisor}",
            )

        if dense_valid and ep_valid:
            report.dimensions = DerivedDimensions(
                config.dp_replicate,
                resolved_dp_shard,
                config.cp,
                config.tp,
                config.pp,
                config.ep,
                config.world_size,
            )
            report.shapes = mesh_shapes(report.dimensions)

    if config.pp >= 1:
        if config.num_layers % config.pp == 0:
            report.add("INFO", f"num_layers={config.num_layers} divides pp={config.pp}")
        else:
            report.add(
                "WARN",
                f"num_layers={config.num_layers} cannot be evenly split across pp={config.pp}",
            )
    report.add(
        "CHECK",
        f"Confirm num_heads={config.num_heads} is compatible with TP/CP "
        f"(tp={config.tp}, cp={config.cp})",
    )
    if config.ep >= 1:
        if config.num_experts % config.ep == 0:
            mapping = config.num_experts // config.ep
            report.add(
                "CHECK",
                f"Confirm expert mapping: {mapping} experts per EP rank",
            )
        else:
            report.add(
                "CHECK",
                f"Confirm expert-to-EP mapping: num_experts={config.num_experts} "
                f"is not divisible by ep={config.ep}",
            )

    if config.tp <= config.gpus_per_node:
        report.add(
            "INFO",
            f"TP degree {config.tp} can fit inside a {config.gpus_per_node}-GPU node",
        )
    else:
        report.add(
            "WARN",
            f"TP degree {config.tp} may cross the {config.gpus_per_node}-GPU node boundary",
        )
    if config.ep > config.gpus_per_node:
        report.add(
            "WARN",
            f"EP degree {config.ep} may require cross-node All-to-All",
        )
    else:
        report.add(
            "CHECK", "EP group placement must be checked against rank-to-node mapping"
        )
    if config.pp >= 1:
        stage_size = config.world_size // config.pp if config.world_size > 0 else 0
        report.add("INFO", f"each PP stage is assigned about {stage_size} ranks")
    report.add(
        "INFO",
        "main collectives: FSDP All-Gather/Reduce-Scatter; loss reduction; "
        "EP All-to-All; PP point-to-point",
    )
    return report


def _print_report(report: AuditReport) -> None:
    config = report.config
    print("ParallelDims configuration audit")
    print(
        f"world={config.world_size}, gpus_per_node={config.gpus_per_node}, "
        f"backend={config.spmd_backend}, rank={config.rank}"
    )
    for finding in report.findings:
        print(f"[{finding.level}] {finding.message}")

    if report.dimensions is None:
        return

    dimensions = report.dimensions
    print("Derived dimensions:")
    print(f"  batch = {dimensions.batch}")
    print(f"  loss  = {dimensions.loss}")
    print(f"  fsdp  = {dimensions.fsdp}")
    print(f"  efsdp = {dimensions.efsdp}")

    labels = mesh_shape_names()
    display_names = {
        "dataloading": "dataloading",
        "loss": "loss",
        "spmd_types_storage": "dense_storage",
        "spmd_types_fwd_bwd": "dense_fwd_bwd",
        "partial_dtensor_dense": "partial_dtensor",
        "sparse": "sparse",
    }
    print("Mesh views:")
    for name, shape in report.shapes.items():
        print(f"  {display_names[name]:16s} = {named_shape(shape, labels[name])}")

    print("Rank visualization:")
    for line in rank_view(config, dimensions):
        print(line)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-size", type=int, default=32)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--dp-replicate", type=int, default=2)
    parser.add_argument("--dp-shard", type=int, default=2)
    parser.add_argument("--cp", type=int, default=2)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--pp", type=int, default=2)
    parser.add_argument("--ep", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument(
        "--spmd-backend",
        choices=("spmd_types", "partial_dtensor"),
        default="spmd_types",
    )
    parser.add_argument("--rank", type=int, default=0)
    return parser


def config_from_args(args: argparse.Namespace) -> AdvisorConfig:
    return AdvisorConfig(
        world_size=args.world_size,
        gpus_per_node=args.gpus_per_node,
        dp_replicate=args.dp_replicate,
        dp_shard=args.dp_shard,
        cp=args.cp,
        tp=args.tp,
        pp=args.pp,
        ep=args.ep,
        seq_len=args.seq_len,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_experts=args.num_experts,
        spmd_backend=args.spmd_backend,
        rank=args.rank,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = audit_config(config_from_args(args))
    _print_report(report)
    return 1 if report.has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
