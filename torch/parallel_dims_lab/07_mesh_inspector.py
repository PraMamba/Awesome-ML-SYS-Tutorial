"""Inspect the source-aligned mesh shapes for the fixed 32-rank example."""

from __future__ import annotations

from parallel_dims_common import (
    DerivedDimensions,
    derive_dimensions,
    mesh_shape_names,
    mesh_shapes,
    named_shape,
    product,
)


def fixed_dimensions() -> DerivedDimensions:
    """Return the configuration used throughout the study plan."""

    return derive_dimensions(
        dp_replicate=2,
        dp_shard=2,
        cp=2,
        tp=2,
        pp=2,
        ep=4,
        world_size=32,
    )


def validate_mesh_products(
    dimensions: DerivedDimensions,
) -> dict[str, bool]:
    """Validate full views against world size and loss against its submesh."""

    shapes = mesh_shapes(dimensions)
    expected = {
        name: dimensions.loss if name == "loss" else dimensions.world_size
        for name in shapes
    }
    return {name: product(shape) == expected[name] for name, shape in shapes.items()}


def run_demo() -> None:
    dimensions = fixed_dimensions()
    shapes = mesh_shapes(dimensions)
    labels = mesh_shape_names()
    checks = validate_mesh_products(dimensions)

    print("Derived dimensions:")
    print(f"  batch = {dimensions.batch}")
    print(f"  loss  = {dimensions.loss}")
    print(f"  fsdp  = {dimensions.fsdp}")
    print(f"  efsdp = {dimensions.efsdp}")
    print("Mesh views:")
    for name, shape in shapes.items():
        expected = dimensions.loss if name == "loss" else dimensions.world_size
        assert checks[name]
        print(
            f"  {name:22s} = {named_shape(shape, labels[name])}; "
            f"product={product(shape)} (expected {expected})"
        )
    print("all mesh products validated")
    print("loss is a (batch, cp) submesh, so its product is 8 rather than 32")


if __name__ == "__main__":
    run_demo()
