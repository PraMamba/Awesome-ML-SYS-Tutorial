"""Mesh shape and singleton-axis tests, including case F."""

from __future__ import annotations

import importlib.util
import sys
from math import prod
from pathlib import Path


LAB_DIR = Path(__file__).resolve().parents[1]
if str(LAB_DIR) not in sys.path:
    sys.path.insert(0, str(LAB_DIR))

from parallel_dims_common import DerivedDimensions, mesh_shapes  # noqa: E402


def load_inspector():
    spec = importlib.util.spec_from_file_location(
        "mesh_inspector_for_tests", LAB_DIR / "07_mesh_inspector.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


inspector = load_inspector()


def test_fixed_mesh_views_have_source_aligned_shapes() -> None:
    dimensions = inspector.fixed_dimensions()
    shapes = mesh_shapes(dimensions)
    assert shapes == {
        "dataloading": (2, 4, 2, 2),
        "loss": (8,),
        "spmd_types_storage": (2, 2, 2, 2, 2),
        "spmd_types_fwd_bwd": (2, 4, 2, 2),
        "partial_dtensor_dense": (2, 2, 4, 2),
        "sparse": (2, 2, 2, 4),
    }
    assert inspector.validate_mesh_products(dimensions) == {
        name: True for name in shapes
    }


def test_case_f_singleton_axes_remain_valid() -> None:
    dimensions = DerivedDimensions(1, 1, 1, 1, 1, 1, 1)
    shapes = mesh_shapes(dimensions)
    assert all(prod(shape) == 1 for shape in shapes.values())
    assert shapes["sparse"] == (1, 1, 1, 1)
