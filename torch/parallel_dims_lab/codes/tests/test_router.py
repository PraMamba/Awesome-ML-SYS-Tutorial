"""MoE routing and load-statistics tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


LAB_DIR = Path(__file__).resolve().parents[1]
if str(LAB_DIR) not in sys.path:
    sys.path.insert(0, str(LAB_DIR))


def load_router():
    spec = importlib.util.spec_from_file_location(
        "router_for_tests", LAB_DIR / "08_moe_router_sim.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


router = load_router()


def test_default_route_buckets_and_restores_order() -> None:
    assignments = router.default_expert_assignments(16, 8)
    result = router.simulate_routing(assignments, num_experts=8, ep=4)
    assert result["rank_loads"] == [4, 4, 4, 4]
    assert result["buckets"] == {
        0: [0, 1, 8, 9],
        1: [2, 3, 10, 11],
        2: [4, 5, 12, 13],
        3: [6, 7, 14, 15],
    }
    assert result["token_to_ep_rank"] == [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
    ]
    assert len(result["restored_values"]) == 16


def test_unbalanced_route_reports_spread() -> None:
    assignments = [0, 0, 0, 1, 1, 2, 2, 2, 4, 4, 5, 5, 6, 6, 7, 7]
    result = router.simulate_routing(assignments, num_experts=8, ep=4)
    assert result["rank_loads"] == [5, 3, 4, 4]
    assert router.load_statistics(result["rank_loads"])["spread"] == 2
