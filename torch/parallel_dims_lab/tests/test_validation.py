"""Validation cases A-E for the ParallelDims lab."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


LAB_DIR = Path(__file__).resolve().parents[1]
if str(LAB_DIR) not in sys.path:
    sys.path.insert(0, str(LAB_DIR))


def load_script(filename: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, LAB_DIR / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


lite = load_script("06_parallel_dims_lite.py", "parallel_dims_lite_for_tests")
advisor = load_script("10_parallel_config_advisor.py", "advisor_for_tests")


def test_case_a_valid_32_rank_configuration() -> None:
    dims = lite.ParallelDimsLite(2, 2, 2, 2, 2, 4, 32)
    assert dims.derived() == {"batch": 4, "loss": 8, "fsdp": 4, "efsdp": 2}


def test_case_b_dense_product_must_match_world_size() -> None:
    with pytest.raises(AssertionError, match="Invalid parallel dims"):
        lite.ParallelDimsLite(2, 2, 2, 2, 1, 4, 32)
    report = advisor.audit_config(advisor.AdvisorConfig(pp=1))
    assert report.has_errors
    assert report.dimensions is None


def test_case_c_ep_must_divide_sparse_region() -> None:
    with pytest.raises(ValueError, match="expert_parallel_degree"):
        lite.ParallelDimsLite(2, 2, 2, 2, 2, 3, 32)
    report = advisor.audit_config(advisor.AdvisorConfig(ep=3))
    assert report.has_errors
    assert report.dimensions is None


def test_case_d_dp_shard_minus_one_is_derived() -> None:
    dims = lite.ParallelDimsLite(2, -1, 2, 2, 2, 4, 32)
    assert dims.dp_shard == 2


def test_case_e_sequence_length_is_a_separate_hard_check() -> None:
    report = advisor.audit_config(advisor.AdvisorConfig(seq_len=8193))
    assert report.has_errors
    assert any(
        finding.level == "ERROR" and "Sequence length" in finding.message
        for finding in report.findings
    )
