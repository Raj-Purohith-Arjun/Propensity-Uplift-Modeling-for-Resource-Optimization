"""
Unit tests for resource allocator.
"""
import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.optimization.resource_allocator import ResourceAllocator


@pytest.fixture
def score_df():
    rng = np.random.default_rng(0)
    n = 1000
    return pd.DataFrame({
        "customer_id": np.arange(1, n + 1),
        "propensity_score": rng.uniform(0.01, 0.99, size=n),
        "uplift_score": rng.uniform(-0.05, 0.25, size=n),
    })


class TestResourceAllocator:
    def test_budget_respected(self, score_df):
        allocator = ResourceAllocator(budget_fraction=0.30)
        result = allocator.allocate(score_df)
        n = len(score_df)
        expected = int(np.ceil(n * 0.30))
        assert result["is_targeted"].sum() == expected

    def test_columns_added(self, score_df):
        allocator = ResourceAllocator()
        result = allocator.allocate(score_df)
        for col in ["composite_score", "rank", "is_targeted", "value_tier"]:
            assert col in result.columns

    def test_value_tiers(self, score_df):
        allocator = ResourceAllocator()
        result = allocator.allocate(score_df)
        assert set(result["value_tier"].unique()).issubset({"high", "medium", "low"})

    def test_summary_keys(self, score_df):
        allocator = ResourceAllocator()
        result = allocator.allocate(score_df)
        summary = allocator.allocation_summary(result)
        required = [
            "n_total", "n_targeted", "targeting_rate_pct",
            "low_value_effort_reduction_pct",
        ]
        for k in required:
            assert k in summary

    def test_invalid_budget_raises(self):
        with pytest.raises(ValueError):
            ResourceAllocator(budget_fraction=0.0)

    def test_invalid_weight_raises(self):
        with pytest.raises(ValueError):
            ResourceAllocator(propensity_weight=1.5)

    def test_composite_score_range(self, score_df):
        allocator = ResourceAllocator()
        result = allocator.allocate(score_df)
        assert result["composite_score"].min() >= 0.0
        assert result["composite_score"].max() <= 1.0

    def test_full_budget(self, score_df):
        allocator = ResourceAllocator(budget_fraction=1.0)
        result = allocator.allocate(score_df)
        assert result["is_targeted"].all()
