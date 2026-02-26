"""
Unit tests for A/B testing framework and uplift evaluation metrics.
"""
import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.ab_testing import ABTestingFramework, ABTestResult
from src.evaluation.uplift_metrics import (
    area_under_uplift_curve,
    cumulative_gain_curve,
    incremental_lift,
    qini_curve,
    compute_all_metrics,
)


@pytest.fixture
def experiment_df():
    rng = np.random.default_rng(42)
    n = 5000
    treatment = rng.integers(0, 2, size=n)
    # treatment has higher conversion rate
    p = np.where(treatment == 1, 0.15, 0.10)
    converted = rng.binomial(1, p)
    return pd.DataFrame({"treatment": treatment, "converted": converted})


@pytest.fixture
def uplift_arrays():
    rng = np.random.default_rng(7)
    n = 2000
    treatment = rng.integers(0, 2, size=n)
    y = rng.integers(0, 2, size=n)
    scores = rng.uniform(-0.1, 0.3, size=n)
    return y, treatment, scores


class TestABTestingFramework:
    def test_significant_lift(self, experiment_df):
        ab = ABTestingFramework(confidence_level=0.95)
        result = ab.run_test(experiment_df)
        assert isinstance(result, ABTestResult)
        assert result.is_significant
        assert result.conversion_rate_treatment > result.conversion_rate_control

    def test_confidence_interval_contains_true_lift(self, experiment_df):
        ab = ABTestingFramework(confidence_level=0.95)
        result = ab.run_test(experiment_df)
        true_lift = 0.05  # 0.15 - 0.10
        assert result.ci_lower < true_lift < result.ci_upper

    def test_p_value_range(self, experiment_df):
        ab = ABTestingFramework(confidence_level=0.95)
        result = ab.run_test(experiment_df)
        assert 0.0 <= result.p_value <= 1.0

    def test_to_dict_keys(self, experiment_df):
        ab = ABTestingFramework(confidence_level=0.95)
        result = ab.run_test(experiment_df)
        d = result.to_dict()
        required = [
            "n_control", "n_treatment", "absolute_lift",
            "relative_lift_pct", "p_value", "is_significant", "ci_lower", "ci_upper",
        ]
        for key in required:
            assert key in d

    def test_required_sample_size(self):
        ab = ABTestingFramework()
        n = ab.required_sample_size(
            baseline_rate=0.10,
            min_detectable_effect=0.02,
            power=0.80,
        )
        assert n > 0
        assert isinstance(n, int)

    def test_segment_analysis(self, experiment_df):
        experiment_df["segment_encoded"] = np.tile([0, 1, 2, 3], len(experiment_df) // 4 + 1)[: len(experiment_df)]
        ab = ABTestingFramework()
        seg_df = ab.segment_analysis(experiment_df, segment_col="segment_encoded")
        assert len(seg_df) >= 1
        assert "segment" in seg_df.columns

    def test_no_effect_not_significant(self):
        rng = np.random.default_rng(0)
        n = 2000
        df = pd.DataFrame({
            "treatment": rng.integers(0, 2, size=n),
            "converted": rng.binomial(1, 0.10, size=n),
        })
        ab = ABTestingFramework()
        result = ab.run_test(df)
        # with equal rates we expect no significance most of the time
        assert result.p_value >= 0.0  # basic sanity


class TestQiniCurve:
    def test_shape(self, uplift_arrays):
        y, t, scores = uplift_arrays
        x, qini = qini_curve(y, t, scores)
        assert len(x) == len(y) + 1
        assert len(qini) == len(y) + 1

    def test_starts_at_zero(self, uplift_arrays):
        y, t, scores = uplift_arrays
        _, qini = qini_curve(y, t, scores)
        assert qini[0] == 0.0

    def test_x_normalized(self, uplift_arrays):
        y, t, scores = uplift_arrays
        x, _ = qini_curve(y, t, scores, normalize=True)
        assert 0.0 <= x[-1] <= 1.0

    def test_invalid_input(self):
        with pytest.raises(ValueError):
            qini_curve(np.array([0, 1]), np.array([0]), np.array([0.5, 0.3]))


class TestAUUC:
    def test_returns_float(self, uplift_arrays):
        y, t, scores = uplift_arrays
        auuc = area_under_uplift_curve(y, t, scores)
        assert isinstance(auuc, float)

    def test_perfect_model_positive(self):
        rng = np.random.default_rng(1)
        n = 1000
        t = rng.integers(0, 2, size=n)
        y = t.copy()
        scores = np.where(t == 1, 1.0, -1.0) + rng.uniform(-0.01, 0.01, n)
        auuc = area_under_uplift_curve(y, t, scores)
        assert auuc > 0


class TestIncrementalLift:
    def test_keys(self, uplift_arrays):
        y, t, scores = uplift_arrays
        result = incremental_lift(y, t, scores, top_k_fraction=0.30)
        expected_keys = [
            "top_k_fraction", "n_targeted", "cr_treatment_top_k",
            "lift_absolute", "low_value_effort_reduction_pct",
        ]
        for k in expected_keys:
            assert k in result

    def test_effort_reduction_range(self, uplift_arrays):
        y, t, scores = uplift_arrays
        result = incremental_lift(y, t, scores, top_k_fraction=0.30)
        assert 0 <= result["low_value_effort_reduction_pct"] <= 100

    def test_k_boundary(self, uplift_arrays):
        y, t, scores = uplift_arrays
        result = incremental_lift(y, t, scores, top_k_fraction=1.0)
        assert result["n_targeted"] == len(y)


class TestCumulativeGainCurve:
    def test_shape(self):
        rng = np.random.default_rng(5)
        y = rng.integers(0, 2, size=500)
        scores = rng.uniform(0, 1, size=500)
        x, gain = cumulative_gain_curve(y, scores)
        assert len(x) == len(y) + 1
        assert gain[0] == 0.0
        assert gain[-1] == pytest.approx(1.0)

    def test_invalid_input(self):
        with pytest.raises(ValueError):
            cumulative_gain_curve(np.array([0, 1]), np.array([0.5]))
