"""
Unit tests for the data pipeline module.
"""
import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from configs.config import ALL_FEATURES, TARGET_COLUMN, TREATMENT_COLUMN
from src.data.data_pipeline import (
    encode_categoricals,
    generate_synthetic_dataset,
    preprocess_features,
    split_dataset,
)


class TestGenerateSyntheticDataset:
    def test_shape(self):
        df = generate_synthetic_dataset(n_samples=1000)
        assert len(df) == 1000

    def test_columns_present(self):
        df = generate_synthetic_dataset(n_samples=500)
        for col in ALL_FEATURES + [TARGET_COLUMN, TREATMENT_COLUMN, "customer_id"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_treatment_ratio(self):
        df = generate_synthetic_dataset(n_samples=10000, treatment_ratio=0.5)
        ratio = df[TREATMENT_COLUMN].mean()
        assert 0.45 <= ratio <= 0.55

    def test_conversion_values(self):
        df = generate_synthetic_dataset(n_samples=500)
        assert set(df[TARGET_COLUMN].unique()).issubset({0, 1})
        assert set(df[TREATMENT_COLUMN].unique()).issubset({0, 1})

    def test_no_nulls(self):
        df = generate_synthetic_dataset(n_samples=500)
        assert df[ALL_FEATURES].isnull().sum().sum() == 0

    def test_reproducibility(self):
        df1 = generate_synthetic_dataset(n_samples=200, random_state=99)
        df2 = generate_synthetic_dataset(n_samples=200, random_state=99)
        pd.testing.assert_frame_equal(df1, df2)


class TestPreprocessFeatures:
    def setup_method(self):
        self.df = generate_synthetic_dataset(n_samples=500)

    def test_output_shape(self):
        X, scaler = preprocess_features(self.df)
        assert X.shape[0] == 500
        assert X.shape[1] == len(ALL_FEATURES)

    def test_no_nulls_after_scaling(self):
        X, _ = preprocess_features(self.df)
        assert X.isnull().sum().sum() == 0

    def test_inference_mode(self):
        X_train, scaler = preprocess_features(self.df.head(400))
        X_test, _ = preprocess_features(self.df.tail(100), scaler=scaler, fit_scaler=False)
        assert X_test.shape[0] == 100


class TestSplitDataset:
    def setup_method(self):
        self.df = generate_synthetic_dataset(n_samples=5000)

    def test_no_overlap(self):
        X_train, X_val, X_test, *_ = split_dataset(self.df)
        idx_train = set(X_train.index)
        idx_val = set(X_val.index)
        idx_test = set(X_test.index)
        assert not idx_train & idx_val
        assert not idx_train & idx_test
        assert not idx_val & idx_test

    def test_sizes(self):
        X_train, X_val, X_test, *_ = split_dataset(self.df)
        total = len(X_train) + len(X_val) + len(X_test)
        assert total == len(self.df)

    def test_treatment_preserved(self):
        _, _, _, _, _, _, t_train, t_val, t_test = split_dataset(self.df)
        for t in (t_train, t_val, t_test):
            assert set(t.unique()).issubset({0, 1})


class TestEncodeCategoricals:
    def test_encoding(self):
        df = pd.DataFrame({"region": ["North", "South", "East", "North"]})
        df_enc, encoders = encode_categoricals(df, categorical_cols=["region"])
        assert "region_encoded" in df_enc.columns
        assert "region" in encoders

    def test_missing_column_skipped(self):
        df = pd.DataFrame({"age": [25, 30]})
        df_enc, encoders = encode_categoricals(df, categorical_cols=["region"])
        assert len(encoders) == 0
