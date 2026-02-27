"""
Unit tests for propensity and uplift models.
"""
import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.data_pipeline import generate_synthetic_dataset, preprocess_features, split_dataset
from src.models.propensity_model import PropensityModel
from src.models.uplift_model import TLearner, SLearner, UpliftModelEnsemble


@pytest.fixture(scope="module")
def prepared_data():
    df = generate_synthetic_dataset(n_samples=3000, random_state=0)
    X_train, X_val, X_test, y_train, y_val, y_test, t_train, t_val, t_test = split_dataset(df)
    X_train_s, scaler = preprocess_features(X_train)
    X_val_s, _ = preprocess_features(X_val, scaler=scaler, fit_scaler=False)
    X_test_s, _ = preprocess_features(X_test, scaler=scaler, fit_scaler=False)
    return X_train_s, X_val_s, X_test_s, y_train, y_val, y_test, t_train, t_val, t_test


class TestPropensityModel:
    def test_fit_predict(self, prepared_data):
        X_train, X_val, X_test, y_train, y_val, y_test, *_ = prepared_data
        model = PropensityModel()
        model.fit(X_train, y_train, X_val, y_val)
        scores = model.predict_proba(X_test)
        assert len(scores) == len(X_test)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_evaluate_metrics(self, prepared_data):
        X_train, X_val, X_test, y_train, y_val, y_test, *_ = prepared_data
        model = PropensityModel()
        model.fit(X_train, y_train, X_val, y_val)
        metrics = model.evaluate(X_test, y_test)
        assert "roc_auc" in metrics
        assert "average_precision" in metrics
        assert "brier_score" in metrics
        assert 0.5 <= metrics["roc_auc"] <= 1.0

    def test_feature_importance(self, prepared_data):
        X_train, X_val, X_test, y_train, y_val, y_test, *_ = prepared_data
        model = PropensityModel()
        model.fit(X_train, y_train, X_val, y_val)
        imp = model.get_feature_importance()
        assert len(imp) == X_train.shape[1]
        assert imp.index[0] == imp.sort_values(ascending=False).index[0]

    def test_uncalibrated(self, prepared_data):
        X_train, X_val, X_test, y_train, y_val, y_test, *_ = prepared_data
        model = PropensityModel()
        model.fit(X_train, y_train, calibrate=False)
        scores = model.predict_proba(X_test)
        assert len(scores) == len(X_test)


class TestTLearner:
    def test_fit_predict(self, prepared_data):
        X_train, _, X_test, y_train, _, y_test, t_train, _, t_test = prepared_data
        model = TLearner()
        model.fit(X_train, y_train, t_train)
        uplift = model.predict_uplift(X_test)
        assert len(uplift) == len(X_test)
        assert uplift.min() >= -1.0
        assert uplift.max() <= 1.0

    def test_feature_importance(self, prepared_data):
        X_train, _, _, y_train, _, _, t_train, _, _ = prepared_data
        model = TLearner()
        model.fit(X_train, y_train, t_train)
        df = model.get_feature_importance()
        assert "avg" in df.columns
        assert len(df) == X_train.shape[1]


class TestSLearner:
    def test_fit_predict(self, prepared_data):
        X_train, _, X_test, y_train, _, y_test, t_train, _, t_test = prepared_data
        model = SLearner()
        model.fit(X_train, y_train, t_train)
        uplift = model.predict_uplift(X_test)
        assert len(uplift) == len(X_test)


class TestUpliftModelEnsemble:
    def test_ensemble_predict(self, prepared_data):
        X_train, _, X_test, y_train, _, y_test, t_train, _, t_test = prepared_data
        ensemble = UpliftModelEnsemble()
        ensemble.fit(X_train, y_train, t_train)
        uplift = ensemble.predict_uplift(X_test)
        assert len(uplift) == len(X_test)
        assert not np.isnan(uplift).any()

    def test_weights_sum_to_one(self):
        ensemble = UpliftModelEnsemble(weights=(0.7, 0.3))
        assert abs(sum(ensemble.weights) - 1.0) < 1e-9
