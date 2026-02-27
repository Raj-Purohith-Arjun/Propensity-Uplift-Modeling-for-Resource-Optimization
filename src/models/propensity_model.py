"""
Propensity model: gradient-boosted classifier (XGBoost) that estimates the
probability that a customer will convert, independent of treatment assignment.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.config import ALL_FEATURES, PROPENSITY_MODEL_PARAMS, RANDOM_STATE

logger = logging.getLogger(__name__)


class PropensityModel:
    """
    XGBoost-based propensity scorer with optional Platt scaling calibration.

    Attributes
    ----------
    params : dict
        XGBoost hyperparameters.
    model : XGBClassifier
        Underlying gradient-boosted classifier.
    _platt_scaler : LogisticRegression | None
        Sigmoid calibration (Platt scaling) fitted after training.
    feature_names : list[str]
        Column names the model was trained on.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self.params = params if params is not None else dict(PROPENSITY_MODEL_PARAMS)
        self._build_model()
        self._platt_scaler: Optional[LogisticRegression] = None
        self.feature_names: list = []

    def _build_model(self) -> None:
        xgb_params = {k: v for k, v in self.params.items()
                      if k != "early_stopping_rounds"}
        self.model = XGBClassifier(
            **xgb_params,
            use_label_encoder=False,
            verbosity=0,
        )

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        calibrate: bool = True,
    ) -> "PropensityModel":
        """
        Train and optionally calibrate the propensity model.

        Parameters
        ----------
        X_train, y_train : training features and labels.
        X_val, y_val : optional validation set for early stopping.
        calibrate : bool
            Apply sigmoid calibration after training.
        """
        self.feature_names = list(X_train.columns)

        fit_kwargs: Dict[str, Any] = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["verbose"] = False

        self.model.fit(X_train, y_train, **fit_kwargs)

        if calibrate:
            # Platt scaling: fit a logistic regression on raw XGBoost log-odds
            raw_scores = self.model.predict_proba(X_train)[:, 1].reshape(-1, 1)
            self._platt_scaler = LogisticRegression(C=1.0, solver="lbfgs")
            self._platt_scaler.fit(raw_scores, y_train)
            logger.info("Propensity model trained and calibrated (Platt scaling).")
        else:
            logger.info("Propensity model trained (uncalibrated).")

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated conversion probability for each record."""
        raw_scores = self.model.predict_proba(X)[:, 1]
        if self._platt_scaler is not None:
            return self._platt_scaler.predict_proba(raw_scores.reshape(-1, 1))[:, 1]
        return raw_scores

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, float]:
        """
        Compute standard classification metrics on the test set.

        Returns
        -------
        dict with keys: roc_auc, average_precision, brier_score
        """
        scores = self.predict_proba(X_test)
        metrics = {
            "roc_auc": roc_auc_score(y_test, scores),
            "average_precision": average_precision_score(y_test, scores),
            "brier_score": brier_score_loss(y_test, scores),
        }
        logger.info(
            "Propensity evaluation — ROC-AUC: %.4f | Avg Precision: %.4f | Brier: %.4f",
            metrics["roc_auc"],
            metrics["average_precision"],
            metrics["brier_score"],
        )
        return metrics

    def get_feature_importance(self) -> pd.Series:
        """Return feature importances sorted descending."""
        importance = self.model.feature_importances_
        return pd.Series(importance, index=self.feature_names).sort_values(ascending=False)

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
    ) -> Dict[str, float]:
        """
        Stratified k-fold cross-validation returning mean ± std of ROC-AUC.
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        auc_scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]

            fold_model = PropensityModel(params=dict(self.params))
            fold_model.fit(X_tr, y_tr, X_vl, y_vl, calibrate=False)
            score = roc_auc_score(y_vl, fold_model.predict_proba(X_vl))
            auc_scores.append(score)
            logger.debug("Fold %d — AUC: %.4f", fold + 1, score)

        result = {
            "mean_roc_auc": float(np.mean(auc_scores)),
            "std_roc_auc": float(np.std(auc_scores)),
        }
        logger.info(
            "Cross-validation ROC-AUC: %.4f ± %.4f",
            result["mean_roc_auc"],
            result["std_roc_auc"],
        )
        return result
