"""
Uplift modeling using T-Learner and S-Learner meta-learner approaches.

Both approaches rely on gradient-boosted models (XGBoost) and estimate the
individual treatment effect (ITE) — i.e. how much a customer's conversion
probability changes as a direct result of treatment.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.config import ALL_FEATURES, RANDOM_STATE, UPLIFT_MODEL_PARAMS

logger = logging.getLogger(__name__)


class TLearner:
    """
    Two-model (T-Learner) uplift estimator.

    Trains separate response models μ₁(x) and μ₀(x) for treatment and control
    populations, then estimates the Conditional Average Treatment Effect (CATE)
    as τ(x) = μ₁(x) − μ₀(x).

    References
    ----------
    Künzel et al. (2019) "Metalearners for Estimating Heterogeneous Treatment
    Effects Using Machine Learning".
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self.params = params if params is not None else dict(UPLIFT_MODEL_PARAMS)
        self._control_model: Optional[XGBClassifier] = None
        self._treatment_model: Optional[XGBClassifier] = None
        self.feature_names: list = []

    def _make_model(self) -> XGBClassifier:
        params = {k: v for k, v in self.params.items() if k != "eval_metric"}
        return XGBClassifier(
            **params,
            use_label_encoder=False,
            verbosity=0,
            eval_metric="logloss",
        )

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treatment: pd.Series,
    ) -> "TLearner":
        """
        Fit separate models on control and treatment subsets.

        Parameters
        ----------
        X : Feature matrix.
        y : Binary conversion target.
        treatment : Binary treatment indicator (1 = treated, 0 = control).
        """
        self.feature_names = list(X.columns)

        mask_treat = treatment == 1
        mask_ctrl = treatment == 0

        self._treatment_model = self._make_model()
        self._treatment_model.fit(X[mask_treat], y[mask_treat])

        self._control_model = self._make_model()
        self._control_model.fit(X[mask_ctrl], y[mask_ctrl])

        treat_size = mask_treat.sum()
        ctrl_size = mask_ctrl.sum()
        logger.info(
            "T-Learner fitted — treatment n=%d, control n=%d", treat_size, ctrl_size
        )
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        """
        Estimate CATE τ(x) = μ₁(x) − μ₀(x) for each observation.

        Returns
        -------
        np.ndarray of shape (n_samples,) with uplift scores in [−1, 1].
        """
        p_treat = self._treatment_model.predict_proba(X)[:, 1]
        p_ctrl = self._control_model.predict_proba(X)[:, 1]
        return p_treat - p_ctrl

    def get_feature_importance(self) -> pd.DataFrame:
        """Return average feature importance across both models."""
        treat_imp = pd.Series(
            self._treatment_model.feature_importances_, index=self.feature_names
        )
        ctrl_imp = pd.Series(
            self._control_model.feature_importances_, index=self.feature_names
        )
        df = pd.DataFrame({"treatment": treat_imp, "control": ctrl_imp})
        df["avg"] = df.mean(axis=1)
        return df.sort_values("avg", ascending=False)


class SLearner:
    """
    Single-model (S-Learner) uplift estimator.

    Trains one model on the full dataset with treatment indicator as a feature.
    CATE is estimated by predicting with ``treatment=1`` minus ``treatment=0``.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self.params = params if params is not None else dict(UPLIFT_MODEL_PARAMS)
        self._model: Optional[XGBClassifier] = None
        self.feature_names: list = []
        self._treatment_col = "treatment"

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treatment: pd.Series,
    ) -> "SLearner":
        """
        Fit a single model on (X, treatment) → y.
        """
        X_aug = X.copy()
        X_aug[self._treatment_col] = treatment.values
        self.feature_names = list(X_aug.columns)

        s_params = {k: v for k, v in self.params.items() if k != "eval_metric"}
        self._model = XGBClassifier(
            **s_params,
            use_label_encoder=False,
            verbosity=0,
            eval_metric="logloss",
        )
        self._model.fit(X_aug, y)
        logger.info("S-Learner fitted on %d samples.", len(X))
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        """
        Estimate CATE as difference between predicted outcomes under
        treatment=1 and treatment=0.
        """
        X_treat = X.copy()
        X_treat[self._treatment_col] = 1
        X_ctrl = X.copy()
        X_ctrl[self._treatment_col] = 0

        p_treat = self._model.predict_proba(X_treat)[:, 1]
        p_ctrl = self._model.predict_proba(X_ctrl)[:, 1]
        return p_treat - p_ctrl


class UpliftModelEnsemble:
    """
    Ensemble of T-Learner and S-Learner predictions via simple averaging.
    Reduces variance and provides more stable CATE estimates.
    """

    def __init__(
        self,
        t_learner_params: Optional[Dict[str, Any]] = None,
        s_learner_params: Optional[Dict[str, Any]] = None,
        weights: Optional[tuple] = None,
    ) -> None:
        self.t_learner = TLearner(params=t_learner_params)
        self.s_learner = SLearner(params=s_learner_params)
        self.weights = weights if weights is not None else (0.6, 0.4)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        treatment: pd.Series,
    ) -> "UpliftModelEnsemble":
        self.t_learner.fit(X, y, treatment)
        self.s_learner.fit(X, y, treatment)
        logger.info("Uplift ensemble (T + S Learner) fitted.")
        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        """Weighted average of T-Learner and S-Learner CATE estimates."""
        w_t, w_s = self.weights
        tau_t = self.t_learner.predict_uplift(X)
        tau_s = self.s_learner.predict_uplift(X)
        return w_t * tau_t + w_s * tau_s
