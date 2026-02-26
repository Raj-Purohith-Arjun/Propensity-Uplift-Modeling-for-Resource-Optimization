"""
Data loading and preprocessing utilities.
Handles both SQL-backed production data and in-memory simulation for testing.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from configs.config import (
    ALL_FEATURES,
    BEHAVIORAL_FEATURES,
    DEMOGRAPHIC_FEATURES,
    OPERATIONAL_FEATURES,
    PRODUCT_FEATURES,
    RANDOM_STATE,
    TARGET_COLUMN,
    TRAIN_RATIO,
    TREATMENT_COLUMN,
    VALIDATION_RATIO,
)

logger = logging.getLogger(__name__)


def generate_synthetic_dataset(
    n_samples: int = 50000,
    treatment_ratio: float = 0.50,
    base_conversion_rate: float = 0.10,
    treatment_lift: float = 0.18,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Generate a realistic synthetic dataset that mimics semiconductor / advanced
    manufacturing customer behaviour at Applied Materials scale.

    Parameters
    ----------
    n_samples : int
        Total number of customer records.
    treatment_ratio : float
        Fraction of customers assigned to the treatment group.
    base_conversion_rate : float
        Baseline conversion probability in the control group.
    treatment_lift : float
        Incremental conversion lift from treatment (absolute).
    random_state : int
        Reproducibility seed.

    Returns
    -------
    pd.DataFrame
        Dataset with feature columns, treatment flag, and conversion target.
    """
    rng = np.random.default_rng(random_state)

    n_treatment = int(n_samples * treatment_ratio)
    n_control = n_samples - n_treatment
    treatment = np.concatenate(
        [np.ones(n_treatment, dtype=int), np.zeros(n_control, dtype=int)]
    )

    # --- demographic features ---
    age = rng.integers(25, 65, size=n_samples)
    tenure_months = rng.exponential(scale=24, size=n_samples).clip(1, 120)
    region_encoded = rng.integers(0, 8, size=n_samples)
    segment_encoded = rng.integers(0, 4, size=n_samples)

    # --- behavioural features (correlated with conversion) ---
    frequency_base = rng.gamma(shape=2.0, scale=3.0, size=n_samples)
    num_interactions_90d = (frequency_base * 5).astype(int).clip(0, 100)
    num_interactions_30d = (num_interactions_90d * rng.uniform(0.2, 0.5, size=n_samples)).astype(int)
    avg_spend_90d = rng.exponential(scale=500, size=n_samples).clip(0, 5000)
    avg_spend_30d = avg_spend_90d * rng.uniform(0.15, 0.45, size=n_samples)
    recency_days = rng.integers(1, 180, size=n_samples)
    frequency_score = np.digitize(
        num_interactions_90d,
        bins=np.percentile(num_interactions_90d, [20, 40, 60, 80]),
    ).clip(1, 5)
    monetary_score = np.digitize(
        avg_spend_90d,
        bins=np.percentile(avg_spend_90d, [20, 40, 60, 80]),
    ).clip(1, 5)

    # --- product features ---
    num_products_owned = rng.integers(1, 10, size=n_samples)
    product_diversity_score = (num_products_owned / 10 + rng.uniform(-0.05, 0.05, size=n_samples)).clip(0, 1)
    cross_sell_ratio = rng.beta(a=2, b=5, size=n_samples)

    # --- operational features (negatively correlated with value) ---
    support_tickets_90d = rng.poisson(lam=1.5, size=n_samples)
    escalation_rate = rng.beta(a=1, b=9, size=n_samples)
    resolution_time_avg = rng.exponential(scale=24, size=n_samples).clip(0, 240)

    # --- build propensity score for realistic conversion ---
    log_odds = (
        -2.5
        + 0.02 * (age - 40)
        + 0.01 * tenure_months
        - 0.003 * recency_days
        + 0.05 * frequency_score
        + 0.04 * monetary_score
        + 0.002 * avg_spend_90d / 100
        + 0.10 * num_products_owned
        + 0.30 * product_diversity_score
        - 0.05 * support_tickets_90d
        - 0.50 * escalation_rate
    )
    propensity = 1.0 / (1.0 + np.exp(-log_odds))
    propensity = propensity.clip(0.01, 0.99)

    # control group converts at base rate driven by propensity
    p_control = (propensity * base_conversion_rate / propensity.mean()).clip(0.01, 0.99)
    # treatment group gets additional absolute lift
    p_treatment = (p_control + treatment_lift).clip(0.01, 0.99)

    conversion_prob = np.where(treatment == 1, p_treatment, p_control)
    converted = rng.binomial(n=1, p=conversion_prob)

    df = pd.DataFrame(
        {
            "customer_id": np.arange(1, n_samples + 1),
            "age": age,
            "tenure_months": tenure_months.round(1),
            "region_encoded": region_encoded,
            "segment_encoded": segment_encoded,
            "num_interactions_30d": num_interactions_30d,
            "num_interactions_90d": num_interactions_90d,
            "avg_spend_30d": avg_spend_30d.round(2),
            "avg_spend_90d": avg_spend_90d.round(2),
            "recency_days": recency_days,
            "frequency_score": frequency_score,
            "monetary_score": monetary_score,
            "num_products_owned": num_products_owned,
            "product_diversity_score": product_diversity_score.round(4),
            "cross_sell_ratio": cross_sell_ratio.round(4),
            "support_tickets_90d": support_tickets_90d,
            "escalation_rate": escalation_rate.round(4),
            "resolution_time_avg": resolution_time_avg.round(2),
            TREATMENT_COLUMN: treatment,
            TARGET_COLUMN: converted,
        }
    )

    logger.info(
        "Generated synthetic dataset: %d samples, treatment rate=%.2f, "
        "overall conversion rate=%.4f",
        n_samples,
        df[TREATMENT_COLUMN].mean(),
        df[TARGET_COLUMN].mean(),
    )
    return df


def preprocess_features(
    df: pd.DataFrame,
    feature_cols: Optional[list] = None,
    scaler: Optional[StandardScaler] = None,
    fit_scaler: bool = True,
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Apply feature scaling to continuous features while leaving encoded
    categorical features untouched.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing at least `feature_cols`.
    feature_cols : list, optional
        Columns to include. Defaults to ``ALL_FEATURES``.
    scaler : StandardScaler, optional
        Pre-fitted scaler for inference-time transformations.
    fit_scaler : bool
        Whether to fit the scaler on ``df``.

    Returns
    -------
    Tuple[pd.DataFrame, StandardScaler]
    """
    if feature_cols is None:
        feature_cols = ALL_FEATURES

    continuous_cols = [
        c for c in feature_cols
        if c not in ("region_encoded", "segment_encoded",
                     "frequency_score", "monetary_score")
    ]

    X = df[feature_cols].copy()
    if scaler is None:
        scaler = StandardScaler()

    if fit_scaler:
        X[continuous_cols] = scaler.fit_transform(X[continuous_cols])
    else:
        X[continuous_cols] = scaler.transform(X[continuous_cols])

    return X, scaler


def split_dataset(
    df: pd.DataFrame,
    feature_cols: Optional[list] = None,
    target_col: str = TARGET_COLUMN,
    treatment_col: str = TREATMENT_COLUMN,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VALIDATION_RATIO,
    random_state: int = RANDOM_STATE,
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.Series, pd.Series, pd.Series,
    pd.Series, pd.Series, pd.Series,
]:
    """
    Stratified split into train / validation / test sets preserving
    treatment-conversion joint distribution.

    Returns
    -------
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    t_train, t_val, t_test
    """
    if feature_cols is None:
        feature_cols = ALL_FEATURES

    X = df[feature_cols]
    y = df[target_col]
    t = df[treatment_col]

    # stratify on (treatment, converted) combination
    strat_key = t.astype(str) + "_" + y.astype(str)

    test_ratio = 1.0 - train_ratio - val_ratio
    X_train_val, X_test, y_train_val, y_test, t_train_val, t_test = train_test_split(
        X, y, t,
        test_size=test_ratio,
        stratify=strat_key,
        random_state=random_state,
    )

    val_relative = val_ratio / (train_ratio + val_ratio)
    strat_key_tv = (t_train_val.astype(str) + "_" + y_train_val.astype(str))
    X_train, X_val, y_train, y_val, t_train, t_val = train_test_split(
        X_train_val, y_train_val, t_train_val,
        test_size=val_relative,
        stratify=strat_key_tv,
        random_state=random_state,
    )

    logger.info(
        "Dataset split — train: %d, val: %d, test: %d",
        len(X_train), len(X_val), len(X_test),
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, t_train, t_val, t_test


def encode_categoricals(
    df: pd.DataFrame,
    categorical_cols: Optional[list] = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Label-encode categorical columns in-place. Returns the modified
    DataFrame and a mapping of column -> fitted LabelEncoder.
    """
    if categorical_cols is None:
        categorical_cols = ["region", "segment"]

    encoders: dict = {}
    df = df.copy()
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    return df, encoders
