# model parameters
PROPENSITY_MODEL_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 1,
    "random_state": 42,
    "eval_metric": "auc",
    "early_stopping_rounds": 50,
}

UPLIFT_MODEL_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "random_state": 42,
    "eval_metric": "logloss",
}

# A/B testing settings
CONFIDENCE_LEVEL = 0.95
MIN_SAMPLE_SIZE = 1000
RANDOM_STATE = 42

# resource optimization settings
RESOURCE_BUDGET_FRACTION = 0.30
LOW_VALUE_EFFORT_THRESHOLD = 0.22
HIGH_VALUE_PROPENSITY_THRESHOLD = 0.60

# data split ratios
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# feature groups
DEMOGRAPHIC_FEATURES = [
    "age",
    "tenure_months",
    "region_encoded",
    "segment_encoded",
]

BEHAVIORAL_FEATURES = [
    "num_interactions_30d",
    "num_interactions_90d",
    "avg_spend_30d",
    "avg_spend_90d",
    "recency_days",
    "frequency_score",
    "monetary_score",
]

PRODUCT_FEATURES = [
    "num_products_owned",
    "product_diversity_score",
    "cross_sell_ratio",
]

OPERATIONAL_FEATURES = [
    "support_tickets_90d",
    "escalation_rate",
    "resolution_time_avg",
]

ALL_FEATURES = (
    DEMOGRAPHIC_FEATURES
    + BEHAVIORAL_FEATURES
    + PRODUCT_FEATURES
    + OPERATIONAL_FEATURES
)

TARGET_COLUMN = "converted"
TREATMENT_COLUMN = "treatment"
