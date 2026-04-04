"""
Model Configuration
Contains hyperparameters and settings for all ML models
"""

# Income Forecasting Model Configuration
INCOME_FORECAST_CONFIG = {
    'rolling_mean': {
        'window': 52,
        'min_periods': 4
    },
    'lstm': {
        'units': [128, 64, 32],
        'dropout': 0.2,
        'recurrent_dropout': 0.2,
        'epochs': 100,
        'batch_size': 32,
        'lookback': 12,
        'learning_rate': 0.001,
        'early_stopping_patience': 10
    },
    'sarimax': {
        'order': (1, 1, 1),
        'seasonal_order': (1, 1, 1, 52),
        'enforce_stationarity': False,
        'enforce_invertibility': False
    },
    'ensemble_weights': {
        'rolling_mean': 0.3,
        'lstm': 0.4,
        'sarimax': 0.3
    }
}

# Risk Scoring Model Configuration
RISK_SCORING_CONFIG = {
    'xgboost': {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.05,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'reg_alpha': 0.05,
        'reg_lambda': 1.0,
        'objective': 'reg:squarederror',
        'random_state': 42
    },
    'lightgbm': {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.05,
        'reg_lambda': 1.0,
        'objective': 'regression',
        'random_state': 42
    },
    'ensemble_weights': {
        'xgboost': 0.5,
        'lightgbm': 0.5
    }
}

# Fraud Detection Model Configuration
FRAUD_DETECTION_CONFIG = {
    'isolation_forest': {
        'n_estimators': 200,
        'max_samples': 'auto',
        'contamination': 0.05,
        'max_features': 1.0,
        'random_state': 42
    },
    'xgb_classifier': {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.05,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'scale_pos_weight': 10,  # Handle class imbalance
        'objective': 'binary:logistic',
        'random_state': 42
    },
    'fraud_threshold': 0.5,
    'trust_rating_multiplier': 5.0
}

# Disruption Impact Model Configuration
DISRUPTION_IMPACT_CONFIG = {
    'xgboost': {
        'n_estimators': 200,
        'max_depth': 7,
        'learning_rate': 0.05,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'reg_alpha': 0.05,
        'reg_lambda': 1.0,
        'objective': 'reg:squarederror',
        'random_state': 42
    },
    'lightgbm': {
        'n_estimators': 200,
        'max_depth': 7,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.05,
        'reg_lambda': 1.0,
        'objective': 'regression',
        'random_state': 42
    },
    'ensemble_weights': {
        'xgboost': 0.55,
        'lightgbm': 0.45
    }
}

# Behavior Analysis Model Configuration (XGBoost + LightGBM; payment & claim behavior)
BEHAVIOR_ANALYSIS_CONFIG = {
    'xgboost': {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.05,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'reg_alpha': 0.05,
        'reg_lambda': 1.0,
        'objective': 'reg:squarederror',
        'random_state': 42
    },
    'lightgbm': {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.05,
        'reg_lambda': 1.0,
        'objective': 'regression',
        'random_state': 42
    },
    'ensemble_weights': {
        'xgboost': 0.5,
        'lightgbm': 0.5
    }
}

# Premium Prediction Model Configuration
PREMIUM_PREDICTION_CONFIG = {
    'base_premium_rate': 0.04,  # 4% of average weekly income
    'slab_multipliers': {
        # final_dataset.csv labels
        'Basic': 0.90,
        'Standard': 1.00,
        'Premium': 1.10,
        'Elite': 1.20,
        # Product-doc slab names
        'Slab 1 (50%)': 0.90,
        'Slab 2 (75%)': 1.00,
        'Slab 3 (100%)': 1.20,
        # Actual dataset slab names
        'Slab_50': 0.90,
        'Slab_75': 1.00,
        'Slab_100': 1.20,
    },
    'consistency_reward': {
        'weeks_required': 4,
        'discount_rate': 0.005  # 0.5% reduction per 4 weeks
    },
    'default_penalty': {
        'penalty_rate': 0.02  # 2% penalty per missed week
    },
    'fraud_penalty': {
        # Doc uses 0–5 trust; dataset may be 0–1 — code maps 0–1 → 0–5 when max<=1
        'threshold_trust_0_5': 2.5,
        'base_penalty': 0.01,
        'incremental_penalty': 0.001  # per 0.1 trust drop below threshold (on 0–5 scale)
    },
    'ml_adjustment_weight': 0.15  # 15% weight for ML-based adjustments
}

# Claim Eligibility Configuration
CLAIM_ELIGIBILITY_CONFIG = {
    'income_threshold': 0.75,  # 75% of 52-week average
    'minimum_loss_percentage': 0.25,  # Must lose at least 25%
    'cooling_period_weeks': 8,
    'minimum_active_weeks': 4
}

# Payout Optimization Configuration
PAYOUT_OPTIMIZATION_CONFIG = {
    'disruption_coverage': {
        'rainfall': {
            0: 0.0,
            5: 0.4,
            10: 0.6,
            15: 0.8,
            20: 1.0
        },
        'cyclone': {
            0: 0.0,
            1: 0.2,
            2: 0.4,
            3: 0.6,
            4: 0.8,
            5: 1.0
        }
    },
    'slab_coverage': {
        'Basic': 0.5,
        'Standard': 0.75,
        'Premium': 0.90,
        'Elite': 1.0,
        'Slab 1 (50%)': 0.5,
        'Slab 2 (75%)': 0.75,
        'Slab 3 (100%)': 1.0,
        # Actual dataset slab names
        'Slab_50': 0.5,
        'Slab_75': 0.75,
        'Slab_100': 1.0,
    },
    'loyalty_bonus_rate': 0.005,  # 0.5% per 4 weeks
    'max_loyalty_bonus': 0.20,    # Maximum 20%
    'penalty_impact_on_coverage': True
}

# General Model Settings
GENERAL_CONFIG = {
    'test_size': 0.2,
    'validation_size': 0.15,
    'random_state': 42,
    'cv_folds': 5,
    'n_jobs': -1,
    'verbose': 1
}

# SHAP Configuration for Explainability
SHAP_CONFIG = {
    'max_samples': 1000,
    'check_additivity': False
}

# MLflow Configuration
MLFLOW_CONFIG = {
    'tracking_uri': 'mlruns',
    'experiment_name': 'gigshield_insurance'
}