"""
Expected on-disk artifacts after training (local or Kaggle).
Copy the `models/<name>/` folders into this repo; filenames must match for load().
"""

ARTIFACT_FILES = {
    "income_forecasting": [
        "lstm_model.h5",
        "sarimax_model.pkl",
        "scaler.pkl",
        "config.pkl",
    ],
    "risk_scoring": [
        "xgb_model.pkl",
        "lgb_model.pkl",
        "scaler.pkl",
        "label_encoders.pkl",
        "config.pkl",
    ],
    "fraud_detection": [
        "isolation_forest.pkl",
        "xgb_classifier.pkl",
        "scaler.pkl",
        "label_encoders.pkl",
        "config.pkl",
    ],
    "disruption_impact": [
        "xgb_model.pkl",
        "lgb_model.pkl",
        "scaler.pkl",
        "label_encoders.pkl",
        "config.pkl",
    ],
    "behavior_analysis": [
        "xgb_model.pkl",
        "lgb_model.pkl",
        "scaler.pkl",
        "label_encoders.pkl",
        "config.pkl",
    ],
    "premium_prediction": [
        "xgb_model.pkl",
        "scaler.pkl",
        "label_encoders.pkl",
        "config.pkl",
    ],
}
