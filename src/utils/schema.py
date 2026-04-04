"""Dataset schema helpers: default columns for training/inference (Kaggle CSV alignment)."""

from __future__ import annotations

import pandas as pd

from config.dataset_schema import DISRUPTION_NONE

# Columns described in product spec; missing values filled for ML blocks
DEFAULT_NUMERIC_FILLS = {
    "default_weeks": 0,
    "claims_past_52_weeks": 0,
    "forecasted_weekly_income": None,  # filled from avg_52week_income if None
    "predicted_income_loss_pct": 0.0,
    "behavior_score": 0.5,
    "predicted_risk_score": 0.5,
    "weeks_active": 26,
    "week_of_year": 20,
    "weekly_income": 5000.0,
    "income_std_dev": 200.0,
    "income_volatility": 0.1,
    "orders_completed_week": 50,
    "active_hours_week": 40.0,
    "disruption_duration_hours": 2.0,
    "rainfall_cm": 5.0,
    "temperature_extreme": 28.0,
    "payment_consistency_score": 0.9,
    "fraud_trust_rating": 0.8,
    "disruption_exposure_risk": 0.1,
    "distance_from_outlet_km": 5.0,
    "order_acceptance_rate": 0.9,
    "order_decline_rate": 0.1,
    "gps_spoofing_score": 0.0,
    "movement_realism_score": 1.0,
    "presence_score": 1.0,
    "peer_group_activity_ratio": 1.0,
    "consecutive_payment_weeks": 10,
    "coordinated_fraud_cluster_id": 0,
    "ip_gps_mismatch": 0,
    "device_sharing_flag": 0,
    "cyclone_alert_level": 0,
    "premium_paid": 1,
    "cooling_period_completed": 1,
    "overall_risk_score": 0.2,
    "highest_weekly_income": 7000.0,
    "lowest_weekly_income": 3000.0,
    "base_premium": 200.0,
    "premium_amount": 200.0,
    "outlet_id": 0,
    "worker_lat": 19.076,
    "worker_lon": 72.877,
    "outlet_lat": 19.076,
    "outlet_lon": 72.877,
    "income_loss_amount": 0.0,
    "coverage_percentage": 50.0,
    "loyalty_bonus_percentage": 0.0,
    "penalty_percentage": 0.0,
    "final_payout_amount": 0.0,
}

# String defaults — exact values from dataset that LabelEncoders were trained on
DEFAULT_STRING_FILLS = {
    "platform": "Zepto",
    "employment_type": "Full-Time",
    "selected_slab": "Slab_50",
    "disruption_type": "Heavy_Rain",
    "city": "Mumbai",
}


def normalize_gigshield_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Align uploaded CSV quirks with model expectations:
    - Many rows have NaN disruption_type (no event): fill with 'Heavy_Rain'
    - String columns stripped
    """
    out = df.copy()
    if "disruption_type" in out.columns:
        out["disruption_type"] = out["disruption_type"].fillna("Heavy_Rain")
        out["disruption_type"] = out["disruption_type"].astype(str).str.strip()
        out.loc[out["disruption_type"].isin(["", "nan", "NaN"]), "disruption_type"] = "Heavy_Rain"
    for col in ("city", "platform", "employment_type", "selected_slab"):
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip()
    return out


def ensure_worker_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add missing optional columns so all models receive stable inputs."""
    out = normalize_gigshield_dataframe(df)
    for col, default in DEFAULT_NUMERIC_FILLS.items():
        if col not in out.columns:
            if col == "forecasted_weekly_income" and "avg_52week_income" in out.columns:
                out[col] = out["avg_52week_income"]
            elif default is not None:
                out[col] = default
    for col, default in DEFAULT_STRING_FILLS.items():
        if col not in out.columns:
            out[col] = default
    if "forecasted_weekly_income" in out.columns and "avg_52week_income" in out.columns:
        m = out["forecasted_weekly_income"].isna()
        if m.any():
            out.loc[m, "forecasted_weekly_income"] = out.loc[m, "avg_52week_income"]
    return out
