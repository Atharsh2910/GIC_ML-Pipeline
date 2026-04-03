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
}


def normalize_gigshield_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Align uploaded CSV quirks with model expectations:
    - Many rows have NaN disruption_type (no event): fill with 'none'
    - String columns stripped
    """
    out = df.copy()
    if "disruption_type" in out.columns:
        out["disruption_type"] = out["disruption_type"].fillna(DISRUPTION_NONE)
        out["disruption_type"] = out["disruption_type"].astype(str).str.strip()
        out.loc[out["disruption_type"].isin(["", "nan", "NaN"]), "disruption_type"] = DISRUPTION_NONE
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
    if "forecasted_weekly_income" in out.columns and "avg_52week_income" in out.columns:
        m = out["forecasted_weekly_income"].isna()
        if m.any():
            out.loc[m, "forecasted_weekly_income"] = out.loc[m, "avg_52week_income"]
    return out
