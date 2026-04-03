"""
Disruption Impact (XGB + LGBM), Behavior (XGB + LGBM), Premium (deterministic actuarial core + XGB residual).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

import joblib
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

import sys

sys.path.append("..")
from config.model_config import DISRUPTION_IMPACT_CONFIG, BEHAVIOR_ANALYSIS_CONFIG, PREMIUM_PREDICTION_CONFIG, GENERAL_CONFIG
from config.data_config import MODEL_FEATURES


class DisruptionImpactModel:
    """Predicts income_loss_percentage from disruption + context features."""

    def __init__(self, config: Dict | None = None):
        self.config = config or DISRUPTION_IMPACT_CONFIG
        self.xgb_model = None
        self.lgb_model = None
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_names: List[str] = []
        self.weights = self.config["ensemble_weights"]

    def _encode(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        out = df.copy()
        for col in out.select_dtypes(include=["object"]).columns:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                out[col] = self.label_encoders[col].fit_transform(out[col].astype(str))
            elif col in self.label_encoders:
                le = self.label_encoders[col]
                out[col] = out[col].astype(str).map(lambda x: x if x in le.classes_ else le.classes_[0])
                out[col] = le.transform(out[col])
        return out.fillna(0)

    def fit(self, df: pd.DataFrame, val_df: pd.DataFrame | None = None):
        print("Training Disruption Impact Model (XGB + LightGBM)...")
        train_enc = self._encode(df, fit=True)
        features = [f for f in MODEL_FEATURES["disruption_impact"] if f in train_enc.columns]
        self.feature_names = features
        X = train_enc[features].values
        y = train_enc["income_loss_percentage"].values

        if val_df is not None:
            val_enc = self._encode(val_df, fit=False)
            X_val = val_enc[features].values
            y_val = val_enc["income_loss_percentage"].values
            X_train, y_train = X, y
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=GENERAL_CONFIG["validation_size"], random_state=GENERAL_CONFIG["random_state"]
            )

        Xs_train = self.scaler.fit_transform(X_train)
        Xs_val = self.scaler.transform(X_val)
        self.xgb_model = xgb.XGBRegressor(**self.config["xgboost"])
        self.xgb_model.fit(Xs_train, y_train, eval_set=[(Xs_val, y_val)], verbose=False)

        self.lgb_model = lgb.LGBMRegressor(**self.config["lightgbm"])
        self.lgb_model.fit(
            Xs_train,
            y_train,
            eval_set=[(Xs_val, y_val)],
            callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(False)],
        )
        print("✓ Disruption Impact Model trained")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        enc = self._encode(X, fit=False)
        features = [f for f in self.feature_names if f in enc.columns]
        Xs = self.scaler.transform(enc[features].values)
        pred = (
            self.weights["xgboost"] * self.xgb_model.predict(Xs)
            + self.weights["lightgbm"] * self.lgb_model.predict(Xs)
        )
        return np.clip(pred, 0.0, 1.0)

    def save(self, path: str):
        joblib.dump(self.xgb_model, f"{path}/xgb_model.pkl")
        joblib.dump(self.lgb_model, f"{path}/lgb_model.pkl")
        joblib.dump(self.scaler, f"{path}/scaler.pkl")
        joblib.dump(self.label_encoders, f"{path}/label_encoders.pkl")
        joblib.dump({"feature_names": self.feature_names, "config": self.config}, f"{path}/config.pkl")
        print(f"✓ Disruption Impact saved to {path}")

    def load(self, path: str):
        self.xgb_model = joblib.load(f"{path}/xgb_model.pkl")
        self.lgb_model = joblib.load(f"{path}/lgb_model.pkl")
        self.scaler = joblib.load(f"{path}/scaler.pkl")
        self.label_encoders = joblib.load(f"{path}/label_encoders.pkl")
        meta = joblib.load(f"{path}/config.pkl")
        self.feature_names = meta.get("feature_names", [])
        self.config = meta.get("config", self.config)
        return self


class BehaviorAnalysisModel:
    """
    Behavioral score for premium: ensemble regressor on defaults, claims history, activity.
    Target: payment_consistency_score (or proxy engineered label).
    """

    def __init__(self, config: Dict | None = None):
        self.config = config or BEHAVIOR_ANALYSIS_CONFIG
        self.xgb_model = None
        self.lgb_model = None
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_names: List[str] = []
        self.weights = self.config["ensemble_weights"]

    def _encode(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        out = df.copy()
        for col in out.select_dtypes(include=["object"]).columns:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                out[col] = self.label_encoders[col].fit_transform(out[col].astype(str))
            elif col in self.label_encoders:
                le = self.label_encoders[col]
                out[col] = out[col].astype(str).map(lambda x: x if x in le.classes_ else le.classes_[0])
                out[col] = le.transform(out[col])
        return out.fillna(0)

    def fit(self, df: pd.DataFrame, val_df: pd.DataFrame | None = None):
        print("Training Behavior Model (XGB + LightGBM)...")
        train_enc = self._encode(df, fit=True)
        features = [f for f in MODEL_FEATURES["behavior_analysis"] if f in train_enc.columns]
        self.feature_names = features
        X = train_enc[features].values
        y = train_enc["payment_consistency_score"].values

        if val_df is not None:
            val_enc = self._encode(val_df, fit=False)
            X_val = val_enc[features].values
            y_val = val_enc["payment_consistency_score"].values
            X_train, y_train = X, y
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=GENERAL_CONFIG["random_state"]
            )

        Xs_train = self.scaler.fit_transform(X_train)
        Xs_val = self.scaler.transform(X_val)
        self.xgb_model = xgb.XGBRegressor(**self.config["xgboost"])
        self.xgb_model.fit(Xs_train, y_train, eval_set=[(Xs_val, y_val)], verbose=False)

        self.lgb_model = lgb.LGBMRegressor(**self.config["lightgbm"])
        self.lgb_model.fit(
            Xs_train,
            y_train,
            eval_set=[(Xs_val, y_val)],
            callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(False)],
        )
        print("✓ Behavior Model trained")
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        enc = self._encode(X, fit=False)
        features = [f for f in self.feature_names if f in enc.columns]
        Xs = self.scaler.transform(enc[features].values)
        ens = (
            self.weights["xgboost"] * self.xgb_model.predict(Xs)
            + self.weights["lightgbm"] * self.lgb_model.predict(Xs)
        )
        ens = np.clip(ens, 0.0, 1.0)
        return pd.DataFrame(
            {
                "behavior_score": ens,
                "behavior_tier": pd.cut(ens, bins=[0, 0.45, 0.75, 1.0], labels=["high_risk", "medium", "low_risk"]),
            }
        )

    def save(self, path: str):
        joblib.dump(self.xgb_model, f"{path}/xgb_model.pkl")
        joblib.dump(self.lgb_model, f"{path}/lgb_model.pkl")
        joblib.dump(self.scaler, f"{path}/scaler.pkl")
        joblib.dump(self.label_encoders, f"{path}/label_encoders.pkl")
        joblib.dump({"feature_names": self.feature_names, "config": self.config}, f"{path}/config.pkl")
        print(f"✓ Behavior Model saved to {path}")

    def load(self, path: str):
        self.xgb_model = joblib.load(f"{path}/xgb_model.pkl")
        self.lgb_model = joblib.load(f"{path}/lgb_model.pkl")
        self.scaler = joblib.load(f"{path}/scaler.pkl")
        self.label_encoders = joblib.load(f"{path}/label_encoders.pkl")
        meta = joblib.load(f"{path}/config.pkl")
        self.feature_names = meta.get("feature_names", [])
        self.config = meta.get("config", self.config)
        return self


class PremiumPredictionModel:
    """
    Actuarial core (slab × 4% × rewards/penalties) + XGBoost residual on premium_amount / base_premium.
    At inference, optionally consumes forecasts from income, disruption, behavior, risk models.
    """

    def __init__(self, config: Dict | None = None):
        self.config = config or PREMIUM_PREDICTION_CONFIG
        self.ml_model: xgb.XGBRegressor | None = None
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.residual_feature_names: List[str] = []

    def _trust_0_5(self, fraud_trust_rating: float) -> float:
        if fraud_trust_rating is None or (isinstance(fraud_trust_rating, float) and np.isnan(fraud_trust_rating)):
            return 2.5
        if fraud_trust_rating <= 1.0:
            return float(fraud_trust_rating) * 5.0
        return float(fraud_trust_rating)

    def calculate_base_premium(self, avg_income: float, slab: str) -> float:
        base_rate = self.config["base_premium_rate"]
        multiplier = self.config["slab_multipliers"].get(slab, 1.0)
        return float(avg_income) * base_rate * multiplier

    def apply_rewards_penalties(self, base_premium: float, worker_data: Dict) -> Dict:
        adjusted = float(base_premium)
        consecutive_weeks = int(worker_data.get("consecutive_payment_weeks", 0) or 0)
        reward_cycles = consecutive_weeks // self.config["consistency_reward"]["weeks_required"]
        discount = reward_cycles * self.config["consistency_reward"]["discount_rate"]
        adjusted *= 1.0 - min(discount, 0.5)

        default_weeks = int(worker_data.get("default_weeks", 0) or 0)
        penalty = default_weeks * self.config["default_penalty"]["penalty_rate"]
        adjusted *= 1.0 + penalty

        fraud_penalty = 0.0
        tr = worker_data.get("fraud_trust_rating", 1.0)
        tr5 = self._trust_0_5(float(tr))
        th = self.config["fraud_penalty"]["threshold_trust_0_5"]
        if tr5 < th:
            drop = th - tr5
            fraud_penalty = self.config["fraud_penalty"]["base_penalty"] + (drop / 0.1) * self.config["fraud_penalty"]["incremental_penalty"]
            adjusted *= 1.0 + fraud_penalty

        return {
            "base_premium": base_premium,
            "consistency_discount": discount,
            "default_penalty": penalty,
            "fraud_penalty": fraud_penalty,
            "adjusted_premium": adjusted,
        }

    def _prepare_residual_frame(self, df: pd.DataFrame, fit: bool) -> Tuple[pd.DataFrame, np.ndarray | None]:
        d = df.copy()
        for col in MODEL_FEATURES["premium_prediction"]:
            if col not in d.columns:
                if col == "forecasted_weekly_income":
                    d[col] = d["avg_52week_income"] if "avg_52week_income" in d.columns else 0.0
                elif col == "predicted_income_loss_pct":
                    d[col] = 0.0
                elif col in ("behavior_score", "predicted_risk_score"):
                    d[col] = 0.5
                else:
                    d[col] = 0
        enc = d.copy()
        for col in list(enc.columns):
            if enc[col].dtype == object or str(enc[col].dtype) == "category":
                if col not in MODEL_FEATURES["premium_prediction"]:
                    continue
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    enc[col] = self.label_encoders[col].fit_transform(enc[col].astype(str))
                elif col in self.label_encoders:
                    le = self.label_encoders[col]
                    enc[col] = enc[col].astype(str).map(lambda x: x if x in le.classes_ else le.classes_[0])
                    enc[col] = le.transform(enc[col])
        self.residual_feature_names = [c for c in MODEL_FEATURES["premium_prediction"] if c in enc.columns]
        X = enc[self.residual_feature_names].apply(pd.to_numeric, errors="coerce").fillna(0)
        y = None
        if "premium_amount" in d.columns:
            bp = (d["avg_52week_income"].astype(float) * self.config["base_premium_rate"]).replace(0, np.nan)
            y = (d["premium_amount"].astype(float) / bp).values
            y = np.nan_to_num(y, nan=1.0)
        return X, y

    def fit(self, df: pd.DataFrame):
        print("Training Premium residual model (XGBoost)...")
        X, y = self._prepare_residual_frame(df, fit=True)
        Xv = self.scaler.fit_transform(X.values)
        self.ml_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=42,
        )
        if y is None:
            raise ValueError("premium_amount and avg_52week_income required to train residual head")
        X_tr, X_va, y_tr, y_va = train_test_split(Xv, y, test_size=0.2, random_state=42)
        self.ml_model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        print("✓ Premium residual trained")
        return self

    def predict(self, worker_data: pd.DataFrame) -> pd.DataFrame:
        rows: List[Dict] = []
        X_df, _ = self._prepare_residual_frame(worker_data, fit=False)
        X_mat = self.scaler.transform(X_df[self.residual_feature_names].values)

        for idx, (_, row) in enumerate(worker_data.iterrows()):
            slab = row.get("selected_slab", "Standard")
            base = self.calculate_base_premium(float(row.get("avg_52week_income", 0) or 0), str(slab))
            breakdown = self.apply_rewards_penalties(base, row.to_dict())
            adj = breakdown["adjusted_premium"]
            if self.ml_model is not None:
                mult = float(self.ml_model.predict(X_mat[idx : idx + 1])[0])
                w = self.config["ml_adjustment_weight"]
                final = adj * ((1.0 - w) + w * mult)
            else:
                final = adj
            rows.append({**breakdown, "final_premium": float(final)})
        return pd.DataFrame(rows)

    def save(self, path: str):
        joblib.dump(self.ml_model, f"{path}/xgb_model.pkl")
        joblib.dump(self.scaler, f"{path}/scaler.pkl")
        joblib.dump(self.label_encoders, f"{path}/label_encoders.pkl")
        joblib.dump(
            {"residual_feature_names": self.residual_feature_names, "config": self.config},
            f"{path}/config.pkl",
        )
        print(f"✓ Premium model saved to {path}")

    def load(self, path: str):
        try:
            self.ml_model = joblib.load(f"{path}/xgb_model.pkl")
        except OSError:
            self.ml_model = None
        self.scaler = joblib.load(f"{path}/scaler.pkl")
        self.label_encoders = joblib.load(f"{path}/label_encoders.pkl")
        meta = joblib.load(f"{path}/config.pkl")
        self.residual_feature_names = meta.get("residual_feature_names", [])
        self.config = meta.get("config", self.config)
        return self
