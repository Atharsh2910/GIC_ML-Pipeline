"""
Risk Scoring Model
Ensemble: XGBoost + LightGBM
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import joblib
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import shap

import sys
sys.path.append('..')
from config.model_config import RISK_SCORING_CONFIG, GENERAL_CONFIG
from config.data_config import MODEL_FEATURES


class RiskScoringModel:
    """
    Ensemble risk scoring using XGBoost and LightGBM
    Predicts overall_risk_score (0-1)
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or RISK_SCORING_CONFIG
        self.xgb_model = None
        self.lgb_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = MODEL_FEATURES['risk_scoring']
        self.ensemble_weights = self.config['ensemble_weights']
        
    def preprocess_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Preprocess and encode features"""
        df = df.copy()
        
        # Handle categorical features
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col in df.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    if col in self.label_encoders:
                        df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        # Fill missing values
        df = df.fillna(df.median(numeric_only=True))
        
        return df
    
    def prepare_data(self, df: pd.DataFrame, fit: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target"""
        df = self.preprocess_features(df, fit=fit)
        
        # Get features that exist in dataframe
        available_features = [f for f in self.feature_names if f in df.columns]
        
        X = df[available_features]
        y = df['overall_risk_score'] if 'overall_risk_score' in df.columns else None
        
        if fit:
            X = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
        
        return X, y
    
    def fit(self, df: pd.DataFrame, val_df: pd.DataFrame = None):
        """Train both XGBoost and LightGBM models"""
        print("Training Risk Scoring Ensemble...")

        if val_df is not None:
            X_train, y_train = self.prepare_data(df, fit=True)
            X_val, y_val = self.prepare_data(val_df, fit=False)
            val_df_raw = val_df
        else:
            train_part, val_part = train_test_split(
                df,
                test_size=GENERAL_CONFIG["validation_size"],
                random_state=GENERAL_CONFIG["random_state"],
            )
            X_train, y_train = self.prepare_data(train_part, fit=True)
            X_val, y_val = self.prepare_data(val_part, fit=False)
            val_df_raw = val_part
        
        # Train XGBoost
        print("Training XGBoost...")
        xgb_config = self.config['xgboost']
        
        self.xgb_model = xgb.XGBRegressor(**xgb_config)
        
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        xgb_score = r2_score(y_val, self.xgb_model.predict(X_val))
        print(f"✓ XGBoost R² Score: {xgb_score:.4f}")
        
        # Train LightGBM
        print("Training LightGBM...")
        lgb_config = self.config['lightgbm']
        
        self.lgb_model = lgb.LGBMRegressor(**lgb_config)
        
        self.lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(False)]
        )
        
        lgb_score = r2_score(y_val, self.lgb_model.predict(X_val))
        print(f"✓ LightGBM R² Score: {lgb_score:.4f}")
        
        ensemble_pred = self.predict(val_df_raw)
        ensemble_score = r2_score(y_val, ensemble_pred["risk_score"].values)
        print(f"✓ Ensemble R² Score: {ensemble_score:.4f}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict risk scores using ensemble"""
        # Prepare data
        if 'overall_risk_score' in X.columns:
            X_processed, _ = self.prepare_data(X, fit=False)
        else:
            X_processed = self.preprocess_features(X, fit=False)
            available_features = [f for f in self.feature_names if f in X_processed.columns]
            X_processed = X_processed[available_features]
            X_processed = pd.DataFrame(
                self.scaler.transform(X_processed),
                columns=X_processed.columns,
                index=X_processed.index
            )
        
        # Predictions from each model
        xgb_pred = self.xgb_model.predict(X_processed)
        lgb_pred = self.lgb_model.predict(X_processed)
        
        # Ensemble prediction
        ensemble_pred = (
            xgb_pred * self.ensemble_weights['xgboost'] +
            lgb_pred * self.ensemble_weights['lightgbm']
        )
        
        # Clip to [0, 1]
        ensemble_pred = np.clip(ensemble_pred, 0, 1)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'risk_score': ensemble_pred,
            'xgb_risk': xgb_pred,
            'lgb_risk': lgb_pred,
            'risk_category': pd.cut(ensemble_pred, 
                                    bins=[0, 0.3, 0.6, 1.0],
                                    labels=['Low', 'Medium', 'High'])
        })
        
        return results
    
    def explain_prediction(self, X: pd.DataFrame, num_samples: int = 100):
        """Generate SHAP explanations"""
        X_processed, _ = self.prepare_data(X, fit=False)
        
        # Sample data if too large
        if len(X_processed) > num_samples:
            X_sample = X_processed.sample(num_samples, random_state=42)
        else:
            X_sample = X_processed
        
        # SHAP for XGBoost
        explainer_xgb = shap.TreeExplainer(self.xgb_model)
        shap_values_xgb = explainer_xgb.shap_values(X_sample)
        
        return {
            'shap_values': shap_values_xgb,
            'feature_names': X_sample.columns.tolist(),
            'base_value': explainer_xgb.expected_value
        }
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from both models"""
        xgb_importance = pd.DataFrame({
            'feature': self.xgb_model.feature_names_in_,
            'xgb_importance': self.xgb_model.feature_importances_
        })
        
        lgb_importance = pd.DataFrame({
            'feature': self.lgb_model.feature_name_,
            'lgb_importance': self.lgb_model.feature_importances_
        })
        
        importance = xgb_importance.merge(lgb_importance, on='feature')
        importance['avg_importance'] = (
            importance['xgb_importance'] * self.ensemble_weights['xgboost'] +
            importance['lgb_importance'] * self.ensemble_weights['lightgbm']
        )
        
        return importance.sort_values('avg_importance', ascending=False)
    
    def save(self, path: str):
        """Save models and preprocessing objects"""
        joblib.dump(self.xgb_model, f"{path}/xgb_model.pkl")
        joblib.dump(self.lgb_model, f"{path}/lgb_model.pkl")
        joblib.dump(self.scaler, f"{path}/scaler.pkl")
        joblib.dump(self.label_encoders, f"{path}/label_encoders.pkl")
        joblib.dump(self.config, f"{path}/config.pkl")
        print(f"✓ Risk Scoring Model saved to {path}")
    
    def load(self, path: str):
        """Load models and preprocessing objects"""
        self.xgb_model = joblib.load(f"{path}/xgb_model.pkl")
        self.lgb_model = joblib.load(f"{path}/lgb_model.pkl")
        self.scaler = joblib.load(f"{path}/scaler.pkl")
        self.label_encoders = joblib.load(f"{path}/label_encoders.pkl")
        self.config = joblib.load(f"{path}/config.pkl")
        print(f"✓ Risk Scoring Model loaded from {path}")
        return self


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    n_samples = 10000
    
    data = {
        'income_volatility': np.random.uniform(0, 1, n_samples),
        'payment_consistency_score': np.random.uniform(0, 1, n_samples),
        'fraud_trust_rating': np.random.uniform(0, 1, n_samples),
        'disruption_exposure_risk': np.random.uniform(0, 1, n_samples),
        'distance_from_outlet_km': np.random.uniform(0, 20, n_samples),
        'order_acceptance_rate': np.random.uniform(0.5, 1, n_samples),
        'order_decline_rate': np.random.uniform(0, 0.5, n_samples),
        'gps_spoofing_score': np.random.uniform(0, 0.3, n_samples),
        'movement_realism_score': np.random.uniform(0.6, 1, n_samples),
        'presence_score': np.random.uniform(0.5, 1, n_samples),
        'peer_group_activity_ratio': np.random.uniform(0.7, 1.3, n_samples),
        'consecutive_payment_weeks': np.random.randint(0, 52, n_samples),
    }
    
    # Calculate synthetic risk score
    data['overall_risk_score'] = (
        0.3 * data['income_volatility'] +
        0.2 * (1 - data['payment_consistency_score']) +
        0.2 * (1 - data['fraud_trust_rating']) +
        0.15 * data['disruption_exposure_risk'] +
        0.15 * data['gps_spoofing_score']
    )
    
    df = pd.DataFrame(data)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Train
    model = RiskScoringModel()
    model.fit(train_df, test_df)
    
    # Predict
    predictions = model.predict(test_df.head())
    print("\nSample Predictions:")
    print(predictions)
    
    # Feature importance
    print("\nFeature Importance:")
    print(model.get_feature_importance().head(10))