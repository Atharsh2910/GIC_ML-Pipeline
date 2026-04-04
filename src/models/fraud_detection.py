"""
Fraud Detection Model
Hybrid: Isolation Forest (Anomaly Detection) + XGBoost Classifier
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import xgboost as xgb
import shap

import sys
sys.path.append('..')
from config.model_config import FRAUD_DETECTION_CONFIG, GENERAL_CONFIG
from config.data_config import MODEL_FEATURES


class FraudDetectionModel:
    """
    Hybrid fraud detection:
    - Isolation Forest for unsupervised anomaly detection
    - XGBoost Classifier for supervised fraud classification
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or FRAUD_DETECTION_CONFIG
        self.isolation_forest = None
        self.xgb_classifier = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = MODEL_FEATURES['fraud_detection']
        
    def preprocess_features(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Preprocess features"""
        df = df.copy()

        # Encode categorical features
        categorical_cols = df.select_dtypes(include=['object']).columns

        for col in categorical_cols:
            if col in df.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        known = set(le.classes_)
                        # Map unseen labels to first known class to avoid crash
                        df[col] = df[col].astype(str).apply(
                            lambda x: x if x in known else le.classes_[0]
                        )
                        df[col] = le.transform(df[col])

        # Fill missing values
        df = df.fillna(0)

        return df
    
    def prepare_data(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Prepare features"""
        df = self.preprocess_features(df, fit=fit)
        
        # Get available features
        available_features = [f for f in self.feature_names if f in df.columns]
        X = df[available_features]
        
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return pd.DataFrame(X_scaled, columns=available_features, index=X.index)
    
    def create_fraud_labels(self, df: pd.DataFrame) -> np.ndarray:
        """
        Create fraud labels based on fraud_trust_rating
        Low trust rating (< 0.4) = Fraud (1)
        High trust rating (>= 0.4) = Legitimate (0)
        """
        if 'fraud_trust_rating' in df.columns:
            # Invert: low trust = high fraud
            return (df['fraud_trust_rating'] < 0.4).astype(int).values
        else:
            raise ValueError("fraud_trust_rating column required for supervised training")
    
    def fit(self, df: pd.DataFrame, val_df: pd.DataFrame = None):
        """Train both Isolation Forest and XGBoost Classifier"""
        print("Training Fraud Detection Models...")
        
        # Prepare data
        X_train = self.prepare_data(df, fit=True)
        
        # 1. Train Isolation Forest (Unsupervised)
        print("Training Isolation Forest...")
        if_config = self.config['isolation_forest']
        
        self.isolation_forest = IsolationForest(**if_config)
        self.isolation_forest.fit(X_train)
        
        # Get anomaly scores
        anomaly_scores = self.isolation_forest.decision_function(X_train)
        anomaly_predictions = self.isolation_forest.predict(X_train)
        
        contamination_rate = (anomaly_predictions == -1).mean()
        print(f"✓ Isolation Forest trained - Anomaly rate: {contamination_rate:.2%}")
        
        # 2. Train XGBoost Classifier (Supervised)
        print("Training XGBoost Classifier...")
        
        # Create labels
        y_train = self.create_fraud_labels(df)
        
        # Handle class imbalance
        fraud_rate = y_train.mean()
        print(f"Fraud rate in training data: {fraud_rate:.2%}")
        
        if val_df is not None:
            X_val = self.prepare_data(val_df, fit=False)
            y_val = self.create_fraud_labels(val_df)
        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train,
                test_size=GENERAL_CONFIG['validation_size'],
                random_state=GENERAL_CONFIG['random_state'],
                stratify=y_train
            )
        
        xgb_config = self.config['xgb_classifier'].copy()
        
        # Adjust scale_pos_weight based on actual class distribution
        if fraud_rate > 0 and fraud_rate < 1:
            xgb_config['scale_pos_weight'] = (1 - fraud_rate) / fraud_rate
        
        self.xgb_classifier = xgb.XGBClassifier(**xgb_config)
        
        self.xgb_classifier.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Evaluation
        y_pred = self.xgb_classifier.predict(X_val)
        y_pred_proba = self.xgb_classifier.predict_proba(X_val)[:, 1]
        
        auc_score = roc_auc_score(y_val, y_pred_proba)
        print(f"✓ XGBoost Classifier trained - AUC: {auc_score:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred, target_names=['Legitimate', 'Fraud']))
        
        return self
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predict fraud using hybrid approach
        Combines Isolation Forest anomaly scores with XGBoost predictions
        """
        # Prepare data
        X_processed = self.prepare_data(X, fit=False)
        
        # Isolation Forest predictions
        if_anomaly_score = self.isolation_forest.decision_function(X_processed)
        if_is_anomaly = self.isolation_forest.predict(X_processed)
        
        # Normalize IF scores to [0, 1] (higher = more anomalous)
        if_anomaly_normalized = 1 / (1 + np.exp(if_anomaly_score))
        
        # XGBoost predictions
        xgb_fraud_proba = self.xgb_classifier.predict_proba(X_processed)[:, 1]
        xgb_fraud_pred = self.xgb_classifier.predict(X_processed)
        
        # Combined fraud score (weighted average)
        fraud_score = 0.4 * if_anomaly_normalized + 0.6 * xgb_fraud_proba
        
        # Calculate trust rating (inverse of fraud score)
        trust_rating = 1 - fraud_score
        
        # Categorize fraud risk
        fraud_category = pd.cut(
            fraud_score,
            bins=[0, 0.3, 0.6, 1.0],
            labels=['Low Risk', 'Medium Risk', 'High Risk']
        )
        
        # Results
        results = pd.DataFrame({
            'fraud_probability': fraud_score,
            'trust_rating': trust_rating,
            'fraud_category': fraud_category,
            'is_fraud': (fraud_score > self.config['fraud_threshold']).astype(int),
            'if_anomaly_score': if_anomaly_normalized,
            'xgb_fraud_score': xgb_fraud_proba,
            'requires_review': (fraud_score > 0.6).astype(int)
        })
        
        return results
    
    def explain_prediction(self, X: pd.DataFrame, num_samples: int = 100):
        """Generate SHAP explanations for XGBoost predictions"""
        X_processed = self.prepare_data(X, fit=False)
        
        if len(X_processed) > num_samples:
            X_sample = X_processed.sample(num_samples, random_state=42)
        else:
            X_sample = X_processed
        
        explainer = shap.TreeExplainer(self.xgb_classifier)
        shap_values = explainer.shap_values(X_sample)
        
        return {
            'shap_values': shap_values,
            'feature_names': X_sample.columns.tolist(),
            'base_value': explainer.expected_value
        }
    
    def detect_coordinated_fraud(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect coordinated fraud clusters
        Groups workers with similar suspicious patterns
        """
        fraud_predictions = self.predict(df)
        
        # Add worker IDs if available
        if 'worker_id' in df.columns:
            fraud_predictions['worker_id'] = df['worker_id'].values
        
        # Identify high-risk workers
        high_risk = fraud_predictions[fraud_predictions['fraud_probability'] > 0.7]
        
        if len(high_risk) > 0:
            # Check for coordinated patterns
            if 'coordinated_fraud_cluster_id' in df.columns:
                cluster_analysis = df.groupby('coordinated_fraud_cluster_id').agg({
                    'worker_id': 'count',
                    'fraud_trust_rating': 'mean',
                    'gps_spoofing_score': 'mean'
                }).rename(columns={'worker_id': 'cluster_size'})
                
                suspicious_clusters = cluster_analysis[
                    (cluster_analysis['cluster_size'] > 1) &
                    (cluster_analysis['fraud_trust_rating'] < 0.4)
                ]
                
                return suspicious_clusters
        
        return pd.DataFrame()
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from XGBoost"""
        importance = pd.DataFrame({
            'feature': self.xgb_classifier.feature_names_in_,
            'importance': self.xgb_classifier.feature_importances_
        })
        
        return importance.sort_values('importance', ascending=False)
    
    def save(self, path: str):
        """Save models"""
        joblib.dump(self.isolation_forest, f"{path}/isolation_forest.pkl")
        joblib.dump(self.xgb_classifier, f"{path}/xgb_classifier.pkl")
        joblib.dump(self.scaler, f"{path}/scaler.pkl")
        joblib.dump(self.label_encoders, f"{path}/label_encoders.pkl")
        joblib.dump(self.config, f"{path}/config.pkl")
        print(f"✓ Fraud Detection Model saved to {path}")
    
    def load(self, path: str):
        """Load models"""
        self.isolation_forest = joblib.load(f"{path}/isolation_forest.pkl")
        self.xgb_classifier = joblib.load(f"{path}/xgb_classifier.pkl")
        self.scaler = joblib.load(f"{path}/scaler.pkl")
        self.label_encoders = joblib.load(f"{path}/label_encoders.pkl")
        self.config = joblib.load(f"{path}/config.pkl")
        print(f"✓ Fraud Detection Model loaded from {path}")
        return self


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)
    n_legitimate = 9000
    n_fraud = 1000
    
    # Legitimate users
    legitimate_data = {
        'gps_spoofing_score': np.random.uniform(0, 0.2, n_legitimate),
        'movement_realism_score': np.random.uniform(0.7, 1, n_legitimate),
        'presence_score': np.random.uniform(0.7, 1, n_legitimate),
        'peer_group_activity_ratio': np.random.uniform(0.8, 1.2, n_legitimate),
        'order_acceptance_rate': np.random.uniform(0.7, 1, n_legitimate),
        'orders_completed_week': np.random.randint(20, 60, n_legitimate),
        'fraud_trust_rating': np.random.uniform(0.6, 1, n_legitimate),
    }
    
    # Fraudulent users
    fraud_data = {
        'gps_spoofing_score': np.random.uniform(0.5, 1, n_fraud),
        'movement_realism_score': np.random.uniform(0, 0.4, n_fraud),
        'presence_score': np.random.uniform(0, 0.4, n_fraud),
        'peer_group_activity_ratio': np.random.uniform(0.2, 0.6, n_fraud),
        'order_acceptance_rate': np.random.uniform(0.2, 0.5, n_fraud),
        'orders_completed_week': np.random.randint(0, 15, n_fraud),
        'fraud_trust_rating': np.random.uniform(0, 0.3, n_fraud),
    }
    
    # Additional common features
    for data in [legitimate_data, fraud_data]:
        n = len(data['gps_spoofing_score'])
        data.update({
            'distance_from_outlet_km': np.random.uniform(0, 15, n),
            'active_hours_week': np.random.uniform(10, 60, n),
            'order_decline_rate': 1 - data['order_acceptance_rate'],
            'coordinated_fraud_cluster_id': np.random.randint(0, 5, n),
            'ip_gps_mismatch': np.random.randint(0, 2, n),
            'device_sharing_flag': np.random.randint(0, 2, n),
            'disruption_duration_hours': np.random.randint(0, 24, n),
            'income_loss_percentage': np.random.uniform(0, 0.5, n)
        })
    
    df_legitimate = pd.DataFrame(legitimate_data)
    df_fraud = pd.DataFrame(fraud_data)
    df = pd.concat([df_legitimate, df_fraud], ignore_index=True)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Train
    model = FraudDetectionModel()
    model.fit(train_df, test_df)
    
    # Predict
    predictions = model.predict(test_df.head(20))
    print("\nSample Predictions:")
    print(predictions[['fraud_probability', 'trust_rating', 'fraud_category', 'is_fraud']])