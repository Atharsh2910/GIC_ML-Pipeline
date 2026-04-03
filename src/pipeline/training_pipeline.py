"""
Complete ML Training Pipeline
Orchestrates training of all models
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import joblib
from pathlib import Path

import sys

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models.income_forecasting import IncomeForecastingModel
from src.models.risk_scoring import RiskScoringModel
from src.models.fraud_detection import FraudDetectionModel
from src.models.additional_models import (DisruptionImpactModel, BehaviorAnalysisModel, 
                                           PremiumPredictionModel)
from src.models.deterministic_models import ClaimEligibilityModel, PayoutOptimizationModel
from config.data_config import DATA_PATHS, MODEL_SAVE_PATHS
from src.utils.schema import ensure_worker_columns


class MLTrainingPipeline:
    """
    End-to-end training pipeline for all GigShield models
    """
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.models = {}
        self.training_results = {}
        
    def load_data(self) -> pd.DataFrame:
        """Load and validate training data"""
        print("=" * 60)
        print("LOADING DATA")
        print("=" * 60)
        
        df = pd.DataFrame(data=pd.read_csv(self.data_path))
        print(f"✓ Loaded {len(df)} records")
        print(f"✓ Features: {list(df.columns)}")
        print(f"✓ Date range: Week {df['week_of_year'].min()} to {df['week_of_year'].max()}")
        
        return df
    
    def split_data(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Split data into train/val/test"""
        print("\nSplitting data...")
        
        # Sort by time
        df = df.sort_values('weeks_active')
        
        # 70% train, 15% val, 15% test (time-based split)
        n = len(df)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)
        
        splits = {
            'train': df.iloc[:train_end],
            'val': df.iloc[train_end:val_end],
            'test': df.iloc[val_end:]
        }
        
        for split_name, split_df in splits.items():
            print(f"  {split_name}: {len(split_df)} records")
        
        return splits
    
    def train_income_forecasting(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """Train income forecasting ensemble"""
        print("\n" + "=" * 60)
        print("TRAINING INCOME FORECASTING MODEL")
        print("=" * 60)
        
        model = IncomeForecastingModel()
        
        # Train for multiple workers
        worker_ids = train_df['worker_id'].unique()[:50]  # Sample of workers
        
        for worker_id in worker_ids:
            model.fit(train_df, worker_id=worker_id)
        
        self.models['income_forecasting'] = model
        self.training_results['income_forecasting'] = {'status': 'success'}
        
        # Save
        save_path = MODEL_SAVE_PATHS['income_forecasting']
        Path(save_path).mkdir(parents=True, exist_ok=True)
        model.save(save_path)
        
        return model
    
    def train_risk_scoring(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """Train risk scoring ensemble"""
        print("\n" + "=" * 60)
        print("TRAINING RISK SCORING MODEL")
        print("=" * 60)
        
        model = RiskScoringModel()
        model.fit(train_df, val_df)
        
        self.models['risk_scoring'] = model
        self.training_results['risk_scoring'] = {'status': 'success'}
        
        # Save
        save_path = MODEL_SAVE_PATHS['risk_scoring']
        Path(save_path).mkdir(parents=True, exist_ok=True)
        model.save(save_path)
        
        return model
    
    def train_fraud_detection(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """Train fraud detection hybrid model"""
        print("\n" + "=" * 60)
        print("TRAINING FRAUD DETECTION MODEL")
        print("=" * 60)
        
        model = FraudDetectionModel()
        model.fit(train_df, val_df)
        
        self.models['fraud_detection'] = model
        self.training_results['fraud_detection'] = {'status': 'success'}
        
        # Save
        save_path = MODEL_SAVE_PATHS['fraud_detection']
        Path(save_path).mkdir(parents=True, exist_ok=True)
        model.save(save_path)
        
        return model
    
    def train_disruption_impact(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """Train disruption impact model"""
        print("\n" + "=" * 60)
        print("TRAINING DISRUPTION IMPACT MODEL")
        print("=" * 60)
        
        model = DisruptionImpactModel()
        model.fit(train_df)
        
        self.models['disruption_impact'] = model
        self.training_results['disruption_impact'] = {'status': 'success'}
        
        # Save
        save_path = MODEL_SAVE_PATHS['disruption_impact']
        Path(save_path).mkdir(parents=True, exist_ok=True)
        model.save(save_path)
        
        return model
    
    def train_behavior_analysis(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """Train behavior analysis model"""
        print("\n" + "=" * 60)
        print("TRAINING BEHAVIOR ANALYSIS MODEL")
        print("=" * 60)
        
        model = BehaviorAnalysisModel()
        model.fit(train_df)
        
        self.models['behavior_analysis'] = model
        self.training_results['behavior_analysis'] = {'status': 'success'}
        
        # Save
        save_path = MODEL_SAVE_PATHS['behavior_analysis']
        Path(save_path).mkdir(parents=True, exist_ok=True)
        model.save(save_path)
        
        return model
    
    def train_premium_prediction(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """Train premium prediction model (enriched with disruption + behavior signals when available)."""
        print("\n" + "=" * 60)
        print("TRAINING PREMIUM PREDICTION MODEL")
        print("=" * 60)

        enriched = ensure_worker_columns(train_df.copy())
        if "predicted_income_loss_pct" not in enriched.columns and "disruption_impact" in self.models:
            try:
                enriched["predicted_income_loss_pct"] = self.models["disruption_impact"].predict(enriched)
            except Exception as exc:
                print(f"⚠ Disruption predict for premium enrichment skipped: {exc}")
                enriched["predicted_income_loss_pct"] = 0.0
        if "behavior_score" not in enriched.columns and "behavior_analysis" in self.models:
            try:
                beh = self.models["behavior_analysis"].predict(enriched)
                enriched["behavior_score"] = beh["behavior_score"].values
            except Exception as exc:
                print(f"⚠ Behavior predict for premium enrichment skipped: {exc}")
                enriched["behavior_score"] = 0.5
        if "predicted_risk_score" not in enriched.columns and "risk_scoring" in self.models:
            try:
                rs = self.models["risk_scoring"].predict(enriched)
                enriched["predicted_risk_score"] = rs["risk_score"].values
            except Exception as exc:
                print(f"⚠ Risk predict for premium enrichment skipped: {exc}")
                enriched["predicted_risk_score"] = enriched.get("overall_risk_score", 0.5)
        if "forecasted_weekly_income" not in enriched.columns:
            enriched["forecasted_weekly_income"] = enriched["avg_52week_income"]

        model = PremiumPredictionModel()
        model.fit(enriched)

        self.models["premium_prediction"] = model
        self.training_results["premium_prediction"] = {"status": "success"}

        save_path = MODEL_SAVE_PATHS["premium_prediction"]
        Path(save_path).mkdir(parents=True, exist_ok=True)
        model.save(save_path)

        return model
    
    def initialize_deterministic_models(self):
        """Initialize deterministic models (no training required)"""
        print("\n" + "=" * 60)
        print("INITIALIZING DETERMINISTIC MODELS")
        print("=" * 60)
        
        self.models['claim_eligibility'] = ClaimEligibilityModel()
        print("✓ Claim Eligibility Model initialized")
        
        self.models['payout_optimization'] = PayoutOptimizationModel()
        print("✓ Payout Optimization Model initialized")
        
        self.training_results['deterministic'] = {'status': 'success'}
    
    def run_complete_training(self):
        """Execute complete training pipeline"""
        print("\n" + "=" * 70)
        print(" " * 15 + "GIGSHIELD ML TRAINING PIPELINE")
        print("=" * 70)
        
        # Load data
        df = self.load_data()
        
        # Split data
        splits = self.split_data(df)
        train_df = splits['train']
        val_df = splits['val']
        test_df = splits['test']
        
        # Train all models
        try:
            # 1. Income Forecasting
            self.train_income_forecasting(train_df, val_df)
            
            # 2. Risk Scoring
            self.train_risk_scoring(train_df, val_df)
            
            # 3. Fraud Detection
            self.train_fraud_detection(train_df, val_df)
            
            # 4. Disruption Impact
            self.train_disruption_impact(train_df, val_df)
            
            # 5. Behavior Analysis
            self.train_behavior_analysis(train_df, val_df)
            
            # 6. Premium Prediction
            self.train_premium_prediction(train_df, val_df)
            
            # 7. Deterministic Models
            self.initialize_deterministic_models()
            
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            raise
        
        # Training Summary
        self.print_training_summary()
        
        return self.models
    
    def print_training_summary(self):
        """Print training summary"""
        print("\n" + "=" * 70)
        print(" " * 20 + "TRAINING SUMMARY")
        print("=" * 70)
        
        for model_name, result in self.training_results.items():
            status = "✓" if result['status'] == 'success' else "✗"
            print(f"{status} {model_name.replace('_', ' ').title()}: {result['status']}")
        
        print("\n✓ All models trained successfully!")
        print(f"✓ Models saved to: models/")
        print("=" * 70)


class InferencePipeline:
    """
    Load PKL/H5 artifacts from models/ (train locally or copy from Kaggle).
    """

    def __init__(self, model_paths: Dict[str, str]):
        self.models: Dict[str, object] = {}
        self.model_paths = model_paths
        self.load_models(model_paths)

    def load_models(self, model_paths: Dict[str, str]) -> None:
        print("Loading models...")
        self.models["income_forecasting"] = IncomeForecastingModel().load(
            model_paths.get("income_forecasting", "models/income_forecasting")
        )
        self.models["risk_scoring"] = RiskScoringModel().load(model_paths.get("risk_scoring", "models/risk_scoring"))
        self.models["fraud_detection"] = FraudDetectionModel().load(
            model_paths.get("fraud_detection", "models/fraud_detection")
        )
        self.models["disruption_impact"] = DisruptionImpactModel().load(
            model_paths.get("disruption_impact", "models/disruption_impact")
        )
        self.models["behavior_analysis"] = BehaviorAnalysisModel().load(
            model_paths.get("behavior_analysis", "models/behavior_analysis")
        )
        self.models["premium_prediction"] = PremiumPredictionModel().load(
            model_paths.get("premium_prediction", "models/premium_prediction")
        )
        self.models["claim_eligibility"] = ClaimEligibilityModel()
        self.models["payout_optimization"] = PayoutOptimizationModel()
        print("✓ All models loaded")

    def predict_for_worker(self, worker_data: pd.DataFrame) -> Dict[str, object]:
        """Ordered pipeline: income → risk, fraud, disruption, behavior → premium (with upstream features)."""
        df = ensure_worker_columns(worker_data.copy())
        wid = int(df["worker_id"].iloc[0])

        results: Dict[str, object] = {}

        results["income_forecast"] = self.models["income_forecasting"].predict(df, wid)
        ens = float(results["income_forecast"].get("ensemble", df["avg_52week_income"].iloc[0]))
        df["forecasted_weekly_income"] = ens

        results["risk_score"] = self.models["risk_scoring"].predict(df)
        df["predicted_risk_score"] = results["risk_score"]["risk_score"].values

        results["fraud_analysis"] = self.models["fraud_detection"].predict(df)

        dip = self.models["disruption_impact"].predict(df)
        results["disruption_impact"] = dip
        df["predicted_income_loss_pct"] = np.asarray(dip).reshape(-1)

        beh = self.models["behavior_analysis"].predict(df)
        results["behavior_score"] = beh
        df["behavior_score"] = beh["behavior_score"].values

        results["premium"] = self.models["premium_prediction"].predict(df)

        if "weekly_income" in df.columns:
            results["claim_eligibility"] = self.models["claim_eligibility"].evaluate_eligibility(df.iloc[0].to_dict())

        return results


if __name__ == "__main__":
    # Generate synthetic training data for testing
    print("Generating synthetic training data...")
    
    np.random.seed(42)
    n_workers = 1000
    n_weeks_per_worker = 52
    
    data = []
    for worker_id in range(n_workers):
        base_income = np.random.uniform(5000, 15000)
        
        for week in range(n_weeks_per_worker):
            # Simulate income with seasonality
            seasonal_factor = 1 + 0.2 * np.sin(week / 52 * 2 * np.pi)
            weekly_income = base_income * seasonal_factor + np.random.normal(0, 1000)
            
            data.append({
                'worker_id': worker_id,
                'weeks_active': week + 1,
                'week_of_year': (week % 52) + 1,
                'weekly_income': max(0, weekly_income),
                'avg_52week_income': base_income,
                'income_std_dev': 1000,
                'income_volatility': np.random.uniform(0.1, 0.5),
                'base_premium': base_income * 0.04,
                'premium_amount': base_income * 0.04 * np.random.uniform(0.9, 1.2),
                'consecutive_payment_weeks': np.random.randint(0, week + 1),
                'payment_consistency_score': np.random.uniform(0.5, 1),
                'fraud_trust_rating': np.random.uniform(0.3, 1),
                'overall_risk_score': np.random.uniform(0, 1),
                'disruption_duration_hours': np.random.randint(0, 24),
                'rainfall_cm': np.random.uniform(0, 25),
                'orders_completed_week': np.random.randint(10, 60),
                'active_hours_week': np.random.uniform(20, 60),
                'gps_spoofing_score': np.random.uniform(0, 0.5),
                'movement_realism_score': np.random.uniform(0.6, 1),
                'presence_score': np.random.uniform(0.6, 1),
                'peer_group_activity_ratio': np.random.uniform(0.8, 1.2),
                'income_loss_percentage': np.random.uniform(0, 0.5),
                'selected_slab': np.random.choice(['Basic', 'Standard', 'Premium', 'Elite']),
                'premium_paid': np.random.randint(0, 2),
                'cooling_period_completed': 1 if week >= 8 else 0,
                'order_acceptance_rate': np.random.uniform(0.6, 1),
                'order_decline_rate': np.random.uniform(0, 0.4),
                'distance_from_outlet_km': np.random.uniform(0, 15),
                'coordinated_fraud_cluster_id': np.random.randint(0, 5),
                'disruption_exposure_risk': np.random.uniform(0, 1),
                'city': np.random.choice(['Mumbai', 'Delhi', 'Bengaluru']),
                'platform': np.random.choice(['Zepto', 'Blinkit', 'Instamart']),
                'disruption_type': np.random.choice(['rainfall', 'heat', 'cyclone']),
                'temperature_extreme': np.random.uniform(0, 45),
                'cyclone_alert_level': np.random.randint(0, 3),
                'default_weeks': np.random.randint(0, 6),
                'claims_past_52_weeks': np.random.randint(0, 4),
                'employment_type': 'full_time',
                'ip_gps_mismatch': np.random.randint(0, 2),
                'device_sharing_flag': np.random.randint(0, 2),
                'highest_weekly_income': base_income * 1.2,
                'lowest_weekly_income': base_income * 0.6,
                'income_loss_amount': 0.0,
                'coverage_percentage': np.random.uniform(0.4, 1.0),
                'loyalty_bonus_percentage': np.random.uniform(0, 0.15),
                'penalty_percentage': np.random.uniform(0, 0.1),
                'final_payout_amount': 0.0,
                'outlet_id': np.random.randint(1, 500),
                'worker_lat': 19.0 + np.random.randn() * 0.01,
                'worker_lon': 72.8 + np.random.randn() * 0.01,
                'outlet_lat': 19.0 + np.random.randn() * 0.01,
                'outlet_lon': 72.8 + np.random.randn() * 0.01,
            })
    
    df = pd.DataFrame(data)
    
    # Save synthetic data
    Path('data/raw').mkdir(parents=True, exist_ok=True)
    df.to_csv('data/raw/worker_data.csv', index=False)
    print(f"✓ Generated {len(df)} training records")
    
    # Run training pipeline
    pipeline = MLTrainingPipeline('data/raw/worker_data.csv')
    trained_models = pipeline.run_complete_training()
    
    print("\n✓ Training pipeline complete!")