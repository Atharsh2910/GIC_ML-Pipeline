"""
Data Configuration
Feature definitions and data paths
"""

# Data Paths
DATA_PATHS = {
    'raw_data': 'data/raw/final_dataset.csv',
    'processed_data': 'data/processed/processed_data.csv',
    'features': 'data/features/engineered_features.csv',
    'outputs': 'data/outputs/'
}

# Feature Groups
FEATURE_GROUPS = {
    'temporal_features': [
        'weeks_active',
        'week_of_year'
    ],
    
    'income_features': [
        'weekly_income',
        'avg_52week_income',
        'income_std_dev',
        'highest_weekly_income',
        'lowest_weekly_income',
        'income_volatility'
    ],
    
    'premium_features': [
        'base_premium',
        'premium_amount',
        'premium_paid'
    ],
    
    'behavior_features': [
        'consecutive_payment_weeks',
        'payment_consistency_score',
        'fraud_trust_rating',
        'overall_risk_score'
    ],
    
    'disruption_features': [
        'disruption_duration_hours',
        'rainfall_cm',
        'temperature_extreme',
        'cyclone_alert_level',
        'disruption_exposure_risk',
        'disruption_type'
    ],
    
    'location_features': [
        'worker_lat',
        'worker_lon',
        'outlet_lat',
        'outlet_lon',
        'distance_from_outlet_km',
        'city'
    ],
    
    'activity_features': [
        'orders_completed_week',
        'active_hours_week',
        'order_acceptance_rate',
        'order_decline_rate'
    ],
    
    'fraud_features': [
        'gps_spoofing_score',
        'movement_realism_score',
        'presence_score',
        'peer_group_activity_ratio',
        'coordinated_fraud_cluster_id',
        'ip_gps_mismatch',
        'device_sharing_flag'
    ],
    
    'claim_features': [
        'income_loss_amount',
        'income_loss_percentage',
        'coverage_percentage',
        'loyalty_bonus_percentage',
        'penalty_percentage',
        'final_payout_amount'
    ],
    
    'categorical_features': [
        'city',
        'employment_type',
        'selected_slab',
        'disruption_type',
        'platform'
    ],
    
    'identifier_features': [
        'worker_id',
        'outlet_id'
    ],
    
    'binary_features': [
        'premium_paid',
        'cooling_period_completed',
        'ip_gps_mismatch',
        'device_sharing_flag'
    ],
    # Rolling history (engineer in pipeline if you have claim-level data)
    'history_features': [
        'default_weeks',
        'claims_past_52_weeks',
    ],
}

# Target Variables for Each Model
TARGET_VARIABLES = {
    'income_forecasting': 'weekly_income',
    'premium_prediction': 'premium_amount',
    'risk_scoring': 'overall_risk_score',
    'fraud_detection': 'fraud_trust_rating',
    'disruption_impact': 'income_loss_percentage',
    'behavior_analysis': 'payment_consistency_score',
    'claim_eligibility': None,  # Deterministic
    'payout_optimization': 'final_payout_amount'
}

# Features for Each Model
MODEL_FEATURES = {
    'income_forecasting': [
        'weeks_active',
        'week_of_year',
        'weekly_income',
        'avg_52week_income',
        'income_std_dev',
        'income_volatility',
        'orders_completed_week',
        'active_hours_week',
        'disruption_duration_hours',
        'rainfall_cm',
        'temperature_extreme'
    ],
    
    'risk_scoring': [
        'income_volatility',
        'payment_consistency_score',
        'fraud_trust_rating',
        'disruption_exposure_risk',
        'distance_from_outlet_km',
        'order_acceptance_rate',
        'order_decline_rate',
        'gps_spoofing_score',
        'movement_realism_score',
        'presence_score',
        'peer_group_activity_ratio',
        'consecutive_payment_weeks'
    ],
    
    'fraud_detection': [
        'gps_spoofing_score',
        'movement_realism_score',
        'presence_score',
        'peer_group_activity_ratio',
        'coordinated_fraud_cluster_id',
        'ip_gps_mismatch',
        'device_sharing_flag',
        'distance_from_outlet_km',
        'orders_completed_week',
        'active_hours_week',
        'order_acceptance_rate',
        'order_decline_rate',
        'disruption_duration_hours',
        'income_loss_percentage'
    ],
    
    'disruption_impact': [
        'disruption_duration_hours',
        'rainfall_cm',
        'temperature_extreme',
        'cyclone_alert_level',
        'disruption_type',
        'distance_from_outlet_km',
        'orders_completed_week',
        'active_hours_week',
        'avg_52week_income',
        'income_volatility',
        'city',
        'platform'
    ],
    
    'behavior_analysis': [
        'consecutive_payment_weeks',
        'premium_paid',
        'weeks_active',
        'payment_consistency_score',
        'income_loss_percentage',
        'fraud_trust_rating',
        'overall_risk_score',
        'avg_52week_income',
        'income_volatility',
        'default_weeks',
        'claims_past_52_weeks',
        'order_acceptance_rate',
        'order_decline_rate',
        'orders_completed_week',
        'active_hours_week',
    ],
    # Hybrid premium: deterministic base + ML residual; upstream columns optional at train time
    'premium_prediction': [
        'avg_52week_income',
        'income_std_dev',
        'income_volatility',
        'payment_consistency_score',
        'fraud_trust_rating',
        'overall_risk_score',
        'consecutive_payment_weeks',
        'disruption_exposure_risk',
        'selected_slab',
        'default_weeks',
        'forecasted_weekly_income',
        'predicted_income_loss_pct',
        'behavior_score',
        'predicted_risk_score',
    ],
}

# Data Validation Rules
VALIDATION_RULES = {
    'weekly_income': {
        'min': 0,
        'max': 50000,
        'allow_null': False
    },
    'fraud_trust_rating': {
        'min': 0,
        'max': 1,
        'allow_null': False
    },
    'rainfall_cm': {
        'min': 0,
        'max': 100,
        'allow_null': True
    },
    'temperature_extreme': {
        'min': -20,
        'max': 60,
        'allow_null': True
    },
    'order_acceptance_rate': {
        'min': 0,
        'max': 1,
        'allow_null': False
    }
}

# Encoding Mappings
CATEGORICAL_ENCODINGS = {
    'employment_type': ['Full-Time', 'Part-Time', 'Occasional', 'full_time', 'part_time'],
    'selected_slab': ['Basic', 'Standard', 'Premium', 'Elite', 'Slab 1 (50%)', 'Slab 2 (75%)', 'Slab 3 (100%)'],
    'disruption_type': [
        'none', 'Heavy_Rain', 'Extreme_Heat', 'Cyclone', 'Flood', 'Infrastructure',
        'rainfall', 'heat', 'cold', 'cyclone', 'strike', 'curfew',
        'rally', 'fuel_shortage', 'network_outage', 'platform_glitch',
    ],
    'platform': ['Zepto', 'Blinkit', 'Instamart'],
    'city': []  # Will be populated from data
}

# Feature Engineering Parameters
FEATURE_ENGINEERING = {
    'income_derived': {
        'income_z_score': True,
        'income_percentile': True,
        'income_trend': True,
        'income_deviation_from_avg': True
    },
    'temporal_derived': {
        'is_month_start': True,
        'is_month_end': True,
        'is_quarter_start': True,
        'season': True
    },
    'interaction_features': {
        'income_volatility_x_disruption_risk': True,
        'fraud_score_x_risk_score': True,
        'distance_x_orders': True
    },
    'aggregation_features': {
        'rolling_windows': [4, 8, 12, 26],  # weeks
        'aggregations': ['mean', 'std', 'min', 'max']
    }
}

# Model Save Paths
MODEL_SAVE_PATHS = {
    'income_forecasting': 'models/income_forecasting/',
    'premium_prediction': 'models/premium_prediction/',
    'risk_scoring': 'models/risk_scoring/',
    'fraud_detection': 'models/fraud_detection/',
    'disruption_impact': 'models/disruption_impact/',
    'behavior_analysis': 'models/behavior_analysis/'
}