"""
Income Forecasting Model
Ensemble: Rolling Mean + LSTM + SARIMAX
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
import joblib
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('..')
from config.model_config import INCOME_FORECAST_CONFIG


class IncomeForecastingModel:
    """
    Ensemble income forecasting using three approaches:
    1. Rolling Mean (52-week)
    2. LSTM Neural Network
    3. SARIMAX Time Series
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or INCOME_FORECAST_CONFIG
        self.scaler = StandardScaler()
        self.lstm_model = None
        self.sarimax_model = None
        self.ensemble_weights = self.config['ensemble_weights']
        
    def prepare_sequences(self, data: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM"""
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i-lookback:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    def build_lstm_model(self, input_shape: Tuple) -> Sequential:
        """Build LSTM architecture"""
        lstm_config = self.config['lstm']
        
        model = Sequential([
            LSTM(lstm_config['units'][0], 
                 return_sequences=True,
                 input_shape=input_shape,
                 dropout=lstm_config['dropout'],
                 recurrent_dropout=lstm_config['recurrent_dropout']),
            
            LSTM(lstm_config['units'][1], 
                 return_sequences=True,
                 dropout=lstm_config['dropout'],
                 recurrent_dropout=lstm_config['recurrent_dropout']),
            
            LSTM(lstm_config['units'][2],
                 dropout=lstm_config['dropout']),
            
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        
        optimizer = Adam(learning_rate=lstm_config['learning_rate'])
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def rolling_mean_forecast(self, income_series: pd.Series) -> float:
        """Calculate 52-week rolling mean forecast"""
        window = self.config['rolling_mean']['window']
        min_periods = self.config['rolling_mean']['min_periods']
        
        if len(income_series) < min_periods:
            return income_series.mean()
        
        rolling_mean = income_series.rolling(
            window=min(window, len(income_series)),
            min_periods=min_periods
        ).mean()
        
        return rolling_mean.iloc[-1]
    
    def train_lstm(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray) -> Sequential:
        """Train LSTM model"""
        lstm_config = self.config['lstm']
        
        # Scale data
        X_train_scaled = self.scaler.fit_transform(
            X_train.reshape(-1, X_train.shape[-1])
        ).reshape(X_train.shape)
        
        X_val_scaled = self.scaler.transform(
            X_val.reshape(-1, X_val.shape[-1])
        ).reshape(X_val.shape)
        
        # Build model
        self.lstm_model = self.build_lstm_model((X_train.shape[1], X_train.shape[2]))
        
        # Early stopping
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=lstm_config['early_stopping_patience'],
            restore_best_weights=True
        )
        
        # Train
        self.lstm_model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=lstm_config['epochs'],
            batch_size=lstm_config['batch_size'],
            callbacks=[early_stop],
            verbose=1
        )
        
        return self.lstm_model
    
    def train_sarimax(self, income_series: pd.Series) -> SARIMAX:
        """Train SARIMAX model"""
        sarimax_config = self.config['sarimax']
        
        try:
            model = SARIMAX(
                income_series,
                order=sarimax_config['order'],
                seasonal_order=sarimax_config['seasonal_order'],
                enforce_stationarity=sarimax_config['enforce_stationarity'],
                enforce_invertibility=sarimax_config['enforce_invertibility']
            )
            
            self.sarimax_model = model.fit(disp=False)
            
        except Exception as e:
            print(f"SARIMAX training failed: {e}")
            self.sarimax_model = None
        
        return self.sarimax_model
    
    def fit(self, df: pd.DataFrame, worker_id: int = None) -> 'IncomeForecastingModel':
        """
        Train all three models
        
        Args:
            df: DataFrame with columns ['worker_id', 'week', 'weekly_income', ...]
            worker_id: Optional specific worker ID to train on
        """
        # Filter by worker if specified
        if worker_id:
            df = df[df['worker_id'] == worker_id].copy()
        
        # Sort by time
        df = df.sort_values('weeks_active')
        income_series = df['weekly_income']
        
        print("Training Income Forecasting Ensemble...")
        
        # 1. Rolling Mean (no training required)
        print("✓ Rolling Mean: Ready")
        
        # 2. Train LSTM
        lookback = self.config['lstm']['lookback']
        
        if len(income_series) >= lookback + 10:
            # Prepare sequences
            income_array = income_series.values.reshape(-1, 1)
            X, y = self.prepare_sequences(income_array, lookback)
            
            # Split
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            self.train_lstm(X_train, y_train, X_val, y_val)
            print("✓ LSTM: Trained")
        else:
            print("⚠ LSTM: Insufficient data, skipping")
        
        # 3. Train SARIMAX
        if len(income_series) >= 60:  # Need sufficient data for SARIMAX
            self.train_sarimax(income_series)
            print("✓ SARIMAX: Trained")
        else:
            print("⚠ SARIMAX: Insufficient data, skipping")
        
        return self
    
    def predict(self, df: pd.DataFrame, worker_id: int) -> Dict[str, float]:
        """
        Predict next week's income using ensemble
        
        Returns:
            Dict with predictions from each model and ensemble prediction
        """
        worker_data = df[df['worker_id'] == worker_id].sort_values('weeks_active')
        income_series = worker_data['weekly_income']
        
        predictions = {}
        
        # 1. Rolling Mean Prediction
        rolling_pred = self.rolling_mean_forecast(income_series)
        predictions['rolling_mean'] = rolling_pred
        
        # 2. LSTM Prediction
        if self.lstm_model is not None:
            lookback = self.config['lstm']['lookback']
            if len(income_series) >= lookback:
                recent_data = income_series.values[-lookback:].reshape(1, lookback, 1)
                recent_scaled = self.scaler.transform(recent_data.reshape(-1, 1)).reshape(1, lookback, 1)
                lstm_pred = self.lstm_model.predict(recent_scaled, verbose=0)[0][0]
                predictions['lstm'] = lstm_pred
            else:
                predictions['lstm'] = rolling_pred  # Fallback
        else:
            predictions['lstm'] = rolling_pred
        
        # 3. SARIMAX Prediction
        if self.sarimax_model is not None:
            try:
                sarimax_forecast = self.sarimax_model.forecast(steps=1)
                predictions['sarimax'] = sarimax_forecast.iloc[0]
            except:
                predictions['sarimax'] = rolling_pred  # Fallback
        else:
            predictions['sarimax'] = rolling_pred
        
        # Ensemble Prediction
        ensemble_pred = (
            predictions['rolling_mean'] * self.ensemble_weights['rolling_mean'] +
            predictions['lstm'] * self.ensemble_weights['lstm'] +
            predictions['sarimax'] * self.ensemble_weights['sarimax']
        )
        
        predictions['ensemble'] = ensemble_pred
        predictions['confidence_interval_lower'] = ensemble_pred * 0.85
        predictions['confidence_interval_upper'] = ensemble_pred * 1.15
        
        return predictions
    
    def batch_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict for all workers in dataframe"""
        worker_ids = df['worker_id'].unique()
        predictions_list = []
        
        for worker_id in worker_ids:
            pred = self.predict(df, worker_id)
            pred['worker_id'] = worker_id
            predictions_list.append(pred)
        
        return pd.DataFrame(predictions_list)
    
    def save(self, path: str):
        """Save all models"""
        # Save LSTM
        if self.lstm_model:
            self.lstm_model.save(f"{path}/lstm_model.h5")
        
        # Save SARIMAX
        if self.sarimax_model:
            self.sarimax_model.save(f"{path}/sarimax_model.pkl")
        
        # Save scaler
        joblib.dump(self.scaler, f"{path}/scaler.pkl")
        
        # Save config
        joblib.dump(self.config, f"{path}/config.pkl")
        
        print(f"✓ Models saved to {path}")
    
    def load(self, path: str):
        """Load all models"""
        try:
            self.lstm_model = load_model(f"{path}/lstm_model.h5")
        except:
            print("⚠ LSTM model not found")
        
        try:
            self.sarimax_model = joblib.load(f"{path}/sarimax_model.pkl")
        except:
            print("⚠ SARIMAX model not found")
        
        self.scaler = joblib.load(f"{path}/scaler.pkl")
        self.config = joblib.load(f"{path}/config.pkl")
        
        print(f"✓ Models loaded from {path}")
        return self


# Example usage
if __name__ == "__main__":
    # Create synthetic data for testing
    np.random.seed(42)
    n_weeks = 104
    n_workers = 100
    
    data = []
    for worker_id in range(n_workers):
        base_income = np.random.uniform(5000, 12000)
        for week in range(n_weeks):
            weekly_income = base_income + np.random.normal(0, 1000) + 500 * np.sin(week / 52 * 2 * np.pi)
            data.append({
                'worker_id': worker_id,
                'weeks_active': week + 1,
                'weekly_income': max(0, weekly_income)
            })
    
    df = pd.DataFrame(data)
    
    # Train model
    model = IncomeForecastingModel()
    model.fit(df, worker_id=0)
    
    # Predict
    predictions = model.predict(df, worker_id=0)
    print("\nPredictions for Worker 0:")
    print(predictions)
    
    # Save
    model.save('models/income_forecasting')