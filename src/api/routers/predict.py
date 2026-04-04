from fastapi import APIRouter, HTTPException, Depends
from typing import Any, Dict, Optional
from pydantic import BaseModel
import pandas as pd
import numpy as np

from src.api.dependencies import get_inference_pipeline, get_worker_data

router = APIRouter(prefix="/predict", tags=["ML Predictions"])

class PredictionRequest(BaseModel):
    worker_id: Any
    city: Optional[str] = None
    disruption_type: Optional[str] = None

class PredictionResponse(BaseModel):
    worker_id: Any
    income_forecast: float
    risk_score: float
    fraud_probability: float
    premium: float
    disruption_loss_pct: float
    raw_predictions: Dict[str, Any]

def _safe_extract(df_or_series, key=None):
    """Safely extract the first element from Pandas/Numpy structures"""
    try:
        if hasattr(df_or_series, 'iloc'):
            if key and key in df_or_series:
                return float(df_or_series[key].iloc[0])
            elif not key:
                return float(df_or_series.iloc[0])
        elif isinstance(df_or_series, dict):
            return float(df_or_series.get(key, 0.0))
        return float(np.asarray(df_or_series).ravel()[0])
    except Exception:
        return 0.0

@router.post("", response_model=PredictionResponse)
def get_prediction(request: PredictionRequest):
    df = get_worker_data()
    if df.empty:
        raise HTTPException(status_code=500, detail="Worker data not loaded")
        
    worker_df = df[df["worker_id"].astype(str) == str(request.worker_id)]
    if worker_df.empty:
        raise HTTPException(status_code=404, detail=f"Worker {request.worker_id} not found")
        
    # Get a copy for inference overrides
    worker_sample = worker_df.copy()
    if request.city is not None:
        worker_sample["city"] = request.city
    if request.disruption_type is not None:
        worker_sample["disruption_type"] = request.disruption_type
        
    inference = get_inference_pipeline()
    
    try:
        # Run ML predictions
        predictions = inference.predict_for_worker(worker_sample)
        
        income = _safe_extract(predictions.get('income_forecast'), 'ensemble')
        risk = _safe_extract(predictions.get('risk_score'), 'risk_score')
        fraud = _safe_extract(predictions.get('fraud_analysis'), 'fraud_probability')
        premium = _safe_extract(predictions.get('premium'), 'final_premium')
        disruption = _safe_extract(predictions.get('disruption_impact'))
        
        # Serialize raw models to primitive dict for arbitrary data
        raw_serializable = {}
        for k, v in predictions.items():
            if hasattr(v, 'to_dict'):
                raw_serializable[k] = v.to_dict(orient="records")
            elif isinstance(v, dict):
                raw_serializable[k] = v
            else:
                raw_serializable[k] = str(v)

        return {
            "worker_id": request.worker_id,
            "income_forecast": income,
            "risk_score": risk,
            "fraud_probability": fraud,
            "premium": premium,
            "disruption_loss_pct": disruption,
            "raw_predictions": raw_serializable
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating predictions: {str(e)}")
