import os
from pathlib import Path
import pandas as pd
from typing import Optional

from fastapi import Request

from src.pipeline.orchestrator import GigShieldOrchestrator
from src.pipeline.training_pipeline import InferencePipeline
from src.rag.rag_system import VectorStore, populate_knowledge_base
from src.utils.schema import ensure_worker_columns

_WORKER_DATA: Optional[pd.DataFrame] = None
_ORCHESTRATOR: Optional[GigShieldOrchestrator] = None

def get_worker_data() -> pd.DataFrame:
    global _WORKER_DATA
    if _WORKER_DATA is None:
        data_path = os.getenv("GIGSHIELD_DATA_PATH", "data/raw/final_dataset.csv")
        p = Path(data_path)
        if not p.is_file():
            # Fallback
            p = Path("data/raw/quick_commerce_synthetic_data52k.csv")
        if p.is_file():
            df = pd.read_csv(p)
            _WORKER_DATA = ensure_worker_columns(df)
        else:
            _WORKER_DATA = pd.DataFrame()
    return _WORKER_DATA

def get_orchestrator() -> GigShieldOrchestrator:
    global _ORCHESTRATOR
    if _ORCHESTRATOR is None:
        model_dir = os.getenv("GIGSHIELD_MODEL_DIR", "models")
        model_paths = {
            "income_forecasting": f"{model_dir}/income_forecasting",
            "risk_scoring": f"{model_dir}/risk_scoring",
            "fraud_detection": f"{model_dir}/fraud_detection",
            "disruption_impact": f"{model_dir}/disruption_impact",
            "behavior_analysis": f"{model_dir}/behavior_analysis",
            "premium_prediction": f"{model_dir}/premium_prediction",
        }
        
        # Load models safely (they might not exist if not trained, but InferencePipeline handles it)
        inference = get_inference_pipeline()
        
        # Vector Store
        vector_store = VectorStore()
        # Optionally populate KB here if empty, but assuming it's populated offline
        
        _ORCHESTRATOR = GigShieldOrchestrator(
            inference_pipeline=inference,
            vector_store=vector_store
        )
    return _ORCHESTRATOR

_INFERENCE_PIPELINE: Optional[InferencePipeline] = None

def get_inference_pipeline() -> InferencePipeline:
    global _INFERENCE_PIPELINE
    if _INFERENCE_PIPELINE is None:
        model_dir = os.getenv("GIGSHIELD_MODEL_DIR", "models")
        model_paths = {
            "income_forecasting": f"{model_dir}/income_forecasting",
            "risk_scoring": f"{model_dir}/risk_scoring",
            "fraud_detection": f"{model_dir}/fraud_detection",
            "disruption_impact": f"{model_dir}/disruption_impact",
            "behavior_analysis": f"{model_dir}/behavior_analysis",
            "premium_prediction": f"{model_dir}/premium_prediction",
        }
        _INFERENCE_PIPELINE = InferencePipeline(model_paths)
    return _INFERENCE_PIPELINE
