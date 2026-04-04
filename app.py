import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 1. Path Setup & Environment
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

load_dotenv(_ROOT / ".env")

# 2. Lazy imports so FastAPI parses without massive ML delays if misconfigured
from src.pipeline.training_pipeline import InferencePipeline
from src.agents.gigshield_langgraph import GigShieldLangGraphOrchestrator
from src.utils.schema import ensure_worker_columns

# Global instances
inference_pipeline: Optional[InferencePipeline] = None
orchestrator: Optional[GigShieldLangGraphOrchestrator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan function to strictly manage startup and shutdown events.
    Loads models into memory once and starts AI orchestration agents.
    """
    global inference_pipeline, orchestrator
    print("\n🚀 Initializing GigShield Backend Models and AI Agents...")

    model_dir = "models"
    model_paths = {
        "income_forecasting": f"{model_dir}/income_forecasting",
        "risk_scoring": f"{model_dir}/risk_scoring",
        "fraud_detection": f"{model_dir}/fraud_detection",
        "disruption_impact": f"{model_dir}/disruption_impact",
        "behavior_analysis": f"{model_dir}/behavior_analysis",
        "premium_prediction": f"{model_dir}/premium_prediction",
    }

    try:
        inference_pipeline = InferencePipeline(model_paths)
        print("✅ Inference models loaded successfully!")
    except Exception as e:
        print(f"⚠️ Warning: Could not fully load InferencePipeline models: {e}")
        inference_pipeline = None

    try:
        # The orchestrator requires OPENAI_API_KEY inside the .env
        orchestrator = GigShieldLangGraphOrchestrator(
            inference_pipeline=inference_pipeline, 
            ensure_kb=True
        )
        print("✅ LangGraph Agents Initialized successfully!")
    except Exception as e:
        print(f"⚠️ Warning: Could not initialize Orchestrator natively (Check API keys): {e}")
        orchestrator = None

    yield
    print("🛑 Shutting down GigShield Backend...")


app = FastAPI(
    title="GigShield Autonomous AI API",
    description="Backend endpoints mapping the RAG + MCP Multi-Agent System architecture via LangGraph.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------

class WorkerRecord(BaseModel):
    worker_id: int
    city: str
    avg_52week_income: float
    disruption_type: Optional[str] = "none"
    selected_slab: Optional[str] = "Diamond"
    income_loss_percentage: Optional[float] = 0.0
    
    # Enable fetching any arbitrary columns like `claims_past_52_weeks`, etc.
    model_config = {
        "extra": "allow"
    }

class EvaluateWorkerRequest(BaseModel):
    worker: WorkerRecord
    city: Optional[str] = None
    context_question: Optional[str] = None

class OrchestrateBatchRequest(BaseModel):
    workers: List[WorkerRecord]
    city: Optional[str] = None
    context_question: Optional[str] = None


# ---------------------------------------------------------
# Endpoints
# ---------------------------------------------------------

@app.get("/health", tags=["System"])
def health_check():
    """Returns the operational status of ML pipelines & AI agents."""
    return {
        "status": "online",
        "models_loaded": inference_pipeline is not None,
        "agents_ready": orchestrator is not None
    }


@app.post("/api/evaluate_worker", tags=["Evaluation"])
async def evaluate_worker(req: EvaluateWorkerRequest):
    """
    Executes the End-to-End Core Architecture on a single Worker:
    1. Triggers real-time monitoring specific to the worker's city
    2. Validates MCP outputs against thresholds
    3. Runs ML + Specialist parallel Agents (Fraud, Risk, Policy)
    4. Persists the decision trajectory autonomously using DecisionOrchestrator
    """
    if not orchestrator:
        raise HTTPException(
            status_code=503, 
            detail="AI Orchestrator is offline. Ensure OPENAI_API_KEY is configured."
        )

    try:
        worker_dict = req.worker.model_dump()
        df_input = pd.DataFrame([worker_dict])
        
        target_city = req.city or req.worker.city
        
        result = await orchestrator.run(
            worker_data=df_input, 
            city=target_city,
            context_question=req.context_question
        )
        
        return {
            "trace_id": result.trace_id,
            "worker_id": result.worker_id,
            "processing_time_ms": result.processing_time_ms,
            "decision": result.final_state.get("decision_code"),
            "confidence": result.final_state.get("confidence"),
            "payout_amount": result.final_state.get("payout_amount"),
            
            # Additional granular metrics extracted from the Graph
            "eligibility_snapshot": result.final_state.get("eligibility_snapshot"),
            "payout_breakdown": result.final_state.get("payout_breakdown"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/orchestrate", tags=["Evaluation"])
async def orchestrate_batch(req: OrchestrateBatchRequest):
    """
    Batch processing Endpoint.
    Sequentially processes a list of affected workers to generate unified ML predictions + Agent recommendations.
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="AI Orchestrator is not initialized.")
    if not req.workers:
        raise HTTPException(status_code=400, detail="Worker payload cannot be empty.")
        
    results = []
    try:
        for worker in req.workers:
            worker_dict = worker.model_dump()
            df_input = pd.DataFrame([worker_dict])
            
            target_city = req.city or worker.city
            
            res = await orchestrator.run(
                worker_data=df_input,
                city=target_city,
                context_question=req.context_question
            )
            
            results.append({
                "trace_id": res.trace_id,
                "worker_id": res.worker_id,
                "decision": res.final_state.get("decision_code"),
                "confidence": res.final_state.get("confidence"),
                "payout_amount": res.final_state.get("payout_amount"),
                "processing_time_ms": res.processing_time_ms
            })
            
        return {
            "batch_size": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch execution failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
