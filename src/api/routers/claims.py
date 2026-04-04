from fastapi import APIRouter, HTTPException, Depends
from typing import Any, Dict

from src.api.schemas import ProcessClaimRequest, WorkflowResultResponse
from src.api.dependencies import get_orchestrator, get_worker_data

router = APIRouter(prefix="/claims", tags=["Claims Workflow"])

@router.post("/process", response_model=WorkflowResultResponse)
async def process_claim(request: ProcessClaimRequest):
    df = get_worker_data()
    if df.empty:
        raise HTTPException(status_code=500, detail="Worker data not loaded")
        
    worker_df = df[df["worker_id"].astype(str) == str(request.worker_id)]
    if worker_df.empty:
        raise HTTPException(status_code=404, detail=f"Worker {request.worker_id} not found")
        
    # Optional override from request
    worker_sample = worker_df.copy()
    if request.city is not None:
        worker_sample["city"] = request.city
    if request.disruption_type is not None:
        worker_sample["disruption_type"] = request.disruption_type
        
    orchestrator = get_orchestrator()
    
    try:
        # Await the process claim workflow
        result = await orchestrator.process_claim(worker_sample)
        
        # Serialize the dataclass output for JSON
        return {
            "trace_id": result.trace_id,
            "worker_id": result.worker_id,
            "decision": result.decision,
            "confidence": result.confidence,
            "payout_amount": result.payout_amount,
            "processing_time_ms": result.processing_time_ms,
            "extras": result.extras
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing claim: {str(e)}")
