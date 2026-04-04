from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class WorkerResponse(BaseModel):
    worker_id: Any
    city: Optional[str] = None
    platform: Optional[str] = None
    employment_type: Optional[str] = None
    selected_slab: Optional[str] = None
    avg_52week_income: Optional[float] = None
    rating: Optional[float] = None
    orders_completed_week: Optional[float] = None
    claims_past_52_weeks: Optional[int] = None
    default_weeks: Optional[int] = None
    disruption_type: Optional[str] = None
    # Add any other fields mapped loosely

class WorkerListResponse(BaseModel):
    total: int
    page: int
    size: int
    workers: List[Dict[str, Any]]

class ProcessClaimRequest(BaseModel):
    worker_id: Any
    city: Optional[str] = None
    disruption_type: Optional[str] = None
    # Any overrides

class WorkflowResultResponse(BaseModel):
    trace_id: str
    worker_id: Any
    decision: str
    confidence: float
    payout_amount: float
    processing_time_ms: float
    extras: Optional[Dict[str, Any]] = None

class MessageResponse(BaseModel):
    message: str
