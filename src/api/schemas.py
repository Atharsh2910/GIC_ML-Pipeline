"""Request/response models for the GigShield HTTP API (module-level for OpenAPI)."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class WorkerRecord(BaseModel):
    worker_id: int
    city: str = "Mumbai"
    avg_52week_income: float = 0.0
    disruption_type: Optional[str] = "none"
    selected_slab: Optional[str] = "Diamond"
    income_loss_percentage: Optional[float] = 0.0

    model_config = {"extra": "allow"}


class EvaluateWorkerRequest(BaseModel):
    worker: WorkerRecord
    city: Optional[str] = None
    context_question: Optional[str] = None
    include_graph_state: bool = Field(
        default=False,
        description="Include full LangGraph state (large JSON).",
    )


class OrchestrateBatchRequest(BaseModel):
    workers: List[WorkerRecord]
    city: Optional[str] = None
    context_question: Optional[str] = None
    include_graph_state: bool = False


class ClassicClaimRequest(BaseModel):
    worker: WorkerRecord
    city: Optional[str] = None


class InferencePredictRequest(BaseModel):
    worker: WorkerRecord


class RAGRetrieveRequest(BaseModel):
    query: str
    categories: Optional[List[str]] = None
    top_k: Optional[int] = None
