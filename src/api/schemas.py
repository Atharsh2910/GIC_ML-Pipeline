"""Request/response models for the GigShield HTTP API (module-level for OpenAPI)."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class WorkerRecord(BaseModel):
    worker_id: int = 1
    city: str = "Mumbai"
    avg_52week_income: float = 7500.0
    disruption_type: Optional[str] = "Heavy_Rain"
    selected_slab: Optional[str] = "Slab_50"
    income_loss_percentage: Optional[float] = 35.0
    employment_type: Optional[str] = "Full-Time"
    platform: Optional[str] = "Zepto"
    premium_paid: Optional[int] = 1
    cooling_period_completed: Optional[int] = 1
    weeks_active: Optional[int] = 26
    week_of_year: Optional[int] = 20
    weekly_income: Optional[float] = 5000.0
    income_std_dev: Optional[float] = 200.0
    income_volatility: Optional[float] = 0.1
    orders_completed_week: Optional[int] = 50
    active_hours_week: Optional[float] = 40.0
    disruption_duration_hours: Optional[float] = 2.0
    rainfall_cm: Optional[float] = 5.0
    temperature_extreme: Optional[float] = 28.0
    cyclone_alert_level: Optional[int] = 0
    payment_consistency_score: Optional[float] = 0.9
    fraud_trust_rating: Optional[float] = 0.8
    overall_risk_score: Optional[float] = 0.2
    disruption_exposure_risk: Optional[float] = 0.1
    distance_from_outlet_km: Optional[float] = 5.0
    order_acceptance_rate: Optional[float] = 0.9
    order_decline_rate: Optional[float] = 0.1
    gps_spoofing_score: Optional[float] = 0.0
    movement_realism_score: Optional[float] = 1.0
    presence_score: Optional[float] = 1.0
    peer_group_activity_ratio: Optional[float] = 1.0
    consecutive_payment_weeks: Optional[int] = 10
    coordinated_fraud_cluster_id: Optional[int] = 0
    ip_gps_mismatch: Optional[int] = 0
    device_sharing_flag: Optional[int] = 0

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
