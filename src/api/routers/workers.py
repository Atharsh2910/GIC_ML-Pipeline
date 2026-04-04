import math
from typing import Optional
from fastapi import APIRouter, Query, HTTPException

from src.api.schemas import WorkerListResponse
from src.api.dependencies import get_worker_data

router = APIRouter(prefix="/workers", tags=["Workers"])

@router.get("/", response_model=WorkerListResponse)
def list_workers(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    city: Optional[str] = None,
    disruption_type: Optional[str] = None
):
    df = get_worker_data()
    if df.empty:
        return {"total": 0, "page": page, "size": size, "workers": []}
    
    filtered_df = df
    if city:
        filtered_df = filtered_df[filtered_df["city"].str.lower() == city.lower()]
    if disruption_type:
        filtered_df = filtered_df[filtered_df["disruption_type"].str.lower() == disruption_type.lower()]
    
    total = len(filtered_df)
    start_idx = (page - 1) * size
    end_idx = start_idx + size
    
    page_df = filtered_df.iloc[start_idx:end_idx]
    
    # Fill NaN with None for JSON serialization
    workers = page_df.where(page_df.notna(), None).to_dict(orient="records")
    
    return {
        "total": total,
        "page": page,
        "size": size,
        "workers": workers
    }

@router.get("/{worker_id}")
def get_worker(worker_id: str):
    df = get_worker_data()
    if df.empty:
        raise HTTPException(status_code=404, detail="Worker data not loaded")
    
    # worker_id might be int or string in dataframe
    # Safe match
    worker_df = df[df["worker_id"].astype(str) == str(worker_id)]
    if worker_df.empty:
        raise HTTPException(status_code=404, detail="Worker not found")
        
    worker_record = worker_df.iloc[0].where(worker_df.iloc[0].notna(), None).to_dict()
    return worker_record
