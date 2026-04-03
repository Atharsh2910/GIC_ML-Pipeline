"""Populate Supabase worker snapshots and Pinecone vectors from the dataset CSV."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


def series_to_record(row: pd.Series) -> dict:
    out: dict = {}
    for k in row.index:
        v = row[k]
        if pd.isna(v):
            out[str(k)] = None
        elif isinstance(v, (np.integer, int)):
            out[str(k)] = int(v)
        elif isinstance(v, (np.floating, float)):
            out[str(k)] = float(v)
        elif isinstance(v, (bool, np.bool_)):
            out[str(k)] = bool(v)
        else:
            out[str(k)] = str(v)
    return out


def worker_summary_text(row: pd.Series) -> str:
    parts = [
        f"Worker {row.get('worker_id')} in {row.get('city')} platform {row.get('platform')}",
        f"slab {row.get('selected_slab')} employment {row.get('employment_type')}",
        f"disruption {row.get('disruption_type')} duration_h {row.get('disruption_duration_hours')} rainfall_cm {row.get('rainfall_cm')}",
        f"income_loss_pct {row.get('income_loss_percentage')} weekly_income {row.get('weekly_income')} avg_52w {row.get('avg_52week_income')}",
        f"risk {row.get('overall_risk_score')} fraud_trust {row.get('fraud_trust_rating')} gps_spoof {row.get('gps_spoofing_score')}",
        f"cooling_ok {row.get('cooling_period_completed')} premium_paid {row.get('premium_paid')} payout {row.get('final_payout_amount')}",
    ]
    return " ".join(str(p) for p in parts)


def run_populate(
    csv_path: Path,
    limit: int | None,
    do_supabase: bool,
    do_pinecone_kb: bool,
    do_worker_vectors: bool,
    worker_vector_limit: int,
) -> None:
    from src.utils.schema import ensure_worker_columns

    df = ensure_worker_columns(pd.read_csv(csv_path))
    if limit is not None and limit > 0:
        df = df.head(limit)
    n = len(df)
    print(f"Loaded {n} rows from {csv_path}")

    if do_supabase:
        from src.persistence.supabase_client import bulk_upsert_workers, is_configured

        if not is_configured():
            raise SystemExit("Supabase not configured: set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in .env")
        rows: list[dict] = []
        for _, row in df.iterrows():
            wid = int(row["worker_id"])
            rows.append({"worker_id": wid, "record": series_to_record(row)})
        inserted = bulk_upsert_workers(rows)
        print(f"Supabase gigshield_workers: upserted {inserted} rows")

    if do_pinecone_kb or do_worker_vectors:
        os.environ["VECTOR_STORE_PROVIDER"] = os.environ.get("VECTOR_STORE_PROVIDER", "pinecone").strip().lower() or "pinecone"
        if os.environ["VECTOR_STORE_PROVIDER"] != "pinecone":
            print("Setting VECTOR_STORE_PROVIDER=pinecone for this run.")
            os.environ["VECTOR_STORE_PROVIDER"] = "pinecone"

    if do_pinecone_kb:
        from src.rag.rag_system import VectorStore, populate_knowledge_base

        vs = VectorStore(provider="pinecone")
        populate_knowledge_base(vs)
        print("Pinecone: upserted curated knowledge_bundle categories (insurance_policies, fraud_cases, etc.)")

    if do_worker_vectors:
        from src.rag.rag_system import VectorStore

        sub = df.head(worker_vector_limit)
        docs: list[dict] = []
        for _, row in sub.iterrows():
            wid = int(row["worker_id"])
            docs.append(
                {
                    "id": f"ds_worker_{wid}",
                    "text": worker_summary_text(row),
                    "metadata": {
                        "worker_id": wid,
                        "city": str(row.get("city") or "")[:128],
                        "disruption_type": str(row.get("disruption_type") or "")[:128],
                    },
                }
            )
        if docs:
            vs = VectorStore(provider="pinecone")
            vs.add_documents(docs, category="dataset_workers")
            print(f"Pinecone: upserted {len(docs)} dataset_workers summary vectors (limit {worker_vector_limit})")
